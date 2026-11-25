"""Maintainer-only CPU conversion of Salesforce BLIP-base for the ncnn Vulkan worker.

Use torch 2.7.1, transformers 4.51.3, onnx 1.21.0 and pnnx 20261020.
The Windows app runs the exported networks directly.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

import torch
from transformers import BlipForConditionalGeneration


class Vision(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixels):
        return self.visual(pixel_values=pixels, return_dict=False)[0]


class Decoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens, features):
        return self.decoder(input_ids=tokens, encoder_hidden_states=features, use_cache=False, return_dict=False)[0]


class Separate(torch.nn.Module):
    """Equivalent Q/K/V projections without the unsupported five-dimensional packed reshape."""
    def __init__(self, original):
        super().__init__()
        self.width = original.embed_dim
        self.heads = original.num_heads
        self.q, self.k, self.v = (torch.nn.Linear(self.width, self.width) for _ in range(3))
        for i, linear in enumerate((self.q, self.k, self.v)):
            linear.weight.copy_(original.qkv.weight[i * self.width:(i + 1) * self.width])
            linear.bias.copy_(original.qkv.bias[i * self.width:(i + 1) * self.width])
        self.proj = original.projection

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        def heads(x):
            return x.reshape(1, 577, self.heads, 64).transpose(1, 2)
        q, k, v = (heads(layer(hidden_states)) for layer in (self.q, self.k, self.v))
        scores = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * 0.125, dim=-1)
        return self.proj(torch.matmul(scores, v).transpose(1, 2).reshape(1, 577, self.width)), None


def verify(path, expected):
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    if digest != expected:
        raise ValueError(f"Wrong SHA-256 for {path}: {digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local Salesforce/blip-image-captioning-base @ 82a37760796d32b1411fe092ab5d4e227313294b snapshot")
    parser.add_argument("pnnx", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    large = (args.source / "model.safetensors").is_file() and not (args.source / "pytorch_model.bin").is_file()
    if large:
        verify(args.source / "model.safetensors", "ebc867586b8318647dd7b75ef8e52f93f89b98ac7dcbd04d10219d33b9a68c6f")
    else:
        verify(args.source / "pytorch_model.bin", "d6638651a5526cc2ede56f2b5104d6851b0755816d220e5e046870430180c767")
    if not large: verify(args.source / "config.json", "f7846f82f4e2c4a2ccbab9ae8b0e44873540a271a76a1288effd078180c13a82")
    verify(args.source / "vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3")
    torch.set_grad_enabled(False)
    torch.set_num_threads(3)
    model = BlipForConditionalGeneration.from_pretrained(args.source, local_files_only=True, use_safetensors=large).eval()
    for layer in model.vision_model.encoder.layers:
        layer.self_attn = Separate(layer.self_attn)

    def export(module, name, inputs, shapes):
        torch.onnx.export(module.eval(), inputs, str(root / (name + ".onnx")), dynamo=False,
                          input_names=[f"in{i}" for i in range(len(inputs))], output_names=["out0"],
                          opset_version=17, do_constant_folding=True)
        subprocess.run([str(args.pnnx.resolve()), name + ".onnx", "inputshape=" + shapes, "fp16=0"], cwd=root, check=True)

    export(Vision(model.vision_model), "vision", (torch.zeros(1, 3, 384, 384),), "[1,3,384,384]")
    tokens = torch.zeros((1, 64), dtype=torch.long)
    tokens[0, 0] = 30522
    export(Decoder(model.text_decoder), "blip-decoder", (tokens, torch.zeros(1, 577, model.config.vision_config.hidden_size)), f"[1,64]i64,[1,577,{model.config.vision_config.hidden_size}]")
    shutil.copyfile(args.source / "vocab.txt", root / "vocab.txt")
    expected = {
        "vision.ncnn.param": "7da371282df5d9ea6d29b661eceadde5e4c83eff91356937d1f020d3de270be6",
        "vision.ncnn.bin": "75fcfd12f40e59a8bd4a7e43f9ce38935da7d49ec92f84e63d9c4eaa1aecf7fd",
        "blip_decoder.ncnn.param": "d253631a0113221d1cb48c95ffa9115c43903ba8e00732f88f4228dccb5808f3",
        "blip_decoder.ncnn.bin": "6dce4d756ee2f9cf097833cb0a246cbca726bd659650d8a34ce54418091f6519",
    }
    if large:
        expected = {
            "vision.ncnn.param": "9aeb227a7bb2925c1f9df8a1974696c3cf09f03e6f105a19da69a53c6bfbf80c",
            "vision.ncnn.bin": "4656300cb9619a81f0efee385f40069a5d4aa3e4ff0823bb91d6fa8af1af55db",
            "blip_decoder.ncnn.param": "4a429dba1d874bf99c3a5df5b20d2ec71f5664b537a02bb02360d112e1c9f7b9",
            "blip_decoder.ncnn.bin": "2b103055538099eb7ef1e9963b9fbeefc8b9a2ce77b2b3e9312c4a2aeede1d4a",
        }
    for filename, digest in expected.items():
        verify(root / filename, digest)


if __name__ == "__main__":
    main()
