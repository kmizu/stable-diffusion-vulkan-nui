"""Build optional ncnn RAFT assets from torchvision weights (build-time Python only)."""
import argparse
from pathlib import Path
import subprocess
import shutil
import tempfile

import torch
from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights


class Encoder(torch.nn.Module):
    def __init__(self, raft):
        super().__init__()
        self.feature = raft.feature_encoder
        self.context = raft.context_encoder

    def forward(self, rgb):
        return torch.cat((self.feature(rgb), self.context(rgb)), dim=1)


class Update(torch.nn.Module):
    def __init__(self, raft, sizes):
        super().__init__()
        self.update = raft.update_block
        self.mask = raft.mask_predictor
        self.sizes = sizes

    def forward(self, inputs):
        hidden, context, corr, flow = torch.split(inputs, self.sizes, dim=1)
        hidden, delta = self.update(hidden, context, corr, flow)
        outputs = [hidden, delta]
        if self.mask is not None:
            outputs.append(self.mask(hidden))
        return torch.cat(outputs, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["small", "large", "both"], default="both")
    parser.add_argument("--weights", type=Path, help="custom torchvision state dictionary for a single variant")
    args = parser.parse_args()
    if args.weights and args.kind == "both":
        parser.error("custom weights require one variant")
    torch.set_num_threads(2)
    for name in (["small", "large"] if args.kind == "both" else [args.kind]):
        small = name == "small"
        factory = raft_small if small else raft_large
        weights = Raft_Small_Weights.C_T_V2 if small else Raft_Large_Weights.C_T_SKHT_V2
        raft = factory(weights=None if args.weights else weights).eval()
        if args.weights:
            raft.load_state_dict(torch.load(args.weights, weights_only=True, map_location="cpu"))
        sizes = [96, 64, 196, 2] if small else [128, 128, 324, 2]
        target = args.output.resolve() / ("raft-" + name)
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch)
            for component, module, channels, side in [
                ("encoder", Encoder(raft), 3, 128),
                ("update", Update(raft, sizes), sum(sizes), 16),
            ]:
                file = work / (component + ".pt")
                with torch.no_grad():
                    torch.jit.trace(module.eval(), torch.zeros(1,channels,side,side), check_trace=False).save(str(file))
                subprocess.run([str(args.pnnx.resolve()), file.name, f"inputshape=[1,{channels},{side},{side}]",
                                f"inputshape2=[1,{channels},{side*2},{side*3}]", "fp16=0"], cwd=work, check=True)
                param = (work / (component + ".ncnn.param")).read_text()
                if "pnnx." in param or "aten::" in param:
                    raise RuntimeError("unconverted operators in " + str(file))
                for suffix in (".ncnn.bin", ".ncnn.param"):
                    shutil.copy2(work / (component + suffix), target / (component + suffix))



if __name__ == "__main__":
    main()
