"""Prepare optional native SD safety checker and NSFWJS model assets from verified originals."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import torch


def verified(file, digest):
    with file.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != digest:
        raise ValueError("Original model hash mismatch: " + str(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety", type=Path)
    parser.add_argument("--nsfwjs", type=Path)
    parser.add_argument("--pnnx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.safety and not args.nsfwjs:
        parser.error("specify --safety and/or --nsfwjs")
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as scratch:
        work = Path(scratch)
        if args.safety:
            verified(args.safety,"64b8393f1afd5a0c1ed2aa5f341fa7c08286839a48f3743162a76a2835c808bd")
            state = torch.load(args.safety, weights_only=True, map_location="cpu")
            concept = {k: v.tolist() for k,v in state.items() if k.startswith(("concept_", "special_care_"))}
            (work / "concepts.json").write_text(json.dumps(concept,separators=(",",":")),encoding="utf-8")
            header, tensors, offset = {}, [], 0
            for k,v in state.items():
                if k in concept or k.endswith("position_ids"):
                    continue
                key = k.removeprefix("vision_model.") if k.startswith("vision_model.vision_model.") else k
                v = v.to(dtype=torch.float32).contiguous()
                size = v.numel() * v.element_size()
                header[key] = dict(dtype="F32",shape=list(v.shape),data_offsets=[offset,offset+size])
                tensors.append((v,size))
                offset += size
            raw = json.dumps(header,separators=(",",":")).encode("utf-8")
            raw += b" " * (-len(raw) % 8)
            with (work / "safety-vision.safetensors").open("wb") as f:
                f.write(struct.pack("<Q",len(raw))); f.write(raw)
                for t,size in tensors:
                    f.write(ctypes.string_at(t.data_ptr(),size))
            for name in ["concepts.json","safety-vision.safetensors"]:
                os.replace(work/name,target/name)
        if args.nsfwjs:
            if not args.pnnx:
                parser.error("--nsfwjs also requires --pnnx")
            verified(args.nsfwjs,"510dce55edcd1cf8281cbe906dedb3f4963f8814bc1edfbd42d3bf261148df2d")
            shutil.copy2(args.nsfwjs,work/"nsfwjs.onnx")
            subprocess.run([str(args.pnnx.resolve()),"nsfwjs.onnx","inputshape=[1,224,224,3]","fp16=0"],cwd=work,check=True)
            # This opset uses four constant ONNX Pad attributes. pnnx retains their names
            # but drops their attributes; all four pad the bottom and right edges by one.
            param = work / "nsfwjs.ncnn.param"
            rows = param.read_text().splitlines()
            pads = [i for i,row in enumerate(rows) if row.split()[:1] == ["Pad"]]
            if len(pads) != 4:
                raise ValueError("Unexpected padding layout in the pinned NSFWJS graph")
            for i in pads:
                rows[i] = rows[i].replace("Pad ", "Padding ", 1) + " 0=0 1=1 2=0 3=1 4=0 5=0"
            param.write_text("\n".join(rows)+"\n", encoding="ascii",newline="\n")
            for name in ["nsfwjs.ncnn.param","nsfwjs.ncnn.bin"]:
                os.replace(work/name,target/name)


if __name__ == "__main__":
    main()
