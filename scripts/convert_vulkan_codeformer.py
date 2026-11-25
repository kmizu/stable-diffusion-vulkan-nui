import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto
from pathlib import Path
import argparse, hashlib, shutil, subprocess, tempfile
parser = argparse.ArgumentParser(description="Convert FaceFusion CodeFormer 3.0.0 for SDV-NUI using pnnx 20261020")
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--pnnx", required=True, type=Path)
args = parser.parse_args()
with args.model.open("rb") as stream:
 digest = hashlib.sha256()
 for chunk in iter(lambda: stream.read(1048576), b""):
  digest.update(chunk)
if digest.hexdigest() != "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f":
 raise ValueError("Expected the models-3.0.0 CodeFormer ONNX export")
args.output.mkdir(parents=True,exist_ok=True)
work = tempfile.TemporaryDirectory(dir=args.output)
src = Path(work.name).resolve()
m = onnx.load(args.model)
g = m.graph
prod={o:n for n in g.node for o in n.output}
users={}
for n in g.node:
 for x in n.input:users.setdefault(x,[]).append(n)
for n in g.node:
 if n.op_type=='Softmax' and '/self_attn/' in n.name:
  flat=prod[n.input[0]];reshape,=users[n.output[0]]
  assert flat.op_type=='Flatten' and reshape.op_type=='Reshape' and flat.attribute[0].i==2
  n.input[0]=flat.input[0];n.output[0]=reshape.output[0];reshape.output[0]+='_unused'
weight=g.input[1];weight.type.tensor_type.elem_type=TensorProto.FLOAT
weight.type.tensor_type.shape.dim.add().dim_value=1
new=[]
for n in g.node:
 if n.op_type=='MatMul' and n.input[1]=='quantize.embedding.weight':
  new.extend([helper.make_node('ArgMax',['logits'],['code_ids'],axis=-1,keepdims=0),
        helper.make_node('Reshape',['code_ids','codes_shape'],['codes_flat']),
        helper.make_node('Gather',['quantize.embedding.weight','codes_flat'],list(n.output),axis=0)])
 else:new.append(n)
g.initializer.append(numpy_helper.from_array(np.array([256],dtype=np.int64),'codes_shape'))
del g.output[1:]; needed={'output'}; kept=[]
for n in reversed(new):
 if needed.intersection(n.output):
  kept.append(n);needed.update(n.input)
del g.node[:];g.node.extend(reversed(kept));onnx.checker.check_model(m);onnx.save(m,src/'codeformer_simple.onnx')

subprocess.run([str(args.pnnx.resolve()),"codeformer_simple.onnx","fp16=0"],cwd=src,check=True)
# pnnx assigns numbered blobs. Only rewrite the graph whose layouts were verified.
graph = (src/'codeformer_simple.ncnn.param').read_text()
if hashlib.sha256(graph.encode()).hexdigest() != "823396081ff6aebf8139358a697ea8916f596110a166b99f94bc777b36c454fe":
 raise ValueError("Unexpected NCNN graph. Use the pinned Windows pnnx release 20261020.")
lines=graph.splitlines()
keep=[]
for line in lines[2:]:
 fields=line.split()
 if fields[0]=='ArgMax' or fields[1]=='reshape_533':continue
 if fields[0]=='Gather':line='SdvCodebookLookup lookup 2 1 540 543 544'
 keep.append(line)
blobs=set()
for line in keep:
 fields=line.split()
 inputs,outputs=map(int,fields[2:4])
 blobs.update(fields[4:4+inputs+outputs])
lines=['7767517',f'{len(keep)} {len(blobs)}',*keep]
for i,l in enumerate(lines):
 a=l.split()
 if len(a)<6:continue
 if a[1]=='/Tile_output_0':lines[i]=' '.join(a[:5])+' 0=512 1=256'
 if a[0]!='Reshape' or not a[1].startswith('reshape_'):continue
 idx=int(a[1].split('_')[-1])
 if idx<449 or idx>532:continue
 shape=' '.join(a[6:])
 if shape in ['0=512 1=1 12=233 13=0 2=256','0=512','0=512 1=1']: shape='0=512 1=256'
 elif shape=='0=64 1=8':shape='0=64 1=8 2=256'
 elif shape in ['0=64 1=256 12=1 13=233 2=8','0=64 1=256 12=233 13=1 2=8']: a[0]='Permute';shape='0=2'
 elif shape=='0=256 1=64 12=2 13=233 2=8':a[0]='Permute';shape='0=3'
 elif idx==532:shape='0=1024 1=256'
 else:print(idx,shape)
 lines[i]=' '.join(a[:6])+' '+shape
param=src/'codeformer.ncnn.param'
param.write_text('\n'.join(lines)+'\n',encoding='ascii',newline='\n')
if not any(line.startswith('SdvCodebookLookup ') for line in lines):
 raise ValueError('Unexpected pnnx conversion layout')
shutil.copyfile(src/'codeformer_simple.ncnn.bin',args.output/'codeformer.ncnn.bin')
shutil.copyfile(param,args.output/'codeformer.ncnn.param')
work.cleanup()
