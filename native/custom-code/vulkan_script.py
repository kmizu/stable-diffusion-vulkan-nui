"""Bridge between a trusted local Custom Code snippet and its selected Vulkan worker."""
import contextlib
import io
import json
import sys
from types import SimpleNamespace

wire_in, wire_out = sys.stdin, sys.stdout
request = json.loads(wire_in.readline())


def send(message):
    print(json.dumps(message, ensure_ascii=True), file=wire_out, flush=True)


class VulkanPipeline:
    def __str__(self):
        return f"VulkanPipeline({request['device']}, model={request.get('model')})"

    def __call__(self, prompt="", **settings):
        for old, new in (("num_inference_steps", "steps"), ("guidance_scale", "cfg_scale")):
            if old in settings:
                settings[new] = settings.pop(old)
        send({"generate": dict(settings, prompt=prompt)})
        reply = json.loads(wire_in.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return SimpleNamespace(images=[item["path"] for item in reply["images"]])


class Capture(io.TextIOBase):
    def __init__(self):
        self.tail = ""

    def write(self, value):
        self.tail = (self.tail + value[-20000:])[-20000:]
        return len(value)


pipe = VulkanPipeline()
engine = SimpleNamespace(pipe=pipe, device=request["device"], model=request.get("model"))
scope = dict(pipe=pipe, engine=engine, params=request["params"])
output = Capture()
try:
    with contextlib.redirect_stdout(output):
        exec(compile(request["params"]["code"], "<custom code>", "exec"), scope)
    result = scope.get("result")
    send({"reply": {"stdout": output.tail, "result": None if result is None else str(result)[:20000],
                    "images": [str(image) for image in (scope.get("images") or [])]}})
except Exception as exc:
    send({"error": f"{type(exc).__name__}: {exc}"})
