#!/usr/bin/env python3
"""One-time export of upstream PyTorch models to ONNX for MLVision.

Run from a venv with torch, transformers, timm, onnx and onnxruntime installed:

    python tools/export_models.py OUT_DIR

Produces, in OUT_DIR:
    nsfw-marqo-vit-tiny-384.onnx       Marqo/nsfw-image-detection-384   (Apache-2.0)
    age-fairface-vit-224.onnx          dima806/fairface_age_image_detection    (Apache-2.0)
    gender-fairface-vit-224.onnx       dima806/fairface_gender_image_detection (Apache-2.0)
    *-int8.onnx                        dynamically quantised copies of the FairFace models

Each export is checked against the PyTorch model on a random input.
"""
import sys, os, numpy as np, torch, onnx, onnxruntime as ort

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)

def check(path, model, x, input_name):
    with torch.no_grad():
        ref = model(x)
        ref = ref.logits if hasattr(ref, "logits") else ref
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {input_name: x.numpy()})[0]
    diff = np.abs(ref.numpy() - got).max()
    print(f"  {os.path.basename(path)}: {os.path.getsize(path)/1e6:.1f} MB, max |torch-onnx| = {diff:.2e}")
    assert diff < 1e-3, "export mismatch"

def export(model, x, path, input_name="input"):
    model.eval()
    torch.onnx.export(model, x, path, input_names=[input_name], output_names=["logits"],
                      dynamic_axes={input_name: {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(path)
    check(path, model, x, input_name)

def quantize(src, dst):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(src, dst, weight_type=QuantType.QUInt8)
    print(f"  {os.path.basename(dst)}: {os.path.getsize(dst)/1e6:.1f} MB")

print("Marqo/nsfw-image-detection-384")
import timm
m = timm.create_model("hf_hub:Marqo/nsfw-image-detection-384", pretrained=True)
print("  labels:", m.pretrained_cfg.get("label_names"), "mean/std:", m.pretrained_cfg["mean"], m.pretrained_cfg["std"])
export(m, torch.randn(1, 3, 384, 384), f"{out}/nsfw-marqo-vit-tiny-384.onnx")

from transformers import ViTForImageClassification
for hub, name in [("dima806/fairface_age_image_detection", "age-fairface-vit-224"),
                  ("dima806/fairface_gender_image_detection", "gender-fairface-vit-224")]:
    print(hub)
    m = ViTForImageClassification.from_pretrained(hub)
    print("  labels:", m.config.id2label)
    export(m, torch.randn(1, 3, 224, 224), f"{out}/{name}.onnx", input_name="pixel_values")
    quantize(f"{out}/{name}.onnx", f"{out}/{name}-int8.onnx")
print("done")
