import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx",
    providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
)

x = np.ones((2, 4), dtype=np.float32)
y = sess.run(None, {"input": x})[0]

print("session_providers", sess.get_providers())
print("output_shape", y.shape)
print("ok")