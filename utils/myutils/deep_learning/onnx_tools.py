import onnxruntime


def load_onnx_session(w):
    session = onnxruntime.InferenceSession(w, None)
    return session
