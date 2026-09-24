"""
ONNX Model Export and Dynamic INT8 Quantization Harness (Phase 3).

Converts GRUMaskNet weights into an optimized ONNX computational graph:
- Inputs: log_mag [1, 257], hidden_in [1, 64]
- Outputs: gain_mask [1, 257], hidden_out [1, 64]
- Generates:
  1. src/models/grumasknet_fp32.onnx (Full precision 32-bit float model)
  2. src/models/grumasknet_int8.onnx (Quantized 8-bit dynamic INT8 model)
"""

from __future__ import annotations
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.denoiser import GRUMaskNet


def build_onnx_model(weights_path: str | None = None) -> tuple[str, str]:
    """Build and serialize both FP32 and INT8 ONNX models for GRUMaskNet.

    Returns
    -------
    tuple[str, str]
        Paths to (fp32_onnx_path, int8_onnx_path).
    """
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime.quantization as ort_quant

    net = GRUMaskNet(weights_path=weights_path)
    weights = net.get_weights()

    W1 = weights["W1"]  # (257, 64)
    b1 = weights["b1"]  # (64,)
    W2 = weights["W2"]  # (64, 64)
    b2 = weights["b2"]  # (64,)
    W_rec = weights["W_rec"]  # (64, 64)
    W3 = weights["W3"]  # (64, 257)
    b3 = weights["b3"]  # (257,)

    # 1. Inputs and Outputs ValueInfo
    log_mag = helper.make_tensor_value_info("log_mag", TensorProto.FLOAT, [1, 257])
    hidden_in = helper.make_tensor_value_info("hidden_in", TensorProto.FLOAT, [1, 64])

    gain_mask = helper.make_tensor_value_info("gain_mask", TensorProto.FLOAT, [1, 257])
    hidden_out = helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT, [1, 64])

    # 2. Constant Initializers (Model Weights & Biases)
    init_W1 = helper.make_tensor("W1", TensorProto.FLOAT, list(W1.shape), W1.flatten().tolist())
    init_b1 = helper.make_tensor("b1", TensorProto.FLOAT, list(b1.shape), b1.flatten().tolist())
    init_W2 = helper.make_tensor("W2", TensorProto.FLOAT, list(W2.shape), W2.flatten().tolist())
    init_b2 = helper.make_tensor("b2", TensorProto.FLOAT, list(b2.shape), b2.flatten().tolist())
    init_Wrec = helper.make_tensor("W_rec", TensorProto.FLOAT, list(W_rec.shape), W_rec.flatten().tolist())
    init_W3 = helper.make_tensor("W3", TensorProto.FLOAT, list(W3.shape), W3.flatten().tolist())
    init_b3 = helper.make_tensor("b3", TensorProto.FLOAT, list(b3.shape), b3.flatten().tolist())

    # Sigmoid scaling constant (1.25 multiplier for sharp thresholding)
    init_scale = helper.make_tensor("sig_scale", TensorProto.FLOAT, [1], [1.25])

    # 3. Graph Nodes
    nodes = [
        # Layer 1: z1 = Gemm(log_mag, W1, b1) -> a1 = Relu(z1)
        helper.make_node("Gemm", ["log_mag", "W1", "b1"], ["z1"], alpha=1.0, beta=1.0),
        helper.make_node("Relu", ["z1"], ["a1"]),

        # Layer 2: z2 = MatMul(a1, W2) + MatMul(hidden_in, W_rec) + b2
        helper.make_node("MatMul", ["a1", "W2"], ["a1_w2"]),
        helper.make_node("MatMul", ["hidden_in", "W_rec"], ["h_wrec"]),
        helper.make_node("Add", ["a1_w2", "h_wrec"], ["z2_pre"]),
        helper.make_node("Add", ["z2_pre", "b2"], ["z2"]),
        helper.make_node("Relu", ["z2"], ["hidden_out"]),

        # Layer 3: z3 = Gemm(hidden_out, W3, b3) -> Sigmoid(1.25 * z3)
        helper.make_node("Gemm", ["hidden_out", "W3", "b3"], ["z3"], alpha=1.0, beta=1.0),
        helper.make_node("Mul", ["z3", "sig_scale"], ["z3_scaled"]),
        helper.make_node("Sigmoid", ["z3_scaled"], ["gain_mask"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="GRUMaskNet_Graph",
        inputs=[log_mag, hidden_in],
        outputs=[gain_mask, hidden_out],
        initializer=[init_W1, init_b1, init_W2, init_b2, init_Wrec, init_W3, init_b3, init_scale],
    )

    model_def = helper.make_model(graph, producer_name="EdgeAI_Audio_Profiler")
    model_def.opset_import[0].version = 17
    model_def.ir_version = 10
    onnx.checker.check_model(model_def)

    # 4. Save FP32 ONNX
    models_dir = os.path.join(PROJECT_ROOT, "src", "models")
    fp32_path = os.path.join(models_dir, "grumasknet_fp32.onnx")
    onnx.save(model_def, fp32_path)
    print(f"[SUCCESS] Exported FP32 ONNX: {fp32_path} ({os.path.getsize(fp32_path)} bytes)")

    # 5. Dynamic INT8 Quantization
    int8_path = os.path.join(models_dir, "grumasknet_int8.onnx")
    ort_quant.quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=ort_quant.QuantType.QInt8,
    )
    # Ensure quantized model IR version conforms to onnxruntime limits
    int8_model = onnx.load(int8_path)
    int8_model.ir_version = 10
    onnx.save(int8_model, int8_path)
    print(f"[SUCCESS] Exported INT8 ONNX: {int8_path} ({os.path.getsize(int8_path)} bytes)")

    return fp32_path, int8_path


if __name__ == "__main__":
    build_onnx_model()
