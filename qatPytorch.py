import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
    QConfig,
    prepare_qat,
    convert,
    QuantStub,
    DeQuantStub,
    fuse_modules,
)
import ipdb

import numpy as np
import torch
from scipy.signal import correlate

def dequantize(q, scale, zp):
    print(scale)
    print(zp)
    return (q.astype(np.float32) - zp) * scale
def quantize(x, scale, zp):
    return np.round(x / scale + zp).astype(np.uint8)
# -----------------------
# Load quantization params
# -----------------------
params = np.load("quant_params.npy", allow_pickle=True).item()

import numpy as np

# ============================================================
# Helper functions for fully integer inference
# ============================================================

def conv2d_int8(x_q, w_q, x_zp, w_zp):
    """
    Fully integer 2D convolution.
    x_q: [N, C, H, W], w_q: [F, C, Kh, Kw], x_zp: scalar, w_zp: [F]
    Returns: int32 output
    """
    N, C, H, W = x_q.shape
    F, _, Kh, Kw = w_q.shape
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    out = np.zeros((N, F, H_out, W_out), dtype=np.int32)
    
    for n in range(N):
        for f in range(F):
            for h in range(H_out):
                for w in range(W_out):
                    patch = x_q[n, :, h:h+Kh, w:w+Kw] - x_zp
                    filt = w_q[f] - w_zp[f]
                    out[n, f, h, w] = np.sum(patch * filt)
    return out

def conv2d_int8_explicit_zp(x_q, w_q, x_zp, w_zp):
    """
    Fully integer 2D convolution with explicit zero-point correction.
    x_q: [N, C, H, W], w_q: [F, C, Kh, Kw], x_zp: scalar, w_zp: [F]
    Returns: int32 output
    """
    N, C, H, W = x_q.shape
    F, _, Kh, Kw = w_q.shape
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    out = np.zeros((N, F, H_out, W_out), dtype=np.int32)
    
    # Pre-calculate the constant sum of 1s (kernel volume)
    C_Kh_Kw = C * Kh * Kw
    
    for n in range(N):
        for f in range(F):
            # The scalar ZP for the current filter
            Z_W_f = w_zp[f]
            
            for h in range(H_out):
                for w in range(W_out):
                    patch = x_q[n, :, h:h+Kh, w:w+Kw]
                    filt = w_q[f]
                    
                    # 1. Term 1: Pure MAC (X_q * W_q)
                    term1 = np.sum(patch.astype(np.int32) * filt.astype(np.int32))
                    
                    # 2. Term 2: Input ZP correction (Z_X * Sum(W_q))
                    term2 = x_zp * np.sum(filt)
                    
                    # 3. Term 3: Weight ZP correction (Z_W[f] * Sum(X_q))
                    term3 = Z_W_f * np.sum(patch)
                    
                    # 4. Term 4: ZP Product (Z_X * Z_W[f] * C*Kh*Kw)
                    term4 = x_zp * Z_W_f * C_Kh_Kw
                    
                    # Result accumulation
                    acc_int32 = term1 - term2 - term3 + term4
                    
                    # 5. Add Bias
                    # acc_int32 += bias_int32[f] # Assuming bias_int32 is a 1D array (F)
                    
                    out[n, f, h, w] = acc_int32
    return out

# The linear layer (FC) should be checked similarly but is simpler due to 2D matrix multiplication.
# The linear_int8 function is already using the centered form, which is usually safe for large NumPy matmuls.
def linear_int8(x_q, w_q, x_zp, w_zp):
    """
    Fully integer linear layer.
    x_q: [N, Din], w_q: [Dout, Din], x_zp: scalar, w_zp: [Dout]
    Returns: int32 output
    """
    x_centered = x_q - x_zp
    w_centered = w_q - w_zp[:, None]
    out = x_centered @ w_centered.T
    return out

def requantize(x_int32, scale, zp, dtype=np.uint8):
    """
    Requantize int32 -> int8/quint8
    """
    print(x_int32.shape)
    x = np.round(x_int32 * scale + zp).astype(dtype)
    x = np.clip(x, 0, 255)  # for quint8
    return x

# ============================================================
# Fully integer NumPy inference
# ============================================================

def run_numpy_model_int(x_q):
    """
    x_q: input quantized tensor (quint8) [N, C, H, W]
    Returns: output quantized tensor (quint8)
    """
    # --------------------
    # Conv layer (int32)
    # --------------------
    conv = params["conv.weight"]
    x_zp = params["quant.activation"]["zp_a"]
    w_q = conv["weight_int"]          # int8 weights
    w_zp = conv["zp_w"]               # per-channel zero points
    print("Input zp: \n", x_zp)
    print("Conv weights int: \n", w_q)
    print("Conv zp: \n", w_zp)
    print("Conv weights scales: \n", conv["scale_w"])
    print("Input scales: \n", params["quant.activation"]["scale_a"])
    print("Conv output scale: \n", params["conv.activation"]["scale_a"])
    scale_conv = conv["scale_w"] * params["quant.activation"]["scale_a"] / params["conv.activation"]["scale_a"]
    scale_conv_reshaped = scale_conv.reshape(1, -1, 1, 1)
    out_conv_int = conv2d_int8_explicit_zp(x_q, w_q, x_zp, w_zp)
    # Requantize conv output to int8/quint8
    out_conv_q = requantize(out_conv_int, scale_conv_reshaped, params["conv.activation"]["zp_a"])
    print("output (quantized):", out_conv_q[0, 0, 0:3, 0:3])
    """out_conv_float = dequantize(out_conv_q, params["conv.activation"]["scale_a"], params["conv.activation"]["zp_a"])
    return out_conv_float"""
    # --------------------
    # FC layer (int32)
    # --------------------
    fc = params["fc.weight"]
    w_q_fc = fc["weight_int"]
    w_zp_fc = fc["zp_w"]                  # per-output zero points
    scale_fc = fc["scale_w"] * params["conv.activation"]["scale_a"] / params["fc.activation"]["scale_a"]
    scale_fc_reshaped = scale_fc.reshape(1, -1)
    out_fc_int = linear_int8(out_conv_q.reshape(1, -1), w_q_fc, params["conv.activation"]["zp_a"], w_zp_fc)

    # Requantize FC output to int8/quint8
    out_fc_q = requantize(out_fc_int, scale_fc_reshaped, params["fc.activation"]["zp_a"])

    return out_fc_q


# Use the usual quantized backend for x86 (change to "qnnpack" on mobile/ARM if needed)
torch.backends.quantized.engine = "fbgemm"

# -------------------------
# Model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(1, 16, kernel_size=3)    # output: (16, 26, 26)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 26 * 26, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        print("Quant stub output (quantized):", x[0, 0, 0:3, 0:3])
        x = self.conv(x)
        print("Conv output (quantized):", x[0, 0, 0:3, 0:3])
        x = self.relu(x)
        # print("ReLU output (quantized):", x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        # print("FC output (quantized):", x)
        x = self.dequant(x)
        # rint("Output after dequant:", x)
        return x

# -------------------------
# Build FakeQuant modules for QAT
# - We simulate symmetric int4 weights (quantized range -8..7) per-channel
# - We simulate symmetric int8 activations (quantized range -128..127) per-tensor
# -------------------------

# Activation fake-quant: per-tensor symmetric int8
# Activation fake-quant: per-tensor unsigned int8 (0..255) for FBGEMM
activation_fake_quant = FakeQuantize.with_args(
    observer=HistogramObserver.with_args(
        dtype=torch.quint8,            # <-- unsigned for FBGEMM
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    ),
    quant_min=0,                      # 0..255
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_symmetric,
)


# Weight fake-quant: per-channel symmetric int4 range (-8 .. 7)
weight_fake_quant = FakeQuantize.with_args(
    observer=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,                      # storage dtype used by observer API
        qscheme=torch.per_channel_symmetric,    # symmetric per-channel
        reduce_range=False,
        # quant_min/quant_max on observer help initialize clamping behavior if needed
        quant_min=-8,
        quant_max=7,
    ),
    quant_min=-8,
    quant_max=7,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
)

# Build QConfig that uses our fake-quant modules
qconfig_int4w_int8a = QConfig(
    activation=activation_fake_quant,
    weight=weight_fake_quant,
)

# -------------------------
# Prepare model for QAT (legacy flow)
# -------------------------
model = SimpleCNN()

# Optional: fuse conv + relu (improves QAT fidelity). fuse_modules expects names of attributes.
# Works because conv and relu are attributes at module root.
# Note: fusing is generally done in eval mode, but for simple conv+relu this is safe to call before QAT.
fuse_modules(model, [["conv", "relu"]], inplace=True)

# Assign qconfig to top-level model (propagates to submodules on prepare_qat)
model.qconfig = qconfig_int4w_int8a

# Prepare for QAT (this inserts FakeQuantize modules according to qconfig)
# inplace=True will modify the model in-place; set to False to get a prepared copy
model.train()
prepare_qat(model, inplace=True)


# -------------------------
# Toy training loop (replace with your real dataloader)
# -------------------------
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for step in range(5000):
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if step % 2 == 0:
        print(f"Step {step} loss: {loss.item():.4f}")

print("Finished QAT training.")

# -------------------------
# Convert to a quantized model for inference
# -------------------------
model.eval()   # convert expects eval mode

print("=== QAT FakeQuantization parameters before convert ===")
for name, module in model.named_modules():
    if isinstance(module, FakeQuantize):
        try:
            scale = module.scale.item()
            zp = module.zero_point.item()
            print(f"{name}: scale={scale:.6f}, zero_point={zp}")
        except Exception:
            # Some modules may not have scale/zero_point initialized yet
            pass
for name, module in model.named_modules():
    if isinstance(module, FakeQuantize):
        if module.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
            print(f"{name} per-channel scale: {module.scale}")
            print(f"{name} per-channel zero_point: {module.zero_point}")
        else:
            print(f"{name} scale: {module.scale.item()}, zero_point: {module.zero_point.item()}")

quantized_model = convert(model, inplace=False)
# Save the full quantized model (recommended)

print("\n=== Extracting quantized weights after convert() ===")

for name, module in quantized_model.named_modules():
    if hasattr(module, "weight") and isinstance(module.weight(), torch.Tensor):

        qw = module.weight()         # quantized tensor
        int_vals = qw.int_repr()     # raw integer values
        # print(int_vals)
        print(f"\nLayer: {name}")
        print(f"  dtype: {qw.dtype}")
        print(f"  shape: {int_vals.shape}")

        # ---- Handle per-channel or per-tensor qscheme ----
        if qw.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine):
            print("  qscheme: per-channel")
            print("  scales:", qw.q_per_channel_scales())
            print("  zero_points:", qw.q_per_channel_zero_points())
            print("  axis:", qw.q_per_channel_axis())
        else:
            print("  qscheme:", qw.qscheme())
            print("  scale:", qw.q_scale())
            print("  zero_point:", qw.q_zero_point())

        # ---- Statistics ----
        print(f"  int8/int4 min={int_vals.min().item()}, max={int_vals.max().item()}")


import numpy as np

def extract_quant_params(qmodel):
    params = {}

    for name, module in qmodel.named_modules():

        # -----------------------------
        # 1. Extract weight quant params
        # -----------------------------
        if hasattr(module, "weight") and isinstance(module.weight(), torch.Tensor):
            qw = module.weight()
            int_w = qw.int_repr().cpu().numpy()
            print(name)
            print(int_w.shape)
            if qw.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine):
                scales = qw.q_per_channel_scales().cpu().numpy()
                zps    = qw.q_per_channel_zero_points().cpu().numpy()
                axis   = qw.q_per_channel_axis()
            else:
                scales = qw.q_scale()
                zps    = qw.q_zero_point()
                axis   = None

            params[name + ".weight"] = {
                "weight_int": int_w,
                "scale_w": scales,
                "zp_w": zps,
                "axis": axis,
            }

        # -------------------------------------------
        # 2. Extract activation quant params for input
        # -------------------------------------------
        if hasattr(module, "scale") and hasattr(module, "zero_point"):
            try:
                act_scale = float(module.scale)
                act_zp    = int(module.zero_point)
                params[name + ".activation"] = {
                    "scale_a": act_scale,
                    "zp_a": act_zp,
                }
            except:
                pass

        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            # PyTorch stores the INT32 bias which is already scaled to the output accumulation space.
            print(name)
            int_b = module.bias.cpu().numpy()
            params[name + ".weight"]["bias_int32"] = int_b
            print(f"Extracted bias for {name}")

    return params



quant_params = extract_quant_params(quantized_model)
np.save("quant_params.npy", quant_params, allow_pickle=True)
print(quant_params)
print("Saved quant_params.npy")

torch.save(quantized_model.state_dict(), "quantized_model.pth")

# Or save the entire model object (includes structure + weights)
torch.save(quantized_model, "quantized_model_full.pth")

print("Converted model type:", type(quantized_model))

# -------------------------
# Optional: inspect a quantized weight tensor (they will be stored as qint8, but values constrained to -8..7)
# -------------------------
summary(model, input_size=(1, 1, 28, 28))
for name, param in quantized_model.named_parameters():
    print(name)
    if "weight" in name:
        try:
            print(name, "dtype:", param.dtype, "min:", float(param.min()), "max:", float(param.max()))
        except Exception:
            # Some quantized modules don't expose .data the same way. Ignore safely.
            pass

fc_act = params["fc.activation"]
x = torch.randn(1, 1, 28, 28)
print(x)
# --- GET QUANTIZED INPUT FOR NUMPY ---
# 1. Pass float input through the QuantStub module.
with torch.no_grad():
    x_quantized_tensor = quantized_model.quant(x)
    # 2. Extract the raw integer representation (quint8)
    x_q_numpy = x_quantized_tensor.int_repr().cpu().numpy()

y_numpy = run_numpy_model_int(x_q_numpy)
print(fc_act["scale_a"])
print(fc_act["zp_a"])
y_numpy_float = dequantize(y_numpy, fc_act["scale_a"], fc_act["zp_a"])
print(y_numpy_float.dtype)
print("NumPy output:\n", y_numpy_float)
quantized_model.eval()
with torch.no_grad():
    y_torch = quantized_model(x).numpy()
print("Torch output:\n", y_torch)

diff = y_numpy_float - y_torch
print("Final Outputs MAE:", np.mean(np.abs(diff)))
print("Final Outputs Max Error:", np.max(np.abs(diff)))

print("TF first values:", y_numpy_float.flatten()[:10])
print("NP first values:", y_torch.flatten()[:10])
print("Diff first values:", diff.flatten()[:10])


ipdb.set_trace()
