import numpy as np
import tensorflow as tf
import h5py
import numpy as np
from scipy.signal import convolve
import soundfile as sf
import collections
import time
import sounddevice as sd
from numpy.lib.stride_tricks import sliding_window_view
model_size_info = [
    5,    # number of conv layers
    144, 10, 4, 3, 3,   # first conv
    144, 3, 3, 1, 1,    # second conv
    144, 3, 3, 1, 1,    # third conv
    144, 3, 3, 1, 1,    # fourth conv
    144, 3, 3, 1, 1,    # fifth conv
]

model_settings = {
    "spectrogram_length": 92,
    "mel_bin_count": 64,
    "label_count": 4,
    "desired_samples": 24000,
    "fingerprint_size": 5888
    }
weights_path = "checkpointsLogmelUpdatedMediumQATV10/best_model_weights_9785_10400.weights.h5"
OLD_TO_STANDARD = {

# ============ Conv1 (Conv2D) ============
"_layer_checkpoint_dependencies/quantize_wrapper_v2_2/vars/2":
    "layers/dscnn/layers/conv2d/vars/0",     # kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_2/layer/vars/0":
    "layers/dscnn/layers/conv2d/vars/1",     # bias


# ============ Separable Block 1 ============
"_layer_checkpoint_dependencies/quantize_wrapper_v2_6/vars/2":
    "layers/dscnn/layers/separable_conv2d_0/vars/0",     # depthwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_8/vars/2":
    "layers/dscnn/layers/separable_conv2d_0/vars/1",     # pointwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_6/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_0/vars/2",     # dw bias
"_layer_checkpoint_dependencies/quantize_wrapper_v2_8/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_0/vars/3",     # pw bias



# ============ Separable Block 2 ============
"_layer_checkpoint_dependencies/quantize_wrapper_v2_12/vars/2":
    "layers/dscnn/layers/separable_conv2d_1/vars/0",     # depthwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_14/vars/2":
    "layers/dscnn/layers/separable_conv2d_1/vars/1",     # pointwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_12/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_1/vars/2",     # dw bias
"_layer_checkpoint_dependencies/quantize_wrapper_v2_14/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_1/vars/3",     # pw bias


# ============ Separable Block 3 ============
"_layer_checkpoint_dependencies/quantize_wrapper_v2_18/vars/2":
    "layers/dscnn/layers/separable_conv2d_2/vars/0",     # depthwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_20/vars/2":
    "layers/dscnn/layers/separable_conv2d_2/vars/1",     # pointwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_18/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_2/vars/2",     # dw bias
"_layer_checkpoint_dependencies/quantize_wrapper_v2_20/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_2/vars/3",     # pw bias

# ============ Separable Block 4 ============
"_layer_checkpoint_dependencies/quantize_wrapper_v2_24/vars/2":
    "layers/dscnn/layers/separable_conv2d_3/vars/0",     # depthwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_26/vars/2":
    "layers/dscnn/layers/separable_conv2d_3/vars/1",     # pointwise kernel
"_layer_checkpoint_dependencies/quantize_wrapper_v2_24/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_3/vars/2",     # dw bias
"_layer_checkpoint_dependencies/quantize_wrapper_v2_26/layer/vars/0":
    "layers/dscnn/layers/separable_conv2d_3/vars/3",     # pw bias


# ============ Dense (FC) ============
"_layer_checkpoint_dependencies/quantize_wrapper_v2_30/vars/2":
    "layers/dscnn/fc/vars/0",                # kernel (144,4)
"_layer_checkpoint_dependencies/quantize_wrapper_v2_30/layer/vars/0":
    "layers/dscnn/fc/vars/1",                # bias (4,)
}



# ---------- Quantization helpers ----------
weights = {}
with h5py.File(weights_path, "r") as f:
    def visit_fn(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape  # <— Get dataset shape
            print(name, shape)
            if name in OLD_TO_STANDARD:
                # print(name, shape)
                standard_name = OLD_TO_STANDARD[name]
                weights[standard_name] = np.array(obj)
                # print(name, standard_name)
            else:
                # print(name)
                weights[name] = np.array(obj)
    f.visititems(visit_fn)



# ----------------------------
# Helper functions (Conv, BN, ReLU, Pool, Dense)
# ----------------------------
# (Keep all conv2d, depthwise_separable_conv2d, batch_norm, relu, global_avg_pool, dense, dscnn_numpy_inference as before)
import numpy as np
from numpy.lib.stride_tricks import as_strided

# ----------------------------
# Helper functions
# ----------------------------

def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def pad_same_keras(in_size, k, stride):
    # Padding formula to match Keras 'same' conv behavior
    out_size = int(np.ceil(in_size / stride))
    pad_total = max((out_size - 1) * stride + k - in_size, 0)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return pad_before, pad_after

def conv2d_valid_strided_vec(x, kernel, Sh=1, Sw=1):
    # Vectorized 2D convolution (valid)
    H, W = x.shape
    kh, kw = kernel.shape
    out_h = (H - kh) // Sh + 1
    out_w = (W - kw) // Sw + 1
    shape = (out_h, out_w, kh, kw)
    strides = (x.strides[0]*Sh, x.strides[1]*Sw, x.strides[0], x.strides[1])
    patches = as_strided(x, shape=shape, strides=strides)
    # Keras Conv2D is actually cross-correlation, so no flipping
    out = np.einsum('ijxy,xy->ij', patches, kernel)
    return out

def conv2d(x, w, bias, stride=(1, 1), padding='same'):
    batch, H, W, C_in = x.shape
    Kh, Kw, _, C_out = w.shape
    print("Conv2d tf shape:", w.shape)
    Sh, Sw = stride

    if padding == 'same':
        pad_top, pad_bottom = pad_same_keras(H, Kh, Sh)
        pad_left, pad_right = pad_same_keras(W, Kw, Sw)
        # print(f"[Conv2d] Input HxW: {H}x{W}, Kernel: {Kh}x{Kw}, Stride: {Sh}x{Sw}, Padding top/bottom: {pad_top}/{pad_bottom}, left/right: {pad_left}/{pad_right}")
        x_padded = np.pad(x, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right),(0,0)), mode='constant')
    else:
        x_padded = x

    H_out = (x_padded.shape[1] - Kh)//Sh + 1
    W_out = (x_padded.shape[2] - Kw)//Sw + 1
    # print(f"[Conv2d] Output HxW: {H_out}x{W_out} formula: (H + pad_top + pad_bottom - Kh)//Sh + 1, (W + pad_left + pad_right - Kw)//Sw + 1")
    out = np.zeros((batch, H_out, W_out, C_out), dtype=np.float32)
    
    for b in range(batch):
        for co in range(C_out):
            for ci in range(C_in):
                out[b,:,:,co] += conv2d_valid_strided_vec(x_padded[b,:,:,ci], w[:,:,ci,co], Sh, Sw)
    if bias is not None:
        out += bias.reshape(1, 1, 1, C_out)
    return out

def batch_norm(x, gamma, beta, mean, var, eps=1e-3):
    # Match Keras default epsilon
    return gamma * (x - mean)/np.sqrt(var + eps) + beta

def relu(x):
    return np.maximum(0, x)

def global_avg_pool(x):
    return np.mean(x, axis=(1,2))
def dense(x, w, b):
    return x @ w + b
def depthwise_separable_conv2d(x, depth_w, bias_dw, point_w, bias_pw, stride=(1,1), padding='same'):
    batch, H, W, C_in = x.shape
    Kh, Kw, _, _ = depth_w.shape
    print("Tensorflow weights shape: ", depth_w.shape)
    Sh, Sw = stride
    if padding == 'same':
        pad_top, pad_bottom = pad_same_keras(H, Kh, Sh)
        pad_left, pad_right = pad_same_keras(W, Kw, Sw)
        x_padded = np.pad(x, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right),(0,0)), mode='constant')
    else:
        x_padded = x

    # Depthwise conv
    H_out = (x_padded.shape[1] - Kh)//Sh + 1
    W_out = (x_padded.shape[2] - Kw)//Sw + 1
    dw_out = np.zeros((batch, H_out, W_out, C_in), dtype=np.float32)
    for b in range(batch):
        for ci in range(C_in):
            dw_out[b,:,:,ci] = conv2d_valid_strided_vec(x_padded[b,:,:,ci], depth_w[:,:,ci,0], Sh, Sw)
    # Pointwise con
    dw_out += bias_dw.reshape(1,1,1,C_in)
    dw_out = relu(dw_out)
    _, H_dw, W_dw, C_in_dw = dw_out.shape
    _, _, C_in_pw, C_out = point_w.shape
    out = np.zeros((batch, H_dw, W_dw, C_out), dtype=np.float32)
    for b in range(batch):
        # Tensordot over input channels
        out[b] = np.tensordot(dw_out[b], point_w[0,0,:,:], axes=([2],[0]))

        # ⭐ Add BN-folded bias now
    out += bias_pw.reshape(1, 1, 1, C_out)
    return out.astype(np.float32)
def dscnn_numpy_inference(x_input, model_settings, model_size_info):
    num_layers = model_size_info[0]
    batch = x_input.shape[0]
    print(x_input.shape)
    print(model_settings['spectrogram_length'])
    print(model_settings['mel_bin_count'])
    # x = tf.reshape(x_input, (-1, model_settings['spectrogram_length'], model_settings['mel_bin_count'], 1))
    x = x_input
    print(x.shape)
    i = 1
    for layer_no in range(num_layers):
        conv_feat = model_size_info[i]; i+=1
        conv_kt   = model_size_info[i]; i+=1
        conv_kf   = model_size_info[i]; i+=1
        conv_st   = model_size_info[i]; i+=1
        conv_sf   = model_size_info[i]; i+=1

        if layer_no == 0:
            w = weights["layers/dscnn/layers/conv2d/vars/0"].astype(np.float32)
            b = weights["layers/dscnn/layers/conv2d/vars/1"].astype(np.float32)
            x = np.pad(x, ((0,0),(5,5),(2,2),(0,0)), mode='constant')
            print("Going in")
            x = conv2d(x, w, b, stride=(conv_st, conv_sf), padding='valid')
            print("Conv2d: ", x.shape)
        else:
            sep_index = layer_no - 1
            sep_name = f"layers/dscnn/layers/separable_conv2d_{sep_index}"
            dw = weights[f"{sep_name}/vars/0"].astype(np.float32)
            pw = weights[f"{sep_name}/vars/1"].astype(np.float32)
            dwb = weights[f"{sep_name}/vars/2"].astype(np.float32)
            pwb = weights[f"{sep_name}/vars/3"].astype(np.float32)
            x = np.pad(x, ((0,0),(1,1),(1,1),(0,0)), mode='constant')
            
            x = depthwise_separable_conv2d(x, dw,dwb, pw, pwb, stride=(conv_st, conv_sf), padding='valid')
            # return x
            print(x.shape)
            print("Depthwise done")
        # x = batch_norm(x, gamma, beta, mean, var)
        x = relu(x)
    x = global_avg_pool(x)
    # print(x.shape)
    w_fc = weights["layers/dscnn/fc/vars/0"].astype(np.float32)
    b_fc = weights["layers/dscnn/fc/vars/1"].astype(np.float32)
    x = dense(x, w_fc, b_fc)
    # print(x.shape)
    return x
# ============================================================
# Helper functions
# ============================================================

import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Any, Tuple, Dict, List # Assuming these are available

# --- Utility Functions (Assumed to be correct/necessary for the pipeline) ---


def conv2d_valid_strided_vec_int8(x_int8: np.ndarray, kernel_int8: np.ndarray, x_zp: int, w_zps: int, Sh: int = 1, Sw: int = 1) -> np.ndarray:
    """
    Performs INT8 x INT8 -> INT32 Accumulation for a single input/output channel pair.
    Assumes w_zps=0 (symmetric weights) for simplification.
    Output shape: [out_h, out_w]
    """
    H, W = x_int8.shape
    kh, kw = kernel_int8.shape
    out_h = (H - kh) // Sh + 1
    out_w = (W - kw) // Sw + 1
    
    # Calculate strides for sliding window view
    shape = (out_h, out_w, kh, kw)
    strides = (x_int8.strides[0]*Sh, x_int8.strides[1]*Sw, x_int8.strides[0], x_int8.strides[1])
    
    # Create sliding windows (patches)
    patches = as_strided(x_int8, shape=shape, strides=strides).astype(np.int32)
    kernel_int32 = kernel_int8.astype(np.int32)
    
    # Accumulation: (Patch - X_ZP) dot (Kernel - W_ZP)
    # Since W_ZP=0, this is: sum( (x_q - Z_x) * w_q ) = sum(x_q * w_q) - Z_x * sum(w_q)
    # This correctly incorporates the ZP correction term when W_ZP=0.
    acc = np.einsum('ijxy,xy->ij', (patches), (kernel_int32))
    return acc

# --- Quantized Layer Implementations (MODIFIED) ---

def quantized_global_average_pooling(x_int8: np.ndarray, input_scale: float) -> np.ndarray:
    """
    Performs Global Average Pooling using dequantize-float-requantize method.
    Output quantization parameters are the same as the input.
    """
    # 1. Dequantize to float
    x_float = (x_int8.astype(np.float32)) * input_scale
    
    # 2. Average across spatial dimensions (H and W)
    avg_float = np.mean(x_float, axis=(1, 2), keepdims=True) # [B, 1, 1, C]
    # return avg_float
    # 3. Re-quantize back to INT8 using the *same* input quantization parameters
    output_scale = 0.055471875
    
    # Clamp to INT8 range [-128, 127]
    avg_int8 = np.clip(
        np.round(avg_float / output_scale), 
        -128, 
        127
    ).astype(np.int8)
    
    return avg_int8

def quantized_separable_conv_layer(
    x_int8: np.ndarray,
    input_scale: float,
    depth_params: Dict[str, Any],
    point_params: Dict[str, Any],
    stride: Tuple[int, int] = (1, 1),
    padding: str = "same",
) -> np.ndarray:
    """
    Fused quantized SeparableConv:
      - Depthwise: INT8 x INT4 -> INT32 (no bias, no requant)
      - Pointwise: INT32 x INT4 -> INT32, + bias
      - Requantize to INT8 with final output scale
      - ReLU

    Assumes:
      * Symmetric quantization (ZP = 0) for activations + weights
      * Bias is already BN-folded and in INT32
      * You have:
          depth_params["W"], depth_params["s_w"]
          point_params["W"], point_params["B"], point_params["s_w"], point_params["s_y"]
    """
    B, H, W, C_in = x_int8.shape
    Wd = depth_params["W"].astype(np.int8)   # [Kh, Kw, Cin, Cmult]
    s_wd = float(depth_params["s_w"])

    Wp = point_params["W"].astype(np.int8)   # [1, 1, Cin*Cmult, Cout]
    B_pw = point_params["B"].astype(np.int32)
    s_wp = float(point_params["s_w"])
    s_y  = float(point_params["s_y"])

    Kh, Kw, _, C_mult = Wd.shape
    Sh, Sw = stride

    x_padded = x_int8

    H_out = (x_padded.shape[1] - Kh) // Sh + 1
    W_out = (x_padded.shape[2] - Kw) // Sw + 1

    # ---------------------------------------------------
    # 1. DEPTHWISE: INT8 x INT4 -> INT32 accumulator
    # ---------------------------------------------------
    # Shape: [B, H_out, W_out, C_in * C_mult]
    dw_acc = np.zeros((B, H_out, W_out, C_in * C_mult), dtype=np.int32)

    for ci in range(C_in):
        for m in range(C_mult):
            co_dw = ci * C_mult + m
            k_dw = Wd[:, :, ci, m]  # [Kh, Kw], int8 (really int4 range)

            for b in range(B):
                dw_acc[b, :, :, co_dw] = conv2d_valid_strided_vec_int8(
                    x_padded[b, :, :, ci],
                    k_dw,
                    x_zp=0,
                    w_zps=0,
                    Sh=Sh,
                    Sw=Sw,
                )
    #  print(dw_acc)
    dw_acc = dw_acc * s_wp
    # ---------------------------------------------------
    # 2. POINTWISE: INT32 (from depthwise) x INT4 -> INT32 + bias
    # ---------------------------------------------------
    Cin_pw = C_in * C_mult
    _, _, Cin_pw_w, Cout = Wp.shape
    assert Cin_pw_w == Cin_pw, "Pointwise kernel Cin must match depthwise output channels"

    # We'll compute:
    # acc[b,h,w,co] = sum_ci dw_acc[b,h,w,ci] * Wp[0,0,ci,co] + B_pw[co]
    acc_pw = np.zeros((B, H_out, W_out, Cout), dtype=np.int64)  # int64 for safe accumulation
    # B_pw = B_pw * (s_wp * input_scale) / s_y
    Wp_int32 = Wp.astype(np.int32)

    for co in range(Cout):
        # Flatten Cin dimension loop for better locality
        for ci in range(Cin_pw):
            w_val = Wp_int32[0, 0, ci, co]  # scalar
            # broadcast multiply-accumulate
            acc_pw[:, :, :, co] += dw_acc[:, :, :, ci].astype(np.int64) * int(w_val)

        acc_pw[:, :, :, co] += B_pw[co].astype(np.int64)

    # ---------------------------------------------------
    # 3. REQUANTIZE: INT32 -> INT8 with combined scale
    #
    # float_y = acc_pw * (s_x * s_wd * s_wp)
    # y_q     = round(float_y / s_y)
    #        = round(acc_pw * (s_x * s_wd * s_wp / s_y))
    # ---------------------------------------------------
    acc_pw = np.maximum(acc_pw, 0)
    combined_rescale = (input_scale * s_wp) / s_y

    acc_pw_f = acc_pw.astype(np.float32) * combined_rescale
    q_out = np.clip(
        np.round(acc_pw_f),
        -128,
        127
    ).astype(np.int8)

    # ---------------------------------------------------
    # 4. ReLU in INT8 domain (since symmetric, ZP=0)
    # ---------------------------------------------------
    # q_out = np.maximum(q_out, 0).astype(np.int8)

    return q_out

def quantized_depthwise_conv(x_int8: np.ndarray, input_scale: float, block_d: Dict[str, Any], stride: Tuple[int, int] = (1, 1), padding: str = "same") -> np.ndarray:
    """
    Performs quantized Depthwise Conv + BiasAdd + ReLU.
    NOTE: Padding is expected to be handled externally if padding='valid' is used.
    """
    B, H, W, C_in = x_int8.shape
    print(x_int8.shape)
    Kh, Kw, _, C_multiplier = block_d['W'].shape
    print(block_d["W"].shape)
    Sh, Sw = stride
    print("in")
    if padding == "same":
        # This block is likely unused if padding is handled externally, but kept for completeness
        pad_top, pad_bottom = pad_same_keras(H, Kh, Sh)
        pad_left, pad_right = pad_same_keras(W, Kw, Sw)
        x_padded = np.pad(x_int8, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                          mode='constant', constant_values=0)
    else:
        x_padded = x_int8

    H_out = (x_padded.shape[1] - Kh) // Sh + 1
    W_out = (x_padded.shape[2] - Kw) // Sw + 1

    acc_out = np.zeros((B, H_out, W_out, C_in * C_multiplier), dtype=np.int8)

    Wd = block_d["W"]
    Bd = block_d["B"]
    scales_w = block_d["s_w"]
    scale_out = block_d["s_y"]
    # Assuming per-tensor weight scale for simplicity based on provided code structure
    scale_w = scales_w
    w_zp_single = 0 # Symmetric weight ZP
    print("DW: ")
    # print(scale_w)
    # print(scale_out)
    rescale_factor = (input_scale * scale_w) / scale_out
    # print(rescale_factor)
    # print(C_multiplier)
    print("Before acc")
    # Optimization: Process all channels and batches together if possible (not done here for simplicity/readability)
    for ci in range(C_in):
        # Output channel index
        co = ci * C_multiplier 
        
        acc = np.zeros((B, H_out, W_out), dtype=np.int32)
        
        for b in range(B):
            # Accumulation for one batch and one input channel
            acc[b] = conv2d_valid_strided_vec_int8(
                x_padded[b, :, :, ci], 
                Wd[:, :, ci, 0],  # Get the [Kh, Kw] kernel for input channel ci
                0, 
                w_zp_single, 
                Sh, Sw
            )
        

        # 3. Add Bias (INT32)
        acc += Bd[ci]
        
        # 4. ReLU (Clip to 0)
        acc_relu = np.maximum(0, acc)
        # print(acc_relu.shape)
        # print(acc_relu)
        # print(rescale_factor.shape)
        # 5. Requantization (INT32 -> INT8)
        acc_rescaled = acc_relu.astype(np.float32) * rescale_factor
        # print(acc_rescaled.shape)
        # print(acc_rescaled)
        # Clamp to INT8 range [-128, 127]
        q_out = np.clip(
            np.round(acc_rescaled), 
            -128, 
            127
        ).astype(np.int8)

        # Store result in the correct output channel slice
        acc_out[:, :, :, co] = q_out
        
    return acc_out
def quantized_conv_relu_layer(x_int8: np.ndarray, input_scale: float, params: Dict[str, Any], i: int, stride: Tuple[int, int] = (1, 1), padding: str = "same") -> np.ndarray:
    """
    Performs the entire quantized Conv2D + BiasAdd + ReLU operation.
    NOTE: Padding is expected to be handled externally if padding='valid' is used.
    """
    B, H, W, C_in = x_int8.shape
    
    # Parameter loading logic
    if i == -1: # Initial Conv1
        prefix = "quant_conv2d"
        w_int8 = params[f"{prefix}_kernel_int8"]
        b_int32 = params[f"{prefix}_bias_int32"]
        scales_w = float(params[f"{prefix}_kernel_scale"])
        scale_out = float(params[f"{prefix}_output_scale"])
    else: # Pointwise Convs
        w_int8 = params["W"]
        b_int32 = params["B"]
        scales_w = params['s_w']
        scale_out = params["s_y"]
        
    print(w_int8)
    Kh, Kw, _, C_out = w_int8.shape
    Sh, Sw = stride

    x_padded_int8 = x_int8
    print("Padded:")
    H_out = (x_padded_int8.shape[1] - Kh) // Sh + 1
    W_out = (x_padded_int8.shape[2] - Kw) // Sw + 1
    
    acc_out = np.zeros((B, H_out, W_out, C_out), dtype=np.int8)
    print(scales_w)
    scale_w = scales_w
    w_zp_single = 0
    rescale_factor = (input_scale * scale_w) / scale_out
    print(rescale_factor)
    for co in range(C_out):
        acc = np.zeros((B, H_out, W_out), dtype=np.int32)
        
        for ci in range(C_in):
            
            w_kernel_int8 = w_int8[:, :, ci, co] # [Kh, Kw] kernel slice
            
            for b in range(B):
                # Accumulation for one batch element
                acc[b] += conv2d_valid_strided_vec_int8(
                    x_padded_int8[b, :, :, ci], 
                    w_kernel_int8, 
                    0, 
                    w_zp_single, 
                    Sh, Sw
                )
        
        # 3. Add Bias (INT32)
        acc += b_int32[co]
        # print(acc)
        # 4. ReLU (Clip to 0)
        acc_relu = np.maximum(0, acc)
        # print(acc)
        # 5. Requantization (INT32 -> INT8)
        acc_rescaled = np.round(acc_relu.astype(np.float32) * rescale_factor).astype(np.int32)
        # print(acc_rescaled)
        # Clamp to INT8 range [-128, 127]
        q_out = np.clip(
            np.round(acc_rescaled), 
            -128, 
            127
        ).astype(np.int8)

        acc_out[:, :, :, co] = q_out
    
    # acc_out = np.maximum(0, acc_out).astype(np.int8)
    return acc_out


def quantized_dense_layer(x_int8: np.ndarray, input_scale: float, params: Dict[str, Any]) -> np.ndarray:
    """
    MODIFIED: Performs quantized Dense (MatMul + BiasAdd) using efficient batch matrix multiplication.
    Assumes Keras weight shape: [Cin, Cout]. Input must be flattened: [B, Cin].
    """
    B, C_in = x_int8.shape
    w_int8 = params["quant_fc_kernel_int8"] # [Cin, Cout]
    b_int32 = params["quant_fc_bias_int32"]
    scales_w = float(params["quant_fc_kernel_scale"])
    scale_out = float(params["quant_fc_output_scale"])
    # zp_out = int(params["quant_dense_output_zp"])
    w_zp_single = 0 # Symmetric weight ZP
    print("Params extracted:")
    C_out = w_int8.shape[-1]
    
    # --- 1. Accumulation (MatMul: INT8 x INT8 -> INT32) ---
    # Input subtraction: [B, Cin] - ZP
    X_q_sub = x_int8.astype(np.int32)
    
    # Weight subtraction: [Cin, Cout] - ZP. Since W_ZP=0, we just use the kernel.
    W_q_sub = w_int8.astype(np.int32) # W_ZP=0 here
    
    # Matrix Multiplication: [B, Cin] @ [Cin, Cout] = [B, Cout]
    # This efficiently performs: sum( (x_q - Z_x) * w_q )
    acc = X_q_sub @ W_q_sub
    
    # --- 2. Add Bias (INT32) ---
    acc += b_int32 # [B, Cout] + [Cout] broadcast = [B, Cout]
    
    # --- 3. Requantization (INT32 -> INT8) ---
    scale_w = scales_w
    rescale_factor = (input_scale * scale_w) / scale_out

    # No ReLU assumed for the final dense layer
    acc_rescaled = acc.astype(np.float32) * rescale_factor
    
    # Clamp to INT8 range [-128, 127]
    q_out = np.clip(
        np.round(acc_rescaled), 
        -128, 
        127
    ).astype(np.int8)
    print("Return")
    return q_out

# ============================================================
# Full model inference
# ============================================================

def run_full_model(params: Dict[str, Any], rand_input: np.ndarray, model_settings: Dict[str, Any]) -> np.ndarray:
    """
    Performs quantized inference for the full DS-CNN-like model.
    """
    H = model_settings["spectrogram_length"]
    W = model_settings["mel_bin_count"]
    params = dict(params)
    # --- Input Quantization ---
    # Define Asymmetric Parameters for range [-1.0, 1.0] (Based on previous analysis)
    s_x = params["Input_scale"]
    z_x = 0
    print(s_x)
        
    
    # Quantize input data
    x_q = np.clip(np.round(rand_input / s_x) + z_x, -128, 127).astype(np.int8)
    # ---------- CONV 1 (Padded and Run) ----------
    # Padding handled externally for Conv1, using the fixed values from the original script
    # This assumes the original model used this specific padding for 'valid' output.
    x_q = np.pad(x_q, ((0, 0), (5, 5), (2, 2), (0, 0)), mode='constant', constant_values=0)
    
    x_q = quantized_conv_relu_layer(x_q, s_x, params, -1, stride=(3, 3), padding="valid")
    s_x = float(params["quant_conv2d_output_scale"])
    print(s_x)
    print(x_q)
    """
    x_float = (x_q.astype(np.float32)) * s_x
    return x_float"""
    # x_float = np.maximum(x_float, 0.0)
    # x_float = batch_norm(x_float, gamma,beta,mean,var,eps)
    # Convert quantized output back to float BEFORE ReLU
    # Apply ReLU in float domain
    # x_q = np.clip(np.round(x_float / s_x), -128, 127).astype(np.int8)
    print("Conv done")
    # ---------- SEPARABLE BLOCKS (4 of them) ----------
    dw_blocks = {}
    pw_blocks = {}
    
    for i in range(4):
        suffix = f"_{i}"
        
        # Load parameters for Depthwise Conv
        dw_blocks[i] = {
            "W": params[f"quant_depthwise_conv2d{suffix}_kernel_int8"],
            "B": params[f"quant_depthwise_conv2d{suffix}_bias_int32"],
            "s_w": float(params[f"quant_depthwise_conv2d{suffix}_kernel_scale"]),
            "s_y": float(params[f"quant_depthwise_conv2d{suffix}_output_scale"]),

        }

        # Load parameters for Pointwise Conv
        pw_blocks[i] = {
            "W": params[f"quant_pointwise_conv2d{suffix}_kernel_int8"],
            "B": params[f"quant_pointwise_conv2d{suffix}_bias_int32"],
            "s_w": float(params[f"quant_pointwise_conv2d{suffix}_kernel_scale"]),
            "s_y": float(params[f"quant_pointwise_conv2d{suffix}_output_scale"]),
        }
        # s_x = 1.0
        print("Separable extraction done")
        # Depthwise Block
        # Padding handled externally for 3x3 kernel, stride 1, same padding -> (1, 1) pad
        x_q = np.pad(x_q, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        x_q = quantized_depthwise_conv(x_q, s_x, dw_blocks[i], stride=(1, 1), padding="valid")
        print("Depthwise without bn")
        s_x = dw_blocks[i]['s_y']
        # x_float = (x_q.astype(np.float32)) * s_x
        # return x_float
        # x_float = np.maximum(x_float, 0.0)
        # Convert quantized output back to float BEFORE ReLU
        # Apply ReLU in float domain
        # x_float = np.maximum(x_float, 0.0)
        # x_q = np.clip(np.round(x_float / s_x), -128, 127).astype(np.int8)
        # Pointwise Block
        print("Before pw")
        x_q = quantized_conv_relu_layer(x_q, s_x, pw_blocks[i], i, stride=(1, 1), padding="valid")
        s_x = pw_blocks[i]['s_y']
        """x_float = (x_q.astype(np.float32)) * s_x
        x_float = np.maximum(x_float, 0.0)

        x_q = np.clip(np.round(x_float / s_x ), -128, 127).astype(np.int8)"""
    # ---------- GAP ----------
    """
    x_float = (x_q.astype(np.float32)) * s_x
    return x_float"""
    x_q = quantized_global_average_pooling(x_q, s_x)
    B, H, W, C = x_q.shape
    print(x_q.shape)
    # Flatten: [B, 1, 1, C] -> [B, C]
    flat_input_int8 = x_q.reshape(B, H * W * C)
    s_x = 0.055471
    # ---------- Dense ----------
    x_q = quantized_dense_layer(flat_input_int8, s_x, params)
    
    # Final Dequantization to logits (float output)
    s_out = float(params["quant_fc_output_scale"])

    logits = (x_q.astype(np.float32)) * s_out

    return logits
    # GAP → convert to int domain

    """print("gap float min/max:", gap.min(), gap.max())
    gap_int8 = np.round(gap / s_yfc + zp_yfc).astype(np.int8)
    print("gap_int8 unique:", np.unique(gap_int8))
    print(gap_int8.shape)
    B, H, W, C = gap_int8.shape
            # Now H and W should be 1, and C should be 276.
    flat_input_int8 = gap_int8.reshape(B, H * W * C)
    # Gap activations scale/zero-point
    s_a = s_yfc
    zp_a = zp_yfc

    # Convert input to int32 and subtract zp
    a_shifted = flat_input_int8.astype(np.int32) - zp_a   # (B, in_dim)

    # Compute int32 logits
    W_int32 = W_fc.astype(np.int32)                        # (in_dim, out_dim)
    logits_int32 = a_shifted @ W_int32 + B_fc              # (B, out_dim)

    # Requantize to int8
    mult = (s_a * s_wfc) / s_out
    logits_q = np.clip(
        np.round(logits_int32 * mult) + zp_out,
        -128, 127
    ).astype(np.int8)

    # Convert back to float for debugging / comparison
    logits_float = (logits_q.astype(np.float32) - zp_out) * s_out
    # int32 accumulation
    return logits_float"""



# ============================================================
# Example Run
# ============================================================

params = np.load("checkpointsLogmelUpdatedMediumQATV10/extracted_params.npz")
"""
rand_input = np.random.randn(1, 92, 64, 1).astype(np.float32)


output = run_full_model(params, rand_input, model_settings)
print("Numpy done")

tf_conv1_float = dscnn_numpy_inference(rand_input,model_settings,model_size_info)
print("TF Conv1 Output Shape:", tf_conv1_float.shape)

y1_np_float = output
print("Float returned")
print(y1_np_float.shape)
# ----------------------------------------
# 5. Compare outputs
# ----------------------------------------
diff = y1_np_float - tf_conv1_float
print("Final Outputs MAE:", np.mean(np.abs(diff)))
print("Final Outputs Max Error:", np.max(np.abs(diff)))

print("TF first values:", tf_conv1_float.flatten()[:10])
print("NP first values:", y1_np_float.flatten()[:10])
print("Diff first values:", diff.flatten()[:10])


import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Prepare dummy input
input_name = session.get_inputs()[0].name
input_data = rand_input

# Run inference
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})

print("Output shape:", output[0].shape)
print(output)"""


import onnxruntime as ort
import numpy as np

def load_wav_file(filename, desired_samples=None):
    wav_data = tf.io.read_file(filename)
    audio, sample_rate = tf.audio.decode_wav(wav_data, desired_channels=1)
    audio = tf.squeeze(audio)
    return audio, tf.cast(sample_rate, tf.int32)
def rms_normalize_frames(audio, window_size, target_rms=0.03, eps=1e-9):
    # Frame the audio with no overlap (or small overlap if desired)
    frames = tf.signal.frame(audio, window_size, window_size, pad_end=True)

    # Compute RMS per frame
    rms = tf.sqrt(tf.reduce_mean(tf.square(frames), axis=1, keepdims=True) + eps)

    # Compute scaling factors frame-wise
    scaling = target_rms / (rms + eps)

    # Apply scaling
    norm_frames = frames * scaling

    # Reconstruct waveform
    norm_audio = tf.reshape(norm_frames, [-1])
    return norm_audio[: tf.shape(audio)[0]]  # trim padded samples
# ----------------------------
# WAV -> Log-Mel in sliding windows
# ----------------------------
def wav_to_log_mel_tf_long(wav_path, model_settings, frame_duration=1.5, stride_duration=0.2):
    # Convert a long WAV to a sequence of log-Mel spectrogram windows.
  
    waveform, sample_rate_tf = load_wav_file(wav_path,model_settings['desired_samples'])
    # waveform = waveform / tf.reduce_max(tf.abs(waveform) + 1e-9)
    waveform = rms_normalize_frames(waveform, window_size=512, target_rms=0.03)
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    Fs = 16000
    # Window parameters
    frame_size = int(frame_duration * Fs)
    stride_size = int(stride_duration * Fs)
    all_windows = []
    desired_samples = model_settings['desired_samples']
    # waveform = waveform[:desired_samples]
    padded_len = tf.shape(waveform)[0]
    waveform = tf.cond(
       padded_len < desired_samples,
       lambda: tf.pad(waveform, [[0, desired_samples - padded_len]]),
       lambda: waveform
    )
    detections = []
    cooldown_until = 0.0
    for start in range(0, len(waveform) - frame_size + 1, stride_size):
        chunk = waveform[start:start + frame_size]
        # chunk = chunk / tf.reduce_max(tf.abs(chunk) + 1e-9)
        # Save as .wav (playable audio ke sure you have this in settings
        # STFT
        stft = tf.signal.stft(chunk, frame_length=512, frame_step=256, fft_length=512)
        spectrogram = tf.abs(stft)
        # Mel filterbank
        num_mel_bins = model_settings["mel_bin_count"]
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=Fs,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7500.0,
        )
        mel_spec = tf.tensordot(spectrogram, mel_matrix, 1)
        mel_spec.set_shape(spectrogram.shape[:-1].concatenate(mel_matrix.shape[-1:]))

        log_mel =  tf.math.log(mel_spec + 1e-6) 
        # log_mel = tf.reshape(log_mel, [-1])
        # print(log_mel.flatten()[:10])
        log_mel = tf.expand_dims(log_mel, -1)  # Add channel
        log_mel = log_mel.numpy()
        H = model_settings["spectrogram_length"]
        W = model_settings["mel_bin_count"]
        log_mel = log_mel.reshape(1, H, W, 1)
        print("First few values of inputs: ", log_mel.flatten()[:10])
        print(log_mel.shape)
        current_time = start/Fs
        if current_time < cooldown_until:
           continue
        # x_input = log_mel[None, :]
        x = tf.reshape(log_mel, (-1, model_settings['spectrogram_length'], model_settings['mel_bin_count'], 1))
        # Load model
        session = ort.InferenceSession("quantized_model_10V4_2.onnx")
        print("Session created")
        # Prepare dummy input
        input_name = session.get_inputs()[0].name
        print(input_name)
        input_data = x.numpy().astype(np.float32)
        # Run inference
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: input_data})

        print("Output shape:", output[0].shape)
        logits_onnx = output[0]
        print(logits_onnx)
        logits_float = dscnn_numpy_inference(log_mel,model_settings,model_size_info)
        print(logits_float)
        # logits= run_full_model(params,log_mel,model_settings) # Output info is not strictly needed here
        # print(logits)
        diff = logits_onnx - logits_float
        print("Final Outputs MAE:", np.mean(np.abs(diff)))
        print("Final Outputs Max Error:", np.max(np.abs(diff)))

        print("TF first values:", logits_float.flatten()[:10])
        print("NP first values:", logits_onnx.flatten()[:10])
        # print("Diff first values:", diff.flatten()[:10])
        probs_float = tf.nn.softmax(logits_float, axis=-1).numpy()
        # probs = tf.nn.softmax(logits_onnx, axis=-1).numpy()
        
        print("Current time: ", current_time)
        print(probs_float)
        # print(probs)
        pred_label = int(np.argmax(probs_float))
        
        conf = float(np.max(probs_float))
        if pred_label == 2 and conf >= 0.70:
           end_time = current_time + frame_duration
           detections.append((end_time, conf))
           print(f"✅ Detected at {end_time:.2f}s (conf {conf:.2f})")
           cooldown_until = current_time + 1.5  # add 2s directly
    print("\nSummary of detections:", detections)# shape: (num_windows, H, W)




# Long WAV input

wav_path = "heysnips_2.wav"
x_windows = wav_to_log_mel_tf_long(wav_path,model_settings)

