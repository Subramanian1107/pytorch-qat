import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import input_data_debug  # your rewritten AudioProcessor
import models_debug             # your model definitions
import tensorflow_model_optimization as tfmot
from qkeras import QConv2D, QDepthwiseConv2D, QDense
from qkeras.quantizers import quantized_bits
# ---------------- GPU Setup ---------------- #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs available:", gpus)
    except RuntimeError as e:
        print(e)
from tensorflow.keras import Model, Sequential
def get_model_type(model):
    """
    Returns the type of a Keras model as a string:
    'Sequential', 'Functional', or 'Subclassed'.
    """
    if isinstance(model, Sequential):
        return 'Sequential'
    elif isinstance(model, Model):
        if getattr(model, "_is_graph_network", False):
            return 'Functional'
        else:
            return 'Subclassed'
    else:
        return 'Unknown'
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_apply, quantize_scope
from tensorflow.keras import layers, Model, Input
# from tensorflow_model_optimization.python.core.quantization.keras import quantizers
 
# from tensorflow_model_optimization.python.core.quantization.keras import QuantizeConfig
class CustomRangeQuantizer(tfmot.quantization.keras.quantizers.LastValueQuantizer):
    def __init__(self, num_bits=4, symmetric=True, narrow_range=True,per_axis=False):
        super().__init__(num_bits=num_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis)
 
class DSCNNQuantizeSepConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.depthwise_kernel, CustomRangeQuantizer())]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.depthwise_kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        return []

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, 
            per_axis=False, 
            symmetric=True, 
            narrow_range=False
        )]

    def get_config(self):
        return {}
class DSCNNQuantizeSeparableConfig(tfmot.quantization.keras.QuantizeConfig):
    """
    QuantizeConfig for SeparableConv2D:
      • INT4 weights for BOTH depthwise + pointwise kernels
      • 8-bit activations (MovingAverageQuantizer)
    """
    def get_weights_and_quantizers(self, layer):
        quantizers = []
        if hasattr(layer, "depthwise_kernel"):
            quantizers.append((layer.depthwise_kernel, CustomRangeQuantizer()))
        if hasattr(layer, "pointwise_kernel"):
            quantizers.append((layer.pointwise_kernel, CustomRangeQuantizer()))
        return quantizers

    def get_activations_and_quantizers(self, layer):
        # No input-activation quantizers here — QAT will add output FakeQuant
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        index = 0
        if hasattr(layer, "depthwise_kernel"):
            layer.depthwise_kernel = quantize_weights[index]
            index += 1
        if hasattr(layer, "pointwise_kernel"):
            layer.pointwise_kernel = quantize_weights[index]

    def set_quantize_activations(self, layer, quantize_activations):
        return []

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8,
            per_axis=False,
            symmetric=True,
            narrow_range=False
        )]

    def get_config(self):
        return {}

class DSCNNInputQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        return []

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, 
            per_axis=False, 
            symmetric=True, 
            narrow_range=False
        )]

    def get_config(self):
        return {}
class DSCNNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, CustomRangeQuantizer())]
 
    def get_activations_and_quantizers(self, layer):
        return []
 
    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]
 
    def set_quantize_activations(self, layer, quantize_activations):
        return []
 
    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, 
            per_axis=False, 
            symmetric=True, 
            narrow_range=False
        )]
 
    def get_config(self):
        return {}
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from typing import Tuple, Optional, Any

def get_kernel_quant_params_fixed(layer: Any) -> Tuple[Optional[float], Optional[float]]:
    """Extracts the kernel's fixed R_min and R_max from a QuantizeWrapper."""
    print(layer.name)
    print(layer.weights)
    weights_float = layer.get_weights()[0]
    qmin = -7
    qmax = 7
    scale = np.max(np.abs(weights_float)) / 7
    return scale
def get_activation_scale_zp(layer) :
    """Extracts the output activation scale and ZP from a QuantizeWrapper."""
    print(layer.name)
    print(layer.weights)
    R_min, R_max = None, None
    for i, w in enumerate(layer.weights):
        name = w.name.lower()
        value = layer.get_weights()[i]
        if 'output_min' in name:
            R_min = value.item()
        elif 'output_max' in name:
            R_max = value.item()
            
    if R_min is None or R_max is None:
        # Fallback for untracked/fixed range activations
        return 1.0
    
    # Activation is assumed 8-bit signed: [-128, 127]
    Q_min_a, Q_max_a = -128, 127
    print("Activation scale range ", R_max, R_min)
    scale_a = calculate_scale_zp(R_min, R_max, Q_min_a, Q_max_a)
    return scale_a

def calculate_scale_zp(R_min, R_max, Q_min, Q_max) :
    """Calculates Scale (alpha) and Zero Point (beta) based on float and integer ranges."""
    scale =  max(abs(R_min), abs(R_max))/Q_max
    if scale == 0:
        return 0.0
    return scale

def quantize_float_weights(weights_float: np.ndarray, scale: float, Q_min: int, Q_max: int) -> np.ndarray:
    """Quantizes float weights using pre-calculated scale and zp."""
    if scale == 0:
        return np.full_like(weights_float, 0, dtype=np.int8)
    print("Weights in float: ", weights_float)
    weights_int8 = np.clip(
        np.round(weights_float / scale), 
        Q_min, 
        Q_max
    ).astype(np.int8)
    return weights_int8

import numpy as np
from typing import Union, Tuple

def extract_all_quantized_params(qat_model, output_file_path: str):
    """
    Extract INT4/INT8 quantization parameters from QAT model.
    Processes only quantized weight layers (Conv2D, DepthwiseConv2D,
    SeparableConv2D, Dense) and stores:
        • weights (int8)
        • bias (int32)
        • weight scale, zp
        • output activation scale, zp
    Activation scale/ZP flow propagates layer-by-layer.
    """

    all_params = {}
    Q_min_w, Q_max_w = -7, 7   # weight int range for INT4/8

    # ---------- Input quantizer -> first activation scale ----------
    # quant_input = qat_model.get_layer("quant_input_quant")
    input_scale_i = 1.0
    all_params["Input_scale"] = np.array(input_scale_i, dtype=np.float32)
    # all_params["Input_zp"] = np.array(input_zp_i, dtype=np.int32)
    print("Input scale: ", all_params["Input_scale"])
    i = 0
    while i < len(qat_model.layers):
        layer = qat_model.layers[i]

        # Must be QAT wrapper
        if not isinstance(layer, tfmot.quantization.keras.QuantizeWrapper):
            i += 1
            continue

        wrapped = layer.layer

        # Weight-bearing layer?
        is_weight_layer = isinstance(
            wrapped,
            (keras.layers.Conv2D,
             keras.layers.SeparableConv2D,
             keras.layers.DepthwiseConv2D,
             keras.layers.Dense)
        )

        # If not weight-bearing, propagate activation scale from ReLU wrappers
        if not is_weight_layer:
            try:
                out_scale = get_activation_scale_zp(layer)
                input_scale_i = out_scale
            except Exception:
                pass
            i += 1
            continue

        layer_prefix = f"{layer.name}_"
        print(f"\n==== Extracting {layer.name} ====")

        # ---------- Weight scale from FakeQuant wrapper ----------
        W_scale_1 = get_kernel_quant_params_fixed(layer)
        A_scale = input_scale_i   # previous activation scale

        # ---------- Extract float weights and bias per layer type ----------
        weights = layer.get_weights()
        # print(weights)
            # Conv2D / DepthwiseConv2D / Dense
        W_float = weights[0]
        B_orig = weights[1] if wrapped.use_bias else np.zeros(W_float.shape[-1], dtype=np.float32)

        # Quantize kernel
        W_int8 = quantize_float_weights(W_float, W_scale_1, Q_min_w, Q_max_w)
        print("Weights: ", W_int8)
        # Quantize bias
        accum_scale = A_scale * W_scale_1
        B_int32 = np.zeros_like(B_orig, dtype=np.int32) if accum_scale == 0 else np.round(B_orig / accum_scale).astype(np.int32)
        print("Biases: ", B_int32)
        all_params[layer_prefix + "kernel_int8"] = W_int8
        all_params[layer_prefix + "bias_int32"] = B_int32
        all_params[layer_prefix + "kernel_scale"] = np.array(W_scale_1, dtype=np.float32)
        print("Kernel scale: ", W_scale_1)

        # ---------- Output activation scale (for next layer) ----------
        out_scale = get_activation_scale_zp(layer)
        all_params[layer_prefix + "output_scale"] = np.array(out_scale, dtype=np.float32)

        input_scale_i = out_scale
        i += 1

    # -------- Save NPZ --------
    np.savez(output_file_path, **all_params)
    n = len([k for k in all_params if "int8" in k])
    print(f"\n✔ Saved weights for {n} QAT layers to {output_file_path}")
    return output_file_path
def quantize_weights(model, min_val, max_val, num_bits=8):
    for layer in model.layers:
        """layer = model.get_layer("quant_conv2d") 
        print(model.summary) # Replace with your layer name"""
        print(layer.name)
        if len(layer.get_weights()) == 0:
           continue
        weights_float = layer.get_weights()[0]
        qmin = -7
        qmax = 7
        scale = (weights_float.max() - weights_float.min()) / (qmax - qmin)
        zero_point = 0
        # scale = 0.000256 
        # zeropoint = 0
        weights_int8 = np.clip(np.round(weights_float / scale + zero_point), qmin, qmax).astype(np.int8)
        print(weights_int8)
        weights_dequant = (weights_int8.astype(np.float32) - zero_point) * scale
        print("Scale:", scale)
        print("Zero point:", zero_point)
        print("Original float range:", weights_float.min(), weights_float.max())
        print("Quantized int8 range:", weights_int8.min(), weights_int8.max())
        print("Dequantized float range:", weights_dequant.min(), weights_dequant.max())     
def quantize_activations(model):
    layer = model.get_layer("quant_conv2d")
    print(model.summary) # Replace with your layer name
    R_min = None
    R_max = None
    
    # 1. Extract R_min and R_max from the layer weights
    for i, w in enumerate(layer.weights):
        name = w.name.lower()
        value = layer.get_weights()[i]
        print(name)
        # Identify the output range variables based on your list
        if 'output_min' in name:
            R_min = value.item()  # Get the scalar float value
        elif 'output_max' in name:
            R_max = value.item()
    if R_min is None or R_max is None:
        raise ValueError("Could not find 'output_min' or 'output_max' in layer weights.")

    # 2. Define the Quantization Range (Q_min, Q_max) for 8-bit signed
    # The default range for weights/activations is [-128, 127]
    Q_min = -128
    Q_max = 127
    print("Activation range ", R_max, R_min)
    # 3. Calculate Scale (alpha)
    scale = (R_max - R_min) / (Q_max - Q_min)
    
    # 4. Calculate Zero Point (beta)
    # The Zero Point is derived from the formula: beta = Q_min - (R_min / scale)
    Z_float = Q_min - (R_min / scale)
    zero_point = int(np.round(Z_float))
    
    # Clip ZP to ensure it's within the target integer range [-128, 127]
    zero_point = np.clip(zero_point, Q_min, Q_max).item()
    
    return scale, zero_point, R_min, R_max
# Example: extract from a QAT model
def convert_int4_model_to_tflite(model, tflite_path):
    weights = model.get_weights()
    int8_weights = [np.clip(np.round(w * 7), -7, 7).astype(np.int8) for w in weights]  # scale if necessary

    int8_model = keras.models.clone_model(model)
    int8_model.set_weights(int8_weights)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(tflite_path,"wb").write(tflite_model)
    print("Success")
# Simulate FakeQuant min/max (replace with actual values if known)
# ---------------- Training Script ---------------- #
def main(FLAGS):
    # ---------------- Model settings ---------------- #
    wanted_words_list = FLAGS.wanted_words.split(',')
    model_settings = models_debug.prepare_model_settings(
        len(input_data_debug.prepare_words_list(wanted_words_list)),
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.dct_coefficient_count,
        FLAGS.mel_bin_count,
        FLAGS.fft_length,
        FLAGS.mel_lower_edge_hertz,
        FLAGS.mel_upper_edge_hertz
    )

    # ---------------- Audio Processor ---------------- #
    audio_processor = input_data_debug.AudioProcessor(
        FLAGS.data_url,
        FLAGS.data_dir,
        FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.partial_percentage,
        wanted_words_list,
        FLAGS.validation_percentage,
        FLAGS.testing_percentage,
        model_settings
    )
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    
   
    fingerprint_input = tf.keras.Input(shape=(fingerprint_size,), dtype=tf.float32, name='fingerprint_input')
    """qat_model = build_ds_cnn_qkeras(model_settings, FLAGS.model_size_info)
    print("Qkeras model built")"""
    print("\n🔹 Loading pretrained float model (.h5)")
    float_model = tf.keras.models.load_model(
        FLAGS.pretrained_h5_path, compile=False
    )
    print("   Loaded model =", get_model_type(float_model))
    
    # ---- Wrap with an input quant Lambda so we can attach a custom QuantizeConfig ----
    float_model.summary()
    inp = float_model.input
    x = keras.layers.Lambda(lambda t: t, name="input_quant")(inp)
    y = float_model.layers[1](x)         # first zero_padding2d
    for lyr in float_model.layers[2:]:   # the rest of the layers
        y = lyr(y)
    model_with_input_quant = keras.Model(inp, y)
    print("✔ Inserted explicit input_quant Lambda after InputLayer")
    # ---- Quantize-annotate only supported layers ----
    # --------------------------------------
    # 3) Per-layer annotation using custom QuantizeConfigs
        # --------------------------------------
    def annotate_layer(layer):
        # 🚫 Never annotate InputLayer — required to avoid graph disconnect
         if isinstance(layer, keras.layers.InputLayer):
            print(f"KEEP InputLayer => {layer.name}")
            return layer
 # optional
        # ⭐ Annotate the Lambda "input_quant" using DSCNNInputQuantizeConfig
         if isinstance(layer, keras.layers.Lambda) and layer.name == "input_quant":
            print(f"Annotating Input Quant => {layer.name}")
            return quantize_annotate_layer(
                 layer, quantize_config=DSCNNInputQuantizeConfig()
            )

    # Annotate Conv2D
         if isinstance(layer, keras.layers.Conv2D):
             print(f"Annotating Conv2D => {layer.name}")
             return quantize_annotate_layer(
                 layer, quantize_config=DSCNNQuantizeConfig()
             )

    # Annotate SeparableConv2D
         if isinstance(layer, keras.layers.DepthwiseConv2D):
             print(f"Annotating SeparableConv2D => {layer.name}")
             return quantize_annotate_layer(
                 layer, quantize_config=DSCNNQuantizeSepConfig()
             )

    # Annotate Dense
         if isinstance(layer, keras.layers.Dense):
             print(f"Annotating Dense => {layer.name}")
             return quantize_annotate_layer(
                 layer, quantize_config=DSCNNQuantizeConfig()
             )

    # Remainder (ZeroPadding2D, ReLU, GAP etc.) unchanged
         return layer


# --------------------------------------
# 4) Build annotated model
# --------------------------------------
    annotated_model = keras.models.clone_model(
       float_model,
       clone_function=annotate_layer
    )
    print("\n🔹 Annotation completed successfully")

    # annotated_model = fuse_and_annotate(model)
    # print(annotated_model.layers)
    print("\n✅ Step 1: Annotated Functional Model Built Successfully")
    with tfmot.quantization.keras.quantize_scope(
        { 'DSCNNQuantizeConfig': DSCNNQuantizeConfig,
          'DSCNNInputQuantizeConfig': DSCNNInputQuantizeConfig,
          'DSCNNQuantizeSeparableConfig': DSCNNQuantizeSeparableConfig,
         'DSCNNQuantizeSepConfig': DSCNNQuantizeSepConfig}
    ):   
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    
    is_quantized = False
    first_wrapped_name = "None"
    qat_model.summary()
    for layer in qat_model.layers:
        if isinstance(layer, tfmot.quantization.keras.QuantizeWrapper):
            is_quantized = True
            first_wrapped_name = layer.name
            break
    if is_quantized:
        print("✅ Step 2: Quantization Applied Successfully")
        print(f"   First quantized layer found: {first_wrapped_name}")
    else:
        print("❌ Step 2: Quantization Failed")
        print("   No QuantizeWrapper layers found in the model.")
# model is now a Keras Functional model, ready for training/inference

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    # ---------------- TensorBoard ---------------- #
    train_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.summaries_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.summaries_dir, 'validation'))

    # ---------------- Training Loop ---------------- #
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    training_steps_max = sum(training_steps_list)
    best_accuracy = 0.0
    
    # Specify which label to monitor during training
    target_word = 'heysnips'
    target_index = wanted_words_list.index(target_word)

    for training_step in range(1, training_steps_max + 1):
        # --- Adjust learning rate ---
        lr = learning_rates_list[0]
        step_sum = 0
        for i, steps in enumerate(training_steps_list):
            step_sum += steps
            if training_step <= step_sum:
                lr = learning_rates_list[i]
                break
        optimizer.learning_rate.assign(lr)
        # --- Fetch batch ---
        train_fingerprints, train_labels, train_filenames = audio_processor.get_data(
            FLAGS.batch_size, 0, training_step,
            background_frequency=FLAGS.background_frequency,
            background_volume_range=FLAGS.background_volume,
            time_shift=time_shift_samples,
            mode='training',
            return_filenames = True
        )

        train_fingerprints = tf.reshape(train_fingerprints, [-1, 92, 64, 1])
        # --- Forward + Backward pass ---
        with tf.GradientTape() as tape:
            print(train_fingerprints.shape)
            logits = qat_model(train_fingerprints, training=True)
            # print(logits)
            loss_value = loss_fn(train_labels, logits)
        
        grads = tape.gradient(loss_value, qat_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, qat_model.trainable_variables))
        train_acc_metric.update_state(train_labels, logits)
        # print("Started training")
        train_acc = train_acc_metric.result().numpy()
        """if training_step % 10 == 0:
            print(f"[Step {training_step}] Loss={loss_value:.4f}, Train Acc={train_acc*100:.2f}%")
            monitor_int4_clustering(qat_model, training_step)"""
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        positive_indices = np.where(np.argmax(train_labels, axis=-1) == target_index)[0]
        for i in range(len(train_filenames)):
              filename = train_filenames[i].decode('utf-8') if isinstance(train_filenames[i], bytes) else train_filenames[i]
              label_index = np.argmax(train_labels[i])
              label_name = audio_processor.words_list[label_index]
              prob = probs[i][label_index]
              print(f"[Step {training_step}] File: {filename} - True label: {label_name}, Prob('{label_name}')={prob:.4f}")


        print(f"[Step {training_step}] Loss={loss_value:.4f}, Train Acc={train_acc*100:.2f}%")
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=training_step)
            tf.summary.scalar('accuracy', train_acc, step=training_step)
        train_acc_metric.reset_state()

        # --- Logging & validation ---
        is_last_step = (training_step == training_steps_max)
        if training_step % FLAGS.eval_step_interval == 0 or is_last_step:

            # Validation
            val_size = audio_processor.set_size('validation')
            val_accuracy_total = 0.0
            print(val_size)
            for i in range(0, val_size, FLAGS.batch_size):
                val_fingerprints, val_labels, val_filenames = audio_processor.get_data(
                    FLAGS.batch_size, i, training_step,
                    background_frequency=0.0,
                    background_volume_range=0.0,
                    time_shift=0,
                    mode='validation',
                    return_filenames = True
                )
                val_fingerprints = tf.reshape(val_fingerprints, [-1, 92, 64, 1])
                val_logits = qat_model(val_fingerprints, training=False)
                probs = tf.nn.softmax(val_logits, axis=-1).numpy()
                for i in range(len(val_filenames)):
                  filename = val_filenames[i].decode('utf-8') if isinstance(val_filenames[i], bytes) else val_filenames[i]
                  label_index = np.argmax(val_labels[i])
                  label_name = audio_processor.words_list[label_index]
                  prob = probs[i][label_index]
                  print(f"[After Step {training_step}] File: {filename} - True label: {label_name}, Prob('{label_name}')={prob:.4f}")
                val_acc_metric.update_state(val_labels, val_logits)
            val_acc = val_acc_metric.result().numpy()
            val_acc_metric.reset_state()
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
            if val_acc >= best_accuracy:
                best_accuracy = val_acc
                os.makedirs(FLAGS.train_dir, exist_ok=True)
                save_path = os.path.join(FLAGS.train_dir, f"best_model_weights_{int(best_accuracy*10000)}_{int(training_step)}.weights.h5")
                qat_model.save_weights(save_path)
                print(f"Saved best model weights to {save_path}")
                save_path = os.path.join(FLAGS.train_dir, f"best_model_{int(best_accuracy*10000)}_{int(training_step)}.h5")
                qat_model.save(save_path)
                # from tensorflow_model_optimization.quantization.keras import quantize as tfmot_quantize
                # stripped = strip_quantization(qat_model)
                # stripped.save(os.path.join(FLAGS.train_dir, "qat_stripped.h5"))

                # --- Convert to INT8 TFLite using QAT ranges ---
                """converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

                tflite_model = converter.convert()
                open(os.path.join(FLAGS.train_dir, "qat_model_int8.tflite"), "wb").write(tflite_model)"""
                # 1. Final Parameter Extraction
                print("\n\n--- Final Parameter Extraction for NumPy Inference ---")
                NPZ_OUTPUT_PATH = os.path.join(FLAGS.train_dir, 'extracted_params.npz')
                # Assuming 'qat_model' holds the final converged state after the loop.
                # Note: If you only want to extract from the *best* model (highest validation accuracy),
                # you would first need to load the best saved weights back into 'qat_model' here.
                extract_all_quantized_params(qat_model, NPZ_OUTPUT_PATH)
                # Run Conversion
                """output_act_params = extract_output_quantizer_params(qat_model)
 
                for layer_name, info in output_act_params.items():
                    details = info["output_activation"]
                    print(f"\nLayer: {layer_name}")
                    print(f"  Output activation quantizer:")
                    print(f"    Min/Max: {details['min']} / {details['max']}")
                    print(f"    Scale: {details['scale']}, Zero-point: {details['zero_point']}")
                inspect_fake_quant_ranges(qat_model)
                min_val = -7.0
                max_val = 7.0
                print("Weights:")
                weights_int8, weights_dequant, scale, zp = quantize_weights(qat_model, min_val, max_val)
                act_scale, out_zero_point, R_min, R_max = quantize_activations(qat_model)
                convert_int4_model_to_tflite(
                      qat_model,
                      tflite_path=os.path.join(FLAGS.train_dir, "qat_new.tflite")
                )"""

    # ---------------- Testing ---------------- #
    test_size = audio_processor.set_size('testing')
    test_accuracy_total = 0.0
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for i in range(0, test_size, FLAGS.batch_size):
        test_fingerprints, test_labels = audio_processor.get_data(
            FLAGS.batch_size, i, 0,
            background_frequency=0.0,
            background_volume_range=0.0,
            time_shift=0,
            mode='testing'
        )
        test_fingerprints = tf.reshape(test_fingerprints, [-1, 92, 64, 1])
        test_logits = qat_model(test_fingerprints, training=False)
        test_acc_metric.update_state(test_labels, test_logits)
        test_accuracy_total += test_acc_metric.result().numpy() * min(FLAGS.batch_size, test_size - i)

    test_accuracy_total /= test_size
    test_accuracy = test_acc_metric.result().numpy()
    print(f"Final Test Accuracy: {test_accuracy_total*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str,
                        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
    parser.add_argument('--data_dir', type=str, default='/tmp/speech_dataset/')
    parser.add_argument('--background_volume', type=float, default=0.1)
    parser.add_argument('--background_frequency', type=float, default=0.8)
    parser.add_argument('--silence_percentage', type=float, default=25.0)
    parser.add_argument('--unknown_percentage', type=float, default=400.0)
    parser.add_argument('--partial_percentage', type=float, default=10.0)
    parser.add_argument('--time_shift_ms', type=float, default=100.0)
    parser.add_argument('--testing_percentage', type=int, default=10)
    parser.add_argument('--validation_percentage', type=int, default=1)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--clip_duration_ms', type=int, default=1000)
    parser.add_argument('--window_size_ms', type=float, default=30.0)
    parser.add_argument('--window_stride_ms', type=float, default=10.0)
    parser.add_argument('--dct_coefficient_count', type=int, default=40)
    parser.add_argument('--how_many_training_steps', type=str, default='15000,3000')
    parser.add_argument('--eval_step_interval', type=int, default=1)
    parser.add_argument('--learning_rate', type=str, default='0.001,0.0001')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--summaries_dir', type=str, default='/tmp/retrain_logs')
    parser.add_argument('--wanted_words', type=str,
                        default='yes,no,up,down,left,right,on,off,stop,go')
    parser.add_argument('--train_dir', type=str, default='/tmp/speech_commands_train')
    parser.add_argument('--model_architecture', type=str, default='dnn')
    parser.add_argument('--model_size_info', type=int, nargs="+", default=[128,128,128])
    parser.add_argument('--check_nans', type=bool, default=False)
    parser.add_argument('--fft_length', type=int, default=1024,
                    help='FFT length for STFT')
    parser.add_argument('--mel_bin_count', type=int, default=40,
                    help='Number of Mel bins for log-mel features')
    parser.add_argument('--mel_lower_edge_hertz', type=float, default=20.0,
                    help='Lowest frequency edge for Mel filters')
    parser.add_argument('--mel_upper_edge_hertz', type=float, default=4000.0,
                    help='Highest frequency edge for Mel filters')
    parser.add_argument('--pretrained_h5_path', type=str, default="dscnn_model_folded_separate.h5", help='Path to pretrained float .h5 model')
    FLAGS = parser.parse_args()
    main(FLAGS)

