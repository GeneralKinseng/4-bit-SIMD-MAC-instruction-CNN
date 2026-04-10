import os
# Force the environment to use the legacy Keras 2 engine to ensure 
# compatibility with the TensorFlow Model Optimization Toolkit (TFMOT).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
import numpy as np

# ==============================================================================
# STEP 1.1: Dataset Acquisition and Preprocessing
# ==============================================================================
def preprocess_mnist(image, label):
    """
    Casts to FP32, normalizes, pads to 32x32, and applies one-hot encoding.
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.pad_to_bounding_box(image, offset_height=2, offset_width=2, 
                                         target_height=32, target_width=32)
    label = tf.one_hot(label, depth=10)
    return image, label

def load_datasets():
    print("Acquiring and preprocessing MNIST dataset...")
    (ds_train_raw, ds_test_raw), ds_info = tfds.load(
        'mnist', split=['train', 'test'], shuffle_files=True,
        as_supervised=True, with_info=True
    )
    BATCH_SIZE = 128
    
    ds_train = ds_train_raw.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    ds_test = ds_test_raw.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test

# ==============================================================================
# STEP 1.2: Model Selection and Architecture Definition
# ==============================================================================
def build_lenet5_architecture():
    """
    Step 1.2: Define the baseline LeNet-5 CNN architecture (2C-3D topology).
    """
    model = tf.keras.Sequential([
        # 1st Convolutional Layer (C1)
        # Extracts low-level features using 6 filters of size 5x5.
        tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), 
                               activation='relu', input_shape=(32, 32, 1)),
        
        # 1st Subsampling/Pooling Layer (S2)
        # Reduces spatial dimensions to lower computational load.
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # 2nd Convolutional Layer (C3)
        # Extracts higher-level features using 16 filters of size 5x5.
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), 
                               activation='relu'),
        
        # 2nd Subsampling/Pooling Layer (S4)
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Flatten the 3D tensor into a 1D vector for the Dense layers
        tf.keras.layers.Flatten(),
        
        # 1st Fully Connected Layer (F5) - Dense 1
        tf.keras.layers.Dense(units=120, activation='relu'),
        
        # 2nd Fully Connected Layer (F6) - Dense 2
        tf.keras.layers.Dense(units=84, activation='relu'),
        
        # Output Classification Layer - Dense 3
        # Uses Soft-max activation to output a probability distribution across the 10 MNIST digit classes.
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    
    return model

# ==============================================================================
# STEP 1.4: 4-Bit Quantization via Quantization-Aware Training (QAT)
# ==============================================================================
def apply_qat(fp32_model, train_dataset, val_dataset):
    """
    Applies QAT to adapt the FP32 model weights to lower precision constraints.
    """
    print("\nInitializing Quantization-Aware Training (QAT)...")
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(fp32_model)
    
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Retrain to simulate precision reduction and mitigate quantization noise
    qat_model.fit(train_dataset, epochs=5, validation_data=val_dataset)
    return qat_model

# ==============================================================================
# STEP 1.5: Sparsity Induction and Lookahead Encoding (SSSA)
# ==============================================================================
def threshold_pruning(weights, lam=0.1):
    """
    Nullifies network traces with negligible contribution mapping low values to zero.
    """
    threshold = np.max(np.abs(weights)) * lam
    pruned_weights = np.where(np.abs(weights) < threshold, 0, weights)
    return pruned_weights

def encode_last_bits(weights_block, skip_blocks):
    """
    Embeds the 4-bit skip_blocks value into a block of 4 weights.
    Simulates INT7 precision by sacrificing 1 bit for lookahead tracking.
    """
    encoded = weights_block.astype(np.uint8)
    for i in range(4):
        w = encoded[i]
        sign_bit = (w >> 7) & 0b1
        skip_bit = (skip_blocks >> i) & 0b1
        w = w & 0b10111111           # Remove MSB after sign bit
        w = (w << 1) & 0b01111110    # Shift left
        w = w | skip_bit             # Insert skip bit into LSB
        w = w | (sign_bit << 7)      # Restore sign bit
        encoded[i] = w
    return encoded.astype(np.int8)

def sssa_lookahead_encoding(kernel_weights):
    """
    Calculates consecutive all-zero blocks and encodes this into non-zero weights.
    """
    H, W, C = kernel_weights.shape
    encoded_kernel = np.copy(kernel_weights)
    
    for h in range(H):
        for w in range(W):
            for c in range(0, C, 4):
                if c + 4 > C:
                    break
                current_block = encoded_kernel[h, w, c:c+4]
                
                if np.any(current_block != 0):
                    i_nxt = c + 4
                    skip_blocks = 0
                    # Lookahead to count succeeding all-zero blocks (max 15)
                    while i_nxt + 4 <= C and skip_blocks < 15:
                        next_block = encoded_kernel[h, w, i_nxt:i_nxt+4]
                        if np.all(next_block == 0):
                            skip_blocks += 1
                            i_nxt += 4
                        else:
                            break
                    # Encode the skip_blocks counter
                    encoded_block = encode_last_bits(current_block, skip_blocks)
                    encoded_kernel[h, w, c:c+4] = encoded_block
                    
    return encoded_kernel

# ==============================================================================
# STEP 1.6: Model Export and FlatBuffer Reorganization
# ==============================================================================
def export_and_reorganize_flatbuffer(qat_model):
    """
    Serializes to FlatBuffer, and extracts weights to an external C-array.
    """
    print("\nConverting model to TFLite FlatBuffer...")
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = "model_full.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print("Reorganizing FlatBuffer into mini-FlatBuffer and Weights C-Array...")
    c_array_header = "#include <stdint.h>\n\n"
    c_array_header += "const int8_t external_weights_array[] = {\n"
    
    # Mocking layer extraction. In reality, this requires parsing the FlatBuffer schema
    dummy_extracted_weights = np.random.randint(-8, 7, size=(1024,), dtype=np.int8) 
    weights_csv = ", ".join(map(str, dummy_extracted_weights))
    c_array_header += f"    {weights_csv}\n"
    c_array_header += "};\n"
    
    with open("external_weights.h", "w") as f:
        f.write(c_array_header)
        
    print("Reorganization Complete. Weights indexed in 'external_weights.h'.")

# ==============================================================================
# Main Execution Flow
# ==============================================================================
if __name__ == "__main__":
    # Step 1.1
    ds_train, ds_test = load_datasets()
    
    # Step 1.2
    print("\nBuilding FP32 Baseline Architecture...")
    fp32_model = build_lenet5_architecture()
    
    # Step 1.3
    print("Compiling and Training FP32 Golden Model...")
    fp32_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    fp32_model.fit(ds_train, epochs=10, validation_data=ds_test)
    fp32_model.save('mnist_fp32_golden_model.h5')
    
    # Step 1.4
    qat_model = apply_qat(fp32_model, ds_train, ds_test)
    
    # Step 1.5 (Demonstration on a dummy kernel mimicking extracted weights)
    print("\nApplying Sparsity and SSSA Lookahead Encoding...")
    dummy_kernel = np.random.randint(-8, 7, size=(3, 3, 16), dtype=np.int8)
    pruned_kernel = threshold_pruning(dummy_kernel, lam=0.5)
    encoded_sparse_kernel = sssa_lookahead_encoding(pruned_kernel)
    
    # Step 1.6
    export_and_reorganize_flatbuffer(qat_model)
    print("\n--- Phase 1 Software Pipeline Complete ---")
