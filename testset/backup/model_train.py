import os
# Force the training script to use the Keras 2 engine to ensure 
# serialization compatibility with the TFMOT quantization pipeline.
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_mnist(image, label):
    """
    Step 1.1: Preprocessing mapping function for the TFDS pipeline.
    """
    # Cast raw integer pixel values to 32-bit floating point (FP32)
    # and normalize to the range [0.0, 1.0] for gradient stability.
    image = tf.cast(image, tf.float32) / 255.0
    
    # Pad native 28x28 MNIST images to 32x32 pixels to match the input 
    # requirements of standard baseline CNN hardware architectures.
    image = tf.image.pad_to_bounding_box(image, offset_height=2, offset_width=2, 
                                         target_height=32, target_width=32)
    
    # Apply One-Hot Encoding to the classification labels (10 classes for MNIST)
    label = tf.one_hot(label, depth=10)
    
    return image, label

def build_baseline_cnn():
    """
    Step 1.2: Define the baseline CNN architecture (LeNet-5 equivalent).
    """
    model = tf.keras.Sequential([
        # First Convolutional Stage
        tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), 
                               activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Second Convolutional Stage
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), 
                               activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Flatten and Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=120, activation='relu'),
        tf.keras.layers.Dense(units=84, activation='relu'),
        
        # Output Soft-max Layer for Probability Classification
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    return model

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

def main():
    # ---------------------------------------------------------
    # Step 1.1: Dataset Acquisition and Preprocessing Pipeline
    # ---------------------------------------------------------
    print("Acquiring MNIST dataset via tensorflow_datasets...")
    (ds_train_raw, ds_test_raw), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    # Construct the data pipelines for highly efficient I/O streaming
    BATCH_SIZE = 128
    
    print("Applying 32x32 padding, FP32 normalization, and one-hot encoding...")
    ds_train = ds_train_raw.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    ds_test = ds_test_raw.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    # ---------------------------------------------------------
    # Step 1.2: Model Selection and Architecture Definition
    # ---------------------------------------------------------
    print("Building the FP32 baseline LeNet-5 CNN architecture...")
    #model = build_baseline_cnn()
    #model.summary()
    model = build_lenet5_architecture()
    model.summary()
    
    # ---------------------------------------------------------
    # Step 1.3: Floating-Point (FP32) Baseline Training
    # ---------------------------------------------------------
    print("Compiling model with Cross-Entropy Loss...")
    # The cross entropy function gives a measure of error between the predicted 
    # output and the true label, used to compute weights during training.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Initiating FP32 Baseline Training...")
    EPOCHS = 10
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test
    )
    
    # Save the FP32 Golden Model in the native Keras format to resolve warnings
    model_path = 'mnist_fp32_golden_model.h5'
    model.save(model_path)
    print(f"\nTraining complete. FP32 Golden Model saved as {model_path}.")

    # ---------------------------------------------------------
    # Prologue to FlatBuffer Conversion (Prep for Step 1.6)
    # ---------------------------------------------------------
    print("Converting the native Keras model to TFLite FlatBuffer format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the initial unquantized TFLite model
    tflite_path = 'mnist_fp32_golden_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"FlatBuffer conversion complete. Saved as {tflite_path}.")

if __name__ == "__main__":
    main()
