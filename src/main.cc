#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <stdint.h>

// Include the mini-FlatBuffer topology generated via xxd
extern unsigned char model_full_tflite[];
extern unsigned int model_full_tflite_len;

extern "C" void putchar_(char c) {
    // Empty for now - TFLM will use its internal mechanisms for MicroPrintf
}

// Include the extracted 4-bit quantized weights mapped to external memory
#include "external_weights.h"

// Define the tensor arena size (2MB safely in global memory)
const int kTensorArenaSize = 2 * 1024 * 1024;   
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int main(int argc, char* argv[]) {
    MicroPrintf("--- System Booting ---");

    const tflite::Model* model = tflite::GetModel(model_full_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Error: Model schema version mismatch.");
        return 1;
    }
    MicroPrintf("Checkpoint 1: Model loaded successfully.");

    tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddAveragePool2D(); 
    resolver.AddMaxPool2D();     
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddTanh();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    MicroPrintf("Checkpoint 2: Resolver and Interpreter created.");

    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("Error: AllocateTensors() failed! Check for missing ops.");
        return -1;
    }
    MicroPrintf("Checkpoint 3: Tensors allocated safely.");

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    if (input == nullptr || output == nullptr) {
        MicroPrintf("Error: Failed to get input/output tensor pointers.");
        return -1;
    }

    MicroPrintf("Checkpoint 3.5: Checking Input Tensor Type...");

    // CRUCIAL CHECK: Did the arena actually allocate physical RAM for this input?
    if (input->data.data == nullptr) {
        MicroPrintf("CRITICAL ERROR: The input data pointer is NULL!");
        MicroPrintf("AllocateTensors() lied to us. The arena might still be unaligned.");
        return -1;
    }

    // Universal Memory Wipe: Fill the buffer with 0s treating it as raw bytes.
    // This safely simulates an empty image whether the model expects Float32, Int8, or Int16.
    uint8_t* raw_input_buffer = static_cast<uint8_t*>(input->data.data);
    for (size_t i = 0; i < input->bytes; ++i) {
        raw_input_buffer[i] = 0;
    }

    MicroPrintf("Checkpoint 4: Starting RISC-V Baseline Inference...");
    
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Error: Invoke failed!");
        return 1;
    }
    
    MicroPrintf("Checkpoint 5: Inference Complete!");

    // Output the results
    for (int i = 0; i < 10; ++i) {
        MicroPrintf("Class %d: %d", i, output->data.int8[i]);
    }

    return 0;
}
