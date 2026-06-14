#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <stdint.h>
#include <unistd.h>

extern unsigned char model_full_tflite[];
extern unsigned int model_full_tflite_len;

extern "C" int putchar_(int c) {
    char ch = static_cast<char>(c);
    return write(1, &ch, 1);
}

#include "external_weights.h"

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

    // Model needs exactly 7 distinct builtins (verified from flatbuffer):
    // QUANTIZE, CONV_2D, AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX, DEQUANTIZE.
    // Each Add* must be called AT MOST ONCE, BEFORE constructing the interpreter.
    tflite::MicroMutableOpResolver<7> resolver;
    if (resolver.AddQuantize()       != kTfLiteOk ||
        resolver.AddConv2D()         != kTfLiteOk ||
        resolver.AddAveragePool2D()  != kTfLiteOk ||
        resolver.AddReshape()        != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax()        != kTfLiteOk ||
        resolver.AddDequantize()     != kTfLiteOk) {
        MicroPrintf("FATAL: failed to register a required op.");
        return 1;
    }

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    MicroPrintf("Checkpoint 2: Resolver and Interpreter created.");

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("Error: AllocateTensors() failed!");
        return 1;
    }

    TfLiteTensor* input_tensor = interpreter.input(0);
    if (input_tensor == nullptr) {
        MicroPrintf("CRITICAL: interpreter.input(0) is NULL!");
        return 1;
    }
    MicroPrintf("Input: type=%d bytes=%d dims=%d",
                input_tensor->type, (int)input_tensor->bytes,
                input_tensor->dims->size);   // expect type=0 (FLOAT32), bytes=4096, dims=4

    MicroPrintf("Checkpoint 3: Tensors allocated safely.");

    // Input is FLOAT32 [1,32,32,1]. Quantization is internal (first op is QUANTIZE).
    float* input = interpreter.typed_input_tensor<float>(0);
    if (input == nullptr) {
        MicroPrintf("CRITICAL: typed_input_tensor<float>(0) is NULL!");
        return 1;
    }
    const int input_elements = (int)(input_tensor->bytes / sizeof(float));  // 1024
    for (int i = 0; i < input_elements; i++) {
        input[i] = 0.0f;   // replace with real image data
    }

    MicroPrintf("Checkpoint 4: Starting RISC-V Baseline Inference...");
    if (interpreter.Invoke() != kTfLiteOk) {
        MicroPrintf("Error: Invoke failed!");
        return 1;
    }

    // Output is FLOAT32 [1,10] (post-DEQUANTIZE).
    float* output = interpreter.typed_output_tensor<float>(0);
    if (output == nullptr) {
        MicroPrintf("CRITICAL: typed_output_tensor<float>(0) is NULL!");
        return 1;
    }
    for (int i = 0; i < 10; i++) {
        MicroPrintf("Class %d score(x1000)=%d", i, (int)(output[i] * 1000.0f));
    }

    MicroPrintf("Checkpoint 5: Inference Complete!");
    return 0;
}
