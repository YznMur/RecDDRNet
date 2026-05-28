import os
import cv2
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from glob import glob
from tqdm import tqdm

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
ENGINE_PATH = "./ddrnet23_slim_cityscapes_fp16.engine"   # Path to your engine
IMAGES_DIR = "data/cityscapes/leftImg8bit/test/"       # Folder with test images
OUTPUT_DIR = "./outputs_trt/"                            # Output folder
IMG_SIZE = (2048, 1024)                                  # Cityscapes: (W,H)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD ENGINE
# -------------------------------------------------
logger = trt.Logger(trt.Logger.INFO)
trt_runtime = trt.Runtime(logger)

print(f"[INFO] Loading TensorRT engine from {ENGINE_PATH}")
with open(ENGINE_PATH, "rb") as f:
    engine = trt_runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Get input/output info
input_binding = engine.get_binding_index(engine[0])
output_binding = engine.get_binding_index(engine[1])

input_shape = engine.get_binding_shape(0)
print(f"[INFO] Engine input shape: {input_shape}")

# -------------------------------------------------
# ALLOCATE BUFFERS
# -------------------------------------------------
def allocate_buffers(engine):
    h_inputs, d_inputs = [], []
    h_outputs, d_outputs = [], []

    for binding in engine:
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = context.get_binding_shape(engine.get_binding_index(binding))
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        if engine.binding_is_input(binding):
            h_inputs.append(host_mem)
            d_inputs.append(device_mem)
        else:
            h_outputs.append(host_mem)
            d_outputs.append(device_mem)
    return h_inputs, d_inputs, h_outputs, d_outputs

h_inputs, d_inputs, h_outputs, d_outputs = allocate_buffers(engine)
stream = cuda.Stream()

# -------------------------------------------------
# INFERENCE FUNCTION
# -------------------------------------------------
def infer_trt(image):
    # Preprocess
    img = cv2.resize(image, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]  # to NCHW

    np.copyto(h_inputs[0], img.ravel())

    # Transfer to GPU
    cuda.memcpy_htod_async(d_inputs[0], h_inputs[0], stream)

    # Execute
    context.execute_async_v2(bindings=[int(d_inputs[0]), int(d_outputs[0])], stream_handle=stream.handle)

    # Transfer back
    cuda.memcpy_dtoh_async(h_outputs[0], d_outputs[0], stream)
    stream.synchronize()

    output = h_outputs[0].reshape(context.get_binding_shape(1))
    return output

# -------------------------------------------------
# RUN ON ALL IMAGES
# -------------------------------------------------
image_paths = glob(os.path.join(IMAGES_DIR, "*.png"))
print(f"[INFO] Found {len(image_paths)} images")

for path in tqdm(image_paths, desc="Running TensorRT inference"):
    img = cv2.imread(path)
    t0 = time.time()
    out = infer_trt(img)
    fps = 1.0 / (time.time() - t0)

    pred = np.argmax(out.squeeze(), axis=0).astype(np.uint8)
    color_map = cv2.applyColorMap((pred * 25).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(cv2.resize(img, IMG_SIZE), 0.5, color_map, 0.5, 0)

    name = os.path.basename(path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), blend)
    print(f"[INFO] Saved: {name}  FPS={fps:.2f}")

print("[✅] Done. All results saved to:", OUTPUT_DIR)
