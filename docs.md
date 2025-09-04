Perfect — let’s build Option B: LiDAR-first + AI depth completion → 2D obstacle mask → LBM. I’ll give you everything you need to implement this pipeline on-device:

a mobile-friendly depth-completion model design + training recipe (TFLite-ready),

a C++/NDK-ready mapping module that turns device LiDAR point clouds into a dense depth map and then into a 2D obstacle mask for your LBM grid, and

the runtime inference & threading plan (call order, batching, confidence handling, perf tips).

I’ll keep code practical and ready to drop into your Android/iOS app. Let’s go.

1) Depth-completion model (mobile, TFLite-friendly)

Goal: RGB + sparse LiDAR → dense depth + per-pixel confidence. Lightweight, <2–5M params, quantizable to INT8.

Architecture (tiny UNet + sparse-aware encoder)

Input: RGB image (H×W×3) + sparse-depth channel (H×W×1) + binary validity mask (H×W×1) → concatenated to 5 channels.

Encoder: MobileNetV2-ish backbone (inverted residuals) but reduced width (0.35×) or use a tiny custom conv stack:

Conv3x3 stride 2 → BN → ReLU6 (out channels 16)

2× InvertedResidual blocks (t=6) (16→24)

2× InvertedResidual blocks (24→32) (stride 2)

Optional one more downsample to 64 feature maps

Decoder: Light UNet upsampling with skip connections:

UpConv (bilinear + 1x1 conv) → concat skip → Conv 3x3 → ReLU

Repeat until original size.

Heads:

Depth head: 1×1 conv → ReLU → final 1×1 conv output (linear) for log-depth (predict log(d + eps))

Confidence head: 1×1 conv → sigmoid (0..1)

Loss:

L1 on depth in pixel units (or log-L1).

Weighted data term: only supervise on pixels where ground-truth depth exists (or synthesized).

Photometric consistency loss (when multi-frame available): warp RGB using predicted depth + pose to adjacent frame and apply L1 or SSIM loss (optional).

Confidence-aware loss: use predicted confidence c: minimize c * |d_pred - d_gt| + λ * (1 - c) (learns to downweight uncertain predictions). Equivalent to NLL of Laplacian with heteroscedastic uncertainty.

Smoothness regularizer: edge-aware smoothness (gradients weighted by image edges).

Training targets:

Dense ground-truth depth (from high-res LiDAR + fused scans, or synthetic scenes).

Randomly sparsify depth input during training to mimic phone LiDAR patterns.

Data augmentation:

Photometric (brightness/hue), geometric (scaling, small rotations), sensor noise, point-dropout, reflective surfaces simulation.

Params:

~2–5M parameters target.

Use depth normalization: predict log-depth or normalized depth to [0,1].

Quantization:

Use post-training quantization to INT8 or quant-aware training (QAT) for best NNAPI/TFLite performance.

Implementation tips

Use separable convs where possible.

Build model in TensorFlow/Keras for straightforward TFLite export, or PyTorch + ONNX → TFLite.

Export two outputs: dense_depth (H×W), confidence_map (H×W).

If running on iOS, convert to CoreML via coremltools or keep TFLite + TensorFlow Lite Metal delegate.

Training recipe (short)

Dataset: Mix real LiDAR scans (sparse) fused to dense depth and synthetic datasets (renderers with known geometry). Domain randomize.

Batch size: as large as GPU allows (e.g., 16 for 256×256).

Optimizer: AdamW, lr=1e-3 warmup → cosine decay to 1e-5.

Loss weights: L1_depth = 1.0, photometric = 0.5 (if used), smooth = 0.1, confidence weight λ = 1e-3 (tune).

Train 50–200k iterations, validate on held-out scenes.

2) C++ mapping module: LiDAR → dense depth → 2D obstacle mask for LBM

This contains:

ingesting device point cloud (sparse points in camera frame),

rasterizing to a depth image and computing a sparse depth channel and validity mask,

calling the TFLite depth-completion model (glue code not shown; assume you can call a TFLite interpreter),

postprocessing depth + confidence → 2D obstacle mask (grid aligned to your LBM domain),

providing a confidence map per LBM cell,

optional voxelization/TSDF alternative if you prefer small volumetric representation.

Below is self-contained C++ (portable) code for point cloud rasterization → grid mask. It assumes you already received:

camera intrinsics (fx, fy, cx, cy),

camera pose (extrinsics) aligning point cloud to RGB frame (or points in camera coordinates),

and a TFLite model to run on the input tensors.

I keep the code straightforward and NDK-friendly (no heavy deps). You can optimize with Eigen/NEON later.

// lidar_to_mask.hpp
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>

// Simple utility: rasterize sparse point cloud (camera coordinates) to depth image & validity mask.
// Then resample/depth-complete via your TFLite model (not included), then convert depth->2D obstacle mask.

struct CameraIntrinsics {
    float fx, fy, cx, cy;
    int width, height; // image resolution
};

// Point in camera coords (meters)
struct Point3 {
    float x, y, z;
};

// Rasterize points to a depth buffer (z in meters). Points with z <= 0 are ignored.
inline void rasterize_points_to_depth(const std::vector<Point3>& points,
                                      const CameraIntrinsics& K,
                                      std::vector<float>& depth_out,
                                      std::vector<uint8_t>& valid_out) {
    const int W = K.width, H = K.height;
    depth_out.assign((size_t)W * H, 0.f);
    valid_out.assign((size_t)W * H, 0);

    for (const auto &p : points) {
        if (p.z <= 0.f) continue;
        float u = (p.x * K.fx) / p.z + K.cx;
        float v = (p.y * K.fy) / p.z + K.cy;
        int ui = (int)std::round(u), vi = (int)std::round(v);
        if (ui < 0 || ui >= W || vi < 0 || vi >= H) continue;
        size_t idx = (size_t)vi * W + ui;
        // keep nearest (min z)
        if (!valid_out[idx] || p.z < depth_out[idx]) {
            depth_out[idx] = p.z;
            valid_out[idx] = 1;
        }
    }
}

// Bilateral-ish / simple hole filling: fill nearest valid depth in small radius (fast).
inline void fill_depth_nearest(std::vector<float>& depth, std::vector<uint8_t>& valid,
                               int W, int H, int radius = 2) {
    std::vector<float> depth_copy = depth;
    std::vector<uint8_t> valid_copy = valid;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t i = (size_t)y*W + x;
            if (valid_copy[i]) continue;
            float best = 0.f; bool found=false;
            for (int dy = -radius; dy <= radius; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= W) continue;
                    size_t j = (size_t)yy*W + xx;
                    if (valid_copy[j]) {
                        if (!found || depth_copy[j] < best) {
                            best = depth_copy[j];
                            found = true;
                        }
                    }
                }
            }
            if (found) { depth[i] = best; valid[i] = 1; }
        }
    }
}

// Convert a dense depth map + confidence into an obstacle mask on an LBM grid.
// - project depth into world (or use camera-aligned slice), then threshold to define obstacle occupancy.
// - simplest: assume you want a "floor-parallel top-down" slice at height h_slice
//
// Parameters:
//  depth: H*W depth in meters (camera z)
//  valid: H*W validity
//  K: intrinsics
//  cam_pose: not included — assume points are already in camera coords and you want camera-centric slice
//
// This function projects depth points to x-y plane (camera x-right, y-down, z-forward).
// Then it rasterizes into an LBM grid (nx x ny) covering x∈[xmin,xmax], y∈[ymin,ymax].
// Cells with any point below z_thresh are marked obstacle (e.g., surfaces closer than z_thresh).
inline void depth_to_lbm_mask(const std::vector<float>& depth,
                              const std::vector<uint8_t>& valid,
                              const CameraIntrinsics& K,
                              int nx, int ny,
                              float xmin, float xmax,
                              float ymin, float ymax,
                              float z_thresh, // anything closer than this is obstacle
                              std::vector<uint8_t>& lbm_mask,
                              std::vector<float>& lbm_conf) {
    // Clear outputs
    lbm_mask.assign((size_t)nx*ny, 0);
    lbm_conf.assign((size_t)nx*ny, 0.f);

    const int W = K.width, H = K.height;
    const float dx = (xmax - xmin) / nx;
    const float dy = (ymax - ymin) / ny;

    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            size_t idx = (size_t)v*W + u;
            if (!valid[idx]) continue;
            float z = depth[idx];
            // compute camera-space x,y
            float x_cam = (u - K.cx) * z / K.fx;
            float y_cam = (v - K.cy) * z / K.fy;
            // Now map x_cam, y_cam into lbm grid coordinates (assuming camera-centered)
            // Depending on your desired slice: here we assume x->x, y->y in some scale
            if (x_cam < xmin || x_cam >= xmax || y_cam < ymin || y_cam >= ymax) continue;
            int ix = (int)((x_cam - xmin) / dx);
            int iy = (int)((y_cam - ymin) / dy);
            if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;
            size_t lidx = (size_t)iy*nx + ix;
            // if the surface is closer than threshold (e.g., z small) mark as obstacle
            if (z <= z_thresh) {
                lbm_mask[lidx] = 1;
            }
            // increment confidence (here we just mark presence — you can use more advanced scoring)
            lbm_conf[lidx] += 1.0f;
        }
    }

    // Normalize confidence by max counts (simple)
    float maxc = 1e-6f;
    for (float v : lbm_conf) if (v>maxc) maxc = v;
    for (size_t i = 0; i < lbm_conf.size(); ++i) {
        lbm_conf[i] = std::min(1.f, lbm_conf[i]/maxc);
    }
}

Explanation & choices

Rasterize LiDAR points into the RGB frame to create sparse depth + validity mask.

Run the TFLite depth-completion model with inputs: RGB, sparse-depth, validity mask. Model outputs dense depth + confidence.

Postprocess: small hole-filling (cheap) and optionally a bilateral filter; then project predicted depth into the camera's coordinate system (or if your input points already in camera coords, projection done during rasterization).

Convert depth→LBM grid: choose a projection plane (camera-aligned slice or top-down). I gave camera-aligned mapping; adjust to top-down if you prefer (apply camera pose transform).

z_thresh: tune this to mark obstacles. For a top-down slice, you may threshold vertical height or thickness instead.

Better alternatives

Build a small TSDF voxel grid in a tightly bounded volume (e.g., 64×64×16). Integrate LiDAR points into TSDF on-device (fast), run a tiny 3D CNN for completion, then project to 2D slice. TSDF is more robust for overhangs and nontrivial topology but costs more memory.

3) Runtime inference & threading plan (order of operations)

Goal: run this pipeline at interactive rates while balancing latency and accuracy.

Pipeline order (per session)

Initial calibration (once or occasionally)

Acquire LiDAR snapshot + RGB; compute intrinsics/pose.

Build an initial dense depth (call the model), compute full obstacle mask.

Initialize LBM grid and set static obstacles.

Display a "hold still for 1–3s" onboarding while we compute.

Per-frame loop (every camera frame; aim 10–30 FPS)
(A) Acquire RGB frame; optionally get latest LiDAR frame if device provides continuous streaming.
(B) If LiDAR update available (sparser frequency), rasterize it and run depth-completion model — update dense depth and confidence. This may happen at lower frequency than RGB (e.g., LiDAR every 100–200 ms).
(C) Run optical flow + segmentation on RGB (fast small models). Convert segmentation → refine obstacle mask (merge with LiDAR-based mask).
(D) Map updated dense depth → LBM mask and per-cell confidence (fast C++ rasterization). Smooth/dilate mask to remove small holes.
(E) Update LBM obstacle field and confidence-aware force scaling.
(F) Run LBM substeps (1–3 substeps). Use worker thread(s).
(G) Renderer reads latest velocity/density arrays (double-buffer or atomic swap) and draws overlay. Interpolate between sim frames if needed.

Threading model

Main thread (UI): rendering and camera frame handling; reads latest overlay buffers (texture).

Sensor thread: ingest LiDAR point cloud & RGB, produce sparse depth buffer.

NN inference thread(s): run depth-completion model, segmentation, optical flow on a small thread pool / NN delegate (NNAPI / Metal). Depth completion can be lower priority and run less frequently.

Sim thread: runs LBM engine in a tight loop; consumes latest obstacle mask + force field. Use double-buffering for inputs to avoid stalls.

Use a lock-free flag (atomic) or double-buffer swap to pass obstacle masks and force maps to the sim thread.

Latency budgeting (example target)

Camera frame: 30 FPS (33 ms)

LiDAR snapshot: 10–50 ms but not necessary every frame

Depth completion (quantized): 10–30 ms (run every 2–4 frames)

Segmentation: 5–15 ms

Optical flow (tiny): 10–30 ms

LBM 64×64 step: 1–3 ms per substep
Total per-frame with depth-completion every 3 frames: ~20–50 ms → 20–50 FPS equivalent depending on parallelism.

Confidence usage & safety

When cell confidence low, attenuate both obstacle hardness and body forces:

Replace hard obstacle cells with soft obstacle (velocity decays toward zero over a few sim steps),

Reduce applied optical-flow-derived forces by factor = confidence (or confidence^alpha).

Visualize confidence overlay so users see where predictions are less reliable.

Putting it together — short checklist & parameter defaults

Camera intrinsics: use device API to get fx,fy,cx,cy.

Grid extents for LBM (nx,ny): 64×64 or 96×96. Map physical extents ~1–3 meters across depending on scene (choose based on camera FOV and typical distance).

z_thresh: 0.1–0.4 m (tune: what counts as obstacle in camera frame).

Dilation radius for mask: 1–3 cells (avoid tiny gaps).

Confidence smoothing: temporal exponential moving average with α = 0.1–0.3.

Force mapping optical flow → body force: scale k ≈ 1e-3 to 1e-2 (tune).

Depth completion model: quantize to INT8 and use NNAPI/Metal delegate for speed.

Example integration notes (Android NDK)

Use Android Sensor / ARCore APIs to get LiDAR / depth points and camera intrinsics.

Use TensorFlow Lite with the NNAPI delegate (or GPU delegate) for depth completion, segmentation, optical flow.

Build LBM C++ code as a native lib and expose thin JNI wrappers to Java/Kotlin UI.

Use AImageReader / SurfaceTexture for low-latency camera frames.

Use a worker thread + double-buffering for sim <-> render handoff.

Next concrete code I can provide (choice — I won’t ask, I’ll pick one unless you prefer another)

I prepared three heavy-lift deliverables earlier; to be maximally useful now I can provide either:

A ready-to-build NDK sample that glues TFLite depth-completion + C++ LBM + simple renderer (lots of code), or

A compact C++ function that converts a dense depth image into a TSDF voxel grid (64×64×16) which can produce more robust obstacle masks (useful if you want overhang support), or

A TFLite/Keras model script (Python) that defines the tiny depth-completion UNet, training loop, and export to TFLite conversion.

I’ll assume you’ll want the model script next (most teams need a training recipe). If you prefer the TSDF code or the NDK glue, say so — otherwise I’ll produce the TFLite/Keras model script next.