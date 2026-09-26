# Architecture design & parameter calculations

Input: 64 × 64 × 3 (EuroSAT RGB). Output: 10 classes.
All 3×3 convolutions use stride 1 and padding 1, so only the 2×2 max-pools change
the spatial size: 64 → 32 → 16 → 8 → 4.
The conv layers have no bias because each one is followed by BatchNorm, which adds
its own shift (β). BatchNorm has 2 trainable parameters per channel (γ, β).

Notation: `k` = kernel size, `Cin` / `Cout` = channels, `H×W` = output size of the conv.

* Standard conv: params = k²·Cin·Cout, MACs = H·W·k²·Cin·Cout
* Depthwise conv: params = k²·Cin, MACs = H·W·k²·Cin
* Pointwise conv: params = Cin·Cout, MACs = H·W·Cin·Cout
* FC: params = in·out + out, MACs = in·out

## Model A – Standard CNN

| # | Layer | Kernel / units | Output | Weights | BN | Params | MACs |
|---|---|---|---|---:|---:|---:|---:|
| 1 | Conv + BN + ReLU | 3×3, 3→32 | 64×64×32 | 864 | 64 | 928 | 3,538,944 |
|   | MaxPool | 2×2 | 32×32×32 | – | – | 0 | – |
| 2 | Conv + BN + ReLU | 3×3, 32→64 | 32×32×64 | 18,432 | 128 | 18,560 | 18,874,368 |
|   | MaxPool | 2×2 | 16×16×64 | – | – | 0 | – |
| 3 | Conv + BN + ReLU | 3×3, 64→128 | 16×16×128 | 73,728 | 256 | 73,984 | 18,874,368 |
|   | MaxPool | 2×2 | 8×8×128 | – | – | 0 | – |
| 4 | Conv + BN + ReLU | 3×3, 128→256 | 8×8×256 | 294,912 | 512 | 295,424 | 18,874,368 |
|   | MaxPool | 2×2 | 4×4×256 | – | – | 0 | – |
| 5 | Global avg pool | – | 256 | – | – | 0 | – |
| 6 | FC + ReLU (+Dropout 0.3) | 64 units | 64 | 16,384 + 64 bias | – | 16,448 | 16,384 |
| 7 | FC (logits) | 10 units | 10 | 640 + 10 bias | – | 650 | 640 |
| | **Total** | | | | | **405,994** | **≈ 60.18 M** |

## Model B – Lightweight CNN (depthwise separable)

Each block: DWConv 3×3 → BN → ReLU → PWConv 1×1 → BN → ReLU → MaxPool 2×2.

| # | Block | DW weights | DW BN | PW weights | PW BN | Params | MACs (DW + PW) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 3→32 @ 64×64 | 27 | 6 | 96 | 64 | 193 | 110,592 + 393,216 = 503,808 |
| 2 | 32→64 @ 32×32 | 288 | 64 | 2,048 | 128 | 2,528 | 294,912 + 2,097,152 = 2,392,064 |
| 3 | 64→128 @ 16×16 | 576 | 128 | 8,192 | 256 | 9,152 | 147,456 + 2,097,152 = 2,244,608 |
| 4 | 128→256 @ 8×8 | 1,152 | 256 | 32,768 | 512 | 34,688 | 73,728 + 2,097,152 = 2,170,880 |
| 5 | GAP → FC 64 + ReLU + Dropout | | | | | 16,448 | 16,384 |
| 6 | FC 10 (logits) | | | | | 650 | 640 |
| | **Total** | | | | | **63,659** (< 100,000 ✔) | **≈ 7.33 M** |

**Reduction from A to B:** 6.4× fewer parameters, 8.2× fewer MACs.
For one layer, the depthwise separable conv costs 1/Cout + 1/k² of a standard conv.
With k = 3 and Cout = 256 that is 1/256 + 1/9 ≈ 0.115, i.e. about 8.7× cheaper.
The first block uses depthwise separable convolution too, as the assignment asks.
With only 3 input channels it saves little: 193 parameters instead of 928. That is
why MobileNet keeps a standard first convolution.

Estimated weight storage: Model A ≈ 1.55 MB in float32 (≈ 0.39 MB in int8).
Model B ≈ 249 KB in float32 (≈ 62 KB in int8), small enough for a microcontroller
with 256 KB of flash once quantised.

## Design choices (hardware-aware)

* **ReLU everywhere.** It is just `max(0, x)`: one comparison, with no exponential,
  division or lookup table. Sigmoid, tanh, Swish and GELU all need `exp()`, which is
  slow or emulated in software on microcontrollers and DSPs that have no FPU. ReLU
  maps exactly onto int8 arithmetic: zero stays zero and nothing needs re-scaling.
  It can also be fused into the conv/BN kernel. Its gradient does not saturate, so
  it also trains faster. ReLU6 would be a good alternative because its bounded range
  makes int8 quantisation easier.
* **No softmax at inference.** Softmax (which needs exponentials) is used only
  inside the cross-entropy loss during training. On the device the predicted class
  is the argmax of the logits, which is the same answer at no extra cost.
* **BatchNorm** has no inference cost: after training it is folded into the
  preceding convolution's weights and bias.
* **Global average pooling instead of Flatten.** A Flatten layer would give a
  4×4×256 = 4,096-wide vector, and a 64-unit FC layer on that would need 262k
  weights, so Model B would break its budget from the classifier alone. GAP needs
  only 256 inputs.
* **Max-pooling** is a comparison only, with no multiplications. It also reduces the
  activation memory (RAM) that the next layer needs.
* **Fixed channel widths 32-64-128-256** (multiples of 8/16) map well onto SIMD
  lanes and NPU tile sizes.
