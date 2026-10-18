# C++ Neural Network — XOR from first principles

A completed learning exercise implementing a fully connected network with handwritten forward propagation, backpropagation, and momentum updates. This repository is retained as a reference, with a bounded reproducibility closeout and no active feature roadmap. It uses the C++ standard library, without an ML framework.

The [multi-language parent](https://github.com/lonlydwolf/XOR-First-Principles-Network) links historical revisions of the C++, C#, and Python implementations. For the repaired C++ example described here, use this repository directly. The implementations have different settings; **cross-language numerical parity is not claimed**.

## Build and run

Use a C++17 compiler such as Clang or GCC. From the repository root, these commands build the example and tests without extra packages or CMake:

```sh
mkdir -p build
c++ -std=c++17 -Wall -Wextra -pedantic -I NN NN/Network.cpp NN/Neuron.cpp examples/xor_demo.cpp -o build/xor_demo
c++ -std=c++17 -Wall -Wextra -pedantic -I NN NN/Network.cpp NN/Neuron.cpp tests/network_tests.cpp -o build/network_tests
./build/xor_demo --seed 0
./build/network_tests
```

The commands above use a POSIX shell. The CI configuration checks Linux and macOS builds. The original Visual Studio project remains historical; the new demo and tests are standalone source files and have not been added to that project.

The demo accepts a nonnegative seed (default `0`) and trains the original `2 → 4 → 1` network over 4,000 samples by repeating `00, 01, 10, 11`. It then prints every truth-table prediction and its classification at threshold 0.5. It returns a nonzero status if any classification is incorrect. It does not read or write training files, so the built executable can run from another working directory.

One local Clang/macOS run with seed 0 produced:

| Inputs | Target | Prediction |
| --- | --- | --- |
| 0, 0 | 0 | -0.000441 |
| 0, 1 | 1 | 0.984851 |
| 1, 0 | 1 | 0.984910 |
| 1, 1 | 0 | 0.000349 |

The seed controls the original C `rand` weight initialization. A seed repeats that sequence on the same C runtime; different platforms may use different random sequences and produce different numerical results. These are the complete four XOR cases, not evidence of generalization to another problem or guaranteed convergence for every seed.

## How it works

- [Network](NN/Network.cpp) traverses layers, propagates errors, and exposes predictions.
- [Neuron](NN/Neuron.cpp) computes `tanh(weighted_sum + 1)`, its derivative `1 - output²`, gradients, and weight updates.
- **Bias is fixed at +1; it is not trained.** Initial weights use the original range `[0, 1]`, learning rate is `0.2`, and momentum is `0.5`.
- A connection update is `0.2 * input * gradient + 0.5 * previous_update`. The logged error statistic is an exponentially smoothed root-mean-square error across output neurons; it is separate from the weight update.

The closeout initializes the error state, sums squared errors across all outputs, rejects incompatible dimensions, adds prediction access, and removes verbose debug printing. It preserves the original activation and weight-update equations and parameters.

## Checks and limits

[The tests](tests/network_tests.cpp) verify that error starts at zero, calculate expected two-output RMSE and smoothing over two updates, reject invalid layer/input/target dimensions, and check all XOR outputs for seeds 0–4. Predictions must classify correctly and fall within 0.1 of their targets. The checks use explicit failures even in release builds.

This is a small numerical exercise, not a validated general-purpose network library. There is no persistence, batching, large-data evaluation, or comprehensive gradient check. Call forward propagation before backpropagation; the API does not enforce that sequencing contract.

The original [NN.cpp](NN/NN.cpp) and [Text.txt](NN/Text.txt) remain historical artifacts. That entry point uses time-seeded random inputs and replaces `Text.txt` in its working directory. Prefer the seeded example above for evaluation without changing files.
