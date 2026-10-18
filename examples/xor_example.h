#pragma once

#include "Network.h"
#include <array>
#include <cstdlib>

namespace xor_example {
// Each row is {left input, right input, target}.
constexpr std::array<std::array<double, 3>, 4> cases{{
    {{0.0, 0.0, 0.0}}, {{0.0, 1.0, 1.0}},
    {{1.0, 0.0, 1.0}}, {{1.0, 1.0, 0.0}}
}};
constexpr unsigned trainingSteps = 4000;

inline Network train(unsigned seed)
{
    std::srand(seed);
    Network network({2, 4, 1});
    for (unsigned step = 0; step < trainingSteps; ++step) {
        const auto& row = cases[step % cases.size()];
        network.feedForward({row[0], row[1]});
        network.backPropagation({row[2]});
    }
    return network;
}
}
