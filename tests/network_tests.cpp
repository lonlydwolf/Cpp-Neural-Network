#include "Network.h"
#include "../examples/xor_example.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Action>
void requireInvalidArgument(Action action)
{
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid dimensions were accepted");
}

void checkDimensions()
{
    requireInvalidArgument([] { Network network({}); });
    requireInvalidArgument([] { Network network({2}); });
    requireInvalidArgument([] { Network network({2, 0, 1}); });
    Network network({2, 4, 1});
    require(network.getResults() == std::vector<double>{0.0}, "initial results must be defined");
    requireInvalidArgument([&] { network.feedForward({1.0}); });
    requireInvalidArgument([&] { network.feedForward({1.0, 0.0, 1.0}); });
    network.feedForward({1.0, 0.0});
    requireInvalidArgument([&] { network.backPropagation({}); });
    requireInvalidArgument([&] { network.backPropagation({1.0, 0.0}); });
}

void checkErrorMetric()
{
    std::srand(7);
    Network network({2, 3, 2});
    require(network.getAvarageError() == 0.0, "initial smoothed error must be zero");
    double expectedAverage = 0.0;
    const std::vector<double> targets{0.0, 1.0};
    for (unsigned step = 0; step < 2; ++step) {
        network.feedForward({1.0, 0.0});
        const auto outputs = network.getResults();
        require(outputs.size() == 2, "expected two output neurons");
        const double first = outputs[0] - targets[0];
        const double second = outputs[1] - targets[1];
        const double rmse = std::sqrt((first * first + second * second) / 2.0);
        expectedAverage = (100.0 * expectedAverage + rmse) / 101.0;
        network.backPropagation(targets);
        require(std::abs(network.getAvarageError() - expectedAverage) < 1e-12,
                "smoothed error must include both outputs and the previous average");
    }
}

void checkXorConvergence()
{
    for (unsigned seed = 0; seed < 5; ++seed) {
        Network network = xor_example::train(seed);
        for (const auto& row : xor_example::cases) {
            network.feedForward({row[0], row[1]});
            const double prediction = network.getResults()[0];
            require(std::isfinite(prediction), "prediction must be finite");
            require((prediction >= 0.5 ? 1 : 0) == static_cast<int>(row[2]),
                    "incorrect XOR classification for seed " + std::to_string(seed));
            require(std::abs(prediction - row[2]) < 0.1,
                    "prediction too far from target for seed " + std::to_string(seed));
        }
    }
}
}

int main()
{
    try {
        checkDimensions();
        checkErrorMetric();
        checkXorConvergence();
        std::cout << "PASS: dimension checks, initialized/multi-output error metric and XOR convergence for seeds 0-4\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
