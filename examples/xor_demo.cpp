#include "xor_example.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    unsigned seed = 0;
    try {
        if (argc != 1) {
            if (argc != 3 || std::string(argv[1]) != "--seed") {
                throw std::invalid_argument("expected --seed followed by an unsigned integer");
            }
            const std::string value(argv[2]);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                throw std::invalid_argument("seed must contain only decimal digits");
            }
            const auto parsed = std::stoull(value);
            if (parsed > std::numeric_limits<unsigned>::max()) {
                throw std::out_of_range("seed is too large");
            }
            seed = static_cast<unsigned>(parsed);
        }
    } catch (const std::exception& error) {
        std::cerr << "Usage: xor_demo [--seed N]\n" << error.what() << '\n';
        return 2;
    }

    Network network = xor_example::train(seed);
    std::cout << "Seed: " << seed << "; training steps: " << xor_example::trainingSteps << '\n';
    std::cout << "Inputs  Target  Prediction  Class (threshold 0.5)\n";
    unsigned correct = 0;
    for (const auto& row : xor_example::cases) {
        network.feedForward({row[0], row[1]});
        const double prediction = network.getResults()[0];
        const int result = prediction >= 0.5 ? 1 : 0;
        correct += result == static_cast<int>(row[2]);
        std::cout << static_cast<int>(row[0]) << ' ' << static_cast<int>(row[1])
                  << "     " << static_cast<int>(row[2]) << "       "
                  << std::fixed << std::setprecision(6) << prediction << "    " << result << '\n';
    }
    std::cout << "Correct: " << correct << "/4\n";
    return correct == xor_example::cases.size() ? 0 : 1;
}
