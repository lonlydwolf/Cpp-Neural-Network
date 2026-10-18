#pragma once
#include <iostream>
#include <time.h>
#include <vector>
#include <cstdlib>

using namespace std;
class Neuron;
typedef vector<Neuron> Layer;
struct Connection
{
    double Weight;
    double deltaWeight;
};

class Neuron
{
public:
    Neuron(unsigned numberOfOutputs, unsigned neuronIndex);
    void setOutputValue(const double& inputValue) { outputValue = inputValue; }
    void feedForward(const Layer& previousLayer);
    double getOutputValue() const { return outputValue; }
    double getOutputWeightof(unsigned n) const { return outputWeights[n].Weight; }
    void calculateOutputGradients(double targetVAlue);
    void calculateHiddenGradients(const Layer& nextLayer);
    void updateWeights(Layer& pervLayer);
private:
    static double traningRate;
    static double Momentum;
    double Gradient = 0.0;
    static double activationFunction(double sum);
    static double activationFunctionDerivative(double sum);
    double randomWeight() { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer& nextLayer)const;
    unsigned Index;
    double outputValue = 0.0;
    vector<Connection> outputWeights;
    double bias;
};

