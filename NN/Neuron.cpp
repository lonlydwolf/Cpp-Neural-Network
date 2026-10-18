#include "Neuron.h"
#include <cmath>

double Neuron::traningRate = 0.2;
double Neuron::Momentum = 0.5;
Neuron::Neuron(unsigned numberOfOutputs, unsigned neuronIndex)
{
    for (unsigned i = 0; i < numberOfOutputs; i++)
    {
        outputWeights.push_back(Connection());
        outputWeights.back().Weight = randomWeight();
    }
    Index = neuronIndex;
    bias = 1.0;
}
void Neuron::feedForward(const Layer& previousLayer)
{
    double sum = 0.0;
    for (unsigned i = 0; i < previousLayer.size(); i++)
    {
        sum += previousLayer[i].getOutputValue() * previousLayer[i].getOutputWeightof(this->Index);//sum of (a_i*w_i)
    }
    sum += bias;
    outputValue = activationFunction(sum);
}
double Neuron::activationFunction(double sum)
{
    return std::tanh(sum);
}
double Neuron::activationFunctionDerivative(double sum)
{
    return 1.0 - sum * sum;
}

void Neuron::calculateOutputGradients(double targetValue)
{
    double delta = targetValue - outputValue;
    Gradient = delta * activationFunctionDerivative(outputValue);
}
void Neuron::calculateHiddenGradients(const Layer& nextLayer)
{
    double dow = sumDOW(nextLayer);
    Gradient = dow * activationFunctionDerivative(outputValue);
}
double Neuron::sumDOW(const Layer& nextLayer) const
{
    double sum = 0.0;
    for (unsigned i = 0; i < nextLayer.size(); i++)
    {
        sum += outputWeights[i].Weight * nextLayer[i].Gradient;
    }
    return sum;
}
void Neuron::updateWeights(Layer& previousLayer)
{
    for (unsigned i = 0; i < previousLayer.size(); i++)
    {
        Neuron& neuron = previousLayer[i];
        double oldDeltaWeight = neuron.outputWeights[Index].deltaWeight;
        double newDeltaWeight = traningRate * neuron.getOutputValue() * Gradient + Momentum * oldDeltaWeight;
        neuron.outputWeights[Index].deltaWeight = newDeltaWeight;
        neuron.outputWeights[Index].Weight += newDeltaWeight;
        //cout << neuron.outputWeights[Index].deltaWeight << endl;
        //cout << neuron.outputWeights[Index].Weight << endl;
    }
}