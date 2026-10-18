#include "Network.h"
#include <cmath>
#include <stdexcept>
double Network::numberToAverage = 100.0;
Network::Network(const vector<unsigned>& topology)
{
	if (topology.size() < 2) {
		throw std::invalid_argument("topology needs input and output layers");
	}
	for (unsigned width : topology) {
		if (width == 0) {
			throw std::invalid_argument("layer widths must be positive");
		}
	}
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		Layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];


		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum) {
			Layers.back().push_back(Neuron(numOutputs, neuronNum));
			//cout << "Neuron " << neuronNum << " from layer " << layerNum << " was born!" << endl;
		}
	}
}

void Network::feedForward(const vector<double>& inputValues)
{
	if (inputValues.size() != Layers[0].size()) {
		throw std::invalid_argument("input count does not match the input layer");
	}
	for (unsigned i = 0; i < inputValues.size(); i++)
	{
		Layers[0][i].setOutputValue(inputValues[i]);
	}
	for (unsigned i = 1; i < Layers.size(); i++)
	{
		Layer& perviousLayer = Layers[i - 1];
		for (unsigned j = 0; j < Layers[i].size(); j++)
			Layers[i][j].feedForward(perviousLayer);
	}
}
void Network::printResult(ofstream& data)
{
	Layer& outputLayer = Layers.back();
	data << "Result: " << outputLayer[0].getOutputValue() << endl;
}

vector<double> Network::getResults() const
{
	vector<double> results;
	for (const Neuron& neuron : Layers.back())
	{
		results.push_back(neuron.getOutputValue());
	}
	return results;
}

void Network::backPropagation(const vector<double>& targetValues)
{
	Layer& outputLayer = Layers.back();
	if (targetValues.size() != outputLayer.size()) {
		throw std::invalid_argument("target count does not match the output layer");
	}
	Error = 0.0;
	for (unsigned i = 0; i < outputLayer.size(); i++)
	{
		double delta = targetValues[i] - outputLayer[i].getOutputValue();
		Error += delta * delta;
	}
	Error /= outputLayer.size();
	Error = std::sqrt(Error);

	averageError = (averageError * numberToAverage + Error) / (numberToAverage + 1.0);

	for (unsigned i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i].calculateOutputGradients(targetValues[i]);
	}

	for (unsigned layerNumber = Layers.size() - 2; layerNumber > 0; layerNumber--)
	{
		Layer& hiddenLayer = Layers[layerNumber];
		Layer& nextLayer = Layers[layerNumber + 1];

		for (unsigned i = 0; i < hiddenLayer.size(); i++)
		{
			hiddenLayer[i].calculateHiddenGradients(nextLayer);
		}
	}

	for (unsigned layerNumber = Layers.size() - 1; layerNumber > 0; layerNumber--)
	{
		Layer& layer = Layers[layerNumber];
		Layer& previousLayer = Layers[layerNumber - 1];

		for (unsigned i = 0; i < layer.size(); i++)
		{
			layer[i].updateWeights(previousLayer);
		}
	}

}
