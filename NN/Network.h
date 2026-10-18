#pragma once
#include <iostream>
#include <time.h>
#include <vector>
#include "Neuron.h"
#include <cassert>
#include<fstream>
using namespace std;
typedef vector<Neuron> Layer;


class Network
{
public:
	Network(const vector<unsigned>& topology);
	void feedForward(const vector<double>& inputValues);
	void backPropagation(const vector<double>& targetValues);
	void printResult(ofstream &data);
	vector<double> getResults() const;
	double getAvarageError()const { return averageError; }
private:
	vector<Layer> Layers;
	double Error = 0.0;
	double averageError = 0.0;
	static double numberToAverage;
};

