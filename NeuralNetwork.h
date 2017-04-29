#pragma once
#include<Eigen/Dense>
#include<random>
#include<assert.h>
#include<iostream>
#include <ctime>
#include"pseudorehearsal.h"
#define INPUTS (*inputs)
#define OUTPUTS (*outputs)
#define NEURONS (*neurons)
#define THETA (*theta)

using namespace std;
using namespace Eigen;

enum functions { logist, gauss, tang, lin };

inline double randDouble() {//random [0,1]

	return static_cast<double>(rand() % 10) / 100;
}

inline double logistic(double z) {//transform to sigmoid
	double d = (1 / (1 + exp(-z)));
	if (d >= 1) return 0.9999;
	if (d <= -1) return -0.9999;
	return d;
}

inline double tanhf(double z) {//transform to sigmoid
	return tanh(z);
}

inline double linear(double z) {
	return z;
}

inline double gaussian(double z) {
	return exp(-((2 * z)*(2 * z)));
}

struct neuralNetwork {

	double reg = 0;
	struct neuron {
		VectorXd* inputs;
		VectorXd weights;
		double* output;

		neuron() {

		}

		neuron(VectorXd *inp) : inputs(inp), weights(VectorXd::Random((*inputs).size() + 1)) {
			output = new double;
			*output = 0;
		}

		neuron(const neuron &oth) :inputs(oth.inputs), weights(oth.weights), output(oth.output) {

		}

		inline double act() {//calculate output
			VectorXd ins(INPUTS.size() + 1);
			ins << 1, *inputs;
			ins = ins.cwiseProduct(weights);
			*output = ins.sum();
			return *output;
		}

		inline void learn(double upd, int at) {//update weight at num by upd
			weights[at] += upd;
		}
	};//end of neuron struct


	struct layer {
		functions layersFunc;
		int size = 0;
		vector<neuron> *neurons;
		VectorXd *inputs;
		VectorXd *outputs;
		MatrixXd *theta;
		layer() {//default

		}


		layer(int layerSize, VectorXd *in) {//initial
			neurons = new vector<neuron>;
			inputs = new VectorXd;
			outputs = new VectorXd(layerSize);
			inputs = in;
			size = layerSize;
			for (int i = 0; i < size; i++) {
				NEURONS.push_back(neuron(inputs));
				OUTPUTS[i] = (*NEURONS[i].output);
			}
			theta = new MatrixXd;
			*theta = countTheta();
		}

		layer(const layer &oth) {
			neurons = oth.neurons;
			inputs = oth.inputs;
			outputs = oth.outputs;
			theta = oth.theta;
			size = oth.size;
			layersFunc = oth.layersFunc;
		}


		inline MatrixXd countTheta() {//get matrix of weights
			(*theta).resize(0, 0);
			(*theta).resize(INPUTS.size() + 1, size);
			for (unsigned i = 0; i < (*neurons).size(); i++) {
				(*theta).col(i) = ((*neurons)[i].weights);
			}
			return (*theta);
		}

		inline VectorXd run() {//run input vector and return output vector
			for (int i = 0; i < size; i++) {
				(*outputs)[i] = sigmoid((*neurons)[i].act());
			}
			return *outputs;
		}

		inline double sigmoid(double d) {
			if (layersFunc == logist) {
				return logistic(d);
			}
			else if (layersFunc == gauss) {
				return gaussian(d);
			}
			else if (layersFunc == tang) {
				return tanhf(d);
			}
			else {
				return d;
			}
		}

	};//end of layer struct

	vector<layer> layers;
	VectorXd *inputs;//inputs vector
	VectorXd *outputs;
	double learningRate;
	int depth = 1;
	int iterations = 0;
	double meanError = 1000;
	double oldErr = 1100;
	int epohs = 0;

	neuralNetwork() {}

	neuralNetwork(int featuresNum, double LR, functions f) {//basic constructor - empty NN
		inputs = new VectorXd;
		outputs = new VectorXd;
		func = f;
		learningRate = LR;
		INPUTS = VectorXd::Zero(featuresNum);
		OUTPUTS = VectorXd::Zero(featuresNum);
		assert(INPUTS.size() == featuresNum);
		assert(INPUTS[0] == 0);
		layers.push_back(layer(featuresNum, inputs));
		outputs = layers[0].outputs;
		assert(inputs == layers[0].inputs);
		layers[0].layersFunc = func;
	}
	neuralNetwork(const neuralNetwork &oth) {
		inputs = oth.inputs;
		outputs = oth.outputs;
		func = oth.func;
		learningRate = oth.learningRate;
		layers = oth.layers;
	}

	inline void addLayer(int neurons) {

		VectorXd *prevOuts = layers[depth - 1].outputs;
		layer newLayer = layer(neurons, prevOuts);
		newLayer.layersFunc = func;
		layers.push_back(newLayer);
		assert((*layers[depth].neurons)[0].weights[0] == (*newLayer.neurons)[0].weights[0]);
		depth++;
		outputs = layers[depth - 1].outputs;
	}

	inline VectorXd run(VectorXd inVector) {//stochastic
		*inputs = inVector;
		VectorXd result;
		for (int i = 0; i < depth - 1; i++) {
			layers[i].run();
		}
		result = layers[depth - 1].run();
		assert(*outputs == result);
		countActivations();
		return result;
	}

	inline void changeFunc(functions f) {
		func = f;
		for (int i = 0; i < depth; i++) {
			layers[i].layersFunc = f;
		}
	}

	inline MatrixXd batchRun(MatrixXd& inMatrix) {//batch m rows n features
		MatrixXd outMatrix(OUTPUTS.size(), inMatrix.cols());
		for (int i = 0; i < inMatrix.cols(); i++) {
			*inputs = inMatrix.col(i); //i-th example vector
			VectorXd result;
			for (int j = 0; j < depth - 1; j++) {
				layers[j].run();
			}
			result = layers[depth - 1].run();
			assert(*outputs == result);
			outMatrix.col(i) = (result);
		}

		return outMatrix;
	}

	vector<MatrixXd> backpropagation(VectorXd in, VectorXd expectedResult) {//stochastic

		*inputs = in;
		run(*inputs);
		activations = countActivations();
		err = errors(expectedResult);


		vector<MatrixXd> delta;
		for (int l = 0; l < depth; l++) {
			int neurSize = (*layers[l].neurons).size();
			int weightsSize = (*layers[l].neurons)[0].weights.size();
			MatrixXd del(weightsSize, neurSize);
			for (int j = 0; j < neurSize; j++) {
				VectorXd d(weightsSize);
				for (int i = 0; i < weightsSize; i++) {

					double update = activations[l][i] * err[l + 1][j] + reg*(*layers[l].neurons)[j].weights[i];

					d[i] = (update);
					if (stochastic) {
						
						(*layers[l].neurons)[j].weights[i] -= (0.5*(learningRate*update));
					}
				}
				del.col(j) = (d);
			}

			delta.push_back(del);
		}

		if (stochastic) {
			VectorXd errV = activations[depth] - expectedResult;
			meanError = 0;
			for (int i = 0; i < errV.size(); i++) {
				meanError += errV[i] * errV[i];
			}
		}
		iterations++;
		return delta;

	}//single bp

	void flushIt() {
		iterations = 0;
	}
	int det;
	bool learned;
	void batchBackpropagation(MatrixXd inputsVec, MatrixXd expectedresults, double acceptableErr) {
        double start = clock();
        double diff = 0;
		double initlr = learningRate;
		det = 0;
		while (1) {
            double step = clock();
			double k;
			while ((k = abs(learn(inputsVec, expectedresults)))>acceptableErr) {
				learningRate = learningRate*0.99;
                 diff = clock()-step;
                 if(clock() - start + diff >=10){//time constraint for env
                     break;
                 }
                 step = clock();
			}
            if(clock() - start + diff >=10){
                     break;
                 }
            
			//cout << "increase!" << endl;
			//cout << "errs = " << k << endl;
			//cout << "det = " << det << endl;
			//cout << "learning rate = " << learningRate << endl;
			if ((abs(k))>acceptableErr) {
				learningRate = learningRate / 5;
				
			}
			else {
				break;
			}

		}
		//cout << "learned in " << epohs << " epohs" << endl;
		//cout << "learned in " << iterations << "iterations" << endl;
		epohs = 0;
		iterations = 0;
		meanError = 1000;
		oldErr = 1100;
		stochastic = true;
		learningRate = initlr;
	}

	double learn(MatrixXd inputsVec, MatrixXd expectedresults) {
		stochastic = false;
		unsigned m = inputsVec.cols();
		vector<vector<MatrixXd>> deltas;
		for (unsigned i = 0; i < m; i++) {
			deltas.push_back(backpropagation(inputsVec.col(i), expectedresults.col(i)));
		}
		vector<MatrixXd> delta = deltas[0];
		for (unsigned i = 0; i < m; i++) {
			for (unsigned k = 0; k < delta.size(); k++) {
				delta[k] = delta[k] + deltas[i][k];
			}
		}

		for (int l = 0; l < depth; l++) {

			for (unsigned j = 0; j < (*layers[l].neurons).size(); j++) {

				for (int i = 0; i < (*layers[l].neurons)[j].weights.size(); i++) {


					double update = delta[l].col(j)[i] / m;


					(*layers[l].neurons)[j].weights[i] -= (0.5*(learningRate / (det + 1))*update);

				}

			}
		}
		MatrixXd results = batchRun(inputsVec);
		MatrixXd err = results - expectedresults;
		//cout << "res" << results << endl;
		//cout << "err" << err << endl;



		double E = err.squaredNorm();
		E = (E) / m;
		oldErr = meanError;
		meanError = E;
			//cout << "mean " << meanError << endl;
		//	cout << "old " << oldErr << endl;
		epohs++;
		/*if (abs(meanError - oldErr) > acceptableErr) {
		batchBackpropagation(inputsVec, expectedresults, acceptableErr);

		}
		else {
		//cout << "learned in " << epohs << " epohs" << endl;
		//cout << "learned in " << iterations << "iterations" << endl;
		epohs = 0;
		iterations = 0;
		meanError = 1000;
		stochastic = true;
		}*/
		return oldErr - meanError;

	}

	vector<vector<VectorXd>> activationsRun(MatrixXd PRIn) {
		vector<vector<VectorXd>> result;
		for (int i = 0; i < PRIn.cols(); i++) {
			this->run(PRIn.col(i));
			result.push_back(this->activations);
		}
		return result;
	}

	void prBackpropagation(MatrixXd PRMatr, vector<vector<VectorXd>> PRActivations, MatrixXd expectedresult, double acceptableErr)
	{
		*inputs = PRMatr.col(0);
		run(*inputs);
		activations = countActivations();
		err = errors(expectedresult.col(0));
		vector<VectorXd> errb = err;
		
		//cout << "actual " << expectedresult.col(0).transpose() << endl;
		//cout << "got " << activations[depth].transpose() << endl;

		for (int x = 1; x < PRMatr.cols(); x++) {
			for (int l = 0; l < depth; l++) {
				int neurSize = (*layers[l].neurons).size();
				int weightsSize = (*layers[l].neurons)[0].weights.size();
                //double theta = acos(prActivations[x][l].dot(activations[l])/(prActivations[x][l].squaredNorm()*activations[l].squaredNorm()))
				VectorXd bxx = activations[l] * (PRActivations[x][l].dot(PRActivations[x][l]));
				VectorXd xxb = PRActivations[x][l] * (PRActivations[x][l].dot(activations[l]));
				VectorXd wght = ((bxx - xxb) / ((activations[l].dot(activations[l]))*(PRActivations[x][l].dot(PRActivations[x][l])) - (activations[l].dot(PRActivations[x][l]))*(activations[l].dot(PRActivations[x][l]))));

				for (int j = 0; j < neurSize; j++) {

					VectorXd upd = errb[l + 1][j] * wght;
					for (int i = 0; i < weightsSize; i++) {

						double update = learningRate*(upd[i] / PRMatr.cols() + reg*(*layers[l].neurons)[j].weights[i]);



						(*layers[l].neurons)[j].weights[i] -= learningRate*update;

					}

				}


			}
		}
		iterations++;

	}


private:

	int PRSize;//number of PR set
	default_random_engine generator;
	vector<VectorXd> activations;
	vector<VectorXd> err;//errors matrix
	vector<VectorXd> derivative;
	functions func;
	bool stochastic = true;
	MatrixXd PRIn, PROut;
	bool PR = false;


	inline vector<VectorXd> countActivations() {//gets activation matrix
		activations.clear();//clear matrix
		assert(*inputs == *layers[0].inputs);
		VectorXd in(INPUTS.size() + 1);//place for layers inputs + bias element
		in << 1, INPUTS;
		activations.push_back(in);//a[0]=in
		for (int i = 0; i < depth - 1; i++) {//for each layer except last - add outputs
			in.resize(0);//clear vector after previous insertion
			in.resize((*layers[i].outputs).size() + 1);//get place
			in << 1, *layers[i].outputs;//add bias and data
			activations.push_back(in);//a[i+1]=out[i]
		}
		activations.push_back(*layers[depth - 1].outputs);//add output without bias
		return activations;
	}

	inline vector<VectorXd> countDerivatives() {
		derivative.clear();
		derivative.push_back(VectorXd::Zero(INPUTS.size()));//inputs can't have errors
		for (int l = 1; l <= depth; l++) {//no derivative[0] because inputs haven't error
			derivative.push_back(derivativeVec(activations[l]));
		}
		return derivative;
	}

	vector<VectorXd> errors(VectorXd expectedResult) {

		vector<MatrixXd> thetas; //theta matrices
		double error = 0; //squared total error
		err.clear();
		for (int i = 0; i < depth; i++) {//get thetas and init errors vector
			thetas.push_back((layers[i].countTheta()));
			err.push_back(VectorXd::Zero(activations[i].size()));//initialize row of errors for each neuron and bias unit

		}
		//output layer
		thetas.push_back((layers[depth - 1].countTheta()));
		err.push_back(VectorXd::Zero(activations[depth].size()));//output - no care about bias
		assert(err[depth].size() == activations[depth].size());

		err[depth] = activations[depth] - expectedResult;//output error
	//	cout << "err " << err[depth].transpose() << endl;
		countDerivatives();

		for (int l = depth - 1; l > 0; l--) {

			MatrixXd er;
			int n = (l == depth - 1) ? 0 : 1;
			er = err[l + 1].tail(err[l + 1].size() - n);

			MatrixXd m = (thetas[l])*(er);

			err[l] = m.cwiseProduct(derivative[l]);

		}

		VectorXd der = derivativeVec(activations[depth]);
		err[depth] = err[depth].cwiseProduct(der);

		return err;
	}

	inline double derivativeDot(double a) {
		if (func == logist) {
			return (a)*(1 - a);
		}
		else if (func == gauss) {
			return (8 * a)*exp(-4 * a*a);
		}
		else if (func == tang) {
			return 1 - (tanh(a)*tanh(a));
		}
		else {
			return a;
		}

	}

	inline VectorXd derivativeVec(VectorXd a) {
		if (func == logist) {
			return (a.cwiseProduct(VectorXd::Ones(a.size()) - a));
		}
		else if (func == gauss) {
			VectorXd res(a.size());
			for (int i = 0; i < a.size(); i++) {
				res[i] = derivativeDot(a[i]);
			}
			return res;
		}
		else if (func == tang) {
			VectorXd res(a.size());
			for (int i = 0; i < a.size(); i++) {
				res[i] = derivativeDot(a[i]);
			}
			return res;
		}
		else {
			return (VectorXd::Ones(a.size()));
		}
	}

};
