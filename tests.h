#pragma once
#include"ActorCritic.h"
#include<iostream>
struct tests {
	void test1() {
		VectorXd k(3);
		k << 3.0, 7.0, 9.0;
		VectorXd* p = &k;
		neuralNetwork::neuron n(p);

		cout << (*n.inputs)[1] << endl;
		cout << (n.weights)[1] << endl;
	}

	void test2() {
		VectorXd k(3);
		k << 3.0, 7.0, 9.0;
		VectorXd* p = &k;
		neuralNetwork NN(3, 0.1, logist);
		NN.run(k);
		cout << "out" << endl;
		cout << (*NN.outputs)[1] << endl;
		system("pause");
	}

	void test3() {
		VectorXd k(5);
		k << 3.0, 7.0, 9.0, -3.0, -5.0;
		VectorXd* p = &k;
		neuralNetwork NN(5, 0.1, logist);
		NN.addLayer(3);
		VectorXd out(3);
		out << 0, 1, 0;
		NN.run(k);
		cout << "out" << endl;
		cout << (*NN.outputs) << endl;
		for (int i = 0; i < 100; i++) {
			NN.backpropagation(k, out);
		}
		NN.run(k);
		cout << "out" << endl;
		cout << (*NN.outputs) << endl;
		system("pause");
	}

	void test4() {
		MatrixXd k(5, 1);
		k.col(0) << 3.0, 7.0, 9.0, -3.0, -5.0;
		MatrixXd* p = &k;
		neuralNetwork NN(5, 0.1, logist);
		NN.addLayer(3);
		MatrixXd out(3, 1);
		out.col(0) << 0, 1, 0;
		NN.run(k.col(0));
		cout << "out" << endl;
		cout << (*NN.outputs) << endl;


		NN.batchBackpropagation(k, out, 0.1);


		NN.run(k.col(0));
		cout << "out" << endl;
		cout << (*NN.outputs) << endl;


		system("pause");
	}

	void test5() {
		
	}

	void test6() {
		MatrixXd k(5, 1);
		
		k.col(0) << rand(), rand(), rand(), rand(), rand();
		k.normalize();
		MatrixXd* p = &k;
		neuralNetwork NN(5, 0.1, logist);
		NN.addLayer(3);
		MatrixXd out(3, 1);
		out.col(0) << 0.5, 0.5, 0.5;
		NN.run(k.col(0));
		cout << "out" << endl;
		cout << (*NN.outputs) << endl;


		NN.batchBackpropagation(k, out, 0.1);


		NN.run(k.col(0));
		cout << "out" << endl;
		cout << (*NN.outputs) << endl;


		system("pause");
	}

	void test7() {
		cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
		qAgent ag(1, cp.current, 0.3, 0.9, agent::epsilonGreedy, &cp);
		ag.epsilon = 0.4;
		while ((ag).learn() != 1) {
		}
		
		system("pause");
	}

	tests() {};
};