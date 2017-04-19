#pragma once
#include <cstdlib>
#include <windows.h>
#include <string>
#include <sstream>
#include <ctime>
#include <fstream>
#include "Environment3.h"
using namespace std;

class agent
{ // abstract agent class

public:
	int first = 0;
	int steps = 0;
	int runs = 0;
	int learnedOn = -1;
	int avgSteps = 0;
	int goals = 0;
	int mistakes = 0;
	int oldGoals = 0;
	bool won = false;
	int goalsRow = 0;
	int relearn = 1;
	vector<int> current_time;
	int startTime, stepTime;
	MatrixXd PRMatr;
	MatrixXd PRTarget;
	bool PR = false; // if we use pseudorehearsals
	int PRsize = 0;
	bool max = true;

	vector<vector<VectorXd> > PRactivations;
	MatrixXd makePR(int ps)
	{
		PRMatr = classicPR(ps, (*actionValues.inputs));
		PRactivations = actionValues.activationsRun(PRMatr);
		PRTarget = actionValues.batchRun(PRMatr);
		return PRMatr;
	}

	void nullize()
	{
		first = 0;
		steps = 0;
		runs = 0;
		learnedOn = -1;
		avgSteps = 0;
		goals = 0;
		mistakes = 0;
		oldGoals = 0;
		won = false;
		goalsRow = 0;
		relearn = 0;
		PR = false;
		PRsize = 0;
		actionValues = neuralNetwork(currentState.value.size(), learningRate, tang);
		actionValues.addLayer(20);
		actionValues.addLayer(env->actions.size());
		policy = neuralNetwork(currentState.value.size(), learningRate, logist);
		policy.addLayer(20);
		policy.addLayer(env->actions.size());
		epsilon = 0.4;
	}

	VectorXd predicted;
	enum moveType { Greedy, epsilonGreedy, softmax, policyBased };
	neuralNetwork policy; // policy vector
	neuralNetwork actionValues; // action-value vector we can use same structures both for policy & actionValues,
								// the only diffrence is that for actionValues probs show values, nit actual probability
	int size; // number of states
	state* map; // map map

	environment* env;
	int buf = 0; // actionbuffer
	double learningRate;
	double discountingFactor;
	moveType currentMoveType = Greedy;
	double epsilon = 0;
	int actionsNum;
	state* pointerToState = new state();
	state currentState = *pointerToState;
	double temperature = 2; // for softmax
	action* last;
	bool learned = false;
	int epsilons = 0;
	int stacked = 0;

	agent()
	{
	}

	agent(int sz, state curSt, double LR, double DF, int mt, environment* e)
		: currentState(curSt)
		, policy(curSt.value.size(), LR, logist)
		, actionValues(curSt.value.size(), LR, tang)
	{ // constr
		env = e;
		learningRate = LR;
		discountingFactor = DF;
		size = sz;
		currentMoveType = (moveType)mt;
		policy.addLayer(70);
		policy.addLayer(env->actions.size());
		actionValues.addLayer(70);
		actionValues.addLayer(env->actions.size());
		actionsNum = env->actions.size();
	}

	agent(const agent &oth) : agent(oth.size, oth.currentState, oth.learningRate, oth.discountingFactor,
		oth.currentMoveType, oth.env){
		actionValues = oth.actionValues;
		policy = oth.policy;
	}

	//int avgSteps = 0; // avg steps per episode (last more valuable)
	//int runs = 0; // episodes run

	double gamble()
	{ // random number from 0 to 1, with step 0.001
		return (static_cast<double>(rand() % 1000)) / 1000;
	}

	void setEpsilon(double eps)
	{ // for e-greedy
		epsilon = eps;
	}

	void setTemperature(double tmp)
	{ // for softmax
		temperature = tmp;
	}

	void changePolicy(moveType mt)
	{ // change traversal type
		currentMoveType = mt;
	}

	virtual int reward(int rw)
	{ // count reward
		return 0;
	}

	int getSize()
	{ // getter
		return size;
	}

	double move()
	{ // make move according to current moveType
		if (currentMoveType == Greedy) {
			return greedyMove();
		}
		else if (currentMoveType == epsilonGreedy) {
			return epsilonGreedyMove(epsilon);
		}
		else if (currentMoveType == policyBased) {
			return policyMove();
		}
		else {
			return softMax();
		}
	}

	int moves = 0;

	double makeAction(int act)
	{ // change state

		last = &(env->actions[act]);
		currentState = (*last).to;
		moves++;

		return env->reward(act);
	}

	double greedyMove()
	{

		VectorXd current = actionValues.run(currentState.norm()); // take entity from policies for current state
																 // cout << "predict vec " << current.transpose() << endl;
		state old = currentState;

		double max = current[0]; // choose highest valued action, initially - zeroes
		int actions = current.size();
		vector<int> maxIndexes;
		int m = 0;
		maxIndexes.push_back(m);
		int n = 2; // for distribution close to uniform
		for (int i = 1; i < actions; i++) { // choose all moves with maximum values
			if (current[i] > max) { // if new action will give bigger error
				max = current[i]; // make it new max
				m = i;
				maxIndexes.clear();
				maxIndexes.push_back(m);

			}
			else if (current[i] == max) {
				// to make random choise in case of multiple actions with the same msximum value
				maxIndexes.push_back(i);
			}
		}
		// choose action
		m = maxIndexes[(rand() % maxIndexes.size())];
		// put action in buffer and do it
		makeAction(m);
		buf = m; // save last action

		return env->reward(); // return reward
	}

	double epsilonGreedyMove(double epsilon)
	{

		if (gamble() > epsilon) { // do greedy move
			return greedyMove();
		}
		epsilons++;
		return randomMove();
	}

	virtual double randomMove()
	{
		makeAction(rand() % actionsNum);
		return env->reward();
	}

	virtual double policyMove()
	{ // policy based
		double chance = gamble();
		VectorXd current = policy.run(currentState.norm()); // take entity from policies for current state
														   // cout << "predict vec " << current.transpose() << endl;
		state old = currentState;

		double max = current[0]; // choose highest valued action, initially - zeroes
		int actions = current.size();

		current.normalize();
		chance = chance / current.norm();
		// choose action
		int i;
		for (i = 0; i < actions; i++) {
			if (chance < current[i]) {
				makeAction(i);
				break;
			}
			else {
				chance -= current[i];
			}
		}

		return env->reward(); // return reward
	}

	double softMax()
	{ // policy based softmax
		state old = currentState;
		int actions = env->actions.size();
		VectorXd chanses(actions);
		double denominator = 0;
		double check = 0;
		VectorXd res = policy.run(currentState.norm());
		for (int i = 0; i < actions; i++) { // choose probabilities by softmax formula
			denominator += exp(res[i] / temperature);
		}
		for (int i = 0; i < actions; i++) {
			chanses[i] = exp(res[i] / temperature) / denominator;
			check += chanses[i];
		}
		//cout << check << endl;
		assert((check - 1)<0.01);

		double dice = gamble() / 1000;
		for (int i = 0; i < actions; i++) { // if gamble < chance - move; else check next variant
			if (chanses[i] >= dice) {

				makeAction(i);
				break;
			}
			else {
				dice -= chanses[i];
			}
		}
		return env->reward();
	}

//	int steps = 0;

	virtual int learn()
	{
		return 0;
	}

	virtual VectorXd getRewards()
	{
		return VectorXd::Zero(1);
	}

	virtual VectorXd maxQ(int st)
	{
		return VectorXd::Zero(1);
	}
};

class qAgent : public agent
{
	

public:
	

	qAgent(int sz, state curSt, double LR, double DF, int mt, environment* e) : agent(sz, curSt, LR, DF, mt, e)
	{ 
	}

	qAgent(const qAgent& oth) : agent(oth)
	{ 
	}

	

	void improvement(state last, VectorXd target)
	{
		VectorXd run = last.norm();
	
		// cout << "in " << run.transpose() << endl;
		// cout << "target " << target.transpose() << endl;
		if (PR) {
			PRMatr.col(0) = run;
			PRTarget.col(0) = target;
			actionValues.batchBackpropagation(PRMatr, PRTarget, 0.001);
		}
		else {
			actionValues.backpropagation(run, target);
		}
	}

	void primprovement(state last, VectorXd target)
	{
		VectorXd run = last.norm();
	
		// cout << "in " << run.transpose() << endl;
		// cout << "target " << target.transpose() << endl;
		if (PR) {
			PRMatr.col(0) = run;
			PRTarget.col(0) = target;
			actionValues.prBackpropagation(PRMatr, PRactivations, PRTarget, 0.00001);
		}
		else {
			actionValues.backpropagation(run, target);
		}
	}
	bool FreanRobins;

	void qLearn(string fileout){//random
		ofstream fout;
		fout.open(fileout);
		VectorXd stepVec = VectorXd::Zero(5000);
		VectorXd etVec = VectorXd::Zero(5000);
		VectorXd esVec = VectorXd::Zero(5000);
		for (int iter = 0; iter < 1; iter++) {
			
			for (int i = 0; i < 5000; i++) {

				while (randrun() != -1) {
				}

				stepVec[i] += steps;
				etVec[i] += startTime;
				esVec[i] += epsilons;

				stacked = 0;
				currentState.num = 0;
				steps = 0;
				epsilons = 0;
			}
			cout << "finished" << endl;
			this->nullize();
		}
		for (int i = 0; i < 5000; i++) {
			fout << i << ";";
			fout << "steps done "
				<< ";" << stepVec[i] / 1 << ";";
			fout << "episode time "
				<< ";" << etVec[i] / 1 << ";";
			fout << " epsilon steps "
				<< ";" << esVec[i] / 1 << endl;
		}

		fout.close();
	}
	void qLearn(int PRSz, int inrelearn, bool maxin, bool fr, int iterations, int averaging, string fileout)
	{
		FreanRobins = fr;
		ofstream fout;
		fout.open(fileout);
		VectorXd stepVec = VectorXd::Zero(iterations);
		VectorXd etVec = VectorXd::Zero(iterations);
		VectorXd esVec = VectorXd::Zero(iterations);
		for (int iter = 0; iter < averaging; iter++) {

			learn();
			max = maxin;
			relearn = inrelearn;
			PRsize = PRSz;
			for (int i = 0; i < iterations; i++) {
				if (PRsize != 0) {
					PR = true;
					makePR(PRsize);
				}
				while (learn() != -1) {
				}

				stepVec[i] += steps;
				etVec[i] += startTime;
				esVec[i] += epsilons;

				stacked = 0;
				currentState.num = 0;
				steps = 0;
				epsilons = 0;
			}
			cout << "fineshed" << endl;
			this->nullize();
		}
		for (int i = 0; i < iterations; i++) {
			fout << i << ";";
			fout << "steps done "
				<< ";" << stepVec[i] / averaging << ";";
			fout << "episode time "
				<< ";" << etVec[i] / averaging << ";";
			fout << " epsilon steps "
				<< ";" << esVec[i] / averaging << endl;
		}
		fout.close();
	}

	int randrun() {
		if (steps == 0) {
			startTime = clock();
			current_time.clear();
		}
		stepTime = clock();

	
		state last = currentState; // save state was current on the enering evaluation function
	

		double rew = randomMove(); // move and save reward
		int done = buf; // save last action done
		state alloc = currentState; // save value of cell we came to

		if (env->isFinal(alloc)) { // if finished episode

			
			steps++;
			avgSteps = steps;
			startTime = clock() - startTime;
			current_time.push_back(clock() - stepTime);
			double time = 0;
			for (unsigned i = 0; i < current_time.size(); i++) {
				time += current_time[i];
			}
			time = time / current_time.size();

			cout << "steps done " << steps << " fail" << endl;
			cout << "average steps time " << time << endl;
			cout << "episode time " << startTime << endl;
			cout << " epsilon steps " << epsilons << endl;


			runs++;
			
			if (steps >= 3000) {
				env->makeFinal();
			}
			return -1;

		}
		else { // not terminal

			steps++;
			return 0;
		}
	}


	int learn()
	{
		if (steps == 0) {
			startTime = clock();
			current_time.clear();
		}
		stepTime = clock();

		VectorXd target(actionsNum);
		VectorXd oldValues = actionValues.run(currentState.norm());
		target = oldValues; // to not relearn values of actions not done

		state last = currentState; // save state was current on the enering evaluation function
		VectorXd Q(actionsNum);

		
			Q = maxQ();
		
		double rew = move(); // move and save reward
		int done = buf; // save last action done
		state alloc = currentState; // save value of cell we came to
		target[done] = 0;

		if (env->isFinal(alloc) || steps == 5000) { // if finished episode

			if ((learnedOn == -1) && steps >= 5000) { // if handled pole for more than 8000 steps - reached the goal
				goalsRow++;
				learnedOn = runs;
				learned = true;
				goals++;

				if (first == 0) {
					first = runs;
				}

			}
			else if (steps >= 5000) {
				mistakes = 0;
				goals++;
				goalsRow++;

			}
			else if (learnedOn != -1 &&
				steps < 5000) { // if we didn't come three times in a row - we possibly didn't learned
				if (mistakes >= 3) {
					learnedOn = -1;
					mistakes = 0;
					learned = false;
					goalsRow = 0;
					won = false;
				}
				else {
					mistakes++;
					goalsRow = 0;
				}
			}
			if (PRsize != 0) {
				if (relearn == -1) {
					cout << "make PR!" << endl;
					makePR(PRsize);
				}
				else if (relearn != 0) {
					if (PR && runs % relearn == 0) {
						cout << "make PR!" << endl;
						makePR(PRsize);
					}
				}
			}

			steps++;
			avgSteps = steps;
			startTime = clock() - startTime;
			current_time.push_back(clock() - stepTime);
			double time = 0;
			for (unsigned i = 0; i < current_time.size(); i++) {
				time += current_time[i];
			}
			time = time / current_time.size();

			cout << "steps done " << steps << " fail" << endl;
			cout << "average steps time " << time << endl;
			cout << "episode time " << startTime << endl;
			cout << " epsilon steps " << epsilons << endl;


			runs++;
			target[done] += rew;
			target[done] = (target[done] - oldValues[done]);
			/*if (runs > 800) {
			cout << "target" << target << endl;
			}*/
			if (FreanRobins) {
				primprovement(last, target);
			}
			else if (PR) {
				target[done] = target[done] * learningRate + oldValues[done];
				improvement(last, target);
			}
			else {
				improvement(last, target);
			}
			setEpsilon(epsilon * 0.80);
			// system("pause");
			if (steps >= 3000) {
				env->makeFinal();
			}
			return -1;

		}
		else { // not terminal

			steps++;

			VectorXd res = discountingFactor * Q;

			target[done] = (target[done] + res[done] - oldValues[done]); // target[done] here gives nothing as our
																		 // reward is 0 here, but it will be useful in
																		 // other problems
																		 /*if (runs > 800) {
																		 cout << "target" << target << endl;
																		 }*/
																		 /*target = target / 3;*/
			if (FreanRobins) {
				primprovement(last, target);
			}
			else if (PR) {
				target[done] = target[done] * learningRate + oldValues[done];
				improvement(last, target);
			}
			else {
				improvement(last, target);
			}
			current_time.push_back(clock() - stepTime);
			return 0;
		}
	}

	VectorXd getRewards()
	{
		VectorXd res(actionsNum);
		for (int i = 0; i < actionsNum; i++) {
			res[i] = env->actions[i].reward;
		}
		return res;
	}

	VectorXd maxQ()
	{
		VectorXd res = VectorXd::Zero(actionsNum);
		state st = env->actions[buf].to;
		res[buf] = actionValues.run(st.norm()).maxCoeff();

		return res;
	}

};
