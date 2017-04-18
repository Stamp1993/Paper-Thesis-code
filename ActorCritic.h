#pragma once
#include"agent.h"

class acAgent : public qAgent{
	//state currentState;
	//state oldState;
	//neuralNetwork actor; we have actionValues instead

public:
	

	acAgent(int sz, state curSt, double LR, double DF, environment* e) : qAgent(sz, curSt, LR, DF, softmax, e)
	{ // constr
		currentMoveType = softmax;
	}

	acAgent(const acAgent& oth) : qAgent(oth)
	{ // constr
		currentMoveType = softmax;
		PR = false;
	}
	
	void policyImprovement(state last, VectorXd target)
	{
		VectorXd run = last.norm();
		
		// cout << "in " << run.transpose() << endl;
		// cout << "target " << target.transpose() << endl;
		if (PR) {
			PRMatr.col(1) = run;
			PRTarget.col(1) = target;
			policy.batchBackpropagation(PRMatr, PRTarget, 0.001);
		}
		else {
			policy.backpropagation(run, target);
		}
	}

	void policyPRimprovement(state last, VectorXd target)
	{
		VectorXd run = last.norm();
		
		
		// cout << "in " << run.transpose() << endl;
		// cout << "target " << target.transpose() << endl;
		if (PR) {
			PRMatr.col(1) = run;
			PRTarget.col(1) = target;
			policy.prBackpropagation(PRMatr, PRactivations, PRTarget, 0.00001);
		}
		else {
			policy.backpropagation(run, target);
		}
	}

	void acLearn(int PRSz, int inrelearn, bool fr,  int len, int avgs, string fileout)
	{
		srand(runs*steps);
		FreanRobins = fr;
		ofstream fout;
		fout.open(fileout);
		VectorXd stepVec = VectorXd::Zero(len);
		VectorXd etVec = VectorXd::Zero(len);
		VectorXd esVec = VectorXd::Zero(len);
		for (int iter = 0; iter < avgs; iter++) {

			learn();
			relearn = inrelearn;
			PRsize = PRSz;
			for (int i = 0; i < len; i++) {

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

		for (int i = 0; i < len; i++) {
			fout << i << ";";
			fout << "steps done "
				<< ";" << stepVec[i] / avgs << ";";
			fout << "episode time "
				<< ";" << etVec[i] / avgs << ";";
			fout << " epsilon steps "
				<< ";" << esVec[i] / avgs << endl;
		}

		fout.close();
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

		VectorXd policytarget(actionsNum);
		VectorXd oldPolicy = policy.run(currentState.norm());
		policytarget = oldPolicy; // to not relearn values of actions not done

		state last = currentState; // save state was current on the enering evaluation function

		

		double rew = move(); // policy move and save reward
		int done = buf; // save last action done

		state alloc = currentState; // save value of cell we came to
		target[done] = rew;
		//policytarget[done] = 0;

		if (env->isFinal(alloc) || steps == 3000) { // if finished episode

			if ((learnedOn == -1) && steps >= 3000) { // if handled pole for more than 8000 steps - reached the goal
				goalsRow++;
				learnedOn = runs;
				learned = true;
				goals++;

				if (first == 0) {
					first = runs;
				}

			}
			else if (steps >= 3000) {
				mistakes = 0;
				goals++;
				goalsRow++;

			}
			else if (learnedOn != -1 &&
				steps < 3000) { // if we didn't come three times in a row - we possibly didn't learned
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
			target[done] = (target[done] - oldValues[done]);
			double td = target[done];
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

			oldValues = actionValues.run(currentState.norm());
			policytarget[done] = log(oldPolicy[done])*td;

			if (FreanRobins) {
				policyPRimprovement(last, policytarget);
			}
			else if (PR) {
				policytarget[done] = policytarget[done] * learningRate + oldPolicy[done];
				improvement(last, policytarget);
			}
			else {
				improvement(last, policytarget);
			}

			setEpsilon(epsilon * 0.80);
			// system("pause");
			if (steps >= 3000) {
				env->makeFinal();
			}

			//TODO: //policy things
			return -1;

		}
		else { // not terminal

			steps++;

			VectorXd res = discountingFactor * SARSA();

			target[done] = (target[done] + res[done] - oldValues[done]); // target[done] here gives nothing as our
																		 // reward is 0 here, but it will be useful in
																		 // other problems
																		 /*if (runs > 800) {
																		 cout << "target" << target << endl;
																		 }*/
			double td = target[done];															 /*target = target / 3;*/
			if (FreanRobins) {
				primprovement(last, target);
			}
			else if (PR) {
				target[done] * learningRate + oldValues[done];
				improvement(last, target);
			}
			else {
				improvement(last, target);
			}

			oldValues = actionValues.run(currentState.norm());
			policytarget[done] = log(oldPolicy[done])*td;

			if (FreanRobins) {
				policyPRimprovement(last, policytarget);
			}
			else if (PR) {
				policytarget[done] = policytarget[done] * learningRate + oldPolicy[done];
				improvement(last, policytarget);
			}
			else {
				improvement(last, policytarget);
			}

			current_time.push_back(clock() - stepTime);

			//TODO: //policy things
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

	VectorXd SARSA()
	{
		VectorXd res = VectorXd::Zero(actionsNum);
		state st = env->actions[buf].to;
		res[buf] = actionValues.run(st.norm())[buf];

		return res;
	}
	

};