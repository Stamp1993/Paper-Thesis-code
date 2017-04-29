#pragma once
#include"agent.h"

class acAgent : public qAgent{
	//state currentState;
	//state oldState;
	//neuralNetwork actor; we have actionValues instead

public:
	 MatrixXd PolPRMatr;
     MatrixXd PolPRTarget;

	acAgent(int sz, state curSt, double LR, double DF, environment* e) : qAgent(sz, curSt, LR, DF, softmax, e)
	{ // constr
		currentMoveType = softmax;
	}

	acAgent(const acAgent& oth) : qAgent(oth)
	{ // constr
		currentMoveType = softmax;
		PR = false;
	}
    
    vector<vector<VectorXd> > polPRactivations;
    
    MatrixXd makePolPR(int ps)
	{
		PolPRMatr = classicPR(ps, (*policy.inputs));
		polPRactivations = policy.activationsRun(PolPRMatr);
		PolPRTarget = policy.batchRun(PolPRMatr);
		return PolPRMatr;
	}
    
	
	void policyImprovement(state last, VectorXd target)
	{
		VectorXd run = last.norm();
		
		// cout << "in " << run.transpose() << endl;
		// cout << "target " << target.transpose() << endl;
		if (PR) {
			PolPRMatr.col(0) = run;
			PolPRTarget.col(0) = target;
			policy.batchBackpropagation(PolPRMatr, PolPRTarget, 0.0000001);
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
			PolPRMatr.col(0) = run;
			PolPRTarget.col(0) = target;
			policy.prBackpropagation(PolPRMatr, polPRactivations, PolPRTarget, 0.0000001);
		}
		else {
			policy.backpropagation(run, target);
		}
	}

	void acLearn(int PRSz, int inrelearn, bool fr,  int len, int avgs, string fileout)
	{
        policy.learningRate = policy.learningRate/70;
        PRsize = PRSz;
        cout << "init" << endl;
       
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
			if (PRsize != 0) {
            PR = true;
        }
        else{
            PR = false;
        }
				if(PR){
                    cout << "make pr " << endl;
					makePR(PRsize);
                    cout << "make pol pr" << endl;
                    makePolPR(PRsize);
                    cout << "done" << endl;
				}
                
			for (int i = 0; i < len; i++) {
                 
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
        int oldRew = 0;
		if (steps == 0) {
			startTime = clock();
			current_time.clear();
		}
		stepTime = clock();

		
		VectorXd target(actionsNum); //last state
		VectorXd oldValues = actionValues.run(currentState.norm());
		target = oldValues; // to not relearn values of actions not done

		VectorXd policytarget(actionsNum);
		VectorXd oldPolicy = policy.run(currentState.norm());
		policytarget = oldPolicy; // to not relearn values of actions not done

		state last = currentState; // save state was current on the enering evaluation function

		int oldAct = buf;

		double rew = move(); // policy move and save reward
		int done = buf; // save last action done
        VectorXd newValues = actionValues.run(currentState.norm());
        VectorXd newPolicy = policy.run(currentState.norm());

		state alloc = currentState; // save value of cell we came to
		target[done] = rew+oldRew;
		//policytarget[done] = 0;
        //conditioning for eval
		if (env->isFinal(alloc) || steps == 50000) { // if finished episode

			if (PRsize != 0) {
				if (relearn == -1) {
					cout << "make PR!" << endl;
					makePR(PRsize);
                    makePolPR(PRsize);
				}
				else if (relearn != 0) {
					if (PR && runs % relearn == 0) {
						cout << "make PR!" << endl;
						makePR(PRsize);
                        makePolPR(PRsize);
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
			target[done] = (discountingFactor*target[done] - oldValues[oldAct]);// gamma*reward - q(oldstate, oldAct)
            double td = target[done];

			/*if (runs > 800) {
			cout << "target" << target << endl;
			}*/
			if (FreanRobins) {
                target[done] = target[done] + newValues[done];
				primprovement(last, target);
			}
			else if (PR) {
				target[done] = target[done] * learningRate + newValues[done];
				improvement(last, target);
			}
			else {
                target[done] = target[done] + newValues[done];
				improvement(last, target);
			}

			policytarget[done] = td;
           
			

			if (FreanRobins) {
                policytarget[done] = policytarget[done] + (oldPolicy[done]);
				policyPRimprovement(last, policytarget);//stochastic - only update
			}
			else if (PR) {
				policytarget[done] = policytarget[done] * policy.learningRate + (oldPolicy[done]);//batch bp - targ
				improvement(last, policytarget);
			}
			else {
                policytarget[done] = policytarget[done] + (oldPolicy[done]);
				improvement(last, policytarget);
			}

			
			// system("pause");
			if (steps >= 50000) {
				env->makeFinal();
                oldRew = 1;//success!
			}

			//TODO: //policy things
			return -1;

		}
		else { // not terminal

			steps++;

			VectorXd res = discountingFactor * actionValues.run(currentState.norm());

			target[done] = (target[done] + res[done] - oldValues[oldAct]);
            double td = target[done];
           
																		 // reward is 0 here, but it will be useful in
																		 // other problems
																		 /*if (runs > 800) {
																		 cout << "target" << target << endl;
																		 }*/
																		 /*target = target / 3;*/
			if (FreanRobins) {
                target[done] = target[done] + oldValues[done];
				primprovement(last, target);
			}
			else if (PR) {
				target[done] = target[done] * learningRate + oldValues[done];
				improvement(last, target);
			}
			else {
                target[done] = target[done] + oldValues[done];
				improvement(last, target);
			}

			oldValues = actionValues.run(last.norm());
			
            
			policytarget[done] = td;

			if (FreanRobins) {
                policytarget[done] = policytarget[done]  + (newPolicy[done]);
				policyPRimprovement(last, policytarget);
			}
			else if (PR) {
				policytarget[done] = policytarget[done] * policy.learningRate + (newPolicy[done]);
				improvement(last, policytarget);
			}
			else {
                policytarget[done] = policytarget[done] + (newPolicy[done]);
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