#pragma once
#include"NeuralNetwork.h"
static int nums = 0;

struct state {//two element pair for state allocation and its value
	VectorXd value;//type of state
	VectorXd info;
	int num;
	bool failed = false;
	double pos;
	double vel;
	double acc;
	double angle1;
	double angVel1;
	double angAcc1;

	state(int a) {
		num = nums;
		nums++;
		value = VectorXd(a);
		info = VectorXd(12);
	}

	state(int a, bool f) {
		num = nums;
		nums++;
		value = VectorXd(a);
		failed = true;
		info = VectorXd(12);
	}

	state(VectorXd vals) : value(vals) {
		num = nums;
		nums++;
		info = VectorXd(12);
	}

	state() {
		num = nums;
		nums++;
		value = VectorXd(1);
		value << 0;
		info = VectorXd(12);
	}
	state(const state &oth) : num(oth.num) {
		value = oth.value;
		failed = oth.failed;
		info = oth.info;
	}



};

struct action {
	state* fromP;
	state* toP;
	state from;
	state to;
	double reward;

	action(state *f, state *t, double r) : fromP(f), toP(t), reward(r) {
		from = *fromP;
		to = *toP;
	}

	action(const action &oth) :fromP(oth.fromP), toP(oth.toP), reward(oth.reward), from(oth.from), to(oth.to) {

	}
};

class environment {
protected:
	int size;
	double lastReward;


public:
	state current;
	vector<action> actions;
	environment(int sz) {
		size = sz;
	}

	int getSize() {
		return size;
	}

	environment() {}

	double reward(int act) {

		//	cout << "action " << act << endl;
		action* acti = &actions[act];
		action a = *acti;
		state* curr = &a.to;
		current = *curr;
		//	cout << current.value.transpose() << endl;
		return (*this).getReward(a);
	}

	virtual double getReward(action act) {
		cout << "I'm in base!" << endl;
		lastReward = act.reward;
		return act.reward;
	}

	double reward() {

		return lastReward;
	}

	virtual bool isFinal(state &st) {
		state* s = &st;
		state sta = *s;
		return sta.failed;
	}

	bool won(state st) {
		return 0;
	}

	void makeFinal() {}

};

enum punchDir { push_left, push_right, dont_touch };

class cart_pole : public environment {

	const double cartM;
	const double pole1L;
	const double pole1M;
	/*const double pole2L;
	const double pole2M;*/
	const double punch;
	const double track;
	const double angLim = 50.000; // degrees
	const double gravAcc = 9.81; //m / sq sec.
	double time = 0;
	const double poleFailAng;
	double timestep = 0.05;

public:
	struct pole_state : public state {
		double pos;
		double vel;
		double acc;
		double angle1;
		double angVel1;
		double angAcc1;
		/*double angle2;
		double angVel2;
		double angAcc2;*/
		//bool failed = false;

		pole_state() :state(4), pos(0), vel(0), acc(0), angle1(0), angVel1(0), angAcc1(0)/*, angle2(0), angVel2(0), angAcc2(0)*/ {
			angle1 += noise();//pole should start falling
							  //	angle2 -= noise();//pole should start falling
			value = VectorXd::Zero(4);
			info = VectorXd::Zero(12);
			makeVal();
		}

		void makeVal() {
			if (pos > 0) {
				value(0) = pos;
				info(0) = pos;
			}
			else {
				value(1) = -pos;
				info(1) = -pos;
			}
			if (vel > 0) {
				info(2) = vel;
			}
			else {
				info(3) = -vel;
			}
			if (acc > 0) {
				info(4) = acc;
			}
			else {
				info(5) = -acc;
			}
			if (angle1 > 0) {
				value(2) = angle1;
				info(6) = angle1;
			}
			else {
				value(3) = -angle1;
				info(7) = -angle1;
			}
			if (angVel1 > 0) {
				info(8) = angVel1;
			}
			else {
				info(9) = -angVel1;
			}

			if (angAcc1 > 0) {
				info(10) = angAcc1;
			}
			else {
				info(11) = -angAcc1;
			}
			/*	if (angle2 > 0) {
			value(12) = angle2;
			}
			else {
			value(13) = angle2;
			}
			if (angVel2 > 0) {
			value(14) = angVel2;
			}
			else {
			value(15) = angVel2;
			}

			if (angAcc2 > 0) {
			value(16) = angAcc2;
			}
			else {
			value(17) = angAcc2;*/


		}

		pole_state(bool f) : state(4, true), pos(0), vel(0), angle1(0), angVel1(0), angAcc1(0) /*angle2(0), angVel2(0), angAcc2(0)*/ {
			angle1 += noise();//pole should start falling
							  //angle2 -= noise();
			value = VectorXd::Zero(4);
			info = VectorXd::Zero(12);
			makeVal();
		}

		pole_state(double posIn, double velIn, double accIn, double angleIn1, double angVelIn1, double angAccIn1/*, double angleIn2, double angVelIn2, double angAccIn2*/)
			: state(4), pos(posIn), vel(velIn), acc(accIn), angle1(angleIn1), angVel1(angVelIn1), angAcc1(angAccIn1)/*, angle2(angleIn2), angVel2(angVelIn2), angAcc2(angAccIn2) */ {

			value = VectorXd::Zero(4);
			info = VectorXd::Zero(12);
			makeVal();
			//	cout << value.transpose() << endl;
		}




		pole_state(const pole_state &oth) : state(4), pos(oth.pos), vel(oth.vel), acc(oth.acc), angle1(oth.angle1), angVel1(oth.angVel1), angAcc1(oth.angAcc1)/*, angle2(oth.angle2), angVel2(oth.angVel2), angAcc2(oth.angAcc2)*/ {
			//copy constructor
			//	cout << "copy" << endl;
			value = VectorXd::Zero(4);
			info = VectorXd::Zero(12);
			makeVal();
			failed = oth.failed;
		}

	};


	void init_actions() {
		actions.clear();
		//cout << "init" << current.value.transpose() << endl;
		pole_state pst1 = t_step(&current, push_left);
		pole_state* l = &pst1;
		pole_state pst2 = t_step(&current, push_right);
		pole_state* r = &pst2;
		pole_state pst3 = t_step(&current, dont_touch);
		pole_state* st = &pst3;
		double lr, rr, sr;

		lr = (l->failed) ? -1 : 0;
		rr = (r->failed) ? -1 : 0;
		sr = (st->failed) ? -1 : 0;
		//cout << lr + rr + sr << " rewards " << endl;
		action act = action(&current, l, lr);
		actions.push_back(act);
		act = action(&current, r, rr);
		actions.push_back(act);
		act = action(&current, st, sr);
		actions.push_back(act);
	}

	pole_state t_step(state *in, punchDir dir) {
		time += timestep;
		state* curr = in;
		state currentState = *curr;
		double pos = currentState.info[0] + currentState.info[1];
		double vel = currentState.info[2] + currentState.info[3];
		double acc = currentState.info[4] + currentState.info[5];
		double angle1 = currentState.info[6] + currentState.info[7];
		double angVel1 = currentState.info[8] + currentState.info[9];
		double angAcc1 = currentState.info[10] + currentState.info[11];
		//cout << "state " << pos << " " << vel << " " << angle1 << endl;
		/*double angle2 = currentState.value[6];
		double angVel2 = currentState.value[7];
		double angAcc2 = currentState.value[8];*/
		double cosAn1, sinAn1, temp1/*, cosAn2, sinAn2, temp2*/;
		double punchDone = (dir == push_left) ? (-punch) : ((dir == push_right) ? punch : 0);

		//cout << "inside step " << currentState.value.transpose() << endl;
		cosAn1 = cos(angle1);
		sinAn1 = sin(angle1);
		/*	cosAn2 = cos(angle2);
		sinAn2 = sin(angle2);*/

		temp1 = (punchDone + pole1M*pole1L * angVel1 * angVel1 * sinAn1)
			/ (pole1M +/* pole2M*/ +cartM);

		/*temp2 = (punchDone + pole2M*pole2L * angVel2 * angVel2 * sinAn2)
		/ (pole1M + pole2M + cartM);*/

		angAcc1 = (gravAcc * sinAn1 - cosAn1 * temp1)
			/ (pole1L * ((4.0 / 3.0) - pole1M * cosAn1 * cosAn1
				/ (pole1M + /*pole2M */+cartM)));

		/*angAcc2 = (gravAcc * sinAn2 - cosAn2 * temp2)
		/ (pole2L * ((4.0 / 3.0) - pole2M * cosAn2 * cosAn2
		/ (pole1M + pole2M + cartM)));*/

		acc = (punchDone + pole1M*pole1L*((angVel1*angVel1)*sinAn1 - angAcc1*cosAn1) /*+pole2M*pole2L*((angVel2*angVel2)*sinAn2 - angAcc1*cosAn2)*/) / (pole1M +/* pole2M */+cartM);

		/*** Update the four state variables, using Euler's method. ***/

		pos += timestep * vel;
		vel += timestep * acc;
		angle1 += timestep * angVel1;
		angVel1 += timestep * angAcc1;
		/*angle2 += timestep * angVel2;
		angVel2 += timestep * angAcc2;*/
		pole_state res;
		if ((abs(angle1) > poleFailAng)/* || (abs(angle2) > poleFailAng)*/ || abs(pos)>track) {
			res = pole_state(true);
			// cout << "must stop here!!!" << endl;

			// cout << "must stop here!!!" << endl;
			// cout << "must stop here!!!" << endl;

		}
		else {
			res = pole_state(pos, vel, acc, angle1, angVel1, angAcc1/*, angle2, angVel2, angAcc2*/);

		}
		//cout << inside timestep" << endl;
		//cout << res.value.transpose() << endl;
		return res;
	}

	pole_state step(pole_state& in, punchDir dir) {

	}

	cart_pole(double cM, double p1M, double p2M, double pch, double p1L, double p2L, double tr, double pfr) :
		cartM(cM), pole1M(p1M), /*pole2M(p2M), */punch(pch),
		pole1L(p1L), /*pole2L(p2L),*/ track(tr), poleFailAng(pfr) {
		current = pole_state();
		init_actions();
	}

	cart_pole(cart_pole &oth) :cartM(oth.cartM), pole1M(oth.pole1M), /*pole2M(oth.pole2M), */punch(oth.punch),
		pole1L(oth.pole1L),/* pole2L(oth.pole2L), */track(oth.track), poleFailAng(oth.poleFailAng) {
		current = pole_state();
		init_actions();
	}

	double getReward(action act) override {
		init_actions();
		lastReward = act.reward;
		return act.reward;
	}

	void makeFinal() {
		current = pole_state();
		init_actions();
	}

	bool won(pole_state st) {
		return false;
	}

};

