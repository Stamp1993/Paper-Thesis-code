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
		info = VectorXd(18);
	}

	state(int a, bool f) {
		num = nums;
		nums++;
		value = VectorXd(a);
		failed = true;
		info = VectorXd(18);
	}

	state(VectorXd vals) : value(vals) {
		num = nums;
		nums++;
		info = VectorXd(18);
	}

	state() {
		num = nums;
		nums++;
		value = VectorXd(6);
		value.Zero(6);
		info = VectorXd(18);
	}
	state(const state &oth) : num(oth.num) {
		value = oth.value;
		failed = oth.failed;
		info = oth.info;
	}

	virtual VectorXd norm() {
		VectorXd res = value;
		for (int i = 1; i < 6; i++) {
			res[i] = res[i] / 20;
		}
		for (int i = 6; i < 18; i++) {
			res[i] = res[i] / 60;
		}

		return res;
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
	const double angLim = 50.000; // degrees
	const double gravAcc = 9.81;
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
	const double pole2L;
	const double pole2M;
	const double punch;
	const double track;
	const double angLim = 50.000; // degrees
	const double gravAcc = 9.81; //m / sq sec.
	double time = 0;
	const double poleFailAng;
	double timestep = 0.01;

public:
	struct pole_state : public state {
		double pos;
		double vel;
		double acc;
		double angle1;
		double angVel1;
		double angAcc1;
		double angle2;
		double angVel2;
		double angAcc2;
		//bool failed = false;

		pole_state() :state(18), pos(0), vel(0), acc(0), angle1(0), angVel1(0), angAcc1(0), angle2(0), angVel2(0), angAcc2(0) {
			pos += noise();
			vel += noise();
			acc += noise();
			angle1 += noise() * 5 - 1;//pole should start falling
			angle2 += noise() * 5;//pole should start falling
			angVel1 += noise() * 5;
			angVel2 += noise() * 5;
			angAcc1 += noise() * 5;
			angAcc2 += noise() * 5;
			//value = VectorXd::Zero(6);
			info = VectorXd::Zero(18);

			makeVal();
			value = info;
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
			if (angle2 > 0) {
				value(4) = angle2;
				info(12) = angle2;
			}
			else {
				value(5) = -angle2;
				info(13) = -angle2;
			}
			if (angVel2 > 0) {
				info(14) = angVel2;
			}
			else {
				info(15) = -angVel2;
			}

			if (angAcc2 > 0) {
				info(16) = angAcc2;
			}
			else {
				info(17) = -angAcc2;
			}

		}

		pole_state(bool f) : pole_state() {
			failed = true;
			pos += noise();
			vel += noise();
			acc += noise();
			angle1 += noise() * 5;//pole should start falling
			angle2 += noise() * 5;//pole should start falling
			angVel1 += noise() * 5;
			angVel2 += noise() * 5;
			angAcc1 += noise() * 5;
			angAcc2 += noise() * 5;

			info = VectorXd::Zero(18);
			makeVal();
			value = info;
		}

		pole_state(double posIn, double velIn, double accIn, double angleIn1, double angVelIn1, double angAccIn1, double angleIn2, double angVelIn2, double angAccIn2)
			: state(18), pos(posIn), vel(velIn), acc(accIn), angle1(angleIn1), angVel1(angVelIn1), angAcc1(angAccIn1), angle2(angleIn2), angVel2(angVelIn2), angAcc2(angAccIn2) {


			info = VectorXd::Zero(18);
			makeVal();
			value = info;
			//cout << value.transpose() << endl;
		}




		pole_state(const pole_state &oth) : pole_state(oth.pos, oth.vel, oth.acc, oth.angle1, oth.angVel1, oth.angAcc1, oth.angle2, oth.angVel2, oth.angAcc2) {
			failed = oth.failed;
			//copy constructor
			//	cout << "copy" << endl;

			info = VectorXd::Zero(18);
			makeVal();
			value = info;
			failed = oth.failed;
		}

		VectorXd norm() {
			VectorXd res = value;
			for (int i = 1; i < 6; i++) {
				res[i] = res[i] / 20;
			}
			for (int i = 6; i < 18; i++) {
				res[i] = res[i] / 60;
			}

			return res;
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
		double pi = acos(-1);
		double pos = currentState.info[0] - currentState.info[1];
		double vel = currentState.info[2] - currentState.info[3];
		double acc = currentState.info[4] - currentState.info[5];
		double angle1 = (currentState.info[6] - currentState.info[7])*pi / 180;
		double angVel1 = (currentState.info[8] - currentState.info[9])*pi / 180;
		double angAcc1 = (currentState.info[10] - currentState.info[11])*pi / 180;
		//cout << "state " << pos << " " << vel << " " << angle1 << endl;
		double angle2 = (currentState.info[12]-currentState.info[13])*pi/180;
		double angVel2 = (currentState.info[14]-currentState.info[15])*pi/180;
		double angAcc2 = (currentState.info[16]-currentState.info[17])*pi/180;
		double cosAn1, sinAn1, temp1, cosAn2, sinAn2, temp2;
		double punchDone = (dir == push_left) ? (-punch) : ((dir == push_right) ? punch : 0);
		// cout << "dir" << dir << endl;
		//cout << "inside step " << currentState.value.transpose() << endl;

		cosAn1 = cos(angle1);
		sinAn1 = sin(angle1);
		cosAn2 = cos(angle2);
		sinAn2 = sin(angle2);
		double poleFric = 2 * pow(10, -6);//friction
		double cartFric = 5 * pow(10, -4);
		double tempMul1 = 2 * pole1M*pole1L*angVel1*angVel1*sinAn1 + (3.0 / 4.0)*pole1M*cosAn1*((poleFric*angVel1) / (pole1M*pole1L) + gravAcc*sinAn1);//effective force
		double tempMul2 = 2*pole2M*pole2L*angVel2*angVel2*sinAn2 + (3.0/4.0)*pole2M*cosAn2*((poleFric*angVel2)/(pole2M*pole2L) +gravAcc*sinAn2 );

		double tempDev1 = pole1M*(1 - (3.0 / 4.0)*cosAn1*cosAn1);
		double tempDev2 = pole2M*(1 - (3.0/4.0)*cosAn2*cosAn2);
		int sing = (vel>0) ? 1 : (vel<0) ? -1 : 0;
		acc = (punchDone - cartFric*sing + tempMul1 + tempMul2) / (cartM + tempDev1 + tempDev2);

		angAcc1 = (3.0 / (4.0*pole1L))*(acc*cosAn1 + gravAcc*sinAn1 + (poleFric*angVel1) / (pole1M*pole1L));
		angAcc2 = (3.0/(4.0*pole2L))*(acc*cosAn2 + gravAcc*sinAn2 + (poleFric*angVel2)/(pole2M*pole2L));




					/*** Update the four state variables, using Euler's method. ***/

		vel += timestep * acc;
		pos += timestep * vel;
		angVel1 += timestep * angAcc1;

		angle1 += timestep * angVel1;
		angle1 = angle1 * 180 / pi;
		angVel2 += timestep * angAcc2;
		angle2 += timestep * angVel2;
		angle2 = angle2*180/pi;
		angVel1 = angVel1 * 180 / pi;
		angVel2 = angVel2*180/pi;
					/*	cout << "done" << endl;
					cout << angAcc1 << endl;
					cout << angVel1 << endl;
					cout << angle1 << endl;
					cout << angAcc2 << endl;
					cout << angVel2 << endl;
					cout << angle2 << endl;
					cout << pos << endl;
					cout << vel << endl;
					cout << acc<< endl;*/
		pole_state res;
		if ((abs(angle1) > poleFailAng) || (abs(angle2) > poleFailAng) || abs(pos) > track) {
			res = pole_state(true);
			// cout << "must stop here!!!" << endl;

			// cout << "must stop here!!!" << endl;
			// cout << "must stop here!!!" << endl;

		}
		else {
			res = pole_state(pos, vel, acc, angle1, angVel1, angAcc1 * 180 / pi, angle2, angVel2, angAcc2 * 180 / pi);

		}
		//cout << inside timestep" << endl;
		//cout << res.value.transpose() << endl;
		return res;
	}

	pole_state step(pole_state& in, punchDir dir) {

	}

	cart_pole(double cM, double p1M, double p2M, double pch, double p1L, double p2L, double tr, double pfr) :
		cartM(cM), pole1M(p1M), pole2M(p2M), punch(pch),
		pole1L(p1L), pole2L(p2L), track(tr), poleFailAng(pfr) {
		current = pole_state();
		init_actions();
	}

	cart_pole(cart_pole &oth) :cartM(oth.cartM), pole1M(oth.pole1M), pole2M(oth.pole2M), punch(oth.punch),
		pole1L(oth.pole1L), pole2L(oth.pole2L), track(oth.track), poleFailAng(oth.poleFailAng) {
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

