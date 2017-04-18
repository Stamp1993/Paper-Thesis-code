#include"tests.h"

void learn(bool max, bool fr, string fileadd) {
	int averaging = 1;
	int length = 1000;
	double lr = 0.01;
	double df = 0.999;

	{string file1 = (fileadd + "NoPR1ag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag1(1, cp.current, lr, df,   &cp);
	ag1.epsilon = 0.4;
	ag1.acLearn(0, 0, fr, length, averaging, file1); }

	{string file2 = (fileadd + "10PR10Relearn1ag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(10, 100, fr, length, averaging, file2); }

	/*{string file3 = (fileadd + "10PR100Relearning1ag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, lr, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(10, 100, max, fr, file3); }
	

	{string file4 = (fileadd + "100PR10Relearn1ag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, lr, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(100, 10, max, fr, file4); }
	
	{string file5 = (fileadd + "100PR100Relearn1ag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, lr, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(100, 100, max, fr, file5); }

	{string file6 = (fileadd + "100PRRelearningUntilNotLearned1ag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, 0.01, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(100, -1, max, fr, file6); }

	{string file7 = (fileadd + "30PR30Relearning.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, 0.01, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(30, 30, max, fr, file7); }

	{string file8 = (fileadd + "30PR10RelearnMultag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, 0.01, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(30, 10, max, fr, file8); }

	{string file9 = (fileadd + "30PR100RelearningMultag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, 0.01, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(30, 100, max, fr, file9); }

	{string file10 = (fileadd + "50PR10RelearnMultag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, 0.01, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(50, 10, max, fr, file10); }

	{string file11 = (fileadd + "500PR50RelearnMultag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, 0.01, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(50, 50, max, fr, file11); }

	{string file12 = (fileadd + "30PRRelearningUntilNotLearnedMultag.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag(1, cp.current, 0.01, df,   &cp);
	ag.epsilon = 0.4;
	ag.qLearn(30, -1, max, fr, file12); }*/
}

int main(){
	tests test = tests();
	//test.test1();
	//test.test2();
	//test.test3();
	//test.test4();
	//test.test5();
	//test.test6();
	//test.test7();

	

	//max
	learn(true, true, "FRavg");
	learn(true, false, "avg");
	//min
	/*{string file1 = ("random run.csv");
	cart_pole cp(1, 0.1, 0.3, 10, 0.5, 1.2, 5, 25);
	acAgent ag1(1, cp.current, 0.01, df,   &cp);
	ag1.epsilon = 0.4;
	ag1.qLearn(file1); }
	*/

}
