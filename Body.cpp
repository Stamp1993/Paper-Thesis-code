#include"tests.h"

void learn(bool max, bool fr, string fileadd) {
    
    int averaging = 1;
	int length = 10000;
	double lr = 0.05;
    double df = 0.99;
    double punch = 5;
    double pole1M = 0.1;
     double pole2M = 0.01;
     double cartM = 1; 
     double failAng = 50;
     double pole1L = 1;
     double pole2L = 0.1;
    
    srand(17);
	{string file1 = (fileadd + "NoPR1ag.csv");
	cart_pole cp(cartM , pole1M,  pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag1(1, cp.current, lr, df,   &cp);
	ag1.epsilon = 0.4;
	ag1.acLearn(0, 0, fr, length, averaging, file1);
    }
    cout << "ag1.acLearn(0, 0, fr, length, averaging, file1);" << endl;
     srand(17);
	{string file2 = (fileadd + "10PR10Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,  &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(10, 10, fr, length, averaging, file2); }
    cout << "ag2.acLearn(10, 10, fr, length, averaging, file2); }" << endl;
     srand(17);
    {string file3 = (fileadd + "10PR100Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(10, 100, fr, length, averaging, file3); }
	cout << "ag2.acLearn(10, 100, fr, length, averaging, file3); }" << endl;
     srand(17);
    {string file4 = (fileadd + "10PR1Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(10, 1, fr, length, averaging, file4); }
    cout << "ag2.acLearn(10, 1, fr, length, averaging, file4); };" << endl;
 srand(17);
	{string file5 = (fileadd + "30PR10Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(30, 10, fr, length, averaging, file5); }
    cout << "ag2.acLearn(30, 10, fr, length, averaging, file5); }" << endl;
     srand(17);
    {string file6 = (fileadd + "30PR30Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(30, 30, fr, length, averaging, file6); }
    cout << "ag2.acLearn(30, 30, fr, length, averaging, file6); }" << endl;
     srand(17);
    {string file7 = (fileadd + "30PR1Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(30, 1, fr, length, averaging, file7); }
    cout << "ag2.acLearn(30, 1, fr, length, averaging, file7); }" << endl;
     srand(17);
    {string file8 = (fileadd + "50PR10Relearn1ag.csv");
	cart_pole cp(cartM, pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(50, 10, fr, length, averaging, file8); }
    cout << "ag2.acLearn(50, 10, fr, length, averaging, file8); }" << endl;
     srand(17);
    {string file9 = (fileadd + "50PR30Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(50, 30, fr, length, averaging, file9); }
    cout << "ag2.acLearn(50, 30, fr, length, averaging, file9); }" << endl;
     srand(17);
    {string file10 = (fileadd + "100PR10Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(100, 10, fr, length, averaging, file10); }
    cout << "ag2.acLearn(100, 10, fr, length, averaging, file10); }" << endl;
     srand(17);
    {string file11 = (fileadd + "100PR100Relearn1ag.csv");
	cart_pole cp(cartM , pole1M, pole2M, punch, pole1L, pole2L, 5, failAng);
	acAgent ag2(1, cp.current, lr, df,   &cp);
	ag2.epsilon = 0.4;
	ag2.acLearn(100, 100, fr, length, averaging, file11); }
    cout << "ag2.acLearn(100, 100, fr, length, averaging, file11); }" << endl;
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
	

}
