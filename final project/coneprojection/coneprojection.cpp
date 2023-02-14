// extract.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define _CRT_SECURE_NO_WARNINGS

// extract.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include<fstream>
#include <sstream>
#include<vector>
#include<algorithm>
#include<random>
#include<chrono>
#include<math.h>
#include<algorithm>
using namespace std;

int cnt1 = 0;
int cnt2 = 0;
int trainlabels[22000];
int testlabels[2200];

vector<double> x1;
vector<double> yy1;
vector<double> z1;
vector<double> e1;
vector<double> x2;
vector<double> y2;
vector<double> z2;
vector<double> e2;
vector<vector<int>> label;
int shapetotal = 30;
int trytotal = 19;

int main()
{
	

	srand(time(NULL));

	for (int j = 0; j < shapetotal; j++)
	{
		string file1 = "./output/C" + to_string(j) + ".txt";
		ifstream in(file1);
		string line;
		while (getline(in, line))
		{
			stringstream ss(line);
			string tmp;
			int i = 0;
			while (getline(ss, tmp, ' ')) {
				if (i == 0)
				{
					double x = stod(tmp);
					x1.push_back(x);
				}
				else if (i == 1)
				{
					double y = stod(tmp) ;
					yy1.push_back(y);
				}
				else if (i == 2)
				{
					double z = stod(tmp);
					z1.push_back(z);
				}
				else if (i == 3)
				{
					double e = stod(tmp) ;
					e1.push_back(e);
				}
				else if (i == 4)
				{
					double x = stod(tmp);
					x2.push_back(x );
				}
				else if (i == 5)
				{
					double y = stod(tmp) ;
					y2.push_back(y);
				}
				else if (i == 6)
				{
					double z = (stod(tmp)) ;
					z2.push_back(z);
				}
				else if (i == 7)
				{
					double e = stod(tmp) ;
					e2.push_back(e);
				};
				i++;
			}
		}

		int len = x1.size();

		for (int iter = 0; iter < len; iter++) 
		{		double xaxis = x1[iter] - x2[iter];
				double yaxis = yy1[iter] - y2[iter];
				double zaxis = z1[iter] - z2[iter];
				int mat[100];
		for (int x=0;x<10;x++)
			for (int y = 0; y < 10; y++) 
			{
				double CosineCompton = 1 - 0.511 * (1 / e2[iter] - 1 / (e1[iter] + e2[iter]));
				double ximage = (x - 4.5) * 5;
				double yimage = (y - 4.5) * 5;

				double xvarify = ximage - x1[iter];
				double yvarify = yimage - yy1[iter];
				double zvarify = 0 - z1[iter];
				double CosineActual = (xaxis * xvarify + yaxis * yvarify + zaxis * zvarify)/(sqrt(xaxis*xaxis+ yaxis * yaxis+ zaxis * zaxis)* sqrt(xvarify * xvarify + yvarify * yvarify + zvarify * zvarify));
				if (abs(acos(CosineActual) - acos(CosineCompton)) < 0.1)
				
					mat[x  + (y) * 10] = 1;
				
				else
					mat[x  + (y) * 10] = 0;
			}
		vector<int> trans;
		for (int pp = 0; pp < 100; pp++)
			trans.push_back(mat[pp]);

		label.push_back(trans);
		}


			string testfile = "./test/" + to_string(j) + ".txt";
			FILE* fp2;
			fp2 = fopen(testfile.c_str(), "w");
			for (int a = 0; a < len; a++)
				for (int b = 0; b < 100; b++) {
					if (b != 99)
						fprintf(fp2, "%f ", float(label[a][b]));
					else
						fprintf(fp2, "%f\n", float(label[a][b]));
				}

			vector<int> res;
			for (int loc = 0; loc < 100; loc++) 
			{
				int num = 0;
			for(int iter=0;iter<len;iter++)
			{
				num += label[iter][loc];
			
			}
			res.push_back(num);

			}


			string valifile = "./vali/" + to_string(j) + ".txt";
			FILE* fp3;
			fp3 = fopen(valifile.c_str(), "w");
			for (int q = 0; q < trytotal; q++) {
				vector<int>::iterator itMax = max_element(res.begin(), res.end());
				int dis = distance(res.begin(), itMax);
				fprintf(fp3, "%f %f\n", double(1+dis%10),double(1+dis/10));//crucial part
				for(int m=0;m<len;m++)
				{
					if (label[m][dis] > 0)
					{
						double random = (rand() % 10) / 10;
						if (random >= 0.7)
						{
							for (int kk = 0; kk < 100; kk++)
							{
								res[kk] -= label[m][kk];

							}

						}
					}
				}//end
				res[dis] = 0;
			}
			res.clear();
			label.clear();
			x1.clear();
			yy1.clear();
			z1.clear();
			e1.clear();
			x2.clear();
			y2.clear();
			z2.clear();
			e2.clear();
		}



		}












// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
