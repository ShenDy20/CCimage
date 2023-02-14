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
vector<int> trainrdm;
vector<int> testrdm;
vector<int>shuffle1;
vector<int>shuffle2;
vector<vector<int>> label;



int main()
{
	for (int i = 0; i < 350; i++)
	{
		trainrdm.push_back(i);
	}
	for (int i = 0; i < 350; i++)
	{
		testrdm.push_back(i);
	}


	for (int j = 0; j < 50; j++)
	{
		string labelfile = "./label/" + to_string(j) + ".txt";
		ifstream in2(labelfile);
		string line2;
		while (getline(in2, line2))
		{
			stringstream ss(line2);
			string tmp;
			vector<int> xx;
			while (getline(ss, tmp, ' ')) 
			{
				xx.push_back(int(stod(tmp)));
			}
			label.push_back(xx);
		}




	}


	for (int j = 0; j < 50; j++)
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
					double x = stod(tmp) + 70.4;
					x1.push_back(x * 256 / 140.8);
				}
				else if (i == 1)
				{
					double y = stod(tmp) + 70.4;
					yy1.push_back(y * 256 / 140.8);
				}
				else if (i == 2)
				{
					double z = (stod(tmp) - 50) * 128;
					z1.push_back(z);
				}
				else if (i == 3)
				{
					double e = stod(tmp) * 4302.52;
					e1.push_back(e);
				}
				else if (i == 4)
				{
					double x = stod(tmp) + 70.4;
					x2.push_back(x * 256 / 140.8);
				}
				else if (i == 5)
				{
					double y = stod(tmp) + 70.4;
					y2.push_back(y * 256 / 140.8);
				}
				else if (i == 6)
				{
					double z = (stod(tmp) - 50) * 128;
					z2.push_back(z);
				}
				else if (i == 7)
				{
					double e = stod(tmp) * 4302.52;
					e2.push_back(e);
				};
				i++;
			}
		}



	

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		for (int I = 0; I < 5; I++)
		{
			
			std::shuffle(testrdm.begin(), testrdm.end(), std::default_random_engine(seed));
			vector<int> temp;
			for (int tt = 0; tt < 256; tt++)
			{
				temp.push_back(testrdm[tt]);
			}
			double DD[64][32];
			for (int J = 0; J < 64; J++)
			{
				for (int K = 0; K < 4; K++)
				{
					int base = K * 8;
					int num = J * 4 + K;
					DD[J][base] = x1[temp[num]];
					DD[J][base + 1] = yy1[temp[num]];
					DD[J][base + 2] = z1[temp[num]];
					DD[J][base + 3] = e1[temp[num]];
					DD[J][base + 4] = x2[temp[num]];
					DD[J][base + 5] = y2[temp[num]];
					DD[J][base + 6] = z2[temp[num]];
					DD[J][base + 7] = e2[temp[num]];
				}

			}

			string testfile = "./test/" + to_string(cnt2) + ".txt";
			//string testfile = "./test/" + to_string(shuffle2[cnt2]) + ".txt";
			FILE* fp2;
			fp2 = fopen(testfile.c_str(), "w");
			for (int a = 0; a < 64; a++)
				for (int b = 0; b < 32; b++) {
					if (b != 31)
						fprintf(fp2, "%f ", DD[a][b]);
					else
						fprintf(fp2, "%f\n", DD[a][b]);
				}
			testlabels[cnt2]=j;
			//testlabels[shuffle2[cnt2]] = j;
			cnt2++;
			fclose(fp2);
		}



		for (int I = 0; I < 200; I++)
		{
			std::shuffle(trainrdm.begin(), trainrdm.end(), std::default_random_engine(seed));
			vector<int> temp;
			for (int tt = 0; tt < 256; tt++)
			{
				temp.push_back(trainrdm[tt]);
			}
			double DD[64][32];
			for (int J = 0; J < 64; J++)
			{
				for (int K = 0; K < 4; K++)
				{
					int base = K * 8;
					int num = J * 4 + K;
					DD[J][base] = x1[temp[num]];
					DD[J][base + 1] = yy1[temp[num]];
					DD[J][base + 2] = z1[temp[num]];
					DD[J][base + 3] = e1[temp[num]];
					DD[J][base + 4] = x2[temp[num]];
					DD[J][base + 5] = y2[temp[num]];
					DD[J][base + 6] = z2[temp[num]];
					DD[J][base + 7] = e2[temp[num]];
				}

			}

			string trainfile = "./train/" + to_string(cnt1) + ".txt";
			//string trainfile = "./train/" + to_string(shuffle1[cnt1]) + ".txt";
			FILE* fp1;

			fp1 = fopen(trainfile.c_str(), "w");
			for (int a = 0; a < 64; a++)
				for (int b = 0; b < 32; b++) {
					if (b != 31)
						fprintf(fp1, "%f ", DD[a][b]);
					else
						fprintf(fp1, "%f\n", DD[a][b]);
				}
			trainlabels[cnt1] = j;
			//trainlabels[shuffle1[cnt1]] = j;
			cnt1++;
			fclose(fp1);
		}










	}


	string testl = "./test/test.txt";
	FILE* fp2;
	fp2 = fopen(testl.c_str(), "w");
	for (int i = 0; i < 250; i++)
	{
		for (int j = 0; j < 512; j++)
		{			fprintf(fp2, "%d ", label[testlabels[i]][j]);}
	fprintf(fp2, "\n");
	}
		fclose(fp2);

		/*vector<float> weight;
		string pos_weight = "./pos_weight.txt";
		FILE* fp3;
		fp3 = fopen(pos_weight.c_str(), "w");
		for (int j = 0; j < 512; j++) 
		{int temp = 0;
			for (int i = 0; i < 10000; i++)
			{
				
				if (label[trainlabels[i]][j] > 0)
					temp++;

			}
			//float w = float(temp) / (2500-temp);
			//weight.push_back(w);
		}
		for (int j = 0; j<512; j++)
		{
			fprintf(fp3, "%f ", weight[j]);
		
		}*/
		string trainl = "./train/train.txt";
		FILE* fp1;
		fp1 = fopen(trainl.c_str(), "w");
		for (int i = 0; i < 10000; i++)
		{
			for (int j = 0; j < 512; j++)
			{
				fprintf(fp1, "%d ", label[trainlabels[i]][j]);
			}
			fprintf(fp1, "\n");
		}


		fclose(fp1);


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
