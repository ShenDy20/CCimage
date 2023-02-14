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
vector<vector<int>> label;
int xylabel[50][64];
int yzlabel[50][64];
int main()
{
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
		for (int k = 0; k < 512; k++)
		{
			if (label[j][k] > 0)
			{
				int temp = k % 64;
				xylabel[j][temp]++;

				int temp2 = k / 8;
				yzlabel[j][temp2]++;

			}



		}




	}


	string testl = "./testxy/test.txt";
	FILE* fp2;
	fp2 = fopen(testl.c_str(), "w");
	string test2 = "./testyz/test.txt";
	FILE* fp4;
	fp4 = fopen(test2.c_str(), "w");
	for (int i = 0; i < 50; i++)
	{
		for (int k = 0; k < 5; k++) {
			for (int j = 0; j < 64; j++)
			{
				if (xylabel[i][j] > 0)
					fprintf(fp2, "1 ");
				else
					fprintf(fp2, "0 ");
			}
			fprintf(fp2, "\n");
			for (int j = 0; j < 64; j++)
			{
				if (yzlabel[i][j] > 0)
					fprintf(fp4, "1 ");
				else
					fprintf(fp4, "0 ");
			}
			fprintf(fp4, "\n");

		}
	}
	fclose(fp2);
	fclose(fp4);
	string trainl = "./trainxy/train.txt";
	FILE* fp1;
	fp1 = fopen(trainl.c_str(), "w");
	string train2 = "./trainyz/train.txt";
	FILE* fp3;
	fp3 = fopen(train2.c_str(), "w");
	for (int i = 0; i < 50; i++)
	{
		for (int k = 0; k < 50; k++) {
			for (int j = 0; j < 64; j++)
			{
				if (xylabel[i][j] > 0)
					fprintf(fp1, "1 ");
				else
					fprintf(fp1, "0 ");
			}
			fprintf(fp1, "\n");
			for (int j = 0; j < 64; j++)
			{
				if (yzlabel[i][j] > 0)
					fprintf(fp3, "1 ");
				else
					fprintf(fp3, "0 ");
			}
			fprintf(fp3, "\n");

		}
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
