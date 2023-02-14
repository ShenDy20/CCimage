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

vector<double> x1;
vector<double> yy1;
vector<double> z1;
vector<double> e1;
vector<double> x2;
vector<double> y2;
vector<double> z2;
vector<double> e2;
double E0 = 0.0595;
vector<vector<int>> label;
int shapetotal = 30;
int trytotal = 25;
double pi = 3.141592653;
int step = 2000000;
vector<vector<double>> sysmat;
vector<double> sigma;
int mat[100];
vector<int> origins;
double fCC(double theta)
{
	return 1 / (1 + E0 / ((1 - cos(theta)) * 0.511));

}
double KNpro(double theta)
{
	double f = fCC(theta);
	return 0.5 * f * f * (f + 1 / f - sin(theta) * sin(theta));

}

double uncertainty(double theta,double E1,double E2,double reso)
{
	double delta1 = reso * E1;
	double delta2 = reso * E2;
	double temp = delta1 * delta1 + E1 * E1 * (E2 + E0) * (E2 + E0) * delta2 * delta2 / (E2 * E2 * E2 * E2);
	return 0.511 * sqrt(temp) / (E0 * E0 * sin(theta));

}

double conditionpro(double delta,double dtheta)
{
	return exp(-(dtheta) * (dtheta) / (2 * delta * delta)) / (delta * sqrt(2 * pi));

}

/*double statep(int incidentnum, int voxeltotal)
{
	double P = 1;
	for (int i = 0; i < incidentnum; i++)
	{
		
		for (int j = 0; j < voxeltotal; j++)
		{
			if (mat[j] == 0)
				continue;
			else
			P *= pow(mat[j] / sigma[j], mat[j]);


		}
		P *= sysmat[i][origins[i]];
	}
	return P;
}
*/
double pp(int n, int newloc)
{
	double P = 1;
	int i = origins[n];
	int csi = mat[i];
	int csj = mat[newloc];
	double temp1 =1;
	if(csi<=1)
	{
		temp1 = sysmat[n][newloc] * pow((csj + 1), csj + 1)/sigma[csj];
	}
	else temp1 = sysmat[n][newloc] * pow((csj + 1), csj + 1)* pow((csi - 1), csi - 1)/sigma[csj];

	double temp2 = sysmat[n][i] * pow((csj), csj) * pow((csi), csi) / sigma[csi];
	P = temp1 / temp2;
	return min(1.0, P);

}

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
					double y = stod(tmp);
					yy1.push_back(y);
				}
				else if (i == 2)
				{
					double z = stod(tmp);
					z1.push_back(z);
				}
				else if (i == 3)
				{
					double e = stod(tmp);
					e1.push_back(e);
				}
				else if (i == 4)
				{
					double x = stod(tmp);
					x2.push_back(x);
				}
				else if (i == 5)
				{
					double y = stod(tmp);
					y2.push_back(y);
				}
				else if (i == 6)
				{
					double z = (stod(tmp));
					z2.push_back(z);
				}
				else if (i == 7)
				{
					double e = stod(tmp);
					e2.push_back(e);
				};
				i++;
			}
		}

		int len = x1.size();

		
		for (int iter = 0; iter < len; iter++)
		{
			vector<double> tempvec;
			double xaxis = x1[iter] - x2[iter];
			double yaxis = yy1[iter] - y2[iter];
			double zaxis = z1[iter] - z2[iter];
			vector<int> posloc;
			for (int y = 0; y < 10; y++)
			{
				for (int x = 0; x < 10; x++)
				{
					double CosineCompton = 1 - 0.511 * (1 / e2[iter] - 1 / (e1[iter] + e2[iter]));
					double ximage = (x - 4.5) * 5;
					double yimage = (y - 4.5) * 5;

					double xvarify = ximage - x1[iter];
					double yvarify = yimage - yy1[iter];
					double zvarify = 0 - z1[iter];
					double CosineActual = (xaxis * xvarify + yaxis * yvarify + zaxis * zvarify) / (sqrt(xaxis * xaxis + yaxis * yaxis + zaxis * zaxis) * sqrt(xvarify * xvarify + yvarify * yvarify + zvarify * zvarify));
					double theta = acos(CosineCompton);
					double beta = acos(CosineActual);
					double dtheta = theta - beta;
					double delta = uncertainty(theta, e1[iter], e2[iter], 0.1);
					
					int number = x + y * 10;
					//if (10*delta>dtheta)
					{
						posloc.push_back(number);
					
					}
					tempvec.push_back(KNpro(theta) * conditionpro(delta, dtheta));
				}
			}
			int len2 = posloc.size();
			int random = rand()%len2;
			int pos = posloc[random];
			mat[pos]++;
			origins.push_back(pos);
			sysmat.push_back(tempvec);
		}

		for (int sig=0;sig<100;sig++)
		{
			double tempd=0;
		for(int nn=0;nn<len;nn++)
		{
			tempd += sysmat[nn][sig];
		}
		sigma.push_back(tempd);
		}




		for (int s=0;s<step;s++)
		{
			int random = rand() % 100;
			int incident = rand() % len;
			int decision = (rand() % 10000) / 10000;
			if (decision < pp(incident, random))
			{	
				mat[origins[incident]]--;
				origins[incident] = random;
				mat[random]++;
				
			}
		
		
		}

		vector<int> res(mat,mat+100);

		string valifile = "./vali/" + to_string(j) + ".txt";
		FILE* fp3;
		fp3 = fopen(valifile.c_str(), "w");
		for (int q = 0; q < trytotal; q++) {
			vector<int>::iterator itMax = max_element(res.begin(), res.end());
			int dis = distance(res.begin(), itMax);
			fprintf(fp3, "%f %f\n", double(1 + dis % 10), double(1 + dis / 10));
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
		sysmat.clear();
		sigma.clear();
		origins.clear();
		memset(mat, 0, sizeof(mat));
	}



}












// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

