// codegene.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
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
struct point {
	double x, y, z;



};
int point2num(point a)
{
	return int(a.x-1 + (a.y - 1) * 8 + (a.z - 1) * 64);
}

point list[20];
string s1 = R"(/gate/source/addSource gammaP
/gate/source/gammaP/gps/particle	gamma
/gate/source/gammaP/gps/energytype	Mono
/gate/source/gammaP/gps/monoenergy	59.5 keV
/gate/source/gammaP/setActivity	1 MBq

/gate/source/gammaP/gps/type	Volume
/gate/source/gammaP/gps/shape	Sphere
/gate/source/gammaP/gps/radius	1 mm
/gate/source/gammaP/gps/angtype	iso
/gate/source/gammaP/gps/mintheta 0. deg
/gate/source/gammaP/gps/maxtheta 90. deg
/gate/source/gammaP/gps/minphi 0. deg
/gate/source/gammaP/gps/maxphi 360. deg
/gate/source/gammaP/gps/centre	)";
int label[512];


int main()
{
    for (int N=0;N<=500;N++)
    {
        string inputfile ="./data/" +to_string(N) + "LowQ.txt";
       // FILE* fp1;
        //fp1 = fopen(outputfile.c_str(), "w");
		ifstream in(inputfile);
		string line;
		int k = 0;
		while (getline(in, line))
		{
			point source;
			stringstream ss(line);
			string tmp;
			int i = 0;
			while (getline(ss, tmp, ' ')) {
				if (i == 0)
				{
					double x = stod(tmp);
					source.x=x ;
				}
				else if (i == 1)
				{
					double y = stod(tmp);
					source.y = y;
				}
				else if (i == 2)
				{
					double z = stod(tmp);
					source.z = z;
				}
				
				i++;
			}
			list[k] = source;

				k++;
		}

		string trainfile = "./output/myCC_source_pointsource_centered_d2mm_" + to_string(N) + ".mac";
		FILE* fp1;
		string val = "./validate/"+ to_string(N) + ".txt";
		FILE* fp2;
		fp1 = fopen(trainfile.c_str(), "w");
		fp2 = fopen(val.c_str(), "w");
		for (int m = 0; m < 20; m++) {
			/*(/gate/source/addSource gammaP
/gate/source/gammaP/gps/particle	gamma
/gate/source/gammaP/gps/energytype	Mono
/gate/source/gammaP/gps/monoenergy	59.5 keV
/gate/source/gammaP/setActivity	1 MBq

/gate/source/gammaP/gps/type	Volume
/gate/source/gammaP/gps/shape	Sphere
/gate/source/gammaP/gps/radius	1 mm
/gate/source/gammaP/gps/angtype	iso
/gate/source/gammaP/gps/mintheta 0. deg
/gate/source/gammaP/gps/maxtheta 90. deg
/gate/source/gammaP/gps/minphi 0. deg
/gate/source/gammaP/gps/maxphi 360. deg
/gate/source/gammaP/gps/centre	)*/

			fprintf(fp1, "/gate/source/addSource gammaP%d\n",m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/particle	gamma\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/energytype	Mono\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/setActivity	1 MBq\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/monoenergy	59.5 keV", m);
			fprintf(fp1, "\n\n");
			fprintf(fp1, "/gate/source/gammaP%d/gps/type	Volume\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/shape	Sphere\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/radius	1 mm\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/angtype	iso\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/mintheta 0. deg\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/maxtheta 90. deg\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/minphi 0. deg\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/maxphi 360. deg\n", m);
			fprintf(fp1, "/gate/source/gammaP%d/gps/centre	%f %f %f mm\n\n", m,(list[m].x-4.5)*10,(list[m].y - 4.5) * 10, (list[m].z - 4.5) * 10);

			int temp = point2num(list[m]);
			label[temp] = 1;
		}
		for (int q = 0; q < 512; q++)
		{fprintf(fp2, "%d ", label[q]);
		}

		memset(label, 0, sizeof(label));
		fclose(fp2);
		fclose(fp1);
		
		





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
