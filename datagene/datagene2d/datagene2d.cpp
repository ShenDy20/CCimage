// datagene.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include<fstream>
#include <sstream>
#include<math.h>
using namespace std;
struct eve
{
    int
        eventID;
    double
        energy,
        globalPosX,
        globalPosY,
        globalPosZ;

    string layerName;



};
eve table[500000];
struct com {
    double x1, y1, z1, x2, y2, z2, e1, e2;



};
com tab[30000];
int main()
{
    string inputfile;
    string outputfile;
    string suffix = ".txt";
    string prefix = "./processdata/C";
    string prefix1 = "./rawdata/";


    for (int NN = 0; NN < 50; NN++) {
        inputfile = prefix1 + to_string(NN) + suffix;


        outputfile = prefix + to_string(NN) + suffix;
        ifstream file(inputfile);
        string str_line;
        string rubb;
        getline(file, rubb);
        int l = 0;
        while (getline(file, str_line))
        {
            stringstream ss(str_line);
            string str_tmp;
            int i = 0;
            int id;
            double  ene, x, y, z;

            while (getline(ss, str_tmp, ','))
            {

                switch (i)
                {
                case 1:
                    id = stoi(str_tmp);
                    table[l].eventID = id;
                case 3:
                    ene = stod(str_tmp);
                    table[l].energy = ene;
                case 6:
                    x = stod(str_tmp);
                    table[l].globalPosX = x;
                case 7:
                    y = stod(str_tmp);
                    table[l].globalPosY = y;
                case 8:
                    z = stod(str_tmp);
                    table[l].globalPosZ = z;
                case 20:
                    table[l].layerName = str_tmp;
                }
                i++;
            }
            l++;
        }
        int CC = 0;
        for (int i = 0; i < 500000; i++)
        {
            if (table[i].eventID == 0 && i != 0)
                break;
            if (table[i].eventID == table[i + 1].eventID)
            {
                int j = 2;
                while (table[i].eventID == table[i + j].eventID)
                {
                    j++;
                }
                if (j > 2)
                {
                    i += j - 1;
                    continue;
                }
                else
                {
                    if (table[i].layerName == "scatterer_phys" && table[i + 1].layerName == "absorber_phys")
                    {
                        double ang = 1 - 0.511 * (1 / table[i + 1].energy - 1 / (table[i + 1].energy + table[i].energy));
                        if (abs(ang)<=1)
                        {
                            tab[CC].e1 = table[i].energy;
                            tab[CC].x1 = table[i].globalPosX;
                            tab[CC].y1 = table[i].globalPosY;
                            tab[CC].z1 = table[i].globalPosZ;
                            tab[CC].e2 = table[i + 1].energy;
                            tab[CC].x2 = table[i + 1].globalPosX;
                            tab[CC].y2 = table[i + 1].globalPosY;
                            tab[CC].z2 = table[i + 1].globalPosZ;
                            CC++;
                        }
                    }
                    i += 1;
                }
            }
        }
        FILE* fp;
        fp = fopen(outputfile.c_str(), "w");
        for (int i = 0; i < 30000; i++)
        {
            if (tab[i].e1 == 0) break;
            fprintf(fp, "%f %f %f %f %f %f %f %f\n", tab[i].x1, tab[i].y1, tab[i].z1, tab[i].e1, tab[i].x2, tab[i].y2, tab[i].z2, tab[i].e2);

        }
        memset(table, 0, sizeof(table));
        memset(tab, 0, sizeof(tab));
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
