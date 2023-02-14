
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include<fstream>
#include<string>
#include<algorithm>
#include<iomanip>
#include<cstdio>
#include <stdio.h>
#include<cmath>
#include <stdlib.h>

using namespace std;
double k1 = 25.524840;
double k2 = -1.120820;
double e0 = 2.361244;
int total =10000;
struct entry
{
    int x;
    int y;
    double energy;
    int index;
    double time;
};
entry table[10000];

struct pixel
{
    double a;
    double b;
    double c;
    double t;


};
pixel matrix[65536];
int main()
{
    ifstream file("Muon_bias200V_1800s_r000.t3pa");//选择对应原始数据，转换为X,Y,T,E形式的数据
    ifstream af("Si_a.txt");
    ifstream bf("Si_b.txt");
    ifstream cf("Si_c.txt");                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    ifstream tf("Si_t.txt");

    int quan = 0;
    for (int i = 0; i < 65536; i++)
    {
        af >> matrix[i].a;
        bf >> matrix[i].b;
        cf >> matrix[i].c;
        tf >> matrix[i].t;

    }

    for (int i = 0; i < total; i++)
    {
        if (file.eof()) break;
        int rub;
        int mat;
        long long toa;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        int ftoa;                                     
        int tot;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        file >> rub;
        file >> mat;
        file >> toa;
        file >> tot;
        file >> ftoa;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        file >> rub;
        int x, y;
        x = (mat + 1) % 256;
        if (x != 0)
        {
            table[i].x = x;
            table[i].y = 1 + ((mat + 1) >> 8);
        }
        else
        {
            table[i].x = 256;                                              
            table[i].y = (mat + 1) >> 8;
        }
        table[i].time = (double)(toa * 25 - ftoa * 1.5625);
        double a, b, c, t;
        a = matrix[mat].a;
        b = matrix[mat].b;               
        c = matrix[mat].c;
        t = matrix[mat].t;
        table[i].energy = (tot + a * t - b + sqrt((b - a * t - tot) * (b - a * t - tot) - 4 * a * (tot * t - b * t - c))) / (2 * a);
        table[i].time -= k1 * pow(table[i].energy - e0, k2);
        quan++;
    }
    FILE* fp;
    fp = fopen("atoms0.txt", "w");
    for (int i = 0; i < 1400000; i++)
    {
        if (table[i].time == 0) break;
        fprintf(fp, "%d %d %.4f %f\n", table[i].x, table[i].y, table[i].time, table[i].energy);

    }

}