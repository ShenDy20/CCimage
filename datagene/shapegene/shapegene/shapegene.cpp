// datagene.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include<fstream>
#include <sstream>
#include<vector>
using namespace std;
int boundaryTotal;
int lowTotal;
struct point {
    double x, y, z;



}; 
vector<point>boundary;
point lowQ[50];
point highQ[100];
bool ep(point a, point b)
{
    if (a.x == b.x && a.y == b.y && a.z == b.z)
        return true;
    else
        return false;
}
int point2num(point a)
{
    return a.x + (a.y - 1) * 8 + (a.z - 1) * 64;
}
bool touched(point a)
{
    for (int i = 0; i < lowTotal; i++)
    {
        if (ep(a, lowQ[i]))
            return true;
    }
    if (a.x < 1 || a.x>8 || a.y < 1 || a.y>8 || a.z < 1 || a.z>8)
        return true;
    return false;
}
bool bounded(point x)
{
    for (int i = 0; i < boundaryTotal; i++)
    {
        if (ep(x, boundary[i]))
            return true;
    }
    return false;
}
int pfind(point a)
{
    for (int i = 0; i < boundaryTotal; i++)
    {
        if (ep(a, boundary[i]))
            return i;
    }
    return -1;
}

int main()
{

    string outputfile1;
    string outputfile2;
    string suffix1 = "LowQ.txt";
    string suffix2 = "HighQ.txt";
            srand(unsigned(time(NULL)));
        for (int num = 0; num <= 500; num++) {
            outputfile1 = to_string(num) + suffix1;
            outputfile2 = to_string(num) + suffix2;
            FILE* fp1;
            FILE* fp2;
            fp1 = fopen(outputfile1.c_str(), "w");
            fp2 = fopen(outputfile2.c_str(), "w");

            point initial;
            initial.x = 3 + rand() % 4;
            initial.y = 3 + rand() % 4;
            initial.z = 3 + rand() % 4;
            boundary.push_back(initial);
            lowQ[0] = initial;
            boundaryTotal = 1;
            lowTotal = 1;
            for (int u = 1; u <= 19; u++)
            {
                int s = rand() % boundaryTotal;
                point cur = boundary[s];
                point p[7];
                p[1].x = cur.x - 1;
                p[1].y = cur.y;
                p[1].z = cur.z;
                p[2].x = cur.x + 1;
                p[2].y = cur.y;
                p[2].z = cur.z;
                p[3].x = cur.x;
                p[3].y = cur.y - 1;
                p[3].z = cur.z;
                p[4].x = cur.x;
                p[4].y = cur.y + 1;
                p[4].z = cur.z;
                p[5].x = cur.x;
                p[5].y = cur.y;
                p[5].z = cur.z - 1;
                p[6].x = cur.x;
                p[6].y = cur.y;
                p[6].z = cur.z + 1;
                point next;
                int ss = rand() % 6;
                switch (ss)
                {
                case 1:
                    next = p[1];
                    break;
                case 2:
                    next = p[2];
                    break;
                case 3:
                    next = p[3];
                    break;
                case 4:
                    next = p[4];
                    break;
                case 5:
                    next = p[5];
                    break;
                case 0:
                    next = p[6];
                    break;

                }
                while (next.x < 1 || next.x>8 || next.y > 8 || next.y < 1 || next.z>8 || next.z < 1 || touched(next))
                {

                    int ss = rand() % 6;
                    switch (ss)
                    {
                    case 1:
                        next = p[1];
                        break;
                    case 2:
                        next = p[2];
                        break;
                    case 3:
                        next = p[3];
                        break;
                    case 4:
                        next = p[4];
                        break;
                    case 5:
                        next = p[5];
                        break;
                    case 0:
                        next = p[6];
                        break;
                    }

                }
                lowQ[lowTotal] = next;
                lowTotal++;
                cur = next;
                p[1].x = cur.x - 1;
                p[1].y = cur.y;
                p[1].z = cur.z;
                p[2].x = cur.x + 1;
                p[2].y = cur.y;
                p[2].z = cur.z;
                p[3].x = cur.x;
                p[3].y = cur.y - 1;
                p[3].z = cur.z;
                p[4].x = cur.x;
                p[4].y = cur.y + 1;
                p[4].z = cur.z;
                p[5].x = cur.x;
                p[5].y = cur.y;
                p[5].z = cur.z - 1;
                p[6].x = cur.x;
                p[6].y = cur.y;
                p[6].z = cur.z + 1;
                if (!(touched(p[1]) && touched(p[2]) && touched(p[3]) && touched(p[4]) && touched(p[5]) && touched(p[6])))
                {
                    boundary.push_back(cur);
                    boundaryTotal++;
                }
                point b[7];
                b[1] = p[1];
                b[2] = p[2];
                b[3] = p[3];
                b[4] = p[4];
                b[5] = p[5];
                b[6] = p[6];
                for (int i = 1; i < 7; i++)
                {
                    cur = b[i];
                    if (bounded(b[i]) == 0)
                        continue;
                    p[1].x = cur.x - 1;
                    p[1].y = cur.y;
                    p[1].z = cur.z;
                    p[2].x = cur.x + 1;
                    p[2].y = cur.y;
                    p[2].z = cur.z;
                    p[3].x = cur.x;
                    p[3].y = cur.y - 1;
                    p[3].z = cur.z;
                    p[4].x = cur.x;
                    p[4].y = cur.y + 1;
                    p[4].z = cur.z;
                    p[5].x = cur.x;
                    p[5].y = cur.y;
                    p[5].z = cur.z - 1;
                    p[6].x = cur.x;
                    p[6].y = cur.y;
                    p[6].z = cur.z + 1;
                    if (touched(p[1]) && touched(p[2]) && touched(p[3]) && touched(p[4]) && touched(p[5]) && touched(p[6]))
                    {
                        int k = pfind(cur);
                        boundary.erase(boundary.begin() + k);
                        boundaryTotal--;

                    }
                }

            }

            for (int j = 0; j < 19; j++)
            {

                fprintf(fp1, "%f %f %f\n", lowQ[j].x, lowQ[j].y, lowQ[j].z);

            }
            for (int j = 0; j < 19; j++)
            {
                int ss = rand() % 8;
                point temp = lowQ[j];
                double xx = temp.x * 2 - 1;
                double yy = temp.y * 2 - 1;
                double zz = temp.z * 2 - 1;
                double xxx = temp.x * 2 - 1;
                double yyy = temp.y * 2 - 1;
                double zzz = temp.z * 2 - 1;
                switch (ss)
                {
                case 1:
                    xx++;
                    break;
                case 2:
                    yy++;
                    break;
                case 3:
                    zz++;
                    break;
                case 4:
                    xx++;
                    yy++;
                    break;
                case 5:
                    xx++;
                    zz++;
                    break;
                case 6:
                    yy++;
                    zz++;
                    break;
                case 7:
                    xx++;
                    yy++;
                    zz++;
                    break;
                case 0:
                    break;
                }
                ss = rand() % 8;
                switch (ss)
                {
                case 1:
                    xxx++;
                    break;
                case 2:
                    yyy++;
                    break;
                case 3:
                    zzz++;
                    break;
                case 4:
                    xxx++;
                    yyy++;
                    break;
                case 5:
                    xxx++;
                    zzz++;
                    break;
                case 6:
                    yyy++;
                    zzz++;
                    break;
                case 7:
                    xxx++;
                    yyy++;
                    zzz++;
                    break;
                case 0:
                    break;
                }
                while ((xx == xxx) && (yy == yyy) && (zz = zzz))
                {
                    xxx = temp.x * 2 - 1;
                    yyy = temp.y * 2 - 1;
                    zzz = temp.z * 2 - 1;
                    ss = rand() % 8;
                    switch (ss)
                    {
                    case 1:
                        xxx++;
                        break;
                    case 2:
                        yyy++;
                        break;
                    case 3:
                        zzz++;
                        break;
                    case 4:
                        xxx++;
                        yyy++;
                        break;
                    case 5:
                        xxx++;
                        zzz++;
                        break;
                    case 6:
                        yyy++;
                        zzz++;
                        break;
                    case 7:
                        xxx++;
                        yyy++;
                        zzz++;
                        break;
                    case 0:
                        break;
                    }
                }

                fprintf(fp2, "%f %f %f\n%f %f %f\n", xx, yy, zz, xxx, yyy, zzz);

            }
            boundary.clear();
            memset(lowQ, 0, sizeof(lowQ));
            fclose(fp1);
                fclose(fp2);
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
