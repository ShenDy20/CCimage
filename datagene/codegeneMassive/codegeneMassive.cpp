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

string s1 = R"(# ==============================
# == VERBOSITY              ==
# ==============================
/control/verbose          0
/run/verbose              1
/event/verbose            0
/tracking/verbose         0
/gate/application/verbose 0
/gate/generator/verbose   0
/gate/random/verbose 0



# ==============================
# ==  VISUALIZATION            ==
# ==============================
#/vis/open OGLIQt
#/vis/drawVolume

#/vis/viewer/set/style wireframe
#/vis/viewer/set/viewpointThetaPhi 60 60
#/vis/viewer/zoom 4

#/vis/viewer/set/auxiliaryEdge True
#/vis/scene/add/trajectories
#/vis/scene/endOfEventAction   accumulate
#/vis/scene/add/axes

# ==============================
# == LOAD DATABASE           ==
# ==============================
/gate/geometry/setMaterialDatabase    GateMaterials.db



# ==============================
# ==   GEOMETRY             ==
# ==============================
/control/execute                     myCC_geometry_TPX3.mac



# ==============================
# ==   PHANTOM            ==
# ==============================
#No Phantom



# ==============================
# ==  PHYSICS             ==
# ==============================
#Physics List
/gate/physics/addPhysicsList emstandard_opt4

#cuts
/control/execute                  cuts.mac



# ==============================
#== ACTOR & ACTOR OUTPUT
# ==============================
/control/execute        myCC_actor_CCMod_adder.mac

/control/execute        myCC_actor_statistics.mac



# ==============================
# == INITIALIZE             ==
# ==============================
/gate/run/initialize



# ==============================
# == SOURCE                 ==
# ==============================
)";
string s2 = R"(/control/execute                   source/myCC_source_pointsource_centered_d2mm_)";


string s3 = R"(# ==============================
# == OUTPUT                ==
# ==============================
#/gate/application/noGlobalOutput
# Only output summary, others output in actor
/gate/output/summary/enable
/gate/output/summary/setFileName 	output/)";


string s4=R"(
/gate/output/summary/addCollection 	Hits
/gate/output/summary/addCollection 	Singles

#/gate/output/tree/enable
#/gate/output/tree/addFileName             output/111.txt
#/gate/output/tree/hits/disable
#/gate/output/tree/addCollection     Singles
#/gate/output/tree/addCollection     Coincidences

/gate/actor/addActor     SimulationStatisticActor    stat 
/gate/actor/stat/save     output/myCCdata_TPX3_stats.txt 
/gate/actor/stat/saveEveryNSeconds 100 

#====================================================
#  R A N D O M   E N G I N E  A N D  S E E D
#====================================================
/gate/random/setEngineName MersenneTwister
/gate/random/setEngineSeed auto



#=====================================================
#   M E A S U R E M E N T   S E T T I N G S   
#=====================================================
/gate/application/setTimeSlice        50  s
/gate/application/setTimeStart        0.00   s
/gate/application/setTimeStop        50  s

/gate/application/startDAQ

exit

)";




int main()
{string finalfile = "./macro/Compton.mac";
		FILE* fp1;
		fp1 = fopen(finalfile.c_str(), "w");
	for (int N = 0; N <= 5; N++)
	{
		

		
		string mainslice = "./run/" + to_string(N) + ".mac";
		FILE* fp2;
		
		fp2 = fopen(mainslice.c_str(), "w");
		

			fprintf(fp2, "%s\n", s1.c_str());
			fprintf(fp2, "%s%d.mac\n", s2.c_str(), N);
			fprintf(fp2, "%s%d.txt\n", s3.c_str(), N);
			fprintf(fp2, "%s", s4.c_str());
		
		
		
			fprintf(fp1,"cd /home/run")
			fprintf(fp1, "Gate /home/run/%d.mac\n",N);

		fclose(fp2);








	}		fclose(fp1);
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
