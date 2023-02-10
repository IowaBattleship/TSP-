#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <random>
#include<vector>
#include<omp.h>
#include<chrono>
using namespace std;

#define dataFile "D:\\test.txt"
#define numThread 16
#define PopSize 10		//粒子数
#define iterNum 2000	//最大迭代次数
#define C_1 2			//自身学习因子
#define C_2 2			//群体学习因子
#define _W 0.8			//速度惯性

double vmax = 1, vmin = -1;
class City
{
public:
	int index;	//城市序号
	double x, y;//城市点的二维坐标

	City() {}
	City(int _i, double _x, double _y)
	{
		index = _i;
		x = _x;
		y = _y;
	}

	void init(int _i, double _x, double _y)
	{
		index = _i;
		x = _x;
		y = _y;
	}

	void printCity()
	{
		std::cout << index << ":" << "(" << x << "," << y << ")" << endl;
	}

};
class Graph
{
public:
	vector<City> city;//城市数组
	double** distance;//城市间的距离矩阵
	int numCity;

	void init(string filename)
	{
		ReadFile(filename);
		calDistance();
	}
	void ReadFile(string txtfilename)//读取城市坐标文件的函数
	{
		ifstream file(txtfilename, ios::in);
		double x = 0, y = 0;
		if (!file.fail())
		{
			int i = 0;
			while (!file.eof() && (file >> x >> y))
			{
				city.push_back(City(++i, x, y));
			}
			numCity = i;
		}
		else {
			cout << "文件不存在";
			perror("!no such file");
		}
		file.close();

		for (int i = 0; i < numCity; ++i)
			cout << city[i].x << ',';
		cout << endl;
		for (int i = 0; i < numCity; ++i)
			cout << city[i].y << ',';
	}
	void calDistance()
	{
		distance = new double* [numCity];
		for (int i = 0; i < numCity; ++i)
		{
			distance[i] = new double[numCity];
			distance[i][i] = 0;
		}
		for (int i = 0; i < numCity; ++i)
		{
			for (int j = 0; j < i; ++j)
			{
				distance[i][j] = distance[j][i] = sqrt(pow((city[i].x - city[j].x), 2) + pow((city[i].y - city[j].y), 2));
			}
		}
	}
	void printGraph()
	{
		cout << "城市编号 " << "坐标x" << " " << "坐标y" << endl;
		for (int i = 0; i < numCity; i++)
			city[i].printCity();
		cout << "距离矩阵： " << endl;
		for (int i = 0; i < numCity; i++)
		{
			for (int j = 0; j < numCity; j++)
			{
				if (j == numCity - 1)
					std::cout << distance[i][j] << endl;
				else
					std::cout << distance[i][j] << "  ";
			}
		}
	}
};
Graph graph;//定义全局对象图,放在Graph类后

int randomInt(const int& l, const int& r) // 生成[l, r]之间的一个随机整数
{
	random_device rd;
	mt19937 e(rd());
	uniform_int_distribution<int> dist(l, r);
	return dist(e);
}
int* doShuffle(int n)
{
	int* res;
	res = new int[n];
	for (int i = 0; i < n; ++i)
		res[i] = i + 1;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(res, res + n, std::default_random_engine(seed));
	return res;
}

//归到Particle
class Particle
{
public:
	int* x;//粒子的位置
	int* v;//粒子的速度
	double fitness;//适应度，越小越好

	void Init()
	{
		x = new int[graph.numCity];
		v = new int[graph.numCity];
		int* M = doShuffle(graph.numCity);
		for (int i = 0; i < graph.numCity; i++)
			x[i] = *(M + i);
		fitness = Evaluate();
		for (int i = 0; i < graph.numCity; i++)
		{
			v[i] = (int)randomInt(vmin, vmax);
		}
	}
	void printParticle()
	{
		for (int i = 0; i < graph.numCity; i++)
		{
			if (i == graph.numCity - 1)
				std::cout << x[i] << ") = " << fitness << endl;
			else if (i == 0)
				std::cout << "f(" << x[i] << ",";
			else
				std::cout << x[i] << ",";
		}
		cout << fitness << endl;
	}
	double Evaluate()//计算粒子适应值的函数
	{
		double fitnessvalue = 0;
		//#pragma omp parallel for reduction(+:fitnessvalue)
		for (int i = 0; i < graph.numCity - 1; i++)
			fitnessvalue += graph.distance[x[i] - 1][x[i + 1] - 1];
		fitnessvalue += graph.distance[x[graph.numCity - 1] - 1][x[0] - 1];
		fitness = fitnessvalue;
		return fitnessvalue;
	}
};
void AdjustParticle(Particle p, int citycount)//调整粒子有效性的函数，使得粒子的位置符合TSP问题解的一个排列
{
	int* route = new int [citycount];//1-citycount
	bool* flag = new bool [citycount] ;//对应route数组中是否在粒子的位置中存在的数组，参考数组为route
	int* biaoji = new int [citycount];//对粒子每个元素进行标记的数组,参考数组为粒子位置x
	for (int j = 0; j < citycount; j++)
	{
		route[j] = j + 1;
		flag[j] = false;
		biaoji[j] = 0;
	}
	//首先判断粒子p的位置中是否有某个城市且唯一，若有且唯一，则对应flag的值为true,
	for (int j = 0; j < citycount; j++)
	{
		int num = 0;
		for (int k = 0; k < citycount; k++)
		{
			if (p.x[k] == route[j])
			{
				biaoji[k] = 1;//说明粒子中的k号元素对应的城市在route中，并且是第一次出现才进行标记
				num++; break;
			}
		}
		if (num == 0) flag[j] = false;//粒子路线中没有route[j]这个城市
		else if (num == 1) flag[j] = true;//粒子路线中有route[j]这个城市
	}
	for (int k = 0; k < citycount; k++)
	{
		if (flag[k] == false)//粒子路线中没有route[k]这个城市，需要将这个城市加入到粒子路线中
		{
			int i = 0;
			for (; i < citycount; i++)
			{
				if (biaoji[i] != 1)break;
			}
			p.x[i] = route[k];//对于标记为0的进行替换
			biaoji[i] = 1;
		}
	}	
}
class PSO
{
public:
	Particle* particle;		//所拥有的所有粒子
	Particle* pbest, gbest;//每个粒子的历史最优以及全局最优
	double c1, c2, w;		//学习参数
	int NumIter;			//迭代次数
	int popsize;			//粒子数

	void Init(int Pop_Size, int numIter, double C1, double C2, double W)
	{
		//参数初始化
		NumIter = numIter;
		c1 = C1;
		c2 = C2;
		w = W;
		popsize = Pop_Size;


		particle = new Particle[popsize];
		pbest = new Particle[popsize];
		for (int i = 0; i < popsize; i++)
		{
			//每一个粒子进行初始化，并用第一次的适应度代表该粒子的自身历史最佳
			particle[i].Init();
			pbest[i].Init();
			for (int j = 0; j < graph.numCity; j++)
			{
				pbest[i].x[j] = particle[i].x[j];
				pbest[i].fitness = particle[i].fitness;
			}
		}
		gbest.Init();
		gbest.fitness = INFINITY;//为全局最优粒子初始化
		for (int i = 0; i < popsize; i++)
		{
			if (pbest[i].fitness < gbest.fitness)
			{
				gbest.fitness = pbest[i].fitness;
				for (int j = 0; j < graph.numCity; j++)
					gbest.x[j] = pbest[i].x[j];
			}
		}
	}
	void printPSO()
	{
		for (int i = 0; i < popsize; i++)
		{
			std::cout << "粒子" << i + 1 << "->";
			particle[i].printParticle();
		}
	}
	void PSO_TSP(int Pop_size, int itetime, double C1, double C2, double W, double Vlimitabs)
	{
		//Map_City.ReadFile(filename);
		//Map_City.printGraph();
		vmax = Vlimitabs; vmin = -Vlimitabs;
		Init(Pop_size, itetime, C1, C2, W);
		//std::cout << "初始化后的种群如下：" << endl;
		//printPSO();
		omp_set_num_threads(numThread);
		
			for (int iter = 0; iter < NumIter; iter++)//迭代次数
			{
			#pragma omp parallel
			{
				#pragma omp for
				for (int i = 0; i < popsize; i++)//粒子数量
				{
					//更新粒子速度和位置
					for (int j = 0; j < graph.numCity; j++)
					{
						//公式1更新速度
						particle[i].v[j] = (int)(w * particle[i].v[j] + 
							c1 * randomInt(0, 1) * (pbest[i].x[j] - particle[i].x[j]) + 
							c2 * randomInt(0, 1) * (gbest.x[j] - particle[i].x[j]));
						if (particle[i].v[j] > vmax)//粒子速度越界调整
							particle[i].v[j] = (int)vmax;
						else if (particle[i].v[j] < vmin)
							particle[i].v[j] = (int)vmin;

						//公式2更新位置
						particle[i].x[j] += particle[i].v[j];
						if (particle[i].x[j] > graph.numCity)particle[i].x[j] = graph.numCity;//粒子位置越界调整
						else if (particle[i].x[j] < 1) particle[i].x[j] = 1;
					}
					//粒子位置有效性调整，必须满足解空间的条件
					AdjustParticle(particle[i], graph.numCity);
					particle[i].Evaluate();
					//pbest[i].Evaluate();
					//更新单个粒子的历史极值
					if (particle[i].fitness < pbest[i].fitness)
					{
						for (int j = 0; j < graph.numCity; j++)
							pbest[i].x[j] = particle[i].x[j];
						pbest[i].fitness = particle[i].fitness;
					}
				}
				//int id = omp_get_thread_num();
				//cout << "thread: " << id << '\t';
				//pbest[id].printParticle();
			}
			//更新全局最优解
			for (int k = 0; k < popsize; k++)
			{
				if (pbest[k].fitness < gbest.fitness)
				{
					for (int j = 0; j < graph.numCity; j++)
						gbest.x[j] = pbest[k].x[j];
					gbest.fitness = pbest[k].fitness;
				}
			}
			if ((iter + 1) % 1000 == 0) 
			//if(iter == iterNum - 1)
			{
				std::cout << "第" << iter + 1 << "次迭代后的最好粒子：";
				
				//printPSO();
				gbest.printParticle();
			}
		}
	}
};
//PSO pso;
int main()
{
	std::cout << "粒子群优化算法求解TSP旅行商问题" << endl;

	graph.init(dataFile);
	//graph.printGraph();
	cout << endl;;
	PSO pso;
	//int start = clock();
	auto start = chrono::steady_clock::now();
	pso.PSO_TSP(PopSize, iterNum, C_1, C_2, _W, 1.5);
	//int end = clock();
	auto end = chrono::steady_clock::now();
	chrono::duration<double, milli> dur = end - start;
	cout << dur.count() << endl;;
	return 0;
}
