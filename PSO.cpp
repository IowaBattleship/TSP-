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
#define PopSize 10		//������
#define iterNum 2000	//����������
#define C_1 2			//����ѧϰ����
#define C_2 2			//Ⱥ��ѧϰ����
#define _W 0.8			//�ٶȹ���

double vmax = 1, vmin = -1;
class City
{
public:
	int index;	//�������
	double x, y;//���е�Ķ�ά����

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
	vector<City> city;//��������
	double** distance;//���м�ľ������
	int numCity;

	void init(string filename)
	{
		ReadFile(filename);
		calDistance();
	}
	void ReadFile(string txtfilename)//��ȡ���������ļ��ĺ���
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
			cout << "�ļ�������";
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
		cout << "���б�� " << "����x" << " " << "����y" << endl;
		for (int i = 0; i < numCity; i++)
			city[i].printCity();
		cout << "������� " << endl;
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
Graph graph;//����ȫ�ֶ���ͼ,����Graph���

int randomInt(const int& l, const int& r) // ����[l, r]֮���һ���������
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

//�鵽Particle
class Particle
{
public:
	int* x;//���ӵ�λ��
	int* v;//���ӵ��ٶ�
	double fitness;//��Ӧ�ȣ�ԽСԽ��

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
	double Evaluate()//����������Ӧֵ�ĺ���
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
void AdjustParticle(Particle p, int citycount)//����������Ч�Եĺ�����ʹ�����ӵ�λ�÷���TSP������һ������
{
	int* route = new int [citycount];//1-citycount
	bool* flag = new bool [citycount] ;//��Ӧroute�������Ƿ������ӵ�λ���д��ڵ����飬�ο�����Ϊroute
	int* biaoji = new int [citycount];//������ÿ��Ԫ�ؽ��б�ǵ�����,�ο�����Ϊ����λ��x
	for (int j = 0; j < citycount; j++)
	{
		route[j] = j + 1;
		flag[j] = false;
		biaoji[j] = 0;
	}
	//�����ж�����p��λ�����Ƿ���ĳ��������Ψһ��������Ψһ�����Ӧflag��ֵΪtrue,
	for (int j = 0; j < citycount; j++)
	{
		int num = 0;
		for (int k = 0; k < citycount; k++)
		{
			if (p.x[k] == route[j])
			{
				biaoji[k] = 1;//˵�������е�k��Ԫ�ض�Ӧ�ĳ�����route�У������ǵ�һ�γ��ֲŽ��б��
				num++; break;
			}
		}
		if (num == 0) flag[j] = false;//����·����û��route[j]�������
		else if (num == 1) flag[j] = true;//����·������route[j]�������
	}
	for (int k = 0; k < citycount; k++)
	{
		if (flag[k] == false)//����·����û��route[k]������У���Ҫ��������м��뵽����·����
		{
			int i = 0;
			for (; i < citycount; i++)
			{
				if (biaoji[i] != 1)break;
			}
			p.x[i] = route[k];//���ڱ��Ϊ0�Ľ����滻
			biaoji[i] = 1;
		}
	}	
}
class PSO
{
public:
	Particle* particle;		//��ӵ�е���������
	Particle* pbest, gbest;//ÿ�����ӵ���ʷ�����Լ�ȫ������
	double c1, c2, w;		//ѧϰ����
	int NumIter;			//��������
	int popsize;			//������

	void Init(int Pop_Size, int numIter, double C1, double C2, double W)
	{
		//������ʼ��
		NumIter = numIter;
		c1 = C1;
		c2 = C2;
		w = W;
		popsize = Pop_Size;


		particle = new Particle[popsize];
		pbest = new Particle[popsize];
		for (int i = 0; i < popsize; i++)
		{
			//ÿһ�����ӽ��г�ʼ�������õ�һ�ε���Ӧ�ȴ�������ӵ�������ʷ���
			particle[i].Init();
			pbest[i].Init();
			for (int j = 0; j < graph.numCity; j++)
			{
				pbest[i].x[j] = particle[i].x[j];
				pbest[i].fitness = particle[i].fitness;
			}
		}
		gbest.Init();
		gbest.fitness = INFINITY;//Ϊȫ���������ӳ�ʼ��
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
			std::cout << "����" << i + 1 << "->";
			particle[i].printParticle();
		}
	}
	void PSO_TSP(int Pop_size, int itetime, double C1, double C2, double W, double Vlimitabs)
	{
		//Map_City.ReadFile(filename);
		//Map_City.printGraph();
		vmax = Vlimitabs; vmin = -Vlimitabs;
		Init(Pop_size, itetime, C1, C2, W);
		//std::cout << "��ʼ�������Ⱥ���£�" << endl;
		//printPSO();
		omp_set_num_threads(numThread);
		
			for (int iter = 0; iter < NumIter; iter++)//��������
			{
			#pragma omp parallel
			{
				#pragma omp for
				for (int i = 0; i < popsize; i++)//��������
				{
					//���������ٶȺ�λ��
					for (int j = 0; j < graph.numCity; j++)
					{
						//��ʽ1�����ٶ�
						particle[i].v[j] = (int)(w * particle[i].v[j] + 
							c1 * randomInt(0, 1) * (pbest[i].x[j] - particle[i].x[j]) + 
							c2 * randomInt(0, 1) * (gbest.x[j] - particle[i].x[j]));
						if (particle[i].v[j] > vmax)//�����ٶ�Խ�����
							particle[i].v[j] = (int)vmax;
						else if (particle[i].v[j] < vmin)
							particle[i].v[j] = (int)vmin;

						//��ʽ2����λ��
						particle[i].x[j] += particle[i].v[j];
						if (particle[i].x[j] > graph.numCity)particle[i].x[j] = graph.numCity;//����λ��Խ�����
						else if (particle[i].x[j] < 1) particle[i].x[j] = 1;
					}
					//����λ����Ч�Ե��������������ռ������
					AdjustParticle(particle[i], graph.numCity);
					particle[i].Evaluate();
					//pbest[i].Evaluate();
					//���µ������ӵ���ʷ��ֵ
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
			//����ȫ�����Ž�
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
				std::cout << "��" << iter + 1 << "�ε������������ӣ�";
				
				//printPSO();
				gbest.printParticle();
			}
		}
	}
};
//PSO pso;
int main()
{
	std::cout << "����Ⱥ�Ż��㷨���TSP����������" << endl;

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