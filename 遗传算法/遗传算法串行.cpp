#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <random>
#include <chrono>
#include <fstream>
#include <CL/sycl.hpp>
using namespace std;
using namespace sycl;
typedef pair<double, double> pdd;
typedef vector<int> vi;
typedef vector<vi> vvi;

const int INDIV_NUM = 10; // 种群个体数目
const int PC = 90; // 交叉概率 /100
const int PM = 5; // 变异概率 /100
const int MAX_ITERATION_NUM = 5000; // 最大迭代次数
const double FITNESS_BASE = 1000; //适应度基数

vector<pdd> city; // 城市坐标
int cityNum;

double square(const double& x) { return x * x; } // 平方
double calDistance(const pdd& a, const pdd& b) // 计算距离
{
	return sqrt(square(a.first - b.first)
		+ square(a.second - b.second));
}
vector<int> generateSeq(const int& n) //生成1~n的一个随机排列
{
	vector<int> tmp(n);
	for (int i = 0; i < n; i++)
		tmp[i] = i;
	random_shuffle(tmp.begin(), tmp.end());
	return tmp;
}

int randomInt(const int& l, const int& r) // 生成[l, r]之间的一个随机整数
{
	random_device rd;
	mt19937 e(rd());
	uniform_int_distribution<int> dist(l, r);
	return dist(e);
}

// Matrix Functions

void randomShuffle(vvi& a) { random_shuffle(a.begin(), a.end()); }
double calDis(const vector<int>& route, const vector<vector<double> >& dis, const int& n)
{
	double sum = 0;
	for (int i = 0; i < n - 1; i++)
		sum += dis[route[i]][route[i + 1]];
	sum += dis[route[n - 1]][route[0]];
	return sum;
}
// 计算各个体路径长度
vector<double> calSumDis(const vvi& a, const vector<vector<double> >& dis, const int& n, const int& m)
{
	vector<double> sum(m);
	for (int i = 0; i < m; i++)
		sum[i] = calDis(a[i], dis, n);
	return sum;
}
// 计算各个体适应度
vector<double> calFitness(const vvi& a, const vector<vector<double> >& dis, const int& n, const int& m)
{
	vector<double> fitness(m);
	vector<double> sum = calSumDis(a, dis, n, m);
	for (int i = 0; i < m; i++)
		fitness[i] = FITNESS_BASE / sum[i];
	return fitness;
}
// 修正交叉后的冲突
void resloveConflict(vector<int>& visited, vector<int>& x, const vector<int>& y, const int& n, const int& l, const int& r)
{
	// 不断遍历寻找有无冲突
	while (1)
	{
		for (int i = 0; i < n; i++)
			visited[i] = -1;

		bool ifConflict = false;
		for (int i = 0; i < n; i++)
		{
			if (~visited[x[i]])
			{
				ifConflict = true;
				// 修正交叉部分
				if (l <= i && i <= r)
					x[i] = y[visited[x[i]]];
				else
					x[visited[x[i]]] = y[i];
			}
			else
				visited[x[i]] = i;
		}
		if (!ifConflict)
			break;
	}
}
// 交叉
void cross(vvi& a)
{
	// n个城市 m个个体
	int m = a.size();
	if (!m)
		return;
	int n = a[0].size();
	if (m & 1) // 奇数个体 最后一个不参与交叉
		m--;

	vvi visited(m, vi(n));

	// 相邻交叉
	for (int i = 0; i < m; i += 2)
		if (randomInt(1, 100) <= PC) // 随机数小于PC则交叉
		{
			// 确定交叉范围
			int l = randomInt(0, n - 1);
			int r = randomInt(l, n - 1);

			// 交叉
			swap_ranges(a[i].begin() + l, a[i].begin() + r + 1, a[i + 1].begin() + l);

			// 修正冲突
			resloveConflict(visited[i], a[i], a[i + 1], n, l, r);
			resloveConflict(visited[i + 1], a[i + 1], a[i], n, l, r);
		}
}
// 变异
void mutate(vvi& a)
{
	// n个城市 m个个体
	int m = a.size();
	if (!m)
		return;
	int n = a[0].size();

	for (int i = 0; i < m; i++)
		if (randomInt(1, 100) <= PM) // 随机数小于PM则变异
		{
			// 确定变异范围
			int l = randomInt(0, n - 2);
			int r = randomInt(l, n - 1);

			//变异
			reverse(a[i].begin() + l, a[i].begin() + r + 1);
		}
}
// 选择
void select(vvi& a, const vector<vector<double> >& dis)
{
	// n个城市 m个个体
	int m = a.size();
	if (!m)
		return;
	int n = a[0].size();

	vector<double> fitness = calFitness(a, dis, n, m);

	// 求适应度总和 并找出最大适应度的个体
	double sumFitness = 0, maxFitness = 0;
	int maxPos = 0;
	for (int i = 0; i < m; i++)
	{
		sumFitness += fitness[i];
		if (maxFitness < fitness[i])
			maxFitness = fitness[i], maxPos = i;
	}

	// 计算选中的概率
	vector<int> possibility(m);
	possibility[0] = fitness[0] / sumFitness * 1e3;
	for (int i = 1; i < m; i++)
		possibility[i] = possibility[i - 1] + fitness[i] / sumFitness * 1e3;

	// 选 m - 1 个个体
	vector<vector<int> > tmp;
	for (int i = 0; i < m - 1; i++)
	{
		int rd = randomInt(0, possibility[m - 1] - 1);
		// 找到第一个大于随机数概率的个体
		int pos = lower_bound(possibility.begin(), possibility.end(), rd) - possibility.begin();
		tmp.push_back(a[pos]);
	}

	// 选最大适应度个体
	tmp.push_back(a[maxPos]);

	copy(tmp.begin(), tmp.end(), a.begin());
}

int main()
{
	auto start = chrono::steady_clock::now();

	srand(time(0));

	ifstream infile("poi.txt");
	// 输入
	infile >> cityNum;
	double x, y;
	for (int i = 0; i < cityNum; i++)
	{
		infile >> x >> y;
		city.push_back(make_pair(x, y));
	}
	infile.close();

	// 初始化城市间距离
	vector<vector<double> > cityDistance(cityNum, vector<double>(cityNum));
	for (int i = 0; i < cityNum; i++)
		for (int j = i + 1; j < cityNum; j++)
			cityDistance[i][j] = cityDistance[j][i] = calDistance(city[i], city[j]);

	// 初始化种群个体
	vvi indiv;
	for (int i = 0; i < INDIV_NUM; i++)
		indiv.push_back(generateSeq(cityNum));

	vector<int> ansCity(cityNum);
	double minDis = 1e9;
	// 迭代
	for (int it = 1; it <= MAX_ITERATION_NUM; it++)
	{
		// 打乱个体顺序
		randomShuffle(indiv);

		// 交叉
		cross(indiv);

		// 变异
		mutate(indiv);

		// 选择
		select(indiv, cityDistance);

		// 记录最优个体
		auto& tmp = indiv[INDIV_NUM - 1];
		double dis = calDis(tmp, cityDistance, cityNum);
		if (dis < minDis)
		{
			copy(tmp.begin(), tmp.end(), ansCity.begin());
			minDis = dis;
		}
	}

	cout << "最优路径：";
	for (const auto& c : ansCity)
		cout << c + 1 << ' ';
	cout << "\n总路程：" << minDis;

	auto end = chrono::steady_clock::now();
	chrono::duration<double, milli> dur = end - start;
	cout << "\n程序运行时间：" << dur.count() << "ms\n";

	return 0;
}
