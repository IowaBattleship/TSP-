#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <random>
#include <unordered_map>
#include <chrono>
#include <CL/sycl.hpp>
#include <oneapi/dpl/random>
#include "header.h"
using namespace std;
using namespace sycl;
typedef pair<double, double> pdd;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef unsigned long long ull;

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
int randomInt(const int& l, const int& r, const ull& seed, const ull& offset) // 生成[l, r]之间的一个随机整数
{
	oneapi::dpl::minstd_rand engine(seed, offset);
	oneapi::dpl::uniform_int_distribution<int> distr(l, r);
	return distr(engine);
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
vector<double> calFitness(queue& q, const vvi& a, const vector<vector<double> >& dis, const int& n, const int& m)
{
	vector<double> fitness(m);
	vector<double> sum = calSumDis(a, dis, n, m);
    
	buffer buf(a);
	buffer bufF(fitness);
    buffer bufS(sum);
	q.submit([&](handler& h)
    {
        accessor A(buf, h, read_only);
        accessor F(bufF, h, write_only);
        accessor S(bufS, h, read_only);
        
        h.parallel_for(range<1>(m), [=](auto i)
        {
            F[i] = FITNESS_BASE / S[i];
        });
    });
    q.wait();
    
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
void cross(queue& q, vvi& a)
{
	// n个城市 m个个体
	int m = a.size();
	if (!m)
		return;
	int n = a[0].size();
	if (m & 1) // 奇数个体 最后一个不参与交叉
		m--;

	vvi visited(m, vi(n));
    ull seed = rand();

	buffer buf(a);
	buffer bufV(visited);
	q.submit([&](handler& h)
    {
        accessor A(buf, h, read_write);
        accessor V(bufV, h, read_write);
        h.parallel_for(range<1>(m >> 1), [=](auto j)
        {
            int i = j << 1;
            int k = j * 3;
            if (randomInt(1, 100, seed, k) <= PC) // 随机数小于PC则交叉
            {
                // 确定交叉范围
                int l = randomInt(0, n - 1, seed, k + 1);
                int r = randomInt(l, n - 1, seed, k + 2);

                // 交叉
                swap_ranges(A[i].begin() + l, A[i].begin() + r + 1, A[i + 1].begin() + l);

                // 修正冲突
                resloveConflict(V[i], A[i], A[i + 1], n, l, r);
                resloveConflict(V[i + 1], A[i + 1], A[i], n, l, r);
            }
        });
    });
	q.wait();
}
// 变异
void mutate(queue& q, vvi& a)
{
	// n个城市 m个个体
	int m = a.size();
	if (!m)
		return;
	int n = a[0].size();
    
    ull seed = rand();

	buffer buf(a);
	q.submit([&](handler& h)
    {
        accessor A(buf, h, read_write);
        h.parallel_for(range<1>(m), [=](auto i)
        {
            int k = i * 3;
            if (randomInt(1, 100, seed, k) <= PM) // 随机数小于PM则变异
            {
                // 确定变异范围
                int l = randomInt(0, n - 2, seed, k + 1);
                int r = randomInt(l, n - 1, seed, k + 2);

                //变异
                reverse(A[i].begin() + l, A[i].begin() + r + 1);
            }
        });
    });
	q.wait();
}
// 选择
void select(queue& q, vvi& a, const vector<vector<double> >& dis)
{
	// n个城市 m个个体
	int m = a.size();
	if (!m)
		return;
	int n = a[0].size();

	vector<double> fitness = calFitness(q, a, dis, n, m);

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
	vector<vector<int> > tmp(m - 1, vector<int>(n));
    
    ull seed = rand();
    
	buffer buf(a);
	buffer bufP(possibility);
	buffer bufT(tmp);
    
	q.submit([&](handler& h)
    {
        accessor A(buf, h, read_only);
        accessor P(bufP, h, read_write);
        accessor T(bufT, h, write_only);
        
        h.parallel_for(range<1>(m - 1), [=](auto i)
        {
            int rd = randomInt(0, P[m - 1] - 1, seed, i);
            // 找到第一个大于随机数概率的个体
            int l = 0, r = m - 1, pos = 0;
            while (l <= r)
            {
                int mid = (l + r) >> 1;
                if (P[mid] > rd)
                    pos = mid, r = mid - 1;
                else
                    l = mid + 1;
            }
            copy(A[pos].begin(), A[pos].end(), T[i].begin());
        });
    });
    q.wait();

	// 选最大适应度个体
    copy(a[maxPos].begin(), a[maxPos].end(), a[m - 1].begin());

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

	cpu_selector selector;
	//gpu_selector selector;
	queue q(selector);

	// 迭代
	for (int it = 1; it <= MAX_ITERATION_NUM; it++)
	{
		// 打乱个体顺序
		randomShuffle(indiv);

		// 交叉
		cross(q, indiv);

		// 变异
		mutate(q, indiv);

		// 选择
		select(q, indiv, cityDistance);

		// 记录最优个体
		auto& tmp = indiv[INDIV_NUM - 1];
		double dis = calDis(indiv[INDIV_NUM - 1], cityDistance, cityNum);
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
