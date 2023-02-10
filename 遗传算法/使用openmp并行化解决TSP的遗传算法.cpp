%%writefile lab/simple.cpp

//#include<bits/stdc++.h>
#include<iostream>
#include<algorithm>
#include<vector>
#include<random>
#include<ctime>
#include<omp.h>

using namespace std;

//最大节点数
#define maxn 100
//种群大小上限(总的way数)
#define UNIT 3000
//单条路径上的最大代价
#define maxVar 50
const double Pacs = 0.8;//杂交概率
const double Pvan = 0.05;//变异概率
int n, m, T_T = 100;///n为点数/城市数，m为个体数,T_T为迭代次数、


int is_cpu = true;

//路径的数据结构：记录了代价和路径数组
struct way
{
    int q[maxn];
    double val;
}ans;
vector<way> a(UNIT);//存储全部路径
vector<way> ret(UNIT);
int waySize = 0;//存储向量大小
bool operator < (way a, way b)
{
    return a.val > b.val;
}

//生成城市之间的距离矩阵
double dis[maxn][maxn] = { 0 };//距离矩阵：存储城市地图
int make_data(int n, int kind = 2)//生成n个点的无向边权完全图
{
    //memset(dis, 0, sizeof(dis));

    //生成平面n点，满足三角形不等式：随机生成两点坐标
    if (kind == 1)
    {
        int x[maxn], y[maxn];
        for (int i = 1; i <= n; i++)
        {
            x[i] = ((int)rand()) % maxVar;
            y[i] = ((int)rand()) % maxVar;
        }
        for (int i = 1; i <= n; i++)
        {
            for (int j = i; j <= n; j++)
            {
                int sx = x[i] - x[j], sy = y[i] - y[j];
                dis[j][i] = dis[i][j] = sqrt(sx * sx + sy * sy);
            }
        }
    }
    //随机生成完全图：非平面图形
    if (kind == 2)
    {
#pragma omp parallel for
        for (int i = 1; i <= n; i++)
        {
            for (int j = i; j <= n; j++)
            {
                int sx = ((int)rand()) % maxVar, sy = ((int)rand()) % maxVar;
                //dis[j][i]=dis[i][j]=sqrt(sx*sx+sy*sy);
                dis[j][i] = dis[i][j] = sx + sy;
            }
        }
    }
    return n;
}
//不可使用多线程
void outmap()//显示地图
{
    //#pragma omp parallel for
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            printf("%d ", (int)dis[i][j]);
        }
        printf("\n");
    }
}

//显示（最佳）路径：代价->路径
void outway(way x)
{
    printf("%lf : ", x.val);
    for (int i = 0; i < n; i++)
    {
        printf("%d -> ", x.q[i]);
    }
    printf("%d\n", x.q[0]);
}
//取得给定路径上的总代价（适应度）
double Get_val(way x)
{
    double val = 0;
#pragma omp parallel for reduction(+:val)
    for (int i = 0; i < n - 1; i++)
    {
        val += dis[x.q[i]][x.q[i + 1]];//适应度为从i->i+1上的代价之和
    }
    val += dis[x.q[n - 1]][x.q[0]];//加上从n-1->0的代价
    return val;
}
//随机生成x条路径：路径是一个从1开始的顺序数的排列
int creat_way(int x)
{
    cout << a.size() << endl;
    way now;
    for (int i = 0; i < maxn; i++) now.q[i] = 0;
    now.val = 0;

    int m = 0;
    x--;
#pragma omp parallel for
    for (int i = 0; i < n; i++) now.q[i] = i + 1;//第一条路径是顺序的
    now.val = Get_val(now);//计算适应度
    a[waySize++] = now;//存入种群
    ans = now;

    //outway(ans);
    //#pragma omp parallel
    for (int i = x; i > 0; i--)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++) { now.q[i] = a[m].q[i]; }//now总是基于上一条确定的路径生成
        m++;//"上一条"路径的计数变更
        random_shuffle(now.q, now.q + n);//随机重新排列now.q[]中的元素，即随机组成新路径
        now.val = Get_val(now);//计算适应度
        a[waySize++] = now;//存入种群//存入种群
        if (ans < now) ans = now;//ans中记录的是最优解
    }
    cout << "初始路径组生成完成......\n";
    return 1000;//种群大小
}

//个体变异：交换一条路径两个位置的节点
void variation(way& x)
{
    int p1 = rand() % n, p2 = rand() % n;
    int temp = x.q[p1]; x.q[p1] = x.q[p2]; x.q[p2] = temp;
    //将变异简化为交换自身的两个结点的位置

    //重新计算适应度
    double val = 0;
#pragma omp parallel for reduction(+:val)
    for (int i = 0; i < n - 1; i++)
    {
        val += dis[x.q[i]][x.q[i + 1]];//适应度为从i->i+1上的代价之和
    }
    val += dis[x.q[n - 1]][x.q[0]];//加上从n-1->0的代价
    x.val = val;
}

//交叉重组（自动修复）
way across(way a, way b)
{
    //思路：从a中选择一段，再将未选中的点按照b中的顺序加入新路径
    way ret;//交叉后生成的新路径
    bool included[maxn] = { false };//当前所有节点都不在新路径中
    int p1 = rand() % n, p2 = rand() % n;//随机指定起点和终点
    //cout << p1 << '\t' << p2 << endl;
    int start_pos = min(p1, p2), end_pos = max(p1, p2), cnt = 0;
    //将路径a中指定的一段路径直接放入ret中
    //#pragma omp parallel for
    for (int i = start_pos; i <= end_pos; i++)
    {
        ret.q[cnt++] = a.q[i];
        included[a.q[i]] = true;
    }
    //将所有没有被放入ret中的路径按照在b中出现的先后顺序存入ret中
    //#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        if (!included[b.q[i]])
        {
            ret.q[cnt++] = b.q[i];
        }
    }

    double val = 0;
#pragma omp parallel for reduction(+:val)
    for (int i = 0; i < n - 1; i++)
    {
        val += dis[ret.q[i]][ret.q[i + 1]];//适应度为从i->i+1上的代价之和
    }
    val += dis[ret.q[n - 1]][ret.q[0]];//加上从n-1->0的代价
    ret.val = val;

    return ret;
}

void work_Ga()//一次迭代
{
    double s[UNIT] = { 0 };
    int populationSize = waySize;
    int j = populationSize;
    for (int i = 0; i < populationSize; i++) { s[i + 1] = s[i] + a[i].val; }//s[i]中存放前i条路径的代价总和，如s[1]=a[0],s[2]=a[0]+a[1];

    //杂交生成子代：每两个进行杂交重组
    //需要的变量：a、s、j、populationSize
    for (int i = 0; i + 1 < populationSize; i += 2)
    {
        if ((rand() % 1000) > Pacs * 1000) continue;//控制交叉互换的发生
        way temp = across(a[i], a[i + 1]);
        if ((rand() % 1000) < Pvan * 1000) variation(temp);//控制变异的发生
        s[j + 1] = s[j] + temp.val;//计算所有路径的代价和
        j++;//路径增加
        a[waySize++] = temp;//将路径存入种群
        if (ans < temp) { ans = temp; }//ans中始终保存最优路径
    }

//#pragma omp target map(from:is_cpu) map(to:ans)
//    {
//        int a[1000]; for (int i = 0; i < 1000; i++) a[i] = i + 1;
//        is_cpu = omp_is_initial_device();
//    }

    double limit = s[waySize - 1];//limit记录了所有路径的代价总和
    double tempWaySize = waySize;
    int aim_num = m;//下一代个数
    waySize = 0;//新种群再次由0开始
    while (aim_num--)
    {//随机选出aim_num个新个体作为存活的下一代，其余的淘汰(注意是使用rand随机选择)
        double pos = limit + 1;
        while (pos > limit)
        {
            pos = 0.0001 * (rand() % 10000) + ((long long int)rand() * rand()) % ((int)limit + 1);
            //pos = (0.0000~0.9999) + (0~(int)limit)
        }//将pos置于0~limit之间
        int l = 0, r = tempWaySize - 1;//l和r分别标记了s[]的有效范围的首末端
        while (l <= r)
        {
            int mid = (l + r) / 2;
            if (s[mid] >= pos) r = mid - 1;
            else l = mid + 1;
            //循环操作，改变l和r的值，使s[r]为最接近pos值的一项
        }
        if (ans < a[r + 1]) { ans = a[r + 1]; }//记录最优解
        ret[waySize++] = a[r + 1];//将当前路径置入下一代
    }
    a = ret;
}

//信息输出
void OutInformation()
{
    printf("交叉概率：%.2lf\n变异概率： %.2lf\n完全图点数：%d\n种群个数： %d\n迭代次数： %d\n",
        Pacs, Pvan, n, m, T_T);
}

int main()
{
    srand(3);

    time_t start, end;
    double timeRun = 0;
    start = clock();

    n = make_data(90); //make_data参数为节点(城市)数目
    outmap(); cout << endl;
    m = creat_way(1000);//种群大小为1000
    OutInformation(); cout << endl;

    //在迭代次数归零前持续迭代并记录所有存活的子代（用于下一次迭代）
    //omp_set_num_threads(4);
//#pragma omp parallel for
    for (T_T = 100; T_T > 0; T_T--) {
        //printf("当前进程编号为%d\n", omp_get_thread_num());
        work_Ga();
    }

    printf("遗传算法得到的结果："); outway(ans);

    end = clock();
    timeRun = double(end - start) / CLOCKS_PER_SEC;
    printf("\n程序运行时间为%f\n", timeRun);

    //std::cout << "Running on " << (is_cpu ? "CPU" : "GPU") << "\n";

    return 0;
}