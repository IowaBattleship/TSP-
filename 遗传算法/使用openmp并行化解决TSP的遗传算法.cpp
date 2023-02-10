%%writefile lab/simple.cpp

//#include<bits/stdc++.h>
#include<iostream>
#include<algorithm>
#include<vector>
#include<random>
#include<ctime>
#include<omp.h>

using namespace std;

//���ڵ���
#define maxn 100
//��Ⱥ��С����(�ܵ�way��)
#define UNIT 3000
//����·���ϵ�������
#define maxVar 50
const double Pacs = 0.8;//�ӽ�����
const double Pvan = 0.05;//�������
int n, m, T_T = 100;///nΪ����/��������mΪ������,T_TΪ����������


int is_cpu = true;

//·�������ݽṹ����¼�˴��ۺ�·������
struct way
{
    int q[maxn];
    double val;
}ans;
vector<way> a(UNIT);//�洢ȫ��·��
vector<way> ret(UNIT);
int waySize = 0;//�洢������С
bool operator < (way a, way b)
{
    return a.val > b.val;
}

//���ɳ���֮��ľ������
double dis[maxn][maxn] = { 0 };//������󣺴洢���е�ͼ
int make_data(int n, int kind = 2)//����n����������Ȩ��ȫͼ
{
    //memset(dis, 0, sizeof(dis));

    //����ƽ��n�㣬���������β���ʽ�����������������
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
    //���������ȫͼ����ƽ��ͼ��
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
//����ʹ�ö��߳�
void outmap()//��ʾ��ͼ
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

//��ʾ����ѣ�·��������->·��
void outway(way x)
{
    printf("%lf : ", x.val);
    for (int i = 0; i < n; i++)
    {
        printf("%d -> ", x.q[i]);
    }
    printf("%d\n", x.q[0]);
}
//ȡ�ø���·���ϵ��ܴ��ۣ���Ӧ�ȣ�
double Get_val(way x)
{
    double val = 0;
#pragma omp parallel for reduction(+:val)
    for (int i = 0; i < n - 1; i++)
    {
        val += dis[x.q[i]][x.q[i + 1]];//��Ӧ��Ϊ��i->i+1�ϵĴ���֮��
    }
    val += dis[x.q[n - 1]][x.q[0]];//���ϴ�n-1->0�Ĵ���
    return val;
}
//�������x��·����·����һ����1��ʼ��˳����������
int creat_way(int x)
{
    cout << a.size() << endl;
    way now;
    for (int i = 0; i < maxn; i++) now.q[i] = 0;
    now.val = 0;

    int m = 0;
    x--;
#pragma omp parallel for
    for (int i = 0; i < n; i++) now.q[i] = i + 1;//��һ��·����˳���
    now.val = Get_val(now);//������Ӧ��
    a[waySize++] = now;//������Ⱥ
    ans = now;

    //outway(ans);
    //#pragma omp parallel
    for (int i = x; i > 0; i--)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++) { now.q[i] = a[m].q[i]; }//now���ǻ�����һ��ȷ����·������
        m++;//"��һ��"·���ļ������
        random_shuffle(now.q, now.q + n);//�����������now.q[]�е�Ԫ�أ�����������·��
        now.val = Get_val(now);//������Ӧ��
        a[waySize++] = now;//������Ⱥ//������Ⱥ
        if (ans < now) ans = now;//ans�м�¼�������Ž�
    }
    cout << "��ʼ·�����������......\n";
    return 1000;//��Ⱥ��С
}

//������죺����һ��·������λ�õĽڵ�
void variation(way& x)
{
    int p1 = rand() % n, p2 = rand() % n;
    int temp = x.q[p1]; x.q[p1] = x.q[p2]; x.q[p2] = temp;
    //�������Ϊ�����������������λ��

    //���¼�����Ӧ��
    double val = 0;
#pragma omp parallel for reduction(+:val)
    for (int i = 0; i < n - 1; i++)
    {
        val += dis[x.q[i]][x.q[i + 1]];//��Ӧ��Ϊ��i->i+1�ϵĴ���֮��
    }
    val += dis[x.q[n - 1]][x.q[0]];//���ϴ�n-1->0�Ĵ���
    x.val = val;
}

//�������飨�Զ��޸���
way across(way a, way b)
{
    //˼·����a��ѡ��һ�Σ��ٽ�δѡ�еĵ㰴��b�е�˳�������·��
    way ret;//��������ɵ���·��
    bool included[maxn] = { false };//��ǰ���нڵ㶼������·����
    int p1 = rand() % n, p2 = rand() % n;//���ָ�������յ�
    //cout << p1 << '\t' << p2 << endl;
    int start_pos = min(p1, p2), end_pos = max(p1, p2), cnt = 0;
    //��·��a��ָ����һ��·��ֱ�ӷ���ret��
    //#pragma omp parallel for
    for (int i = start_pos; i <= end_pos; i++)
    {
        ret.q[cnt++] = a.q[i];
        included[a.q[i]] = true;
    }
    //������û�б�����ret�е�·��������b�г��ֵ��Ⱥ�˳�����ret��
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
        val += dis[ret.q[i]][ret.q[i + 1]];//��Ӧ��Ϊ��i->i+1�ϵĴ���֮��
    }
    val += dis[ret.q[n - 1]][ret.q[0]];//���ϴ�n-1->0�Ĵ���
    ret.val = val;

    return ret;
}

void work_Ga()//һ�ε���
{
    double s[UNIT] = { 0 };
    int populationSize = waySize;
    int j = populationSize;
    for (int i = 0; i < populationSize; i++) { s[i + 1] = s[i] + a[i].val; }//s[i]�д��ǰi��·���Ĵ����ܺͣ���s[1]=a[0],s[2]=a[0]+a[1];

    //�ӽ������Ӵ���ÿ���������ӽ�����
    //��Ҫ�ı�����a��s��j��populationSize
    for (int i = 0; i + 1 < populationSize; i += 2)
    {
        if ((rand() % 1000) > Pacs * 1000) continue;//���ƽ��滥���ķ���
        way temp = across(a[i], a[i + 1]);
        if ((rand() % 1000) < Pvan * 1000) variation(temp);//���Ʊ���ķ���
        s[j + 1] = s[j] + temp.val;//��������·���Ĵ��ۺ�
        j++;//·������
        a[waySize++] = temp;//��·��������Ⱥ
        if (ans < temp) { ans = temp; }//ans��ʼ�ձ�������·��
    }

//#pragma omp target map(from:is_cpu) map(to:ans)
//    {
//        int a[1000]; for (int i = 0; i < 1000; i++) a[i] = i + 1;
//        is_cpu = omp_is_initial_device();
//    }

    double limit = s[waySize - 1];//limit��¼������·���Ĵ����ܺ�
    double tempWaySize = waySize;
    int aim_num = m;//��һ������
    waySize = 0;//����Ⱥ�ٴ���0��ʼ
    while (aim_num--)
    {//���ѡ��aim_num���¸�����Ϊ������һ�����������̭(ע����ʹ��rand���ѡ��)
        double pos = limit + 1;
        while (pos > limit)
        {
            pos = 0.0001 * (rand() % 10000) + ((long long int)rand() * rand()) % ((int)limit + 1);
            //pos = (0.0000~0.9999) + (0~(int)limit)
        }//��pos����0~limit֮��
        int l = 0, r = tempWaySize - 1;//l��r�ֱ�����s[]����Ч��Χ����ĩ��
        while (l <= r)
        {
            int mid = (l + r) / 2;
            if (s[mid] >= pos) r = mid - 1;
            else l = mid + 1;
            //ѭ���������ı�l��r��ֵ��ʹs[r]Ϊ��ӽ�posֵ��һ��
        }
        if (ans < a[r + 1]) { ans = a[r + 1]; }//��¼���Ž�
        ret[waySize++] = a[r + 1];//����ǰ·��������һ��
    }
    a = ret;
}

//��Ϣ���
void OutInformation()
{
    printf("������ʣ�%.2lf\n������ʣ� %.2lf\n��ȫͼ������%d\n��Ⱥ������ %d\n���������� %d\n",
        Pacs, Pvan, n, m, T_T);
}

int main()
{
    srand(3);

    time_t start, end;
    double timeRun = 0;
    start = clock();

    n = make_data(90); //make_data����Ϊ�ڵ�(����)��Ŀ
    outmap(); cout << endl;
    m = creat_way(1000);//��Ⱥ��СΪ1000
    OutInformation(); cout << endl;

    //�ڵ�����������ǰ������������¼���д����Ӵ���������һ�ε�����
    //omp_set_num_threads(4);
//#pragma omp parallel for
    for (T_T = 100; T_T > 0; T_T--) {
        //printf("��ǰ���̱��Ϊ%d\n", omp_get_thread_num());
        work_Ga();
    }

    printf("�Ŵ��㷨�õ��Ľ����"); outway(ans);

    end = clock();
    timeRun = double(end - start) / CLOCKS_PER_SEC;
    printf("\n��������ʱ��Ϊ%f\n", timeRun);

    //std::cout << "Running on " << (is_cpu ? "CPU" : "GPU") << "\n";

    return 0;
}