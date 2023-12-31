#include <iostream>
#include <vector>
#include <string>
#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include<chrono>
#include<omp.h>

using namespace std;

struct Point{
    double x;
    double y;
};

int threads = 1;


void create_grid(int M, int N, double A1, double A2, double B1, double B2, double h1, double h2, std::vector<std::vector<Point>> &grid){

    Point point;

    for(int i=0;i < M+1;i++){
        for(int j=0;j < N+1;j++){
            point.x = A1 + j * h1;
            point.y = A2 + i * h2;
            grid[i][j] = point;
        }
    }
    return;
}

void print_pars(std::vector<std::vector<Point>> matrix, int M, int N){
    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++)
            std::cout<<"("<<matrix[i][j].x<<","<<matrix[i][j].y<<")"<<",";
        std::cout << "\n";
    }
}

void print_matrix(std::vector<std::vector<double>> matrix, int M, int N){
    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++)
            std::cout<<matrix[i][j]<<" ";
        std::cout << "\n";
    }
}

bool if_point_inside_shape(Point point){
    return abs(point.x) + abs(point.y) < 2 && point.y < 1;
}

double monte_carlo(double x_min, double x_max, double y_min, double y_max, int npoints){
    int points_inside = 0;
    int total_points = 0;

    default_random_engine gen;
    uniform_real_distribution<double> dist_x(x_min, x_max);
    uniform_real_distribution<double> dist_y(y_min, y_max);


    Point point;
    for(int i=0; i<npoints;i++){
        point.x = dist_x(gen);
        point.y = dist_y(gen);

        if (if_point_inside_shape(point))
            points_inside++;
        
        total_points++;
    }
    double x_dist = x_max - x_min;
    if (x_min == x_max)
        x_dist = 1.0;
    double y_dist = y_max - y_min;
    if (y_min == y_max)
        y_dist = 1.0;

    return (double(points_inside) / total_points) * (x_dist * y_dist);
}

double get_a(double h2, double eps, double x_min, double x_max, double y_min, double y_max, int n_points){
    double l;
    int points_inside;
    int total_points;
    double res;
    l= monte_carlo(x_min, x_max, y_min, y_max, n_points);
    res = (l/h2) + ((1 - (l/h2))/eps);
    return res;
}

double get_b(double h1, double eps, double x_min, double x_max, double y_min, double y_max, int n_points){
    double l;
    int points_inside;
    int total_points;
    double res;

    l = monte_carlo(x_min, x_max, y_min, y_max, n_points);
    res = (l/h1) + ((1 - (l/h1))/eps);
    return res;
}

double get_F(double h1, double h2, double eps, double x_min, double x_max, double y_min, double y_max, int n_points){
    double S;
    int points_inside;
    int total_points;
    double res;
    double delim;

    S = monte_carlo(x_min, x_max, y_min, y_max, n_points);
    res = S/(h1*h2);
    return res;
}



void Vect_multi_scalar(const std::vector<double> &V, double num, int size, std::vector<double> &res){
    #pragma omp parallel for
    for(int i=0; i < size; i++)
        res[i] = V[i] * num;

    return;
}

void Vect_dif(const std::vector<double> &A, const std::vector<double> &B, int size, int grid_size, std::vector<double> &res){

    #pragma omp parallel for
    for(int i=grid_size+2; i < size - grid_size - 2; i++)
        res[i] = A[i] - B[i];
    return;
}

double Vect_scalar(const std::vector<double> &A, const std::vector<double> &B, int size, double h1, double h2){
    double res = 0.0;

    #pragma omp parallel for reduction(+:res)
    for(int i=0; i<size; i++)
        res += A[i] * B[i];
    return res * h1 * h2;
}

double Vector_Euclid(const std::vector<double> &V, int size, double h1, double h2){
    auto res = Vect_scalar(V, V, size, h1, h2);
    return sqrt(res);
}

double Tau_count(const std::vector<std::vector<double>> &A, const std::vector<double> &w, const std::vector<double> &F, double grid_size, std::vector<double> &r){
    double top = 0.0;
    double delim = 0.0;
    int size = A.size() - grid_size - 2;

    #pragma omp parallel for reduction(+:delim) reduction(+:top)
    for (int i = grid_size + 2; i < size; i++) {
        double tmp = 0.0;
        tmp += A[i][i] * w[i];
        tmp += A[i][i - 1] * w[i - 1];
        tmp += A[i][i + 1] * w[i + 1];
        tmp += A[i][i - grid_size - 1] * w[i - grid_size - 1];
        tmp += A[i][i + grid_size + 1] * w[i + grid_size + 1];
        r[i] = tmp - F[i];
        tmp = 0.0;
        tmp += A[i][i] * r[i];
        tmp += A[i][i - 1] * r[i - 1];
        tmp += A[i][i + 1] * r[i + 1];
        tmp += A[i][i - grid_size - 1] * r[i - grid_size - 1];
        tmp += A[i][i + grid_size + 1] * r[i + grid_size + 1];
        delim += tmp * tmp;
        top += tmp * r[i];
    }
    return top/delim;
}

double Condition_count(const std::vector<double> &w, const std::vector<double> &r, double tau, int size, std::vector<double> &w_plus1, double h1, double h2){
    double condition = 0.0;
    #pragma omp parallel for reduction(+:condition)
    for(int i=0; i<size; i++){
        w_plus1[i] = w[i] - (r[i] * tau);
        condition += (w_plus1[i] - w[i]) * (w_plus1[i] - w[i]);
    }
    return sqrt(condition * h1 * h2);
}

std::tuple<const std::vector<double>, int> MinNev(int size, const std::vector<std::vector<double>> &A, const std::vector<double> &F, double sigma, double h1, double h2, int grid_size){
    std::vector<double> w(size, 0.0);
    std::vector<double> w_plus1(size, 0.0);
    std::vector<double> r(size, 0.0);
    double tau = 0.0;
    double condition = 1.0;
    int counter = 0;

    do{
        tau = Tau_count(A, w, F, grid_size, r);

        condition = Condition_count(w, r, tau, size, w_plus1, h1, h2);

        if ((counter % 10000) == 0)
            cout<<condition<<endl;
        w = w_plus1;
        counter++;

    }while (condition >= sigma);
    return std::make_tuple(w, counter);
    
    
}

void main_cicle(double h_1, double h_2, double eps, int grid_size, const std::vector<std::vector<Point>> &grid, double &sigma){
    int n_points = 1000;
    std::vector<std::vector<double>> A((grid_size + 1) * (grid_size + 1), std::vector<double>((grid_size + 1) * (grid_size + 1), 0.0));

    std::vector<double> F((grid_size + 1) * (grid_size + 1), 0.0);
    std::vector<double> ans((grid_size + 1) * (grid_size + 1), 0.0);

    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double a_ij;
    double b_ij;
    double a_i1_j;
    double b_i_j1;
    int number_of_iterations;
    #pragma omp parallel for private(a_ij) private(b_ij) private(a_i1_j) private(b_i_j1)
    for(int i=1;i < grid_size;i++){
        for(int j=1;j < grid_size;j++){

            a_ij = get_a(h_2, eps, grid[i][j].x - 0.5 * h_1, grid[i][j].x - 0.5 * h_1, grid[i][j].y - 0.5 * h_2, grid[i][j].y + 0.5 * h_2, n_points);
            b_ij = get_b(h_1, eps, grid[i][j].x - 0.5 * h_1, grid[i][j].x + 0.5 * h_1, grid[i][j].y - 0.5 * h_2, grid[i][j].y - 0.5 * h_2, n_points);
            F[i *(grid_size + 1) + j] = get_F(h_1, h_2, eps, grid[i][j].x - 0.5 * h_1, grid[i][j].x + 0.5 * h_1, grid[i][j].y - 0.5 * h_2, grid[i][j].y + 0.5 * h_2, n_points);

            a_i1_j = get_a(h_2, eps, grid[i+1][j].x - 0.5 * h_1, grid[i+1][j].x - 0.5 * h_1, grid[i+1][j].y - 0.5 * h_2, grid[i+1][j].y + 0.5 * h_2, n_points);

            b_i_j1 = get_b(h_1, eps, grid[i][j+1].x - 0.5 * h_1, grid[i][j+1].x + 0.5 * h_1, grid[i][j+1].y - 0.5 * h_2, grid[i][j+1].y - 0.5 * h_2, n_points);

            A[i * (grid_size + 1) + j][i * (grid_size + 1) + j] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
            if (i == 1) {
                A[i * (grid_size + 1) + j][(i + 1) * (grid_size + 1) + j] = -a_i1_j / (h_1 * h_1);
                if (j == 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
                else if (j == grid_size - 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                }
                else {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
            }
            else if (i == grid_size - 1) {
                A[i * (grid_size + 1) + j][(i - 1) * (grid_size + 1) + j] = -a_ij / (h_1 * h_1);;
                if (j == 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
                else if (j == grid_size - 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                }
                else {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
            }
            else {
                A[i * (grid_size + 1) + j][(i - 1) * (grid_size + 1) + j] = -a_ij / (h_1 * h_1);;
                A[i * (grid_size + 1) + j][(i + 1) * (grid_size + 1) + j] = -a_i1_j / (h_1 * h_1);
                if (j == 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
                else if (j == grid_size - 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                }
                else {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
            }
        }
    }

    std::tie(ans, number_of_iterations) = MinNev((grid_size + 1) * (grid_size + 1),A, F, sigma, h_1, h_2, grid_size);
    cout<<"MinNev_iterations -> "<<number_of_iterations<<endl;
    
    std::ofstream out;
    out.open("results.txt");
    if(out.is_open()){
        for(int i=0; i < (grid_size + 1) * (grid_size + 1); i++)
            out<<ans[i]<<",";
    }
    out.close();
}

int main(int argc, char *argv[])
{
    int grid_size;
    grid_size = std::atoi(argv[1]);
    std::vector<std::vector<Point>> grid(grid_size + 1, std::vector<Point>(grid_size+1, Point{0.0, 0.0}));
    double A1 = -4.0;
    double A2 = -4.0;
    double B1 = 4.0;
    double B2 = 4.0;
    double h1 = (B1 - A1)/grid_size;
    double h2 = (B2 - A2)/grid_size;
    double eps = max(h1,h2)*max(h1,h2);
    double sigma = 1.e-6;
    int max_threads = omp_get_max_threads();
    int num_threads = std::stoi(argv[2]);
    omp_set_num_threads(num_threads);

    auto start = std::chrono::system_clock::now();

    create_grid(grid_size, grid_size, A1, A2, B1, B2, h1, h2, grid);
    main_cicle(h1, h2, eps, grid_size, grid, sigma);

    auto end = std::chrono::system_clock::now();
    std::cout <<"Execution_time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << "s" << std::endl;
}