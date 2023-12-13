#include <iostream>
#include <vector>
#include <string>
#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include<chrono>
#include<omp.h>
#include"mpi.h"

using namespace std;

struct Point{
    double x;
    double y;
};

int threads = 1;


void create_grid(int M, int N, double A1, double A2, double B1, double B2, double h1, double h2, std::vector<std::vector<Point>> &grid){

    Point point;

    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++){
            point.x = A1 + j * h1;
            point.y = A2 + i * h2;
            grid[i][j] = point;
        }
    }
    return;
}

void print_grid(std::vector<std::vector<Point>> matrix, int M, int N){
    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++)
            std::cout<<"("<<matrix[i][j].x<<" "<<matrix[i][j].y<<")"<<" ";
        std::cout << "\n";
    }
    std::cout<<endl;
}

void print_matrix(std::vector<std::vector<double>> matrix, int M, int N){
    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++)
            std::cout<<matrix[i][j]<<" ";
        std::cout << "\n";
    }
    std::cout<<endl;
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

double Tau_count(const std::vector<std::vector<double>> &A, const std::vector<double> &w, const std::vector<double> &F, double grid_size, std::vector<double> &r, int start, int stop, int rank){
    double top = 0.0;
    double delim = 0.0;
    //int size = A.size() - grid_size - 2;
    int shift;
    double buf_top, buf_delim;
    MPI_Status status;

    if (rank==0){
        //#pragma omp parallel for reduction(+:delim) reduction(+:top)
        for (int i = start; i < stop; i++) {
            double tmp = 0.0;
            tmp += A[i][3] * w[i];
            tmp += A[i][2] * w[i - 1];
            tmp += A[i][4] * w[i + 1];
            tmp += A[i][1] * w[i - grid_size - 1];
            tmp += A[i][5] * w[i + grid_size + 1];
            r[i] = tmp - F[i];
            tmp = 0.0;
            tmp += A[i][3] * r[i];
            tmp += A[i][2] * r[i - 1];
            tmp += A[i][4] * r[i + 1];
            tmp += A[i][1] * r[i - grid_size - 1];
            tmp += A[i][5] * r[i + grid_size + 1];
            delim += tmp * tmp;
            top += tmp * r[i];
        }

        MPI_Recv(&buf_delim, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&buf_top, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
        MPI_Send(&delim, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&top, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
        return (top + buf_top)/(delim + buf_delim);
    }
    else{
        //#pragma omp parallel for reduction(+:delim) reduction(+:top)
        for (int i = start; i < stop; i++) {
            double tmp = 0.0;
            tmp += A[i][3] * w[i + A.size() + grid_size + 1];
            tmp += A[i][2] * w[i + A.size() + grid_size + 1 - 1];
            tmp += A[i][4] * w[i + A.size() + grid_size + 1 + 1];
            tmp += A[i][1] * w[i + A.size() + grid_size + 1 - grid_size - 1];
            tmp += A[i][5] * w[i + A.size() + grid_size + 1 + grid_size + 1];
            r[i + A.size() + grid_size + 1] = tmp - F[i];
            tmp = 0.0;
            tmp += A[i][3] * r[i + A.size() + grid_size + 1];
            tmp += A[i][2] * r[i + A.size() + grid_size + 1 - 1];
            tmp += A[i][4] * r[i + A.size() + grid_size + 1 + 1];
            tmp += A[i][1] * r[i + A.size() + grid_size + 1 - grid_size - 1];
            tmp += A[i][5] * r[i + A.size() + grid_size + 1 + grid_size + 1];
            delim += tmp * tmp;
            top += tmp * r[i + A.size() + grid_size + 1];
        }

        MPI_Send(&delim, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&top, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Recv(&buf_delim, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&buf_top, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        return (top + buf_top)/(delim + buf_delim);
    }

}

double Condition_count(const std::vector<double> &w, const std::vector<double> &r, double tau, std::vector<double> &w_plus1, double h1, double h2, int start, int stop, int rank){
    double condition = 0.0;
    double buf_condition;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    //#pragma omp parallel for reduction(+:condition)
    for(int i=start; i<stop; i++){
        w_plus1[i] = w[i] - (r[i] * tau);
        condition += (w_plus1[i] - w[i]) * (w_plus1[i] - w[i]);
    }

    if (rank == 1){
        MPI_Send(&condition, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&buf_condition, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        condition += buf_condition;
        return sqrt(condition * h1 * h2);
    }
    else{
        MPI_Recv(&buf_condition, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&condition, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
        condition += buf_condition;
        return sqrt(condition * h1 * h2);
    }
    
}

std::tuple<const std::vector<double>, int> MinNev(const std::vector<std::vector<double>> &A, const std::vector<double> &F, double sigma, double h1, double h2, int grid_size, int rank){
    std::vector<double> w((grid_size+1) * (grid_size+1), 0.0);
    std::vector<double> w_plus1((grid_size+1) * (grid_size+1), 0.0);
    std::vector<double> r((grid_size+1) * (grid_size+1), 0.0);
    std::vector<double> buf_w_plus1((grid_size+1) * (grid_size+1), 0.0);
    std::vector<double> buf_r((grid_size+1) * (grid_size+1), 0.0);
    double tau = 0.0;
    double condition = 1.0;
    int counter = 0;
    int start_tau, stop_tau, start_cond, stop_cond;
    MPI_Status status;
    
    if(rank==0){
        start_tau = grid_size+2;
        stop_tau = A.size();

        start_cond = 0;
        stop_cond = A.size();

    }
    else{
        start_tau = 0;
        stop_tau = A.size() - grid_size - 2;

        start_cond = A.size() + grid_size + 1;
        stop_cond = (grid_size+1) * (grid_size+1);
    }
    

    do{
        tau = Tau_count(A, w, F, grid_size, r,start_tau, stop_tau, rank);

        condition = Condition_count(w, r, tau, w_plus1, h1, h2, start_cond, stop_cond, rank);

        if ((counter % 10000) == 0)
            cout<<condition<<endl;

        if(rank==0){
            MPI_Send(w_plus1.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Send(r.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
            MPI_Recv(buf_w_plus1.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(buf_r.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
            for(int i=A.size(); i < (grid_size+1) * (grid_size+1); i++){
                r[i] = buf_r[i];
                w_plus1[i] = buf_w_plus1[i];
            }
        
        }
        else{
            MPI_Send(w_plus1.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(r.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(buf_w_plus1.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(buf_r.data(), (grid_size+1) * (grid_size+1), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            for(int i=0; i < A.size() + grid_size+1; i++){
                r[i] = buf_r[i];
                w_plus1[i] = buf_w_plus1[i];
            }
        }

        w = w_plus1;
        counter++;

    }while (condition >= sigma);
    return std::make_tuple(w, counter);
    
    
}

void matrix_init(std::vector<std::vector<double>> &A, std::vector<double> &F,const std::vector<std::vector<Point>> &grid, int grid_size, double h_1, double h_2, double eps, int start, int stop, int rank, int shift){
    double a_ij;
    double b_ij;
    double a_i1_j;
    double b_i_j1;
    int n_points = 1000;
    //#pragma omp parallel for private(a_ij) private(b_ij) private(a_i1_j) private(b_i_j1)
    for(int i=1;i < stop;i++){
        for(int j=1;j < grid_size;j++){
            a_ij = get_a(h_2, eps, grid[i][j].x - 0.5 * h_1, grid[i][j].x - 0.5 * h_1, grid[i][j].y - 0.5 * h_2, grid[i][j].y + 0.5 * h_2, n_points);
            b_ij = get_b(h_1, eps, grid[i][j].x - 0.5 * h_1, grid[i][j].x + 0.5 * h_1, grid[i][j].y - 0.5 * h_2, grid[i][j].y - 0.5 * h_2, n_points);
            F[i *(grid_size + 1) + j-shift] = get_F(h_1, h_2, eps, grid[i][j].x - 0.5 * h_1, grid[i][j].x + 0.5 * h_1, grid[i][j].y - 0.5 * h_2, grid[i][j].y + 0.5 * h_2, n_points);

            a_i1_j = get_a(h_2, eps, grid[i+1][j].x - 0.5 * h_1, grid[i+1][j].x - 0.5 * h_1, grid[i+1][j].y - 0.5 * h_2, grid[i+1][j].y + 0.5 * h_2, n_points);

            b_i_j1 = get_b(h_1, eps, grid[i][j+1].x - 0.5 * h_1, grid[i][j+1].x + 0.5 * h_1, grid[i][j+1].y - 0.5 * h_2, grid[i][j+1].y - 0.5 * h_2, n_points);
            //cout<< a_ij<<" "<<a_i1_j<<" "<<b_ij<<" "<<b_i_j1<<endl;

            A[i * (grid_size + 1) + j-shift][3] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
            if (i == 1) {
                if(rank==0){
                    A[i * (grid_size + 1) + j-shift][5] = -a_i1_j / (h_1 * h_1);
                    if (j == 1) {
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                    else if (j == grid_size - 1) {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                    }
                    else {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                }
                else{
                    A[i * (grid_size + 1) + j-shift][5] = -a_i1_j / (h_1 * h_1);
                    A[i * (grid_size + 1) + j-shift][1] = -a_ij / (h_1 * h_1);
                    if (j == 1) {
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                    else if (j == grid_size - 1) {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                    }
                    else {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                }
            }
            else if (i == stop - 1) {
                if (rank==0){
                    A[i * (grid_size + 1) + j-shift][1] = -a_ij / (h_1 * h_1);
                    A[i * (grid_size + 1) + j-shift][5] = -a_i1_j / (h_1 * h_1);
                    if (j == 1) {
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                    else if (j == grid_size - 1) {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                    }
                    else {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                }
                else{
                    A[i * (grid_size + 1) + j-shift][1] = -a_ij / (h_1 * h_1);;
                    if (j == 1) {
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                    else if (j == grid_size - 1) {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                    }
                    else {
                        A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                        A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                    }
                }
            }
            else {
                A[i * (grid_size + 1) + j-shift][1] = -a_ij / (h_1 * h_1);;
                A[i * (grid_size + 1) + j-shift][5] = -a_i1_j / (h_1 * h_1);
                if (j == 1) {
                    A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                }
                else if (j == grid_size - 1) {
                    A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                }
                else {
                    A[i * (grid_size + 1) + j-shift][2] = -b_ij / (h_2 * h_2);
                    A[i * (grid_size + 1) + j-shift][4] = -b_i_j1 / (h_2 * h_2);
                }
            }
        }
    }
}

void main_cicle(double h_1, double h_2, double eps, int grid_size, const std::vector<std::vector<Point>> &grid, double &sigma, int rank){
    int number_of_iterations;
    int start, stop, shift;

    if (rank==0){
        std::vector<std::vector<double>> A((grid_size/2+1) * (grid_size + 1), std::vector<double>(6, 0.0));

        std::vector<double> F((grid_size/2+1) * (grid_size + 1), 0.0);
        std::vector<double> ans((grid_size/2 + 1) * (grid_size + 1), 0.0);
        std::vector<double> buf_ans((grid_size/2) * (grid_size + 1), 0.0);
        stop = grid_size/2+1;
        shift = 0;

        matrix_init(A, F, grid, grid_size, h_1, h_2, eps, start, stop, rank, shift);
        //print_matrix(A, (grid_size/2+1) * (grid_size+1), 6);

        /*for(int i=0; i< (grid_size/2+1) * (grid_size + 1); i++){
            cout<<F[i]<<",";
        }
        cout<<endl;*/

        std::tie(ans, number_of_iterations) = MinNev(A, F, sigma, h_1, h_2, grid_size, rank);
        cout<<"MinNev_iterations -> "<<number_of_iterations<<endl;
        
        std::ofstream out;
        out.open("results.txt");
        if(out.is_open()){
            for(int i=0; i < (grid_size + 1) * (grid_size + 1); i++)
                out<<ans[i]<<",";
        }
        //out<<"0";
        out.close();
    }
    else{
        std::vector<std::vector<double>> A((grid_size/2) * (grid_size + 1), std::vector<double>(6, 0.0));

        std::vector<double> F((grid_size/2) * (grid_size + 1), 0.0);
        std::vector<double> ans((grid_size/2) * (grid_size + 1), 0.0);
        std::vector<double> buf_ans((grid_size/2) * (grid_size + 1), 0.0);

        stop=grid_size/2;
        shift=grid_size+1;

        matrix_init(A, F, grid, grid_size, h_1, h_2, eps, start, stop, rank,shift);

        /*for(int i=0; i< (grid_size/2) * (grid_size + 1); i++){
            cout<<F[i]<<",";
        }
        cout<<endl;*/
        //print_matrix(A, (grid_size/2) * (grid_size+1), 6);

        std::tie(ans, number_of_iterations) = MinNev(A, F, sigma, h_1, h_2, grid_size, rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    //matrix_init(A, F, grid, grid_size, h_1, h_2, eps, start, stop, shift);
    /*if(rank==1){
        for(int i=0; i< (grid_size+1) * (grid_size + 2)/size; i++){
            cout<<F[i]<<",";
        }
        cout<<endl;}*/

    /*if(rank==0)
        print_matrix(A, (grid_size+1) * (grid_size+2)/size, (grid_size+1) * (grid_size+1));*/

    /*std::tie(ans, number_of_iterations) = MinNev((grid_size + 1) * (grid_size + 1)/size,A, F, sigma, h_1, h_2, grid_size);

    if (rank == 0){
        cout<<"MinNev_iterations -> "<<number_of_iterations<<endl;
        MPI_Recv(buf_ans.data(), (grid_size + 1) * (grid_size + 1)/size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
        
        std::ofstream out;
        out.open("results.txt");
        if(out.is_open()){
            for(int i=0; i < (grid_size + 1) * (grid_size + 1)/size; i++)
                out<<ans[i]<<",";
            out<<endl;
            for(int i=0; i < ((grid_size + 1) * (grid_size + 1)/size); i++)
                out<<buf_ans[i]<<",";
        }
        //out<<"0";
        out.close();
    }
    else
        MPI_Send(ans.data(), (grid_size + 1) * (grid_size + 2)/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);*/
}

int main(int argc, char *argv[])
{
    int grid_size;
    grid_size = std::atoi(argv[1]);
    double A1 = -2.0;
    double A2 = -2.0;
    double B1 = 2.0;
    double B2 = 1.0;
    double h1 = (B1 - A1)/grid_size;
    double h2 = (B2 - A2)/grid_size;
    double eps = max(h1,h2)*max(h1,h2);
    double sigma = 1.e-6;
    int max_threads = omp_get_max_threads();
    int num_threads = std::stoi(argv[2]);
    omp_set_num_threads(num_threads);
    int rank;

    //auto start = std::chrono::system_clock::now();


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto start = MPI_Wtime();
    if(rank==0){
        std::vector<std::vector<Point>> grid((grid_size/2) + 2, std::vector<Point>(grid_size+1, Point{0.0, 0.0}));
        create_grid((grid_size/2)+2, grid_size+1, A1, A2, B1, B2, h1, h2, grid);
        main_cicle(h1, h2, eps, grid_size, grid, sigma, rank);

        //print_grid(grid, (grid_size/2)+2, grid_size+1);
    }
    else{
        std::vector<std::vector<Point>> grid((grid_size/2+1), std::vector<Point>(grid_size+1, Point{0.0, 0.0}));
        create_grid(grid_size/2+1, grid_size+1, A1, A2+h2*(grid_size/2), B1, B2, h1, h2, grid);
        main_cicle(h1, h2, eps, grid_size, grid, sigma, rank);

        //print_grid(grid, grid_size/2+1, grid_size+1);
    }
    //create_grid(grid_size, grid_size, A1, A2, B1, B2, h1, h2, grid);
    //main_cicle(h1, h2, eps, grid_size, grid, sigma);
    auto end = MPI_Wtime();

    MPI_Finalize();
    //auto end = std::chrono::system_clock::now();
    printf("That took %f seconds\n",end-start);
}