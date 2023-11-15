#include <iostream>
#include <vector>
#include <string>
#include <bits/stdc++.h>
#include <cmath>
#include <fstream>

using namespace std;

struct Point{
    float x;
    float y;
};


std::vector<std::vector<Point>> create_grid(int M, int N, float A1, float A2, float B1, float B2, float h1, float h2){

    std::vector<std::vector<Point>> grid;
    Point point;
    grid.resize(M+1);
    for(int i=0;i < M+1;++i){
        grid[i].resize(N+1);
    }

    for(int i=0;i < M+1;i++){
        for(int j=0;j < N+1;j++){
            point.x = A1 + i * h1;
            point.y = A2 + j * h2;
            grid[i][j] = point;
        }
    }
    return grid;
}

void print_pars(std::vector<std::vector<Point>> matrix, int M, int N){
    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++)
            std::cout<<"("<<matrix[i][j].x<<","<<matrix[i][j].y<<")"<<",";
        std::cout << "\n";
    }
}

void print_matrix(std::vector<std::vector<float>> matrix, int M, int N){
    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++)
            std::cout<<matrix[i][j]<<" ";
        std::cout << "\n";
    }
}

bool if_point_inside_shape(Point point){
    return abs(point.x) + abs(point.y) < 2 && point.y < 1;
}

std::tuple<float, int, int> monte_carlo(float x_min, float x_max, float y_min, float y_max, int npoints){
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
    float x_dist = x_max - x_min;
    if (x_min == x_max)
        x_dist = 1.0;
    float y_dist = y_max - y_min;
    if (y_min == y_max)
        y_dist = 1.0;

    return std::make_tuple((float(points_inside) / total_points) * (x_dist * y_dist), points_inside, total_points);
}

float get_a(float h2, float eps, float x_min, float x_max, float y_min, float y_max, int n_points){
    float l;
    int points_inside;
    int total_points;
    float res;
    std::tie(l, points_inside, total_points) = monte_carlo(x_min, x_max, y_min, y_max, n_points);
    if (points_inside == total_points)
        res = 1;
    else if (points_inside == 0)
        res = 1/eps;
    else
        res = (l/h2) - ((1 - (l/h2))/eps);
    return res;
}

float get_b(float h1, float eps, float x_min, float x_max, float y_min, float y_max, int n_points){
    float l;
    int points_inside;
    int total_points;
    float res;

    std::tie(l, points_inside, total_points) = monte_carlo(x_min, x_max, y_min, y_max, n_points);
    if (points_inside == total_points)
        res = 1;
    else if (points_inside == 0)
        res = 1/eps;
    else
        res = (l/h1) - ((1 - (l/h1))/eps);
    return res;
}

float get_F(float h1, float h2, float eps, float x_min, float x_max, float y_min, float y_max, int n_points){
    float S;
    int points_inside;
    int total_points;
    float res;
    float delim;

    std::tie(S, points_inside, total_points) = monte_carlo(x_min, x_max, y_min, y_max, n_points);
    if (points_inside == total_points)
        res = 1.0;
    else if (points_inside == 0)
        res = 0.0;
    else{
        res = S/(h1*h2);
    }
    return res;
}


std::vector<float> Vect_multi(std::vector<std::vector<float>> A, std::vector<float> V, int size){
    std::vector<float> res(size, 0.0);

    for(int i=0; i < size; i++){
        for(int j=0; j < size; j++)
            res[i] += A[i][j] * V[j];
    }

    return res;
}

std::vector<float> Vect_multi_w(std::vector<std::vector<float>> A, std::vector<float> V, int size, int N){
    std::vector<float> res(size, 0.0);

    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            res[i*N+j] += A[i*N+j][i*N+j] * V[i * N + j];
            if(i+1 != N)
                res[i*N+j] += A[i*N+j][(i+1)*N+j] * V[(i+1) * N + j];
            if(j+1 != N)
                res[i*N+j] += A[i*N+j][j+1+i*N] * V[i * N + (j+1)];
            if(i-1 > 0)
                res[i*N+j] += A[i*N+j][(i-1)*N+j] * V[(i-1) * N + j];
            if(j-1 > 0)
                res[i*N+j] += A[i*N+j][j-1+i*N] * V[i * N + (j-1)];
        }
    }

    return res;
}

std::vector<float> Vect_multi_scalar(std::vector<float> V, float num, int size){
    std::vector<float> res(size, 0.0);

    for(int i=0; i < size; i++)
        res[i] = V[i] * num;

    return res;
}

std::vector<float> Vect_dif(std::vector<float> A, std::vector<float> B, int size){
    std::vector<float> res(size, 0.0);
    for(int i=0; i < size; i++)
        res[i] = A[i] - B[i];
    return res;
}

float Vect_scalar(std::vector<float> A, std::vector<float> B, int size, float h1, float h2){
    float res = 0.0;

    for(int i=0; i<size; i++)
        res += A[i] * B[i];
    return res * h1 * h2;
}

float Vector_Euclid(std::vector<float> V, int size, float h1, float h2){
    auto res = Vect_scalar(V, V, size, h1, h2);
    return sqrt(res);
}

std::tuple<std::vector<float>, int> MinNev(int size, std::vector<std::vector<float>> A, std::vector<float> F, float sigma, float h1, float h2, int N){
    std::vector<float> w(size, 0.0);
    std::vector<float> w_plus1(size, 0.0);
    std::vector<float> r(size, 0.0);
    std::vector<float> Ar(size, 0.0);
    float delim = 0.0;
    float tau = 0.0;
    float condition = 1.0;
    int counter = 0;
    
    do{
        r = Vect_dif(Vect_multi_w(A, w, size, N), F, size);
        Ar = Vect_multi(A, r, size);
        delim = Vector_Euclid(Ar, size, h1, h2);
        tau = Vect_scalar(Ar, r, size, h1, h2) / (delim * delim);
        w_plus1 = Vect_dif(w, Vect_multi_scalar(r, tau, size), size);
        condition = Vector_Euclid(Vect_dif(w_plus1, w, size), size, h1, h2);
        if ((counter % 100) == 0)
            cout<<condition<<endl;
        w = w_plus1;
        counter++;
    }while (condition >= sigma);

    return std::make_tuple(w, counter);
    
    
}

void get_A(float h_1, float h_2, float eps, int grid_size, std::vector<std::vector<Point>> grid){
    int n_points = 1000;
    std::vector<std::vector<float>> A((grid_size + 1) * (grid_size + 1), std::vector<float>((grid_size + 1) * (grid_size + 1), 0.0));

    std::vector<float> F((grid_size + 1) * (grid_size + 1), 0.0);
    std::vector<float> ans((grid_size + 1) * (grid_size + 1), 0.0);

    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float a_ij;
    float b_ij;
    float a_i1_j;
    float b_i_j1;
    float sigma = 1/100000.0;
    int number_of_iterations;

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
    cout<<endl;
    print_matrix(A, (grid_size+1)*(grid_size+1), (grid_size+1)*(grid_size+1));
    for(int i=0; i< grid_size+1;i++){
        for(int j=0; j<grid_size+1;j++)
            cout<<F[i * (grid_size+1) + j]<<",";
        //cout<<endl;
    }
    cout << endl;
    std::tie(ans, number_of_iterations) = MinNev((grid_size + 1) * (grid_size + 1),A, F, sigma, h_1, h_2, grid_size+1);
    cout<<"MinNev_iterations -> "<<number_of_iterations<<endl;
    
    std::ofstream out;
    out.open("results.txt");
    if(out.is_open()){
        for(int i=0; i < (grid_size + 1) * (grid_size + 1); i++)
            out<<ans[i]<<endl;
    }
    out.close();

    for(int i=0; i < (grid_size + 1) * (grid_size + 1); i++)
        cout<<ans[i]<<",";


}

int main()
{
    int grid_size = 20;
    float A1 = -2.0;
    float A2 = -2.0;
    float B1 = 2.0;
    float B2 = 1.0;
    float h1 = (B1 - A1)/grid_size;
    float h2 = (B2 - A2)/grid_size;
    float eps = max(h1,h2)*max(h1,h2);

    std::vector<std::vector<Point>> grid = create_grid(grid_size, grid_size, A1, A2, B1, B2, h1, h2);
    //print_matrix(grid, grid_size+1, grid_size+1);
    get_A(h1, h2, eps, grid_size, grid);
 
}