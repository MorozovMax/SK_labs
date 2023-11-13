#include <iostream>
#include <vector>
#include <string>
#include <bits/stdc++.h>

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

void print_matrix(std::vector<std::vector<float>> matrix, int M, int N){
    for(int i=0;i < M;i++){
        for(int j=0;j < N;j++)
            if (matrix[i][j] != 0)
                std::cout<<"+"<<" ";
            else
                std::cout<<"-"<<" ";
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
    int x_dist = x_max - x_min;
    if (x_min == x_max)
        x_dist = 1;
    int y_dist = y_max - y_min;
    if (y_min == y_max)
        y_dist = 1;
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
        res = (l/h2) - (1 - (l/h2))/eps;
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
        res = (l/h1) - (1 - (l/h1))/eps;
    return res;
}

float get_F(float h1, float h2, float eps, float x_min, float x_max, float y_min, float y_max, int n_points){
    float S;
    int points_inside;
    int total_points;
    float res;

    std::tie(S, points_inside, total_points) = monte_carlo(x_min, x_max, y_min, y_max, n_points);
    if (points_inside == total_points)
        res = 1;
    else if (points_inside == 0)
        res = 0;
    else
        res = S/(h1*h2);
    return res;
}


void get_A(float h1, float h2, float eps, int grid_size, std::vector<std::vector<Point>> grid){
    int n_points = 1000;
    std::vector<std::vector<float>> A;
    A.resize((grid_size+1)*(grid_size+1));
    for(int i=0;i < (grid_size+1)*(grid_size+1);++i){
        A[i].resize((grid_size+1)*(grid_size+1));
    }

    for(int i=0;i < (grid_size+1)*(grid_size+1);i++){
        for(int j=0;j < (grid_size+1)*(grid_size+1);j++){
            A[i][j] = 0;
        }
    }
    std::vector<std::vector<float>> F;
    F.resize(grid_size+1);
    for(int i=0;i < (grid_size+1);i++){
        F[i].resize(grid_size+1);
    }

    for(int i=0;i < (grid_size+1);i++){
        for(int j=0;j < (grid_size+1);j++){
            F[i][j] = 0;
        }
    }

    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float a_i_j;
    float b_i_j;
    float a_iplus1_j;
    float b_iplus1_j;


    for(int i=1;i < grid_size+1;i++){
        for(int j=1;j < grid_size+1;j++){
            x_min = min(grid[i][j].x - 0.5 * h1, grid[i][j].x + 0.5 * h1);
            x_max = max(grid[i][j].x - 0.5 * h1, grid[i][j].x + 0.5 * h1);
            y_min = min(grid[i][j].y - 0.5 * h2, grid[i][j].y + 0.5 * h2);
            y_max = max(grid[i][j].y - 0.5 * h2, grid[i][j].y + 0.5 * h2);
            a_i_j = get_a(h2, eps, x_min, x_max, y_min, y_max, n_points);
            b_i_j = get_b(h1, eps, x_min, x_max, y_min, y_max, n_points);
            F[i][j] = get_F(h1, h2, eps, x_min, x_max, y_min, y_max, n_points);

            if (i < grid_size){
                x_min = min(grid[i+1][j].x - 0.5 * h1, grid[i+1][j].x + 0.5 * h1);
                x_max = max(grid[i+1][j].x - 0.5 * h1, grid[i+1][j].x + 0.5 * h1);
                y_min = min(grid[i+1][j].y - 0.5 * h2, grid[i+1][j].y + 0.5 * h2);
                y_min = max(grid[i+1][j].y - 0.5 * h2, grid[i+1][j].y + 0.5 * h2);
                a_iplus1_j = get_a(h2, eps, x_min, x_max, y_min, y_max, n_points);

                x_min = min(grid[i+1][j].x - 0.5 * h1, grid[i][j].x + 0.5 * h1);
                x_max = max(grid[i+1][j].x - 0.5 * h1, grid[i][j].x + 0.5 * h1);
                y_min = min(grid[i+1][j].y - 0.5 * h2, grid[i][j].y + 0.5 * h2);
                y_min = max(grid[i+1][j].y - 0.5 * h2, grid[i][j].y + 0.5 * h2);
                b_iplus1_j = get_b(h1, eps, x_min, x_max, y_min, y_max, n_points);
            }
            else{
                a_iplus1_j = 0;
                b_iplus1_j = 0;
            }

            A[i * (grid_size + 1)+j][i*(grid_size + 1)+j] = ((a_iplus1_j + a_i_j)/(h1*h1) + (b_iplus1_j + b_i_j)/(h2*h2));
            A[i * (grid_size + 1)+j][i*(grid_size + 1)+j+1] = -b_iplus1_j/(h2*h2);
            A[i * (grid_size + 1)+j][i*(grid_size + 1)+j-1] = -b_i_j/(h2*h2);
            A[i * (grid_size + 1)+j][(i+1)*(grid_size + 1)+j] = -a_iplus1_j/(h1*h1);
            A[i * (grid_size + 1)+j][(i-1)*(grid_size + 1)+j] = -a_i_j/(h1*h1);
        }
    }
    print_matrix(A, (grid_size+1)*(grid_size+1), (grid_size+1)*(grid_size+1));
    //print_matrix(F, grid_size+1, grid_size+1);

}

int main()
{
    int M = 10;
    int N = 10;
    int grid_size = 10;
    float A1 = -2;
    float A2 = -2;
    float B1 = 2;
    float B2 = 1;
    float h1 = (B1 - A1)/M;
    float h2 = (B2 - A2)/N;
    float eps = 1/1000000.0;

    std::vector<std::vector<Point>> grid = create_grid(M, N, A1, A2, B1, B2, h1, h2);
    //print_matrix(grid, 11, 11);
    get_A(h1, h2, eps, 10, grid);
}