#pragma once

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>

using namespace std;

class Simplex {
private:
    vector<vector<double>> matrix;
    vector<int> base;
    int rows, cols;

    bool isOptimal();
    int findEnteringVariable();
    int findLeavingVariable(int entering);
    void pivot(int leaving, int entering);

public:
    Simplex(const vector<double>& objective, const vector<vector<double>>& A, const vector<double>& b);
    void calculate();
    void printMatrix();
    void printSolution();
};