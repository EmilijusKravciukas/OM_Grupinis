#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;
using namespace Eigen;

class Affinine {
public:
    Affinine(double gamma = 0.5, double epsilon = 1e-6, int max_iter = 1000);
    void setupProblem(const MatrixXd& A, const VectorXd& b, const VectorXd& c, const vector<char>& constraints);
    void initFeasiblePoint();
    bool solve();
    VectorXd computeDescentDirection(const VectorXd& x_k);
    VectorXd getSolution() const;
    double getObjectiveValue() const;
    void printSolution() const;


private:
    MatrixXd A; // Pilna simpleksine matrica su slackais
    VectorXd b; // Tiesiog apribojimu rezultatu vektorius
    VectorXd c; // Musu tikslo funkcijos koeficientu vektorius su slackais
    VectorXd x; // taskas
    double gamma; // spindulio (zingsnio) ilgis. Kitur zymima y
    double epsilon; // bendra paklaida
    int max_iterations; // kiek iteraciju sukasi
};