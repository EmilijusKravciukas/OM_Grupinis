#ifndef AFFINE_SCALE_H
#define AFFINE_SCALE_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <iomanip>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


class Affinine {
    
public:
    Affinine(double alpha = 0.5, double tol = 1e-6, int maxIter = 1000);

    void setupProblem(const MatrixXd& A, const VectorXd& b, const VectorXd& c, const vector<char>& constraints);
    bool solve();
    void printSolution() const;

private:
    MatrixXd A;
    VectorXd b, c, x;
    vector<char> constraints;
    double alpha, tol;
    int maxIter;
};

#endif