#include "affine_scale.h"
#include <iostream>

using namespace std;

Affinine::Affinine(double alpha, double tol, int maxIter)
    : alpha(alpha), tol(tol), maxIter(maxIter) {}

void Affinine::setupProblem(const MatrixXd& A_, const VectorXd& b_, const VectorXd& c_, const vector<char>& constraints_) {
    A = A_;
    b = b_;
    c = c_;
    constraints = constraints_;
    x = VectorXd::Constant(c_.size(), 1.0);  // basic init
}

bool Affinine::solve() {
    for (int iter = 0; iter < maxIter; ++iter) {
        VectorXd grad = c;
        VectorXd direction = -grad.normalized();

        double stepSize = alpha;
        x += stepSize * direction;

        cout << "Iteration " << iter << ": Objective = " << c.dot(x) << ", Direction norm = " << direction.norm() << endl;

        if (direction.norm() < tol)
            return true;
    }
    return false;
}

void Affinine::printSolution() const {
    cout << "Final solution:\n" << x.transpose() << endl;
    cout << "Objective value: " << c.dot(x) << endl;
}