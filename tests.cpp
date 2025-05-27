#include "affine_scale.h"
#include <vector>
#include <string>
#include <iostream>

using namespace std;

void runAffineScale(const std::string& name, const MatrixXd& A, const VectorXd& b, const VectorXd& c, const vector<char>& constraints) {
    cout << "\n==============================" << endl;
    cout << "Running test: " << name << endl;
    cout << "==============================" << endl;

    Affinine solver(0.5, 1e-6, 1000);
    solver.setupProblem(A, b, c, constraints);
    bool result = solver.solve();
    solver.printSolution();
}

int main() {
    // ---------- Test 1: Simple 2D ----------
    MatrixXd A1(2, 2);
    A1 << -1, -2,
          -4, -1;
    VectorXd b1(2);
    b1 << -4, -4;
    VectorXd c1(2);
    c1 << -1, -1;
    runAffineScale("Basic 2D LP", A1, b1, c1, { 'L', 'L' });

    // ---------- Test 2: Degeneracy ----------
    MatrixXd A2(2, 3);
    A2 << 1, 1, 1,
           1, 1, 0;
    VectorXd b2(2);
    b2 << 1, 1;
    VectorXd c2(3);
    c2 << 10, 0, 0;
    runAffineScale("Degeneracy Test", A2, b2, c2, { 'E', 'E' });

    // ---------- Test 3: Unbounded ----------
    MatrixXd A3(1, 2);
    A3 << -1, 1;
    VectorXd b3(1);
    b3 << -2;
    VectorXd c3(2);
    c3 << 1, 1;
    runAffineScale("Unbounded LP", A3, b3, c3, { 'L' });

    // ---------- Test 4: Simplex Favoring ----------
    MatrixXd A4(2, 3);
    A4 << 1, 1, 1,
          2, 2, 1;
    VectorXd b4(2);
    b4 << 100, 150;
    VectorXd c4(3);
    c4 << -100, -10, -1;
    runAffineScale("Simplex Friendly LP", A4, b4, c4, { 'L', 'L' });

    // ---------- Test 5: Interior Point ----------
    MatrixXd A5(1, 3);
    A5 << 1, 1, 1;
    VectorXd b5(1);
    b5 << 1;
    VectorXd c5(3);
    c5 << 2, 3, 1;
    runAffineScale("Interior Test", A5, b5, c5, { 'E' });

    // ---------- Test 6: Multiple Optima ----------
    MatrixXd A6(2, 3);
    A6 << 1, 1, 0,
          1, 0, 1;
    VectorXd b6(2);
    b6 << 1, 1;
    VectorXd c6(3);
    c6 << 0, 0, 1;
    runAffineScale("Multiple Optima", A6, b6, c6, { 'E', 'E' });

    // ---------- Test 7: Diet Problem ----------
    MatrixXd A7(2, 2);
    A7 << -400, -200,
          -3, -1;
    VectorXd b7(2);
    b7 << -500, -6;
    VectorXd c7(2);
    c7 << 50, 20;
    runAffineScale("Mini Diet LP", A7, b7, c7, { 'L', 'L' });

    return 0;
}