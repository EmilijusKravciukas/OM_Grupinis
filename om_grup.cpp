#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace Eigen;
using namespace std;

class Affinine {
private:
    MatrixXd A; // Pilna simpleksine matrica su slackais
    VectorXd b; // Tiesiog apribojimu rezultatu vektorius
    VectorXd c; // Musu tikslo funkcijos koeficientu vektorius su slackais
    VectorXd x; // taskas
    double gamma; // spindulio (zingsnio) ilgis. Kitur zymima y
    double epsilon; // bendra paklaida
    int max_iterations; // kiek iteraciju sukasi

public:
    Affinine(double gamma = 0.5, double epsilon = 1e-6, int max_iter = 1000) : gamma(gamma), epsilon(epsilon), max_iterations(max_iter) {
    }

    void setupProblem(const MatrixXd& A_orig, const VectorXd& b_orig, const VectorXd& c_orig, const vector<char>& constraint_types) {
        int ProblemRows = A_orig.rows();
        int ProblemColumns = A_orig.cols();

        int slack_vars = 0;
        for (char type : constraint_types) {
            if (type == '<' || type == 'L') slack_vars++;
        }

        A = MatrixXd::Zero(ProblemRows, ProblemColumns + slack_vars);
        A.leftCols(ProblemColumns) = A_orig;

        b = b_orig;
        c = VectorXd::Zero(ProblemColumns + slack_vars);
        c.head(ProblemColumns) = c_orig;

        int slack_id = 0;
        for (int i = 0; i < ProblemRows; i++) {
            if (constraint_types[i] == '<' || constraint_types[i] == 'L') {
                A(i, ProblemColumns + slack_id) = 1.0;
                slack_id++;
            }
        }
        initFeasiblePoint();
    }

    void initFeasiblePoint() {
        int n = A.cols();
        x = VectorXd::Constant(n, 1.0);

        for (int iter = 0; iter < 100; ++iter) {
            VectorXd Ax = A * x;
            VectorXd violation = Ax - b;

            if (violation.maxCoeff() <= epsilon) {
                cout << "Starting point found after " << iter << " iterations." << endl;
                return;
            }

            //Idealiai gauname pazeidimu vektoriu, kuri atemus is x vektoriaus gausime reiksmes, kurios neturi pazeidimu, tokiu atveju toliau seka skaiciavimai su matrica A.
            VectorXd adjustment = A.transpose() * ((A * A.transpose()).ldlt().solve(violation));
            x -= adjustment;

            for (int i = 0; i < x.size(); ++i) {
                if (x(i) <= 0) x(i) = 1e-3;
            }
        }

        cout << "Warning: Could not find a strictly feasible point. Proceeding with best effort." << endl;
    }

    // Compute steepest descent direction in transformed space
    VectorXd computeDescentDirection(const VectorXd& x_k) {
        int n = x_k.size();
        int m = A.rows();

        // Create diagonal matrix D
        MatrixXd D = x_k.asDiagonal();

        // Transformed constraint matrix
        MatrixXd A_tilde = A * D;

        // Transformed objective vector
        VectorXd c_tilde = D * c;

        // Compute projection matrix P = I - A_tilde^T(A_tilde * A_tilde^T)^(-1) * A_tilde
        MatrixXd AAt = A_tilde * A_tilde.transpose();
        MatrixXd AAt_inv = AAt.completeOrthogonalDecomposition().pseudoInverse();
        MatrixXd P = MatrixXd::Identity(n, n) - A_tilde.transpose() * AAt_inv * A_tilde;

        // Projected gradient
        VectorXd grad_proj = P * c_tilde;

        // Normalize direction
        double norm = grad_proj.norm();
        if (norm > epsilon) {
            grad_proj = -grad_proj / norm;
        }

        return grad_proj;
    }

    bool solve() {
        cout << "Starting Affinine Algorithm..." << endl;
        cout << "Initial point: " << x.transpose() << endl;
        cout << "Initial objective value: " << c.dot(x) << endl << endl;

        for (int k = 0; k < max_iterations; k++) {
            // Check convergence
            VectorXd constraint_violation = A * x - b;
            double max_violation = constraint_violation.cwiseAbs().maxCoeff();

            if (max_violation > 1e-3) {
                cout << "Warning: Constraint violation = " << max_violation << endl;
            }

            // Compute descent direction in transformed space
            VectorXd direction = computeDescentDirection(x);

            // Check stopping criterion
            double direction_norm = direction.norm();
            if (direction_norm < epsilon) {
                cout << "Converged after " << k << " iterations." << endl;
                cout << "Final objective value: " << c.dot(x) << endl;
                return true;
            }

            VectorXd step = x.asDiagonal() * direction;

            // Compute maximum safe step to stay in feasible region
            double alpha = 1.0;
            for (int i = 0; i < x.size(); ++i) {
                if (step(i) < 0) {
                    alpha = std::min(alpha, -0.99 * x(i) / step(i));
                }
            }

            VectorXd x_new = x + std::min(gamma, alpha) * step;

            // Ensure feasibility and positivity
            double min_val = x_new.minCoeff();
            if (min_val <= 0) {
                double alpha = 0.99 * x.cwiseQuotient(-direction).minCoeff();
                if (alpha > 0 && alpha < gamma) {
                    x_new = x + alpha * x.cwiseProduct(direction);
                }
            }

            x = x_new;

            if (k % 10 == 0) {
                cout << "Iteration " << k << ": ";
                cout << "Objective = " << fixed << setprecision(6) << c.dot(x);
                cout << ", Direction norm = " << direction_norm << endl;
            }
        }

        cout << "Maximum iterations reached." << endl;
        return false;
    }

    VectorXd getSolution() const {
        return x;
    }

    double getObjectiveValue() const {
        return c.dot(x);
    }

    void printSolution() const {
        cout << "\n=== SOLUTION ===" << endl;
        cout << "Optimal point:" << endl;
        for (int i = 0; i < x.size(); i++) {
            cout << "x[" << i << "] = " << fixed << setprecision(6) << x(i) << endl;
        }
        cout << "Optimal objective value: " << fixed << setprecision(6) << c.dot(x) << endl;

        // Check constraints
        cout << "\nConstraint verification:" << endl;
        VectorXd Ax = A * x;
        for (int i = 0; i < A.rows(); i++) {
            cout << "Constraint " << i << ": " << fixed << setprecision(6)
                << Ax(i) << " ≤ " << b(i) << " (violation: "
                << max(0.0, Ax(i) - b(i)) << ")" << endl;
        }
    }
};

// Example usage with the given problem
int main() {
    // Problem: min 2*x1 - 3*x2 - 5*x4
    // Subject to:
    // -x1 + x2 - x3 - x4 ≤ 8
    // 2*x1 + 4*x2 ≤ 10  
    // x3 + x4 ≤ 3
    // xi ≥ 0

    // Koeficientu Matrica apribojimams
    MatrixXd A(3, 4);
    A << -1, 1, -1, -1,
        2, 4, 0, 0,
        0, 0, 1, 1;

    // Apribojimu reiksmes
    VectorXd Res(3);
    Res << 8, 10, 3;

    // Tikslo funkcijos koeficientai  
    VectorXd coeffs(4);
    coeffs << 2, -3, 0, -5;

    // Apribojimu tipai, Cia rasom L jeigu <=, < jeigu < arba E jeigu =
    // pvz 2x1 + 4x2 <= 8 yra L
    vector<char> constraint_types = { 'L', 'L', 'L' };

    // iki dabar visas pasiruosimas identiskas tiesinio programavimo simpleksui.

    Affinine Aff(0.5, 1e-6, 1000);
    Aff.setupProblem(A, Res, coeffs, constraint_types);

    bool success = Aff.solve();

    if (success) {
        Aff.printSolution();
    }
    else {
        cout << "No Optimal,\n";
        cout << "Reached feasible solution:\n";
        Aff.printSolution();
    }

    return 0;
}