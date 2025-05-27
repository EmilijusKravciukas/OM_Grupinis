#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace Eigen;
using namespace std;

int coolVar;

class KarmarkarSolver {
private:
    MatrixXd A;
    VectorXd b;
    VectorXd c;
    VectorXd x;
    double gamma;
    double epsilon;
    int max_iterations;

public:
    KarmarkarSolver(double gamma = 0.5, double epsilon = 1e-6, int max_iter = 1000)
        : gamma(gamma), epsilon(epsilon), max_iterations(max_iter) {}

    // Convert inequality constraints to standard form
    void setupProblem(const MatrixXd& A_orig, const VectorXd& b_orig, const VectorXd& c_orig,
        const vector<char>& constraint_types) {
        int m = A_orig.rows();
        int n = A_orig.cols();

        // Count slack variables needed
        int slack_vars = 0;
        for (char type : constraint_types) {
            if (type == '<' || type == 'L') slack_vars++;
        }

        // Setup matrices with slack variables
        A = MatrixXd::Zero(m, n + slack_vars);
        A.leftCols(n) = A_orig;

        b = b_orig;
        c = VectorXd::Zero(n + slack_vars);
        c.head(n) = c_orig;

        // Add slack variables for inequality constraints
        int slack_idx = 0;
        for (int i = 0; i < m; i++) {
            if (constraint_types[i] == '<' || constraint_types[i] == 'L') {
                A(i, n + slack_idx) = 1.0;
                slack_idx++;
            }
        }

        // Initialize feasible starting point
        initializeFeasiblePoint();
    }

    void initializeFeasiblePoint() {
        int n = A.cols();
        x = VectorXd::Constant(n, 1.0); // start with a positive vector

        for (int iter = 0; iter < 100; ++iter) {
            VectorXd Ax = A * x;
            VectorXd violation = Ax - b;

            // If feasible, we're done
            if (violation.maxCoeff() <= epsilon) {
                cout << "Feasible starting point found after " << iter << " iterations." << endl;
                return;
            }

            // Project violation back toward feasibility
            VectorXd adjustment = A.transpose() * ((A * A.transpose()).ldlt().solve(violation));
            x -= adjustment;

            // Ensure positivity
            for (int i = 0; i < x.size(); ++i) {
                if (x(i) <= 0) x(i) = 1e-3;
            }
        }

        cout << "Warning: Could not find a strictly feasible point. Proceeding with best effort." << endl;
    }

    // Projective transformation
    VectorXd projectiveTransform(const VectorXd& x_k) {
        VectorXd D_inv = x_k.cwiseInverse();
        VectorXd x_prime = VectorXd::Ones(x_k.size());

        for (int i = 0; i < x_k.size(); i++) {
            x_prime(i) = 1.0 / x_k(i);
        }
        x_prime = x_prime / x_prime.sum() * x_k.size();

        return x_prime;
    }

    // Inverse transformation  
    VectorXd inverseTransform(const VectorXd& x_prime, const VectorXd& x_k) {
        VectorXd x_new(x_k.size());
        double sum_x_prime = x_prime.sum();

        for (int i = 0; i < x_k.size(); i++) {
            x_new(i) = x_k(i) * x_prime(i) / sum_x_prime;
        }

        return x_new;
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
        cout << "Starting Karmarkar's Algorithm..." << endl;
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

            // Update point using step size gamma
            VectorXd x_new = x + gamma * x.cwiseProduct(direction);

            // Ensure feasibility and positivity
            double min_val = x_new.minCoeff();
            if (min_val <= 0) {
                double alpha = 0.99 * x.cwiseQuotient(-direction).minCoeff();
                if (alpha > 0 && alpha < gamma) {
                    x_new = x + alpha * x.cwiseProduct(direction);
                }
            }

            // Ensure all components remain positive
            for (int i = 0; i < x_new.size(); i++) {
                if (x_new(i) <= 0) {
                    x_new(i) = 1e-8;
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

    // Coefficient matrix (without slack variables)
    MatrixXd A_orig(3, 4);
    A_orig << -1, 1, -1, -1,
        2, 4, 0, 0,
        0, 0, 1, 1;

    // Right-hand side
    VectorXd b_orig(3);
    b_orig << 8, 10, 3;

    // Objective coefficients  
    VectorXd c_orig(4);
    c_orig << 2, -3, 0, -5;

    // Constraint types ('L' for ≤, 'E' for =)
    vector<char> constraint_types = { 'L', 'L', 'L' };

    // Create and solve
    KarmarkarSolver solver(0.5, 1e-6, 1000);
    solver.setupProblem(A_orig, b_orig, c_orig, constraint_types);

    bool success = solver.solve();

    if (success) {
        solver.printSolution();
    }
    else {
        cout << "Algorithm did not converge to optimal solution." << endl;
        cout << "Current best solution:" << endl;
        solver.printSolution();
    }

    return 0;
}