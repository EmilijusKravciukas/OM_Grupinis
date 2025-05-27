#include "affine_scale.h"

Affinine::Affinine(double gamma, double epsilon, int max_iter) : gamma(gamma), epsilon(epsilon), max_iterations(max_iter) {}

void Affinine::setupProblem(const MatrixXd& A_orig, const VectorXd& b_orig, const VectorXd& c_orig, const vector<char>& constraint_types) {
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

void Affinine::initFeasiblePoint() {
    int n = A.cols();

    // Start with a better initial point - use constraint bounds
    x = VectorXd::Zero(n);

    // Set original variables to reasonable values
    int orig_vars = c.size() - A.rows();
    for (int i = 0; i < orig_vars; i++) {
        x(i) = min(10.0, b.minCoeff() * 0.1); // More aggressive initial point
    }

    // Calculate slack variables
    VectorXd Ax = A.leftCols(orig_vars) * x.head(orig_vars);
    for (int i = 0; i < A.rows(); i++) {
        x(orig_vars + i) = max(1.0, b(i) - Ax(i)); // Set slack to remaining capacity
    }

    cout << "Initial feasible point found." << endl;
    cout << "Initial constraint satisfaction:" << endl;
    VectorXd check = A * x;
    for (int i = 0; i < check.size(); i++) {
        cout << "  " << check(i) << " <= " << b(i) << endl;
    }
}

// Compute steepest descent direction in transformed space
VectorXd Affinine::computeDescentDirection(const VectorXd& x_k) {
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

bool Affinine::solve() {
    cout << "Starting Affinine Algorithm..." << endl;
    cout << "Initial point: " << x.transpose() << endl;
    cout << "Initial objective value: " << c.dot(x) << endl << endl;

    double prev_obj = c.dot(x);

    for (int k = 0; k < max_iterations; k++) {
        // Compute descent direction in transformed space
        VectorXd direction = computeDescentDirection(x);

        // Check stopping criterion
        double direction_norm = direction.norm();
        if (direction_norm < epsilon) {
            cout << "Converged due to small direction norm after " << k << " iterations." << endl;
            cout << "Final objective value: " << c.dot(x) << endl;
            return true;
        }

        // Compute step size to maintain feasibility
        double alpha = gamma; // Start with default step size


        // Take the step (proper affine scaling step)
        VectorXd x_new = x + alpha * x.cwiseProduct(direction);

        // Ensure all variables remain strictly positive
        for (int i = 0; i < x_new.size(); ++i) {
            if (x_new(i) <= 1e-8) {
                x_new(i) = 1e-8;
            }
        }

        // Check constraint feasibility
        VectorXd Ax_new = A * x_new;
        bool feasible = true;
        for (int i = 0; i < b.size(); ++i) {
            if (Ax_new(i) > b(i) + epsilon) {
                feasible = false;
                break;
            }
        }

        x = x_new;
        double current_obj = c.dot(x);

        // Additional convergence check based on objective improvement
        if (abs(current_obj - prev_obj) < epsilon && k > 10) {
            cout << "Converged due to small objective change after " << k << " iterations." << endl;
            cout << "Final objective value: " << current_obj << endl;
            return true;
        }

        prev_obj = current_obj;

        if (k % 10 == 0) {
            cout << "Iteration " << k << ": ";
            cout << "Objective = " << fixed << setprecision(6) << current_obj;
            cout << ", Step size = " << alpha << endl;
        }
    }

    cout << "Maximum iterations reached." << endl;
    return false;
}

VectorXd Affinine::getSolution() const {
    return x;
}

double Affinine::getObjectiveValue() const {
    return c.dot(x);
}

void Affinine::printSolution() const {
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