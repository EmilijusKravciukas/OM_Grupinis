#include "affine_scale.h"  // Your affine solver header
#include "simplex.h" // Your Simplex solver implementation

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;

// Affine Solver Wrapper
void runAffineTest(const string& name, const MatrixXd& A, const VectorXd& b, const VectorXd& c, const vector<char>& constraints) {
    cout << "\n===== [Affine Solver] Test: " << name << " =====\n";
    Affinine solver(0.5, 1e-6, 1000);
    solver.setupProblem(A, b, c, constraints);
    solver.solve();
    solver.printSolution();
}

// Simplex Solver Wrapper
void runSimplexTest(const string& name, const MatrixXd& A, const VectorXd& b, const VectorXd& c) {
    cout << "\n===== [Simplex Solver] Test: " << name << " =====\n";

    vector<vector<double>> A_vec(A.rows(), vector<double>(A.cols()));
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            A_vec[i][j] = A(i, j);

    vector<double> b_vec(b.data(), b.data() + b.size());
    vector<double> c_vec(c.data(), c.data() + c.size());

    Simplex solver(c_vec, A_vec, b_vec);
    solver.calculate();
}

// Shared runner
void runTest(const string& name, const MatrixXd& A, const VectorXd& b, const VectorXd& c, const vector<char>& constraints) {
    runAffineTest(name, A, b, c, constraints);
    runSimplexTest(name, A, b, c);
}

class Simplex{
private:
    vector<vector<double>> matrix;
    vector<int> base;
    int rows, cols;

    bool isOptimal(){
        for (int j = 0; j < cols - 1; ++j){
            if (matrix[0][j] > 1e-4){
                return false;
            }
        }
        return true;
    }

    int findEnteringVariable(){
        int entering = -1;
        double best = 0;
        for (int j = 0; j < cols - 1; ++j){
            if (matrix[0][j] > best){
                best = matrix[0][j];
                entering = j;
            }
        }
        return entering;
    }

    int findLeavingVariable(int entering){
        int leaving = -1;
        double minRatio = INFINITY; //uz sita tikrai nieko didesnio nera
        for (int i = 1; i < rows; ++i){
            if (matrix[i][entering] > 1e-4){
                double ratio = matrix[i][cols - 1] / matrix[i][entering];
                if (ratio < minRatio){
                    minRatio = ratio;
                    leaving = i;
                }
            }
        }
        return leaving;
    }

    void pivot(int leaving, int entering){
        double pivot = matrix[leaving][entering];
        for (int j = 0; j < cols; ++j){
            matrix[leaving][j] /= pivot;
        }
        for (int i = 0; i < rows; ++i){
            if (i != leaving) {
                double factor = matrix[i][entering];
                for (int j = 0; j < cols; ++j){
                    matrix[i][j] -= factor * matrix[leaving][j]; // matricos atimties veiksmas su pivotinimo(pamirsau kaip lietuviskai vadinasi) eilute
                }
            }
        }
        base[leaving - 1] = entering;
    }

public:
    Simplex(const vector<double>& objective, const vector<vector<double>>& A, const vector<double>& b){
        int n = objective.size(); 
        int m = A.size(); 

        rows = m + 1; // ribojimai ir funkcija
        cols = n + m + 1; // kintamieji, fiktyvus kintamieji ir sprendinys
        matrix.assign(rows, vector<double>(cols, 0.0)); // kurti matrica
        base.resize(m);

        for (int j = 0; j < n; ++j){
            matrix[0][j] = -objective[j]; //Pritaikom simpleksa minimizavimui
        }
        for (int i = 0; i < m; ++i){
            for (int j = 0; j < n; ++j){
                matrix[i + 1][j] = A[i][j];
            }
            matrix[i + 1][n + i] = 1.0;
            matrix[i + 1][cols - 1] = b[i];
            base[i] = n + i; //s1, s2, s3 yra pradine baze
        }
    }

    void calculate() {
        cout << "Pradine simplekso matrica:\n";
        printMatrix();

        int iteration = 0;
        while (!isOptimal()) {
            int entering = findEnteringVariable();
            int leaving = findLeavingVariable(entering);

            if (entering == -1 || leaving == -1) {
                cout << "Sprendinys nerastas.\n";
                return;
            }

            cout << "\nIteracija " << iteration << ":\n";
            iteration++;
            cout << "Ieinantis kintamasis: ";
            
            int vars = cols - 1 - base.size();
            if (entering < vars){
                cout << "x_" << entering + 1;
            } else{
                cout << "s_" << entering - vars + 1;
            }
            
            cout << ", Iseinantis kintamasis: ";
            int leavingVar = base[leaving - 1];
            if (leavingVar < vars) {
                cout << "x_" << leavingVar + 1;
            } else {
                cout << "s_" << leavingVar - vars + 1;
            }
            cout << endl;

            pivot(leaving, entering);
            printMatrix();
        }

        printSolution();
    }

    void printMatrix(){
        int vars = cols - 1 - base.size();

        cout << setw(8) <<left<< "Baze";
        cout << " |";
        for (int j = 0; j < vars; ++j) cout << setw(8) << "x_" + to_string(j + 1);
        for (int j = 0; j < base.size(); ++j) cout << setw(8) << "s_" + to_string(j + 1);
        cout << setw(6) << "Rez\n";

        cout << setw(6) <<left<< "f";
        cout <<  " |";
        for (int j = 0; j < cols; ++j){
            cout << setw(8) << fixed << setprecision(3) << matrix[0][j];
        }
        cout << endl;

        for (int i = 1; i < rows; ++i){
            string var;
            if (base[i - 1] < vars){
                var = "x_" + to_string(base[i - 1] + 1);
            } else{
                var = "s_" + to_string(base[i - 1] - vars + 1);
            }
            cout << setw(8) <<left<< var << " |";
            for (int j = 0; j < cols; ++j){
                cout << setw(8) << fixed << setprecision(3) << matrix[i][j];
            }
            cout << endl;
        }
    }

    void printSolution() {
        int vars = cols - 1 - base.size();
        vector<double> solution(vars, 0.0);

        for (int i = 0; i < base.size(); ++i){
            if (base[i] < vars){
                solution[base[i]] = matrix[i + 1][cols - 1];
            }
        }
            

        double f = matrix[0][cols - 1];

        cout << "Minimali funkcijos reiksme: " << fixed << setprecision(3) << f << "\n";
        cout << "Optimalus sprendinys:\n";
        for (int i = 0; i < vars; ++i)
            cout << "x_" << i + 1 << " = " << fixed << setprecision(3) << solution[i] << "\n";
            
        cout << "Optimalus baziniai kintamieji: {";
        for (int i = 0; i < base.size(); ++i) {
            if (base[i] < vars) {
                cout << "x_" << base[i] + 1;
            } else {
                cout << "s_" << base[i] - vars + 1;
            }
            if (i < base.size() - 1) cout << ", ";
        }
        cout << "}\n";
    }
};

int main() {
    vector<double> objective = {2, -3, 0, -5};
    vector<vector<double>> constraints = {
        {-1, 1, -1, -1},
        { 2, 4,  0,  0},
        { 0, 0,  1,  1}
    };
    vector<double> constraintSolutions = {8, 10, 3};

    Simplex solver(objective, constraints, constraintSolutions);
    solver.calculate();
    cout << "\n\n";

    return 0;
}