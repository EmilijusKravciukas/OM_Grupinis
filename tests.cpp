#include "affine_scale.h"
#include "simplex.h"

using namespace std;

void runAffineScale(const std::string& name, const MatrixXd& A, const VectorXd& b, const VectorXd& c, const vector<char>& constraints) {
      cout << "\n==============================" << endl;
      cout << "Running test: " << name << endl;
      cout << "==============================" << endl;

      Affinine solver(0.5, 1e-6, 1000);
      solver.setupProblem(A, b, c, constraints);
      if (solver.solve()) {
            solver.printSolution();
      }
      else {
            cout << "No Optimal,\n";
            cout << "Reached feasible solution:\n";
            solver.printSolution();
      }
}


void runSimplexTest(const std::string& name, const MatrixXd& A, const VectorXd& b, const VectorXd& c) {
    std::cout << "\n==============================" << std::endl;
    std::cout << "Running Simplex test: " << name << std::endl;
    std::cout << "==============================" << std::endl;

    // Convert Eigen data to std::vector (Simplex uses std::vector)
    std::vector<std::vector<double>> A_vec(A.rows(), std::vector<double>(A.cols()));
    std::vector<double> b_vec(b.size());
    std::vector<double> c_vec(c.size());

    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            A_vec[i][j] = A(i, j);

    for (int i = 0; i < b.size(); ++i)
        b_vec[i] = b(i);

    for (int i = 0; i < c.size(); ++i)
        c_vec[i] = c(i);

    // Call the Simplex class
    Simplex solver(c_vec, A_vec, b_vec);
    solver.calculate();
}

int main() {

      // ---------- Test 0: Original ----------
      MatrixXd A0(3, 4);
      A0 << -1, 1, -1, -1,
        2, 4, 0, 0,
        0, 0, 1, 1;

      VectorXd b0(3);
      b0 << 8, 10, 3;

      VectorXd c0(4);
      c0 << 2, -3, 0, -5;

      vector<char> ct0 = { 'L', 'L', 'L' };

      // --------- Test 1: 2D Matrix ---------

      MatrixXd A1(2, 2);
      A1 << -1, -2,
          -4, -1;
      VectorXd b1(2);
      b1 << -4, -4;
      VectorXd c1(2);
      c1 << -1, -1;

      vector<char> ct1 = {'L', 'L'};

      // ---------- Test 2: Degeneracy ----------
      MatrixXd A2(2, 3);
      A2 << 1, 1, 1,
           1, 1, 0;
      VectorXd b2(2);
      b2 << 1, 1;
      VectorXd c2(3);
      c2 << 10, 0, 0;

      vector<char> ct2 = {'E', 'E'};

      // ---------- Test 3: Unbounded ----------
      MatrixXd A3(1, 2);
      A3 << -1, 1;
      VectorXd b3(1);
      b3 << -2;
      VectorXd c3(2);
      c3 << 1, 1;

      vector<char> ct3 = {'L'};

      // ---------- Test 4: Simplex Favoring ----------
      MatrixXd A4(2, 3);
      A4 << 1, 1, 1,
          2, 2, 1;
      VectorXd b4(2);
      b4 << 100, 150;
      VectorXd c4(3);
      c4 << -100, -10, -1;

      vector<char> ct4 = {'L', 'L'};

      // ---------- Test 5: Interior Point ----------
      MatrixXd A5(1, 3);
      A5 << 1, 1, 1;
      VectorXd b5(1);
      b5 << 1;
      VectorXd c5(3);
      c5 << 2, 3, 1;

      vector<char> ct5 = {'E'};

      // ---------- Test 6: Multiple Optima ----------
      MatrixXd A6(2, 3);
      A6 << 1, 1, 0,
          1, 0, 1;
      VectorXd b6(2);
      b6 << 1, 1;
      VectorXd c6(3);
      c6 << 0, 0, 1;

      vector <char> ct6 = {'E', 'E'};

      // ---------- Test 7: Diet Problem ----------
      MatrixXd A7(2, 2);
      A7 << -400, -200,
          -3, -1;
      VectorXd b7(2);
      b7 << -500, -6;
      VectorXd c7(2);
      c7 << 50, 20;

      vector<char> ct7 = {'L', 'L'};


      int userInput;
      cin>>userInput;

      switch (userInput){
            case 1:
                  runAffineScale("Test", A1, b1, c1, ct1);
                  runSimplexTest("Test", A1, b1, c1);
                  break;

            case 2:
                  runAffineScale("Test", A2, b2, c2, ct2);
                  runSimplexTest("Test", A2, b2, c2);
                  
                  break;
            case 3:
                  runAffineScale("Test", A3, b3, c3, ct3);
                  runSimplexTest("Test", A3, b3, c3);

                  break;
            case 4:
                  runAffineScale("Test", A4, b4, c4, ct4);
                  runSimplexTest("Test", A4, b4, c4);

                  break;
            case 5: // This one
                  runAffineScale("Test", A5, b5, c5, ct5);
                  runSimplexTest("Test", A5, b5, c5);

                  break;
            case 6:
                  runAffineScale("Test", A6, b6, c6, ct6);
                  runSimplexTest("Test", A6, b6, c6);

                  break;
            case 7:
                  runAffineScale("Test", A7, b7, c7, ct7);
                  runSimplexTest("Test", A7, b7, c7);
                  
                  break;
            case 0:
                  runAffineScale("Test", A0, b0, c0, ct0);
                  runSimplexTest("Test", A0, b0, c0);
                  break;
      }

      return 0;
}