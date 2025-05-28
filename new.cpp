bool solve() {
     cout << "Starting Affinine Algorithm" << endl;
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
             cout << "Final objective value: " << prev_obj << endl;
             return true;
         }

         // Take the step (proper affine scaling step)
         VectorXd x_new = x + gamma * x.cwiseProduct(direction);

         // Ensure all variables remain strictly positive
         for (int i = 0; i < x_new.size(); ++i) {
             if (x_new(i) <= 1e-8) {
                 x_new(i) = 1e-8;
             }
         }

         x = x_new;
         double current_obj = c.dot(x);

         if (abs(current_obj - prev_obj) < epsilon && k > 10) {
             cout << "Converged due to small objective change after " << k << " iterations." << endl;
             cout << "Final objective value: " << current_obj << endl;
             return true;
         }
         if (k % 10 == 0) {
             cout << "Iteration " << k << ": ";
             cout << "Objective = " << fixed << setprecision(6) << current_obj;
             cout << ", Step size = " << gamma << endl;
         }
         prev_obj = current_obj;
     }

     cout << "Maximum iterations reached." << endl;
     return false;
 }
