#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Constants
#define PI 3.14159265358979323846

// Compute Rastrigin function and gradient
double rastrigin(double *x, double *s, int n, double *grad) {
    double f = 10.0 * n;  // Base term: 10 * dimension
    double z;             // Temp: x_i - s_i

    for (int i = 0; i < n; i++) {
        z = x[i] - s[i];
        f += z * z - 10.0 * cos(2.0 * PI * z);  // Function value
        if (grad != NULL) {                      // Compute gradient if requested
            grad[i] = 2.0 * z + 20.0 * PI * sin(2.0 * PI * z);
        }
    }
    return f;
}

// Example trajectory generation with Adam-like updates
void generate_trajectory(double *x0, double *s, int n, int steps, double lr,
                        double *x_traj, double *f_traj, double *grad_traj) {
    double *x = (double *)malloc(n * sizeof(double));    // Current x
    double *grad = (double *)malloc(n * sizeof(double)); // Gradient
    double m[n], v[n];                                   // Adam moments

    // Initialize
    for (int i = 0; i < n; i++) {
        x[i] = x0[i];
        m[i] = 0.0;  // First moment (momentum)
        v[i] = 0.0;  // Second moment (RMSProp)
    }

    // Adam parameters (simplified)
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

    // Iterate
    for (int t = 0; t < steps; t++) {
        // Compute function and gradient
        double f = rastrigin(x, s, n, grad);
        
        // Store trajectory
        f_traj[t] = f;
        for (int i = 0; i < n; i++) {
            x_traj[t * n + i] = x[i];
            grad_traj[t * n + i] = grad[i];
        }

        // Adam update (simplified, no bias correction for brevity)
        for (int i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            x[i] -= lr * m[i] / (sqrt(v[i]) + eps);
        }
    }

    free(x);
    free(grad);
}

// Example usage
int main() {
    int n = 50;      // Dimension
    int steps = 50000;  // Trajectory length
    double lr = 0.01;

    // Allocate memory
    double *x0 = (double *)malloc(n * sizeof(double));
    double *s = (double *)malloc(n * sizeof(double));
    double *x_traj = (double *)malloc(steps * n * sizeof(double));
    double *f_traj = (double *)malloc(steps * sizeof(double));
    double *grad_traj = (double *)malloc(steps * n * sizeof(double));

    // Initialize x0 and s with random values (simplified)
    for (int i = 0; i < n; i++) {
        x0[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;  // [-5, 5]
        s[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;    // [-1, 1]
    }

    // Generate trajectory
    generate_trajectory(x0, s, n, steps, lr, x_traj, f_traj, grad_traj);

    // Print first few steps (for verification)
    for (int t = 45000; t < 50000; t++) {
        printf("Step %d: f(x) = %f\n", t, grad_traj[t]);
    }

    // Clean up
    free(x0);
    free(s);
    free(x_traj);
    free(f_traj);
    free(grad_traj);
    return 0;
}