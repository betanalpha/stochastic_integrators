#include <iostream>
#include <fstream>

#include <cmath>

#include <boost/random/additive_combine.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>

inline double V(double q, double sigma, double B, double N, double ave) {
  double m = 0;
  double s = 1;
  double mu = (ave * N * s * s + m * sigma * sigma) / (sigma * sigma + N * s * s);
  double tau_inv = /*(B / N) * */(sigma * sigma + N * s * s) / (sigma * sigma * s * s);
  return 0.5 * tau_inv * (q - mu) * (q - mu);
}

inline double grad_V(double q, double sigma, double B, double N, double ave) {
  double m = 0;
  double s = 1;
  double mu = (ave * N * s * s + m * sigma * sigma) / (sigma * sigma + N * s * s);
  double tau_inv = /*(B / N) * */(sigma * sigma + N * s * s) / (sigma * sigma * s * s);
  return tau_inv * (q - mu);
}

inline void evolve(double& q, double& p,
                   double sigma, double B, double N, double ave,
                   double epsilon) {
  //std::cout << "EVOLVE: " << q << "\t" << p << std::endl;
  p -= 0.5 * epsilon * grad_V(q, sigma, B, N, ave);
  q += epsilon * p;
  p -= 0.5 * epsilon * grad_V(q, sigma, B, N, ave);
}

int main(int argc, const char * argv[]) {

  typedef boost::ecuyer1988 rng_t;
  rng_t rng(85738293);
  
  boost::variate_generator<rng_t&, boost::normal_distribution<> >
  rand_unit_gauss(rng, boost::normal_distribution<>());
  
  boost::random::uniform_real_distribution<double>
    uniform_double(-3.141592653589793, 3.141592653589793);
  
  boost::variate_generator<rng_t&, boost::random::uniform_real_distribution<double> >
  rand_uniform(rng, uniform_double);
 
  double mu = 1;
  double sigma = 2;
  
  const double N = 500;
  const int b_small = 20;
  const int n_small_batch = N / b_small;
  const int b_large = 250;
  const int n_large_batch = N / b_large;
  
  boost::random::uniform_int_distribution<> uniform_int(0, 4);
  boost::variate_generator<rng_t&, boost::random::uniform_int_distribution<> >
  rand_int(rng, uniform_int);
  
  double ave_full = 0;
  
  double ave_small_batch[n_small_batch];
  for (int n = 0; n < n_small_batch; ++n) ave_small_batch[n] = 0;

  double ave_large_batch[n_large_batch];
  for (int n = 0; n < n_large_batch; ++n) ave_large_batch[n] = 0;
  
  for (int n = 0; n < N; ++n) {
    double x = mu + sigma * rand_unit_gauss();
    ave_full += x;
    ave_small_batch[n / b_small] += x;
    ave_large_batch[n / b_large] += x;
  }
  
  ave_full /= static_cast<double>(N);
  for (int n = 0; n < n_small_batch; ++n) ave_small_batch[n] /= static_cast<double>(b_small);
  for (int n = 0; n < n_large_batch; ++n) ave_large_batch[n] /= static_cast<double>(b_large);
  
  std::cout << ave_full << std::endl;
  std::cout << ave_small_batch[0] << std::endl;
  std::cout << ave_large_batch[0] << std::endl;
  
  double m = 0;
  double s = 1;
  double q0 = (ave_full * N * s * s + m * sigma * sigma) / (sigma * sigma + N * s * s);
  
  double q = q0 + std::sqrt(6);
  double p = 0;
  double epsilon = 0.05;
  
  std::ofstream output;
  output.open("full_trajectory.dat");
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 60; ++n) {
    evolve(q, p, sigma, N, N, ave_full, epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  output.close();
  
  output.open("full_trajectory_big_stepsize.dat");
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 60; ++n) {
    evolve(q, p, sigma, N, N, ave_full, epsilon * static_cast<double>(n_small_batch));
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  output.close();
  
  output.open("small_batch_sym_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  double g_full = grad_V(q, sigma, N, N, ave_full);
  double g_sym = 0;
  
  for (int n = 0; n < 25; ++n) {
    g_sym += grad_V(q, sigma, b_small, N, ave_small_batch[n]);
  }
  
  std::cout << "Symmetric: " << g_full << "\t" << g_sym / static_cast<double>(n_small_batch) << std::endl;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 25; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  for (int n = 24; n >= 0; --n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  for (int n = 0; n < 25; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  for (int n = 24; n >= 0; --n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  output.close();

  // Plots above only at end of each update, identifying
  // the corresponding numerical level set of the full
  // symmetric composition
  output.open("small_batch_sym_loop_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int m = 0; m < 50; ++m) {
  
    for (int n = 0; n < 25; ++n) {
      evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    }
    
    for (int n = 24; n >= 0; --n) {
      evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    }
    
    output << q << "\t" << p << "\t" << std::endl;
    
  }
  
  output.close();
  
  // Plots above only at end of each update, identifying
  // the corresponding numerical level set of the full
  // symmetric composition -- step size scaled by the
  // number of batches
  output.open("small_batch_sym_loop_scaled_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int m = 0; m < 50; ++m) {
    
    for (int n = 0; n < 25; ++n) {
      evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon / static_cast<double>(n_small_batch));
    }
    
    for (int n = 24; n >= 0; --n) {
      evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon / static_cast<double>(n_small_batch));
    }
    
    output << q << "\t" << p << "\t" << std::endl;
    
  }
  
  output.close();
  
  output.open("small_batch_asym_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 25; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  for (int n = 0; n < 25; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  for (int n = 0; n < 25; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  for (int n = 0; n < 25; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[n], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }

  output.close();
  
  output.open("small_batch_asym_subsample_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  double g_asym_sub = 0;
  for (int n = 0; n < 5; ++n) {
    g_asym_sub += grad_V(q, sigma, b_small, N, ave_small_batch[n]);
  }
  
  std::cout << "Aymmetric subsample: " << g_full << "\t" << g_asym_sub / 5.0 << std::endl;
  
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 500; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[rand_int()], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  output.close();
  
  output.open("small_batch_asym_subsample_scaled_step1_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 500; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[rand_int()], epsilon / 5.0);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  output.close();
  
  output.open("small_batch_asym_subsample_scaled_step2_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 500; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[rand_int()], epsilon / 10.0);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  output.close();
  
  output.open("small_batch_asym_subsample_scaled_step3_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 5000; ++n) {
    evolve(q, p, sigma, b_small, N, ave_small_batch[rand_int()], epsilon / 100.0);
    output << q << "\t" << p << "\t" << std::endl;
  }
  
  output.close();
  
  output.open("large_batch_sym_trajectory.dat");
  
  q = q0 + std::sqrt(6);
  p = 0;
  
  output << q << "\t" << p << "\t" << std::endl;
  
  for (int n = 0; n < 15; ++n) {
    evolve(q, p, sigma, b_large, N, ave_large_batch[0], epsilon);
    output << q << "\t" << p << "\t" << std::endl;

    evolve(q, p, sigma, b_large, N, ave_large_batch[1], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
    
    evolve(q, p, sigma, b_large, N, ave_large_batch[1], epsilon);
    output << q << "\t" << p << "\t" << std::endl;

    evolve(q, p, sigma, b_large, N, ave_large_batch[0], epsilon);
    output << q << "\t" << p << "\t" << std::endl;
    
  }
  
  output.close();
  
  return 0;
  
}




