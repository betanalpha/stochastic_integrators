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
    uniform_two_pi(0, 2 * 3.141592653589793);
  
  boost::variate_generator<rng_t&, boost::random::uniform_real_distribution<double> >
  rand_period(rng, uniform_two_pi);
  
  boost::random::uniform_real_distribution<double>
  uniform_unit(0, 1);
  
  boost::variate_generator<rng_t&, boost::random::uniform_real_distribution<double> >
  rand_unit(rng, uniform_unit);
  
  //(Subsampled Expectation - Expectation of Full Data) vs Subsample Size
 
  double mu = 1;
  double sigma = 2;
  
  const int B = 5;
  int batch_size[B] = {20, 200, 2000, 20000, 200000};
  double epsilon[B] = {0.15, 0.05, 0.015, 0.005, 0.0015};
  
  const int n_small_batch = 25;
  double ave_small_batch[n_small_batch];
  
  boost::random::uniform_int_distribution<> uniform_small(0, 4);
  boost::variate_generator<rng_t&, boost::random::uniform_int_distribution<> >
  rand_small(rng, uniform_small);
  
  boost::random::uniform_int_distribution<> uniform_full(0, n_small_batch - 1);
  boost::variate_generator<rng_t&, boost::random::uniform_int_distribution<> >
  rand_full(rng, uniform_full);
  
  std::ofstream output;
  output.open("subsample_expectations.dat");

  for (int b = 0; b < B; ++b) {
    
    int N = 25 * batch_size[b];
    
    double ave_full = 0;
    for (int n = 0; n < n_small_batch; ++n) ave_small_batch[n] = 0;
    
    for (int n = 0; n < N; ++n) {
      double x = mu + sigma * rand_unit_gauss();
      ave_full += x;
      ave_small_batch[n / batch_size[b]] += x;
    }
    
    ave_full /= static_cast<double>(N);
    for (int n = 0; n < n_small_batch; ++n) ave_small_batch[n] /= static_cast<double>(batch_size[b]);
    
    std::cout << b << std::endl;
    //std::cout << ave_full << "\t" << ave_small_batch[0] << std::endl;
    
    double m = 0;
    double s = 1;
    double q0 = (ave_full * N * s * s + m * sigma * sigma) / (sigma * sigma + N * s * s);
    
    double q;
    
    //
    //  Single Subsample
    //
    
    q = q0 + std::sqrt(6);
    
    // Warmup
    for (int n = 0; n < 500; ++n) {
    
      int L = rand_period() / epsilon[b];
      
      double q_propose = q;
      double p_propose = rand_unit_gauss();
      double H = 0.5 * p_propose * p_propose + V(q_propose, sigma, batch_size[b], N, ave_small_batch[0]);
      
      for (int l = 0; l < L; ++l)
        evolve(q_propose, p_propose, sigma, batch_size[b], N, ave_small_batch[0], epsilon[b]);

      double delta_H = 0.5 * p_propose * p_propose
                       + V(q_propose, sigma, batch_size[b], N, ave_small_batch[0]) - H;
      
      double accept_prob = std::exp(-delta_H);
      accept_prob = accept_prob > 1 ? 1 : accept_prob;
      
      if (rand_unit() < accept_prob)
        q = q_propose;
      
    }
    
    // Sampling
    double single_mean = 0;
    double ave_accept = 0;
    
    for (int n = 0; n < 5000; ++n) {
      
      int L = rand_period() / epsilon[b];
      
      double q_propose = q;
      double p_propose = rand_unit_gauss();
      double H = 0.5 * p_propose * p_propose + V(q_propose, sigma, batch_size[b], N, ave_small_batch[0]);
      
      for (int l = 0; l < L; ++l)
        evolve(q_propose, p_propose, sigma, batch_size[b], N, ave_small_batch[0], epsilon[b]);
      
      double delta_H = 0.5 * p_propose * p_propose
                       + V(q_propose, sigma, batch_size[b], N, ave_small_batch[0]) - H;
      
      double accept_prob = std::exp(-delta_H);
      accept_prob = accept_prob > 1 ? 1 : accept_prob;
      
      if (rand_unit() < accept_prob)
        q = q_propose;
      
      single_mean += 0.0002 * q;
      ave_accept += 0.0002 * accept_prob;
      
    }

    //
    //  Stochastic From Five Subsamples
    //
    
    q = q0 + std::sqrt(6);
    
    // Warmup
    for (int n = 0; n < 500; ++n) {
      int L = rand_period() / epsilon[b];
      double p = rand_unit_gauss();
      for (int l = 0; l < L; ++l)
        evolve(q, p, sigma, batch_size[b], N, ave_small_batch[rand_small()], epsilon[b]);
    }
    
    // Sampling
    double sub_five_mean = 0;
    
    for (int n = 0; n < 5000; ++n) {
      int L = rand_period() / epsilon[b];
      double p = rand_unit_gauss();
      
      for (int l = 0; l < L; ++l)
        evolve(q, p, sigma, batch_size[b], N, ave_small_batch[rand_small()], epsilon[b]);
      
      sub_five_mean += 0.0002 * q;
      
    }
    
    //
    //  Stochastic From All Subsamples
    //
    
    q = q0 + std::sqrt(6);
    
    // Warmup
    for (int n = 0; n < 500; ++n) {
      int L = rand_period() / epsilon[b];
      double p = rand_unit_gauss();
      for (int l = 0; l < L; ++l)
        evolve(q, p, sigma, batch_size[b], N, ave_small_batch[rand_full()], epsilon[b]);
    }
    
    // Sampling
    double sub_full_mean = 0;
    
    for (int n = 0; n < 5000; ++n) {
      int L = rand_period() / epsilon[b];
      double p = rand_unit_gauss();
      
      for (int l = 0; l < L; ++l)
        evolve(q, p, sigma, batch_size[b], N, ave_small_batch[rand_full()], epsilon[b]);
      
      sub_full_mean += 0.0002 * q;
      
    }

    //std::cout << "ave_accept = " << ave_accept << std::endl;
    std::cout << q0 - single_mean << std::endl;
    std::cout << q0 - sub_five_mean << std::endl;
    std::cout << q0 - sub_full_mean << std::endl;
    std::cout << std::endl;
    
    output << q0 - single_mean << "\t" << q0 - sub_five_mean << "\t" << q0 - sub_full_mean << std::endl;
    
  }

  output.close();
  
  return 0;
  
}




