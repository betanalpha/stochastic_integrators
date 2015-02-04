#include <iostream>
#include <fstream>

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include <boost/random/additive_combine.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>

inline double V(Eigen::VectorXd& q, double sigma, double B, double N, const Eigen::VectorXd& ave) {
  double m = 0;
  double s = 1;
  double V = 0;
  
  Eigen::VectorXd mu(q.size());
  for (int d = 0; d < q.size(); ++d) mu(d) = (N * s * s * ave(d) + m * sigma * sigma) / (sigma * sigma + N * s * s);
  double tau_inv = (sigma * sigma + N * s * s) / (sigma * sigma * s * s);
  return 0.5 * tau_inv * (q - mu).squaredNorm();
}

inline Eigen::VectorXd grad_V(Eigen::VectorXd& q, double sigma, double B, double N, const Eigen::VectorXd ave) {
  double m = 0;
  double s = 1;
  
  Eigen::VectorXd mu(q.size());
  for (int d = 0; d < q.size(); ++d) mu(d) = (N * s * s * ave(d) + m * sigma * sigma) / (sigma * sigma + N * s * s);
  double tau_inv = (sigma * sigma + N * s * s) / (sigma * sigma * s * s);
  
  return tau_inv * (q - mu);
}

inline void evolve(Eigen::VectorXd& q, Eigen::VectorXd& p,
                   double sigma, double B, double N, const Eigen::VectorXd& ave,
                   double epsilon) {
  p -= 0.5 * epsilon * grad_V(q, sigma, B, N, ave);
  q += epsilon * p;
  p -= 0.5 * epsilon * grad_V(q, sigma, B, N, ave);
}

int main(int argc, const char * argv[]) {

  // Set up the generators
  typedef boost::ecuyer1988 rng_t;
  rng_t rng(85738293);
  
  boost::variate_generator<rng_t&, boost::normal_distribution<> >
  rand_unit_gauss(rng, boost::normal_distribution<>());

  boost::variate_generator<rng_t&, boost::random::uniform_real_distribution<double> >
    rand_uniform(rng, boost::random::uniform_real_distribution<double>(0, 1));

  boost::variate_generator<rng_t&, boost::random::uniform_real_distribution<double> >
    rand_period(rng, boost::random::uniform_real_distribution<double>(0, 2 * 3.141592653589793));
  
  boost::variate_generator<rng_t&, boost::random::uniform_int_distribution<> >
    rand_subsample(rng, boost::random::uniform_int_distribution<>(0, 4));
 
  const int n_experiments = 7;
  int D[n_experiments] = {1, 5, 10, 50, 100, 500, 1000};
  double epsilon[n_experiments] = {0.14, 0.1, 0.08, 0.06, 0.05, 0.035, 0.0275};
  
  std::ofstream full_output;
  full_output.open("full_accept_probs.dat");
  
  std::ofstream subsampled_between_output;
  subsampled_between_output.open("subsampled_between_accept_probs.dat");
  
  std::ofstream subsampled_within_output;
  subsampled_within_output.open("subsampled_within_accept_probs.dat");
  
  for (int e = 0; e < n_experiments; ++e) {
  
    Eigen::VectorXd mu(D[e]);
    for (int d = 0; d < D[e]; ++d) mu(d) = rand_unit_gauss();
    
    double sigma = 2;
    
    // Generate pseudodata and aggregate sufficient statistics
    const int N = 500;
    const int n_small_batch = 25;
    const int b_small = N / n_small_batch;
    
    Eigen::VectorXd ave_full = Eigen::VectorXd::Zero(D[e]);
    Eigen::MatrixXd ave_small_batch = Eigen::MatrixXd::Zero(D[e], n_small_batch);
    
    for (int n = 0; n < N; ++n) {
      for (int d = 0; d < D[e]; ++d) {
        double x = mu(d) + sigma * rand_unit_gauss();
        ave_full(d) += x;
        ave_small_batch(d, n / b_small) += x;
      }
    }
    
    ave_full /= static_cast<double>(N);
    ave_small_batch /= static_cast<double>(b_small);
    
    double m = 0;
    double s = 1;

    Eigen::VectorXd q0(D[e]);
    for (int d = 0; d < D[e]; ++d) q0(d) = (N * s * s * ave_full(d) + m * sigma * sigma) / (sigma * sigma + N * s * s);
    
    Eigen::VectorXd q(D[e]);
    for (int d = 0; d < D[e]; ++d) q(d) = q0(d) + std::sqrt(6);
    
    // Warmup
    for (int n = 0; n < 500; ++n) {
      
      int L = rand_period() / epsilon[e];
      
      Eigen::VectorXd q_propose = q;
      
      Eigen::VectorXd p(D[e]);
      for (int d = 0; d < D[e]; ++d) p(d) = rand_unit_gauss();
      
      double H = 0.5 * p.squaredNorm() + V(q_propose, sigma, N, N, ave_full);
      
      for (int l = 0; l < L; ++l)
        evolve(q_propose, p, sigma, N, N, ave_full, epsilon[e]);
      
      double delta_H = 0.5 * p.squaredNorm() + V(q_propose, sigma, N, N, ave_full) - H;
      
      double accept_prob = std::exp(-delta_H);
      accept_prob = accept_prob > 1 ? 1 : accept_prob;
      
      if (rand_uniform() < accept_prob)
        q = q_propose;
      
    }
    
    Eigen::VectorXd q_init = q;
    
    // Sampling (Full)
    double ave_accept_prob = 0;
    double var_accept_prob = 0;
    
    int n_samples = 5000;
    
    for (int n = 0; n < n_samples; ++n) {
      
      int L = rand_period() / epsilon[e];
      
      Eigen::VectorXd q_propose = q;
      
      Eigen::VectorXd p(D[e]);
      for (int d = 0; d < D[e]; ++d) p(d) = rand_unit_gauss();
      
      double H = 0.5 * p.squaredNorm() + V(q_propose, sigma, N, N, ave_full);
      
      for (int l = 0; l < L; ++l)
        evolve(q_propose, p, sigma, N, N, ave_full, epsilon[e]);
      
      double delta_H = 0.5 * p.squaredNorm() + V(q_propose, sigma, N, N, ave_full) - H;
      
      double accept_prob = std::exp(-delta_H);
      accept_prob = accept_prob > 1 ? 1 : accept_prob;
      
      if (rand_uniform() < accept_prob)
        q = q_propose;
      
      double delta = accept_prob - ave_accept_prob;
      ave_accept_prob += delta / static_cast<double>(n + 1);
      var_accept_prob += delta * (accept_prob - ave_accept_prob);

    }
    
    var_accept_prob /= static_cast<double>(n_samples);

    std::cout << D[e] << std::endl;
    std::cout << "Full: " << ave_accept_prob << " +/- " << std::sqrt(var_accept_prob) << std::endl;

    full_output << D[e] << "\t" << ave_accept_prob << "\t" << std::sqrt(var_accept_prob) << std::endl;

    // Sampling (Subsampled Between Trajectories)
    q = q_init;
    
    ave_accept_prob = 0;
    var_accept_prob = 0;
    
    for (int n = 0; n < n_samples; ++n) {
      
      int L = rand_period() / (epsilon[e] / 5);
      
      Eigen::VectorXd q_propose = q;
      
      Eigen::VectorXd p(D[e]);
      for (int d = 0; d < D[e]; ++d) p(d) = rand_unit_gauss();
      
      double H = 0.5 * p.squaredNorm() + V(q_propose, sigma, N, N, ave_full);
      
      int subsample = rand_subsample();
      for (int l = 0; l < L; ++l)
        evolve(q_propose, p, sigma, b_small, N, ave_small_batch.col(subsample), epsilon[e] / 5);
   
      double delta_H = 0.5 * p.squaredNorm() + V(q_propose, sigma, b_small, N, ave_full) - H;
      
      double accept_prob = std::exp(-delta_H);
      accept_prob = accept_prob > 1 ? 1 : accept_prob;
      
      if (rand_uniform() < accept_prob)
        q = q_propose;
      
      double delta = accept_prob - ave_accept_prob;
      ave_accept_prob += delta / static_cast<double>(n + 1);
      var_accept_prob += delta * (accept_prob - ave_accept_prob);
      
    }
    
    var_accept_prob /= static_cast<double>(n_samples);
    
    std::cout << "Subsampled Between: " << ave_accept_prob << " +/- " << std::sqrt(var_accept_prob) << std::endl;
    
    subsampled_between_output << D[e] << "\t" << ave_accept_prob << "\t" << std::sqrt(var_accept_prob) << std::endl;
    
    // Sampling (Subsampled Within Trajectories)
    q = q_init;
    
    ave_accept_prob = 0;
    var_accept_prob = 0;
    
    for (int n = 0; n < n_samples; ++n) {
      
      int L = rand_period() / (epsilon[e] / 5);
      
      Eigen::VectorXd q_propose = q;
      
      Eigen::VectorXd p(D[e]);
      for (int d = 0; d < D[e]; ++d) p(d) = rand_unit_gauss();
      
      double H = 0.5 * p.squaredNorm() + V(q_propose, sigma, N, N, ave_full);
      
      for (int l = 0; l < L; ++l) {
        evolve(q_propose, p, sigma, b_small, N, ave_small_batch.col(rand_subsample()), epsilon[e] / 5);
      }
      double delta_H = 0.5 * p.squaredNorm() + V(q_propose, sigma, b_small, N, ave_full) - H;
      
      double accept_prob = std::exp(-delta_H);
      accept_prob = accept_prob > 1 ? 1 : accept_prob;
      
      if (rand_uniform() < accept_prob)
        q = q_propose;
      
      double delta = accept_prob - ave_accept_prob;
      ave_accept_prob += delta / static_cast<double>(n + 1);
      var_accept_prob += delta * (accept_prob - ave_accept_prob);
      
    }
    
    var_accept_prob /= static_cast<double>(n_samples);

    std::cout << "Subsampled Within: " << ave_accept_prob << " +/- " << std::sqrt(var_accept_prob) << std::endl;
    std::cout << std::endl;
    
    subsampled_within_output << D[e] << "\t" << ave_accept_prob << "\t" << std::sqrt(var_accept_prob) << std::endl;

  }
  
  full_output.close();
  subsampled_between_output.close();
  subsampled_within_output.close();
  
  return 0;
  
}




