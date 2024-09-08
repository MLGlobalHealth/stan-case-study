#ifndef STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_2_RNG_HPP
#define STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_2_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/prob/beta_rng.hpp>
// #include <stan/math/prim/prob/neg_binomial_rng.hpp>
#include <limits>

namespace example_model_namespace {

/** \ingroup prob_dists
 * Return a beta-negative binomial random variate with given number of successes,
 * success, and failure parameters using the given random number generator.
 *
 * r, sigma, and beta can each be a scalar or a one-dimensional container. Any
 * non-scalar inputs must be the same size.
 *
 * @tparam T_mu type of mean parameter
 * @tparam T_sigma type of tail parameter
 * @tparam T_gamma type of dispersion parameter
 * @tparam RNG type of random number generator
 * 
 * @param mu (Sequence of) mean parameter(s)
 * @param sigma (Sequence of) tail parameter(s)
 * @param gamma (Sequence of) dispersion parameter(s)
 * @param rng random number generator
 * @return (Sequence of) beta-binomial random variate(s)
 * @throw std::domain_error if any of mu, sigma or gamma are nonpositive
 * @throw std::invalid_argument if non-scalar arguments are of different sizes
 */
template <typename T_mu, typename T_sigma, typename T_gamma, class RNG>
inline typename stan::VectorBuilder<true, int, T_mu, T_sigma, T_gamma>::type
beta_neg_binomial_2_rng(const T_mu &mu, const T_sigma &sigma, const T_gamma &gamma,
                  RNG &rng, std::ostream* pstream__) {

  using stan::ref_type_t;
  using stan::VectorBuilder;
  using stan::scalar_seq_view;
  using stan::math::is_inf;
  using stan::math::size;
  using namespace stan::math;

  using T_mu_ref = ref_type_t<T_mu>;
  using T_sigma_ref = ref_type_t<T_sigma>;
  using T_gamma_ref = ref_type_t<T_gamma>;
  static const char *function = "beta_neg_binomial_2_rng";
  check_consistent_sizes(function, "Mean parameter", mu,
                         "Tail parameter", sigma,
                         "Dispersion parameter", gamma);

  T_mu_ref mu_ref = mu;
  T_sigma_ref sigma_ref = sigma;
  T_gamma_ref gamma_ref = gamma;
  check_positive_finite(function, "Population size parameter", mu_ref);
  check_positive_finite(function, "First prior sample size parameter", sigma_ref);
  check_positive_finite(function, "Second prior sample size parameter", gamma_ref);

  // convert parameterization {mu, sigma, gamma} to {r, alpha, beta}
  scalar_seq_view<T_mu_ref> mu_vec(mu_ref);
  scalar_seq_view<T_sigma_ref> sigma_vec(sigma_ref);
  scalar_seq_view<T_gamma_ref> gamma_vec(gamma_ref);
  size_t size_sigma = stan::math::size(sigma);
  size_t max_size_seq_view = max_size(mu, sigma, gamma);

  VectorBuilder<true, double, T_mu, T_sigma, T_gamma> nu(max_size_seq_view);
  for (size_t n = 0; n < max_size_seq_view; n++) {
    nu[n] = gamma_vec.val(n) + sqrt(sigma_vec.val(n)/mu_vec.val(n)); //temporary variable
  }
  VectorBuilder<true, double, T_mu, T_sigma, T_gamma> r(max_size_seq_view);
  for (size_t n = 0; n < max_size_seq_view; ++n) {
    r[n] = (nu[n] * mu_vec.val(n)) / sigma_vec.val(n);
  }
  VectorBuilder<true, double, T_sigma> alpha(size_sigma);
  for (size_t n = 0; n < size_sigma; ++n) {
    alpha[n] = 1 + 1 / sigma_vec.val(n);
  }
  VectorBuilder<true, double, T_mu, T_sigma, T_gamma> beta(max_size_seq_view);
  for (size_t n = 0; n < max_size_seq_view; ++n) {
    beta[n] = 1 / nu[n];
  }

  // Compute the odds ratio
  using T_p = decltype(beta_rng(alpha.data(), beta.data(), rng));
  T_p p = beta_rng(alpha.data(), beta.data(), rng);

  scalar_seq_view<T_p> p_vec(p);
  size_t size_p = size(p);
  VectorBuilder<true, double, T_p> odds_ratio_p(size_p);
  for (size_t n = 0; n < size_p; ++n) {
    odds_ratio_p[n] = p_vec.val(n) / (1 - p_vec.val(n));
  }
  
  // Replace infinities and zeros
  double dbl_max_finite = std::numeric_limits<double>::max();
  double dbl_min_positive = std::numeric_limits<double>::min();
  for (size_t i = 0; i < size_p; ++i) {
    if (is_inf(odds_ratio_p[i])) {
      odds_ratio_p[i] = dbl_max_finite;
    } else if (odds_ratio_p[i] == 0.0) {
      odds_ratio_p[i] = dbl_min_positive;
    }
  }

  using boost::gamma_distribution;
  using boost::variate_generator;
  using boost::random::poisson_distribution;
  VectorBuilder<true, int, T_mu, T_sigma, T_gamma> output(max_size_seq_view);
  for (size_t n = 0; n < max_size_seq_view; ++n) {
    double rng_from_gamma = variate_generator<RNG&, gamma_distribution<> >(
        rng, gamma_distribution<>(r[n], 1.0 / odds_ratio_p[n]))();
    // const double POISSON_MAX_RATE = std::pow(2.0, 30);
    // max int = 2^31, should prevent output to be larger than it
    // The root cause is that stan only supports <int>
    if (rng_from_gamma > POISSON_MAX_RATE) {
      rng_from_gamma = POISSON_MAX_RATE;
    }
    output[n] = variate_generator<RNG&, poisson_distribution<> >(
        rng, poisson_distribution<>(rng_from_gamma))();
  }


  // return neg_binomial_rng(r.data(), odds_ratio_p.data(), rng);
  return output.data();
}



}
#endif
