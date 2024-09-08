#ifndef STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_2_LPMF_HPP
#define STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_2_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/lbeta.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/operands_and_partials.hpp>
#include <cmath>
#include <stan/math.hpp>


namespace example_model_namespace {

/** \ingroup prob_dists
 * Returns the log PMF of the Beta-Negative Binomial distribution with given 
 * number of observations, location parameter, tail parameter and dispersion parameter. 
 * Given containers of matching sizes, returns the log sum of probabilities.
 *
 * @tparam T_n type of number of observations
 * @tparam T_mu type of location parameter
 * @tparam T_sigma type of tail parameter
 * @tparam T_gamma type of dispersion parameter
 * 
 * @param n number of observations
 * @param mu location parameter
 * @param sigma tail parameter
 * @param gamma dispersion parameter
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if mu, sigma, or gamma fails to be positive
 * @throw std::invalid_argument if container sizes mismatch
 */
template <bool propto, typename T_n, typename T_mu, typename T_sigma,
          typename T_gamma,
          stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_n, T_mu, T_sigma, T_gamma>* = nullptr>
inline stan::return_type_t<T_mu, T_sigma, T_gamma> beta_neg_binomial_2_lpmf(const T_n& n, 
                                                    const T_mu& mu,
                                                    const T_sigma& sigma,
                                                    const T_gamma& gamma,
                                                    std::ostream* pstream__) {

  using stan::partials_return_t;
  using stan::ref_type_t;
  
  using stan::is_constant;
  using stan::is_constant_all;
  using stan::VectorBuilder;
  using stan::scalar_seq_view;
  using stan::math::lgamma;
  using stan::math::lbeta;
  using stan::math::size;
  using stan::math::max_size;
  using namespace stan::math;

  using T_partials_return = partials_return_t<T_n, T_mu, T_sigma, T_gamma>;
  using T_mu_ref = ref_type_t<T_mu>;
  using T_sigma_ref = ref_type_t<T_sigma>;
  using T_gamma_ref = ref_type_t<T_gamma>;
  static const char* function = "beta_neg_binomial_2_lpmf";
  check_consistent_sizes(function, "Number of failures", n,
                         "Location parameter", mu,
                         "Tail parameter", sigma,
                         "Dispersion parameter", gamma);
  if (size_zero(n, mu, sigma, gamma)) {
    return 0.0;
  }

  T_mu_ref mu_ref = mu;
  T_sigma_ref sigma_ref = sigma;
  T_gamma_ref gamma_ref = gamma;
  check_positive_finite(function, "Location parameter", mu_ref);
  check_positive_finite(function, "Tail parameter", sigma_ref);
  check_positive_finite(function, "Dispersion parameter", gamma_ref);

  if (!include_summand<propto, T_mu, T_sigma, T_gamma>::value) {
    return 0.0;
  }

  T_partials_return logp(0.0);
  operands_and_partials<T_mu_ref, T_sigma_ref, T_gamma_ref> ops_partials(mu_ref, sigma_ref, gamma_ref);

  scalar_seq_view<T_n> n_vec(n);
  scalar_seq_view<T_mu_ref> mu_vec(mu_ref);
  scalar_seq_view<T_sigma_ref> sigma_vec(sigma_ref);
  scalar_seq_view<T_gamma_ref> gamma_vec(gamma_ref);
  size_t size_n = stan::math::size(n);
  size_t size_mu = stan::math::size(mu);
  size_t size_sigma = stan::math::size(sigma);
  size_t size_gamma = stan::math::size(gamma);
  size_t size_n_mu = max_size(n, mu);
  size_t size_n_gamma = max_size(n, gamma);
  size_t size_mu_sigma = max_size(mu, sigma);
  size_t size_mu_gamma = max_size(mu, gamma);
  size_t size_sigma_gamma = max_size(sigma, gamma);
  size_t max_size_mu_sigma_gamma = max_size(mu, sigma, gamma);
  size_t max_size_seq_view = max_size(n, mu, sigma, gamma);

  // Determine whether the support is valid one by one
  for (size_t i = 0; i < max_size_seq_view; i++) {
    if (n_vec[i] < 0) {
      return ops_partials.build(LOG_ZERO);
    }
  }


  VectorBuilder<true, T_partials_return, T_mu, T_sigma, T_gamma> nu(max_size_mu_sigma_gamma);
  for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
    nu[i] = gamma_vec.val(i) + sqrt(sigma_vec.val(i)/mu_vec.val(i)); //temporary variable
  }

  VectorBuilder<true, T_partials_return, T_mu, T_sigma, T_gamma> r(max_size_mu_sigma_gamma);
  for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
    r[i] = nu[i] * mu_vec.val(i) / sigma_vec.val(i);
  }
  VectorBuilder<true, T_partials_return, T_sigma> alpha(size_sigma);
  for (size_t i = 0; i < size_sigma; i++) {
    alpha[i] = 1 + 1 / sigma_vec.val(i);
  }
  VectorBuilder<true, T_partials_return, T_mu, T_sigma, T_gamma> beta(max_size_mu_sigma_gamma);
  for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
    beta[i] = 1 / nu[i];
  }


  // Compute the lpmf return value
  // compute gamma(n+1)
  VectorBuilder<include_summand<propto>::value, T_partials_return, T_n>
      normalizing_constant(size_n);
  for (size_t i = 0; i < size_n; i++)
    if (include_summand<propto>::value)
      normalizing_constant[i] = -lgamma(n_vec[i] + 1);

  // compute lbeta denominator with size r and alpha
  VectorBuilder<true, T_partials_return, T_mu, T_sigma, T_gamma> lbeta_denominator(max_size_mu_sigma_gamma);
  for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
    lbeta_denominator[i] = lbeta(r[i], alpha[i]);
  }

  // compute lgamma denominator with size beta
  VectorBuilder<true, T_partials_return, T_mu, T_sigma, T_gamma> lgamma_denominator(max_size_mu_sigma_gamma);
  for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
    lgamma_denominator[i] = lgamma(beta[i]);
  }

  // compute lgamma numerator with size n and beta
  VectorBuilder<true, T_partials_return, T_n, T_mu, T_sigma, T_gamma> lgamma_numerator(max_size_seq_view);
  for (size_t i = 0; i < max_size_seq_view; i++) {
    lgamma_numerator[i] = lgamma(n_vec[i] + beta[i]);
  }

  // compute lbeta numerator with size n and r, alpha, beta
  VectorBuilder<true, T_partials_return, T_n, T_mu, T_sigma, T_gamma> lbeta_diff(max_size_seq_view);
  for (size_t i = 0; i < max_size_seq_view; i++) {
    lbeta_diff[i] = lbeta(n_vec[i] + r[i], alpha[i] + beta[i])
                    + lgamma_numerator[i]
                    - lbeta_denominator[i] - lgamma_denominator[i];
  }





  // Compute the Jacobian y = f(r, alpha, beta)

  // compute digamma(n+r+alpha+beta)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return,
                T_n, T_mu, T_sigma, T_gamma>
      digamma_n_r_alpha_beta(max_size_seq_view);
  if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
    for (size_t i = 0; i < max_size_seq_view; i++) {
      digamma_n_r_alpha_beta[i] = digamma(n_vec[i] + r[i] + alpha[i] + beta[i]);
    }
  }

  // compute digamma(n+r)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return,
                T_n, T_mu, T_sigma, T_gamma>
      digamma_n_r(max_size_seq_view);
  if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
    for (size_t i = 0; i < max_size_seq_view; i++) {
      digamma_n_r[i] = digamma(n_vec[i] + r[i]);
    }
  }

  // compute digamma(n+beta)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return,
                T_n, T_mu, T_sigma, T_gamma>
      digamma_n_beta(max_size_seq_view);
  if (!is_constant_all<T_gamma>::value) {
    for (size_t i = 0; i < max_size_seq_view; i++) {
      digamma_n_beta[i] = digamma(n_vec[i] + beta[i]);
    }
  }

  // compute digamma(alpha+r)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return, 
                T_mu, T_sigma, T_gamma>
      digamma_r_alpha(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      digamma_r_alpha[i] = digamma(r[i] + alpha[i]);
    }
  }

  // compute digamma(alpha+beta)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return, 
                T_mu, T_sigma, T_gamma>
      digamma_alpha_beta(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      digamma_alpha_beta[i] = digamma(alpha[i] + beta[i]);
    }
  }

  // compute digamma(r)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return,
                T_mu, T_sigma, T_gamma> 
      digamma_r(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      digamma_r[i] = digamma(r[i]);
    }
  }

  // compute digamma(alpha)
  VectorBuilder<!is_constant_all<T_sigma>::value, T_partials_return, T_sigma> 
      digamma_alpha(size_sigma);
  if (!is_constant_all<T_sigma>::value) {
    for (size_t i = 0; i < size_sigma; i++) {
      digamma_alpha[i] = digamma(alpha[i]);
    }
  }

  // compute digamma(gamma)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return,
                T_mu, T_sigma, T_gamma>
      digamma_beta(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_gamma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      digamma_beta[i] = digamma(beta[i]);
    }
  }


  // Combine the digamma terms

  // y2r = d(y)/d(r)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return,
                T_n, T_mu, T_sigma, T_gamma>
      y2r(max_size_seq_view);
  if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
    for (size_t i = 0; i < max_size_seq_view; i++) {
      y2r[i] = digamma_n_r[i] - digamma_n_r_alpha_beta[i] - digamma_r[i] + digamma_r_alpha[i];
    }
  }

  // y2alpha = d(y)/d(alpha)
  VectorBuilder<!is_constant_all<T_sigma>::value, T_partials_return,
                T_n, T_mu, T_sigma, T_gamma>
      y2alpha(max_size_seq_view);
  if (!is_constant_all<T_sigma>::value) {
    for (size_t i = 0; i < max_size_seq_view; i++) {
      y2alpha[i] = digamma_alpha_beta[i] - digamma_n_r_alpha_beta[i] - digamma_alpha[i] + digamma_r_alpha[i];
    }
  }

  // y2beta = d(y)/d(beta)
  VectorBuilder<!is_constant_all<T_mu, T_sigma, T_gamma>::value, T_partials_return,
                T_n, T_mu, T_sigma, T_gamma>
      y2beta(max_size_seq_view);
  if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
    for (size_t i = 0; i < max_size_seq_view; i++) {
      y2beta[i] = digamma_alpha_beta[i] - digamma_n_r_alpha_beta[i] + digamma_n_beta[i] - digamma_beta[i];
    }
  }



  // Compute the Jacobian (r, alpha, beta) = F(mu, sigma, gamma)

  // r2mu = d(r)/d(mu)
  VectorBuilder<!is_constant_all<T_mu>::value, T_partials_return, T_mu, T_sigma, T_gamma> 
      r2mu(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_mu>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      r2mu[i] = (nu[i] + gamma_vec.val(i)) / (2*sigma_vec.val(i));
    }
  }

  // r2sigma = d(r)/d(sigma)
  VectorBuilder<!is_constant_all<T_sigma>::value, T_partials_return, T_mu, T_sigma, T_gamma> 
      r2sigma(max_size_mu_sigma_gamma);
  if ( !is_constant_all<T_sigma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      r2sigma[i] = -mu_vec.val(i)*(nu[i] + gamma_vec.val(i)) / (2*sigma_vec.val(i)*sigma_vec.val(i));
    }
  }

  // r2gamma = d(r)/d(gamma)
  VectorBuilder<!is_constant_all<T_gamma>::value, T_partials_return, T_mu, T_sigma, T_gamma> 
      r2gamma(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_gamma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      r2gamma[i] = mu_vec.val(i) / sigma_vec.val(i);
    }
  }


  // alpha2sigma = d(alpha)/d(sigma)
  VectorBuilder<!is_constant_all<T_sigma>::value, T_partials_return, T_sigma> 
      alpha2sigma(size_sigma);
  if (!is_constant_all<T_sigma>::value) {
    for (size_t i = 0; i < size_sigma; i++) {
      alpha2sigma[i] = -1 / (sigma_vec.val(i)*sigma_vec.val(i));
    }
  }


  // beta2mu = d(beta)/d(mu)
  VectorBuilder<!is_constant_all<T_mu>::value, T_partials_return, T_mu, T_sigma, T_gamma> 
      beta2mu(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_mu>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      beta2mu[i] = sqrt(sigma_vec.val(i)/mu_vec.val(i)) / (2*mu_vec.val(i)*nu[i]*nu[i]);
    }
  }

  // beta2sigma = d(beta)/d(sigma)
  VectorBuilder<!is_constant_all<T_sigma>::value, T_partials_return, T_mu, T_sigma, T_gamma> 
      beta2sigma(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_sigma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      beta2sigma[i] = -sqrt(sigma_vec.val(i)/mu_vec.val(i)) / (2*sigma_vec.val(i)*nu[i]*nu[i]);
    }
  }

  // beta2gamma = d(beta)/d(gamma)
  VectorBuilder<!is_constant_all<T_gamma>::value, T_partials_return, T_mu, T_sigma, T_gamma> 
      beta2gamma(max_size_mu_sigma_gamma);
  if (!is_constant_all<T_gamma>::value) {
    for (size_t i = 0; i < max_size_mu_sigma_gamma; i++) {
      beta2gamma[i] = -1 / (nu[i] * nu[i]);
    }
  }

  for (size_t i = 0; i < max_size_seq_view; i++) {
    if (include_summand<propto>::value)
      logp += normalizing_constant[i];
    logp += lbeta_diff[i];

    if (!is_constant_all<T_mu>::value)
      ops_partials.edge1_.partials_[i]
          += y2r[i] * r2mu[i] + y2beta[i] * beta2mu[i];
    if (!is_constant_all<T_sigma>::value)
      ops_partials.edge2_.partials_[i]
          += y2r[i] * r2sigma[i] + y2alpha[i] * alpha2sigma[i] + y2beta[i] * beta2sigma[i];
    if (!is_constant_all<T_gamma>::value)
      ops_partials.edge3_.partials_[i]
          += y2r[i] * r2gamma[i] + y2beta[i] * beta2gamma[i];
  }
  return ops_partials.build(logp);
}

template <typename T_n, typename T_mu, typename T_sigma, typename T_gamma>
inline stan::return_type_t<T_mu, T_sigma, T_gamma> beta_neg_binomial_2_lpmf(const T_n& n, 
                                                   const T_mu& mu,
                                                   const T_sigma& sigma,
                                                   const T_gamma& gamma,
                                                   std::ostream* pstream__) {
  return beta_neg_binomial_2_lpmf<false>(n, mu, sigma, gamma);
}








}
#endif
