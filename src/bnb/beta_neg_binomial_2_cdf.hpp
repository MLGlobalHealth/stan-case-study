#ifndef STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_2_CDF_HPP
#define STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_2_CDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/beta.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/hypergeometric_3F2.hpp>
#include <stan/math/prim/fun/grad_F32.hpp>
#include <stan/math/prim/fun/lbeta.hpp>
#include <stan/math/prim/fun/log.hpp>
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
 * Returns the CDF of the Beta-Negative Binomial distribution with given 
 * number of failures, location parameter, tail parameter and dispersion parameter. 
 * Given containers of matching sizes, returns the sum of probabilities.
 *
 * @tparam T_n type of number of failures
 * @tparam T_mu type of location parameter
 * @tparam T_sigma type of tail parameter
 * @tparam T_gamma type of dispersion parameter
 *
 * @param n number of failures
 * @param mu location parameter
 * @param sigma tail parameter
 * @param gamma dispersion parameter
 * @return probability or sum of probabilities
 * @throw std::domain_error if mu, sigma, or gamma fails to be positive
 * @throw std::invalid_argument if container sizes mismatch
 */
template <typename T_n, typename T_mu, typename T_sigma, typename T_gamma>
inline stan::return_type_t<T_mu, T_sigma, T_gamma> beta_neg_binomial_2_cdf(const T_n& n, 
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
  using stan::math::size;
  using stan::math::max_size;
  using namespace stan::math;

  using T_partials_return = partials_return_t<T_n, T_mu, T_sigma, T_gamma>;
  using std::exp;
  using std::log;
  using T_mu_ref = ref_type_t<T_mu>;
  using T_sigma_ref = ref_type_t<T_sigma>;
  using T_gamma_ref = ref_type_t<T_gamma>;
  static const char* function = "beta_neg_binomial_2_cdf";
  check_consistent_sizes(function, "Number of failures", n,
                         "Location parameter", mu,
                         "Tail parameter", sigma,
                         "Dispersion parameter", gamma);
  if (size_zero(n, mu, sigma, gamma)) {
    return 0;
  }

  T_mu_ref mu_ref = mu;
  T_sigma_ref sigma_ref = sigma;
  T_gamma_ref gamma_ref = gamma;
  check_positive_finite(function, "Location parameter", mu_ref);
  check_positive_finite(function, "Tail parameter", sigma_ref);
  check_positive_finite(function, "Dispersion parameter", gamma_ref);

  T_partials_return P(0.0);
  operands_and_partials<T_mu_ref, T_sigma_ref, T_gamma_ref> ops_partials(mu_ref, sigma_ref, gamma_ref);

  scalar_seq_view<T_n> n_vec(n);
  scalar_seq_view<T_mu_ref> mu_vec(mu_ref);
  scalar_seq_view<T_sigma_ref> sigma_vec(sigma_ref);
  scalar_seq_view<T_gamma_ref> gamma_vec(gamma_ref);
  size_t size_n = stan::math::size(n);
  size_t size_mu = stan::math::size(mu);
  size_t size_sigma = stan::math::size(sigma);
  size_t size_gamma = stan::math::size(gamma);
  size_t max_size_mu_sigma_gamma = max_size(mu, sigma, gamma);
  size_t max_size_seq_view = max_size(n, mu, sigma, gamma);

  // Explicit return for extreme values
  // The gradients are technically ill-defined, but treated as neg infinity
  for (size_t i = 0; i < size_n; i++) {
    if (n_vec.val(i) < 0) {
      return ops_partials.build(negative_infinity());
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

  for (size_t i = 0; i < max_size_seq_view; i++) {
    const T_partials_return n_dbl = n_vec.val(i);
    const T_partials_return r_dbl = r[i];
    const T_partials_return alpha_dbl = alpha[i];
    const T_partials_return beta_dbl = beta[i];
    const T_partials_return b_plus_n = beta_dbl + n_dbl;
    const T_partials_return r_plus_n = r_dbl + n_dbl;
    const T_partials_return a_plus_r = alpha_dbl + r_dbl;
    const T_partials_return one = 1;

    const T_partials_return F = hypergeometric_3F2({one, b_plus_n + 1, r_plus_n + 1},
                                                   {n_dbl + 2, a_plus_r + b_plus_n + 1}, one);
    const T_partials_return C = lgamma(r_plus_n + 1) + lbeta(a_plus_r, b_plus_n + 1)
                          - lgamma(r_dbl) - lbeta(alpha_dbl, beta_dbl) - lgamma(n_dbl + 2);
    const T_partials_return P_i = exp(C) * F; //ccdf
    P += one - P_i;


    T_partials_return digamma_abrn
        = is_constant_all<T_mu, T_sigma, T_gamma>::value
              ? 0
              : digamma(a_plus_r + b_plus_n + 1);
    T_partials_return digamma_ab
        = is_constant_all<T_sigma, T_gamma>::value
              ? 0
              : digamma(alpha_dbl + beta_dbl);
    T_partials_return digamma_ar
        = is_constant_all<T_mu, T_sigma>::value
              ? 0
              : digamma(a_plus_r);

    T_partials_return dF[6];
    if (!is_constant_all<T_mu, T_sigma, T_gamma>::value) {
      grad_F32(dF, one, b_plus_n + 1, r_plus_n + 1, n_dbl + 2, a_plus_r + b_plus_n + 1, one, 1e-3);
    }

    // y2r = d(y)/d(r)
    T_partials_return y2r
        = is_constant_all<T_mu, T_sigma, T_gamma>::value
              ? 0
              : digamma(r_plus_n + 1) + (digamma_ar - digamma_abrn) + (dF[2] + dF[4]) / F - digamma(r_dbl);

    // y2alpha = d(y)/d(alpha)
    T_partials_return y2alpha
        = is_constant_all<T_sigma>::value
              ? 0
              : digamma(a_plus_r) - digamma_abrn + dF[4] / F - digamma(alpha_dbl) + digamma_ab;
    
    // y2beta = d(y)/d(beta)
    T_partials_return y2beta
        = is_constant_all<T_gamma>::value
              ? 0
              : digamma(b_plus_n + 1) - digamma_abrn + (dF[1] + dF[4]) / F - digamma(beta_dbl) + digamma_ab;


    if (!is_constant_all<T_mu>::value) {
      T_partials_return r2mu = (nu[i] + gamma_vec.val(i)) / (2*sigma_vec.val(i));
      T_partials_return beta2mu = sqrt(sigma_vec.val(i)/mu_vec.val(i)) / (2*mu_vec.val(i)*nu[i]*nu[i]);
      T_partials_return g = y2r * r2mu + y2beta * beta2mu;
      printf("edge1: %f ", -g * P_i);
      ops_partials.edge1_.partials_[i] += -g * P_i;
    }
    if (!is_constant_all<T_sigma>::value) {
      T_partials_return r2sigma = -mu_vec.val(i)*(nu[i] + gamma_vec.val(i)) / (2*sigma_vec.val(i)*sigma_vec.val(i));
      T_partials_return alpha2sigma = -1 / (sigma_vec.val(i)*sigma_vec.val(i));
      T_partials_return beta2sigma = -sqrt(sigma_vec.val(i)/mu_vec.val(i)) / (2*sigma_vec.val(i)*nu[i]*nu[i]);
      T_partials_return g = y2r * r2sigma + y2alpha * alpha2sigma + y2beta * beta2sigma;
      printf("edge2: %f ", -g * P_i);
      ops_partials.edge2_.partials_[i] += -g * P_i;
    }
    if (!is_constant_all<T_gamma>::value) {
      T_partials_return r2gamma = mu_vec.val(i) / sigma_vec.val(i);
      T_partials_return beta2gamma = -1 / (nu[i] * nu[i]);
      T_partials_return g = y2r * r2gamma + y2beta * beta2gamma;
      printf("edge3: %f \n", -g * P_i);
      ops_partials.edge3_.partials_[i] += -g * P_i;
    }
  }

  return ops_partials.build(P);
}

}
#endif
