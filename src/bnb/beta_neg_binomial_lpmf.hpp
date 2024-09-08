#ifndef STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_LPMF_HPP
#define STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/binomial_coefficient_log.hpp>
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
 * number of successes, prior success, and prior failure parameters. 
 * Given containers of matching sizes, returns the log sum of probabilities.
 *
 * @tparam T_n type of failure parameter
 * @tparam T_r type of number of successes parameter
 * @tparam T_size1 type of prior success parameter
 * @tparam T_size2 type of prior failure parameter
 * 
 * @param n failure parameter
 * @param r Number of successes parameter
 * @param alpha prior success parameter
 * @param beta prior failure parameter
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if r, alpha, or beta fails to be positive
 * @throw std::invalid_argument if container sizes mismatch
 */
template <bool propto, typename T_n, typename T_r, typename T_size1,
          typename T_size2,
          stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_n, T_r, T_size1, T_size2>* = nullptr>
inline stan::return_type_t<T_r, T_size1, T_size2> beta_neg_binomial_lpmf(const T_n& n, 
                                                    const T_r& r,
                                                    const T_size1& alpha,
                                                    const T_size2& beta,
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

  using T_partials_return = partials_return_t<T_n, T_r, T_size1, T_size2>;
  using T_r_ref = ref_type_t<T_r>;
  using T_alpha_ref = ref_type_t<T_size1>;
  using T_beta_ref = ref_type_t<T_size2>;
  static const char* function = "beta_neg_binomial_lpmf";
  check_consistent_sizes(function, "Successes variable", n,
                         "Number of successes parameter", r,
                         "First prior sample size parameter", alpha,
                         "Second prior sample size parameter", beta);
  if (size_zero(n, r, alpha, beta)) {
    return 0.0;
  }

  T_r_ref r_ref = r;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;
  check_positive_finite(function, "Number of successes parameter", r_ref);
  check_positive_finite(function, "First prior sample size parameter", alpha_ref);
  check_positive_finite(function, "Second prior sample size parameter", beta_ref);

  if (!include_summand<propto, T_r, T_size1, T_size2>::value) {
    return 0.0;
  }

  T_partials_return logp(0.0);
  operands_and_partials<T_r_ref, T_alpha_ref, T_beta_ref> ops_partials(r_ref, alpha_ref, beta_ref);

  scalar_seq_view<T_n> n_vec(n);
  scalar_seq_view<T_r_ref> r_vec(r_ref);
  scalar_seq_view<T_alpha_ref> alpha_vec(alpha_ref);
  scalar_seq_view<T_beta_ref> beta_vec(beta_ref);
  size_t size_n = stan::math::size(n);
  size_t size_r = stan::math::size(r);
  size_t size_alpha = stan::math::size(alpha);
  size_t size_beta = stan::math::size(beta);
  size_t size_n_r = max_size(n, r);
  size_t size_r_alpha = max_size(r, alpha);
  size_t size_n_beta = max_size(n, beta);
  size_t size_alpha_beta = max_size(alpha, beta);
  size_t max_size_seq_view = max_size(n, r, alpha, beta);

  // Determine whether the support is valid one by one
  for (size_t i = 0; i < max_size_seq_view; i++) {
    if (n_vec[i] < 0) {
      return ops_partials.build(LOG_ZERO);
    }
  }

  // compute gamma(n+1)
  VectorBuilder<include_summand<propto>::value, T_partials_return, T_n>
      normalizing_constant(size_n);
  for (size_t i = 0; i < size_n; i++)
    if (include_summand<propto>::value)
      normalizing_constant[i] = -lgamma(n_vec[i] + 1);

  // compute lbeta denominator with size r and alpha
  VectorBuilder<true, T_partials_return, T_r, T_size1> lbeta_denominator(size_r_alpha);
  for (size_t i = 0; i < size_r_alpha; i++) {
    lbeta_denominator[i] = lbeta(r_vec.val(i), alpha_vec.val(i));
  }

  // compute lgamma denominator with size beta
  VectorBuilder<true, T_partials_return, T_size2> lgamma_denominator(size_beta);
  for (size_t i = 0; i < size_beta; i++) {
    lgamma_denominator[i] = lgamma(beta_vec.val(i));
  }

  // compute lgamma numerator with size n and beta
  VectorBuilder<true, T_partials_return, T_n, T_size2> lgamma_numerator(size_n_beta);
  for (size_t i = 0; i < size_n_beta; i++) {
    lgamma_numerator[i] = lgamma(n_vec[i] + beta_vec.val(i));
  }

  // compute lbeta numerator with size n, r, alpha and beta
  VectorBuilder<true, T_partials_return, T_n, T_r, T_size1, T_size2> lbeta_diff(max_size_seq_view);
  for (size_t i = 0; i < max_size_seq_view; i++) {
    lbeta_diff[i] = lbeta(n_vec[i] + r_vec.val(i),
                          alpha_vec.val(i) + beta_vec.val(i)) + lgamma_numerator[i]
                    - lbeta_denominator[i] - lgamma_denominator[i];
  }

  // compute derivative w.r.t. r, alpha and beta

  // compute digamma(n+r+alpha+beta)
  VectorBuilder<!is_constant_all<T_r, T_size1, T_size2>::value, T_partials_return,
                T_n, T_r, T_size1, T_size2>
      digamma_n_r_alpha_beta(max_size_seq_view);
  if (!is_constant_all<T_r, T_size1, T_size2>::value) {
    for (size_t i = 0; i < max_size_seq_view; i++) {
      digamma_n_r_alpha_beta[i]
          = digamma(n_vec[i] + r_vec.val(i) + alpha_vec.val(i) + beta_vec.val(i));
    }
  }

  // compute digamma(alpha+beta)
  VectorBuilder<!is_constant_all<T_size1, T_size2>::value, T_partials_return,
                T_size1, T_size2>
      digamma_alpha_beta(size_alpha_beta);
  if (!is_constant_all<T_size1, T_size2>::value) {
    for (size_t i = 0; i < size_alpha_beta; i++) {
      digamma_alpha_beta[i] = digamma(alpha_vec.val(i) + beta_vec.val(i));
    }
  }

  // compute digamma(n+r)
  VectorBuilder<!is_constant_all<T_r>::value, T_partials_return, T_n, T_r>
      digamma_n_r(size_n_r);
  if (!is_constant_all<T_r>::value) {
    for (size_t i = 0; i < size_n_r; i++) {
      digamma_n_r[i] = digamma(n_vec[i] + r_vec.val(i));
    }
  }

  // compute digamma(r+alpha)
  VectorBuilder<!is_constant_all<T_r, T_size1>::value, T_partials_return, T_r, T_size1>
      digamma_r_alpha(size_r_alpha);
  if (!is_constant_all<T_r, T_size1>::value) {
    for (size_t i = 0; i < size_r_alpha; i++) {
      digamma_r_alpha[i] = digamma(r_vec.val(i) + alpha_vec.val(i));
    }
  }

  // compute digamma(n+beta)
  VectorBuilder<!is_constant_all<T_size2>::value, T_partials_return, T_n, T_size2>
      digamma_n_beta(size_n_beta);
  if (!is_constant_all<T_n, T_size2>::value) {
    for (size_t i = 0; i < size_n_beta; i++) {
      digamma_n_beta[i] = digamma(n_vec[i] + beta_vec.val(i));
    }
  }

  // compute digamma(r)
  VectorBuilder<!is_constant_all<T_r>::value, T_partials_return, T_r> digamma_r(size_r);
  if (!is_constant_all<T_r>::value) {
    for (size_t i = 0; i < size_r; i++) {
      digamma_r[i] = digamma(r_vec.val(i));
    }
  }

  // compute digamma(alpha)
  VectorBuilder<!is_constant_all<T_size1>::value, T_partials_return, T_size1> digamma_alpha(size_alpha);
  if (!is_constant_all<T_size1>::value) {
    for (size_t i = 0; i < size_alpha; i++) {
      digamma_alpha[i] = digamma(alpha_vec.val(i));
    }
  }

  // compute digamma(beta)
  VectorBuilder<!is_constant_all<T_size2>::value, T_partials_return, T_size2> digamma_beta(size_beta);
  if (!is_constant_all<T_size2>::value) {
    for (size_t i = 0; i < size_beta; i++) {
      digamma_beta[i] = digamma(beta_vec.val(i));
    }
  }


  for (size_t i = 0; i < max_size_seq_view; i++) {
    if (include_summand<propto>::value)
      logp += normalizing_constant[i];
    logp += lbeta_diff[i];

    if (!is_constant_all<T_r>::value) {
      ops_partials.edge1_.partials_[i]
          += digamma_n_r[i] - digamma_n_r_alpha_beta[i] - (digamma_r[i] - digamma_r_alpha[i]);
    }
    if (!is_constant_all<T_size1>::value) {
      ops_partials.edge2_.partials_[i]
          += digamma_alpha_beta[i] - digamma_n_r_alpha_beta[i] - (digamma_alpha[i] - digamma_r_alpha[i]);
    }
    if (!is_constant_all<T_size2>::value) {
      ops_partials.edge3_.partials_[i]
          += digamma_alpha_beta[i] - digamma_n_r_alpha_beta[i] + digamma_n_beta[i] - digamma_beta[i];
    }
  }
  return ops_partials.build(logp);
}

template <typename T_n, typename T_r, typename T_size1, typename T_size2>
inline stan::return_type_t<T_r, T_size1, T_size2> beta_neg_binomial_lpmf(const T_n& n, 
                                                   const T_r& r,
                                                   const T_size1& alpha,
                                                   const T_size2& beta,
                                                   std::ostream* pstream__) {
  return beta_neg_binomial_lpmf<false>(n, r, alpha, beta);
}








}
#endif
