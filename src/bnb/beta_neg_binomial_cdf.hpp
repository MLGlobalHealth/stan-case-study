#ifndef STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_CDF_HPP
#define STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_CDF_HPP

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
 * number of successes, prior success, and prior failure parameters. 
 * Given containers of matching sizes, returns the sum of probabilities.
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
 * @return probability or sum of probabilities
 * @throw std::domain_error if r, alpha, or beta fails to be positive
 * @throw std::invalid_argument if container sizes mismatch
 */
template <typename T_n, typename T_r, typename T_size1, typename T_size2>
inline stan::return_type_t<T_r, T_size1, T_size2> beta_neg_binomial_cdf(const T_n& n, 
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
  using stan::math::log1m;
  using namespace stan::math;

  using T_partials_return = partials_return_t<T_n, T_r, T_size1, T_size2>;
  using std::exp;
  using std::log;
  using T_r_ref = ref_type_t<T_r>;
  using T_alpha_ref = ref_type_t<T_size1>;
  using T_beta_ref = ref_type_t<T_size2>;
  static const char* function = "beta_neg_binomial_cdf";
  check_consistent_sizes(function, "Successes variable", n,
                         "Number of successes parameter", r,
                         "First prior sample size parameter", alpha,
                         "Second prior sample size parameter", beta);
  if (size_zero(n, r, alpha, beta)) {
    return 0;
  }

  T_r_ref r_ref = r;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;
  check_positive_finite(function, "Number of successes parameter", r_ref);
  check_positive_finite(function, "First prior sample size parameter", alpha_ref);
  check_positive_finite(function, "Second prior sample size parameter", beta_ref);

  T_partials_return P(0.0);
  operands_and_partials<T_r_ref, T_alpha_ref, T_beta_ref> ops_partials(r_ref, alpha_ref, beta_ref);

  scalar_seq_view<T_n> n_vec(n);
  scalar_seq_view<T_r_ref> r_vec(r_ref);
  scalar_seq_view<T_alpha_ref> alpha_vec(alpha_ref);
  scalar_seq_view<T_beta_ref> beta_vec(beta_ref);
  size_t size_n = stan::math::size(n);
  size_t max_size_seq_view = max_size(n, r, alpha, beta);

  // Explicit return for out of range values 
  for (size_t i = 0; i < size_n; i++) {
    if (n_vec.val(i) < 0) {
      return ops_partials.build(negative_infinity());
    }
  }

  for (size_t i = 0; i < max_size_seq_view; i++) {
    const T_partials_return n_dbl = n_vec.val(i);
    const T_partials_return r_dbl = r_vec.val(i);
    const T_partials_return alpha_dbl = alpha_vec.val(i);
    const T_partials_return beta_dbl = beta_vec.val(i);
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
        = is_constant_all<T_r, T_size1, T_size2>::value
              ? 0
              : digamma(a_plus_r + b_plus_n + 1);
    T_partials_return digamma_ab
        = is_constant_all<T_size1, T_size2>::value
              ? 0
              : digamma(alpha_dbl + beta_dbl);

    T_partials_return dF[6];
    if (!is_constant_all<T_r, T_size1, T_size2>::value) {
      grad_F32(dF, one, b_plus_n + 1, r_plus_n + 1, n_dbl + 2, a_plus_r + b_plus_n + 1, one, 1e-3);
    }
    if (!is_constant_all<T_r>::value) {
      const T_partials_return g
          = digamma(r_plus_n + 1) + (digamma(a_plus_r) - digamma_abrn) + (dF[2] + dF[4]) / F - digamma(r_dbl);
      ops_partials.edge1_.partials_[i] += -g * P_i;
    }
    if (!is_constant_all<T_size1>::value) {
      const T_partials_return g
          = digamma(a_plus_r) - digamma_abrn + dF[4] / F - (digamma(alpha_dbl) - digamma_ab);
      ops_partials.edge2_.partials_[i] += -g * P_i;
    }
    if (!is_constant_all<T_size2>::value) {
      const T_partials_return g
          = digamma(b_plus_n + 1) - digamma_abrn + (dF[1] + dF[4]) / F - (digamma(beta_dbl) - digamma_ab);
      ops_partials.edge3_.partials_[i] += -g * P_i;
    }
  }

  return ops_partials.build(P);
}

}
#endif
