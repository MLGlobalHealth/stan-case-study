#ifndef STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_RNG_HPP
#define STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/prob/neg_binomial_rng.hpp>
#include <stan/math/prim/fun/logit.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/prob/beta_rng.hpp>
#include <limits>

namespace example_model_namespace {

/** \ingroup prob_dists
 * Return a beta-negative binomial random variate with given number of successes,
 * success, and failure parameters using the given random number generator.
 *
 * r, alpha, and beta can each be a scalar or a one-dimensional container. Any
 * non-scalar inputs must be the same size.
 *
 * @tparam T_r type of number of successes parameter
 * @tparam T_size1 type of prior success parameter
 * @tparam T_size2 type of prior failure parameter
 * @tparam RNG type of random number generator
 * 
 * @param r (Sequence of) number of successes parameter(s)
 * @param alpha (Sequence of) positive success parameter(s)
 * @param beta (Sequence of) positive failure parameter(s)
 * @param rng random number generator
 * @return (Sequence of) beta-binomial random variate(s)
 * @throw std::domain_error if r is negative, or alpha or beta are nonpositive
 * @throw std::invalid_argument if non-scalar arguments are of different sizes
 */
template <typename T_r, typename T_shape1, typename T_shape2, class RNG>
inline typename stan::VectorBuilder<true, int, T_r, T_shape1, T_shape2>::type
beta_neg_binomial_rng(const T_r &r, const T_shape1 &alpha, const T_shape2 &beta,
                  RNG &rng, std::ostream* pstream__) {


  using stan::ref_type_t;
  using stan::VectorBuilder;
  using stan::scalar_seq_view;
  using stan::math::size;
  using namespace stan::math;

  using T_r_ref = ref_type_t<T_r>;
  using T_alpha_ref = ref_type_t<T_shape1>;
  using T_beta_ref = ref_type_t<T_shape2>;
  static const char *function = "beta_neg_binomial_rng";
  check_consistent_sizes(function, "Population size parameter", r,
                         "First prior sample size parameter", alpha,
                         "Second prior sample size parameter", beta);

  T_r_ref r_ref = r;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;
  check_positive_finite(function, "Population size parameter", r_ref);
  check_positive_finite(function, "First prior sample size parameter", alpha_ref);
  check_positive_finite(function, "Second prior sample size parameter", beta_ref);

  using T_p = decltype(beta_rng(alpha_ref, beta_ref, rng));
  T_p p = beta_rng(alpha_ref, beta_ref, rng);

  scalar_seq_view<T_p> p_vec(p);
  size_t size_p = size(p);
  VectorBuilder<true, double, T_p> odds_ratio_p(size_p);
  for (size_t n = 0; n < size_p; ++n) {
    odds_ratio_p[n] = p_vec.val(n) / (1 - p_vec.val(n));
  }

  // Replace infinities and zeros
  double max_finite = std::numeric_limits<double>::max();
  double min_positive = std::numeric_limits<double>::min();
  for (size_t i = 0; i < size_p; ++i) {
    if (is_inf(odds_ratio_p[i])) {
      odds_ratio_p[i] = max_finite;
    } else if (odds_ratio_p[i] == 0.0) {
      odds_ratio_p[i] = min_positive;
    }
  }

  return neg_binomial_rng(r_ref, odds_ratio_p.data(), rng);
}



}
#endif
