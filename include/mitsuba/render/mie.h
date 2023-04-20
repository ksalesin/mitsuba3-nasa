#pragma once

#include <mitsuba/core/vector.h>
#include <drjit/complex.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Evaluates the scattering amplitudes of a dielectric sphere using Lorenz-Mie theory.
 *
 * \param mu
 *      Cosine of the angle between the the incident ray and the scattered ray
 *
 * \param wavelength
 *      Wavelength of light (units of distance, must be consistent with radius of spheres)
 * 
 * \param radius
 *      Radius of the sphere (units of distance)
 * 
 * \param ior_med
 *      Index of refraction of the host medium (complex-valued)
 * 
 * \param ior_sph
 *      index of refraction of the sphere (complex-valued)
 * 
 * \param nmax
 *      Number of terms in the infinite series that should be used (-1: automatic)
 * 
 * \param 
 *
 * \return A tuple (S1, S2, Ns) consisting of
 *
 *     S1          Scattering amplitude/phase of the *ordinary* ray
 *
 *     S2          Scattering amplitude/phase of the *extraordinary* ray
 * 
 *     Ns          Normalization coefficient required by the phase function,
 *                 which is defined as ``(abs(S1)^2 + abs(S2)^2) / Ns``
*/
template <typename Value, typename Int>
std::tuple<dr::Complex<Value>, dr::Complex<Value>, Value> mie_s1s2(Value wavelengths, Value mu, Value radius, dr::Complex<Value> ior_med, dr::Complex<Value> ior_sph, Int nmax_) {
    using Complex2v = dr::Complex<Value>;
    using Array2v = dr::Array<Value, 2>;

    // Relative index of refraction
    Complex2v m = ior_sph * dr::rcp(ior_med);

    // Wave numbers
    Complex2v kx = dr::TwoPi<Value> * ior_med * dr::rcp(wavelengths),
              ky = dr::TwoPi<Value> * ior_sph * dr::rcp(wavelengths);

    // Size parameters
    Complex2v x = kx * radius, 
              y = ky * radius;

    // Related constants
    Complex2v m_sq = dr::sqr(m),
              rcp_x = dr::rcp(x),
              rcp_y = dr::rcp(y),
              i(0, 1);

    Value x_norm = dr::norm(x);
    Value y_norm = dr::norm(y);

    Int x_nmax = nmax_;
    Int y_nmax = nmax_;

    // Default stopping criterion, [Mishchenko and Yang 2018]
    if (x_nmax == -1) {
        x_nmax = (Int) dr::max_nested(8 + x_norm + 4.05f * dr::pow(x_norm, 1.f / 3.f));
        y_nmax = (Int) dr::max_nested(8 + y_norm + 4.05f * dr::pow(y_norm, 1.f / 3.f));
    }

    // Default starting n for downward recurrence of ratio j_n(z) / j_{n-1}(z)
    Int x_ndown = x_nmax + 8 * (Int) dr::sqrt(x_nmax) + 3;
    Int y_ndown = y_nmax + 8 * (Int) dr::sqrt(y_nmax) + 3;

    // Calculate ratio j_n(z) / j_{n-1}(z) by downward recurrence for z = x
    std::vector<Complex2v> j_ratio_x(x_ndown);
    Complex2v j_ratio_x_n = x * dr::rcp(2.f * x_ndown + 1);

    for (Int n = x_ndown - 1; n >= 1; --n) {
        Complex2v kx_n = Value(2 * n + 1) * rcp_x;
        j_ratio_x[n] = j_ratio_x_n = dr::rcp(kx_n - j_ratio_x_n);
    }

    // Calculate ratio j_n(z) / j_{n-1}(z) by downward recurrence for z = y
    std::vector<Complex2v> j_ratio_y(y_ndown);
    Complex2v j_ratio_y_n = y * dr::rcp(2.f * y_ndown + 1);

    for (Int n = y_ndown - 1; n >= 1; --n) {
        Complex2v ky_n = Value(2 * n + 1) * rcp_y;
        j_ratio_y[n] = j_ratio_y_n = dr::rcp(ky_n - j_ratio_y_n);
    }

    // Variables for upward recurrences of Bessel fcts.
    Complex2v jx_0 = dr::sin(x) * rcp_x,
              jy_0 = dr::sin(y) * rcp_y;

    // Variables for upward recurrences of Hankel fcts.
    Complex2v h_exp = dr::exp(i * x) * rcp_x,
              hx_0 = -i * h_exp,
              hx_1 = -h_exp * (1.f + i * rcp_x);

    // Upward recurrence for deriv. of Legendre polynomial
    Value pi_0 = 0, pi_1 = 1;

    // Accumulation variables for S1 and S2 terms
    Complex2v S1 = 0, S2 = 0;

    // Accumulation variable for normalization factor
    Value Ns = 0;

    for (Int n = 1; n <= x_nmax; ++n) {
        Value fn = n;
        j_ratio_x_n = j_ratio_x[n];
        j_ratio_y_n = j_ratio_y[n];

        // Upward recurrences for Bessel and Hankel functions
        Complex2v hx_n, hx_dx;
        if (n == 1) {
            hx_n = hx_1;
            hx_dx = x * hx_0 - fn * hx_1;
        } else {
            hx_n = (2 * fn - 1) * rcp_x * hx_1 - hx_0;
            hx_dx = x * hx_1 - fn * hx_n;

            hx_0 = hx_1; hx_1 = hx_n;
        }

        Complex2v jx_n = j_ratio_x_n * jx_0,
                  jy_n = j_ratio_y_n * jy_0,
                  jx_dx = x * jx_0 - fn * jx_n,
                  jy_dy = y * jy_0 - fn * jy_n;

        jx_0 = jx_n; jy_0 = jy_n;

        /* Upward recurrences for angle-dependent terms based on
        Legendre functions (Absorption and Scattering of Light by Small
        Particles, Bohren and Huffman, p. 95) */
        Value pi_n, tau_n;
        if (n == 1) {
            pi_n = pi_1;
            tau_n = mu;
        } else {
            pi_n = ((2 * fn - 1) / (fn - 1)) * mu * pi_1 -
                (fn / (fn - 1)) * pi_0;
            tau_n = fn * mu * pi_n - (fn + 1) * pi_1;
            
            pi_0 = pi_1; pi_1 = pi_n;
        }

        // Lorenz-Mie coefficients (Eqs. 9, 10)
        Complex2v a_n = (m_sq * jy_n * jx_dx - jx_n * jy_dy) /
                        (m_sq * jy_n * hx_dx - hx_n * jy_dy),
                  b_n = (jy_n * jx_dx - jx_n * jy_dy) /
                        (jy_n * hx_dx - hx_n * jy_dy);

        // Calculate i-th term of S1 and S2
        Value kn = (2 * fn + 1) / (fn * (fn + 1));
        S1 += kn * (a_n * tau_n + b_n * pi_n);
        S2 += kn * (a_n * pi_n + b_n * tau_n);

        // Calculate i-th term of factor in denominator
        Ns += (2 * fn + 1) * (dr::squared_norm(a_n) + dr::squared_norm(b_n));
    }

    S1 *= i * dr::rcp(kx);
    S2 *= i * dr::rcp(kx);

    Ns *= 0.5f * dr::FourPi<Value> / dr::squared_norm(kx);

    return { S1, S2, Ns };
}

/**
 * \brief Evaluates the scattering and extinction cross sections using Lorenz-Mie theory.
 * 
 * \param wavelength
 *      Wavelength of light (units of distance, must be consistent with radius of spheres)
 * 
 * \param radius
 *      Radius of the sphere (units of distance)
 * 
 * \param ior_med
 *      Index of refraction of the host medium (complex-valued)
 * 
 * \param ior_sph
 *      index of refraction of the sphere (complex-valued)
 * 
 * \param nmax
 *      Number of terms in the infinite series that should be used (-1: automatic)
 * 
 * \return A tuple (Cs, Ct) consisting of
 *
 *     Cs          Scattering cross section (units of 1/(distance * wavelength))
 *
 *     Ct          Extinction cross section (units of 1/(distance * wavelength))
 * 
*/
template <typename Value, typename Int>
std::tuple<Value, Value> mie_xsections(const Value &wavelengths, Value radius, dr::Complex<Value> ior_med, dr::Complex<Value> ior_sph, Int nmax_) {
    using Complex2v = dr::Complex<Value>;
    using Array2v = dr::Array<Value, 2>;

    // Relative index of refraction
    Complex2v m = ior_sph * dr::rcp(ior_med);

    // Wave numbers
    Complex2v kx = dr::TwoPi<Value> * ior_med * dr::rcp(wavelengths);
    Complex2v ky = dr::TwoPi<Value> * ior_sph * dr::rcp(wavelengths);

    // Size parameters
    Complex2v x = kx * radius;
    Complex2v y = ky * radius;

    // Related constants
    Complex2v m_sq = dr::sqr(m),
              rcp_x = dr::rcp(x),
              rcp_y = dr::rcp(y),
              i(0, 1);

    Value x_norm = dr::norm(x);
    Value y_norm = dr::norm(y);

    Int x_nmax = nmax_;
    Int y_nmax = nmax_;

    // Default stopping criterion, [Mishchenko and Yang 2018]
    if (x_nmax == -1) {
        x_nmax = (Int) dr::max_nested(8 + x_norm + 4.05f * dr::pow(x_norm, 1.f / 3.f));
        y_nmax = (Int) dr::max_nested(8 + y_norm + 4.05f * dr::pow(y_norm, 1.f / 3.f));
    }

    // Default starting n for downward recurrence of ratio j_n(z) / j_{n-1}(z)
    Int x_ndown = x_nmax + 8 * (Int) dr::sqrt(x_nmax) + 3;
    Int y_ndown = y_nmax + 8 * (Int) dr::sqrt(y_nmax) + 3;

    // Calculate ratio j_n(z) / j_{n-1}(z) by downward recurrence for z = x
    std::vector<Complex2v> j_ratio_x(x_ndown);
    Complex2v j_ratio_x_n = x * dr::rcp(2.f * x_ndown + 1);

    for (Int n = x_ndown - 1; n >= 1; --n) {
        Complex2v kx_n = (2 * n + 1) * rcp_x;
        j_ratio_x[n] = j_ratio_x_n = dr::rcp(kx_n - j_ratio_x_n);
    }

    // Calculate ratio j_n(z) / j_{n-1}(z) by downward recurrence for z = y
    std::vector<Complex2v> j_ratio_y(y_ndown);
    Complex2v j_ratio_y_n = y * dr::rcp(2.f * y_ndown + 1);

    for (Int n = y_ndown - 1; n >= 1; --n) {
        Complex2v ky_n = (2 * n + 1) * rcp_y;
        j_ratio_y[n] = j_ratio_y_n = dr::rcp(ky_n - j_ratio_y_n);
    }

    // Variables for upward recurrences of Bessel fcts.
    Complex2v jx_0 = dr::sin(x) * rcp_x,
              jy_0 = dr::sin(y) * rcp_y;

    // Variables for upward recurrences of Hankel fcts.
    Complex2v h_exp = dr::exp(i * x) * rcp_x,
              hx_0 = -i * h_exp,
              hx_1 = -h_exp * (1.f + i * rcp_x);

    // Accumulation variables for cross sections
    Complex2v Cs = 0, Ct = 0;

    for (Int n = 1; n <= x_nmax; ++n) {
        Value fn = n;
        j_ratio_x_n = j_ratio_x[n];
        j_ratio_y_n = j_ratio_y[n];

        // Upward recurrences for Bessel and Hankel functions
        Complex2v hx_n, hx_dx;
        if (n == 1) {
            hx_n = hx_1;
            hx_dx = x * hx_0 - fn * hx_1;
        } else {
            hx_n = (2 * fn - 1) * rcp_x * hx_1 - hx_0;
            hx_dx = x * hx_1 - fn * hx_n;

            hx_0 = hx_1; hx_1 = hx_n;
        }

        Complex2v jx_n = j_ratio_x_n * jx_0,
                  jy_n = j_ratio_y_n * jy_0,
                  jx_dx = x * jx_0 - fn * jx_n,
                  jy_dy = y * jy_0 - fn * jy_n;

        jx_0 = jx_n; jy_0 = jy_n;

        // Lorenz-Mie coefficients (Eqs. 9, 10)
        Complex2v a_n = (m_sq * jy_n * jx_dx - jx_n * jy_dy) /
                        (m_sq * jy_n * hx_dx - hx_n * jy_dy),
                  b_n = (jy_n * jx_dx - jx_n * jy_dy) /
                        (jy_n * hx_dx - hx_n * jy_dy);

        // Calculate i-th term of Cs and Ct
        Value kn = (2 * fn + 1);
        Cs += kn * (dr::squared_norm(a_n) + dr::squared_norm(b_n));
        Ct += dr::real(kn * (a_n + b_n));
    }

    Value k = dr::TwoPi<Value> / dr::sqr(kx);
    Cs *= k;
    Ct *= k;

    return { Cs, Ct };
}

NAMESPACE_END(mitsuba)