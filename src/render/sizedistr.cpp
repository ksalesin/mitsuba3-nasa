
#include <mitsuba/core/properties.h>
#include <mitsuba/render/sizedistr.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT SizeDistribution<Float, Spectrum>::SizeDistribution(const Properties &props) {
    m_g = props.get<uint32_t>("gauss_points", 100);

    if (m_g <= 0)
        Throw("The number of divisions for Gaussian quadrature must be positive!");
}

MI_VARIANT SizeDistribution<Float, Spectrum>::~SizeDistribution() {}

/// Based on the method in:
/// "Numerical Recipes in C: The Art of Scientific Computing." 2nd ed. Press et al. 1992. p. 152.
MI_VARIANT void SizeDistribution<Float, Spectrum>::calculate_gauss() {
    Log(Info, "Calculating %u points and weights for Gaussian quadrature...", m_g);
    ScalarFloat min_radius, max_radius;

    if constexpr(dr::is_jit_v<Float>) {
        min_radius = m_min_radius[0];
        max_radius = m_max_radius[0];
    } else {
        min_radius = m_min_radius;
        max_radius = m_max_radius;
    }

    m_gauss_points = std::vector<double>(m_g);
    m_gauss_weights = std::vector<double>(m_g);

    uint32_t m = (m_g + 1) / 2, i, j;

    double z, p1, p2, p3, dp;
    double eps = 1e-8f;
    double delta = 1.f;
    double shift = 0.5f * (max_radius + min_radius);
    double scale = 0.5f * (max_radius - min_radius);

    for (i = 1; i <= m; i++) {
        // Initial guess
        z = dr::cos(dr::Pi<double> * (i - 0.25f) / (m_g + 0.5f));

        // Newton's method (refine the initial guess)
        do {
            // Get the m_g-th Legendre polynomial by recurrence relations
            p1 = 1.f;
            p2 = 0.f;
            for (j = 1; j <= m_g; j++) {
                p3 = p2;
                p2 = p1;
                p1 = ((2.f * j - 1.f) * z * p2 - (j - 1.f) * p3) / j;
            }
            
            // Get derivative of m_g-th Legendre polynomial
            dp = m_g * (z * p1 - p2) / (z * z - 1.f);

            // Update guess
            double z_tmp = z;
            z = z - p1 / dp;
            delta = abs(z - z_tmp);
        }
        while(delta > eps);

        // Scale the roots to desired interval (roots are symmetric)
        m_gauss_points[i   - 1] = shift - scale * z;
        m_gauss_points[m_g - i] = shift + scale * z;

        // Calculate weights
        m_gauss_weights[i   - 1] = 2.f * scale / ((1.f - z * z) * dp * dp);
        m_gauss_weights[m_g - i] = m_gauss_weights[i - 1];  
    }
    
    // for (i = 0; i < m_g; i++)
    //     Log(Warn, "gauss %s: %s, %s", i+1, m_gauss_points[i], m_gauss_weights[i]);

    Log(Info, " done.");
}

MI_VARIANT void SizeDistribution<Float, Spectrum>::calculate_constant() {
    double integral = 0.f;

    // Use Gaussian quadrature to compute integral of `eval` over radius interval
    for (uint32_t i = 0; i < m_g; i++) {
        double gauss_eval;
        if constexpr (dr::is_jit_v<Float>)
            gauss_eval = eval(m_gauss_points[i], false)[0];
        else
            gauss_eval = eval(m_gauss_points[i], false);

        integral += m_gauss_weights[i] * gauss_eval;
    }

    m_constant = dr::rcp(integral);

    // Use Gaussian quadrature to compute the effective radius and variance
    double G_avg = 0.f; m_reff = 0.f; m_veff = 0.f;

    for (uint32_t i = 0; i < m_g; i++) {
        ScalarFloat radius = m_gauss_points[i];
        ScalarFloat weight = m_gauss_weights[i];

        double gauss_eval;
        if constexpr (dr::is_jit_v<Float>)
            gauss_eval = eval(radius)[0];
        else
            gauss_eval = eval(radius);

        G_avg += weight * gauss_eval * dr::Pi<ScalarFloat> * dr::sqr(radius);
    }

    for (uint32_t i = 0; i < m_g; i++) {
        ScalarFloat radius = m_gauss_points[i];
        ScalarFloat weight = m_gauss_weights[i];

        double gauss_eval;
        if constexpr (dr::is_jit_v<Float>)
            gauss_eval = eval(radius)[0];
        else
            gauss_eval = eval(radius);

        m_reff += weight * gauss_eval * dr::Pi<ScalarFloat> * dr::pow(radius, 3.f);
    }

    m_reff *= dr::rcp(G_avg);

    for (uint32_t i = 0; i < m_g; i++) {
        ScalarFloat radius = m_gauss_points[i];
        ScalarFloat weight = m_gauss_weights[i];

        double gauss_eval;
        if constexpr (dr::is_jit_v<Float>)
            gauss_eval = eval(radius)[0];
        else
            gauss_eval = eval(radius);

        m_veff += weight * gauss_eval * dr::sqr(radius - m_reff) * dr::Pi<ScalarFloat> * dr::sqr(radius);
    }

    m_veff *= dr::rcp(G_avg * dr::sqr(m_reff));

    // Log(Info, "G_avg: %s", G_avg);
    // Log(Info, "m_reff: %s", m_reff);
    // Log(Info, "m_veff: %s", m_veff);

    // Use trapezoid rule to compute integral of `eval` over radius interval
    // ScalarFloat dr = (m_max_radius - m_min_radius) / m_g;
    // ScalarFloat r = m_min_radius;
    // ScalarFloat val_0;

    // for (uint32_t i = 0; i <= m_g; i++) {
    //     ScalarFloat val_1;

    //     if constexpr (dr::is_jit_array_v<Float>)
    //         val_1 = eval(r, false)[0];
    //     else
    //         val_1 = eval(r, false);

    //     if (i > 0)
    //         result += (val_0 + val_1) * 0.5f * dr;

    //     val_0 = val_1;
    //     r += dr;
    // }
}

MI_IMPLEMENT_CLASS_VARIANT(SizeDistribution, Object, "sizedistr")
MI_INSTANTIATE_CLASS(SizeDistribution)
NAMESPACE_END(mitsuba)
