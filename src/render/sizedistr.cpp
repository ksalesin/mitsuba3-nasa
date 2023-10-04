
#include <mitsuba/core/properties.h>
#include <mitsuba/core/quad.h>
#include <mitsuba/render/sizedistr.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT SizeDistribution<Float, Spectrum>::SizeDistribution(const Properties &props) {
    m_g = props.get<uint32_t>("gauss_points", 100);

    auto [nodes, weights] = quad::gauss_legendre<FloatX>(m_g);

    m_gauss_nodes = nodes;
    m_gauss_weights = weights;
}

MI_VARIANT SizeDistribution<Float, Spectrum>::~SizeDistribution() {}

MI_VARIANT void SizeDistribution<Float, Spectrum>::calculate_constants() {
    double integral = 0.f, G_avg = 0.f, m_reff = 0.f, m_veff = 0.f;
    double shift, scale, val;
    
    if constexpr (dr::is_jit_v<Float>) {
        shift = 0.5f * (m_max_radius[0] + m_min_radius[0]);
        scale = 0.5f * (m_max_radius[0] - m_min_radius[0]);
    } else {
        shift = 0.5f * (m_max_radius + m_min_radius);
        scale = 0.5f * (m_max_radius - m_min_radius);
    }

    // Use Gaussian quadrature to compute integral of `eval` over radius interval
    for (uint32_t i = 0; i < m_g; i++) {
        double node = m_gauss_nodes[i] * scale + shift;
        double weight = m_gauss_weights[i] * scale;

        if constexpr (dr::is_jit_v<Float>)
            val = eval(node, false)[0];
        else
            val = eval(node, false);

        integral += weight * val;
    }

    // The normalization constant must be set before calling eval() below
    m_normalization = dr::rcp(integral);

    // for (uint32_t i = 0; i < m_g; i++) {
    //     ScalarFloat radius = m_gauss_nodes[i] * scale + shift;
    //     ScalarFloat weight = m_gauss_weights[i] * scale;

    //     if constexpr (dr::is_jit_v<Float>)
    //         val = eval(radius)[0];
    //     else
    //         val = eval(radius);

    //     G_avg += weight * val * dr::Pi<ScalarFloat> * dr::sqr(radius);
    //     m_reff += weight * val * dr::Pi<ScalarFloat> * dr::pow(radius, 3.f);
    // }

    // m_reff *= dr::rcp(G_avg);

    // for (uint32_t i = 0; i < m_g; i++) {
    //     ScalarFloat radius = m_gauss_nodes[i] * scale + shift;
    //     ScalarFloat weight = m_gauss_weights[i] * scale;

    //     if constexpr (dr::is_jit_v<Float>)
    //         val = eval(radius)[0];
    //     else
    //         val = eval(radius);

    //     m_veff += weight * val * dr::sqr(radius - m_reff) * dr::Pi<ScalarFloat> * dr::sqr(radius);
    // }

    // m_veff *= dr::rcp(G_avg * dr::sqr(m_reff));
}

MI_IMPLEMENT_CLASS_VARIANT(SizeDistribution, Object, "sizedistr")
MI_INSTANTIATE_CLASS(SizeDistribution)
NAMESPACE_END(mitsuba)
