
#include <mitsuba/core/properties.h>
#include <mitsuba/core/quad.h>
#include <mitsuba/render/sizedistr.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT SizeDistribution<Float, Spectrum>::SizeDistribution(const Properties &props) {
    m_min_radius = props.get<ScalarFloat>("min_radius", 500.f);
    m_max_radius = props.get<ScalarFloat>("max_radius", 5000.f);
    m_g = props.get<uint32_t>("gauss_points", 100);

    auto [nodes, weights] = quad::gauss_legendre<FloatX>(m_g);

    m_gauss_nodes = dr::zeros<FloatX>(m_g);
    m_gauss_weights = dr::zeros<FloatX>(m_g);

    ScalarFloat shift = 0.5f * (m_max_radius + m_min_radius);
    ScalarFloat scale = 0.5f * (m_max_radius - m_min_radius);

    for (size_t i = 0; i < m_g; i++) {
        m_gauss_nodes[i] = nodes[i] * scale + shift;
        m_gauss_weights[i] = weights[i] * scale;
    }
}

MI_VARIANT SizeDistribution<Float, Spectrum>::~SizeDistribution() {}

MI_VARIANT void SizeDistribution<Float, Spectrum>::compute_constant() {
    ScalarFloat integral = 0.f;
    ScalarFloat val;

    // Use Gaussian quadrature to compute integral of `eval` over radius interval
    for (uint32_t i = 0; i < m_g; i++) {
        ScalarFloat node = m_gauss_nodes[i];
        ScalarFloat weight = m_gauss_weights[i];
        
        if constexpr (dr::is_jit_v<Float>) {
            val = eval(node, false)[0];
        } else {
            val = eval(node, false);
        }

        integral += weight * val;
    }

    m_normalization = dr::rcp(integral);
}

MI_IMPLEMENT_CLASS_VARIANT(SizeDistribution, Object, "sizedistr")
MI_INSTANTIATE_CLASS(SizeDistribution)
NAMESPACE_END(mitsuba)
