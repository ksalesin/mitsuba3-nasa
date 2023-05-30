#include <mitsuba/core/properties.h>
#include <mitsuba/render/sizedistr.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-powerlaw:

Gamma size distribution function (:monosp:`gamma`)
-----------------------------------------------

.. list-table::
 :widths: 20 15 65
 :header-rows: 1
 :class: paramstable

 * - Parameter
   - Type
   - Description
 * - min_radius
   - |float|
   - Minimum radius (units of distance)
 * - max_radius
   - |float|
   - Maximum radius (units of distance)

This plugin implements a gamma size distribution function.

*/
template <typename Float, typename Spectrum>
class GammaSizeDistr final : public SizeDistribution<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SizeDistribution, m_min_radius, m_max_radius, 
                    m_constant, calculate_gauss, calculate_constant)
    MI_IMPORT_TYPES()

    GammaSizeDistr(const Properties &props) : Base(props) {
        m_min_radius = props.get<ScalarFloat>("min_radius", 500.f);
        m_max_radius = props.get<ScalarFloat>("max_radius", 5000.f);
        m_a = props.get<ScalarFloat>("a", 600.f);
        m_b = props.get<ScalarFloat>("b", 0.25f);

        if (m_a < 0)
            Log(Error, "a must be positive!");

        if (!(m_b > 0 && m_b < 500.0))
            Log(Error, "b must be between 0 and 500!");

        calculate_gauss();
        calculate_constant();
    }

    Float eval(Float r, bool normalize) const override {
        Float value = dr::pow(r, (1 - 3 * m_b) / m_b) * dr::exp(-r / (m_a * m_b));

        if (normalize)
            return Float(m_constant) * value;
        else
            return value;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "GammaSizeDistr[" << std::endl
            << "  min_radius = " << string::indent(m_min_radius) << std::endl
            << "  max_radius = " << string::indent(m_max_radius) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ScalarFloat m_a;
    ScalarFloat m_b;
};

MI_IMPLEMENT_CLASS_VARIANT(GammaSizeDistr, SizeDistribution)
MI_EXPORT_PLUGIN(GammaSizeDistr, "Power law size distribution function")
NAMESPACE_END(mitsuba)
