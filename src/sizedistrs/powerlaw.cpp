#include <mitsuba/core/properties.h>
#include <mitsuba/render/sizedistr.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-powerlaw:

Power law size distribution function (:monosp:`powerlaw`)
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

This plugin implements a power law size distribution function.

*/
template <typename Float, typename Spectrum>
class PowerLawSizeDistr final : public SizeDistribution<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SizeDistribution, m_min_radius, m_max_radius, 
                    m_normalization, calculate_constants)
    MI_IMPORT_TYPES()

    PowerLawSizeDistr(const Properties &props) : Base(props) {
        ScalarFloat min_radius = props.get<ScalarFloat>("min_radius", 500.f);
        ScalarFloat max_radius = props.get<ScalarFloat>("max_radius", 5000.f);
        
        // exponent = 3 is used in [Mishchenko and Yang 2018]
        ScalarFloat exponent = props.get<ScalarFloat>("exponent", 3.f);

        m_min_radius = min_radius;
        m_max_radius = max_radius;
        m_exponent = exponent;

        calculate_constants();
    }

    Float eval(Float r, bool normalize) const override {
        Float value = dr::pow(r, -m_exponent);

        if (normalize)
            return Float(m_normalization) * value;
        else
            return value;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("min_radius", m_min_radius, +ParamFlags::Differentiable);
        callback->put_parameter("max_radius", m_max_radius, +ParamFlags::Differentiable);
        callback->put_parameter("exponent", m_exponent, +ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "PowerLawSizeDistr[" << std::endl
            << "  min_radius = " << string::indent(m_min_radius) << std::endl
            << "  max_radius = " << string::indent(m_max_radius) << std::endl
            << "  exponent = " << string::indent(m_exponent) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Float m_exponent;
};

MI_IMPLEMENT_CLASS_VARIANT(PowerLawSizeDistr, SizeDistribution)
MI_EXPORT_PLUGIN(PowerLawSizeDistr, "Power law size distribution function")
NAMESPACE_END(mitsuba)
