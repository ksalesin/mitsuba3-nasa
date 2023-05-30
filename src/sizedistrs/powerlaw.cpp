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
                    m_constant, calculate_gauss, calculate_constant)
    MI_IMPORT_TYPES()

    PowerLawSizeDistr(const Properties &props) : Base(props) {
        m_min_radius = props.get<ScalarFloat>("min_radius", 500.f);
        m_max_radius = props.get<ScalarFloat>("max_radius", 5000.f);

        if (m_min_radius <= 0 || m_max_radius <= 0)
            Log(Error, "Radii must be positive!");

        calculate_gauss();
        calculate_constant();
    }

    Float eval(Float r, bool normalize) const override {
        Float value = dr::rcp(r * r * r);

        if (normalize)
            return Float(m_constant) * value;
        else
            return value;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "PowerLawSizeDistr[" << std::endl
            << "  min_radius = " << string::indent(m_min_radius) << std::endl
            << "  max_radius = " << string::indent(m_max_radius) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    // None
};

MI_IMPLEMENT_CLASS_VARIANT(PowerLawSizeDistr, SizeDistribution)
MI_EXPORT_PLUGIN(PowerLawSizeDistr, "Power law size distribution function")
NAMESPACE_END(mitsuba)
