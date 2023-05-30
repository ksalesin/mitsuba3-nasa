#include <mitsuba/core/properties.h>
#include <mitsuba/render/sizedistr.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-lognormal:

Log normal size distribution function (:monosp:`lognormal`)
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
 * - mean_radius
   - |float|
   - Mean radius (units of distance)
 * - std
   - |float|
   - Standard deviation (units of distance)

This plugin implements a log normal size distribution function.

*/
template <typename Float, typename Spectrum>
class LogNormalSizeDistr final : public SizeDistribution<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SizeDistribution, m_min_radius, m_max_radius, 
                    m_constant, calculate_gauss, calculate_constant)
    MI_IMPORT_TYPES()

    LogNormalSizeDistr(const Properties &props) : Base(props) {
        m_min_radius = props.get<ScalarFloat>("min_radius", 500.f);
        m_max_radius = props.get<ScalarFloat>("max_radius", 5000.f);
        m_mean_radius = props.get<ScalarFloat>("mean_radius", 1000.f);
        ScalarFloat std = props.get<ScalarFloat>("std", 100.f);

        if (m_min_radius <= 0 || m_max_radius <= 0 || m_mean_radius <= 0)
            Log(Error, "Radii must be positive!");

        if (std <= 0)
            Log(Error, "Standard deviation must be positive!");

        ScalarFloat ln_std = dr::log(std);
        m_std_constant = dr::rcp(2.f * dr::sqr(ln_std));
        
        calculate_gauss();
        calculate_constant();
    }

    Float eval(Float r, bool normalize) const override {
        Float a = dr::log(r) - dr::log(Float(m_mean_radius));
        Float value = dr::exp(-dr::sqr(a) * Float(m_std_constant)) / r;

        if (normalize)
            return Float(m_constant) * value;
        else
            return value;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LogNormalSizeDistr[" << std::endl
            << "  min_radius = " << string::indent(m_min_radius) << std::endl
            << "  max_radius = " << string::indent(m_max_radius) << std::endl
            << "  mean_radius = " << string::indent(m_mean_radius) << std::endl
            << "  std_constant = " << string::indent(m_std_constant) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ScalarFloat m_mean_radius;
    ScalarFloat m_std_constant;
};

MI_IMPLEMENT_CLASS_VARIANT(LogNormalSizeDistr, SizeDistribution)
MI_EXPORT_PLUGIN(LogNormalSizeDistr, "Log normal size distribution function")
NAMESPACE_END(mitsuba)
