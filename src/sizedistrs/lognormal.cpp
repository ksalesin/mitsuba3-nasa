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
                    m_normalization, calculate_constants)
    MI_IMPORT_TYPES()

    LogNormalSizeDistr(const Properties &props) : Base(props) {
        ScalarFloat min_radius = props.get<ScalarFloat>("min_radius", 500.f);
        ScalarFloat max_radius = props.get<ScalarFloat>("max_radius", 5000.f);
        ScalarFloat mean_radius = props.get<ScalarFloat>("mean_radius", 1000.f);
        ScalarFloat std = props.get<ScalarFloat>("std", 100.f);

        m_min_radius = min_radius;
        m_max_radius = max_radius;
        m_mean_radius = mean_radius;
        m_std = std;

        Float ln_std = dr::log(m_std);
        m_std_constant = dr::rcp(2.f * dr::sqr(ln_std));
        
        calculate_constants();
    }

    Float eval(Float r, bool normalize) const override {
        Float a = dr::log(r) - dr::log(Float(m_mean_radius));
        Float value = dr::exp(-dr::sqr(a) * m_std_constant) / r;

        if (normalize)
            return Float(m_normalization) * value;
        else
            return value;
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("min_radius", m_min_radius, +ParamFlags::Differentiable);
        callback->put_parameter("max_radius", m_max_radius, +ParamFlags::Differentiable);
        callback->put_parameter("mean_radius", m_mean_radius, +ParamFlags::Differentiable);
        callback->put_parameter("std", m_std, +ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LogNormalSizeDistr[" << std::endl
            << "  min_radius = " << string::indent(m_min_radius) << std::endl
            << "  max_radius = " << string::indent(m_max_radius) << std::endl
            << "  mean_radius = " << string::indent(m_mean_radius) << std::endl
            << "  std = " << string::indent(m_std) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Float m_mean_radius;
    Float m_std_constant;
    Float m_std;
};

MI_IMPLEMENT_CLASS_VARIANT(LogNormalSizeDistr, SizeDistribution)
MI_EXPORT_PLUGIN(LogNormalSizeDistr, "Log normal size distribution function")
NAMESPACE_END(mitsuba)
