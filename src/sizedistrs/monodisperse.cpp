#include <mitsuba/core/properties.h>
#include <mitsuba/render/sizedistr.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-monodisperse:

Monodisperse size distribution function (:monosp:`monodisperse`)
-----------------------------------------------

.. list-table::
 :widths: 20 15 65
 :header-rows: 1
 :class: paramstable

 * - Parameter
   - Type
   - Description
 * - radius
   - |float|
   - Radius (units of distance)

This plugin implements a monodisperse size distribution function, aka
a single radius.

*/
template <typename Float, typename Spectrum>
class MonodisperseSizeDistr final : public SizeDistribution<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SizeDistribution, m_min_radius, m_max_radius, 
                    m_constant, calculate_gauss)
    MI_IMPORT_TYPES()

    MonodisperseSizeDistr(const Properties &props) : Base(props) {
        m_min_radius = props.get<ScalarFloat>("radius", 1000.f);
        m_max_radius = m_min_radius;

        if (m_min_radius <= 0)
            Log(Error, "The radius must be positive!");

        m_constant = 1.f;

        // calculate_gauss();
    }

    Float eval(Float r, bool /* normalize */) const override {
        Float val = 0.f;
        dr::masked(val, r == m_min_radius) = 1.f;
        return val;
    }
    
    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MonodisperseSizeDistr[" << std::endl
            << "  min_radius = " << string::indent(m_min_radius) << std::endl
            << "  max_radius = " << string::indent(m_max_radius) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:

};

MI_IMPLEMENT_CLASS_VARIANT(MonodisperseSizeDistr, SizeDistribution)
MI_EXPORT_PLUGIN(MonodisperseSizeDistr, "Monodisperse size distribution function")
NAMESPACE_END(mitsuba)
