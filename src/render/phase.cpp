
#include <mitsuba/core/properties.h>
#include <mitsuba/render/phase.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT
PhaseFunction<Float, Spectrum>::PhaseFunction(const Properties &props)
    : m_flags(+PhaseFunctionFlags::Empty), m_id(props.id()) {}

MI_VARIANT PhaseFunction<Float, Spectrum>::~PhaseFunction() {}

MI_VARIANT std::pair<Spectrum, Float>
PhaseFunction<Float, Spectrum>::eval_pdf_old(const PhaseFunctionContext &ctx,
                                             const MediumInteraction3f &mei,
                                             const Vector3f &wo,
                                             Mask active) const {
    return eval_pdf(ctx, mei, wo, active);
}

MI_IMPLEMENT_CLASS_VARIANT(PhaseFunction, Object, "phase")
MI_INSTANTIATE_CLASS(PhaseFunction)
NAMESPACE_END(mitsuba)
