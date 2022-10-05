#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-singleton:

Singleton spectrum (:monosp:`singleton`)
------------------------------------

This spectrum returns a constant reflectance or emission value at a fixed wavelength.

 */

template <typename Float, typename Spectrum>
class SingletonSpectrum final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    SingletonSpectrum(const Properties &props) : Texture(props) {
        m_wavelength = dr::opaque<Float>(props.get<ScalarFloat>("wavelength"));
        m_value = dr::opaque<Float>(props.get<ScalarFloat>("value"));
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &/* si */,
                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_value;
    }

    Float eval_1(const SurfaceInteraction3f & /* it */, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_value;
    }

    Wavelength pdf_spectrum(const SurfaceInteraction3f &/* si */, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return Wavelength(1.f);
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f & /*si*/,
                    const Wavelength &sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);

        if constexpr (is_monochromatic_v<Spectrum> || is_spectral_v<Spectrum>) {
            DRJIT_MARK_USED(sample);
            return { Wavelength(m_wavelength), UnpolarizedSpectrum(m_value) };
        } else {
            DRJIT_MARK_USED(sample);
            NotImplementedError("sample");
        }
    }

    Float mean() const override { return dr::mean(m_value); }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("wavelength", m_wavelength, +ParamFlags::NonDifferentiable);
        callback->put_parameter("value", m_value, +ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SingletonSpectrum[" << std::endl
            << "  wavelength = " << string::indent(m_wavelength) << "," << std::endl
            << "  value = " << string::indent(m_value) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Float m_wavelength;
    Float m_value;
};

MI_IMPLEMENT_CLASS_VARIANT(SingletonSpectrum, Texture)
MI_EXPORT_PLUGIN(SingletonSpectrum, "Singleton spectrum")
NAMESPACE_END(mitsuba)
