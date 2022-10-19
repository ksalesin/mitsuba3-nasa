#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/render/mie.h>
#include <mitsuba/render/phase.h>
#include <drjit/complex.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-mie:

Mie phase function (:monosp:`mie`)
----------------------------------

.. list-table::
 :widths: 20 15 65
 :header-rows: 1
 :class: paramstable

 * - Parameter
   - Type
   - Description
 * - radius
   - |float|
   - Radius of the sphere (units of distance). (Default: 10000)
 * - ior_med
   - |float|
   - Real component of the index of refraction of the host medium. (Default: 1)
 * - ior_med_i
   - |float|
   - Imaginary component of the index of refraction of the host medium (a negative value implies absorption). (Default: 0)
 * - ior_sph
   - |float|
   - Real component of the index of refraction of the sphere. (Default: 1.33)
*  - ior_sph_i
   - |float|
   - Imaginary component of the index of refraction of the sphere (a negative value implies absorption). (Default: 0)
*  - nmax
   - |int|
   - Number of terms in the infinite series that should be used (-1: automatic)

This plugin implements the phase function of a dielectric sphere using Lorenz-Mie theory.

Follows the theory and recurrences outlined in
"Far-field Lorenz-Mie scattering in an absorbing host medium: 
Theoretical formalism and FORTRAN program" by Mishchenko and Yang (JQSRT 2018).

Units of distance are arbitrary but need to be consistent between
``radius``, ``wavelength``, and the imaginary components of the provided
refractive indices.
*/
template <typename Float, typename Spectrum>
class MiePhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    using Complex2f = dr::Complex<ScalarFloat>;

    MiePhaseFunction(const Properties &props) : Base(props) {
        if constexpr(is_rgb_v<Spectrum>)
            Throw("Mie phase function may only be used in monochromatic or spectral mode!");

        m_r = props.get<ScalarFloat>("radius", 10000.f);
        m_nmax = props.get<ScalarInt32>("nmax", -1);
        ScalarFloat ior_med_re = props.get<ScalarFloat>("ior_med", 1.f);
        ScalarFloat ior_med_im = props.get<ScalarFloat>("ior_med_i", 0.f);
        ScalarFloat ior_sph_re = props.get<ScalarFloat>("ior_sph", 1.33f);
        ScalarFloat ior_sph_im = props.get<ScalarFloat>("ior_sph_i", 0.f);

        if (m_r <= 0)
            Log(Error, "The radius of the spheres must be positive!");

        if (m_nmax < -1)
            Log(Error, "The number of series terms must be positive or -1 (automatic)!");

        if (ior_med_re <= 0 || ior_sph_re <= 0)
            Log(Error, "Indices of refraction must be positive!");

        m_ior_med = Complex2f(ior_med_re, ior_med_im);
        m_ior_sph = Complex2f(ior_sph_re, ior_sph_im);
        
        m_flags = +PhaseFunctionFlags::Anisotropic;
        dr::set_attr(this, "flags", m_flags);
        m_components.push_back(m_flags);
    }

    Spectrum eval_mie(const PhaseFunctionContext &ctx, 
                      const MediumInteraction3f &mi, 
                      const Vector3f &wo,
                      Mask active) const {
        Spectrum phase_val;
        UnpolarizedSpectrum wavelengths_u;

        if constexpr(is_rgb_v<Spectrum>) {
            wavelengths_u = 0.f;
        } else {
            wavelengths_u = unpolarized_spectrum(mi.wavelengths);
        }

        // The direction of light propagation is +z in local space
        Float mu = Frame3f::cos_theta(wo);
        
        auto [s1, s2, ns] = mie_s1s2(wavelengths_u, 
                                     UnpolarizedSpectrum(mu), 
                                     UnpolarizedSpectrum(m_r), 
                                     dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
                                     dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
                                     m_nmax);

        if constexpr (is_polarized_v<Spectrum>) {
            phase_val = mueller::mie_scatter(s1, s2, ns);

            /* Due to the coordinate system rotations for polarization-aware
                pBSDFs below we need to know the propagation direction of light.
                In the following, light arrives along `-wo_hat` and leaves along
                `+wi_hat`. */
            Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : mi.wi,
                     wi_hat = ctx.mode == TransportMode::Radiance ? mi.wi : wo;

            /* The Stokes reference frame vector of this matrix lies in the 
                scattering plane spanned by wi and wo.
            
                See Fig. A.1 in "Optical Polarization in Biomedical Applications" (Appendix A)
                by Tuchin, Wang, and Zimnyakov (2006). */
            Vector3f x_hat = dr::cross(-wo_hat, wi_hat),
                     p_axis_in = dr::normalize(dr::cross(x_hat, -wo_hat)),
                     p_axis_out = dr::normalize(dr::cross(x_hat, wi_hat));

            /* Rotate in/out reference vector of weight s.t. it aligns with the
            implicit Stokes bases of -wo_hat & wi_hat. */
            phase_val = mueller::rotate_mueller_basis(phase_val,
                                                     -wo_hat, p_axis_in, mueller::stokes_basis(-wo_hat),
                                                      wi_hat, p_axis_out, mueller::stokes_basis(wi_hat));

            // If the cross product x_hat is too small, M may be NaN due to normalize()
            dr::masked(phase_val, dr::any_nested(dr::isnan(phase_val))) = depolarizer<Spectrum>(0.f);
        } else {
            phase_val = 0.5f * (dr::squared_norm(s1) + dr::squared_norm(s2)) * dr::rcp(ns);
        }
        
        return phase_val;
    }

    std::pair<Vector3f, Spectrum> sample(const PhaseFunctionContext & /* ctx */,
                                      const MediumInteraction3f & /* mi */,
                                      Float /* sample1 */,
                                      const Point2f & /* sample2 */,
                                      Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);
        NotImplementedError("sample");
    }

    Spectrum eval(const PhaseFunctionContext & ctx, const MediumInteraction3f &mi,
               const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        return eval_mie(ctx, mi, wo, active);
    }

     Float pdf(const PhaseFunctionContext & /* ctx */, const MediumInteraction3f & /* mi */,
               const Vector3f & /* wo */, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        NotImplementedError("pdf");
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("radius", m_r, +ParamFlags::Differentiable);
        callback->put_parameter("ior_med", m_ior_med, +ParamFlags::Differentiable);
        callback->put_parameter("ior_sph", m_ior_sph, +ParamFlags::Differentiable);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MiePhaseFunction[" << std::endl
            << "  radius = " << string::indent(m_r) << std::endl
            << "  ior_med = " << string::indent(m_ior_med) << std::endl
            << "  ior_sph = " << string::indent(m_ior_sph) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ScalarFloat m_r;
    ScalarInt32 m_nmax;
    Complex2f m_ior_med, m_ior_sph;
};

MI_IMPLEMENT_CLASS_VARIANT(MiePhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(MiePhaseFunction, "Mie phase function")
NAMESPACE_END(mitsuba)