#include <mitsuba/core/properties.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/quad.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/render/mie.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sizedistr.h>
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
    MI_IMPORT_TYPES(PhaseFunctionContext, SizeDistribution)

    using Complex2f = dr::Complex<Float>;
    using FloatX = dr::DynamicArray<dr::scalar_t<Float>>;

    MiePhaseFunction(const Properties &props) : Base(props) {
        if constexpr(is_rgb_v<Spectrum>)
            Throw("Mie phase function may only be used in monochromatic or spectral mode!");

        for (auto &[name, obj] : props.objects(false)) {
            auto *size_distr = dynamic_cast<SizeDistribution *>(obj.get());

            if (size_distr) {
                if (m_size_distr)
                    Throw("Only a single size distribution can be specified");
                m_size_distr = size_distr;
                props.mark_queried(name);
            }
        }

        auto pmgr = PluginManager::instance();
        if (!m_size_distr) {
            // Instantiate a monodisperse size distribution if none was specified
            ScalarFloat radius = props.get<ScalarFloat>("radius", 1000.f);

            Properties props_mono("monodisperse");
            props_mono.set_float("radius", radius);
            m_size_distr = static_cast<SizeDistribution *>(pmgr->create_object<SizeDistribution>(props_mono));
        }

        m_nmax = props.get<ScalarInt32>("nmax", -1);
        ScalarFloat ior_med_re = props.get<ScalarFloat>("ior_med", 1.f);
        ScalarFloat ior_med_im = props.get<ScalarFloat>("ior_med_i", 0.f);
        ScalarFloat ior_sph_re = props.get<ScalarFloat>("ior_sph", 1.33f);
        ScalarFloat ior_sph_im = props.get<ScalarFloat>("ior_sph_i", 0.f);
        m_wavelength = props.get<ScalarFloat>("wavelength", -1.f);

        if (m_nmax < -1)
            Log(Error, "The number of series terms must be positive or -1 (automatic)!");

        if (ior_med_re <= 0 || ior_sph_re <= 0)
            Log(Error, "Indices of refraction must be positive!");

        m_ior_med_re = ior_med_re;
        m_ior_med_im = ior_med_im;
        m_ior_sph_re = ior_sph_re;
        m_ior_sph_im = ior_sph_im;

        m_ior_med = Complex2f(m_ior_med_re, m_ior_med_im);
        m_ior_sph = Complex2f(m_ior_sph_re, m_ior_sph_im);

        m_flags = +PhaseFunctionFlags::Anisotropic;
        dr::set_attr(this, "flags", m_flags);
        m_components.push_back(m_flags);
    }

    Spectrum eval_mie(const PhaseFunctionContext &ctx, 
                      const MediumInteraction3f &mi, 
                      const Vector3f &wo,
                      Mask active) const {
        Spectrum phase_val(0.f);
        UnpolarizedSpectrum wavelengths_u;

        if constexpr(is_rgb_v<Spectrum>) {
            wavelengths_u = 0.f;
        } else {
            wavelengths_u = unpolarized_spectrum(mi.wavelengths);
        }

        // The direction of light propagation is +z in local space
        Float mu = Frame3f::cos_theta(wo);

        if (m_size_distr->is_monodisperse()) {
            Float radius = m_size_distr->min_radius();

            auto [s1, s2, ns, Cs, Ct] = mie<Float>(wavelengths_u, 
                                            UnpolarizedSpectrum(mu), 
                                            UnpolarizedSpectrum(radius), 
                                            dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
                                            dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
                                            m_nmax);

            if constexpr (is_polarized_v<Spectrum>) {
                phase_val = mueller::mie_scatter(s1, s2, ns);
            } else {
                phase_val = 0.5f * (dr::squared_norm(s1) + dr::squared_norm(s2)) * dr::rcp(ns);
            }
        } else {
            UnpolarizedSpectrum Cs_avg(0.f);
            Spectrum phase_r;

            if constexpr (dr::is_jit_v<Float>) {
                auto [radius, weight, sdf] = m_size_distr->eval_gauss_all();

                auto [radius_grid, mu_grid] = dr::meshgrid(radius, mu);
                auto [weight_grid, unused]  = dr::meshgrid(weight, mu);
                auto [sdf_grid, unused2]    = dr::meshgrid(sdf, mu);
                uint32_t n_gauss = m_size_distr->n_gauss();

                auto [s1, s2, ns, Cs, Ct] = mie<Float>(wavelengths_u, 
                                                       UnpolarizedSpectrum(mu_grid), 
                                                       UnpolarizedSpectrum(radius_grid), 
                                                       dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
                                                       dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
                                                       m_nmax);

                if constexpr (is_polarized_v<Spectrum>) {
                    phase_r = mueller::mie_scatter(s1, s2, ns);
                } else {
                    phase_r = 0.5f * (dr::squared_norm(s1) + dr::squared_norm(s2)) * dr::rcp(ns);
                }

                UnpolarizedSpectrum Cs_tmp = weight_grid * sdf_grid * Cs;
                Spectrum phase_tmp = weight_grid * sdf_grid * Cs * phase_r;

                Cs_avg = dr::block_sum(Cs_tmp, n_gauss);
                phase_val = dr::block_sum(phase_tmp, n_gauss);
            } else {
                // Estimate integral over radius distribution by Gaussian quadrature
                uint32_t g = m_size_distr->n_gauss();
                uint32_t i = 0;

                dr::scoped_set_flag guard(JitFlag::LoopRecord, false);

                dr::Loop<dr::mask_t<uint32_t>> loop_gauss("Integrate over distribution of sizes", 
                                            /* loop state: */ i, Cs_avg, phase_r, phase_val);

                while (loop_gauss(i < g)) {
                    auto [radius, weight, sdf] = m_size_distr->eval_gauss(i);

                    auto [s1, s2, ns, Cs, Ct] = mie<Float>(wavelengths_u, 
                                                    UnpolarizedSpectrum(mu), 
                                                    UnpolarizedSpectrum(radius), 
                                                    dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
                                                    dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
                                                    m_nmax);

                    if constexpr (is_polarized_v<Spectrum>) {
                        phase_r = mueller::mie_scatter(s1, s2, ns);
                    } else {
                        phase_r = 0.5f * (dr::squared_norm(s1) + dr::squared_norm(s2)) * dr::rcp(ns);
                    }

                    Cs_avg += weight * sdf * Cs;
                    phase_val += weight * sdf * Cs * phase_r;

                    i++;
                }
            }

            phase_val /= Cs_avg;
        }

        if constexpr (is_polarized_v<Spectrum>) {
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
            dr::masked(phase_val, dr::isnan(phase_val)) = 0.f;
        }
        
        return phase_val;
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float /* sample1 */,
                                                 const Point2f &sample,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // Otherwise drjit will complain about loops in Mie calculation
        if (jit_flag(JitFlag::Recording))
            return { Vector3f(0.f), Spectrum(0.f), 0.f };
        
        // We use a tabulated version of the Mie phase function for sampling in practice
        auto wo  = warp::square_to_uniform_sphere(sample);
        auto pdf = warp::square_to_uniform_sphere_pdf(wo);

        // Get Mueller matrix
        Spectrum phase_weight = eval_mie(ctx, mi, wo, active);

        return { wo, phase_weight, pdf };
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f & mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        
        // Otherwise drjit will complain about loops in Mie calculation
         if (jit_flag(JitFlag::Recording))
            return { Spectrum(0.f), 0.f };

        Spectrum phase_val = eval_mie(ctx, mi, wo, active);

        // We use a tabulated version of the Mie phase function for sampling in practice
        Float pdf = warp::square_to_uniform_sphere_pdf(wo);

        return { phase_val, pdf };
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("ior_med", m_ior_med_re, +ParamFlags::Differentiable);
        callback->put_parameter("ior_med_i", m_ior_med_im, +ParamFlags::Differentiable);
        callback->put_parameter("ior_sph", m_ior_sph_re, +ParamFlags::Differentiable);
        callback->put_parameter("ior_sph_i", m_ior_sph_im, +ParamFlags::Differentiable);
        callback->put_object("size_distr", m_size_distr.get(), +ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        m_ior_med = Complex2f(m_ior_med_re, m_ior_med_im);
        m_ior_sph = Complex2f(m_ior_sph_re, m_ior_sph_im);
        Base::parameters_changed();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MiePhaseFunction[" << std::endl
            << "  ior_med = " << string::indent(m_ior_med) << std::endl
            << "  ior_sph = " << string::indent(m_ior_sph) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ScalarFloat m_wavelength;
    ScalarInt32 m_nmax;
    Complex2f m_ior_med, m_ior_sph;
    Float m_ior_med_re, m_ior_med_im, m_ior_sph_re, m_ior_sph_im; // For differentiation
    ref<SizeDistribution> m_size_distr;
};

MI_IMPLEMENT_CLASS_VARIANT(MiePhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(MiePhaseFunction, "Mie phase function")
NAMESPACE_END(mitsuba)