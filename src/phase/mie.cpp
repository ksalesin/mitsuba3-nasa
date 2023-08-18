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
    using Warp2D1 = Marginal2D<Float, 1, true>;

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
        m_ndiv = props.get<ScalarInt32>("ndiv", -1);
        m_smax = props.get<ScalarInt32>("smax", 100);
        m_numerical_accuracy = props.get<ScalarFloat>("m_numerical_accuracy", 1e-7f);
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

        // if (m_wavelength != -1.f)
        //     calculate_spherical_expansion();
    }

    /**
     * \brief Calculate next Wigner d-function by upward recurrence.
     * (see "Scattering, Absorption, and Emission of Light by Small Particles," 
     * Mishchenko et al, p. 365)
     */
    MI_INLINE 
    ScalarFloat wignerd(ScalarFloat d_0, ScalarFloat d_1, ScalarFloat var, 
                        ScalarInt32 m, ScalarInt32 n, ScalarInt32 s) const {
        ScalarFloat x = dr::cos(var),
                    spp = s + 1,
                    spp_sq = dr::sqr(spp),
                    s_sq = dr::sqr(s),
                    m_sq = dr::sqr(m),
                    n_sq = dr::sqr(n);

        ScalarFloat a = s * dr::sqrt(spp_sq - m_sq) * dr::sqrt(spp_sq - n_sq),
                    b = (2 * s + 1) * (s * spp * x - m * n) * d_1,
                    c = spp * dr::sqrt(s_sq - m_sq) * dr::sqrt(s_sq - n_sq) * d_0;

        return dr::rcp(a) * (b - c);
    }

    /**
     * \brief Construct generalized spherical functions for efficient evaluation
     * of Mie phase function at runtime.
     */
    // void calculate_spherical_expansion() {
    //     using FloatX = dr::DynamicArray<dr::scalar_t<Float>>;
    //     UnpolarizedSpectrum wavelengths_u(m_wavelength);

    //     ScalarFloat min_radius = m_size_distr->min_radius();
    //     ScalarFloat max_radius = m_size_distr->max_radius();
    //     auto radius_intervals = dr::linspace<FloatX>(min_radius, max_radius, m_ndiv + 1);

    //     // Gaussian quadrature nodes & weights for integration over radius
    //     uint32_t g_r = m_size_distr->n_gauss();
    //     auto [nodes_r, weights_r] = quad::gauss_legendre<FloatX>(g_r);

    //     // Average scattering cross section
    //     UnpolarizedSpectrum Cs_avg(0.f);

    //     // Store expensive variables for later reuse
    //     uint32_t n_r = g_r * m_ndiv;
    //     std::vector<ScalarFloat> radii(n_r), sdf_r(n_r);
    //     std::vector<UnpolarizedSpectrum> Cs_r(n_r);
        
    //     printf("Calculating average scattering cross section");

    //     // Calculate average scattering cross section
    //     for (uint32_t d = 0; d < m_ndiv; ++d) {
    //         printf(".");
    //         min_radius = radius_intervals[d];
    //         max_radius = radius_intervals[d + 1];
    //         ScalarFloat scale = max_radius - min_radius, radius, sdf;

    //         for (uint32_t i = 0; i < g_r; i++) {
    //             uint32_t idx = d * g_r + i;
    //             ScalarFloat node_i = nodes_r[i],
    //                         weight_i = weights_r[i];

    //             // Scale node interval from [-1, 1] to [min_radius, max_radius]
    //             radius = dr::fmadd(node_i, 0.5f, 0.5f);
    //             radius = dr::fmadd(radius, scale, min_radius);

    //             Float sdf_ = m_size_distr->eval(radius);
    //             if constexpr (dr::is_jit_v<Float>)
    //                 sdf = sdf_[0];
    //             else
    //                 sdf = sdf_;

    //             auto [Cs, Ct] = mie_xsections(wavelengths_u,
    //                                           UnpolarizedSpectrum(radius), 
    //                                           dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
    //                                           dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
    //                                           m_nmax);

    //             radii[idx] = radius;
    //             sdf_r[idx] = sdf;
    //             Cs_r[idx]  = Cs;
    //             Cs_avg    += weight_i * sdf * Cs;

    //             dr::schedule(Cs_avg);
    //             dr::eval();
    //         }
    //     }

    //     printf("done.\n");

    //     // Gaussian quadrature nodes & weights for integration over theta
    //     Complex2f kx = dr::TwoPi<Float> * m_ior_med * dr::rcp(m_wavelength);
    //     Complex2f x = kx * max_radius;
    //     Float x_norm = dr::norm(x);

    //     ScalarInt32 g_t = (ScalarInt32) dr::max_nested(2.f * (8 + x_norm + 4.05f * dr::pow(x_norm, 1.f / 3.f)));
    //     auto [nodes_t, weights_t] = quad::gauss_legendre<FloatX>(g_t);
        
    //     // Spherical functions
    //     m_alpha1 = std::vector<ScalarFloat>(m_smax);
    //     m_alpha4 = std::vector<ScalarFloat>(m_smax);
    //     m_beta1  = std::vector<ScalarFloat>(m_smax);
    //     m_beta2  = std::vector<ScalarFloat>(m_smax);

    //     printf("Calculating expansion into spherical funtions");

    //     // Calculate spherical functions
    //     for (uint32_t j = 0; j < g_t; ++j) {
    //         printf(".");
    //         ScalarFloat node_j = nodes_t[j],
    //                     weight_j = weights_t[j],
    //                     cos_theta = node_j,
    //                     theta = dr::acos(node_j);

    //         Spectrum phase_val(0.f);

    //         // Integrate over n_div subintervals of [min_radius, max_radius]
    //         for (uint32_t d = 0; d < m_ndiv; ++d) {
    //             for (uint32_t i = 0; i < g_r; i++) {
    //                 uint32_t idx = d * g_r + i;
    //                 ScalarFloat weight_i   = weights_r[i],
    //                             radius     = radii[idx],
    //                             sdf        = sdf_r[idx];
    //                 UnpolarizedSpectrum Cs = Cs_r[idx];

    //                 // Get phase function value for this node
    //                 auto [s1, s2, ns] = mie_s1s2(wavelengths_u, 
    //                                             UnpolarizedSpectrum(cos_theta), 
    //                                             UnpolarizedSpectrum(radius), 
    //                                             dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
    //                                             dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
    //                                             m_nmax);

    //                 Spectrum phase_r;
    //                 if constexpr (is_polarized_v<Spectrum>) {
    //                     phase_r = mueller::mie_scatter(s1, s2, ns);
    //                 } else {
    //                     phase_r = 0.5f * (dr::squared_norm(s1) + dr::squared_norm(s2)) * dr::rcp(ns);
    //                 }

    //                 phase_val += weight_i * sdf * Cs * phase_r;
    //                 dr::schedule(phase_val);
    //                 dr::eval();
    //             }
    //         }

    //         phase_val /= Cs_avg;

    //         ScalarFloat a1, a4, b1, b2;
    //         if constexpr(is_polarized_v<Spectrum>) {
    //             a1 = phase_val[0][0][0][0];
    //             a4 = phase_val[0][0][3][3];
    //             b1 = phase_val[0][0][0][1];
    //             b2 = phase_val[0][0][3][2]; // TODO: Double check this indexing
    //         } else {
    //             // TODO
    //         }

    //         // Variables for upward recurrences of Wigner d-functions
    //         ScalarFloat d00_0 = 0.f, d00_1 = 0.f, d02_0 = 0.f, d02_1 = 0.f;

    //         // betas are summed from s = 2 and up
    //         m_beta1[0] = 0.f; m_beta1[1] = 0.f; m_beta2[0] = 0.f; m_beta2[1] = 0.f;
            
    //         for (uint32_t s = 0; s < m_smax; s++) {
    //             // Calculate next Wigner d-functions by upward recurrence
    //             ScalarFloat d00_s, d02_s;
    //             if (s == 0) {
    //                 d00_s = 1.f;
    //             } else {
    //                 d00_s = wignerd(d00_0, d00_1, theta, 0, 0, s - 1);
    //             }
    //             d00_0 = d00_1; d00_1 = d00_s;

    //             if (s < 2) {
    //                 d02_s = 0.f;
    //             } else if (s == 2) {
    //                 d02_s = 0.25f * dr::sqrt(6.f) * (1 - cos_theta) * (1 + cos_theta);
    //             } else {
    //                 d02_s = wignerd(d02_0, d02_1, theta, 0, 0, s - 1);
    //             }

    //             m_alpha1[s] += weight_j * a1 * d00_s;
    //             m_alpha4[s] += weight_j * a4 * d00_s;
    //             m_beta1[s]  += weight_j * b1 * d02_s;
    //             m_beta2[s]  += weight_j * b2 * d02_s;
    //         }
    //     }
    //     printf("done.\n");

    //     Log(Info, "S,ALPHA 1,ALPHA 4,BETA 1,BETA 2");

    //     for (uint32_t s = 0; s < m_smax; s++) {
    //         m_alpha1[s] *= (s + 0.5f);
    //         m_alpha4[s] *= (s + 0.5f);
    //         m_beta1[s]  *= -1 * (s + 0.5f);
    //         m_beta2[s]  *= -1 * (s + 0.5f);

    //         Log(Info, "%d,%f,%f,%f,%f", s, m_alpha1[s], m_alpha4[s], m_beta1[s], m_beta2[s]);
    //     }
    // }

    /**
     * \brief Construct a sampling distribution for phase function 
     * due to its spikiness and complexity of evaluation.
     */
    // void tabulate_mie() {
    //     ScalarFloat wavelength_min = MI_CIE_MIN;
    //     ScalarFloat wavelength_max = MI_CIE_MAX;
    //     ScalarFloat wavelength_range = wavelength_max - wavelength_min;
    //     uint32_t wavelength_res = (uint32_t) (wavelength_max - wavelength_min); // 1 nm resolution

    //     uint32_t theta_res = 180;
    //     uint32_t phi_res = 2;

    //     // Phase function evaluated at theta_res x phi_res grid points
    //     std::vector<ScalarFloat> wavelengths(wavelength_res);  
    //     std::vector<ScalarFloat> values(wavelength_res * theta_res * phi_res);

    //     for (size_t v = 0; v < wavelength_res; v++) {
    //         ScalarFloat wavelength = wavelengths[v] = wavelength_min + (ScalarFloat) v * wavelength_range / wavelength_res;
    //         uint32_t pfx = theta_res * phi_res * v;

    //         for (size_t i = 0; i < theta_res; i++) {
    //             ScalarFloat theta = dr::Pi<ScalarFloat> * (ScalarFloat) i / (theta_res - 1);
    //             ScalarFloat mu = dr::cos(theta);

    //             auto [s1, s2, ns] = mie_s1s2(wavelength, mu, m_r, m_ior_med, m_ior_sph, m_nmax);

    //             // Phase function value
    //             auto value = 0.5f * (dr::squared_norm(s1) + dr::squared_norm(s2)) * dr::rcp(ns);

    //             for (size_t j = 0; j < phi_res; j++) {
    //                 // Phase function does not actually depend on phi
    //                 values[pfx + j * theta_res + i] = value;
    //             }
    //         }
    //     }

    //     m_pf = Warp2D1(
    //         values.data(),
    //         ScalarVector2u(theta_res, phi_res),
    //         {{
    //             (uint32_t) wavelength_res
    //         }},
    //         {{
    //             wavelengths.data()
    //         }},
    //         false, true
    //     );
    // }

    /**
     * Numerically stable method computing the elevation of the given
     * (normalized) vector in the local frame.
     * Conceptually equivalent to:
     *     safe_acos(Frame3f::cos_theta(d))
     */
    auto elevation(const Vector3f &d) const {
        auto dist = dr::sqrt(dr::sqr(d.x()) + dr::sqr(d.y()) + dr::sqr(d.z() - 1.f));
        return 2.f * dr::safe_asin(.5f * dist);
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

        // Get radius integration interval
        ScalarFloat min_radius = m_size_distr->min_radius();
        ScalarFloat max_radius = m_size_distr->max_radius();

        if (min_radius == max_radius) {
            auto [s1, s2, ns] = mie_s1s2(wavelengths_u, 
                                         UnpolarizedSpectrum(mu), 
                                         UnpolarizedSpectrum(min_radius), 
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

            // Estimate integral over radius distribution by Gaussian quadrature
            uint32_t g = m_size_distr->n_gauss();

            for (uint32_t i = 0; i < g; i++) {
                auto [radius, weight, sdf] = m_size_distr->eval_gauss(i);

                auto [s1, s2, ns] = mie_s1s2(wavelengths_u, 
                                             UnpolarizedSpectrum(mu), 
                                             UnpolarizedSpectrum(radius), 
                                             dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
                                             dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
                                             m_nmax);

                auto [Cs, Ct] = mie_xsections(wavelengths_u,
                                              UnpolarizedSpectrum(radius), 
                                              dr::Complex<UnpolarizedSpectrum>(m_ior_med), 
                                              dr::Complex<UnpolarizedSpectrum>(m_ior_sph), 
                                              m_nmax);
   
                Spectrum phase_r;
                if constexpr (is_polarized_v<Spectrum>) {
                    phase_r = mueller::mie_scatter(s1, s2, ns);
                } else {
                    phase_r = 0.5f * (dr::squared_norm(s1) + dr::squared_norm(s2)) * dr::rcp(ns);
                }

                // Some rare nan's can pop up in Mie computation
                dr::masked(Cs, dr::isnan(Cs)) = 0.f;
                dr::masked(phase_r, dr::isnan(phase_r)) = 0.f;

                Cs_avg += weight * sdf * Cs;
                phase_val += weight * sdf * Cs * phase_r;
                dr::schedule(Cs_avg, phase_val);
                dr::eval();
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

    std::pair<Vector3f, Spectrum> sample(const PhaseFunctionContext & ctx,
                                      const MediumInteraction3f & mi,
                                      Float /* sample1 */,
                                      const Point2f & sample2,
                                      Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);
        
        // Workaround for now, which wavelength would we choose for full spectral renderings?
        Float params[1] = { mi.wavelengths[0] };

        // Draw direction sample
        auto [u_wo, pdf] = m_pf.sample(sample2, params, active);

        // Unit coordinates -> spherical coordinates
        Float theta_o = u2theta(u_wo.x()),
              phi_o   = u2phi(u_wo.y());

        // Spherical coordinates -> Cartesian coordinates
        auto [sin_theta_o, cos_theta_o] = dr::sincos(theta_o);
        auto [sin_phi_o, cos_phi_o] = dr::sincos(phi_o);

        // Direction in local space (wi in +z direction)
        Vector3f wo(sin_theta_o * cos_phi_o, 
                    sin_theta_o * sin_phi_o, 
                    cos_theta_o);

        // Get Mueller matrix (TODO: use tabulated phase function instead)
        Spectrum phase_val = eval_mie(ctx, mi, wo, active);

        return { wo, (phase_val / pdf) & active };
    }

    Spectrum eval(const PhaseFunctionContext &ctx, const MediumInteraction3f &mi,
               const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        return eval_mie(ctx, mi, wo, active);
    }

     Float pdf(const PhaseFunctionContext &ctx, const MediumInteraction3f &mi,
               const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);
        
        // Cartesian coordinates -> spherical coordinates
        Float theta_o = elevation(wo),
              phi_o   = dr::atan2(wo.y(), wo.x());

        // Spherical coordinates -> unit coordinates
        Vector2f u_wo(theta2u(theta_o), phi2u(phi_o));

        Float params[1] = { mi.wavelengths[0] };

        auto [sample, pdf] = m_pf.invert(u_wo, params, active);

        return dr::select(active, pdf, 0.f);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("ior_med", m_ior_med_re, +ParamFlags::Differentiable);
        callback->put_parameter("ior_med_i", m_ior_med_im, +ParamFlags::Differentiable);
        callback->put_parameter("ior_sph", m_ior_sph_re, +ParamFlags::Differentiable);
        callback->put_parameter("ior_sph_i", m_ior_sph_im, +ParamFlags::Differentiable);
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
    template <typename Value> Value u2theta(Value u) const {
        return u * dr::Pi<Value>;
    }

    template <typename Value> Value u2phi(Value u) const {
        return (2.f * u - 1.f) * dr::Pi<Value>;
    }

    template <typename Value> Value theta2u(Value theta) const {
        return dr::clamp(theta * dr::InvPi<Value>, 0.f, 1.f);
    }

    template <typename Value> Value phi2u(Value phi) const {
        return dr::clamp(phi * dr::InvTwoPi<Value> + 0.5f, 0.f, 1.f);
    }

    ScalarFloat m_wavelength;
    ScalarInt32 m_nmax;
    ScalarInt32 m_ndiv;
    ScalarFloat m_numerical_accuracy;
    Complex2f m_ior_med, m_ior_sph;
    Float m_ior_med_re, m_ior_med_im, m_ior_sph_re, m_ior_sph_im; // For differentiating
    ref<SizeDistribution> m_size_distr;

    // Expansion of ensemble-averaged phase function into generalized spherical fcts.
    // (see "Scattering, Absorption, and Emission of Light by Small Particles," 
    // Mishchenko et al, p. 103-105)
    std::vector<ScalarFloat> m_alpha1;
    std::vector<ScalarFloat> m_alpha4;
    std::vector<ScalarFloat> m_beta1;
    std::vector<ScalarFloat> m_beta2;
    ScalarInt32 m_smax;

    Warp2D1 m_pf;
};

MI_IMPLEMENT_CLASS_VARIANT(MiePhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(MiePhaseFunction, "Mie phase function")
NAMESPACE_END(mitsuba)