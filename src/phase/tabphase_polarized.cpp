#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/phase.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-tabphase_polarized:

Lookup table (polarized) phase function (:monosp:`tabphase_polarized`)
------------------------------------------------

.. pluginparameters::

 * - values
   - |string|
   - A comma-separated list of phase function values parametrized by the
     cosine of the scattering angle.
   - |exposed|, |differentiable|, |discontinuous|

This plugin implements a generic phase function model for isotropic media
parametrized by a lookup table giving values of the phase function as a
function of the cosine of the scattering angle.

.. admonition:: Notes

   * The scattering angle cosine is here defined as the dot product of the
     incoming and outgoing directions, where the incoming, resp. outgoing
     direction points *toward*, resp. *outward* the interaction point.
   * From this follows that :math:`\cos \theta = 1` corresponds to forward
     scattering.
   * Lookup table points are regularly spaced between -1 and 1.
   * Phase function values are automatically normalized.
   * For polarized phase functions, this assumes (for the time being) the structure
     of a phase function with spherically symmetric particles, i.e. there are
     only four unique elements of the Mueller matrix: `M_{11}`, `M_{12}`, 
     `M_{33}`, and `M_{34}`
*/

template <typename Float, typename Spectrum>
class TabulatedPolarizedPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    TabulatedPolarizedPhaseFunction(const Properties &props) : Base(props) {
        if (props.type("m11") == Properties::Type::String) {
            std::vector<std::string> cost_str =
                string::tokenize(props.string("cost"), " ,");
            std::vector<std::string> entry_str1, m11_str =
                string::tokenize(props.string("m11"), " ,");
            std::vector<std::string> entry_str2, m12_str =
                string::tokenize(props.string("m12"), " ,");
            std::vector<std::string> entry_str3, m33_str =
                string::tokenize(props.string("m33"), " ,");
            std::vector<std::string> entry_str4, m34_str =
                string::tokenize(props.string("m34"), " ,");

            // TODO: Check all sizes against each other
            if (cost_str.size() != m11_str.size())
                Throw("TabulatedPolarizedPhaseFunction: 'cost_str' and 'm11_str' parameters must have the same size!");

            std::vector<ScalarFloat> cost, m11, m12, m33, m34;
            cost.reserve(cost_str.size());
            m11.reserve(m11_str.size());
            m12.reserve(m12_str.size());
            m33.reserve(m33_str.size());
            m34.reserve(m34_str.size());

            for (size_t i = 0; i < cost_str.size(); ++i) {
                try {
                    cost.push_back(string::stof<ScalarFloat>(cost_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", cost_str[i]);
                }
                try {
                    m11.push_back(string::stof<ScalarFloat>(m11_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", m11_str[i]);
                }
                try {
                    m12.push_back(string::stof<ScalarFloat>(m12_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", m12_str[i]);
                }
                try {
                    m33.push_back(string::stof<ScalarFloat>(m33_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", m33_str[i]);
                }
                try {
                    m34.push_back(string::stof<ScalarFloat>(m34_str[i]));
                } catch (...) {
                    Throw("Could not parse floating point value '%s'", m34_str[i]);
                }
            }

            m_m11 = IrregularContinuousDistribution<Float>(
                cost.data(), m11.data(), m11.size()
            );
            m_m12 = IrregularContinuousDistribution<Float>(
                cost.data(), m12.data(), m12.size(), false, false
            );
            m_m33 = IrregularContinuousDistribution<Float>(
                cost.data(), m33.data(), m33.size(), false, false
            );
            m_m34 = IrregularContinuousDistribution<Float>(
                cost.data(), m34.data(), m34.size(), false, false
            );
        }

        m_flags = +PhaseFunctionFlags::Anisotropic;
        dr::set_attr(this, "flags", m_flags);
        m_components.push_back(m_flags);
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mi,
                                                 Float /* sample1 */,
                                                 const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        // Sample a direction in physics convention.
        // We sample cos θ' = cos(π - θ) = -cos θ.
        //
        // [Kate] Does this need to change?
        Float cos_theta_prime = m_m11.sample(sample2.x());
        Float sin_theta_prime =
            dr::safe_sqrt(1.f - cos_theta_prime * cos_theta_prime);
        auto [sin_phi, cos_phi] =
            dr::sincos(2.f * dr::Pi<ScalarFloat> * sample2.y());
        Vector3f wo{ sin_theta_prime * cos_phi, sin_theta_prime * sin_phi,
                     cos_theta_prime };

        // Switch the sampled direction to graphics convention and transform the
        // computed direction to world coordinates
        //
        // [Kate] We convert to world space in the integrator 
        // and our mi.sh_frame = ray.d, not -ray.d (as in the base Mitsuba 3), to be consistent with to_world_mueller
        // wo = -mi.to_world(wo);

        auto [ phase_val, phase_pdf ] = eval_pdf(ctx, mi, wo, active);
        Spectrum phase_weight = phase_val * dr::rcp(phase_pdf);

        return { wo, phase_weight, phase_pdf };
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mi,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        // The data is laid out in physics convention
        // (with cos θ = 1 corresponding to forward scattering).
        // This parameterization differs from the convention used internally by
        // Mitsuba and is the reason for the minus sign below.
        Float cos_theta = -dot(wo, mi.wi);

        Float m11 = m_m11.eval_pdf(cos_theta, active);
        Float m12 = m_m12.eval_pdf(cos_theta, active);
        Float m33 = m_m33.eval_pdf(cos_theta, active);
        Float m34 = m_m34.eval_pdf(cos_theta, active);

        Spectrum phase_val(0.f); 

        if constexpr (is_polarized_v<Spectrum>) {
            phase_val = MuellerMatrix<Float>(
                            m11, m12, 0, 0,
                            m12, m11, 0, 0,
                            0, 0, m33, m34,
                            0, 0,-m34, m33
                        );

            /* Due to the coordinate system rotations for polarization-aware
                pBSDFs below we need to know the propagation direction of light.
                In the following, light arrives along `-wo_hat` and leaves along
                `+wi_hat`. */
            Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : mi.wi,
                     wi_hat = ctx.mode == TransportMode::Radiance ? mi.wi : wo;

            /* The Stokes reference frame vector of this matrix lies in the 
                scattering plane spanned by wi and wo. */
            Vector3f x_hat = dr::cross(-wo_hat, wi_hat),
                     p_axis_in = dr::normalize(dr::cross(x_hat, -wo_hat)),
                     p_axis_out = dr::normalize(dr::cross(x_hat, wi_hat));

            /* Rotate in/out reference vector of weight s.t. it aligns with the
            implicit Stokes bases of -wo_hat & wi_hat. */
            phase_val = mueller::rotate_mueller_basis(phase_val,
                                                     -wo_hat, p_axis_in, mueller::stokes_basis(-wo_hat),
                                                      wi_hat, p_axis_out, mueller::stokes_basis(wi_hat));

            // If the cross product x_hat is too small, phase_val may be NaN
            dr::masked(phase_val, dr::isnan(phase_val)) = depolarizer<Spectrum>(0.f);
        }

        Float pdf = m_m11.eval_pdf_normalized(cos_theta, active) 
                    * dr::InvTwoPi<ScalarFloat>;

        return { phase_val, pdf };
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("m11", m_m11.pdf(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("m12", m_m12.pdf(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("m33", m_m33.pdf(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("m34", m_m34.pdf(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    void parameters_changed(const std::vector<std::string> & /*keys*/) override {
        m_m11.update();
        m_m12.update();
        m_m33.update();
        m_m34.update();
    }
    
    std::string to_string() const override {
        std::ostringstream oss;
        oss << "TabulatedPhaseFunction[" << std::endl
            << "  distr = " << string::indent(m_m11) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    IrregularContinuousDistribution<Float> m_m11;
    IrregularContinuousDistribution<Float> m_m12;
    IrregularContinuousDistribution<Float> m_m33;
    IrregularContinuousDistribution<Float> m_m34;
};

MI_IMPLEMENT_CLASS_VARIANT(TabulatedPolarizedPhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(TabulatedPolarizedPhaseFunction, "Tabulated (polarized) phase function")
NAMESPACE_END(mitsuba)
