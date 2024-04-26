#include <mitsuba/core/properties.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _phase-blendphase:

Blended phase function (:monosp:`blendphase`)
---------------------------------------------

.. pluginparameters::

 * - weight
   - |float| or |texture|
   - A floating point value or texture with values between zero and one.
     The extreme values zero and one activate the first and second nested phase
     function respectively, and in-between values interpolate accordingly.
     (Default: 0.5)
   - |exposed|, |differentiable|

 * - (Nested plugin)
   - |phase|
   - Two nested phase function instances that should be mixed according to the
     specified blending weight
   - |exposed|, |differentiable|

This plugin implements a *blend* phase function, which represents linear
combinations of two phase function instances. Any phase function in Mitsuba 3
(be it isotropic, anisotropic, micro-flake ...) can be mixed with others in this
manner. This is of particular interest when mixing components in a participating
medium (*e.g.* accounting for the presence of aerosols in a Rayleigh-scattering
atmosphere).
The association of nested Phase plugins with the two positions in the
interpolation is based on the alphanumeric order of their identifiers.

.. tabs::
    .. code-tab:: xml

        <phase type="blendphase">
            <float name="weight" value="0.5"/>
            <phase name="phase_0" type="isotropic" />
            <phase name="phase_1" type="hg">
                <float name="g" value="0.2"/>
            </phase>
        </phase>

    .. code-tab:: python

        'type': 'blendphase',
        'weight': 0.5,
        'phase_0': {
            'type': 'isotropic'
        },
        'phase_1': {
            'type': 'hg',
            'g': 0.2
        }

*/

template <typename Float, typename Spectrum>
class BlendPhaseFunction final : public PhaseFunction<Float, Spectrum> {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags, m_components)
    MI_IMPORT_TYPES(PhaseFunctionContext, PhaseFunctionPtr, Volume)

    BlendPhaseFunction(const Properties &props) : Base(props) {
        int phase_index = 0;

        for (auto &[name, obj] : props.objects(false)) {
            auto *phase = dynamic_cast<Base *>(obj.get());
            if (phase) {
                m_nested_phase.push_back(phase);
                props.mark_queried(name);
                phase_index++;
            }
        }

        if (phase_index < 2)
            Throw("BlendPhase: At least two child phase functions must be specified!");

        for (size_t i = 0; i < phase_index - 1; ++i) {
            m_weight.push_back(props.volume<Volume>("weight_" + std::to_string(i)));
        }

        m_components.clear();
        for (size_t i = 0; i < phase_index; ++i)
            for (size_t j = 0; j < m_nested_phase[i]->component_count(); ++j)
                m_components.push_back(m_nested_phase[i]->flags(j));

        m_flags = 0;
        for (size_t i = 0; i < phase_index; ++i)
            m_flags |= m_nested_phase[i]->flags();
        
        dr::set_attr(this, "flags", m_flags);

        m_phase_dr = dr::load<DynamicBuffer<PhaseFunctionPtr>>(m_nested_phase.data(),
                                                               m_nested_phase.size());
    }

    void traverse(TraversalCallback *callback) override {
        for (size_t i = 0; i < m_nested_phase.size() - 1; ++i) {
            callback->put_object("weight_" + std::to_string(i), m_weight[i].get(), 
                                 +ParamFlags::Differentiable);
        }
        for (size_t i = 0; i < m_nested_phase.size(); ++i) {
            callback->put_object("phase_" + std::to_string(i), m_nested_phase[i].get(), 
                                 +ParamFlags::Differentiable);
        }
    }

    std::tuple<Vector3f, Spectrum, Float> sample(const PhaseFunctionContext &ctx,
                                                 const MediumInteraction3f &mei,
                                                 Float sample1, const Point2f &sample2,
                                                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionSample, active);

        Float weight_sum = 0.f;
        std::vector<Float> weight;
        std::vector<Float> cdf;

        for (size_t i = 0; i < m_nested_phase.size() - 1; ++i) {
            Float w_i = eval_weight(mei, i, active);
            weight.push_back(w_i);

            weight_sum += w_i;
            cdf.push_back(weight_sum);
        }

        // Infer the weight of the last phase function
        weight.push_back((Float) 1.f - weight_sum);
        cdf.push_back((Float) 1.f);

        if (unlikely(ctx.component != (uint32_t) -1)) {
            PhaseFunctionContext ctx2(ctx);
            uint32_t component_sum = 0, index = 0;

            for (size_t i = 0; i < m_nested_phase.size(); ++i) {
                uint32_t new_sum = component_sum + m_nested_phase[i]->component_count();
                if (ctx.component < new_sum) {
                    index = i;
                    break;
                }
                component_sum = new_sum;
            }

            ctx2.component = ctx.component - component_sum;

            auto [wo, w, pdf] = m_nested_phase[index]->sample(
                ctx2, mei, sample1, sample2, active);

            w *= weight[index];
            pdf *= weight[index];
            
            return { wo, w, pdf };
        }

        // UInt32 idx_u = dr::zeros<UInt32>();
        Float sample1_adjusted = 0.f;
        Float last_cdf = 0.f;

        // for (size_t i = 0; i < m_nested_phase.size(); ++i) {
        //     auto sample_i = sample1 > last_cdf && sample1 < cdf[i];
        //     dr::masked(idx_u, sample_i) = i;
        //     dr::masked(sample1_adjusted, sample_i) = (sample1 - last_cdf) / (cdf[i] - last_cdf);
        //     last_cdf = cdf[i];
        // }

        Vector3f wo(0.f);
        Spectrum w(0.f);
        Float pdf = 0.f;

        for (size_t i = 0; i < m_nested_phase.size(); ++i) {
            Mask sample_i = active && sample1 > last_cdf && sample1 < cdf[i];
            
            if (dr::any_or<true>(sample_i)) {
                dr::masked(sample1_adjusted, sample_i) = (sample1 - last_cdf) / (cdf[i] - last_cdf);
                auto [wo_i, w_i, pdf_i] = m_nested_phase[i]->sample(ctx, mei, sample1_adjusted, sample2, sample_i);
                dr::masked(wo, sample_i) = wo_i;
                dr::masked(w, sample_i) = w_i;
                dr::masked(pdf, sample_i) = pdf_i;
            }
            
            last_cdf = cdf[i];
        }

        // PhaseFunctionPtr phase = dr::gather<PhaseFunctionPtr>(m_phase_dr, idx_u, active);
        // auto [wo, w, pdf] = phase->sample(ctx, mei, sample1_adjusted, sample2, active);

        return { wo, w, pdf };
    }

    MI_INLINE Float eval_weight(const MediumInteraction3f &mei,
                                const uint32_t index,
                                const Mask &active) const {
        return dr::clamp(m_weight[index]->eval_1(mei, active), 0.f, 1.f);
    }

    std::pair<Spectrum, Float> eval_pdf(const PhaseFunctionContext &ctx,
                                        const MediumInteraction3f &mei,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::PhaseFunctionEvaluate, active);

        Float weight_sum = 0.f;
        std::vector<Float> weight;

        for (size_t i = 0; i < m_nested_phase.size() - 1; ++i) {
            Float w_i = eval_weight(mei, i, active);
            weight.push_back(w_i);

            weight_sum += w_i;
        }

        // Infer the weight of the last phase function
        weight.push_back((Float) 1.f - weight_sum);

        if (unlikely(ctx.component != (uint32_t) -1)) {
            PhaseFunctionContext ctx2(ctx);
            uint32_t component_sum = 0, index;

            for (size_t i = 0; i < m_nested_phase.size(); ++i) {
                uint32_t new_sum = component_sum + m_nested_phase[i]->component_count();
                if (ctx.component < new_sum) {
                    index = i;
                    break;
                }
                component_sum = new_sum;
            }

            ctx2.component = ctx.component - component_sum;

            auto [val, pdf] = m_nested_phase[index]->eval_pdf(ctx2, mei, wo, active);

            val *= weight[index];
            pdf *= weight[index];

             return { val, pdf };
        } else {
            Spectrum val = 0.f;
            Float pdf = 0.f;

            for (size_t i = 0; i < m_nested_phase.size(); ++i) {
                auto [val_i, pdf_i] = m_nested_phase[i]->eval_pdf(ctx, mei, wo, active);

                val += val_i * weight[i];
                pdf += pdf_i * weight[i];
            }
            
            return { val, pdf };
        }
    }
    
    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BlendPhase[" << std::endl
            << "  weight = " << string::indent(m_weight) << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    std::vector<ref<Volume>> m_weight;
    std::vector<ref<Base>> m_nested_phase;
    DynamicBuffer<PhaseFunctionPtr> m_phase_dr;
};

MI_IMPLEMENT_CLASS_VARIANT(BlendPhaseFunction, PhaseFunction)
MI_EXPORT_PLUGIN(BlendPhaseFunction, "Blended phase function")
NAMESPACE_END(mitsuba)
