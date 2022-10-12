#include <random>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>


NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-volpath:

Atmosphere-ocean volumetric path tracer (:monosp:`volpathaos`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)

This plugin is basically the same as :ref:`the simple volumetric path tracer <integrator-volpath>`,
except for some optimizations based on assumptions about the scene.

It assumes the only light source is :ref:`directional <emitter-distant>` and enters the scene from above all elements.
It assumes there is at most one refractive BSDF in the scene (:ref:`dielectric <bsdf-dielectric>` or 
:ref:`rough dielectric <bsdf-roughdielectric>`).

*/
template <typename Float, typename Spectrum>
class VolumetricPathAOSIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                     Medium, MediumPtr, PhaseFunctionContext)

    VolumetricPathAOSIntegrator(const Properties &props) : Base(props) {
    }

    MI_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, dr::eq(idx, 1u)) = spec[1];
            dr::masked(m, dr::eq(idx, 2u)) = spec[2];
        } else {
            DRJIT_MARK_USED(idx);
        }
        return m;
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium *initial_medium,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // If there is an environment emitter and emitters are visible: all rays will be valid
        // Otherwise, it will depend on whether a valid interaction is sampled
        Mask valid_ray = !m_hide_emitters && dr::neq(scene->environment(), nullptr);

        // For now, don't use ray differentials
        Ray3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Spectrum throughput(1.f), result(0.f);
        MediumPtr medium = initial_medium;
        MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
        Mask specular_chain = active && !m_hide_emitters;
        UInt32 depth = 0;

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) dr::array_size_v<Spectrum>;
            channel = (UInt32) dr::minimum(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Mask needs_intersection = true;

        // We need to keep track of whether there is a refractive transmissive surface between the current location 
        // and the light source that requires special importance sampling for NEE (e.g. dielectric or microfacet surface)
        BSDFPtr refractive_bsdf = nullptr;

        /* Set up a Dr.Jit loop (optimizes away to a normal loop in scalar mode,
           generates wavefront or megakernel renderer based on configuration).
           Register everything that changes as part of the loop here */
        dr::Loop<Mask> loop("Volpath integrator",
                            /* loop state: */ active, depth, ray, throughput,
                            result, si, mei, medium, eta, needs_intersection,
                            refractive_bsdf, specular_chain, valid_ray, sampler);

        while (loop(active)) {
            // ----------------- Handle termination of paths ------------------
            // Russian roulette: try to keep path weights equal to one, while accounting for the
            // solid angle compression at refractive index boundaries. Stop with at least some
            // probability to avoid  getting stuck (e.g. due to total internal reflection)
            active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            Float q = dr::minimum(dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), .95f);
            Mask perform_rr = (depth > (uint32_t) m_rr_depth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            dr::masked(throughput, perform_rr) *= dr::rcp(dr::detach(q));

            active &= depth < (uint32_t) m_max_depth;
            if (dr::none_or<false>(active))
                break;

            // ----------------------- Sampling the RTE -----------------------
            Mask active_medium  = active && dr::neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            Mask act_null_scatter = false, act_medium_scatter = false,
                 escaped_medium = false;

            // If the medium does not have a spectrally varying extinction,
            // we can perform a few optimizations to speed up rendering
            Mask is_spectral = active_medium;
            Mask not_spectral = false;
            if (dr::any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (dr::any_or<true>(active_medium)) {
                mei = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                dr::masked(ray.maxt, active_medium && medium->is_homogeneous() && mei.is_valid()) = mei.t;
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);
                needs_intersection &= !(active_medium && si.is_valid());

                dr::masked(mei.t, active_medium && (si.t < mei.t)) = dr::Infinity<Float>;
                if (dr::any_or<true>(is_spectral)) {
                    auto [tr, free_flight_pdf] = medium->eval_tr_and_pdf(mei, si, is_spectral);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    dr::masked(throughput, is_spectral) *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();

                // Handle null and real scatter events
                Mask null_scatter = sampler->next_1d(active_medium) >= index_spectrum(mei.sigma_t, channel) / index_spectrum(mei.combined_extinction, channel);

                act_null_scatter |= null_scatter && active_medium;
                act_medium_scatter |= !act_null_scatter && active_medium;

                if (dr::any_or<true>(is_spectral && act_null_scatter))
                    dr::masked(throughput, is_spectral && act_null_scatter) *=
                        mei.sigma_n * index_spectrum(mei.combined_extinction, channel) /
                        index_spectrum(mei.sigma_n, channel);

                dr::masked(depth, act_medium_scatter) += 1;
            }

            // Dont estimate lighting if we exceeded number of bounces
            active &= depth < (uint32_t) m_max_depth;
            act_medium_scatter &= active;

            if (dr::any_or<true>(act_null_scatter)) {
                dr::masked(ray.o, act_null_scatter) = mei.p;
                dr::masked(si.t, act_null_scatter) = si.t - mei.t;
            }

            if (dr::any_or<true>(act_medium_scatter)) {
                if (dr::any_or<true>(is_spectral))
                    dr::masked(throughput, is_spectral && act_medium_scatter) *=
                        mei.sigma_s * index_spectrum(mei.combined_extinction, channel) / index_spectrum(mei.sigma_t, channel);
                if (dr::any_or<true>(not_spectral))
                    dr::masked(throughput, not_spectral && act_medium_scatter) *= mei.sigma_s / mei.sigma_t;

                PhaseFunctionContext phase_ctx(sampler);
                auto phase = mei.medium->phase_function();

                // --------------------- Emitter sampling ---------------------
                Mask sample_emitters = mei.medium->use_emitter_sampling();
                valid_ray |= act_medium_scatter;
                specular_chain &= !act_medium_scatter;
                specular_chain |= act_medium_scatter && !sample_emitters;

                Mask active_e = act_medium_scatter && sample_emitters;
                if (dr::any_or<true>(active_e)) {
                    auto [emitted, ds] = sample_emitter(mei, scene, sampler, medium, channel, refractive_bsdf, active_e);
                    Vector3f wo        = mei.to_local(ds.d);
                    Spectrum phase_val = phase->eval(phase_ctx, mei, wo, active_e);
                    phase_val = mei.to_world_mueller(phase_val, -wo, mei.wi);
                    Float phase_pdf = phase->pdf(phase_ctx, mei, wo, active_e);
                    dr::masked(result, active_e) += throughput * phase_val * emitted;
                }

                // ------------------ Phase function sampling -----------------
                dr::masked(phase, !act_medium_scatter) = nullptr;
                auto [wo, phase_val] = phase->sample(phase_ctx, mei,
                    sampler->next_1d(act_medium_scatter),
                    sampler->next_2d(act_medium_scatter),
                    act_medium_scatter);
                phase_val = mei.to_world_mueller(phase_val, -wo, mei.wi);
                Float phase_pdf = phase->pdf(phase_ctx, mei, wo, act_medium_scatter);
                dr::masked(throughput, act_medium_scatter) *= phase_val;

                act_medium_scatter &= phase_pdf > 0.f;
                Ray3f new_ray  = mei.spawn_ray(mei.to_world(wo));
                dr::masked(ray, act_medium_scatter) = new_ray;
                needs_intersection |= act_medium_scatter;
            }

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);

            active_surface &= si.is_valid();
            if (dr::any_or<true>(active_surface)) {
                // --------------------- Emitter sampling ---------------------
                BSDFContext ctx;
                BSDFPtr bsdf  = si.bsdf(ray);
                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && (depth + 1 < (uint32_t) m_max_depth);

                if (likely(dr::any_or<true>(active_e))) {
                    // Hacky way to check if this is the refractive BSDF (since we know there is only one)
                    Mask refractive = has_flag(bsdf->flags(), BSDFFlags::DeltaTransmission) || 
                                      has_flag(bsdf->flags(), BSDFFlags::GlossyTransmission);
                    Mask reflect_e = active_e && refractive && si.wi.z() > 0;

                    // Only evaluate if this is a surface reflection
                    if (dr::any_or<true>(reflect_e)) {
                        auto [emitted, ds] = sample_emitter(si, scene, sampler, medium, channel, refractive_bsdf, active_e);
                        
                        // Query the BSDF for that emitter-sampled direction
                        Vector3f wo       = si.to_local(ds.d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                        result[reflect_e] += throughput * bsdf_val * emitted;
                    }
                }

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active_surface),
                                                   sampler->next_2d(active_surface), active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

                dr::masked(throughput, active_surface) *= bsdf_val;
                dr::masked(eta, active_surface) *= bs.eta;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                dr::masked(depth, non_null_bsdf) += 1;

                Ray bsdf_ray                    = si.spawn_ray(si.to_world(bs.wo));

                // Refraction occurred, intentionally or not
                // Assumes surface is aligned with x-y plane in world space
                Float cos_theta_old = Frame3f::cos_theta(ray.d);
                Float cos_theta_new = Frame3f::cos_theta(bsdf_ray.d);

                Mask refracted = non_null_bsdf && (cos_theta_old * cos_theta_new > 0);

                // Mask refracted = has_flag(bs.sampled_type, BSDFFlags::DeltaTransmission) || 
                //                  has_flag(bs.sampled_type, BSDFFlags::GlossyTransmission);

                dr::masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                valid_ray |= non_null_bsdf;
                specular_chain |= non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));
                act_null_scatter |= active_surface && has_flag(bs.sampled_type, BSDFFlags::Null);
                Mask has_medium_trans                = active_surface && si.is_medium_transition();
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);

                Mask set_refractive_bsdf = dr::eq(refractive_bsdf, nullptr) && refracted;
                Mask unset_refractive_bsdf = dr::neq(refractive_bsdf, nullptr) && refracted;
                dr::masked(refractive_bsdf, set_refractive_bsdf) = bsdf;
                dr::masked(refractive_bsdf, unset_refractive_bsdf) = nullptr;
            }
            active &= (active_surface | active_medium);
        }
        return { result, valid_ray };
    }

    /// Samples an emitter in the scene and evaluates its attenuated contribution
    std::tuple<Spectrum, DirectionSample3f>
    sample_emitter(const Interaction3f &ref_interaction, const Scene *scene,
                   Sampler *sampler, MediumPtr medium, UInt32 channel, BSDFPtr refractive_bsdf,
                   Mask active) const {
        using EmitterPtr = dr::replace_scalar_t<Float, const Emitter *>;
        Spectrum transmittance(1.0f);

        auto [ds, emitter_val] = scene->sample_emitter_direction(ref_interaction, sampler->next_2d(active), false, active);
        Vector3f emitter_d = dr::normalize(ds.d);

        dr::masked(emitter_val, dr::eq(ds.pdf, 0.f)) = 0.f;
        active &= dr::neq(ds.pdf, 0.f);

        Mask has_refractive_bsdf = active && dr::neq(refractive_bsdf, nullptr);

        // If there is a refractive BSDF between this interaction and the sensor,
        // pick a direction that will refract to the emitter direction
        if (dr::any_or<true>(has_refractive_bsdf)) {
            BSDFContext ctx;
            ctx.type_mask = BSDFFlags::GlossyTransmission | BSDFFlags::DeltaTransmission;

            SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
            si.wi = emitter_d;

            auto [bs, bsdf_val] = refractive_bsdf->sample(ctx, si, sampler->next_1d(has_refractive_bsdf),
                                                        sampler->next_2d(has_refractive_bsdf), has_refractive_bsdf);
            Mask valid_sample = bs.pdf > 0.f;

            // Assumes surface normal is (0, 0, 1) in world space
            DirectionSample3f ds_tmp(ds);
            ds_tmp.d = -bs.wo;

            dr::masked(emitter_val, has_refractive_bsdf && !valid_sample) = 0.f;
            dr::masked(ds, has_refractive_bsdf && valid_sample) = ds_tmp;
            dr::masked(emitter_val, has_refractive_bsdf && valid_sample) /= bs.pdf;
        }

        if (dr::none_or<false>(active)) {
            return { emitter_val, ds };
        }

        Ray3f ray = ref_interaction.spawn_ray(ds.d);

        Float total_dist = 0.f;
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Mask needs_intersection = true;

        dr::Loop<Mask> loop("Volpath integrator emitter sampling",
                            active, ray, total_dist, needs_intersection, medium, si,
                            transmittance, sampler);
        
        while (loop(active)) {
            Float remaining_dist = ds.dist * (1.f - math::ShadowEpsilon<Float>) - total_dist;
            ray.maxt = remaining_dist;
            active &= remaining_dist > 0.f;
            if (dr::none_or<false>(active))
                break;

            Mask escaped_medium = false;
            Mask active_medium  = active && dr::neq(medium, nullptr);
            Mask active_surface = active && !active_medium;

            if (dr::any_or<true>(active_medium)) {
                auto mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                dr::masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = dr::minimum(mi.t, remaining_dist);
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);

                dr::masked(mi.t, active_medium && (si.t < mi.t)) = dr::Infinity<Float>;
                needs_intersection &= !(active_medium && si.is_valid());

                Mask is_spectral = medium->has_spectral_extinction() && active_medium;
                Mask not_spectral = !is_spectral && active_medium;
                if (dr::any_or<true>(is_spectral)) {
                    Float t      = dr::minimum(remaining_dist, dr::minimum(mi.t, si.t)) - mi.mint;
                    UnpolarizedSpectrum tr  = dr::exp(-t * mi.combined_extinction);
                    UnpolarizedSpectrum free_flight_pdf = dr::select(si.t < mi.t || mi.t > remaining_dist, tr, tr * mi.combined_extinction);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    dr::masked(transmittance, is_spectral) *= dr::select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                // Handle exceeding the maximum distance by medium sampling
                dr::masked(total_dist, active_medium && (mi.t > remaining_dist) && mi.is_valid()) = ds.dist;
                dr::masked(mi.t, active_medium && (mi.t > remaining_dist)) = dr::Infinity<Float>;

                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();
                is_spectral &= active_medium;
                not_spectral &= active_medium;

                dr::masked(total_dist, active_medium) += mi.t;

                if (dr::any_or<true>(active_medium)) {
                    dr::masked(ray.o, active_medium)    = mi.p;
                    dr::masked(si.t, active_medium) = si.t - mi.t;

                    if (dr::any_or<true>(is_spectral))
                        dr::masked(transmittance, is_spectral) *= mi.sigma_n;
                    if (dr::any_or<true>(not_spectral))
                        dr::masked(transmittance, not_spectral) *= mi.sigma_n / mi.combined_extinction;
                }
            }

            // Handle interactions with surfaces
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect)    = scene->ray_intersect(ray, intersect);
            needs_intersection &= !intersect;
            active_surface |= escaped_medium;
            dr::masked(total_dist, active_surface) += si.t;

            active_surface &= si.is_valid() && active && !active_medium;
            if (dr::any_or<true>(active_surface)) {
                BSDFContext ctx;
                auto bsdf = si.bsdf(ray);
                Vector3f wo = si.to_local(emitter_d);
                Mask is_null  = active_surface && has_flag(bsdf->flags(), BSDFFlags::Null);
                Mask not_null = active_surface && !is_null;
     
                if (dr::any_or<true>(is_null)) {
                    Spectrum bsdf_val = bsdf->eval_null_transmission(si, is_null);
                    bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi);
                    dr::masked(transmittance, is_null) *= bsdf_val;
                }
                
                if (dr::any_or<true>(not_null)) {
                    // Reached the refractive BDSF
                    Spectrum bsdf_val = bsdf->eval(ctx, si, wo, not_null);
                    bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
                    dr::masked(transmittance, not_null) *= bsdf_val;

                    // Change ray direction since refraction occurred
                    dr::masked(ray.d, not_null) = emitter_d;
                }                
            }

            // Update the ray with new origin & t parameter
            dr::masked(ray, active_surface) = si.spawn_ray(ray.d);
            ray.maxt = remaining_dist;
            needs_intersection |= active_surface;

            // Continue tracing through scene if non-zero weights exist
            active &= (active_medium || active_surface) && dr::any(dr::neq(unpolarized_spectrum(transmittance), 0.f));

            // If a medium transition is taking place: Update the medium pointer
            Mask has_medium_trans = active_surface && si.is_medium_transition();
            if (dr::any_or<true>(has_medium_trans)) {
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
        }
        
        return { transmittance * emitter_val, ds };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("VolumetricPathAOSIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(VolumetricPathAOSIntegrator, MonteCarloIntegrator);
MI_EXPORT_PLUGIN(VolumetricPathAOSIntegrator, "Volumetric Path Tracer AOS integrator");
NAMESPACE_END(mitsuba)
