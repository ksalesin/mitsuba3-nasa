from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import RBIntegrator

def index_spectrum(spec, idx):
    return spec[0]

class PRBPolarizedIntegrator(RBIntegrator):
    r"""
    .. _integrator-prbvolpath:

    Path Replay Backpropagation Volumetric Integrator (:monosp:`prbvolpath`)
    -------------------------------------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)

     * - hide_emitters
       - |bool|
       - Hide directly visible emitters. (Default: no, i.e. |false|)


    This class implements a volumetric Path Replay Backpropagation (PRB) integrator
    with the following properties:

    - Differentiable delta tracking for free-flight distance sampling

    - Emitter sampling (a.k.a. next event estimation).

    - Russian Roulette stopping criterion.

    - No reparameterization. This means that the integrator cannot be used for
      shape optimization (it will return incorrect/biased gradients for
      geometric parameters like vertex positions.)

    - Detached sampling. This means that the properties of ideal specular
      objects (e.g., the IOR of a glass vase) cannot be optimized.

    See the paper :cite:`Vicini2021` for details on PRB and differentiable delta
    tracking.

    .. tabs::

        .. code-tab:: python

            'type': 'prbvolpath',
            'max_depth': 8
    """
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_nee = False
        self.nee_handle_homogeneous = False
        self.handle_null_scattering = False
        self.is_prepared = False

        if mi.is_rgb:
            raise Exception('PRBPolarizedIntegrator can only be used in'
                            'monochromatic or spectral mode!')

        if not mi.is_polarized:
            raise Exception('PRBPolarizedIntegrator can only be used in'
                            'polarized mode!')

    def prepare_scene(self, scene):
        if self.is_prepared:
            return

        for shape in scene.shapes():
            for medium in [shape.interior_medium(), shape.exterior_medium()]:
                if medium:
                    # Enable NEE if a medium specifically asks for it
                    self.use_nee = self.use_nee or medium.use_emitter_sampling()
                    self.nee_handle_homogeneous = self.nee_handle_homogeneous or medium.is_homogeneous()
                    self.handle_null_scattering = self.handle_null_scattering or (not medium.is_homogeneous())
        self.is_prepared = True
        # By default enable always NEE in case there are surfaces
        self.use_nee = True

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:
        self.prepare_scene(scene)

        if mode == dr.ADMode.Forward:
            raise RuntimeError("PRBPolarizedIntegrator doesn't support "
                               "forward-mode differentiation!")

        is_primal = mode == dr.ADMode.Primal

        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if is_primal else state_in) # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        throughput = mi.Spectrum(1)                   # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        medium = dr.zeros(mi.MediumPtr)

        channel = 0
        valid_ray = mi.Bool(False)
        specular_chain = mi.Bool(True)

        loop = mi.Loop(name=f"Path Replay Backpropagation ({mode.name})",
                       state=lambda: (sampler, active, depth, ray, medium, si,
                                      throughput, L, needs_intersection,
                                      last_scatter_event, specular_chain, η,
                                      last_scatter_direction_pdf, valid_ray))
        while loop(active):
            for i in range(1, 4):
                L[:, i] = 0

            active &= dr.any(dr.neq(mi.unpolarized_spectrum(throughput), 0.0))

            #--------------------- Perform russian roulette --------------------

            q = dr.minimum(dr.max(mi.unpolarized_spectrum(throughput)) * dr.sqr(η), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            throughput[perform_rr] = throughput @ mi.Spectrum(dr.rcp(q))

            active_medium = active & dr.neq(medium, None)
            active_surface = active & ~active_medium

            with dr.resume_grad(when=not is_primal):
                #--------------------- Sample medium interaction -------------------

                # Handle medium sampling and potential medium escape
                u = sampler.next_1d(active_medium)
                mei = medium.sample_interaction(ray, u, channel, active_medium)
                mei.t = dr.detach(mei.t)

                ray.maxt[active_medium & medium.is_homogeneous() & mei.is_valid()] = mei.t
                intersect = needs_intersection & active_medium
                si[intersect] = scene.ray_intersect(ray, intersect)

                needs_intersection &= ~active_medium
                mei.t[active_medium & (si.t < mei.t)] = dr.inf

                # Evaluate ratio of transmittance and free-flight PDF
                tr, free_flight_pdf = medium.transmittance_eval_pdf(mei, si, active_medium)
                tr_pdf = index_spectrum(free_flight_pdf, channel)
                weight = mi.Spectrum(1.0)
                weight[active_medium] = weight @ mi.Spectrum(dr.select(tr_pdf > 0.0, tr / dr.detach(tr_pdf), 0.0))

                escaped_medium = active_medium & ~mei.is_valid()
                active_medium &= mei.is_valid()

                # Handle null and real scatter events
                if self.handle_null_scattering:
                    scatter_prob = index_spectrum(mei.sigma_t, channel) / index_spectrum(mei.combined_extinction, channel)
                    act_null_scatter = (sampler.next_1d(active_medium) >= scatter_prob) & active_medium
                    act_medium_scatter = ~act_null_scatter & active_medium
                    weight[act_null_scatter] *= mei.sigma_n / dr.detach(1 - scatter_prob)
                else:
                    scatter_prob = mi.Float(1.0)
                    act_medium_scatter = active_medium

                depth[act_medium_scatter] += 1
                last_scatter_event[act_medium_scatter] = dr.detach(mei)

                # Don't estimate lighting if we exceeded number of bounces
                active &= depth < self.max_depth
                act_medium_scatter &= active
                if self.handle_null_scattering:
                    ray.o[act_null_scatter] = dr.detach(mei.p)
                    si.t[act_null_scatter] = si.t - dr.detach(mei.t)

                weight[act_medium_scatter] = weight @ mi.Spectrum(mei.sigma_s / dr.detach(scatter_prob))
                throughput[active_medium] = throughput @ dr.detach(weight)

                mei = dr.detach(mei)
                if not is_primal and dr.grad_enabled(weight):
                    I = dr.replace_grad(mi.Spectrum(1.0), weight)
                    Lo = I @ mi.Spectrum(dr.detach(dr.select(active_medium | escaped_medium, L, 0.0)))
                    dr.backward(δL @ Lo)

                phase_ctx = mi.PhaseFunctionContext(sampler)
                phase = mei.medium.phase_function()
                phase[~act_medium_scatter] = dr.zeros(mi.PhaseFunctionPtr)

                #--------------------- Surface Interactions ---------------------
                
                active_surface |= escaped_medium
                intersect = active_surface & needs_intersection
                si[intersect] = scene.ray_intersect(ray, intersect)

                active_surface &= si.is_valid()
                ctx = mi.BSDFContext()
                bsdf = si.bsdf(ray)

                # --------------------- Emitter sampling ---------------------
                
                if self.use_nee:
                    active_e_surface = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < self.max_depth)
                    sample_emitters = mei.medium.use_emitter_sampling()
                    specular_chain &= ~act_medium_scatter
                    specular_chain |= act_medium_scatter & ~sample_emitters

                    active_e_medium = act_medium_scatter & sample_emitters
                    active_e = active_e_surface | active_e_medium

                    nee_sampler = sampler if is_primal else sampler.clone()
                    emitted, ds = self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                        scene, sampler, medium, channel, active_e, mode=dr.ADMode.Primal)
                    
                    # Query the BSDF for that emitter-sampled direction
                    bsdf_wo = si.to_local(ds.d)
                    bsdf_val, _ = bsdf.eval_pdf(ctx, si, bsdf_wo, active_e_surface)
                    bsdf_val = si.to_world_mueller(bsdf_val, -bsdf_wo, si.wi)

                    phase_wo = mei.to_local(ds.d)
                    phase_val, _ = phase.eval_pdf(phase_ctx, mei, phase_wo, active_e_medium)
                    phase_val = mei.to_world_mueller(phase_val, -phase_wo, mei.wi)
                    
                    nee_weight = mi.Spectrum(dr.select(active_e_surface, bsdf_val, phase_val))

                    # Calculate NEE contribution to final radiance value
                    contrib = throughput @ nee_weight @ emitted
                    L[active_e] += dr.detach(contrib if is_primal else -contrib)

                    if not is_primal:
                        self.sample_emitter(mei, si, active_e_medium, active_e_surface,
                            scene, nee_sampler, medium, channel, active_e, adj_emitted=contrib,
                            δL=δL, mode=mode)
                        
                        if dr.grad_enabled(nee_weight) or dr.grad_enabled(emitted):
                            dr.backward(δL @ contrib)

                #-------------------- Phase function sampling ------------------

                valid_ray |= act_medium_scatter
                with dr.suspend_grad():
                    wo, phase_weight, phase_pdf = phase.sample(phase_ctx, mei,
                                                               sampler.next_1d(act_medium_scatter),
                                                               sampler.next_2d(act_medium_scatter),
                                                               act_medium_scatter)
                    phase_weight = mei.to_world_mueller(phase_weight, -wo, mei.wi)
                act_medium_scatter &= phase_pdf > 0.0

                # Re evaluate the phase function value in an attached manner
                phase_eval, _ = phase.eval_pdf(phase_ctx, mei, wo, act_medium_scatter)
                phase_eval = mei.to_world_mueller(phase_eval, -wo, mei.wi)
                if not is_primal and dr.grad_enabled(phase_eval):
                    I = dr.replace_grad(mi.Spectrum(1.0), phase_eval)
                    Lo = I @ mi.Spectrum(dr.detach(dr.select(act_medium_scatter, L, 0.0)))
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL @ Lo)

                throughput[act_medium_scatter] = throughput @ phase_weight
                ray[act_medium_scatter] = mei.spawn_ray(mei.to_world(wo))
                needs_intersection |= act_medium_scatter
                last_scatter_direction_pdf[act_medium_scatter] = phase_pdf

                # ----------------------- BSDF sampling ----------------------
                with dr.suspend_grad():
                    bs, bsdf_weight = bsdf.sample(ctx, si,
                                                  sampler.next_1d(active_surface),
                                                  sampler.next_2d(active_surface),
                                                  active_surface)
                    bsdf_weight = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi)
                    active_surface &= bs.pdf > 0

                bsdf_eval = bsdf.eval(ctx, si, bs.wo, active_surface)
                bsdf_eval = si.to_world_mueller(bsdf_eval, -bs.wo, si.wi)

                if not is_primal and dr.grad_enabled(bsdf_eval):
                    I = dr.replace_grad(mi.Spectrum(1.0), bsdf_eval)
                    Lo = I @ mi.Spectrum(dr.detach(dr.select(active, L, 0.0)))
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL @ Lo)

                throughput[active_surface] = throughput @ bsdf_weight
                η[active_surface] *= bs.eta
                bsdf_ray = si.spawn_ray(si.to_world(bs.wo))
                ray[active_surface] = bsdf_ray

                needs_intersection |= active_surface
                non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
                depth[non_null_bsdf] += 1

                # update the last scatter PDF event if we encountered a non-null scatter event
                last_scatter_event[non_null_bsdf] = si
                last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

                valid_ray |= non_null_bsdf
                specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
                has_medium_trans = active_surface & si.is_medium_transition()
                medium[has_medium_trans] = si.target_medium(ray.d)
                active &= (active_surface | active_medium)

        return L if is_primal else δL, valid_ray, L

    def sample_emitter(self, mei, si, active_medium, active_surface, scene, sampler, medium, channel,
                       active, adj_emitted=None, δL=None, mode=None):
        is_primal = mode == dr.ADMode.Primal

        active = mi.Bool(active)

        ref_interaction = dr.zeros(mi.Interaction3f)
        ref_interaction[active_medium] = mei
        ref_interaction[active_surface] = si

        ds, emitter_val = scene.sample_emitter_direction(ref_interaction, 
                                                         sampler.next_2d(active), 
                                                         False, active)
        ds = dr.detach(ds)
        invalid = dr.eq(ds.pdf, 0.0)
        emitter_val[invalid] = 0.0
        active &= ~invalid

        medium = dr.select(active, medium, dr.zeros(mi.MediumPtr))
        medium[(active_surface & si.is_medium_transition())] = si.target_medium(ds.d)

        ray = ref_interaction.spawn_ray(ds.d)
        total_dist = mi.Float(0.0)
        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        transmittance = mi.Spectrum(1.0)
        loop = mi.Loop(name=f"PRB Next Event Estimation ({mode.name})",
                       state=lambda: (sampler, active, medium, ray, total_dist,
                                      needs_intersection, si, transmittance))
        while loop(active):
            remaining_dist = ds.dist * (1.0 - mi.math.ShadowEpsilon) - total_dist
            ray.maxt = dr.detach(remaining_dist)
            active &= remaining_dist > 0.0

            # This ray will not intersect if it reached the end of the segment
            needs_intersection &= active
            si[needs_intersection] = scene.ray_intersect(ray, needs_intersection)
            needs_intersection &= False

            active_medium = active & dr.neq(medium, None)
            active_surface = active & ~active_medium

            # Handle medium interactions / transmittance
            mei = medium.sample_interaction(ray, sampler.next_1d(active_medium), channel, active_medium)
            mei.t[active_medium & (si.t < mei.t)] = dr.inf
            mei.t = dr.detach(mei.t)

            tr_multiplier = mi.Spectrum(1.0)

            # Special case for homogeneous media: directly advance to the next surface / end of the segment
            if self.nee_handle_homogeneous:
                active_homogeneous = active_medium & medium.is_homogeneous()
                mei.t[active_homogeneous] = dr.minimum(remaining_dist, si.t)
                tr_multiplier[active_homogeneous] = medium.transmittance_eval_pdf(mei, si, active_homogeneous)[0]
                mei.t[active_homogeneous] = dr.inf

            escaped_medium = active_medium & ~mei.is_valid()

            # Ratio tracking transmittance computation
            active_medium &= mei.is_valid()
            ray.o[active_medium] = dr.detach(mei.p)
            si.t[active_medium] = dr.detach(si.t - mei.t)
            tr_multiplier[active_medium] = tr_multiplier @ mi.Spectrum(mei.sigma_n / mei.combined_extinction)

            # Handle interactions with surfaces
            active_surface |= escaped_medium
            active_surface &= si.is_valid() & ~active_medium
            bsdf = si.bsdf(ray)
            bsdf_val = bsdf.eval_null_transmission(si, active_surface)
            bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi)
            tr_multiplier[active_surface] = tr_multiplier @ bsdf_val

            if not is_primal and dr.grad_enabled(tr_multiplier):
                active_adj = (active_surface | active_medium) & (mi.unpolarized_spectrum(tr_multiplier) > 0.0)
                I = dr.replace_grad(mi.Spectrum(1.0), tr_multiplier)
                dr.backward(I @ mi.Spectrum(dr.detach(dr.select(active_adj, δL @ adj_emitted, 0.0))))

            transmittance = transmittance @ dr.detach(tr_multiplier)

            # Update the ray with new origin & t parameter
            ray[active_surface] = dr.detach(si.spawn_ray(mi.Vector3f(ray.d)))
            ray.maxt = dr.detach(remaining_dist)
            needs_intersection |= active_surface

            # Continue tracing through scene if non-zero weights exist
            active &= (active_medium | active_surface) & dr.any(dr.neq(mi.unpolarized_spectrum(transmittance), 0.0))
            total_dist[active] += dr.select(active_medium, mei.t, si.t)

            # If a medium transition is taking place: Update the medium pointer
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

        return transmittance @ emitter_val, ds

    def to_string(self):
        return f'PRBPolarizedIntegrator[max_depth = {self.max_depth}]'


mi.register_integrator("prb_polarized", lambda props: PRBPolarizedIntegrator(props))
