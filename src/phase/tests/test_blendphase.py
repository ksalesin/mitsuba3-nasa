import drjit as dr
import mitsuba as mi


def test01_create(variant_scalar_rgb):
    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "isotropic"},
            "weight_0": 0.8,
        }
    )
    assert phase is not None
    assert phase.flags() == int(mi.PhaseFunctionFlags.Isotropic)

    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "hg"},
            "weight_0": 0.8,
        }
    )
    assert phase is not None
    assert phase.component_count() == 2
    assert phase.flags(0) == int(mi.PhaseFunctionFlags.Isotropic)
    assert phase.flags(1) == int(mi.PhaseFunctionFlags.Anisotropic)
    assert (
        phase.flags() == mi.PhaseFunctionFlags.Isotropic | mi.PhaseFunctionFlags.Anisotropic
    )


def test02_eval_2(variant_scalar_rgb):
    weight_0 = 0.8
    g = 0.2

    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "hg", "g": g},
            "weight_0": weight_0,
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    wo = [0, 0, 1]
    ctx = mi.PhaseFunctionContext(None)

    # Evaluate the blend of both components
    expected = weight_0 * dr.inv_four_pi + (1 - weight_0) * dr.inv_four_pi * \
        (1.0 - g) / (1.0 + g) ** 2
    value = phase.eval_pdf(ctx, mei, wo)[0]
    assert dr.allclose(value, expected)


def test03_sample_2(variants_all_rgb):
    weight_0 = 0.8
    g = 0.2

    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "hg", "g": g},
            "weight_0": weight_0,
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    ctx = mi.PhaseFunctionContext(None)

    # Sample using two different values of 'sample1' and make sure correct
    # components are chosen.

    # -- Sample below weight: first component (isotropic) is selected
    expected_a = dr.inv_four_pi
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.3, [0.5, 0.5])
    assert dr.allclose(pdf_a, expected_a)

    # -- Sample above weight: second component (HG) is selected
    expected_b = dr.inv_four_pi * (1 - g) / (1 + g) ** 2
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.9, [0, 0])
    assert dr.allclose(pdf_b, expected_b)


def test04_eval_3(variant_scalar_rgb):
    weight_0 = 0.3
    weight_1 = 0.5
    weight_2 = 1 - (weight_0 + weight_1)
    g_1 = 0.2
    g_2 = 0.9

    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "hg", "g": g_1},
            "phase_2": {"type": "hg", "g": g_2},
            "weight_0": weight_0,
            "weight_1": weight_1
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    wo = [0, 0, 1]
    ctx = mi.PhaseFunctionContext(None)

    # Evaluate the blend of all components
    expected = weight_0 * dr.inv_four_pi + \
               weight_1 * dr.inv_four_pi * (1.0 - g_1) / (1.0 + g_1) ** 2 + \
               weight_2 * dr.inv_four_pi * (1.0 - g_2) / (1.0 + g_2) ** 2
    value = phase.eval_pdf(ctx, mei, wo)[0]
    assert dr.allclose(value, expected)


def test05_sample_3(variants_all_rgb):
    weight_0 = 0.3
    weight_1 = 0.5
    g_1 = 0.2
    g_2 = 0.9

    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "hg", "g": g_1},
            "phase_2": {"type": "hg", "g": g_2},
            "weight_0": weight_0,
            "weight_1": weight_1
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    ctx = mi.PhaseFunctionContext(None)

    # Sample using different values of 'sample1' and make sure correct
    # components are chosen.

    # -- Sample below weight_0: first component (isotropic) is selected
    expected_a = dr.inv_four_pi
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.1, [0.5, 0.5])
    assert dr.allclose(pdf_a, expected_a)

    # -- Sample below (weight_0 + weight_1): second component (HG_1) is selected
    expected_b = dr.inv_four_pi * (1 - g_1) / (1 + g_1) ** 2
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.5, [0, 0])
    assert dr.allclose(pdf_b, expected_b)

    # -- Sample above (weight_0 + weight_1): third component (HG_2) is selected
    expected_c = dr.inv_four_pi * (1 - g_2) / (1 + g_2) ** 2
    wo_c, w_c, pdf_c = phase.sample(ctx, mei, 0.9, [0, 0])
    assert dr.allclose(pdf_c, expected_c)


def test06_eval_components(variant_scalar_rgb):
    weight_0 = 0.3
    weight_1 = 0.5
    weight_2 = 1 - (weight_0 + weight_1)
    g_1 = 0.2
    g_2 = 0.9

    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "hg", "g": g_1},
            "phase_2": {"type": "hg", "g": g_2},
            "weight_0": weight_0,
            "weight_1": weight_1
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    wo = [0, 0, 1]
    ctx = mi.PhaseFunctionContext(None)

    # Evaluate the components separately

    ctx.component = 0
    value0, pdf0 = phase.eval_pdf(ctx, mei, wo)
    expected0 = weight_0 * dr.inv_four_pi
    assert dr.allclose(value0, expected0)
    assert dr.allclose(value0, pdf0)

    ctx.component = 1
    value1, pdf1 = phase.eval_pdf(ctx, mei, wo)
    expected1 = weight_1 * dr.inv_four_pi * (1.0 - g_1) / (1.0 + g_1) ** 2
    assert dr.allclose(value1, expected1)
    assert dr.allclose(value1, pdf1)

    ctx.component = 2
    value2, pdf2 = phase.eval_pdf(ctx, mei, wo)
    expected2 = weight_2 * dr.inv_four_pi * (1.0 - g_2) / (1.0 + g_2) ** 2
    assert dr.allclose(value2, expected2)
    assert dr.allclose(value2, pdf2)


def test07_sample_components(variant_scalar_rgb):
    weight_0 = 0.3
    weight_1 = 0.5
    weight_2 = 1 - (weight_0 + weight_1)
    g_1 = 0.2
    g_2 = 0.9

    phase = mi.load_dict(
        {
            "type": "blendphase",
            "phase_0": {"type": "isotropic"},
            "phase_1": {"type": "hg", "g": g_1},
            "phase_2": {"type": "hg", "g": g_2},
            "weight_0": weight_0,
            "weight_1": weight_1
        }
    )

    mei = mi.MediumInteraction3f()
    mei.t = 0.1
    mei.p = [0, 0, 0]
    mei.sh_frame = mi.Frame3f([0, 0, 1])
    mei.wi = [0, 0, 1]

    ctx = mi.PhaseFunctionContext(None)

    # Sample using different values of 'sample1' and make sure correct
    # components are chosen.

    # -- Select component 0: first component is always sampled
    ctx.component = 0

    expected_a = weight_0 *  dr.inv_four_pi
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.3, [0.5, 0.5])
    assert dr.allclose(pdf_a, expected_a)

    expected_b = weight_0 *  dr.inv_four_pi
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.9, [0.5, 0.5])
    assert dr.allclose(pdf_b, expected_b)

    # -- Select component 1: second component is always sampled
    ctx.component = 1

    expected_a = weight_1 *  dr.inv_four_pi * (1 - g_1) / (1 + g_1) ** 2
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.3, [0.0, 0.0])
    assert dr.allclose(pdf_a, expected_a)

    expected_b = weight_1 * dr.inv_four_pi * (1 - g_1) / (1 + g_1) ** 2
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.9, [0.0, 0.0])
    assert dr.allclose(pdf_b, expected_b)

    # -- Select component 2: third component is always sampled
    ctx.component = 2

    expected_a = weight_2 *  dr.inv_four_pi * (1 - g_2) / (1 + g_2) ** 2
    wo_a, w_a, pdf_a = phase.sample(ctx, mei, 0.3, [0.0, 0.0])
    assert dr.allclose(pdf_a, expected_a)

    expected_b = weight_2 * dr.inv_four_pi * (1 - g_2) / (1 + g_2) ** 2
    wo_b, w_b, pdf_b = phase.sample(ctx, mei, 0.9, [0.0, 0.0])
    assert dr.allclose(pdf_b, expected_b)