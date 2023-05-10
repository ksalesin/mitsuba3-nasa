"""
Overview
--------

This file defines a set of unit tests to assess the correctness of the
different adjoint integrators such as `rb`, `prb`, etc... All integrators will
be tested for their implementation of primal rendering, adjoint forward
rendering and adjoint backward rendering.

- For primal rendering, the output image will be compared to a ground truth
  image precomputed in the `resources/data/tests/integrators` directory.
- Adjoint forward rendering will be compared against finite differences.
- Adjoint backward rendering will be compared against finite differences.

Those tests will be run on a set of configurations (scene description + metadata)
also provided in this file. More tests can easily be added by creating a new
configuration type and add it to the *_CONFIGS_LIST below.

By executing this script with python directly it is possible to regenerate the
reference data (e.g. for a new configurations). Please see the following command:

``python3 test_ad_integrators.py --help``

"""

import drjit as dr
import mitsuba as mi
import importlib

import pytest, os, argparse
from os.path import join, exists
import numpy as np

from mitsuba.scalar_rgb.test.util import fresolver_append_path
# from mitsuba.scalar_rgb.test.util import find_resource

# output_dir = find_resource('resources/data/tests/integrators')

# -------------------------------------------------------------------
#                          Test configs
# -------------------------------------------------------------------

from mitsuba.scalar_rgb import ScalarTransform4f as T
from mitsuba.scalar_rgb import Vector3f

class ConfigBase:
    """
    Base class to configure test scene and define the parameter to update
    """
    require_reparameterization = False

    def __init__(self) -> None:
        self.spp = 8
        self.res = 32
        self.error_mean_threshold = 0.05
        self.error_max_threshold = 0.5
        self.error_max_threshold_bwd = 0.05
        self.ref_fd_epsilon = 1e-3

        self.integrator_dict = {
            'max_depth': 3,
        }

        self.sensor_dict = {
            'type': 'radiancemeter',
            'to_world': T.look_at(origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
            'wavelength': 450.0,
            'film': {
                'type': 'hdrfilm',
                'rfilter': { 'type': 'box' },
                'width': self.res,
                'height': self.res,
                'sample_border': False,
                'pixel_format': 'luminance',
                'component_format': 'float32',
            }
        }

        # Set the config name based on the type
        import re
        self.name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__[:-6]).lower()

    def initialize(self):
        """
        Initialize the configuration, loading the Mitsuba scene and storing a
        copy of the scene parameters to compute gradients for.
        """

        self.sensor_dict['film']['width'] = self.res
        self.sensor_dict['film']['height'] = self.res
        self.scene_dict['sensor'] = self.sensor_dict

        @fresolver_append_path
        def create_scene():
            return mi.load_dict(self.scene_dict)
        self.scene = create_scene()
        self.params = mi.traverse(self.scene)

        if hasattr(self, 'key'):
            self.params.keep([self.key])
            self.initial_state = type(self.params[self.key])(self.params[self.key])

    def update(self, theta):
        """
        This method update the scene parameter associated to this config
        """
        self.params[self.key] = self.initial_state + theta
        dr.set_label(self.params, 'params')
        self.params.update()
        dr.eval()

    def __repr__(self) -> str:
        return f'{self.name}[\n' \
               f'  integrator = { self.integrator_dict },\n' \
               f'  spp = {self.spp},\n' \
               f'  key = {self.key if hasattr(self, "key") else "None"}\n' \
               f']'


# BSDF albedo of a directly visible gray plane illuminated by a constant emitter
class DiffuseAlbedoConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'plane.bsdf.reflectance.value'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': { 'type': 'diffuse' }
            },
            'light': { 'type': 'constant' }
        }

# BSDF albedo of a off camera plane blending onto a directly visible gray plane
class DiffuseAlbedoGIConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'green.bsdf.reflectance.value'
        self.scene_dict = {
            'type': 'scene',
            'plane': { 'type': 'rectangle' },
            'green': {
                'type': 'rectangle',
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': [0.1, 1.0, 0.1]
                    }
                },
                'to_world': T.translate([1.25, 0.0, 1.0]) @ T.rotate([0, 1, 0], -90),
            },
            'light': { 'type': 'constant', 'radiance': 3.0 }
        }
        self.integrator_dict = {
            'max_depth': 3,
        }

# Off camera area light illuminating a gray plane
class AreaLightRadianceConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'light.emitter.radiance.value'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
                }
            },
            'light': {
                'type': 'rectangle',
                'to_world': T.translate([1.25, 0.0, 1.0]) @ T.rotate([0, 1, 0], -90),
                'emitter': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': [3.0, 3.0, 3.0]}
                }
            }
        }

# Directly visible area light illuminating a gray plane
class DirectlyVisibleAreaLightRadianceConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'light.emitter.radiance.value'
        self.scene_dict = {
            'type': 'scene',
            'light': {
                'type': 'rectangle',
                'emitter': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
                }
            }
        }

# Off camera point light illuminating a gray plane
class PointLightIntensityConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'light.intensity.value'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
                }
            },
            'light': {
                'type': 'point',
                'position': [1.25, 0.0, 1.0],
                'intensity': {'type': 'uniform', 'value': 5.0}
            },
        }

# Instensity of a constant emitter illuminating a gray rectangle
class ConstantEmitterRadianceConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'light.radiance.value'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': { 'type': 'diffuse' }
            },
            'light': { 'type': 'constant' }
        }

# Relative index of refraction of a directly visible rough dielectric surface illuminated by a constant emitter
class RoughDielectricEtaConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'plane.bsdf.eta'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'roughdielectric',
                    'int_ior': 1.34,
                    'ext_ior': 1.0,
                    'distribution': 'beckmann',
                    'alpha': 0.1,
                    'sample_visible': False
                }
            },
            'light': {
                'type': 'rectangle',
                'to_world': T.translate([1.25, 0.0, 1.0]) @ T.rotate([0, 1, 0], -90),
                'emitter': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': [3.0, 3.0, 3.0]}
                }
            }
        }

# Roughness of a directly visible rough dielectric surface illuminated by a constant emitter
class RoughDielectricRoughnessConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'plane.bsdf.alpha.value'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'roughdielectric',
                    'int_ior': 1.34,
                    'ext_ior': 1.0,
                    'distribution': 'beckmann',
                    'alpha': 0.1,
                    'sample_visible': False
                }
            },
            'light': {
                'type': 'rectangle',
                'to_world': T.translate([1.25, 0.0, 1.0]) @ T.rotate([0, 1, 0], -90),
                'emitter': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': [3.0, 3.0, 3.0]}
                }
            }
        }

# Relative index of refraction of a directly visible rough dielectric surface illuminated by a directional emitter
class RoughDielectricEtaDirectionalConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'plane.bsdf.eta'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'roughdielectric',
                    'int_ior': 1.34,
                    'ext_ior': 1.0,
                    'distribution': 'beckmann',
                    'alpha': 0.1,
                    'sample_visible': False
                }
            },
            'emitter': {
                'type': 'directional',
                'direction': [-0.5, 0, -0.866],
                'irradiance': 1.0
            }
        }

# Roughness of a directly visible rough dielectric surface illuminated by a directional emitter
class RoughDielectricRoughnessDirectionalConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'plane.bsdf.alpha.value'
        self.scene_dict = {
            'type': 'scene',
            'plane': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'roughdielectric',
                    'int_ior': 1.34,
                    'ext_ior': 1.0,
                    'distribution': 'beckmann',
                    'alpha': 0.1,
                    'sample_visible': False
                }
            },
            'emitter': {
                'type': 'directional',
                'direction': [-0.5, 0, -0.866],
                'irradiance': 1.0
            }
        }

class MediumAlbedoConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.error_max_threshold_bwd = 0.1
        self.key = 'a_medium.albedo.value.value'
        self.scene_dict = {
            'type': 'scene',
            'a_medium': {
                'type': 'homogeneous',
                'phase': {
                    'type': 'hg',
                    'g': 0.5
                },
                'albedo': 0.5,
                'sigma_t': 1.0,
                'has_spectral_extinction': False
            },
            'top': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'null',
                },
                'to_world':  T.translate([0.0, 0.0, 1.0]) @ T.scale([1000000, 1000000, 1]),
                'interior' : {
                    'type' : 'ref',
                    'id' : 'a_medium'
                }
            },
            'bottom': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'diffuse',
                    'reflectance': 0.0
                },
                'to_world': T.scale([1000000, 1000000, 1]),
                'exterior' : {
                    'type' : 'ref',
                    'id' : 'a_medium'
                }
            },
            'light': {
                'type': 'rectangle',
                'to_world': T.translate([0.0, 0.0, 5.0]) @ T.rotate([0, 1, 0], -180),
                'emitter': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': [3.0, 3.0, 3.0]}
                }
            }
        }

class MediumPhaseConfig(ConfigBase):
    def __init__(self) -> None:
        super().__init__()
        self.key = 'a_medium.phase_function.g'
        self.scene_dict = {
            'type': 'scene',
            'a_medium': {
                'type': 'homogeneous',
                'phase': {
                    'type': 'hg',
                    'g': 0.5
                },
                'albedo': 0.5,
                'sigma_t': 1.0,
                'has_spectral_extinction': False
            },
            'top': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'null',
                },
                'to_world': T.translate([0.0, 0.0, 1.0]) @ T.scale([1000000, 1000000, 1]),
                'interior' : {
                    'type' : 'ref',
                    'id' : 'a_medium'
                }
            },
            'bottom': {
                'type': 'rectangle',
                'bsdf': { 
                    'type': 'diffuse',
                    'reflectance': 0.0
                },
                'to_world': T.scale([1000000, 1000000, 1]),
                'exterior' : {
                    'type' : 'ref',
                    'id' : 'a_medium'
                }
            },
            'light': {
                'type': 'rectangle',
                'to_world': T.translate([0.0, 0.0, 5.0]) @ T.rotate([0, 1, 0], -180),
                'emitter': {
                    'type': 'area',
                    'radiance': {'type': 'rgb', 'value': [3.0, 3.0, 3.0]}
                }
            }
        }

# -------------------------------------------------------------------
#                           List configs
# -------------------------------------------------------------------

BASIC_CONFIGS_LIST = [
    DiffuseAlbedoConfig,
    DiffuseAlbedoGIConfig,
    AreaLightRadianceConfig,
    DirectlyVisibleAreaLightRadianceConfig,
    PointLightIntensityConfig,
    ConstantEmitterRadianceConfig,
]

ADVANCED_CONFIGS_LIST = [
    # RoughDielectricEtaConfig,
    # RoughDielectricRoughnessConfig,
    # RoughDielectricEtaDirectionalConfig,
    RoughDielectricRoughnessDirectionalConfig,
    # MediumAlbedoConfig,
    # MediumPhaseConfig
]

# List of integrators to test (also indicates whether it handles discontinuities)
INTEGRATORS = [
    ('prb_volpathaos', False)
]

CONFIGS = []
for integrator_name, reparam in INTEGRATORS:
    todos = BASIC_CONFIGS_LIST
    for config in todos:
        CONFIGS.append((integrator_name, config))

# -------------------------------------------------------------------
#                      Generate reference results
# -------------------------------------------------------------------

ref_spp = 10000

mi.set_variant('llvm_ad_mono_polarized')

# Generate reference primal/forward results for all configs.
for config_base in BASIC_CONFIGS_LIST:
    config = config_base()
    print(f"name: {config.name}")

    config.initialize()

    integrator_ref = mi.load_dict({
        'type': 'volpathaos',
        'max_depth': config.integrator_dict['max_depth']
    })

    # Primal render
    ref_primal = integrator_ref.render_1(config.scene, 
                                         0,        # sensor index
                                         0,        # RNG seed
                                         ref_spp,  # spp override
                                         False,    # develop film
                                         True,     # evaluate
                                         0)        # thread count (0 for auto-detect)
    dr.eval(ref_primal)
    
    config_base.ref_primal = ref_primal

    # Finite difference
    theta = mi.Float(-0.5 * config.ref_fd_epsilon)
    config.update(theta)
    value_1   = integrator_ref.render_1(config.scene, 
                                        0,        # sensor index
                                        0,        # RNG seed
                                        ref_spp,  # spp override
                                        False,    # develop film
                                        True,     # evaluate
                                        0)        # thread count (0 for auto-detect)
    dr.eval(value_1)

    theta = mi.Float(0.5 * config.ref_fd_epsilon)
    config.update(theta)
    value_2   = integrator_ref.render_1(config.scene, 
                                        0,        # sensor index
                                        0,        # RNG seed
                                        ref_spp,  # spp override
                                        False,    # develop film
                                        True,     # evaluate
                                        0)        # thread count (0 for auto-detect)
    dr.eval(value_2)

    ref_fd = (value_2 - value_1) / config.ref_fd_epsilon

    config_base.ref_fd = ref_fd

# -------------------------------------------------------------------
#                           Unit tests
# -------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('integrator_name, config_base', CONFIGS)
def test01_rendering_primal(variants_llvm_ad_mono_polarized, integrator_name, config_base):
    config = config_base()
    config.initialize()

    # dr.set_flag(dr.JitFlag.VCallRecord, False)
    # dr.set_flag(dr.JitFlag.LoopRecord, False)

    import mitsuba
    importlib.reload(mitsuba.ad.integrators)
    config.integrator_dict['type'] = integrator_name
    integrator = mi.load_dict(config.integrator_dict)

    ref_primal = config_base.ref_primal

    primal = integrator.render_1(config.scene, 
                                0,          # sensor index
                                0,          # RNG seed
                                config.spp, # spp override
                                False,      # develop film
                                True,       # evaluate
                                0)          # thread count (0 for auto-detect)

    error = dr.abs(dr.ravel(primal) - dr.ravel(ref_primal)) / dr.maximum(dr.abs(dr.ravel(ref_primal)), 2e-2)
    # error_mean = dr.mean(error)[0]
    error_max = dr.max(error)[0]

    if error_max > config.error_max_threshold:
        print(f"Failure in config: {config.name}, {integrator_name}")
        # print(f"-> error mean: {error_mean} (threshold={config.error_mean_threshold})")
        print(f"-> error max: {error_max} (threshold={config.error_max_threshold})")
        print(f'-> reference value: {ref_primal}')
        print(f'-> output value: {primal}')
        assert False


@pytest.mark.slow
@pytest.mark.skipif(os.name == 'nt', reason='Skip those memory heavy tests on Windows')
@pytest.mark.parametrize('integrator_name, config_base', CONFIGS)
def test02_rendering_backward(variants_llvm_ad_mono_polarized, integrator_name, config_base):
    # dr.set_flag(dr.JitFlag.LoopRecord, False)
    # dr.set_flag(dr.JitFlag.VCallRecord, False)

    config = config_base()
    config.initialize()

    import mitsuba
    importlib.reload(mitsuba.ad.integrators)
    config.integrator_dict['type'] = integrator_name
    integrator = mi.load_dict(config.integrator_dict)

    ref_fd = config_base.ref_fd

    # image_adj = mi.TensorXf(1.0, ref_fd.shape)
    image_adj = dr.full(mi.Spectrum, 1.0)

    theta = mi.Float(0.0)
    dr.enable_grad(theta)
    config.update(theta)

    # dr.set_flag(dr.JitFlag.KernelHistory, True)
    # dr.set_log_level(3)

    integrator.render_1_backward(
        config.scene, grad_in=image_adj, seed=0, spp=config.spp, params=theta)

    grad = dr.grad(theta)[0]

    # FD ref is really a Jacobian, perform one last multiplication
    # dx = dy^T @ J to match the output of dr.grad()
    grad_ref = dr.dot(dr.ravel(ref_fd), dr.ravel(image_adj))[0]
    
    print(f"grad:     {grad}")
    print(f"grad_ref: {grad_ref}")

    error = dr.abs(grad - grad_ref) / dr.maximum(dr.abs(grad_ref), 1e-3)
    if error > config.error_max_threshold_bwd:
        print(f"Failure in config: {config.name}, {integrator_name}")
        print(f"-> grad:     {grad}")
        print(f"-> grad_ref: {grad_ref}")
        print(f"-> error: {error} (threshold={config.error_max_threshold_bwd})")
        print(f"-> ratio: {grad / grad_ref}")
        assert False
