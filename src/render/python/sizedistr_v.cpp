#include <mitsuba/render/sizedistr.h>
#include <mitsuba/python/python.h>
#include <mitsuba/core/properties.h>

// To include this file in build, need to add to CMakeLists

MI_PY_EXPORT(SizeDistribution) {
    MI_PY_IMPORT_TYPES()

    // MI_PY_CLASS(SizeDistribution, Object)
    //     .def(py::init<const Properties &>(), "props"_a)
    //     .def("is_monodisperse", &SizeDistribution::is_monodisperse)
    //     .def("min_radius", &SizeDistribution::min_radius)
    //     .def("max_radius", &SizeDistribution::max_radius)
    //     .def("n_gauss", &SizeDistribution::n_gauss)
    //     .def("eval", &SizeDistribution::eval, "r"_a, "normalize"_a = true)
    //     .def("eval_gauss", &SizeDistribution::eval_gauss, "i"_a);
    //     .def_method(SizeDistribution, min_radius)
    //     .def_method(SizeDistribution, max_radius)
    //     .def_method(SizeDistribution, n_gauss)
    //     .def(SizeDistribution, eval, "r"_a, "normalize"_a = true)
    //     .def(SizeDistribution, eval_gauss, "i"_a);
}