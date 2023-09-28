#pragma once

#include <mitsuba/render/fwd.h>
#include <drjit/vcall.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Abstract size distribution base-class.
 *
 * This class provides an abstract interface to all size distribution plugins. 
 * It exposes functions for evaluating the model.
 */

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB SizeDistribution : public Object {
public:
    MI_IMPORT_TYPES()

    /**
     * \brief Evaluate the size distribution function (sdf)
     *
     * \param r
     *     The radius (size)
     */
    virtual Float eval(Float r, bool normalize = true) const = 0;

    /**
     * \brief Calculate the Gaussian quadrature division points
     *        and weights.
     */
    void calculate_gauss();

    /**
     * \brief Calculate normalization constant.
     */
    void calculate_constant();
    
    /**
     * \brief Evaluate radius, Gaussian quadrature weight,
     *        and value of sdf for i-th division point
     *
     * \return  A tuple (radius, weight, sdf) consisting of
     *
     *     radius      Radius of i-th division point
     *
     *     weight      Weight of i-th division point
     * 
     *     sdf         Value of sdf(radius)
     */
    std::tuple<Float, Float, Float> eval_gauss(ScalarUInt32 i) const {
        if (i > m_g)
            Throw("Index %d out of range", i);

        Float radius = Float(m_gauss_points[i]);
        Float weight = Float(m_gauss_weights[i]);
        Float sdf    = eval(radius);

        return { radius, weight, sdf };
    }
    
    /// Returns whether this medium is monodisperse
    MI_INLINE bool is_monodisperse() const { return m_is_monodisperse; }

    /// Return the minimum radius
    Float min_radius() const { return m_min_radius; }

    /// Return the maximum radius
    Float max_radius() const { return m_max_radius; }

    /// Return the number of Gaussian quadrature division points
    ScalarInt32 n_gauss() const { return m_g; }

    /// Return a string identifier
    std::string id() const override { return m_id; }

    /// Set a string identifier
    void set_id(const std::string& id) override { m_id = id; };

    /// Return a human-readable representation of the phase function
    std::string to_string() const override = 0;

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        calculate_gauss();
        calculate_constant();
    }

    //! @}
    // -----------------------------------------------------------------------

    MI_DECLARE_CLASS()
protected:
    SizeDistribution(const Properties &props);
    virtual ~SizeDistribution();

protected:
    bool m_is_monodisperse = false;

    /// Normalization constant
    double m_constant;

    /// Minimum and maximum radius
    Float m_min_radius;
    Float m_max_radius;

    /// Effective radius and variance
    Float m_reff;
    Float m_veff;

    /// Number of Gaussian quadrature points
    uint32_t m_g;

    /// Gaussian quadrature division weights and points
    std::vector<double> m_gauss_weights;
    std::vector<double> m_gauss_points;

    /// Identifier (if available)
    std::string m_id;
};

MI_EXTERN_CLASS(SizeDistribution)
NAMESPACE_END(mitsuba)