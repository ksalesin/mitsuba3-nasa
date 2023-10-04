#pragma once

#include <mitsuba/render/fwd.h>

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

    using FloatX = dr::DynamicArray<dr::scalar_t<Float>>;

    /**
     * \brief Evaluate the size distribution function (sdf)
     *
     * \param r
     *     The radius (size)
     */
    virtual Float eval(Float r, bool normalize = true) const = 0;

    /**
     * \brief Calculate normalization constant, average geometric cross section,
     *        effective radius, and effective variance.
     */
    void calculate_constants();
    
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

        Float shift = 0.5f * (m_max_radius + m_min_radius);
        Float scale = 0.5f * (m_max_radius - m_min_radius);

        Float radius = Float(m_gauss_nodes[i]) * scale + shift;
        Float weight = Float(m_gauss_weights[i]) * scale;
        Float sdf    = eval(radius);

        return { radius, weight, sdf };
    }
    
    /// Returns whether this size distribution is monodisperse
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

    /// Return a human-readable representation of the size distribution
    std::string to_string() const override = 0;

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        calculate_constants();
    }

    //! @}
    // -----------------------------------------------------------------------

    MI_DECLARE_CLASS()
protected:
    SizeDistribution(const Properties &props);
    virtual ~SizeDistribution();

protected:
    bool m_is_monodisperse = false;

    /// Normalization factor (inverse of integral)
    double m_normalization;

    /// Minimum and maximum radius
    Float m_min_radius;
    Float m_max_radius;

    /// Effective radius and variance
    Float m_reff;
    Float m_veff;

    /// Number of Gaussian quadrature points
    uint32_t m_g;

    /// Gaussian quadrature division nodes and weights
    FloatX m_gauss_nodes;
    FloatX m_gauss_weights;

    /// Identifier (if available)
    std::string m_id;
};

MI_EXTERN_CLASS(SizeDistribution)
NAMESPACE_END(mitsuba)
