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
     * \brief Calculate normalization constant
     */
    void compute_constant();
    
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

        Float radius = Float(m_gauss_nodes[i]);
        Float weight = Float(m_gauss_weights[i]);
        Float sdf    = eval(radius);

        return { radius, weight, sdf };
    }

    /**
     * \brief Return all radii, Gaussian quadrature weights,
     *        and sdf values as arrays
     *
     * \return  A tuple (radius, weight, sdf) consisting of
     *
     *     radius      Radius of all division points
     *
     *     weight      Weight of all division points
     * 
     *     sdf         Value of sdf(radius)
     */
    std::tuple<Float, Float, Float> eval_gauss_all() const {
        Float radius_ = 0.f, weight_ = 0.f, sdf_ = 0.f;

        if constexpr (dr::is_jit_v<Float>) {
            Float radius = dr::load<Float>(m_gauss_nodes.data(), m_gauss_nodes.size());
            Float weight = dr::load<Float>(m_gauss_weights.data(), m_gauss_weights.size());
            Float sdf    = eval(radius);

            return { radius, weight, sdf };
        } else {
            Log(Error, "eval_gauss_all() can only be used with JIT Mitsuba variants!");
        }

        return { radius_, weight_, sdf_ };
    }
    
    /// Returns whether this size distribution is monodisperse
    MI_INLINE bool is_monodisperse() const { return m_is_monodisperse; }

    /// Return the minimum radius
    ScalarFloat min_radius() const { return m_min_radius; }

    /// Return the maximum radius
    ScalarFloat max_radius() const { return m_max_radius; }

    /// Return the number of Gaussian quadrature division points
    ScalarInt32 n_gauss() const { return m_g; }

    /// Return a string identifier
    std::string id() const override { return m_id; }

    /// Set a string identifier
    void set_id(const std::string& id) override { m_id = id; };

    /// Return a human-readable representation of the size distribution
    std::string to_string() const override = 0;

    void parameters_changed(const std::vector<std::string> & /* keys = {} */) override {
        compute_constant();
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
    ScalarFloat m_normalization;

    /// Minimum and maximum radius
    ScalarFloat m_min_radius;
    ScalarFloat m_max_radius;

    /// Number of Gaussian quadrature points
    uint32_t m_g;

    /// Gaussian quadrature division nodes and weights (already scaled and shifted)
    FloatX m_gauss_nodes;
    FloatX m_gauss_weights;

    /// Identifier (if available)
    std::string m_id;
};

MI_EXTERN_CLASS(SizeDistribution)
NAMESPACE_END(mitsuba)
