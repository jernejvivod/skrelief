module IRelief
export IRelief

using LinearAlgebra


"""
    get_mean_m_h_vals(data::Array{<:Real,2}, target::Array{<:Number,1}, 
                           dist_weights::Array{Float64}, sig::Number)::Tuple{Array{Float64}, Array{Float64}}

Get mean m and mean h values (see reference, auxiliary function).
"""
function get_mean_m_h_vals(data::Array{<:Real,2}, target::Array{<:Number,1}, 
                           dist_weights::Array{Float64}, sig::Number)::Tuple{Array{Float64}, Array{Float64}}
    
    # Allocate matrix for storing results.
    mean_h = Array{Float64}(undef, size(data))
    mean_m = Array{Float64}(undef, size(data))

    # Go over rows of pairwise differences.
    for idx = 1:size(data, 1)

        # Compute m values.
        m_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data[target .!= target[idx], :])
        h_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data[(target .== target[idx]) .& (1:size(data, 1) .!= idx), :])

        # Compute kernel function values.
        f_m_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*m_vals, dims=2)/sig)
        f_h_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*h_vals, dims=2)/sig)

        # Compute vector of probabilities of misses being nearest misses.
        pm_vec = f_m_vals./(sum(f_m_vals) + eps(Float64))
        ph_vec = f_h_vals./(sum(f_h_vals) + eps(Float64))

        # Compute mean_m_values for each example
        mean_m[idx, :] = sum(pm_vec.*m_vals, dims=1)
        mean_h[idx, :] = sum(ph_vec.*h_vals, dims=1)
    end

    return mean_m, mean_h
end


"""
    function get_gamma_vals(data::Array{<:Real,2}, target::Array{<:Number,1}, dist_weights::Array{Float64}, sig::Number)

Get gamma values (see reference, auxiliary function)
"""
function get_gamma_vals(data::Array{<:Real,2}, target::Array{<:Number,1}, dist_weights::Array{Float64}, sig::Number)
    
    # Allocate array for storing results.
    po_vals = Array{Float64}(undef, size(data, 1))

    # Go over rows of distance matrix.
    for idx = 1:size(data, 1)
        
        # Compute probability of n-th example being an outlier.
        m_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data[target .!= target[idx], :])
        d_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data)
        f_m_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*m_vals, dims=2)/sig)
        f_d_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*d_vals, dims=2)/sig)
        po_vals[idx] = sum(f_m_vals)/(sum(f_d_vals) + eps(Float64))
    end
    
    # Gamma values are probabilities of examples being inliers.
    return 1 .- po_vals
end


"""
    function get_nu(gamma_vals::Array{Float64}, mean_m_vals::Array{Float64}, mean_h_vals::Array{Float64}, nrow::Int64)

Get nu value (see reference, auxiliary function).
"""
function get_nu(gamma_vals::Array{Float64}, mean_m_vals::Array{Float64}, mean_h_vals::Array{Float64}, nrow::Int64)
    return (1/nrow) .* sum(reshape(gamma_vals, length(gamma_vals), 1) .* (mean_m_vals .- mean_h_vals), dims=1)
end

"""
    function irelief(data::Array{<:Real,2}, target::Array{<:Number,1}, max_iter::Int64, k_width::Number, conv_condition::Number, initial_w_div::Number)::Array{Float64}

Compute feature weights using Iterative Relief algorithm.

---
# Reference:
- Yijun Sun and Jian Li. Iterative RELIEF for feature weighting. In ICML
2006 - Proceedings of the 23rd International Conference on Machine
Learning, volume 2006, pages 913–920, 2006.
"""
function irelief(data::Array{<:Real,2}, target::Array{<:Number,1}, max_iter::Int64, k_width::Number, conv_condition::Number, initial_w_div::Number)::Array{Float64}

    # Intialize convergence indicator and distance weights for features.
    convergence = false 
    dist_weights = ones(Float64, size(data, 2))/initial_w_div

    # Initialize iteration counter.
    iter_count = 0

    ### Main iteration loop. ###
    while iter_count < max_iter && !convergence

        # Get gamma values and compute nu.
        gamma_vals = get_gamma_vals(data, target, dist_weights, k_width)
        
        # Get mean m and mean h vals for all examples.
        mean_m_vals, mean_h_vals = get_mean_m_h_vals(data, target, dist_weights, k_width)
        
        # Get nu vector.
        nu = get_nu(gamma_vals, mean_m_vals, mean_h_vals, size(data, 1)) 

        # Update distance weights.
        dist_weights_nxt = clamp.(nu, 0, Inf)/(LinearAlgebra.norm(clamp.(nu, 0, Inf)) + eps(Float64))

        # Check if convergence criterion satisfied. If not, continue with next iteration.
        if sum(abs.(dist_weights_nxt .- dist_weights)) < conv_condition
            dist_weights = dist_weights_nxt
            convergence = true
        else
            dist_weights = dist_weights_nxt
            iter_count += 1
        end 
    end

    ############################

    # Return feature ranks and last distance weights.
    return vec(dist_weights)
end

end

