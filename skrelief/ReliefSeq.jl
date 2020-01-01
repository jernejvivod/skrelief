module ReliefSeq
export reliefseq

push!(LOAD_PATH, @__DIR__)
using Relieff
using StatsBase
using Printf


"""
    reliefseq(data::Array{<:Real, 2}, target::Array{<:Number, 1}, m::Integer=-1, 
                   k_min::Integer=5, k_max::Integer=10, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2))::Array{Float64,1}

Compute feature weights using ReliefSeq algorithm.

---
# Reference:
- Brett A. McKinney, Bill C. White, Diane E. Grill, Peter W. Li, Ri-
chard B. Kennedy, Gregory A. Poland, and Ann L. Oberg. ReliefSeq: a
gene-wise adaptive-k nearest-neighbor feature selection tool for finding
gene-gene interactions and main effects in mRNA-Seq gene expression
data. PloS ONE, 8(12):e81527–e81527, Dec 2013.
"""
function reliefseq(data::Array{<:Real, 2}, target::Array{<:Number, 1}, m::Integer=-1, 
                   k_min::Integer=5, k_max::Integer=10, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2))::Array{Float64,1}

    # Check if k nearest misses and hits can be found for each class.
    upper_k_lim = minimum(counts(Int64.(target))) - 1
    if k_min > upper_k_lim
        throw(ArgumentError(k_min, "Insufficient number of instances to respect lower bound k_min.")) 
    end

    if k_max > upper_k_lim
        k_max = upper_k_lim
        @warn @sprintf "k_max reduced to %d because of insufficient number of instances." upper_k_lim
    end

    # Allocate array for storing weights for different values of k.
    res_mat = Array{Float64}(undef, size(data, 2), k_max - k_min + 1)
   
    # Go over interval of k values and compute ReliefF feature weights.
    col_idx = 1
    @inbounds for k = k_min:k_max
        res_mat[:, col_idx] = Relieff.relieff(data, target, m, k, "k_nearest", 1.0, dist_func)
        col_idx += 1
    end

    # For each feature choose highest weight.
    # Return computed feature weights.
    return vec(maximum(res_mat, dims=2))
end

end

