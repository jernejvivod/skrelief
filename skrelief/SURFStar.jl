module SURFStar
export surfstar

using Statistics
using StatsBase
include("./utils/square_to_vec.jl")


"""
    surfstar(data::Array{<:Real,2}, target::Array{<:Number,1}, 
                  dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2))::Array{Float64,1}

Compute feature weights using SURFStar algorithm.

---
# Reference:
- Casey S. Greene, Daniel S. Himmelstein, Jeff Kiralis, and Jason H. Mo-
ore. The informative extremes: Using both nearest and farthest indivi-
duals can improve Relief algorithms in the domain of human genetics.
In Clara Pizzuti, Marylyn D. Ritchie, and Mario Giacobini, editors,
Evolutionary Computation, Machine Learning and Data Mining in Bi-
oinformatics, pages 182–193. Springer, 2010.
"""
function surfstar(data::Array{<:Real,2}, target::Array{<:Number,1}, 
                  dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2))::Array{Float64,1}

    # Initialize feature weights vector.
    weights = zeros(Float64, 1, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = maximum(data, dims=1)
    min_f_vals = minimum(data, dims=1)
    
    # Compute pairwise distances between samples (vector form).
    dists = Array{Float64}(undef, Int64((size(data, 1)^2 - size(data, 1))/2 + 1))
    dists[1] = 0  # Set first value of distances vector to 0 - accessed when i == j in square form indices.

    # Construct pairwise distances vector using vectorized distance function.
    top_ptr = 2
    @inbounds for idx = 1:size(data,1)-1
        upper_lim = top_ptr + size(data, 1) - idx - 1
        dists[top_ptr:upper_lim] = dist_func(data[idx:idx, :], data[idx+1:end, :])
        top_ptr = upper_lim + 1
    end

    # Get mean distance between all samples.
    mean_dist = Statistics.mean(dists[2:end])

    # Go over training samples.
    @inbounds for idx = 1:size(data, 1)
        
        # Row and column indices for querying pairwise distance vector.
        row_idxs = repeat([idx - 1], size(data, 1))
        col_idxs = collect(0:size(data, 1)-1)

        # Get indices in distance vector (from square form indices).
        neigh_idx = Int64.(square_to_vec(row_idxs, col_idxs, size(data, 1)) .+ 2)

        # Query distances to neighbours and get masks for neighbours that fall within
        # hypersphere with radius mean_dist and those that fall outside hypersphere.
        neigh_mask_near = dists[neigh_idx[neigh_idx .!= 0]] .< mean_dist
        neigh_mask_near[idx] = false
        neigh_mask_far = .!neigh_mask_near

        # Get masks of for near and far hits/misses.
        hit_neigh_mask_near = neigh_mask_near .& (target .== target[idx])
        miss_neigh_mask_near = neigh_mask_near .& (target .!= target[idx])
        hit_neigh_mask_far = neigh_mask_far .& (target .== target[idx])
        miss_neigh_mask_far = neigh_mask_far .& (target .!= target[idx])

        # Ignore samples that have no hits or misses in radius.
        if sum(hit_neigh_mask_near) == 0 || sum(miss_neigh_mask_near) == 0 || 
            sum(hit_neigh_mask_far) == 0 || sum(miss_neigh_mask_far) == 0
            continue
        end

        # Get class values of near and far misses.
        miss_classes_near = target[miss_neigh_mask_near]
        miss_classes_far = target[miss_neigh_mask_far]

        # Compute weights for near misses and compute weighting vector.
        weights_mult1 = Array{Float64}(undef, length(miss_classes_near))  # Allocate weights multiplier vector.
        cm = countmap(miss_classes_near)  # Count unique values.
        u = collect(keys(cm))
        c = collect(values(cm)) 
        neighbour_weights = c ./ length(miss_classes_near)  # Compute misses' weights.
        @inbounds for (i, val) = enumerate(u)                    # Build multiplier vector.
            find_res = findall(miss_classes_near .== val)
            weights_mult1[find_res] .= neighbour_weights[i]
        end

        # Compute weights for far misses and compute weighting vector.
        weights_mult2 = Array{Float64}(undef, length(miss_classes_far))  # Allocate weights multiplier vector.
        cm = countmap(miss_classes_far)  # Count unique values.
        u = collect(keys(cm))
        c = collect(values(cm)) 
        neighbour_weights = c ./ length(miss_classes_far)  # Compute misses' weights.
        @inbounds for (i, val) = enumerate(u)                    # Build multiplier vector.
            find_res = findall(miss_classes_far .== val)
            weights_mult2[find_res] .= neighbour_weights[i]
        end


        ### Weights Update ###

        # Penalty term for near neighbours.
        penalty_near = sum(abs.(data[idx:idx, :] .- data[hit_neigh_mask_near, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

        # Reward term for near neighbours.
        reward_near = sum(weights_mult1 .* abs.(data[idx:idx, :] .- data[miss_neigh_mask_near, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

        # Weights values for near neighbours.
        weights_near = weights .- penalty_near ./ (size(data, 1)*size(data[hit_neigh_mask_near, :], 1) + eps(Float64)) .+ 
            reward_near ./ (size(data, 1)*size(data[miss_neigh_mask_near, :], 1) + eps(Float64))


        # Penalty term for far neighbours.
        penalty_far = sum(abs.(data[idx:idx, :] .- data[hit_neigh_mask_far, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

        # Reward term for far neighbours.
        reward_far = sum(weights_mult2 .* abs.(data[idx:idx, :] .- data[miss_neigh_mask_far, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

        # Weights values for far neighbours.
        weights_far = weights .- penalty_far ./ (size(data, 1)*size(data[hit_neigh_mask_far, :], 1) + eps(Float64)) .+ 
            reward_far ./ (size(data, 1)*size(data[miss_neigh_mask_far, :], 1) + eps(Float64))

        # Update feature weights. 
        weights = weights_near - (weights_far - weights)

        ######################

    end

    # Return computed feature weights.
    return vec(weights)
end

end
