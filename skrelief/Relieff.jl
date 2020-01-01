module Relieff
export relieff

using StatsBase
include("./utils/square_to_vec.jl")


"""
    relieff(data::Array{<:Real,2}, target::Array{<:Number,1}, m::Integer=-1, 
                 k::Integer=5, mode::String="k_nearest", sig::Float64=1.0, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2))::Array{Float64,1}

Compute feature weights using ReliefF algorithm.

---
# Reference:
- Marko Robnik-Šikonja and Igor Kononenko. Theoretical and empirical
analysis of ReliefF and RReliefF. Machine Learning, 53(1):23–69, Oct
2003.
"""
function relieff(data::Array{<:Real,2}, target::Array{<:Number,1}, m::Integer=-1, 
                 k::Integer=5, mode::String="k_nearest", sig::Float64=1.0, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2))::Array{Float64,1}


    # Initialize feature weights vector.
    weights = zeros(Float64, 1, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = maximum(data, dims=1)
    min_f_vals = minimum(data, dims=1)
    
    # Sample m examples without replacement.
    sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)
    if (m == -1) m = size(data, 1) end
    
    # Compute probabilities of class values in training set.
    classes_map = countmap(target)
    num_samples = length(target)
    p_classes = Array{Float64}(undef, length(keys(classes_map)), 2)
    @inbounds for (idx, p) = enumerate(classes_map)  # Compute matrix representation of probabilities.
        p_classes[idx, :] = [p[1], p[2]/num_samples]
    end

    # Compute pairwise distances between samples (vector form).
    dists = Array{Float64}(undef, Int64((size(data, 1)^2 - size(data, 1))/2 + 1))
    dists[1] = 0  # Set first value of distances vector to 0 - accessed when i == j in square form indices.

    # Construct pairwise distances vector using vectorized distance function.
    top_ptr = 2
    for idx = 1:size(data,1)-1
        upper_lim = top_ptr + size(data, 1) - idx - 1
        dists[top_ptr:upper_lim] = dist_func(data[idx:idx, :], data[idx+1:end, :])
        top_ptr = upper_lim + 1
    end


    # Go over sampled indices.
    @inbounds for idx = sample_idxs

        # Row and column indices for querying pairwise distance vector.
        row_idxs = repeat([idx - 1], size(data, 1))
        col_idxs = collect(0:size(data, 1)-1)

        # Get indices in distance vector (from square form indices).
        neigh_idx = Int64.(square_to_vec(row_idxs[target .== target[idx]], col_idxs[target .== target[idx]], size(data, 1))) .+ 2
        idx_k_nearest_same = partialsortperm(dists[neigh_idx], 1:k+1)[2:end]
        
        # Get k nearest hits.
        k_nearest_same = data[target .== target[idx], :][idx_k_nearest_same, :]
       
        # Allocate matrix for storing the k nearest misses.
        k_nearest_other = Array{Float64}(undef, k * (length(keys(classes_map)) - 1), size(data, 2))

        # Go over class values not equal to class value of currently sampled sample. 
        top_ptr = 1
        @inbounds for cl = keys(classes_map)
            if cl != target[idx]
                neigh_idx_nxt = Int64.(square_to_vec(row_idxs[target .== cl], col_idxs[target .== cl], size(data, 1))) .+ 2
                idx_k_nearest_other_nxt = partialsortperm(dists[neigh_idx_nxt], 1:k)
                k_nearest_other_nxt = data[target .== cl, :][idx_k_nearest_other_nxt, :]

                # Find k closest samples from same class.
                k_nearest_other[top_ptr:top_ptr+k-1, :] = k_nearest_other_nxt
                top_ptr += k
            end
        end

        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 1] .!= target[idx], 2]
        
        # Compute diff sum weights for closest examples from different classes.
        p_weights = p_classes_other./(1 .- p_classes[p_classes[:, 1] .== target[idx], 2])

        # Compute diff sum weights for closest examples from different class.
        weights_mult = reshape(repeat(p_weights, inner=k), :, 1)


        ### Weights Update - K-NEAREST #######
        if mode == "k_nearest"

            # Penalty term.
            penalty = sum(abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=1)

            # Reward term.
            reward = sum(weights_mult .* (abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)

            # Weights update.
            weights = weights .- penalty./(m*k) .+ reward./(m*k)

        ### Weights Update - DIFF ############
        elseif mode == "diff"

            # Compute weights for each nearest hit
            d_vals_closest_same = 1 ./ (sum(abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=2) .+ eps(Float64))
            dist_weights_penalty = d_vals_closest_same ./ sum(d_vals_closest_same)

            # Distance weights for reward term
            d_vals_closest_other = 1 ./ (sum(abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=2) .+ eps(Float64))
            dist_weights_reward = d_vals_closest_other ./ sum(d_vals_closest_other)
                
            # Penalty term
            penalty = sum(dist_weights_penalty .* (abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)
            
            # Reward term
            reward = sum(weights_mult .* (dist_weights_reward .* (abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)))), dims=1)

            # Weights update
            weights = weights .- penalty./m .+ reward./m

        ### Weights Update - EXP-RANK ########
        elseif mode == "exp_rank"
            
            # Compute weights for nearest hits.
            # NOTE: nearest hits are already sorted by distance.
            exp_rank_same = exp.(-(1:size(k_nearest_same, 1)/sig))
            closest_same_weights = exp_rank_same/sum(exp_rank_same)
            
            # Compute weights for nearest misses.
            sp = sortperm(vec(dist_func(data[idx:idx, :], k_nearest_other)))
            closest_other_ranks = collect(1:size(k_nearest_other, 1))
            closest_other_ranks[sp] = closest_other_ranks
            exp_rank_other = exp.(-(closest_other_ranks/sig))
            closest_other_weights = exp_rank_other/sum(exp_rank_other)
            
            # Penalty term
            penalty = sum(closest_same_weights .* (abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)
            
            # Reward term
            reward = sum(weights_mult .* (closest_other_weights .* (abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)))), dims=1)

            # Weights update
            weights = weights .- penalty./m .+ reward./m

        end

        ######################################


    end
   
    # Return computed feature weights.
    return vec(weights)

end

end

