module VLSRelief
export vlsrelief
using StatsBase


"""
    vlsrelief(data::Array{<:Real,2}, target::Array{<:Number,1}, num_partitions_to_select::Integer, 
               num_subsets::Integer, partition_size::Integer, rba)::Array{Float64,1}

Compute feature weights using VLSRelief algorithm.

---
# Reference:
- Margaret Eppstein and Paul Haake. Very large scale ReliefF for genome-
wide association analysis. In 2008 IEEE Symposium on Computational
Intelligence in Bioinformatics and Computational Biology, CIBCB ’08,
2008.
"""
function vlsrelief(data::Array{<:Real,2}, target::Array{<:Number,1}, num_partitions_to_select::Integer, 
                   num_subsets::Integer, partition_size::Integer, rba)::Array{Float64,1}

    # Initialize feature weights vector.
    weights = zeros(Float64, size(data, 2))

    # Get vector of feature indices.
    feat_ind = collect(1:size(data, 2))

    # Get indices of partition starting indices.
    feat_ind_start_pos = collect(1:partition_size:size(data, 2))

    # Go over subsets and compute local ReliefF scores.
    @inbounds for i = 1:num_subsets

        # Randomly select k partitions and combine them to form a subset of features of examples.
        ind_sel = [collect(el:el+partition_size-1) for el in sample(feat_ind_start_pos, num_partitions_to_select, replace=false)]

        # Flatten list of indices' lists.
        ind_sel_unl = Array{Int64}(undef, num_partitions_to_select*partition_size)
        ptr = 1
        @inbounds for sel = ind_sel
            ind_sel_unl[ptr:ptr+partition_size-1] = sel
            ptr += partition_size
        end
        ind_sel_unl = ind_sel_unl[ind_sel_unl .<= feat_ind[end]]
        
        # Perform ReliefF algorithm on subset to obtain local weights.
        rba_weights = rba(data[:, ind_sel_unl], target)

        # Update weights using computed local weights.
        weights[ind_sel_unl] = vec(maximum(hcat(weights[ind_sel_unl], rba_weights), dims=2))
    end

    # Return computed feature weights.
    return weights
end

end

