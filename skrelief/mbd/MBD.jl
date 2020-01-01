module MBD
export get_dist_func

using Pkg, Random, Distributions, StatsBase, Parameters

ENV["PYTHON"] = "python3.7"
Pkg.build("PyCall")

# Set seed for random value generator.
Random.seed!(123)


# Structure representing a node in an i-tree.
@with_kw mutable struct It_node
	l::Union{It_node, Nothing}
	r::Union{It_node, Nothing}
	split_attr::Union{Int, Nothing}
	split_val::Union{Float64, Nothing}
	level::Int
	mass::Int = 0 
end


function get_random_itree(data_sub)

    # Define auxiliary function to implement recursion.
    function random_itree(x_in; current_height, lim)

        # Base case - reached height limit or single sample left.
		if current_height >= lim || size(x_in, 1) <= 1
        	return It_node(l=nothing, r=nothing, split_attr=nothing, split_val=nothing, level=current_height)
        else

        # Randomly select an attribute q.
        q = rand(1:size(x_in, 2))

        # Randomly select a split point p between min and max values of attribute q in X.
		min_q_x_in = minimum(x_in[:, q])
		max_q_x_in = maximum(x_in[:, q])
		if min_q_x_in == max_q_x_in
			p = min_q_x_in
		else
        	p = rand(Uniform(minimum(x_in[:, q]), maximum(x_in[:, q])))
		end

        # Get left and right subtrees.
        xl = x_in[x_in[:, q] .< p, :]
        xr = x_in[x_in[:, q] .>= p, :]

        # Recursive case - build node and make recursive case for subtrees.
        return It_node(l=random_itree(xl, current_height=current_height+1, lim=lim),
                   r=random_itree(xr, current_height=current_height+1, lim=lim),
                   split_attr=q,
                   split_val=p,
                   level=current_height)
        end
    end

    # Build i-tree.
    return random_itree(data_sub, current_height=0, lim=10)
end


function get_n_random_itrees(n, subs_size, data)

    # Make array of i-tree nodes 
    random_itrees = Array{It_node, 1}(undef, n)

    # Build n i-trees using random data subsets.
    for k = 1:n
        # Get a random sample of training examples to build next random itree.
        data_sub = data[sample(1:size(data, 1), subs_size, replace=false), :]
        # Get next random itree.
        random_itrees[k] = get_random_itree(data_sub)  
    end

    # Return array of built i-trees and size of data subsets used
    # to build the i-trees.
    return random_itrees, subs_size

end


function get_lowest_common_node_mass(itree, x1, x2)

    # Base case #1: if node is a leaf, return its mass.
    if itree.split_val == nothing
    	return itree.mass
    end

    # Base case #2: if x1 and x2 are in different subtrees, return current node's mass.
	if (x1[itree.split_attr] < itree.split_val) != (x2[itree.split_attr] < itree.split_val)
    	return itree.mass
    end

    # Recursive case #1: if both examples if left subtree, make recursive call.
	if (x1[itree.split_attr] < itree.split_val) && (x2[itree.split_attr] < itree.split_val)
    	return get_lowest_common_node_mass(itree.l, x1, x2)
    end

    # Recursive case #2: if both examples in right subtree, make recursive call.
	if (x1[itree.split_attr] >= itree.split_val) && (x2[itree.split_attr] >= itree.split_val)
    	return get_lowest_common_node_mass(itree.r, x1, x2)
    end

end


function mass_based_dissimilarity(x1, x2, itrees, subs_size)

    # In each i-tree, find lowest nodes containing both examples and accumulate masses.
    sum_masses = 0
    for i = 1:length(itrees)
    	sum_masses += get_lowest_common_node_mass(itrees[i], x1, x2)/subs_size
    end

    # Divide by number of space partitioning models.
    return (2/length(itrees)) * sum_masses 

end


function get_node_masses(itrees, data)

    # Traverse i-tree with example and increment masses of visited nodes.
    function traverse(example, it_node)

        # Base case: in leaf.
        if it_node.l == nothing && it_node.r == nothing
        	it_node.mass += 1

        # Recursive case #1: if split attribute value lower than split value.
        elseif example[it_node.split_attr] < it_node.split_val
        	it_node.mass += 1
        	traverse(example, it_node.l)  # Traverse left subtree.

        # Recursive case #2: if split attribute value greater or equal to split value.
        else
        	it_node.mass += 1
        	traverse(example, it_node.r)  # Traverse right subtree.
        end
    end

    # Compute masses of nodes in itree.
    function compute_masses(itree, data)

        # Go over indices of samples in training data
        # and traverse i-tree with each sample.
		for example_idx in 1:size(data, 1)
			traverse(data[example_idx, :], itree)
        end
    end

    # Go over itrees and set masses of nodes.
    for itree in itrees  
        compute_masses(itree, data)
    end
end


"""
    get_dist_func(data::Array{<:Real,2}, num_itrees::Integer=10)

Get metric function that computes mass based dissimilarity.

---
# Reference:
- Kai Ming Ting, Ye Zhu, Mark Carman, Yue Zhu, and Zhi-Hua Zhou.
Overcoming key weaknesses of distance-based neighbourhood methods
using a data dependent dissimilarity measure. In Proceedings of the 22.
ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining, KDD ’16, pages 1205–1214, New York, NY, USA, 2016.
ACM.

---
# Arguments
- `data::Array{<:Real,2}`: matrix containing training samples as rows.
- `m::Integer=10`: number of i-trees used in space partitioning.
---

# Examples
```julia-repl
julia> num_itrees = 15
15
julia> dist_func = get_dist_func(data, num_itrees)
julia> dist_func(data[1:1,:], data[2:2,:])
1.2860000000000003
julia> dist_func(data[1:1,:], data[1:1,:])
0.022000000000000002
```
"""

function get_dist_func(data::Array{<:Real,2}, num_itrees::Integer=10)

    # Compute num_itrees random i-trees.
    itrees, subs_size = get_n_random_itrees(num_itrees, size(data, 1), data)

    # Compute masses of i-tree nodes in i-tree forest.
    get_node_masses(itrees, data)
     
    # Dissimilarity function returned as result.
    function res_func(e1, e2)

        # If both e1 and e2 are vectors, compute single dissimilarity.
        if size(e1, 1) == 1 && size(e2, 1) == 1
            dists = [mass_based_dissimilarity(e1, e2, itrees, subs_size)]
        elseif size(e1, 1) == 1  # If e2 is a matrix, compute dissimilarity of e1 with each row vector forming e2.
            dists = zeros(Float64, size(e2, 1))
            for idx in 1:size(e2, 1)
                dists[idx] = mass_based_dissimilarity(e1, e2[idx, :], itrees, subs_size) 
            end
        elseif size(e2, 1) == 1  # If e1 is a matrix, compute dissimilarity of e2 with each row vector forming e1.
            dists = zeros(Float64, size(e1, 1))
            for idx in 1:size(e1, 1)
                dists[idx] = mass_based_dissimilarity(e2, e1[idx, :], itrees, subs_size) 
            end
        end

        # Return distances (either single value or vector).
        return dists
    end

    # Return dissimilarity function.
    return (e1, e2) -> res_func(e1, e2)
end

end
