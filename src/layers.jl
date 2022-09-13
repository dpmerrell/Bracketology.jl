
import SparseMatFac: collect_view_gradients

"""
    layers.jl

    Layers for modeling teams in competition.
    Each of these is a row or column transformation
    appended to the base matrix factorization layer.
"""

#########################################
# Row/Column Shifts 

mutable struct ShiftLayer
    b::AbstractVector{<:Number}
    inv_view_mat::AbstractMatrix
end

@functor ShiftLayer

Flux.trainable(sl::ShiftLayer) = (b=sl.b, )

function ShiftLayer(M::Int)
    return ShiftLayer(randn(M) ./ 100.0, 
                      SparseMatrixCSC{Float32,Int32}(sparse(I, M, M)))
end

function (sl::ShiftLayer)(A::AbstractArray)
    return sl.b .+ A
end

function Base.view(sl::ShiftLayer, idx)
    nnz = length(idx)
    M = length(sl.b)
    nz_to_idx = sparse(idx, 1:nnz, ones(nnz), M, nnz)

    if typeof(sl.b) <: CuArray
        nz_to_idx = gpu(nz_to_idx)
    end

    return ShiftLayer(view(sl.b, idx), nz_to_idx)
end

function collect_view_gradients(sl::ShiftLayer, gradient_view)
    gradients = (b = sl.inv_view_mat * gradient_view.b, )
    return gradients
end



