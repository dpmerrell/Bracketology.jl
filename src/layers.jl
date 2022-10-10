
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
    return ShiftLayer(randn(M) .* 1e-5, 
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


#########################################
# Covariate linear regression layer

mutable struct LinearCovariateLayer
    data::AbstractMatrix{<:Number}   # K x M
    w::AbstractMatrix{<:Number}      # K x M
    inv_view_mat::AbstractMatrix
end

@functor LinearCovariateLayer
Flux.trainable(lcl::LinearCovariateLayer) = (w=lcl.w,)

function LinearCovariateLayer(data::AbstractMatrix)
    M = size(data, 1)
    return LinearCovariateLayer(data, randn(size(data)) .* 1e-5,
                                SparseMatrixCSC{Float32,Int32}(sparse(I,M,M)))
end

function (lcl::LinearCovariateLayer)(A::AbstractArray)
    return A .+ vec(sum(lcl.w .* lcl.data, dims=1))
end

function Base.view(lcl::LinearCovariateLayer, idx)
    nnz = length(idx)
    K, M = size(lcl.data)
    nz_to_idx = sparse(idx, 1:nnz, ones(nnz), M, nnz)

    if typeof(lcl.data) <: CuArray
        nz_to_idx = gpu(nz_to_idx)
    end

    return LinearCovariateLayer(view(lcl.data, :, idx),
                                view(lcl.w, :, idx), nz_to_idx)
end

function collect_view_gradients(lcl::LinearCovariateLayer, gradient_view)
    gradients = (w = lcl.inv_view_mat * gradient_view.w, )
    return gradients
end


#########################################
# Combined layer

mutable struct CombinedLayer
    shift::ShiftLayer
    covariates::LinearCovariateLayer 
end


function (cl::CombinedLayer)(A::AbstractArray)
    return cl.covariates(cl.shift(A))
end

function Base.view(cl::LinearCovariateLayer, idx)
    return CombinedLayer(view(cl.shift, idx),
                         view(cl.covariates, idx))
end

function collect_view_gradients(cl::CombinedLayer, gradient_view)
    return (shift=collect_view_gradients(gradient_view.shift), 
            covariates=collect_view_gradients(gradient_view.covariates))
end


