
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
    inv_view_mat::AbstractMatrix     # M x M (at construction)
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
    nz_to_idx = sparse(idx, 1:nnz, ones(nnz), M, nnz) # M x M'

    if typeof(lcl.data) <: CuArray
        nz_to_idx = gpu(nz_to_idx)
    end

    return LinearCovariateLayer(view(lcl.data, :, idx),
                                view(lcl.w, :, idx), nz_to_idx)
end

function collect_view_gradients(lcl::LinearCovariateLayer, gradient_view)
    gradients = (w = gradient_view.w * lcl.inv_view_mat, )
    return gradients
end

#########################################
mutable struct ConstantLayer
    constant::Vector{<:Number}
end

function ConstantLayer(v::Number)
    return ConstantLayer([v])
end

function (cl::ConstantLayer)(A::AbstractVector)
    return cl.constant .+ A
end

function ChainRulesCore.rrule(cl::ConstantLayer, A::AbstractVector)
    result = cl.constant .+ A

    function constantlayer_pullback(result_bar)
        constant_bar = zero(cl.constant)
        constant_bar .+= sum(result_bar)
        A_bar = copy(result_bar)
        return ChainRulesCore.Tangent{ConstantLayer}(constant=constant_bar), A_bar
    end

    return result, constantlayer_pullback
end

function collect_view_gradients(cl::ConstantLayer, gradient_view)
    return gradient_view
end 

#########################################
# Combined layer

mutable struct CombinedLayer
    constant::ConstantLayer
    shift::ShiftLayer
    covariates::LinearCovariateLayer 
end

@functor CombinedLayer
Flux.trainable(cl::CombinedLayer) = (constant=cl.constant,
                                     shift=cl.shift, covariates=cl.covariates)

function CombinedLayer(constant_term::Number, covariates::AbstractMatrix)
    Kc, M = size(covariates)
    return CombinedLayer(ConstantLayer(constant_term), 
                         ShiftLayer(M), 
                         LinearCovariateLayer(covariates))
end

function (cl::CombinedLayer)(A::AbstractArray)
    return cl.constant(cl.covariates(cl.shift(A)))
end



function Base.view(cl::CombinedLayer, idx)
    return CombinedLayer(cl.constant,
                         view(cl.shift, idx),
                         view(cl.covariates, idx))
end

function collect_view_gradients(cl::CombinedLayer, gradient_view)
    return (constant=gradient_view.constant,
            shift=collect_view_gradients(cl.shift, gradient_view.shift), 
            covariates=collect_view_gradients(cl.covariates, gradient_view.covariates))
end


