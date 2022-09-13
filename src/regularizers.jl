
"""
    Assemble a regularizer matrix that encourages
    a team's parameters to vary smoothly over time.
    That is: for a particular team, parameters that are
    near each other in time are more strongly encouraged
    to be similar to each other.
"""
function assemble_regmat(team_dates; epsilon=0.0)

    M = length(team_dates)

    diag = ones(M).*epsilon
    I = Int32[]
    J = Int32[]
    V = Float32[]

    prev_team, prev_date = team_dates[1]
    for idx=2:M
        team, date = team_dates[idx]
        
        if team == prev_team
            delta_t = Dates.days(Date(date) - Date(prev_date))/365.0
            weight = 1/delta_t
            diag[idx-1] += weight 
            diag[idx] += weight

            push!(I, idx-1)
            push!(J, idx)
            push!(V, -weight)
            
            push!(I, idx)
            push!(J, idx-1)
            push!(V, -weight)
        end

        prev_team = team
        prev_date = date
    end

    for (i, d) in enumerate(diag)
        push!(I, i)
        push!(J, i)
        push!(V, d)
    end

    regmat = sparse(I,J,V)
    return regmat
end


########################################
# Regularizer for X, Y

mutable struct MatrixRegularizer
    mat::AbstractMatrix
end

@functor MatrixRegularizer
Flux.trainable(mr::MatrixRegularizer) = ()

function MatrixRegularizer(team_dates::Vector{<:Tuple}; epsilon=0.0)
    mat = assemble_regmat(team_dates, epsilon=epsilon)
    return MatrixRegularizer(mat)
end


function (mr::MatrixRegularizer)(A::AbstractMatrix)
    B = A * mr.mat
    loss = 0.5*sum(A .* B)
    return loss
end


#######################################
# Layer regularizers

mutable struct ShiftRegularizer
    mat::AbstractMatrix
end

@functor ShiftRegularizer
Flux.trainable(sr::ShiftRegularizer) = ()

function ShiftRegularizer(team_dates::Vector{<:Tuple}; epsilon=0.0)
    mat = assemble_regmat(team_dates, epsilon=epsilon)
    return ShiftRegularizer(mat)
end

function (sr::ShiftRegularizer)(sl::ShiftLayer)
    return 0.5*dot(sl.b, sr.mat * sl.b)
end



