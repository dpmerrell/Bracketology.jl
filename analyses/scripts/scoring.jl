
include("util.jl")


function win_prob_many(model, a_vec, b_vec, date_vec)
    
    N = length(a_vec)
    preds = zeros(N)
    for (i, (a, b, d)) in enumerate(zip(a_vec, b_vec, date_vec))
        preds[i], _, _ = win_prob(model, a, b, d)
    end
    # The model tends to be overconfident.
    # It seems unreasonable to ever be more than
    # 65% sure of the outcome of any game.
    preds[preds .> 0.625] .= 0.625
    preds[preds .< 0.375] .= 0.375

    return preds
end


function brier_score_kernel(pred_vec, true_vec)
    return sum(pred_vec .- true_vec) / length(pred_vec)
end


function brier_score(model, df)
    
    team_a_vec, team_b_vec, a_scores, b_scores, date_vec = unpack_df(df)
    pred_vec = win_prob_many(model, team_a_vec, team_b_vec, date_vec)
    true_vec = (a_scores .> b_scores)

    return brier_score_kernel(pred_vec, true_vec)
end

# This is the function used by 538 for NFL forecasting
function brier_score_538_kernel(pred_vec, true_vec)
    return sum(25.0 .- (100 .*(pred_vec .- true_vec).^2))
end

# We negate it for reasons of consistency -- "scores" in this
# file are supposed to be minimized
function brier_score_538(model, df)
    team_a_vec, team_b_vec, a_scores, b_scores, date_vec = unpack_df(df)
    pred_vec = win_prob_many(model, team_a_vec, team_b_vec, date_vec)
    true_vec = (a_scores .> b_scores)
    
    return -brier_score_538_kernel(pred_vec, true_vec)
end

