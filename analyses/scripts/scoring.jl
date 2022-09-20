
include("util.jl")


function win_prob_many(model, a_vec, b_vec)
    
    N = length(a_vec)
    preds = zeros(N) 
    for (i, (a, b)) in enumerate(zip(a_vec, b_vec))
        preds[i], _, _ = win_prob(model, a, b)
    end

    return preds
end


function brier_score_kernel(pred_vec, true_vec)
    return sum(pred_vec .- true_vec) / length(pred_vec)
end


function brier_score(model, df)
    
    team_a_vec, team_b_vec, a_scores, b_scores, date_vec = unpack_df(df)
    pred_vec = win_prob_many(model, team_a_vec, team_b_vec)
    true_vec = (a_scores .> b_scores)

    return brier_score_kernel(pred_vec, true_vec)
end

