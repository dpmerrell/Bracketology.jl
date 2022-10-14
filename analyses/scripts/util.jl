
function unpack_df(train_df)
    team_a_vec = train_df[:,:TeamA]
    team_b_vec = train_df[:,:TeamB]
    date_vec = train_df[:,:Date]
    a_scores = train_df[:,:ScoreA]
    b_scores = train_df[:,:ScoreB]
    a_covariates = Matrix{Float32}(train_df[:,[:IsHomeA]])
    b_covariates = Matrix{Float32}(train_df[:,[:IsHomeB]])
    return team_a_vec, team_b_vec, a_scores, b_scores, date_vec, a_covariates, b_covariates
end


