
using Bracketology, SparseMatFac, Zygote, ScikitLearnBase, Test

BTY = Bracketology


function layer_tests()
    
    @testset "Layer tests" begin
       M = 10

       sl = BTY.ShiftLayer(M)
       @test size(sl.b) == (M,)
       
       A = randn(M)
       shifted = sl(A)
       @test isapprox(shifted, A .+ sl.b)

       idx = rand(1:M, 50)
       sl_view = view(sl, idx)
       @test size(sl_view.inv_view_mat) == (M, length(idx))
       
       view_grads = ones(length(idx))
       grads = SparseMatFac.collect_view_gradients(sl_view, view_grads)
       @test size(grads) == (M,)

    end
end


function regularizer_tests()

    teams_a = ["badgers", "alligators", "cougars", "badgers", "cougars", "badgers"]
    teams_b = ["alligators", "cougars", "badgers", "cougars", "badgers", "alligators"]
    dates = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]

    @testset "Regularizer tests" begin
    
        unq_team_dates = BTY.unique_team_dates(teams_a, teams_b, dates)
        M = length(unq_team_dates)

        ############################
        # Matrix regularizer 
        mat_reg = BTY.MatrixRegularizer(unq_team_dates)
        @test size(mat_reg.mat) == (M,M)
        
        A = zeros(3,M)
        @test mat_reg(A) == 0.0

        A = randn(M,M)
        loss = mat_reg(A) 
        @test loss == 0.5*sum(A .* (A * mat_reg.mat))
        @test loss > 0

        (grad,) = Zygote.gradient(mat_reg, A)
        @test isapprox(grad, A*mat_reg.mat)

        ############################
        # Shift layer regularizer
        layer_reg = BTY.ShiftRegularizer(unq_team_dates)
        @test size(layer_reg.mat) == (M,M)
        
        b = zeros(M)
        sl = BTY.ShiftLayer(M)
        sl.b .= b
        @test layer_reg(sl) == 0.0

        b = randn(M)
        sl = BTY.ShiftLayer(M)
        sl.b .= b
        loss = layer_reg(sl) 
        @test loss == 0.5*sum(sl.b .* (layer_reg.mat * sl.b))
        @test loss > 0
        
        (grad,) = Zygote.gradient(layer_reg, sl)
        @test isapprox(grad.b, layer_reg.mat * b)

    end
end


function assemble_model_tests()

    teams_a = ["badgers", "alligators", "cougars", "badgers", "cougars", "badgers"]
    teams_b = ["alligators", "cougars", "badgers", "cougars", "badgers", "alligators"]
    dates = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]
    K = 2

    @testset "Assemble model tests" begin

        unq_team_dates = BTY.unique_team_dates(teams_a, teams_b, dates)
        @test length(unq_team_dates) == length(teams_a) + length(teams_b)

        model = BTY.assemble_model(teams_a, teams_b, dates; K=K, reg_weight=3.14)
        @test size(model.matfac.X_reg.mat, 1) == length(unq_team_dates)
        @test size(model.matfac.X) == (K, length(unq_team_dates))
    end
end


function fit_tests()

    teams_a = ["badgers", "alligators", "cougars", "badgers", "cougars", "badgers"]
    teams_b = ["alligators", "cougars", "badgers", "cougars", "badgers", "alligators"]
    dates = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]
    a_scores = rand(1:100, length(teams_a))
    b_scores = rand(1:100, length(teams_b))
    K = 2

    @testset "Fit tests" begin
    
        model = BTY.CompetitionModel(teams_a, teams_b, dates; K=2, reg_weight=0.01)
        start_X = copy(model.matfac.X)
        start_Y = copy(model.matfac.Y)
        start_b_row = copy(model.matfac.row_transform.b)
        start_b_col = copy(model.matfac.col_transform.b)

        fit!(model, teams_a, teams_b, a_scores, b_scores, dates; max_iter=10, verbosity=1)

        @test !isapprox(model.matfac.X, start_X)
        @test !isapprox(model.matfac.Y, start_Y)
        @test !isapprox(model.matfac.row_transform.b, start_b_row)
        @test !isapprox(model.matfac.col_transform.b, start_b_col)
    end    

end


function impute_tests()
    
    teams_a = ["badgers", "alligators", "cougars", "badgers", "cougars", "badgers"]
    teams_b = ["alligators", "cougars", "badgers", "cougars", "badgers", "alligators"]
    dates = ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]
    a_scores = rand(1:100, length(teams_a))
    b_scores = rand(1:100, length(teams_b))
    K = 2

    model = BTY.CompetitionModel(teams_a, teams_b, dates; K=2, reg_weight=0.01)
    fit!(model, teams_a, teams_b, a_scores, b_scores, dates; max_iter=10, verbosity=1)

    @testset "Imputation tests" begin
        # Just make sure it runs and spits out nonnegative numbers
        a_score, b_score = BTY.impute_game(model, "badgers", "cougars") 
       
        @test a_score >= 0
        @test b_score >= 0 
    end

    @testset "Win probability tests" begin
        # Just make sure it runs and spits out a probability
        pA = BTY.win_prob(model, "alligators", "badgers")
        @test (pA > 0.0) & (pA < 1.0)
    end

end


function main()

    layer_tests()
    regularizer_tests()
    assemble_model_tests()
    fit_tests()
    impute_tests()

end


main()


