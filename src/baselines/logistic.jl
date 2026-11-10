# linear baselines: logistic regression (elog / rlog) and shrinkage LDA (elda / rlda)

# one Dense layer, softmax via logitcrossentropy
logistic_model(G::Integer, C::Integer) = Flux.Chain(Flux.Dense(G => C))

# class means (G x C), pooled within-class covariance (Float64) and class counts
function lda_stats(X::AbstractMatrix{Float32}, y::AbstractVector{<:Integer}, C::Integer; chunk::Int = 50_000)
    G, N = size(X)
    M = zeros(Float64, G, C); n = zeros(Int, C)
    @inbounds for j in 1:N
        c = y[j]; n[c] += 1
        for g in 1:G
            M[g, c] += X[g, j]
        end
    end
    M ./= max.(n, 1)'
    S = zeros(Float64, G, G)
    for s in 1:chunk:N
        c = s:min(s + chunk - 1, N)
        Xc = Float64.(X[:, c]) .- M[:, y[c]]
        BLAS.syrk!('U', 'N', 1.0, Xc, 1.0, S)
    end
    S = Symmetric(S ./ max(N - C, 1), :U)
    return M, Matrix(S), n
end

# discriminant weights for shrinkage a: Sigma = (1 - a) S + a (tr S / G) I
function lda_weights(E::Eigen, M::Matrix{Float64}, logprior::Vector{Float64}, a::Float64)
    nu = mean(E.values)
    d = 1 ./ max.((1 - a) .* E.values .+ a * nu, 1e-12)
    W = E.vectors * (d .* (E.vectors' * M))
    b = -0.5 .* vec(sum(M .* W, dims=1)) .+ logprior
    return Float32.(W), Float32.(b)
end

lda_scores(W, b, X) = W' * X .+ b

# shrinkage LDA: fit on train, shrinkage chosen on val accuracy (ties -> lower val loss), test reported
function lda_baseline(cfg::Dict; data::Union{Nothing,LincsData} = nothing)
    model = cfg["model"]; level = cfg["level"]; seed = get(cfg, "seed", 1)
    model in ("elda", "rlda") || error("lda model must be elda or rlda, got $model")
    seed_all!(seed)
    d = isnothing(data) ? load_data(cfg["data_path"]) : data
    idx, y, names = labels(d, level)
    sp = finetune_split(d, idx)
    C = length(names)
    Xtr, Xva, Xte, info = baseline_inputs(d, idx, sp, model, cfg)
    ytr, yva, yte = y[sp.train], y[sp.val], y[sp.test]
    name = get(cfg, "run_name", baseline_name(cfg))
    dir = run_dir(cfg, "baseline", name)
    @printf("lda %s | %d classes | train %d val %d test %d | %s | dir %s\n", name, C, length(ytr), length(yva), length(yte), info, dir)

    t0 = time()
    M, S, n = lda_stats(Xtr, ytr, C)
    E = eigen(Symmetric(S))
    logprior = log.(max.(n, 1) ./ sum(n))
    grid = Float64.(get(cfg, "shrinkage_grid", [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]))
    evalset(W, b, Xs, ys) = (s = lda_scores(W, b, Xs); (Float64(Flux.logitcrossentropy(s, Flux.onehotbatch(ys, 1:C))), mean(Flux.onecold(s) .== ys), Flux.onecold(s)))
    curve = NamedTuple[]; best = (acc = -Inf, loss = Inf, a = NaN)
    for a in grid
        W, b = lda_weights(E, M, logprior, a)
        vl, va, _ = evalset(W, b, Xva, yva)
        push!(curve, (shrinkage = a, val_loss = vl, val_acc = va))
        @printf("shrinkage %.4g val %.4f val_acc %.4f\n", a, vl, va); flush(stdout)
        if va > best.acc || (va == best.acc && vl < best.loss)
            best = (acc = va, loss = vl, a = a)
        end
    end
    W, b = lda_weights(E, M, logprior, best.a)
    _, acc, p = evalset(W, b, Xte, yte)
    wall = time() - t0
    jldsave(joinpath(dir, "model.jld2"); W=W, b=b, shrinkage=best.a, config=cfg)
    jldsave(joinpath(dir, "predstrues.jld2"); preds_final=p, preds_best=p, trues=yte, test_sample_idx=idx[sp.test], label_names=names)
    write_curve(joinpath(dir, "curve.tsv"), curve)
    cfg = merge(cfg, Dict("shrinkage" => best.a))
    open(io -> TOML.print(io, Dict(k => v for (k, v) in cfg if v isa Union{Number,String,Bool})), joinpath(dir, "config.toml"), "w")
    row = (name = name, input = baseline_input_name(model), mode = "baseline", objective = model, level = level, seed = seed, steps = 0,
           final_test = acc, best_test = acc, best_step = 0, best_val_acc = best.acc,
           head_init_sd = NaN, gpu = "cpu", job = get(ENV, "SLURM_JOB_ID", "local"), wall_s = round(Int, wall),
           git = git_hash(), dir = dir)
    append_row(joinpath(cfg["results_dir"], "finetune.tsv"), row)
    @printf("test %.4f (shrinkage %.4g, val_acc %.4f) | %.0f s\n", acc, best.a, best.acc, wall)
    return row
end
