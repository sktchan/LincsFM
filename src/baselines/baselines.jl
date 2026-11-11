# no-pretraining baselines: e* = expression, r* = rank features; log = logistic regression, mlp = relu MLP, lda = shrinkage LDA

include("logistic.jl")
include("mlp.jl")
include("trivial.jl")

_flag(x) = x isa Bool ? x : x != 0

baseline_input_name(model::AbstractString) = model[1] == 'e' ? "etf" : model[1] == 'r' ? "rtf" : error("unknown baseline $model")
baseline_name(cfg::Dict) = "$(cfg["model"])$(cfg["model"][1] == 'e' && _flag(get(cfg, "standardize", false)) ? "_z" : "")_$(cfg["level"])"

# train / val / test inputs: rank features, or expression z-scored per gene with the stored pretrain-train mean/sd (as ETF A) and clipped to +-clip (0 = no clip)
function baseline_inputs(d::LincsData, idx, sp, model::AbstractString, cfg::Dict)
    X = d.X[:, idx]
    if model[1] == 'r'
        X = rank_features(gene_ranks(X, d.medians), X); info = "rank features"
    elseif _flag(get(cfg, "standardize", false))
        clip = Float32(get(cfg, "clip", 10))
        X = (X .- d.gene_mean) ./ d.gene_sd
        frac = mean(abs.(X[:, sp.train]) .> clip)
        clip > 0 && (X = clamp.(X, -clip, clip))
        info = @sprintf("z-scored, clip %g (%.4f%% of train values)", clip, 100 * (clip > 0 ? frac : 0.0))
    else
        info = "raw expression"
    end
    return X[:, sp.train], X[:, sp.val], X[:, sp.test], info
end

# train one logistic / MLP baseline with the finetune() loop conventions (train/classify.jl)
function baseline(cfg::Dict; data::Union{Nothing,LincsData} = nothing)
    set_math!(cfg)
    model_name = cfg["model"]; level = cfg["level"]
    model_name in ("elda", "rlda") && return lda_baseline(cfg; data)
    model_name in ("elog", "rlog", "emlp", "rmlp") || error("model must be elog, rlog, emlp, rmlp, elda or rlda, got $model_name")
    is_log = endswith(model_name, "log")
    seed = cfg["seed"]; bs = cfg["batch_size"]; total = cfg["max_steps"]
    lr = Float64(is_log ? get(cfg, "lr", 1e-3) : cfg["lr"])
    drop = Float64(get(cfg, "drop_prob", 0.0)); wd = Float64(get(cfg, "weight_decay", 0.0))
    cfg = merge(cfg, Dict("lr" => lr, "job_type" => get(cfg, "job_type", "baseline")))
    d = isnothing(data) ? load_data(cfg["data_path"]) : data
    idx, y, names = labels(d, level)
    sp = finetune_split(d, idx)
    G = n_genes(d); C = length(names)
    Xtr, Xva, Xte, info = baseline_inputs(d, idx, sp, model_name, cfg)
    ytr, yva, yte = y[sp.train], y[sp.val], y[sp.test]

    # model
    seed_all!(seed)
    if is_log
        model = logistic_model(G, C); arch = "linear $G => $C"
    else
        sizes = mlp_sizes(G, C, cfg["n_layers"], get(cfg, "mlp_hidden_dim", 0); shape=get(cfg, "mlp_shape", "const"))
        model = mlp_model(sizes, drop); arch = "MLP $(join(sizes, "-"))"
    end
    model = to_device(model)
    opt = Flux.setup(Optimisers.AdamW(lr, (0.9, 0.999), wd), model)

    name = get(cfg, "run_name", baseline_name(cfg))
    dir = run_dir(cfg, "baseline", name)
    cfg = merge(cfg, Dict("wandb_name" => "$(name)_s$(seed)"))
    @printf("baseline %s | %d classes | train %d val %d test %d | %s | %s | lr %.3g | dir %s\n",
            name, C, length(ytr), length(yva), length(yte), arch, info, lr, dir)

    grad(m, x, t) = (r = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), t), m); (r.val, r.grad[1]))
    res = train_classifier!(model, opt, grad, (Xtr, ytr, Xva, yva, Xte, yte), cfg, C)
    return save_classifier_run(dir, cfg, res, idx, sp, yte, names,
                               (name = name, input = baseline_input_name(model_name), mode = "baseline", objective = model_name), NaN)
end
