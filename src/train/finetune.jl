# end-to-end finetuning on lvl1 (cell line) or lvl2 (drug)
# mode: "scratch" (no pretraining, T0), "pretrained" (load a pretrain checkpoint, T2), "random" (same arch and init scheme, random weights, T1)

function finetune(cfg::Dict; data::Union{Nothing,LincsData} = nothing)
    set_math!(cfg)
    seed = cfg["seed"]; bs = cfg["batch_size"]; total = cfg["max_steps"]; lr = Float64(cfg["lr"])
    mode = cfg["mode"]; input = parse_input(cfg["input"]); level = cfg["level"]
    drop = Float64(cfg["drop_prob"]); micro = get(cfg, "micro_bs", 0)
    d = isnothing(data) ? load_data(cfg["data_path"]) : data
    idx, y, names = labels(d, level)
    sp = finetune_split(d, idx)
    G = n_genes(d); C = length(names)

    # model; a pretrained / random encoder keeps its pretraining norm and ETF input scaling
    obj_name = "none"
    if mode == "scratch"
        seed_all!(seed)
        enc = Encoder(input; n_genes=G, embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
                      n_layers=cfg["n_layers"], drop_prob=drop, norm=cfg["norm"])
    elseif mode in ("pretrained", "random")
        enc, pcfg, _ = load_encoder(cfg["ckpt"]; random=(mode == "random"), init_seed=get(cfg, "init_seed", 0))
        pcfg["input"] == cfg["input"] || error("checkpoint input $(pcfg["input"]) != finetune input $(cfg["input"])")
        cfg = merge(cfg, Dict("norm" => get(pcfg, "norm", "pre"), "etf_scale" => get(pcfg, "etf_scale", "raw")))
        enc = set_dropout(enc, drop); obj_name = pcfg["objective"]
        seed_all!(seed)
    else
        error("mode must be scratch, pretrained or random")
    end

    Xtr, Xva, Xte = finetune_inputs(cfg, d, idx, sp)
    ytr, yva, yte = y[sp.train], y[sp.val], y[sp.test]
    model = Classifier(enc, C; hidden_dim=cfg["hidden_dim"], drop_prob=drop, pool=Symbol(get(cfg, "pool", "mean")))
    model = to_device(model)
    hsd = get(cfg, "head_init", false) ? head_init!(model, batches(dev, Xtr[:, 1:min(5000, size(Xtr, 2))], bs)) : NaN

    opt = Flux.setup(Optimisers.AdamW(lr, (0.9, 0.999), Float64(get(cfg, "weight_decay", 0.0))), model)
    # per-token tables reach the mean-pooled output at 1/G, so their lr is scaled (gene table; RTF rank table)
    Optimisers.adjust!(opt.encoder.gene, lr * get(cfg, "gene_lr_mult", 1.0))
    input isa RTF && Optimisers.adjust!(opt.encoder.value, lr * get(cfg, "rank_lr_mult", 1.0))

    name = get(cfg, "run_name", obj_name == "none" ? "$(mode)_$(cfg["input"])_$(level)" : "$(mode)_$(obj_name)_$(cfg["input"])_$(level)")
    dir = run_dir(cfg, "finetune", name)
    cfg = merge(cfg, Dict("wandb_name" => "$(name)_s$(seed)"))
    @printf("finetune %s | %d classes | train %d val %d test %d | head_init dim-sd %.3g | dir %s\n",
            name, C, length(ytr), length(yva), length(yte), hsd, dir)

    r = train_classifier!(model, opt, (m, x, t) -> _ft_grad(m, x, t, micro), (Xtr, ytr, Xva, yva, Xte, yte), cfg, C)
    return save_classifier_run(dir, cfg, r, idx, sp, yte, names, (name = name, input = cfg["input"], mode = mode, objective = obj_name), hsd)
end

# batch gradient, optionally accumulated over micro-batches (same gradient as one pass)
function _ft_grad(model, x, y, micro::Integer)
    B = size(x, 2)
    if micro <= 0 || micro >= B
        r = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), y), model)
        return r.val, r.grad[1]
    end
    lv = 0f0; g = nothing
    for s in 1:micro:B
        c = s:min(s + micro - 1, B); w = Float32(length(c) / B)
        xb = x[:, c]; yb = y[:, c]
        r = Flux.withgradient(m -> Flux.logitcrossentropy(m(xb), yb) * w, model)
        lv += r.val; g = isnothing(g) ? r.grad[1] : Flux.Zygote.accum(g, r.grad[1])
    end
    return lv, g
end

# finetune inputs (train, val, test): the model input of the labelled samples (cfg etf_scale for ETF)
function finetune_inputs(cfg::Dict, d::LincsData, idx, sp)
    X = model_input(parse_input(cfg["input"]), d.X[:, idx], d, cfg)
    return X[:, sp.train], X[:, sp.val], X[:, sp.test]
end
