# pretraining loop for every objective and input
# required cfg keys: input, objective, lr, drop_prob, mask_ratio, max_steps; lrecon also ema_decay

function pretrain(cfg::Dict; data::Union{Nothing,LincsData} = nothing)
    set_math!(cfg)
    seed = cfg["seed"]; bs = cfg["batch_size"]; total = cfg["max_steps"]
    eval_every = get(cfg, "eval_every", 1000); max_val = get(cfg, "max_val_samples", 5000)
    wd = Float64(get(cfg, "weight_decay", 0.0)); clip = Float64(get(cfg, "clip", 1.0))
    input = parse_input(cfg["input"]); obj = objective_from(cfg)
    d = isnothing(data) ? load_data(cfg["data_path"]) : data
    G = n_genes(d)
    sp = pretrain_split(d)
    Xtr = d.X[:, sp.train]
    nv = min(max_val, length(sp.val)); Xva = d.X[:, sp.val[1:nv]]

    # per-gene train stats (erecon target)
    mu_g = vec(mean(Xtr, dims=2)); sd_g = max.(vec(std(Xtr, dims=2)), 1f-6)
    # global stats of the ETF model input (tok_norm init)
    x_mean, x_sd = cfg["etf_scale"] == "raw" ? (Float64(mean(Xtr)), Float64(std(Xtr))) :
                   (xs = etf_scale(Xtr, d, cfg["etf_scale"]); (Float64(mean(xs)), Float64(std(xs))))
    ctx0 = (G = G, medians = d.medians, mu_g = mu_g, sd_g = sd_g, input = input)

    # model: encoder + objective heads, built right after the seed
    seed_all!(seed)
    enc = Encoder(input; n_genes=G, embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
                  n_layers=cfg["n_layers"], drop_prob=Float64(cfg["drop_prob"]), norm=cfg["norm"])
    scheme = Symbol(get(cfg, "init", string(init_scheme(obj))))
    init_encoder!(enc, scheme; seed, x_mean, x_sd)
    heads = make_heads(obj, cfg["embed_dim"], cfg["hidden_dim"], G)
    model = (encoder = enc, heads = heads)

    name = "$(cfg["objective"])_$(cfg["input"])"
    dir = run_dir(cfg, "pretrain", get(cfg, "run_name", name))
    extra = Dict{String,Any}("n_genes" => G, "init_scheme" => string(scheme), "x_mean" => x_mean, "x_sd" => x_sd,
                             "git" => git_hash(), "gpu" => gpu_name())
    save_checkpoint(joinpath(dir, "init"), enc, cfg; heads, extra)
    cn = channel_norms(enc, model_input(input, Xtr[:, 1:bs], d, cfg))
    @printf("pretrain %s | init %s: gene %.3f value %.3f mask %.3f | train %d val %d (eval on %d) | dir %s\n",
            name, scheme, cn.gene, cn.value, cn.mask, length(sp.train), length(sp.val), nv, dir)
    flush(stdout)

    model = to_device(model)
    teacher = uses_teacher(obj) ? Flux.testmode!(deepcopy(model.encoder)) : nothing
    opt = Flux.setup(OptimiserChain(ClipNorm(clip), Optimisers.AdamW(Float64(cfg["lr"]), (0.9, 0.999), wd)), model)
    warmup = max(1, div(total, 10))

    # fixed val masks and inputs; same masks for ETF and RTF
    hv = hide_mask(G, nv, obj.mask_ratio, MersenneTwister(1001))
    xin_va = model_input(input, Xva, d, cfg)
    zero_mse = obj isa ERecon ? Float64(mean(abs2, ((Xva .- mu_g) ./ sd_g)[hv])) : NaN
    ctx = merge(ctx0, (zero_mse = zero_mse,))
    triv = trivial_loss(obj, ctx)

    # val loss as the mean over hidden tokens, with per-gene / per-rank / per-sample sums (diag_*.jld2)
    R_va = gene_ranks(Xva, d.medians)
    function val_eval()
        Flux.testmode!(model)
        D = new_diag(G, nv)
        for s in 1:bs:nv
            c = s:min(s + bs - 1, nv)
            t = targets(obj, input, Xva[:, c], xin_va[:, c], hv[:, c], ctx)
            st = token_stats(obj, input, model, dev(xin_va[:, c]), hv[:, c], map(dev, t); teacher)
            add_diag!(D, st, t.idx, c, R_va, G)
        end
        Flux.trainmode!(model)
        return sum(D["gene_loss"]) / sum(D["gene_n"]), D
    end
    # per-eval scale checks: pooled sd of the downstream encoder, raw teacher target sd (lrecon)
    xin_sd = xin_va[:, 1:min(nv, 1000)]
    function scale_stats()
        Flux.testmode!(model)
        r = (pool_sd = pooled_sd(_ckpt_encoder(model, teacher), xin_sd, bs),
             tgt_raw_sd = isnothing(teacher) ? NaN : teacher_raw_sd(teacher, dev(xin_va[:, 1:min(nv, bs)])))
        Flux.trainmode!(model)
        return r
    end
    val_idx = sp.val[1:nv]

    println("model on $(gpu_name()) ($(CUDA.math_mode()) $(CUDA.math_precision())), first eval"); flush(stdout)
    log = pretrain_log(dir, cfg)
    crng = MersenneTwister(seed + 1000)          # hidden genes (shared by ETF and RTF)
    brng = MersenneTwister(seed + 3000)          # batch order (shared by ETF and RTF)
    perm = randperm(brng, size(Xtr, 2)); pos = 1
    best = Inf; step = 0; t0 = time(); Dv = nothing
    v0, D0 = val_eval()
    save_diag(joinpath(dir, "diag_init.jld2"), D0, 0, val_idx)
    log_eval!(log, 0, NaN, v0, triv, 0.0, 0.0; extra=merge((lr = 0.0,), scale_stats()))
    while step < total
        losses = Float64[]; gmax = 0.0
        for _ in 1:min(eval_every, total - step)
            pos + bs - 1 > length(perm) && (perm = randperm(brng, size(Xtr, 2)); pos = 1)
            cols = perm[pos:pos+bs-1]; pos += bs
            Xb = Xtr[:, cols]; xin = model_input(input, Xb, d, cfg)
            hide = hide_mask(G, bs, obj.mask_ratio, crng)
            t = map(dev, targets(obj, input, Xb, xin, hide, ctx))
            x = dev(xin)
            step += 1
            Optimisers.adjust!(opt, cosine_lr(step, total, Float64(cfg["lr"]), warmup))
            l, g = Flux.withgradient(m -> pretext_loss(obj, input, m, x, hide, t; teacher), model)
            isfinite(l) || error("non-finite loss at step $step")
            gmax = max(gmax, grad_norm(g[1]))
            Flux.update!(opt, model, g[1])
            isnothing(teacher) || ema_update!(teacher, model.encoder, obj.tau)
            push!(losses, l)
        end
        vl, Dv = val_eval()
        log_eval!(log, step, mean(losses), vl, triv, gmax, time() - t0; extra=merge((lr = cosine_lr(step, total, Float64(cfg["lr"]), warmup),), scale_stats()))
        if vl < best
            best = vl
            save_diag(joinpath(dir, "diag_best.jld2"), Dv, step, val_idx)
            save_checkpoint(joinpath(dir, "best"), _ckpt_encoder(model, teacher), cfg; heads=model.heads, extra=merge(extra, Dict("step" => step)))
        end
    end
    save_diag(joinpath(dir, "diag_final.jld2"), Dv, step, val_idx)
    save_checkpoint(joinpath(dir, "final"), _ckpt_encoder(model, teacher), cfg; heads=model.heads,
                    extra=merge(extra, Dict("step" => step, "s_per_step" => (time() - t0) / max(step, 1))))
    isnothing(teacher) || save_checkpoint(joinpath(dir, "final_student"), model.encoder, cfg; heads=model.heads, extra=extra)
    close_log!(log)
    @printf("done %s: best val %.4f (trivial %.4f), %.3f s/step, final ckpt %s\n", name, best, triv, (time() - t0) / max(step, 1), joinpath(dir, "final"))
    return joinpath(dir, "final")
end

# the encoder that downstream uses: the EMA teacher for objectives with a teacher, else the student
_ckpt_encoder(model, teacher) = isnothing(teacher) ? model.encoder : teacher
