# supervised training loop shared by finetune() and baseline(): sampling, periodic val eval, best-val tracking, test eval

# mean loss, accuracy and predictions over all columns of Xs; full = also the true-class probability and the top-10 classes / probabilities
function evaluate_classifier(m, Xs, ys, C::Int, bs::Int; full::Bool = false)
    Flux.testmode!(m); loss = 0.0; preds = Int[]
    k = min(10, C); n = size(Xs, 2)
    p_true = full ? zeros(Float32, n) : nothing
    topk = full ? zeros(Int16, k, n) : nothing; topk_p = full ? zeros(Float16, k, n) : nothing
    for s in 1:bs:n
        c = s:min(s + bs - 1, n)
        lg = m(dev(Xs[:, c]))
        loss += Float64(Flux.logitcrossentropy(lg, dev(Flux.onehotbatch(ys[c], 1:C)))) * length(c)
        L = Array(lg)
        append!(preds, Flux.onecold(L))
        if full
            P = softmax(L; dims=1)
            for (j, i) in enumerate(c)
                p_true[i] = P[ys[i], j]
                o = partialsortperm(view(P, :, j), 1:k; rev=true)
                topk[:, i] .= o; topk_p[:, i] .= P[o, j]
            end
        end
    end
    Flux.trainmode!(m)
    return loss / n, mean(preds .== ys), preds, (p_true = p_true, top10 = topk, top10_p = topk_p)
end

# grad(model, x, y) -> (loss, gradient); lvl2 samples class-balanced (uniform class, then uniform sample), lvl1 walks shuffled epochs
# eval_every (steps, default one epoch); best = highest val acc, ties by lower val loss; always runs max_steps
function train_classifier!(model, opt, grad, data, cfg::Dict, C::Int)
    Xtr, ytr, Xva, yva, Xte, yte = data
    bs = cfg["batch_size"]; total = cfg["max_steps"]; level = cfg["level"]
    bpe = div(length(ytr), bs); eval_every = get(cfg, "eval_every", bpe)
    cidx = Dict{Int,Vector{Int}}(); for (i, c) in enumerate(ytr); push!(get!(cidx, c, Int[]), i); end
    classes = collect(keys(cidx))
    perm = randperm(length(ytr)); pos = 1
    function next_batch()
        level == "lvl2" && return [rand(cidx[rand(classes)]) for _ in 1:bs]
        pos + bs - 1 > length(perm) && (perm = randperm(length(ytr)); pos = 1)
        b = perm[pos:pos+bs-1]; pos += bs
        return b
    end

    curve = NamedTuple[]; best = (acc = -Inf, loss = Inf, step = 0); best_state = nothing
    step = 0; losses = Float64[]; t0 = time()
    while step < total
        b = next_batch()
        l, g = grad(model, dev(Xtr[:, b]), dev(Flux.onehotbatch(ytr[b], 1:C)))
        Flux.update!(opt, model, g)
        push!(losses, l); step += 1
        (step % eval_every == 0 || step == total) || continue
        vl, va, _ = evaluate_classifier(model, Xva, yva, C, bs)
        ep = step / bpe
        push!(curve, (step = step, epoch = ep, train = mean(losses), val_loss = vl, val_acc = va))
        @printf("step %d (epoch %.2f) train %.4f val %.4f val_acc %.4f | %.0f s\n", step, ep, mean(losses), vl, va, time() - t0); flush(stdout)
        log_wandb(cfg, Dict("step" => step, "epoch" => ep, "train_loss" => mean(losses), "val_loss" => vl, "val_acc" => va, "best_val_acc" => max(best.acc, va)))
        losses = Float64[]
        if va > best.acc || (va == best.acc && vl < best.loss)
            best = (acc = va, loss = vl, step = step); best_state = Flux.state(cpu(model))
        end
    end
    wall = time() - t0
    _, acc_final, p_final, e_final = evaluate_classifier(model, Xte, yte, C, bs; full=true)
    final_state = Flux.state(cpu(model))
    Flux.loadmodel!(model, best_state)
    _, acc_best, p_best, e_best = evaluate_classifier(model, Xte, yte, C, bs; full=true)
    return (curve = curve, steps = step, wall = wall, best = best, final_state = final_state, best_state = best_state,
            acc_final = acc_final, p_final = p_final, acc_best = acc_best, p_best = p_best, e_final = e_final, e_best = e_best)
end

# save model states, test predictions, curve, config and the results row; returns the row
function save_classifier_run(dir::String, cfg::Dict, r, idx, sp, yte, names, row_head::NamedTuple, hsd::Real)
    get(cfg, "save_model", true) && jldsave(joinpath(dir, "model.jld2"); final=r.final_state, best=r.best_state, config=cfg)
    jldsave(joinpath(dir, "predstrues.jld2"); preds_final=r.p_final, preds_best=r.p_best, trues=yte,
            ptrue_final=r.e_final.p_true, ptrue_best=r.e_best.p_true, top10_final=r.e_final.top10, top10_best=r.e_best.top10,
            top10p_final=r.e_final.top10_p, top10p_best=r.e_best.top10_p,
            test_sample_idx=idx[sp.test], label_names=names)
    write_curve(joinpath(dir, "curve.tsv"), r.curve)
    open(io -> TOML.print(io, Dict(k => v for (k, v) in cfg if v isa Union{Number,String,Bool})), joinpath(dir, "config.toml"), "w")
    row = merge(row_head, (level = cfg["level"], seed = cfg["seed"], steps = r.steps, final_test = r.acc_final, best_test = r.acc_best,
           best_step = r.best.step, best_val_acc = r.best.acc, head_init_sd = hsd, gpu = gpu_name(),
           job = get(ENV, "SLURM_JOB_ID", "local"), wall_s = round(Int, r.wall), git = git_hash(), dir = dir))
    append_row(joinpath(cfg["results_dir"], "finetune.tsv"), row)
    log_wandb_summary(cfg, Dict("final_test" => r.acc_final, "best_test" => r.acc_best, "best_step" => r.best.step))
    @printf("test final %.4f best %.4f (step %d) | %.0f s\n", r.acc_final, r.acc_best, r.best.step, r.wall)
    return row
end
