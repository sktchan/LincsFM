# CPU unit tests (no GPU): julia --threads=1,0 test/runtests.jl [nodata]  (nodata skips the tests on data/lincs.jld2)
ENV["CUDA_VISIBLE_DEVICES"] = ""
include(joinpath(@__DIR__, "..", "scripts", "setup.jl"))
using Test, Random, Statistics, LinearAlgebra, Flux
const M = LincsFM
const t_start = time()

# direct ranks: inverse of a stable descending sort of x / median
function direct_ranks(X, med)
    R = similar(X, Int)
    for j in axes(X, 2)
        p = sortperm(X[:, j] ./ med; rev=true, alg=MergeSort)
        R[p, j] = 1:size(X, 1)
    end
    return R
end

colnorms(A) = vec(sqrt.(sum(abs2, A; dims=1)))
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))

tiny_encoder(input; G=40, D=16, seed=1) = (Random.seed!(seed); M.Encoder(input; n_genes=G, embed_dim=D, hidden_dim=16, n_heads=2, n_layers=2, drop_prob=0.1))

@testset "LincsFM" begin

@testset "config" begin
    o = M.parse_overrides(["lr=0.001", "seed=3", "head_init=true", "input=rtf", "ckpt=/a/b=c"])
    @test o["lr"] === 0.001 && o["seed"] === 3 && o["head_init"] === true && o["input"] == "rtf" && o["ckpt"] == "/a/b=c"
    cfg = M.load_config(joinpath(M.PKG_DIR, "configs", "finetune", "scratch_rtf.toml"), o)
    @test cfg["lr"] == 0.001 && cfg["input"] == "rtf" && cfg["embed_dim"] == 128
end

@testset "gene_ranks and rank_features" begin
    rng = MersenneTwister(1)
    G, N = 60, 25
    X = rand(rng, Float32, G, N) .* 10; X[3, :] .= X[7, :]          # ties
    med = rand(rng, Float32, G) .+ 0.5f0; med[3] = med[7]
    R = M.gene_ranks(X, med)
    @test R == direct_ranks(X, med)
    @test all(sort(R[:, j]) == 1:G for j in 1:N)
    F = M.rank_features(R, X)
    @test all(0 .< F .<= 1)
    @test all(F[R .== 1] .== 1f0) && all(F[R .== G] .≈ 1f0 / G)
    @test eltype(M.encode_input(M.RTF(), X, med)) == Int32
    @test M.encode_input(M.ETF(), X, med) == X
end

@testset "hide_mask" begin
    for (G, B, r) in ((978, 50, 0.15), (978, 7, 0.3), (40, 10, 0.15))
        h = M.hide_mask(G, B, r, MersenneTwister(5))
        @test size(h) == (G, B)
        @test all(sum(h; dims=1) .== ceil(Int, G * r))
    end
    # the mask depends only on the rng, so ETF and RTF lanes at the same seed hide the same genes
    a = M.hide_mask(978, 20, 0.15, MersenneTwister(1001)); b = M.hide_mask(978, 20, 0.15, MersenneTwister(1001))
    @test a == b
    @test a != M.hide_mask(978, 20, 0.15, MersenneTwister(1002))
end

@testset "tokens with hidden genes" begin
    G, B = 40, 6
    X = rand(MersenneTwister(2), Float32, G, B) .* 8; med = fill(4f0, G)
    hide = M.hide_mask(G, B, 0.3, MersenneTwister(3)); vis = .!hide
    for input in (M.ETF(), M.RTF())
        e = tiny_encoder(input; G)
        x = M.encode_input(input, X, med)
        t0 = M.tokens(e, x); vc = M.value_channel(e, x); gc = M.gene_channel(e, x)
        D = M.embed_dim(e)
        for what in (:value, :gene)
            t = M.tokens(e, x, hide, what)
            @test size(t) == (D, G, B)
            for j in 1:B, g in 1:G
                if vis[g, j]
                    @test t[:, g, j] ≈ t0[:, g, j]
                elseif what == :value
                    @test t[:, g, j] ≈ e.mask .+ gc[:, g]
                else
                    @test t[:, g, j] ≈ vc[:, g, j] .+ e.mask
                end
            end
        end
    end
end

@testset "init: tok_norm and rank_init_lin" begin
    G = 978
    for input in (M.ETF(), M.RTF())
        e = tiny_encoder(input; G, D=128)
        w0 = input isa M.ETF ? (copy(vec(e.value.weight)), copy(e.value.bias)) : nothing
        M.init_encoder!(e, :tok_norm; seed=1, x_mean=5.0, x_sd=2.0)
        @test isapprox(mean(colnorms(e.gene.weight)), 1; atol=0.05)
        @test isapprox(norm(e.mask), 1; atol=1e-5)
        if input isa M.ETF
            # value projection acts on (x - x_mean) / x_sd
            x = Float32[5 7; 9 3]
            @test M.value_channel(e, x) ≈ w0[1] .* reshape((x .- 5) ./ 2, 1, 2, 2) .+ w0[2] rtol=1e-4
        else
            W = e.value.weight
            c = [cosine(W[:, r], W[:, r+1]) for r in 1:G-1]
            @test isapprox(mean(c), 0.998; atol=1e-3)
            @test count(<(0), c) == 1                         # one sign change at the middle rank
            @test norm(W[:, 1]) > norm(W[:, G ÷ 2])           # magnitude grows away from the middle
        end
    end
    # same seed -> same mask vector for ETF and RTF
    a = M.init_mask!(tiny_encoder(M.ETF(); D=32); seed=4); b = M.init_mask!(tiny_encoder(M.RTF(); D=32); seed=4)
    @test a.mask == b.mask
end

@testset "fast attention on cpu = Flux reference" begin
    e = tiny_encoder(M.ETF()); Flux.testmode!(e)
    x = rand(MersenneTwister(4), Float32, 40, 5)
    M.FastAttn.USE_FAST_ATTN[] = true;  h1 = M.encode(e, x)
    M.FastAttn.USE_FAST_ATTN[] = false; h2 = M.encode(e, x)
    M.FastAttn.USE_FAST_ATTN[] = true
    @test h1 ≈ h2 rtol=1e-4
end

@testset "pretext losses on a tiny encoder" begin
    G, B = 40, 8
    rng = MersenneTwister(6)
    X = rand(rng, Float32, G, B) .* 8; med = vec(median(X; dims=2))
    mu_g = vec(mean(X; dims=2)); sd_g = vec(std(X; dims=2))
    for (name, obj) in (("mlm", M.MLM(0.15)), ("erecon", M.ERecon(0.3)), ("lrecon", M.LRecon(0.15, 0.99f0))), input in (M.ETF(), M.RTF())
        e = tiny_encoder(input; G)
        M.init_encoder!(e, M.init_scheme(obj); seed=1, x_mean=mean(X), x_sd=std(X))
        model = (encoder = e, heads = M.make_heads(obj, 16, 16, G))
        teacher = M.uses_teacher(obj) ? Flux.testmode!(deepcopy(e)) : nothing
        hide = M.hide_mask(G, B, obj.mask_ratio, MersenneTwister(1))
        xin = M.encode_input(input, X, med)
        ctx = (G = G, medians = med, mu_g = mu_g, sd_g = sd_g, input = input, zero_mse = 1.0)
        t = M.targets(obj, input, X, xin, hide, ctx)
        @test length(t.idx) == count(hide)
        l, g = Flux.withgradient(m -> M.pretext_loss(obj, input, m, xin, hide, t; teacher), model)
        @test isfinite(l)
        @test isfinite(M.grad_norm(g[1])) && M.grad_norm(g[1]) > 0
        @test isfinite(M.trivial_loss(obj, ctx))
        if obj isa M.MLM
            @test all(1 .<= t.y .<= G)
            input isa M.RTF && @test t.y == Int32.(mod1.(findall(vec(hide)), G))
        end
        if obj isa M.LRecon
            before = deepcopy(Flux.trainables(teacher)[1])
            M.ema_update!(teacher, e, 0.5f0)
            @test Flux.trainables(teacher)[1] ≈ before   # student == teacher copy, so EMA leaves it unchanged
        end
    end
end

@testset "norm variants" begin
    G, B = 40, 4
    X = rand(MersenneTwister(3), Float32, G, B) .* 8; med = fill(4f0, G)
    for input in (M.ETF(), M.RTF()), norm in ("pre", "pre_final", "post")
        Random.seed!(1)
        e = M.Encoder(input; n_genes=G, embed_dim=16, hidden_dim=16, n_heads=2, n_layers=2, drop_prob=0.1, norm)
        Flux.testmode!(e)
        x = M.encode_input(input, X, med)
        h = M.encode(e, x); hs = M.run_layers(e, M.tokens(e, x); states=true)
        @test size(h) == (16, G, B) && hs[end] ≈ h && length(hs) == 3
        @test isnothing(e.final) == (norm != "pre_final") && all(l.post == (norm == "post") for l in e.layers.layers)
        # normed outputs: per-token mean ~0, sd ~1 (LayerNorm at init has unit scale, zero bias)
        norm == "pre" || @test maximum(abs, mean(h; dims=1)) < 1e-4
        # checkpoint round trip keeps the arch
        dir = mktempdir()
        cfg = M.load_config(nothing, Dict{String,Any}("input" => M.input_name(input), "objective" => "mlm", "drop_prob" => 0.1,
                            "embed_dim" => 16, "hidden_dim" => 16, "norm" => norm))
        M.save_checkpoint(joinpath(dir, "final"), e, cfg; extra=Dict{String,Any}("n_genes" => G, "init_scheme" => "default", "x_mean" => 0.0, "x_sd" => 1.0))
        e2, c2, _ = M.load_encoder(joinpath(dir, "final")); Flux.testmode!(e2)
        @test c2["norm"] == norm && M.encode(e2, x) ≈ h
    end
end

@testset "classifier forward" begin
    G, B, C = 40, 5, 7
    X = rand(MersenneTwister(7), Float32, G, B) .* 8; med = fill(4f0, G)
    for input in (M.ETF(), M.RTF()), pool in (:mean, :attn)
        m = M.Classifier(tiny_encoder(input; G), C; hidden_dim=16, pool)
        x = M.encode_input(input, X, med)
        y = m(x)
        @test size(y) == (C, B) && all(isfinite, y)
    end
end

@testset "checkpoint round trip" begin
    dir = mktempdir()
    cfg = M.load_config(joinpath(M.PKG_DIR, "configs", "pretrain", "erecon_rtf.toml"), Dict{String,Any}("embed_dim" => 16, "hidden_dim" => 16))
    e = tiny_encoder(M.RTF(); G=40, D=16)
    M.init_encoder!(e, :tok_norm; seed=1)
    heads = M.make_heads(M.ERecon(0.3), 16, 16, 40)
    extra = Dict{String,Any}("n_genes" => 40, "init_scheme" => "tok_norm", "x_mean" => 0.0, "x_sd" => 1.0)
    M.save_checkpoint(joinpath(dir, "final"), e, cfg; heads, extra)
    @test isfile(joinpath(dir, "final", "config.toml"))
    e2, cfg2, ex2 = M.load_encoder(joinpath(dir, "final"))
    @test cfg2["input"] == "rtf" && ex2["n_genes"] == 40
    @test all(a == b for (a, b) in zip(Flux.trainables(e), Flux.trainables(e2)))
    x = rand(Int32(1):Int32(40), 40, 3)
    Flux.testmode!(e); Flux.testmode!(e2)
    @test M.encode(e, x) ≈ M.encode(e2, x)
    # random mode: same architecture and init scheme, different weights, reproducible per init_seed
    r1, _, _ = M.load_encoder(joinpath(dir, "final"); random=true, init_seed=0)
    r2, _, _ = M.load_encoder(joinpath(dir, "final"); random=true, init_seed=0)
    @test all(a == b for (a, b) in zip(Flux.trainables(r1), Flux.trainables(r2)))
    @test r1.gene.weight != e.gene.weight
    @test size(r1.value.weight) == size(e.value.weight)
    @test isapprox(mean(colnorms(r1.gene.weight)), 1; atol=0.1)
end

@testset "pretrain diagnostics" begin
    G, D, B = 40, 16, 6
    X = rand(Float32, G, B) .* 5; med = vec(median(X; dims=2)); R = M.gene_ranks(X, med)
    ctx = (G = G, medians = med, mu_g = vec(mean(X; dims=2)), sd_g = vec(std(X; dims=2)), input = M.ETF(), zero_mse = 1.0)
    for input in (M.ETF(), M.RTF()), obj in (M.MLM(0.15), M.ERecon(0.3), M.LRecon(0.15, 0.99f0))
        e = tiny_encoder(input; G, D)
        model = (encoder = e, heads = M.make_heads(obj, D, 16, G))
        teacher = M.uses_teacher(obj) ? Flux.testmode!(deepcopy(e)) : nothing
        Flux.testmode!(model)
        xin = M.encode_input(input, X, med)
        hide = M.hide_mask(G, B, obj.mask_ratio, MersenneTwister(5))
        t = M.targets(obj, input, X, xin, hide, ctx)
        l = M.pretext_loss(obj, input, model, xin, hide, t; teacher)
        st = M.token_stats(obj, input, model, xin, hide, t; teacher)
        @test length(st.loss) == length(t.idx)
        @test isapprox(mean(st.loss), l; rtol=1e-4)
        Dg = M.add_diag!(M.new_diag(G, B), st, t.idx, 1:B, R, G)
        @test sum(Dg["gene_n"]) == sum(Dg["rank_n"]) == sum(Dg["sample_n"]) == count(hide)
        @test sum(Dg["gene_loss"]) ≈ sum(Dg["rank_loss"]) && sum(Dg["sample_loss"]) ≈ sum(st.loss) && sum(Dg["gene_loss"]) ≈ sum(st.loss)
        @test Dg["gene_n"] == vec(sum(hide; dims=2))
        obj isa M.MLM && @test all(0 .<= Dg["gene_correct"] .<= Dg["gene_n"])
        obj isa M.MLM && input isa M.ETF && @test haskey(Dg, "rank_rank_abserr")
        obj isa M.ERecon && @test isapprox(sum(Dg["gene_z2"]), sum(abs2, t.z); rtol=1e-4)
    end
end

@testset "classifier eval extras" begin
    G, C, B = 30, 12, 25
    e = tiny_encoder(M.ETF(); G, D=16)
    m = M.Classifier(e, C; hidden_dim=16)
    X = randn(Float32, G, B); y = rand(1:C, B)
    l, a, p, ex = M.evaluate_classifier(m, X, y, C, 7; full=true)
    @test size(ex.top10) == (10, B) && all(ex.top10[1, :] .== p)
    @test all(0 .< ex.p_true .<= 1) && all(ex.top10_p[1, :] .>= ex.top10_p[2, :])
    @test mean(Float32.(ex.p_true)) > 0 && a == mean(p .== y)
end

if !("nodata" in ARGS)
    @testset "real data: labels and splits" begin
        d = M.load_data()
        G, N = size(d.X)
        @test G == 978 && length(d.medians) == G
        tr = M.pretrain_split(d).train; gs = M.gene_stats(d.X[1:3, :], tr)
        @test gs.medians ≈ d.medians[1:3] && gs.mean ≈ d.gene_mean[1:3] && gs.sd ≈ d.gene_sd[1:3]
        idx, y, names = M.labels(d, "lvl2")
        sp = M.finetune_split(d, idx)
        @test length(names) == 322
        @test (length(sp.train), length(sp.val), length(sp.test)) == (245130, 30904, 30730)
        @test isempty(intersect(sp.train, sp.val)) && isempty(intersect(sp.train, sp.test)) && isempty(intersect(sp.val, sp.test))
        @test extrema(y) == (1, 322)
        for (sc, per_gene) in (("gene", true), ("global", false))
            Xtr, _, _ = M.finetune_inputs(Dict{String,Any}("input" => "etf", "etf_scale" => sc), d, idx, sp)
            mg = vec(mean(Xtr; dims=2))
            @test abs(mean(Xtr)) < 0.1 && abs(std(Xtr) - 1) < 0.1 && (maximum(abs, mg) < 0.1) == per_gene
        end
        Xr, _, _ = M.finetune_inputs(Dict{String,Any}("input" => "etf", "etf_scale" => "raw"), d, idx, sp)
        @test Xr == d.X[:, idx[sp.train]]
        bad = Set(d.bad_plates)
        @test !any(p -> p in bad, d.meta.det_plate[idx])
        @test all(d.meta.pert_type[idx] .== "trt_cp")
        idx1, y1, names1 = M.labels(d, "lvl1")
        @test length(names1) == 236   # cell_iname, bad plate dropped
        sp1 = M.finetune_split(d, idx1)
        @test !any(p -> p in bad, d.meta.det_plate[idx1])
        @test length(idx1) == N - count(p -> p in bad, d.meta.det_plate)
        ps = M.pretrain_split(d)
        @test !any(p -> p in bad, d.meta.det_plate[ps.train])
        println("lvl1: $(length(names1)) classes, train $(length(sp1.train)) val $(length(sp1.val)) test $(length(sp1.test))")
        cols = rand(MersenneTwister(8), 1:N, 50)
        @test M.gene_ranks(d.X[:, cols], d.medians) == direct_ranks(d.X[:, cols], d.medians)
    end
end

end
println("tests done in $(round(time() - t_start, digits=1)) s")
