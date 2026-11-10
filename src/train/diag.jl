# pretraining diagnostics on the fixed val set: per hidden token loss (and extras), summed per gene, per rank and per sample

# per hidden token stats (cpu vectors over t.idx); loss averages to pretext_loss
function token_stats(o::MLM, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, hide_what(o, input))
    lg = model.heads.cls(hidden_tokens(h, t.idx))
    oh = Flux.onehotbatch(t.y, 1:size(lg, 1))
    ls = logsoftmax(lg; dims=1)
    loss = -vec(sum(ls .* oh; dims=1))
    correct = vec(sum(lg .* oh; dims=1)) .>= vec(maximum(lg; dims=1))
    st = (loss = Array(loss), correct = Array(Float32.(correct)))
    # ETF classes are ranks: abs error of the expected rank
    if input isa ETF
        er = vec(sum(exp.(ls) .* Float32.(dev(collect(1:size(lg, 1)))); dims=1))
        st = merge(st, (rank_abserr = Array(abs.(er .- Float32.(t.y))),))
    end
    return st
end

function token_stats(o::ERecon, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, :value)
    pred = Array(vec(model.heads.reg(hidden_tokens(h, t.idx))))
    z = Array(t.z)
    return (loss = (pred .- z) .^ 2, pred = pred, z = z)
end

function token_stats(o::LRecon, input::Input, model, x, hide, t; teacher = nothing)
    Y = hidden_tokens(teacher_targets(teacher, x), t.idx)
    P = model.heads.dec(hidden_tokens(encode(model.encoder, x, hide, :value), t.idx))
    c = vec(sum(P .* Y; dims=1) ./ (sqrt.(sum(P .^ 2; dims=1)) .* sqrt.(sum(Y .^ 2; dims=1)) .+ 1f-6))
    return (loss = Array(1f0 .- c),)
end

# sums per gene, per rank (rank of the gene in the sample by x / median, both inputs) and per sample
function new_diag(G::Int, n::Int)
    z(k) = zeros(Float64, k)
    return Dict{String,Any}("gene_n" => z(G), "rank_n" => z(G), "sample_n" => z(n),
                            "gene_loss" => z(G), "rank_loss" => z(G), "sample_loss" => z(n))
end

function add_diag!(D::Dict, st::NamedTuple, idx::AbstractVector, cols::AbstractVector{<:Integer}, R::AbstractMatrix, G::Int)
    for (j, li) in enumerate(idx)
        g = mod1(li, G); s = cols[div(li - 1, G) + 1]; r = R[g, s]
        D["gene_n"][g] += 1; D["rank_n"][r] += 1; D["sample_n"][s] += 1
        D["gene_loss"][g] += st.loss[j]; D["rank_loss"][r] += st.loss[j]; D["sample_loss"][s] += st.loss[j]
        for k in keys(st)
            k in (:loss, :pred, :z) && continue
            v = st[k][j]
            get!(D, "gene_$k", zeros(G))[g] += v; get!(D, "rank_$k", zeros(G))[r] += v
        end
        # erecon: sums for per-gene R2 / correlation (trivial predict-zero = gene_z2)
        if haskey(st, :pred)
            p = st.pred[j]; y = st.z[j]
            get!(D, "gene_p", zeros(G))[g] += p; get!(D, "gene_p2", zeros(G))[g] += p^2
            get!(D, "gene_z", zeros(G))[g] += y; get!(D, "gene_z2", zeros(G))[g] += y^2; get!(D, "gene_pz", zeros(G))[g] += p * y
            get!(D, "rank_z2", zeros(G))[r] += y^2
        end
    end
    return D
end

save_diag(path::String, D::Dict, step::Int, val_idx) = jldsave(path; step, val_sample_idx=collect(val_idx), sums=D)

# median over features of the across-sample sd of the mean-pooled last layer (unmasked; caller sets testmode)
function pooled_sd(enc, xin, bs::Int)
    P = reduce(hcat, [Array(meanpool(encode(enc, dev(xin[:, c])))) for c in Iterators.partition(1:size(xin, 2), bs)])
    return Float64(median(vec(std(P; dims=2))))
end

# lrecon: median across-sample sd of the raw (unnormalised) layer-averaged teacher states
function teacher_raw_sd(teacher, x)
    hs = run_layers(teacher, tokens(teacher, x); states=true)
    H = Array(sum(hs) ./ Float32(length(hs)))
    return Float64(median(vec(std(H; dims=3))))
end
