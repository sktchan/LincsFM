# finetune classifier: encoder (pretraining tokens) + pool + head

struct Classifier{E<:Encoder,P,H}
    encoder::E
    pool::P      # AttnPool, ClsToken, MaxPool, or nothing = mean pool
    head::H
end
Flux.@layer Classifier

function Classifier(enc::Encoder, n_classes::Int; hidden_dim::Int = 128, drop_prob::Float64 = 0.18, pool::Symbol = :mean)
    D = embed_dim(enc)
    pl = pool == :attn ? AttnPool(D) : pool == :cls ? ClsToken(D) : pool == :max ? MaxPool() : pool == :mean ? nothing : error("pool must be mean, attn, cls or max")
    return Classifier(enc, pl, ft_head(D, hidden_dim, n_classes, drop_prob))
end

function features(m::Classifier{<:Encoder,<:ClsToken}, x)
    t = tokens(m.encoder, x)
    c = repeat(reshape(m.pool.vec, :, 1, 1), 1, 1, size(t, 3))
    return run_layers(m.encoder, hcat(c, t))[:, 1, :]
end
function features(m::Classifier, x)
    h = encode(m.encoder, x)
    return isnothing(m.pool) ? meanpool(h) : m.pool(h)
end
(m::Classifier)(x) = m.head(features(m, x))

# head_init: rescale the first head layer so it sees z-scored pooled features at step 0
function head_init!(m::Classifier, batches)
    Flux.testmode!(m)
    E = reduce(hcat, [Array(features(m, x)) for x in batches])
    Flux.trainmode!(m)
    mu = mean(E, dims=2); sd = std(E, dims=2); sd[sd .< 1f-6] .= 1f0
    d1 = m.head.layers[1]
    Wn = Array(d1.weight) ./ permutedims(sd)
    copyto!(d1.weight, Wn); copyto!(d1.bias, Array(d1.bias) .- Wn * vec(mu))
    return median(vec(sd))
end
