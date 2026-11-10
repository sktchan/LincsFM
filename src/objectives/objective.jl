# pretraining objectives: each defines its corruption, target, head and loss; one training loop (train/pretrain.jl)

abstract type Objective end

# defaults for every objective (overridden by the objective or the config)
init_scheme(::Objective) = :tok_norm
hide_what(::Objective, ::Input) = :value
uses_teacher(::Objective) = false

# hidden genes: k = ceil(G * ratio) per column, one randperm per column; true = hidden
function hide_mask(G::Int, B::Int, ratio::Real, rng::AbstractRNG)
    k = ceil(Int, G * ratio); m = falses(G, B)
    for b in 1:B
        @inbounds for g in view(randperm(rng, G), 1:k)
            m[g, b] = true
        end
    end
    return m
end

# gather the hidden tokens: D x (number hidden)
hidden_tokens(h, idx) = reshape(h, size(h, 1), :)[:, idx]

function objective_from(cfg::Dict)
    name = cfg["objective"]
    r = Float64(cfg["mask_ratio"])
    name == "mlm" && return MLM(r)
    name == "erecon" && return ERecon(r)
    name == "lrecon" && return LRecon(r, Float32(cfg["ema_decay"]))
    error("unknown objective $name (mlm, erecon, lrecon)")
end
