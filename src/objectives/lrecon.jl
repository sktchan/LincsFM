# lrecon: hide values of a fraction of genes; predict an EMA teacher's normalized, layer-averaged states of those tokens (cosine loss)

struct LRecon <: Objective
    mask_ratio::Float64
    tau::Float32
end

uses_teacher(::LRecon) = true
make_heads(::LRecon, D::Int, H::Int, G::Int) = (dec = latent_head(D, H),)

function targets(::LRecon, ::Input, Xraw, xin, hide::AbstractMatrix{Bool}, ctx)
    return (idx = findall(vec(hide)),)
end

# per-sample token-mean removal, then batch z per (feature, gene)
function _dbl(h)
    t = h .- mean(h, dims=2)
    return (t .- mean(t, dims=3)) ./ (std(t, dims=3) .+ 1f-6)
end

# layer-averaged normalised teacher states (D x G x B)
teacher_targets(teacher::Encoder, x) = (hs = run_layers(teacher, tokens(teacher, x); states=true); sum(_dbl, hs) ./ Float32(length(hs)))

_cos_loss(a, b) = mean(1f0 .- sum(a .* b, dims=1) ./ (sqrt.(sum(a .^ 2, dims=1)) .* sqrt.(sum(b .^ 2, dims=1)) .+ 1f-6))

function pretext_loss(o::LRecon, input::Input, model, x, hide, t; teacher = nothing)
    Y = Flux.ChainRulesCore.ignore_derivatives(() -> hidden_tokens(teacher_targets(teacher, x), t.idx))
    h = encode(model.encoder, x, hide, :value)
    return _cos_loss(model.heads.dec(hidden_tokens(h, t.idx)), Y)
end

trivial_loss(::LRecon, ctx) = 1.0   # cosine loss of an unrelated direction

# teacher <- tau * teacher + (1 - tau) * student
function ema_update!(teacher::Encoder, student::Encoder, tau::Float32)
    for (pt, ps) in zip(Flux.trainables(teacher), Flux.trainables(student))
        @. pt = tau * pt + (1f0 - tau) * ps
    end
    return teacher
end
