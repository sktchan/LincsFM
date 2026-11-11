# erecon: hide the value of a fraction of genes and predict their per-gene z-scored expression (same masks and targets for ETF and RTF)

struct ERecon <: Objective
    mask_ratio::Float64
end

make_heads(::ERecon, D::Int, H::Int, G::Int) = (reg = reg_head(D, H),)

# z target of each hidden gene (cpu); ctx.mu_g / ctx.sd_g = train mean / sd per gene
function targets(::ERecon, ::Input, Xraw, xin, hide::AbstractMatrix{Bool}, ctx)
    idx = findall(vec(hide))
    return (idx = idx, z = Float32.(vec((Xraw .- ctx.mu_g) ./ ctx.sd_g)[idx]))
end

function pretext_loss(o::ERecon, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, :value)
    pred = vec(model.heads.reg(hidden_tokens(h, t.idx)))
    return mean((pred .- t.z) .^ 2)
end

trivial_loss(::ERecon, ctx) = ctx.zero_mse   # predict-zero on the val masks
