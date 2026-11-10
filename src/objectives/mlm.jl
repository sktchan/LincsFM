# mlm (as in the original runs): ETF hides the value and predicts the gene's rank (G classes);
# RTF hides the gene and keeps its rank, and predicts which gene it is (G classes)

struct MLM <: Objective
    mask_ratio::Float64
end

init_scheme(::MLM) = :default
hide_what(::MLM, ::RTF) = :gene
make_heads(::MLM, D::Int, H::Int, G::Int) = (cls = class_head(D, G),)

# class of each hidden token (cpu): ETF = rank of the gene, RTF = gene id
function targets(::MLM, input::Input, Xraw, xin, hide::AbstractMatrix{Bool}, ctx)
    idx = findall(vec(hide))
    y = input isa ETF ? vec(gene_ranks(Xraw, ctx.medians))[idx] : mod1.(idx, size(hide, 1))
    return (idx = idx, y = Int32.(y))
end

function pretext_loss(o::MLM, input::Input, model, x, hide, t; teacher = nothing)
    h = encode(model.encoder, x, hide, hide_what(o, input))
    logits = model.heads.cls(hidden_tokens(h, t.idx))
    return Flux.logitcrossentropy(logits, Flux.onehotbatch(t.y, 1:size(logits, 1)))
end

trivial_loss(::MLM, ctx) = log(Float64(ctx.G))   # uniform over G classes
