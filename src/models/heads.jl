# heads and pooling

# pretext heads (as the original pretrain models)
class_head(d::Int, n::Int) = Flux.Chain(Flux.Dense(d => d, gelu), Flux.LayerNorm(d), Flux.Dense(d => n))
reg_head(d::Int, h::Int) = Flux.Chain(Flux.Dense(d => h, gelu), Flux.LayerNorm(h), Flux.Dense(h => 1))
latent_head(d::Int, h::Int) = Flux.Chain(Flux.Dense(d => h, gelu), Flux.LayerNorm(h), Flux.Dense(h => d))

# finetune classification head
ft_head(d::Int, h::Int, n::Int, p::Float64) = Flux.Chain(Flux.Dense(d => h, gelu), Flux.Dropout(p), Flux.Dense(h => n))

# attention pooling: heads = chunks of the embedding, softmax over tokens of score(LN(h)); zero init = mean pool
struct AttnPool{N,D}
    norm::N
    score::D
end
Flux.@layer AttnPool
AttnPool(d::Int; heads::Int = 4) = AttnPool(Flux.LayerNorm(d), Flux.Dense(zeros(Float32, heads, d), zeros(Float32, heads)))
function (p::AttnPool)(h)
    d, L, B = size(h); H = size(p.score.weight, 1)
    a = NNlib.softmax(reshape(p.score(reshape(p.norm(h), d, :)), H, L, B); dims=2)
    return reshape(sum(reshape(h, d ÷ H, H, L, B) .* reshape(a, 1, H, L, B); dims=3), d, B)
end

meanpool(h) = dropdims(mean(h, dims=2), dims=2)

# max pooling over tokens (BulkFormer, scFoundation)
struct MaxPool end
(::MaxPool)(h) = dropdims(maximum(h, dims=2), dims=2)

# CLS readout: a learned token prepended to the gene tokens; its encoder output is the sample vector
struct ClsToken{V<:AbstractVector}
    vec::V
end
Flux.@layer ClsToken
ClsToken(d::Int; seed::Int = 1) = ClsToken(randn(MersenneTwister(seed + 3001), Float32, d) .* 0.02f0)
