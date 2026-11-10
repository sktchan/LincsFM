# inputs: ETF = expression values, RTF = rank of each gene by x / gene median (rank 1 = highest); both kept in gene order

abstract type Input end
struct ETF <: Input end
struct RTF <: Input end

parse_input(s::AbstractString) = s == "etf" ? ETF() : s == "rtf" ? RTF() : error("input must be etf or rtf, got $s")
input_name(::ETF) = "etf"
input_name(::RTF) = "rtf"

# rank of gene g in sample j (gene order); ties keep gene order, as sortperm in the original rank_genes
function gene_ranks(X::AbstractMatrix{<:Real}, medians::AbstractVector{<:Real})
    G, N = size(X)
    R = Matrix{Int32}(undef, G, N)
    v = Vector{Float32}(undef, G); p = Vector{Int}(undef, G)
    @inbounds for j in 1:N
        @. v = X[:, j] / medians
        sortperm!(p, v, rev=true)
        for r in 1:G
            R[p[r], j] = Int32(r)
        end
    end
    return R
end

# model input for one set of columns
encode_input(::ETF, X, medians) = Matrix{Float32}(X)
encode_input(::RTF, X, medians) = gene_ranks(X, medians)

# model input of raw columns X: ETF scaled by cfg etf_scale, RTF gene ranks
model_input(input::Input, X, d, cfg::AbstractDict) = input isa RTF ? gene_ranks(X, d.medians) : etf_scale(X, d, get(cfg, "etf_scale", "raw"))

# rank baseline features as the original rmlp/rlog: (n_det + 1 - r) / G for detected genes (x > 0), 0 for undetected
function rank_features(R::AbstractMatrix{<:Integer}, X::AbstractMatrix{<:Real})
    G = size(R, 1); nd = vec(sum(X .> 0, dims=1))
    return Float32.(ifelse.((R .<= nd') .& (X .> 0), (nd' .+ 1 .- R) ./ G, 0))
end
