# config, schedules and run directories shared by pretraining and finetuning

const DEFAULTS = Dict{String,Any}(
    "embed_dim" => 128, "hidden_dim" => 128, "n_heads" => 2, "n_layers" => 2, "norm" => "pre", "etf_scale" => "raw",
    "batch_size" => 100, "seed" => 1, "wandb_project" => "", "wandb_group" => "", "results_dir" => joinpath(PKG_DIR, "results"),
    "data_path" => joinpath(PKG_DIR, "data", "lincs.jld2"))

# config = defaults < toml file < overrides (key = value pairs)
function load_config(path::Union{Nothing,AbstractString} = nothing, overrides::AbstractDict = Dict{String,Any}(); kw...)
    cfg = copy(DEFAULTS)
    isnothing(path) || merge!(cfg, TOML.parsefile(path))
    for (k, v) in Iterators.flatten((overrides, kw))
        cfg[string(k)] = v
    end
    return cfg
end

# parse "key=value" command-line overrides (numbers and bools parsed, else strings)
function parse_overrides(args)
    out = Dict{String,Any}()
    for a in args
        k, v = split(a, "="; limit=2)
        out[k] = something(tryparse(Int, v), tryparse(Float64, v), v == "true" ? true : v == "false" ? false : String(v))
    end
    return out
end

# linear warmup then cosine to 0
cosine_lr(step::Int, total::Int, base::Float64, warmup::Int) = step <= warmup ? base * step / warmup :
    base * 0.5 * (1 + cos(pi * min(1.0, (step - warmup) / max(1, total - warmup))))

# global gradient norm
function grad_norm(g)
    s = Ref(0.0)
    fmap(x -> (x isa AbstractArray{<:Number} && (s[] += Float64(sum(abs2, x))); x), g; exclude = x -> x isa AbstractArray)
    return sqrt(s[])
end

seed_all!(seed::Int) = (Random.seed!(seed); CUDA.functional() && CUDA.seed!(seed); nothing)

# results/<stage>/<name>/<timestamp>_s<seed>_j<job>
function run_dir(cfg::Dict, stage::String, name::String)
    ts = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    job = get(ENV, "SLURM_JOB_ID", "local")
    d = joinpath(cfg["results_dir"], stage, name, "$(ts)_s$(cfg["seed"])_j$(job)")
    mkpath(d)
    return d
end

gpu_name() = CUDA.functional() ? CUDA.name(CUDA.device()) : "cpu"
git_hash() = (try readchomp(pipeline(`git -C $PKG_DIR rev-parse --short HEAD`; stderr=devnull)) catch; "nogit" end)

# columns of X in batches of bs, as device inputs
batches(f, X, bs) = (f(X[:, s:min(s + bs - 1, size(X, 2))]) for s in 1:bs:size(X, 2))

# TF32 matmuls (as the Nov 2026 fastobj runs); cfg tf32 = false keeps full Float32
set_math!(cfg::Dict) = (CUDA.functional() && get(cfg, "tf32", true) ? CUDA.math_mode!(CUDA.FAST_MATH; precision=:TensorFloat32) : CUDA.math_mode!(CUDA.DEFAULT_MATH); nothing)
