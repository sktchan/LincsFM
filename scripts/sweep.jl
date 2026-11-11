# writes a wandb sweep yaml from configs/sweeps/<stage>.yaml and creates the sweep
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM, Printf

name = ARGS[1]
fixed = Dict{String,String}(String.(split(a, "="; limit=2)) for a in ARGS[2:end])
dry = lowercase(pop!(fixed, "dry", "false")) == "true"

stage, arm = String.(split(name, '_'; limit=2))
level = stage == "pt" ? "" : String(split(arm, '_')[end])
config = stage == "pt" ? "configs/pretrain/$arm.toml" :
         stage == "ft" ? "configs/finetune/$(chopsuffix(arm, "_" * level)).toml" :
         stage == "bl" ? "configs/baselines/$arm.toml" : error("sweep name must start with pt_, ft_ or bl_")
stage == "pt" || level in ("lvl1", "lvl2") || error("sweep name must end in _lvl1 or _lvl2")
stage == "bl" && !occursin("mlp", arm) && !occursin("log", arm) && error("bl sweeps are for the mlp and logistic baselines")
isfile(joinpath(LincsFM.PKG_DIR, config)) || error("no config $config")

# pretrained / random finetunes default to the erecon checkpoint of the same input
ftarm = chopsuffix(arm, "_" * level)
stage == "ft" && !startswith(ftarm, "scratch_") && get!(fixed, "ckpt", "@erecon_" * split(ftarm, '_')[end])

lr = LincsFM.read_toml(joinpath(LincsFM.PKG_DIR, config))["lr"]
vals = Dict("NAME" => name, "CONFIG" => config, "LEVEL" => level,
            "MAX_STEPS" => level == "lvl1" ? "10000" : "40000", "EVAL_EVERY" => level == "lvl1" ? "1000" : "500",
            "LR_GRID" => @sprintf("[%.3g, %.3g, %.3g]", lr / 3, lr, lr * 3),
            "FIXED" => join(["  $k:\n    value: \"$v\"\n" for (k, v) in sort(collect(fixed))]))
template = stage == "bl" && occursin("log", arm) ? "bl_log" : stage
yaml = replace(read(joinpath(LincsFM.PKG_DIR, "configs", "sweeps", "$template.yaml"), String), ("\${$k}" => v for (k, v) in vals)...)
out = joinpath(LincsFM.PKG_DIR, "logs", "sweeps", "$name.yaml")
mkpath(dirname(out)); write(out, yaml)
println(out)
dry || cd(() -> run(`wandb sweep $out`), LincsFM.PKG_DIR)
