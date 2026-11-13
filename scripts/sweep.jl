# writes a wandb sweep yaml from configs/sweeps/<stage>.yaml and creates the sweep
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM, Printf

name = ARGS[1]
fixed = Dict{String,String}(String.(split(a, "="; limit=2)) for a in ARGS[2:end])
dry = lowercase(pop!(fixed, "dry", "false")) == "true"
# W&B project for the sweep and its trials (finals go to lincs-lvl1-runs / lincs-lvl2-runs, tests to lincs-test)
project = pop!(fixed, "project", "lincs-lvl2-sweeps")

stage, arm = String.(split(name, '_'; limit=2))
level = stage == "pt" ? "" : String(split(arm, '_')[end])
# ft names may end in the pretraining objective: ft_pretrained_etf_mlm_lvl2 -> configs/finetune/pretrained_etf.toml, ckpt=@mlm_etf
ftparts = stage == "ft" ? split(chopsuffix(arm, "_" * level), '_') : String[]
objective = length(ftparts) == 3 && ftparts[3] in ("mlm", "erecon", "lrecon") ? String(ftparts[3]) : "erecon"
ftarm = join(ftparts[1:min(2, length(ftparts))], '_')
config = stage == "pt" ? "configs/pretrain/$arm.toml" :
         stage == "ft" ? "configs/finetune/$ftarm.toml" :
         stage == "bl" ? "configs/baselines/$arm.toml" : error("sweep name must start with pt_, ft_ or bl_")
stage == "pt" || level in ("lvl1", "lvl2") || error("sweep name must end in _lvl1 or _lvl2")
stage == "bl" && !occursin("mlp", arm) && !occursin("log", arm) && error("bl sweeps are for the mlp and logistic baselines")
isfile(joinpath(LincsFM.PKG_DIR, config)) || error("no config $config")

# pretrained / random finetunes default to the checkpoint of the named objective (erecon if none) and the same input
stage == "ft" && !startswith(ftarm, "scratch_") && get!(fixed, "ckpt", "@$(objective)_" * split(ftarm, '_')[end])

lr = LincsFM.read_toml(joinpath(LincsFM.PKG_DIR, config))["lr"]
vals = Dict("NAME" => name, "CONFIG" => config, "PROJECT" => project, "LEVEL" => level,
            "MAX_STEPS" => level == "lvl1" ? "10000" : "40000", "EVAL_EVERY" => level == "lvl1" ? "1000" : "500",
            "LR_GRID" => @sprintf("[%.3g, %.3g, %.3g]", lr / 3, lr, lr * 3),
            "FIXED" => join(["  $k:\n    value: \"$v\"\n" for (k, v) in sort(collect(fixed))]))
template = stage == "bl" && occursin("log", arm) ? "bl_log" : stage
yaml = replace(read(joinpath(LincsFM.PKG_DIR, "configs", "sweeps", "$template.yaml"), String), ("\${$k}" => v for (k, v) in vals)...)
out = joinpath(LincsFM.PKG_DIR, "logs", "sweeps", "$name.yaml")
mkpath(dirname(out)); write(out, yaml)
println(out)
dry || cd(() -> run(`wandb sweep $out`), LincsFM.PKG_DIR)
