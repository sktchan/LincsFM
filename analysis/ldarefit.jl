# LDA refit of a finished finetune run's features
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM
run_ldarefit(LincsFM.cli_config(ARGS))
