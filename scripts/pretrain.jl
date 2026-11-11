# pretrain an encoder from a config, then probe and health-check its final checkpoint
using Pkg; Pkg.activate(joinpath(dirname(@__DIR__), "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
using LincsFM
pretrain(LincsFM.cli_config(ARGS))
