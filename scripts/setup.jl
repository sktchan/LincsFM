# activate the tested environment for this cpu architecture and load LincsFM (included as a module, no package install)
using Pkg
const LINCS_DIR = dirname(@__DIR__)
Pkg.activate(joinpath(LINCS_DIR, "env", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"); io=devnull)
include(joinpath(LINCS_DIR, "src", "LincsFM.jl"))
using .LincsFM
