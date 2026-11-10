module LincsFM

# LINCS L1000 transformer experiments: expression (ETF) vs rank (RTF) input, pretraining objectives, finetuning, baselines

using CUDA, Flux, Functors, JLD2, LinearAlgebra, NNlib, Optimisers, Printf, Random, Statistics, StatsBase, TOML, Dates
import PyCall

const PKG_DIR = dirname(@__DIR__)   # thesis/lincs

include("models/fastattn.jl")
using .FastAttn: attend

include("data/data.jl")
include("inputs.jl")
include("models/transformer.jl")
include("models/encoder.jl")
include("models/heads.jl")
include("models/classifier.jl")
include("objectives/objective.jl")
include("objectives/mlm.jl")
include("objectives/erecon.jl")
include("objectives/lrecon.jl")
include("train/common.jl")
include("train/checkpoint.jl")
include("train/diag.jl")
include("train/pretrain.jl")
include("train/classify.jl")
include("train/finetune.jl")
include("log.jl")
include("eval/eval.jl")
include("baselines/baselines.jl")

export LincsData, load_data, labels, pretrain_split, finetune_split
export Input, ETF, RTF, parse_input, gene_ranks, rank_features
export Encoder, Classifier, MLM, ERecon, LRecon, objective_from
export pretrain, finetune, load_config, load_encoder
export baseline, lda_baseline, trivial_baselines
export run_probe, run_health, run_ldarefit, run_bootstrap

end
