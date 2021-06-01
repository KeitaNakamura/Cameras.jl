module Cameras

using Reexport
@reexport using StaticArrays
using LinearAlgebra, Statistics, MappedArrays, OffsetArrays, Interpolations, ForwardDiff, DiffResults
using FileIO, ImageMagick, ImageIO
using ImageView # imshow
using ImageCore, ImageDraw
using GtkReactive

using Base: @_propagate_inbounds_meta

# reexport
export load, imshow

export
# Camera
    Camera,
    nsamples,
    calibrate!,
# Chessboard
    Chessboard,
# PointsFromImage
    PointsFromImage,
# DIC
    zncc,
    coarse_search,
    fine_search

include("utils.jl")
include("contour.jl")
include("Chessboard.jl")
include("Camera.jl")
include("PointsFromImage.jl")
include("DIC.jl")
include("stereo.jl")

end # module
