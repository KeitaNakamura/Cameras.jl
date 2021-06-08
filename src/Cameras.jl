module Cameras

using Reexport
@reexport using Tensorial
using LinearAlgebra, Statistics
using MappedArrays, OffsetArrays
using Interpolations, ImageFiltering
using ForwardDiff, DiffResults
using Colors
using FileIO, ImageMagick, ImageIO # for io stream
using ImageView # for imshow
using ImageDraw # for draw
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

const PixelIndices{dim} = AbstractArray{<: CartesianIndex{dim}}

include("utils.jl")
include("contour.jl")
include("Chessboard.jl")
include("Camera.jl")
include("PointsFromImage.jl")
include("DIC.jl")
include("stereo.jl")

end # module
