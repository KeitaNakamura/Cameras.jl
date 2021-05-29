module Cameras

using Reexport
@reexport using StaticArrays
using LinearAlgebra, MappedArrays, Interpolations, ForwardDiff, DiffResults
using FileIO, ImageMagick, ImageIO
using ImageView # imshow
using ImageCore
using GtkReactive

# reexport
export load, imshow

export
# Camera
    Camera,
    nsamples,
    calibrate!,
# PointsFromImage
    PointsFromImage,
# DIC
    zncc,
    coarse_search,
    fine_search

include("utils.jl")
include("contour.jl")
include("Camera.jl")
include("PointsFromImage.jl")
include("DIC.jl")
include("stereo.jl")

end # module
