module Cameras

using LinearAlgebra, StaticArrays, MappedArrays
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
    coarse_search

include("Camera.jl")
include("PointsFromImage.jl")
include("DIC.jl")

end # module
