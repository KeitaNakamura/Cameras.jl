module Cameras

using LinearAlgebra, StaticArrays
using FileIO, ImageMagick, ImageIO
using ImageView # imshow
using GtkReactive

export
    load,
    imshow

export
    Camera,
    nsamples,
    calibrate!,
    PointsFromImage

include("Camera.jl")
include("PointsFromImage.jl")

end # module
