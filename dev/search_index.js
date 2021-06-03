var documenterSearchIndex = {"docs":
[{"location":"Utilities/","page":"Utilities","title":"Utilities","text":"DocTestSetup = :(using Cameras)","category":"page"},{"location":"Utilities/#Utilities","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"Utilities/#Reading-coordinates-from-image","page":"Utilities","title":"Reading coordinates from image","text":"","category":"section"},{"location":"Utilities/","page":"Utilities","title":"Utilities","text":"PointsFromImage","category":"page"},{"location":"Utilities/#Cameras.PointsFromImage","page":"Utilities","title":"Cameras.PointsFromImage","text":"PointsFromImage(image::AbstractArray)\n\nRead coordinates from image by clicking points on image.\n\n\n\n\n\n","category":"type"},{"location":"DIC/","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"DocTestSetup = :(using Cameras)","category":"page"},{"location":"DIC/#Digital-Image-Correlation-(DIC)","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"","category":"section"},{"location":"DIC/","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"Order = [:type, :function]\nPages = [\"DIC.md\"]","category":"page"},{"location":"DIC/","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"Modules = [Cameras]\nOrder   = [:type, :function]\nPages   = [\"DIC.jl\"]","category":"page"},{"location":"DIC/#Cameras.coarse_search-Tuple{AbstractArray, AbstractArray}","page":"Digital Image Correlation (DIC)","title":"Cameras.coarse_search","text":"coarse_search(subset, image; region = CartesianIndices(image)) -> indices, C\n\nPerform coarse search subset in image using DIC. Return the indices which has the highest correlation with subset. Use image[indices] to get the found part of image. The searching region (entire image by default) can also be specified by CartesianIndices to reduce computations.\n\nSee also neighborindices.\n\nExamples\n\njulia> image = rand(10,10);\n\njulia> subset = image[3:5, 2:3];\n\njulia> coarse_search(subset, image)\n(CartesianIndex{2}[CartesianIndex(3, 2) CartesianIndex(3, 3); CartesianIndex(4, 2) CartesianIndex(4, 3); CartesianIndex(5, 2) CartesianIndex(5, 3)], 1.0)\n\n\n\n\n\n","category":"method"},{"location":"DIC/#Cameras.fine_search-Union{Tuple{dim}, Tuple{T}, Tuple{AbstractArray{T, dim}, AbstractArray{T, dim}, AbstractArray{var\"#s1\", N} where {var\"#s1\"<:CartesianIndex{dim}, N}}} where {T<:Real, dim}","page":"Digital Image Correlation (DIC)","title":"Cameras.fine_search","text":"fine_search(subset, image, first_guess::PixelIndices) -> center, C\n\nPerform fine search subset in image based on the Newton-Raphson method. The results by coarse_search can be used as first_guess. Note that returned center is a center coordinates (not integer any more) of searched subset in image.\n\nExamples\n\njulia> image = Cameras.testimage(\"buffalo\");\n\njulia> subset = image[100:300, 300:500];\n\njulia> center, C = fine_search(subset, image, CartesianIndices((101:301, 301:501)))\n([200.00000782067005, 400.00001094427904], 0.9999999999437896)\n\n\n\n\n\n","category":"method"},{"location":"DIC/#Cameras.zncc-Union{Tuple{U}, Tuple{T}, Tuple{AbstractArray{T, N} where N, AbstractArray{U, N} where N}} where {T<:Real, U<:Real}","page":"Digital Image Correlation (DIC)","title":"Cameras.zncc","text":"zncc(image1, image2)\n\nPerform zero-mean normalized cross-correlation between two images.\n\nC = fracsum(A_ij - barA_ij) (B_ij - barB_ij)sqrtsum(A_ij - barA_ij)^2 sum(B_ij - barB_ij)^2\n\n\n\n\n\n","category":"method"},{"location":"Camera/","page":"Camera","title":"Camera","text":"DocTestSetup = :(using Cameras)","category":"page"},{"location":"Camera/#Camera","page":"Camera","title":"Camera","text":"","category":"section"},{"location":"Camera/","page":"Camera","title":"Camera","text":"Order = [:type, :function]\nPages = [\"Camera.md\"]","category":"page"},{"location":"Camera/","page":"Camera","title":"Camera","text":"Modules = [Cameras]\nOrder   = [:type, :function]\nPages   = [\"Camera.jl\"]","category":"page"},{"location":"Camera/#Cameras.Camera","page":"Camera","title":"Cameras.Camera","text":"Camera()\nCamera{T}()\n\nConstruct Camera object.\n\n\n\n\n\n","category":"type"},{"location":"Camera/#Cameras.Camera-Tuple{AbstractVector{T} where T}","page":"Camera","title":"Cameras.Camera","text":"camera(X)\n\nCalculate coordinates in image from actual coordinates X. camera should be calibrate!d before using this function.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.calibrate!-Tuple{Camera, Vector{var\"#s18\"} where var\"#s18\"<:Chessboard}","page":"Camera","title":"Cameras.calibrate!","text":"calibrate!(camera::Camera, chessboards::Vector{<: Chessboard}; [gridspace = 1])\n\nCalibrate camera from chessboards.\n\nSee also calibrate_intrinsic! and calibrate_extrinsic!.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.calibrate!-Union{Tuple{U}, Tuple{T}, Tuple{Camera, Pair{Array{SVector{2, T}, 1}, Array{SVector{3, U}, 1}}}} where {T, U}","page":"Camera","title":"Cameras.calibrate!","text":"calibrate!(camera::Camera, xᵢ => Xᵢ)\n\nCalibrate camera from the pair of coordinates of image xᵢ and its corresponding actual coordinates Xᵢ. The elements of xᵢ should be vector of length 2 and those of Xᵢ should be vector of length 3.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.calibrate_extrinsic!-Tuple{Camera, Chessboard}","page":"Camera","title":"Cameras.calibrate_extrinsic!","text":"calibrate!(camera::Camera, chessboard::Chessboard; [gridspace = 1])\n\nCalibrate extrinsic parameters of camera from chessboard.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.calibrate_intrinsic!-Tuple{Camera, Vector{var\"#s21\"} where var\"#s21\"<:Chessboard}","page":"Camera","title":"Cameras.calibrate_intrinsic!","text":"calibrate!(camera::Camera, chessboards::Vector{<: Chessboard})\n\nCalibrate intrinsic parameters of camera from chessboards.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.compute_homogeneous_matrix-Union{Tuple{Pair{Array{SVector{2, T}, 1}, Array{SVector{DIM, U}, 1}}}, Tuple{U}, Tuple{T}, Tuple{DIM}} where {DIM, T<:Real, U<:Real}","page":"Camera","title":"Cameras.compute_homogeneous_matrix","text":"compute_homogeneous_matrix(xᵢ => Xᵢ)\n\nCompute H in bmx simeq bmH bmX.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.rq-Tuple{Any}","page":"Camera","title":"Cameras.rq","text":"rq(A)\n\nCompute the RQ factorization of the matrix A.\n\n\n\n\n\n","category":"method"},{"location":"#Cameras","page":"Home","title":"Cameras","text":"","category":"section"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/KeitaNakamura/Cameras.jl.git","category":"page"}]
}