var documenterSearchIndex = {"docs":
[{"location":"Utilities/","page":"Utilities","title":"Utilities","text":"DocTestSetup = :(using Cameras)","category":"page"},{"location":"Utilities/#Utilities","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"Utilities/#Reading-coordinates-from-image","page":"Utilities","title":"Reading coordinates from image","text":"","category":"section"},{"location":"Utilities/","page":"Utilities","title":"Utilities","text":"PointsFromImage","category":"page"},{"location":"Utilities/#Cameras.PointsFromImage","page":"Utilities","title":"Cameras.PointsFromImage","text":"PointsFromImage(image::AbstractArray)\n\nRead coordinates from image by clicking points on image.\n\n\n\n\n\n","category":"type"},{"location":"DIC/","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"DocTestSetup = :(using Cameras)","category":"page"},{"location":"DIC/#Digital-Image-Correlation-(DIC)","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"","category":"section"},{"location":"DIC/","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"Order = [:type, :function]\nPages = [\"DIC.md\"]","category":"page"},{"location":"DIC/","page":"Digital Image Correlation (DIC)","title":"Digital Image Correlation (DIC)","text":"Modules = [Cameras]\nOrder   = [:type, :function]\nPages   = [\"DIC.jl\"]","category":"page"},{"location":"DIC/#Cameras.coarse_search-Tuple{AbstractArray, AbstractArray}","page":"Digital Image Correlation (DIC)","title":"Cameras.coarse_search","text":"coarse_search(subset, image; region = CartesianIndices(image)) -> indices, C\n\nPerform coarse search subset in image using DIC. Return the indices which has the highest correlation with subset. Use image[indices] to get the found part of image. The searching region (entire image by default) can also be specified by CartesianIndices to reduce computations.\n\nSee also neighborindices.\n\nExamples\n\njulia> image = rand(10,10);\n\njulia> subset = image[3:5, 2:3];\n\njulia> coarse_search(subset, image)\n(CartesianIndex{2}[CartesianIndex(3, 2) CartesianIndex(3, 3); CartesianIndex(4, 2) CartesianIndex(4, 3); CartesianIndex(5, 2) CartesianIndex(5, 3)], 1.0)\n\n\n\n\n\n","category":"method"},{"location":"DIC/#Cameras.fine_search-Union{Tuple{dim}, Tuple{T}, Tuple{AbstractArray{T, dim}, AbstractArray{T, dim}, AbstractArray{var\"#s14\", N} where {var\"#s14\"<:CartesianIndex{dim}, N}}} where {T<:Real, dim}","page":"Digital Image Correlation (DIC)","title":"Cameras.fine_search","text":"fine_search(subset, image, first_guess::PixelIndices) -> center, C\n\nPerform fine search subset in image based on the Newton-Raphson method. The results by coarse_search can be used as first_guess. Note that returned center is a center coordinates (not integer any more) of searched subset in image.\n\nExamples\n\njulia> image = Cameras.testimage(\"buffalo\");\n\njulia> subset = image[100:300, 300:500];\n\njulia> center, C = fine_search(subset, image, CartesianIndices((101:301, 301:501)))\n([200.00000782067005, 400.00001094427904], 0.9999999999437896)\n\n\n\n\n\n","category":"method"},{"location":"DIC/#Cameras.neighborindices-Tuple{AbstractArray{var\"#s14\", N} where {dim, var\"#s14\"<:CartesianIndex{dim}, N}, AbstractArray, Int64}","page":"Digital Image Correlation (DIC)","title":"Cameras.neighborindices","text":"Cameras.neighborindices(subset::PixelIndices, image, npixels::Int)\n\nReturn npixels outer indices around subset. Violated indices in image are cut automatically. This is useful to give region in coarse_search.\n\njulia> image = rand(10,10);\n\njulia> Cameras.neighborindices(CartesianIndices((4:6, 3:6)), image, 2)\n7×8 CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}:\n CartesianIndex(2, 1)  CartesianIndex(2, 2)  …  CartesianIndex(2, 8)\n CartesianIndex(3, 1)  CartesianIndex(3, 2)     CartesianIndex(3, 8)\n CartesianIndex(4, 1)  CartesianIndex(4, 2)     CartesianIndex(4, 8)\n CartesianIndex(5, 1)  CartesianIndex(5, 2)     CartesianIndex(5, 8)\n CartesianIndex(6, 1)  CartesianIndex(6, 2)     CartesianIndex(6, 8)\n CartesianIndex(7, 1)  CartesianIndex(7, 2)  …  CartesianIndex(7, 8)\n CartesianIndex(8, 1)  CartesianIndex(8, 2)     CartesianIndex(8, 8)\n\n\n\n\n\n","category":"method"},{"location":"DIC/#Cameras.walkindices-Tuple{AbstractArray, AbstractArray}","page":"Digital Image Correlation (DIC)","title":"Cameras.walkindices","text":"Cameras.walkindices(subset, image; region = CartesianIndices(image))\n\nReturn indices to walk image with size of subset.\n\njulia> image = rand(4,4);\n\njulia> subset = rand(2,2);\n\njulia> Cameras.walkindices(subset, image)\n3×3 Matrix{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}:\n [CartesianIndex(1, 1) CartesianIndex(1, 2); CartesianIndex(2, 1) CartesianIndex(2, 2)]  …  [CartesianIndex(1, 3) CartesianIndex(1, 4); CartesianIndex(2, 3) CartesianIndex(2, 4)]\n [CartesianIndex(2, 1) CartesianIndex(2, 2); CartesianIndex(3, 1) CartesianIndex(3, 2)]     [CartesianIndex(2, 3) CartesianIndex(2, 4); CartesianIndex(3, 3) CartesianIndex(3, 4)]\n [CartesianIndex(3, 1) CartesianIndex(3, 2); CartesianIndex(4, 1) CartesianIndex(4, 2)]     [CartesianIndex(3, 3) CartesianIndex(3, 4); CartesianIndex(4, 3) CartesianIndex(4, 4)]\n\n\n\n\n\n","category":"method"},{"location":"DIC/#Cameras.zncc-Union{Tuple{U}, Tuple{T}, Tuple{AbstractArray{T, N} where N, AbstractArray{U, N} where N}} where {T<:Real, U<:Real}","page":"Digital Image Correlation (DIC)","title":"Cameras.zncc","text":"zncc(image1, image2)\n\nPerform zero-mean normalized cross-correlation between two images.\n\nC = fracsum(A_ij - barA_ij) (B_ij - barB_ij)sqrtsum(A_ij - barA_ij)^2 sum(B_ij - barB_ij)^2\n\n\n\n\n\n","category":"method"},{"location":"Camera/","page":"Camera","title":"Camera","text":"DocTestSetup = :(using Cameras)","category":"page"},{"location":"Camera/#Camera","page":"Camera","title":"Camera","text":"","category":"section"},{"location":"Camera/","page":"Camera","title":"Camera","text":"Order = [:type, :function]\nPages = [\"Camera.md\"]","category":"page"},{"location":"Camera/","page":"Camera","title":"Camera","text":"Modules = [Cameras]\nOrder   = [:type, :function]\nPages   = [\"Camera.jl\"]","category":"page"},{"location":"Camera/#Cameras.Camera","page":"Camera","title":"Cameras.Camera","text":"Camera()\nCamera{T}()\n\nConstruct Camera object.\n\n\n\n\n\n","category":"type"},{"location":"Camera/#Cameras.Camera-Tuple{AbstractVector{T} where T}","page":"Camera","title":"Cameras.Camera","text":"camera(X)\n\nCalculate coordinates in image from actual coordinates X. camera should be calibrate!d before using this function.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.calibrate!-Tuple{Camera, Pair{var\"#s1\", var\"#s12\"} where {var\"#s1\"<:(AbstractVector{var\"#s13\"} where var\"#s13\"<:(SVector{2, T} where T)), var\"#s12\"<:(AbstractVector{var\"#s14\"} where var\"#s14\"<:(SVector{3, T} where T))}}","page":"Camera","title":"Cameras.calibrate!","text":"calibrate!(camera::Camera, xᵢ => Xᵢ)\n\nCalibrate camera from the pair of coordinates of image xᵢ and its corresponding actual coordinates Xᵢ. The elements of xᵢ should be vector of length 2 and those of Xᵢ should be vector of length 3.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.compute_homogeneous_matrix-Union{Tuple{Pair{var\"#s3\", var\"#s2\"} where {var\"#s3\"<:AbstractArray{SVector{2, T}, 1}, var\"#s2\"<:AbstractArray{SVector{DIM, U}, 1}}}, Tuple{U}, Tuple{T}, Tuple{DIM}} where {DIM, T<:Real, U<:Real}","page":"Camera","title":"Cameras.compute_homogeneous_matrix","text":"compute_homogeneous_matrix(xᵢ => Xᵢ)\n\nCompute H in bmx simeq bmH bmX.\n\n\n\n\n\n","category":"method"},{"location":"Camera/#Cameras.rq-Tuple{Any}","page":"Camera","title":"Cameras.rq","text":"rq(A)\n\nCompute the RQ factorization of the matrix A.\n\n\n\n\n\n","category":"method"},{"location":"#Cameras","page":"Home","title":"Cameras","text":"","category":"section"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/KeitaNakamura/Cameras.jl.git","category":"page"}]
}