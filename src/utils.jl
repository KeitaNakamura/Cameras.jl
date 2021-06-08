function testimage(name::String)
    if splitext(name)[1] == "buffalo"
        return load(joinpath(dirname(@__FILE__), "../images/buffalo.tif"))
    end
    throw(ArgumentError("test image $name is not exist"))
end

"""
    Cameras.walkindices(subset, image; region = CartesianIndices(image))

Return indices to walk `image` with size of `subset`.

```jldoctest
julia> image = rand(4,4);

julia> subset = rand(2,2);

julia> Cameras.walkindices(subset, image)
3×3 Matrix{CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}}:
 [CartesianIndex(1, 1) CartesianIndex(1, 2); CartesianIndex(2, 1) CartesianIndex(2, 2)]  …  [CartesianIndex(1, 3) CartesianIndex(1, 4); CartesianIndex(2, 3) CartesianIndex(2, 4)]
 [CartesianIndex(2, 1) CartesianIndex(2, 2); CartesianIndex(3, 1) CartesianIndex(3, 2)]     [CartesianIndex(2, 3) CartesianIndex(2, 4); CartesianIndex(3, 3) CartesianIndex(3, 4)]
 [CartesianIndex(3, 1) CartesianIndex(3, 2); CartesianIndex(4, 1) CartesianIndex(4, 2)]     [CartesianIndex(3, 3) CartesianIndex(3, 4); CartesianIndex(4, 3) CartesianIndex(4, 4)]
```
"""
function walkindices(subset::AbstractArray, image::AbstractArray; region::PixelIndices = CartesianIndices(image))
    checkbounds(image, region)
    checkbounds(region, CartesianIndices(subset))
    origins = first(region):first(region)+CartesianIndex(size(region) .- size(subset))
    map(origins) do I
        I:I+CartesianIndex(size(subset).-1)
    end
end

"""
    Cameras.neighborindices(subset::PixelIndices, image, npixels::Int)

Return `npixels` outer indices around `subset`.
Violated indices in `image` are cut automatically.
This is useful to give `region` in [`coarse_search`](@ref).

```jldoctest
julia> image = rand(10,10);

julia> Cameras.neighborindices(CartesianIndices((4:6, 3:6)), image, 2)
7×8 CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}:
 CartesianIndex(2, 1)  CartesianIndex(2, 2)  …  CartesianIndex(2, 8)
 CartesianIndex(3, 1)  CartesianIndex(3, 2)     CartesianIndex(3, 8)
 CartesianIndex(4, 1)  CartesianIndex(4, 2)     CartesianIndex(4, 8)
 CartesianIndex(5, 1)  CartesianIndex(5, 2)     CartesianIndex(5, 8)
 CartesianIndex(6, 1)  CartesianIndex(6, 2)     CartesianIndex(6, 8)
 CartesianIndex(7, 1)  CartesianIndex(7, 2)  …  CartesianIndex(7, 8)
 CartesianIndex(8, 1)  CartesianIndex(8, 2)     CartesianIndex(8, 8)
```
"""
function neighborindices(subset::PixelIndices, image::AbstractArray, npixels::Int)
    start = Tuple(first(subset)) .- npixels
    stop = Tuple(last(subset)) .+ npixels
    firsts = first.(axes(image))
    lasts = last.(axes(image))
    newstart = clamp.(start, firsts, lasts)
    newstop = clamp.(stop, firsts, lasts)
    CartesianIndex(newstart):CartesianIndex(newstop)
end

function neighborindices(point::CartesianIndex, image::AbstractArray, npixels::Int)
    neighborindices(point:point, image, npixels)
end

function perpendicular_distance(x::Vec{dim}, line::Pair{<: Vec{dim}, <: Vec{dim}}) where {dim}
    start, stop = line
    v1 = stop - start
    v2 = x - start
    n = normalize(v1)
    norm((v2 ⋅ n) * n - v2)
end

function _douglas_peucker(list::AbstractVector{Vec{dim, T}}, ϵ::Real) where {dim, T}
    # Find the point with the maximum distance
    index = 0
    dmax = zero(T)
    for i in 2:length(list)-1
        @inbounds d = perpendicular_distance(list[i], list[1] => list[end])
        if d > dmax
            index = i
            dmax = d
        end
    end
    # If max distance is greater than ϵ, recursively simplify
    if dmax > ϵ
        lhs = _douglas_peucker(@view(list[1:index]), ϵ)
        rhs = _douglas_peucker(@view(list[index:end]), ϵ)
        pop!(lhs)
        append!(lhs, rhs)
    else
        [list[1], list[end]]
    end
end

function douglas_peucker(list::AbstractVector{Vec{dim, T}}; thresh::Real, isclosed::Bool) where {dim, T}
    list = _douglas_peucker(list, thresh)
    if isclosed
        dmax = zero(T)
        lhs_start = 1
        rhs_start = 1
        # TODO: more clever way
        for i in 1:length(list)÷2 # `÷2` is for faster computation. it can get approximately farthest pair?
            for j in i+1:length(list)
                @inbounds x = list[i] - list[j]
                dd = dot(x, x)
                if dd > dmax*dmax
                    dmax = sqrt(dd)
                    lhs_start = i
                    rhs_start = j
                end
            end
        end
        lhs = _douglas_peucker(list[lhs_start:rhs_start], thresh)
        rhs = _douglas_peucker(vcat(list[rhs_start:end], list[1:lhs_start]), thresh)
        list = vcat(lhs[1:end-1], rhs[1:end-1])
    end
    list
end

function arclength(list::AbstractVector; isclosed::Bool)
    l = sum(norm(list[i]-list[i+1]) for i in 1:length(list)-1)
    if isclosed
        l += norm(list[1] - list[end])
    end
    l
end

function contourarea(list::AbstractVector{<: AbstractVector{T}}) where {T}
    poly = [list; [list[1]]]
    A = zero(T)
    for i in 1:length(poly)-1
        @inbounds begin
            Xᵢ = poly[i]
            Xᵢ₊₁ = poly[i+1]
            xᵢ, yᵢ = Xᵢ[1], Xᵢ[2]
            xᵢ₊₁, yᵢ₊₁ = Xᵢ₊₁[1], Xᵢ₊₁[2]
        end
        a = (xᵢ * yᵢ₊₁ - xᵢ₊₁ * yᵢ)
        A += a
    end
    A /= 2
    abs(A)
end

function harris_subpixel(img, k, pos::CartesianIndex{2}, npixels::Int)
    getindex_offset(A, inds) = OffsetArray(A[inds], inds)

    # use image around of `pos`
    img = getindex_offset(img, neighborindices(pos, img, 10*npixels))

    blurred = imfilter(Float64.(Gray.(img)), Kernel.gaussian(1))
    Iy, Ix = imgradients(blurred, KernelFactors.sobel, "replicate")
    IxIx = imfilter(Ix .* Ix, Kernel.gaussian(1))
    IxIy = imfilter(Ix .* Iy, Kernel.gaussian(1))
    IyIy = imfilter(Iy .* Iy, Kernel.gaussian(1))
    R = @. (IxIx * IyIy - IxIy * IxIy) - k * (IxIx + IyIy)^2

    R′ = getindex_offset(R, neighborindices(pos, img, npixels))
    x, y = Tuple(argmax(R′)) # do harris just in case

    # quadratic approximation
    # http://www.ipol.im/pub/art/2018/229/article_lr.pdf
    Rx = (R[x+1, y] - R[x-1, y]) / 2
    Ry = (R[x, y+1] - R[x, y-1]) / 2
    Rxx = R[x+1, y] + R[x-1, y] - 2R[x, y]
    Ryy = R[x, y+1] + R[x, y-1] - 2R[x, y]
    Rxy = (R[x+1, y+1] - R[x-1, y-1] - R[x+1, y-1] - R[x-1, y+1]) / 4
    ∇R = @Vec [Rx, Ry]
    ∇²R = @Mat [Rxx Rxy
                Rxy Ryy]
    [x, y] - inv(∇²R) ⋅ ∇R
end
