const PixelIndices{dim} = AbstractArray{<: CartesianIndex{dim}}

tofloat(A::AbstractArray{<: Gray}) = mappedarray(Float64, A)
tofloat(A::AbstractArray) = tofloat(mappedarray(Gray, A))

"""
    zncc(image1, image2)

Perform zero-mean normalized cross-correlation between two images.

```math
C = \\frac{\\sum{(A_{ij} - \\bar{A}_{ij}) (B_{ij} - \\bar{B}_{ij})}}{\\sqrt{\\sum{(A_{ij} - \\bar{A}_{ij})^2} \\sum{(B_{ij} - \\bar{B}_{ij})^2}}}
```
"""
function zncc(A::AbstractArray{T}, B::AbstractArray{T}) where {T <: Real}
    size(A) == size(B) || throw(DimensionMismatch("Dimensions must match."))

    # mean values
    Ā = zero(T)
    B̄ = zero(T)
    @inbounds @simd for i in eachindex(A, B)
        Ā += A[i]
        B̄ += B[i]
    end
    Ā /= length(A)
    B̄ /= length(B)

    # numerator/denominator
    n = zero(T)
    d_A = zero(T)
    d_B = zero(T)
    @inbounds @simd for i in eachindex(A, B)
        Aᵢ = A[i]
        Bᵢ = B[i]
        AA = Aᵢ - Ā
        BB = Bᵢ - B̄
        n += AA * BB
        d_A += AA^2
        d_B += BB^2
    end

    n / sqrt(d_A * d_B)
end
zncc(A::AbstractArray, B::AbstractArray) = zncc(tofloat(A), tofloat(B))

"""
    walkindices(subset, image; region = CartesianIndices(image))

Return indices to walk `image` with size of `subset`.

```jldoctest
julia> image = rand(4,4);

julia> subset = rand(2,2);

julia> walkindices(subset, image)
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
    neighborindices(subset::PixelIndices, image, npixels::Int)

Return `npixels` outer indices around `subset`.
Violated indices in `image` are cut automatically.
This is useful to give `region` in [`coarse_search`](@ref).

```jldoctest
julia> image = rand(10,10);

julia> neighborindices(CartesianIndices((4:6, 3:6)), image, 2)
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
    newstart = clamp.(start, 1, size(image))
    newstop = clamp.(stop, 1, size(image))
    CartesianIndex(newstart):CartesianIndex(newstop)
end

function neighborindices(point::CartesianIndex, image::AbstractArray, npixels::Int)
    neighborindices(point:point, image, npixels)
end

"""
    coarse_search(subset, image; region = CartesianIndices(image)) -> indices, C

Perform coarse search `subset` in `image` using DIC.
Return the `indices` which has the highest correlation with `subset`.
Use `image[indices]` to get the found part of image.
The searching `region` (entire image by default) can also be specified
by `CartesianIndices` to reduce computations.

See also [`neighborindices`](@ref).

# Examples

```jldoctest
julia> image = rand(10,10);

julia> subset = image[3:5, 2:3];

julia> coarse_search(subset, image)
(CartesianIndex{2}[CartesianIndex(3, 2) CartesianIndex(3, 3); CartesianIndex(4, 2) CartesianIndex(4, 3); CartesianIndex(5, 2) CartesianIndex(5, 3)], 1.0)
```
"""
function coarse_search(subset::AbstractArray, image::AbstractArray; region::PixelIndices = CartesianIndices(image))
    inds = walkindices(subset, image; region)
    Cs = similar(inds, Float64)
    Threads.@threads for i in eachindex(inds, Cs)
        @inbounds Cs[i] = zncc(view(image, inds[i]), subset)
    end
    I = argmax(Cs)
    inds[I], Cs[I]
end
