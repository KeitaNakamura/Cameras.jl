tofloat(A::AbstractArray{<: Gray}) = mappedarray(Float64, A)
tofloat(A::AbstractArray) = tofloat(mappedarray(Gray, A))

"""
    zncc(image1, image2)

Perform zero-mean normalized cross-correlation between two images.

```math
C = \\frac{\\sum{(A_{ij} - \\bar{A}_{ij}) (B_{ij} - \\bar{B}_{ij})}}{\\sqrt{\\sum{(A_{ij} - \\bar{A}_{ij})^2} \\sum{(B_{ij} - \\bar{B}_{ij})^2}}}
```
"""
function zncc(A::AbstractArray{T}, B::AbstractArray{U}) where {T <: Real, U <: Real}
    size(A) == size(B) || throw(DimensionMismatch("Dimensions must match."))

    # mean values
    Ā = zero(T)
    B̄ = zero(U)
    @inbounds @simd for i in eachindex(A, B)
        Ā += A[i]
        B̄ += B[i]
    end
    Ā /= length(A)
    B̄ /= length(B)

    # numerator/denominator
    num = zero(promote_type(T, U))
    denA = zero(T)
    denB = zero(U)
    @inbounds @simd for i in eachindex(A, B)
        Aᵢ = A[i]
        Bᵢ = B[i]
        AA = Aᵢ - Ā
        BB = Bᵢ - B̄
        num += AA * BB
        denA += AA^2
        denB += BB^2
    end

    num / sqrt(denA * denB)
end
zncc(A::AbstractArray, B::AbstractArray) = zncc(tofloat(A), tofloat(B))

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

# for 2D
solution_vector(::Type{T}, ::Val{2}) where {T} = zero(Vec{6, T})
function compute_correlation(subset::AbstractArray{<: Real, 2}, image_itp::AbstractArray{T, 2}, first_guess::PixelIndices{2}, X::Vec{6}) where {T}
    xc, yc = Tuple(first(first_guess) + last(first_guess)) ./ 2
    u, v, dudx, dudy, dvdx, dvdy = Tuple(X)
    sol = map(first_guess) do I
        x, y = Tuple(I)
        dx = x - xc
        dy = y - yc
        x′ = x + u + dudx*dx + dudy*dy
        y′ = y + v + dvdx*dx + dvdy*dy
        # If calculted coordinates are out side of image, just return zero.
        # This means that out side of image are filled with black color.
        checkbounds(Bool, image_itp, x′, y′) ? image_itp(x′, y′) : zero(T)
    end
    zncc(subset, sol)
end

# TODO: for 3D
# solution_vector
# compute_correlation

"""
    fine_search(subset, image, first_guess::PixelIndices) -> center, C

Perform fine search `subset` in `image` based on the Newton-Raphson method.
The results by [`coarse_search`](@ref) can be used as `first_guess`.
Note that returned `center` is a center coordinates (not integer any more) of searched subset in `image`.

# Examples

```jldoctest
julia> image = Cameras.testimage("buffalo");

julia> subset = image[100:300, 300:500];

julia> center, C = fine_search(subset, image, CartesianIndices((101:301, 301:501)))
([200.00000782067005, 400.0000109442791], 0.999999999943781)
```
"""
function fine_search(subset::AbstractArray{T, dim}, image::AbstractArray{T, dim}, first_guess::PixelIndices{dim}) where {T <: Real, dim}
    @assert size(subset) == size(first_guess)
    image_itp = interpolate(image, BSpline(Linear())) # sub-pixel interpolation
    x = solution_vector(T, Val(dim))
    C = 0.0
    for i in 1:20
        H, ∇x, C = hessian(x -> compute_correlation(subset, image_itp, first_guess, x), x, :all)
        C ≈ 1 && break
        x += -H \ ∇x
    end
    center = Tuple(first(first_guess) + last(first_guess)) ./ 2
    Vec(ntuple(i -> center[i] + x[i], Val(dim))), C
end

function fine_search(subset::AbstractArray, image::AbstractArray, first_guess::PixelIndices)
    fine_search(tofloat(subset), tofloat(image), first_guess)
end
