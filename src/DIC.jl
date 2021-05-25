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
        A = Aᵢ - Ā
        B = Bᵢ - B̄
        n += A * B
        d_A += A^2
        d_B += B^2
    end

    n / sqrt(d_A * d_B)
end
zncc(A::AbstractArray, B::AbstractArray) = zncc(tofloat(A), tofloat(B))
