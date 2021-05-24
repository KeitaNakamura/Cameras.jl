mutable struct Camera{T}
    # internal parameters
    f_δx::MVector{2, T}     # (focal length) * (pixel per distance)
    x₀::MVector{2, T}       # offsets in image
    # external parameters
    t::MVector{3, T}        # translation
    R::MMatrix{3, 3, T, 9}  # rotation
    # lhs * P_vector = x
    P::MMatrix{3, 4, T, 12} # projection matrix
    lhs::Vector{T}          # lhs matrix (reshaped later)
    rhs::Vector{T}          # rhs vector
end

function Camera{T}() where {T}
    # f = zero(T)
    f_δx = x₀ = zero(MVector{2, T})
    t = zero(MVector{3, T})
    R = zero(MMatrix{3, 3, T})
    P = zero(MMatrix{3, 4, T})
    lhs = T[]
    rhs = T[]
    Camera{T}(f_δx, x₀, t, R, P, lhs, rhs)
end

Camera() = Camera{Float64}()

function Base.push!(camera::Camera, (x, X)::Pair{<: AbstractVector, <: AbstractVector})
    @assert length(x) == 2
    @assert length(X) == 3
    𝟘 = zeros(length(X) + 1)
    append!(camera.lhs, [X; 1;    𝟘; -x[1]*X])
    append!(camera.lhs, [   𝟘; X; 1; -x[2]*X])
    append!(camera.rhs, x)
    camera
end

function reset!(camera::Camera)
    camera.lhs = []
    camera.rhs = []
    camera
end

nsamples(camera::Camera) = length(camera.rhs) ÷ 2

function calibrate!(camera::Camera)
    n = nsamples(camera)
    2n > 11 || @warn "number of sample points should be at least 6, but $n"

    # solve projection matrix
    lhs = reshape(camera.lhs, Int(length(camera.lhs) / 2n), 2n)'
    sol = lhs \ camera.rhs
    push!(sol, 1) # fill element at (3,4)
    camera.P .= reshape(sol, 4, 3)'

    R, A = qr(camera.P[1:3, 1:3])
    # internal parameters
    camera.f_δx .= [A[1,1], A[2,2]]
    camera.x₀ .= A[1:2, 3]
    # external parameters
    camera.R .= R
    camera.t .= inv(A) * camera.P[1:3, 4]

    camera
end

function calibrate!(camera::Camera, (xᵢ, Xᵢ)::Pair{<: AbstractVector{<: AbstractVector}, <: AbstractVector{<: AbstractVector}})
    reset!(camera)
    for (x, X) in zip(xᵢ, Xᵢ)
        push!(camera, x => X)
    end
    calibrate!(camera)
    camera
end

function (camera::Camera)(X::AbstractVector)
    @assert length(X) == 3
    b = [X; 1]
    h = camera.P[3, :]' * b
    x = camera.P * b / h
    x[1:2]
end
