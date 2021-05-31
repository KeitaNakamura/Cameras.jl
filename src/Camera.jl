const SAME_POINT_THRESH = Ref(5)

"""
    Camera()
    Camera{T}()

Construct `Camera` object.
"""
mutable struct Camera{T}
    # internal parameters
    params::SMatrix{3, 3, T, 9}
    # external parameters
    t::SVector{3, T}       # translation
    Q::SMatrix{3, 3, T, 9} # rotation
end

function Camera{T}() where {T}
    params = zero(SMatrix{3, 3, T})
    t = zero(SVector{3, T})
    R = zero(SMatrix{3, 3, T})
    Camera{T}(params, t, R)
end

Camera() = Camera{Float64}()

"""
    rq(A)

Compute the RQ factorization of the matrix `A`.
"""
function rq(A)
    # https://math.stackexchange.com/questions/1640695/rq-decomposition
    Q, R = qr(reverse(A, dims = 1)')
    reverse(reverse(R', dims = 1), dims = 2), reverse(Q', dims = 1)
end

"""
    compute_homogeneous_matrix(xᵢ => Xᵢ)

Compute `H` in ``\\bm{x} \\simeq \\bm{H} \\bm{X}``.
"""
function compute_homogeneous_matrix((xᵢ, Xᵢ)::Pair{<: AbstractVector{SVector{2, T}}, <: AbstractVector{SVector{DIM, U}}}) where {DIM, T <: Real, U <: Real}
    n = length(eachindex(xᵢ, Xᵢ)) # number of samples
    ElType = promote_type(T, U)
    A = Array{ElType}(undef, 2n, 2(DIM+1)+DIM)
    b = Vector{ElType}(undef, 2n)
    @assert size(A, 1) ≥ size(A, 2)
    for i in 1:n
        x = xᵢ[i]
        X = Xᵢ[i]
        I = 2i - 1
        A[I:I+1, :] .= vcat([X; SVector(1); zero(X); SVector(0); -x[1]*X]',
                            [zero(X); SVector(0); X; SVector(1); -x[2]*X]')
        b[I:I+1] .= x
    end
    SMatrix{3, DIM+1}(reshape(push!(A \ b, 1), DIM+1, 3)')
end

"""
    calibrate!(camera::Camera, xᵢ => Xᵢ)

Calibrate `camera` from the pair of coordinates of image `xᵢ` and its corresponding actual coordinates `Xᵢ`.
The elements of `xᵢ` should be vector of length `2` and those of `Xᵢ` should be vector of length `3`.
"""
function calibrate!(camera::Camera, (xᵢ, Xᵢ)::Pair{<: AbstractVector{<: SVector{2}}, <: AbstractVector{<: SVector{3}}})
    P = compute_homogeneous_matrix(xᵢ => Xᵢ)
    R, Q = rq(P[SOneTo(3), SOneTo(3)])
    M = diagm(sign.(diag(R)))
    R = R * M
    Q = M * Q

    # internal parameters
    camera.params = R
    # external parameters
    camera.Q = Q
    camera.t = inv(R) * P[:, 4]

    camera
end

function projection_matrix(camera)
    A = camera.params
    Q = camera.Q
    t = camera.t
    [A*Q A*t]
end

"""
    camera(X)

Calculate coordinates in image from actual coordinates `X`.
`camera` should be [`calibrate!`](@ref)d before using this function.
"""
function (camera::Camera)(X::AbstractVector)
    @assert length(X) == 3
    P = projection_matrix(camera)
    b = [X; 1]
    scale = P[3, :]' * b
    x = (P * b) / scale
    SVector(x[1], x[2])
end

# see `generateQuads` in https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibinit.cpp
function find_contourquads(image)
    contours, painted = find_contours(image)

    contours_approx = map(contours) do contour
        pts = SVector{2}.(Tuple.(contour))
        local maybequad::typeof(pts)
        for ϵ in 1:7 # 7 is refered from `MAX_CONTOUR_APPROX` in calibinit.cpp
            # if maybequad is found, break loop
            maybequad = douglas_peucker(pts; thresh = ϵ, isclosed = true)
            length(maybequad) == 4 && break
            maybequad = douglas_peucker(maybequad; thresh = ϵ, isclosed = true)
            length(maybequad) == 4 && break
        end
        maybequad
    end

    # separate contours if possible to handle connected quad contours
    separated_list = Int[]
    for (I, contour) in enumerate(contours_approx)
        separated = false
        for i in 1:length(contour)
            xi = contour[i]
            for j in i+1:length(contour)
                xj = contour[j]
                v = xi - xj
                if sqrt(v[1]^2 + v[2]^2) < SAME_POINT_THRESH[]
                    push!(contours_approx, vcat(contour[1:i], contour[j+1:end]))
                    push!(contours_approx, contour[i+1:j])
                    separated = true
                    break
                end
            end
            separated && break
        end
        separated && push!(separated_list, I)
    end
    deleteat!(contours_approx, separated_list)

    quads = Vector{SVector{2, Int}}[]
    for maybequad in contours_approx
        length(maybequad) != 4 && continue

        min_size = 25
        l = arclength(maybequad; isclosed = true)
        A = contourarea(maybequad)
        d1 = norm(maybequad[1] - maybequad[3])
        d2 = norm(maybequad[2] - maybequad[4])
        d3 = norm(maybequad[1] - maybequad[2])
        d4 = norm(maybequad[2] - maybequad[3])
        if !(d3*4 > d4 && d4*4 > d3 && d3*d4 < A*1.5 && A > min_size &&
             d1 >= 0.15 * l && d2 >= 0.15 * l)
            continue
        end

        push!(quads, maybequad)
    end

    quads

    # painted = copy(image)
    # draw_contours(painted, RGB(1,0,0), quads)
    # quads, painted
end

struct ChessBoardQuad <: AbstractVector{SVector{2, Float64}}
    poins::Vector{SVector{2, Float64}}
    index::CartesianIndex{2}
end
Base.size(quad::ChessBoardQuad) = size(quad.poins)
Base.getindex(quad::ChessBoardQuad, i::Int) = quad.poins[i]

function connectedquad(dest::ChessBoardQuad, src::AbstractVector)
    @assert length(dest) == 4 && length(src) == 4
    function compute_index(j, I)
        inds = [0,0,0,0]
        inds[I] = j
        for _ in 1:3
            j += 1
            I += 1
            j > 4 && (j = 1)
            I > 4 && (I = 1)
            inds[I] = j
        end
        inds
    end
    for i in 1:length(dest)
        norms = [norm(dest[i] - src[j]) for j in 1:length(src)]
        inds = findall(norms .< SAME_POINT_THRESH[])
        isempty(inds) && continue
        j = only(inds)
        i == 1 && return ChessBoardQuad(src[compute_index(j, 3)], dest.index + CartesianIndex(-1, -1))
        i == 2 && return ChessBoardQuad(src[compute_index(j, 4)], dest.index + CartesianIndex(-1,  1))
        i == 3 && return ChessBoardQuad(src[compute_index(j, 1)], dest.index + CartesianIndex( 1,  1))
        i == 4 && return ChessBoardQuad(src[compute_index(j, 2)], dest.index + CartesianIndex( 1, -1))
    end
    nothing
end

function find_connectedquads(image)::Vector{ChessBoardQuad}
    quads = find_contourquads(image)
    found = false
    groups = []
    local group
    while !isempty(quads)
        if !found
            group = [ChessBoardQuad(pop!(quads), CartesianIndex(0, 0))]
            push!(groups, group)
        end
        found = false
        for i_dest in 1:length(group)
            for i_src in 1:length(quads)
                quad = connectedquad(group[i_dest], quads[i_src])
                if quad !== nothing
                    popat!(quads, i_src)
                    push!(group, quad)
                    found = true
                    break
                end
            end
            found && break
        end
    end
    groups[argmax(map(length, groups))] # extract the largest group
end

function paint_foundcorners(image, corners::Matrix{<: SVector{2}})
    painted = copy(image)
    cmap = cmap_rainbow()
    if size(corners, 1) < size(corners, 2)
        corners = oftype(corners, corners')
    end
    ToPoint(x) = Point(round(x[2]), round(x[1]))
    lastpoint = ToPoint(corners[1,1])
    for j in 1:size(corners, 2)
        slice = corners[:,j]
        for point in slice
            p = ToPoint(point)
            color = convert(eltype(painted), cmap[j])
            draw!(painted, Ellipse(CirclePointRadius(p, 10, thickness = 5, fill = false)), color)
            draw!(painted, LineSegment(lastpoint, p), color)
            lastpoint = p
        end
    end
    imshow(painted)
end

function _find_chessboardcorners(image)::Matrix
    quads = find_connectedquads(image)
    i_min, i_max = extrema(quad.index[1] for quad in quads)
    j_min, j_max = extrema(quad.index[2] for quad in quads)
    cornerlayout = CartesianIndices((i_min:i_max+1, j_min:j_max+1))
    corners = OffsetArray([SVector{2, Float64}[] for i in cornerlayout], cornerlayout)
    for i in eachindex(quads)
        quad = quads[i]
        push!(corners[quad.index + CartesianIndex(0, 0)], quad[1])
        push!(corners[quad.index + CartesianIndex(0, 1)], quad[2])
        push!(corners[quad.index + CartesianIndex(1, 1)], quad[3])
        push!(corners[quad.index + CartesianIndex(1, 0)], quad[4])
    end
    isinner = map(x -> length(x) > 1, corners)
    @assert !any(isinner[:,begin]) && !any(isinner[:,end]) &&
            !any(isinner[begin,:]) && !any(isinner[end,:]) && all(isinner[begin+1:end-1, begin+1:end-1])
    map(mean, corners[begin+1:end-1, begin+1:end-1])
end

function find_chessboardcorners(image)
    corners = _find_chessboardcorners(image)

    # check order of detected corners (not necessary?)
    diff = x -> abs(-(extrema(x)...))
    if diff(getindex.(corners[:,1], 1)) < diff(getindex.(corners[:,1], 2))
        corners = oftype(corners, corners')
    end
    if corners[1,1][1] > corners[end,1][1]
        corners = reverse(corners, dims = 1)
    end
    if corners[1,1][2] > corners[1,end][2]
        corners = reverse(corners, dims = 2)
    end

    paint_foundcorners(image, corners)

    corners
end

function cmap_rainbow()
    [RGB(255/255,   0/255,   0/255),
     RGB(255/255, 150/255,   0/255),
     RGB(255/255, 240/255,   0/255),
     RGB(  0/255, 135/255,   0/255),
     RGB(  0/255, 145/255, 255/255),
     RGB(  0/255, 100/255, 190/255),
     RGB(145/255,   0/255, 130/255)]
end
