struct Chessboard{T} <: AbstractMatrix{T}
    image::Matrix{T}
    corners::Matrix{Vec{2, Float64}}
end
Base.size(x::Chessboard) = size(x.image)
Base.IndexStyle(::Type{<: Chessboard}) = IndexLinear()
Base.getindex(x::Chessboard, i::Int) = (@_propagate_inbounds_meta; x.image[i])
imagepoints(x::Chessboard) = x.corners
function objectpoints(x::Chessboard)
    dims = size(imagepoints(x)) .- 1
    Vec{2, Float64}.(Tuple.(CartesianIndices(UnitRange.(0, dims))))
end

struct ChessboardQuad <: AbstractVector{Vec{2, Float64}}
    poins::Vector{Vec{2, Float64}}
    index::CartesianIndex{2}
end
Base.size(quad::ChessboardQuad) = size(quad.poins)
Base.getindex(quad::ChessboardQuad, i::Int) = (@_propagate_inbounds_meta; quad.poins[i])

# see `generateQuads` in https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibinit.cpp
function find_contourquads(image; binary_thresh)
    contours, painted = find_contours(image; thresh = binary_thresh)

    contours_approx = map(contours) do contour
        pts = Vec{2, Float64}.(Tuple.(contour))
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
                if sqrt(v[1]^2 + v[2]^2) < SAME_POINT_THRESH
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

    quads = Vector{Vec{2, Int}}[]
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

function connectedquad(dest::ChessboardQuad, src::AbstractVector)
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
        inds = findall(norms .< SAME_POINT_THRESH)
        isempty(inds) && continue
        j = only(inds)
        i == 1 && return ChessboardQuad(src[compute_index(j, 3)], dest.index + CartesianIndex(-1, -1))
        i == 2 && return ChessboardQuad(src[compute_index(j, 4)], dest.index + CartesianIndex(-1,  1))
        i == 3 && return ChessboardQuad(src[compute_index(j, 1)], dest.index + CartesianIndex( 1,  1))
        i == 4 && return ChessboardQuad(src[compute_index(j, 2)], dest.index + CartesianIndex( 1, -1))
    end
    nothing
end

function find_connectedquads(image; binary_thresh)::Vector{ChessboardQuad}
    quads = find_contourquads(image; binary_thresh)
    found = false
    groups = []
    local group
    while !isempty(quads)
        if !found
            group = [ChessboardQuad(pop!(quads), CartesianIndex(0, 0))]
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

function paint_foundcorners(image, corners::Matrix{<: Vec{2}})
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
    painted
end

function _find_chessboardcorners(image; binary_thresh)::Matrix
    quads = find_connectedquads(image; binary_thresh)
    i_min, i_max = extrema(quad.index[1] for quad in quads)
    j_min, j_max = extrema(quad.index[2] for quad in quads)
    cornerlayout = CartesianIndices((i_min:i_max+1, j_min:j_max+1))
    corners = OffsetArray([Vec{2, Float64}[] for i in cornerlayout], cornerlayout)
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

function find_chessboardcorners_with_several_binary_thresh(image, thresh_list)
    for binary_thresh in thresh_list
        corners = try
            _find_chessboardcorners(image; binary_thresh)
        catch e
            nothing
        end
        corners !== nothing && return corners
    end
    error("checkbound corners couldn't be found correctly")
end

function find_chessboardcorners(image)
    # Changing threshold for binarization improves detection of chessboard corners in some cases.
    corners = find_chessboardcorners_with_several_binary_thresh(image, (0.45, 0.4, 0.5, 0.3, 0.6, 0.2, 0.7))

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

    corners
end

function Chessboard(image; subpixel::Bool = false)
    corners = find_chessboardcorners(image)
    if subpixel
        corners = map(corners) do corner
            harris_subpixel(image, 0.04, CartesianIndex(Tuple(round.(Int, corner))), 2)
        end
    end
    painted = paint_foundcorners(image, corners)
    Chessboard(painted, corners)
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
