struct PointsFromImage{T} <: AbstractVector{SVector{2, T}}
    signal::Signal
    data::Vector{SVector{2, T}}
end

Base.size(x::PointsFromImage) = size(x.data)

@inline function Base.getindex(x::PointsFromImage, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.data[i]
end

Base.empty!(x::PointsFromImage) = empty!(x.data)

"""
    PointsFromImage(image::AbstractArray)

Read coordinates from `image` by clicking points on `image`.
"""
function PointsFromImage(img::AbstractArray)
    c = imshow(img)["gui"]["canvas"]
    output = SVector{2, Float64}[]
    signal = map(c.mouse.buttonpress) do btn
        btn.button == 1 && push!(output, SVector(btn.position.y, btn.position.x))
        nothing
    end
    PointsFromImage(signal, output)
end
