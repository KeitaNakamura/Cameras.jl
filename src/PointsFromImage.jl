struct PointsFromImage{T} <: AbstractVector{Vec{2, T}}
    signal::Signal
    data::Vector{Vec{2, T}}
end

Base.size(x::PointsFromImage) = size(x.data)

@inline function Base.getindex(x::PointsFromImage, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.data[i]
end

Base.empty!(x::PointsFromImage) = (empty!(x.data); x)

"""
    PointsFromImage(image::AbstractArray)

Read coordinates from `image` by clicking points on `image`.
"""
function PointsFromImage(img::AbstractArray)
    c = imshow(img)["gui"]["canvas"]
    output = Vec{2, Float64}[]
    signal = map(c.mouse.buttonpress) do btn
        btn.button == 1 && push!(output, Vec(btn.position.y, btn.position.x))
        nothing
    end
    PointsFromImage(signal, output)
end
