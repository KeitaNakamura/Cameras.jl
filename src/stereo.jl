function stereo(cameraL::Camera, cameraR::Camera, xL::AbstractVector, xR::AbstractVector)
    @assert length(xL) == length(xR) == 2
    pL = projection_matrix(cameraL)
    pR = projection_matrix(cameraR)
    A = @SMatrix [pL[3,1]*xL[1]-pL[1,1] pL[3,2]*xL[1]-pL[1,2] pL[3,3]*xL[1]-pL[1,3]
                  pL[3,1]*xL[2]-pL[2,1] pL[3,2]*xL[2]-pL[2,2] pL[3,3]*xL[2]-pL[2,3]
                  pR[3,1]*xR[1]-pR[1,1] pR[3,2]*xR[1]-pR[1,2] pR[3,3]*xR[1]-pR[1,3]
                  pR[3,1]*xR[2]-pR[2,1] pR[3,2]*xR[2]-pR[2,2] pR[3,3]*xR[2]-pR[2,3]]
    b = @SVector [pL[1,4]-pL[3,4]*xL[1],
                  pL[2,4]-pL[3,4]*xL[2],
                  pR[1,4]-pR[3,4]*xR[1],
                  pR[2,4]-pR[3,4]*xR[2]]
    A \ b
end
