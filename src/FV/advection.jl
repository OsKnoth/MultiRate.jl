# Upwind values of advected quantities at the flux points: at every face a left and a right value
# from the third-order (κ = 1/3) reconstruction of the cell averages, then the upwind one.

"""Third-order (κ = 1/3) value at the right face of cell 0 from cells -1, 0, 1 (left state)."""
@inline recon_left(qm1, q0, qp1) = (-qm1 + 5 * q0 + 2 * qp1) / 6

"""Third-order (κ = 1/3) value at the left face of cell 0 from cells -1, 0, 1 (right state)."""
@inline recon_right(qm1, q0, qp1) = (2 * qm1 + 5 * q0 - qp1) / 6

"""Upwind value at the face between qm1 and q0 (stencil qm2, qm1, q0, qp1): the left state
`recon_left(qm2, qm1, q0)` for v ≥ 0, the right state `recon_right(qm1, q0, qp1)` for v < 0
(the sign bit of v decides, so v = -0.0 takes the right state)."""
@inline interp3(qm2, qm1, q0, qp1, v) = signbit(v) ? recon_right(qm1, q0, qp1) : recon_left(qm2, qm1, q0)

"""Second-order (centred) value between qm1 and q0."""
@inline interp2(qm1, q0) = 0.5 * (qm1 + q0)
