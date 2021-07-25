using NetCDF
using Statistics, Printf
using SeawaterPolynomials.TEOS10
using FFTW

# Constructors
mutable struct SQG_model
    c2
    c4
    c8
    K
    K2
    K4
    kx
    ky
    Nx
end

# Functions
function coarsen_xy(fld,fac)
    fld = coarsen_first_dim(fld,fac)
    fld = permutedims(fld, [2, 1])
    fld = coarsen_first_dim(fld,fac)
    out = permutedims(fld, [2, 1])
    return out
end

function coarsen_first_dim(fld,fac)
    newShape = collect(size(fld)) # Collect turns the tuple into an array
    newShape[1] = Int(newShape[1]/fac);
    newShape = [fac; newShape]
    out = reshape(fld,Tuple(newShape))
    out = dropdims(mean(out,dims=1),dims=1)
    return out
end

function timestep_sqg(m,theta,F,dt)
    theta1 = rhs_sqg(m,theta,0)
    theta2 = rhs_sqg(m,theta + 0.5*dt*theta1 , 0.5*dt)
    theta3 = rhs_sqg(m,theta + 0.5*dt*theta2 , 0.5*dt)
    theta4 = rhs_sqg(m,theta + dt*theta3, dt)
    dthetadt = (1/6)*(theta1 + 2*theta2 + 2*theta3 + theta4) + F
    theta += dthetadt*dt
    return theta
end

function timestep_sqg_with_ML(m,theta,F,P,dt)
    theta1 = rhs_sqg(m,theta,0)
    theta2 = rhs_sqg(m,theta + 0.5*dt*theta1 , 0.5*dt)
    theta3 = rhs_sqg(m,theta + 0.5*dt*theta2 , 0.5*dt)
    theta4 = rhs_sqg(m,theta + dt*theta3, dt)
    dthetadt = (1/6)*(theta1 + 2*theta2 + 2*theta3 + theta4) + P + F
    theta += dthetadt*dt
    return theta
end

function rhs_sqg(m,theta,dt)

    theta_hat = fft(reshape(theta, (m.Nx,m.Nx) ))

    # Operations in the Fourier domain
    psi_hat = - theta_hat ./ m.K
    psi_hat[1,1] = 0
    psix_hat = psi_hat .* m.kx * 1im # d(psi)/dx in Fourier
    psiy_hat = psi_hat .* m.ky * 1im # d(psi)/dy in Fourier
    theta_x_hat = theta_hat  .* m.kx * 1im # dw/dx in Fourier
    theta_y_hat = theta_hat  .* m.ky * 1im # dw/dy in Fourier

    psix_hat[:,Int(m.Nx/2)] .= 0
    psiy_hat[Int(m.Nx/2),:] .= 0
    theta_x_hat[:,Int(m.Nx/2)] .= 0
    theta_y_hat[Int(m.Nx/2),:] .= 0

    # Space domain
    v = real(ifft(psix_hat))
    u = -real(ifft(psiy_hat))
    theta_x = real(ifft(theta_x_hat))
    theta_y = real(ifft(theta_y_hat))
    psi = real(ifft(psi_hat))

    rhs = -(theta_x.*u + theta_y.*v) - m.c2*ifft(m.K2.*theta_hat) - m.c4*ifft(m.K4.*theta_hat)
    rhs = real(rhs)

    return rhs

end

function get_adv_term(m,theta,dt)
    # m.c2 = 0
    # m.c4 = 0
    # m.c8 = 0
    theta1 = rhs_sqg(m,theta,0)
    theta2 = rhs_sqg(m,theta + 0.5*dt*theta1 , 0.5*dt)
    theta3 = rhs_sqg(m,theta + 0.5*dt*theta2 , 0.5*dt)
    theta4 = rhs_sqg(m,theta + dt*theta3, dt)
    dthetadt = (1/6)*(theta1 + 2*theta2 + 2*theta3 + theta4)
    return dthetadt
end


function get_flow_field(m,theta)

    w_hat = fft(theta)

    # Operations in the Fourier domain
    psi_hat = - w_hat ./ m.K
    psi_hat[1,1] = 0
    psix_hat = psi_hat  .* m.kx * 1im # d(psi)/dx in Fourier
    psiy_hat = psi_hat .* m.ky * 1im # d(psi)/dy in Fourier
    psix_hat[:,Int(m.Nx/2)] .= 0
    psiy_hat[Int(m.Nx/2),:] .= 0
    vort_hat = psi_hat.*m.K2

    # TODO check signs for u and v
    u = -real(ifft(psiy_hat))
    v = real(ifft(psix_hat))
    psi = real(ifft(psi_hat))
    vort = real(ifft(vort_hat))
    return u,v,psi,vort

end

# function h = circle(x,y,r)
#     d = r*2
#     px = x-r
#     py = y-r
#     h = rectangle('Position',[px py d d],'Curvature',[1,1])
# #     daspect([1,1,1])
# end

# function [rhs,u,v,psi] = rhs_sqg_trunc(theta,dt)
#
#     global c2 c4 c8 X Y K K2 K4 K8 KX KY n
#
#     theta_hat = fft(reshape(theta,n,n))
#
#     # Operations in the Fourier domain
#     psi_hat = - theta_hat ./ K
#     psi_hat(1,1) = 0
#     psix_hat = psi_hat  .* KX * 1i # d(psi)/dx in Fourier
#     psiy_hat = psi_hat .* KY * 1i # d(psi)/dy in Fourier
#     theta_x_hat = theta_hat  .* KX * 1i # dw/dx in Fourier
#     theta_y_hat = theta_hat  .* KY * 1i # dw/dy in Fourier
#
#     psix_hat(:,n/2) = 0
#     psiy_hat(n/2,:) = 0
#     theta_x_hat(:,n/2) = 0
#     theta_y_hat(n/2,:) = 0
#
#     # Space domain
#     v = real(ifft(psix_hat))
#     u = -real(ifft(psiy_hat))
#     theta_x = real(ifft(theta_x_hat))
#     theta_y = real(ifft(theta_y_hat))
#     psi = real(ifft(psi_hat))
#
#     a = AntiAlias_trunc(psix_hat,theta_y_hat)
#     b = AntiAlias_trunc(-psiy_hat,theta_x_hat)
#
#     rhs = -(a + b) - c2*ifft(K2.*theta_hat) - c4*ifft(K4.*theta_hat) - c8*ifft(K8.*theta_hat)
#     rhs = real(rhs)
#
# end
#
# function [rhs,u,v,psi] = rhs_sqg_pad(theta,dt)
#
#     global c2 c4 c8 X Y K K2 KX KY n
#
#     theta_hat = fft(reshape(theta,n,n))
#
#     # Operations in the Fourier domain
#     psi_hat = - theta_hat ./ K
#     psi_hat(1,1) = 0
#     psix_hat = psi_hat  .* KX * 1i # d(psi)/dx in Fourier
#     psiy_hat = psi_hat .* KY * 1i # d(psi)/dy in Fourier
#     theta_x_hat = theta_hat  .* KX * 1i # dw/dx in Fourier
#     theta_y_hat = theta_hat  .* KY * 1i # dw/dy in Fourier
#
#     psix_hat(:,n/2) = 0
#     psiy_hat(n/2,:) = 0
#     theta_x_hat(:,n/2) = 0
#     theta_y_hat(n/2,:) = 0
#
#     # Space domain
#     v = real(ifft(psix_hat))
#     u = -real(ifft(psiy_hat))
#     theta_x = real(ifft(theta_x_hat))
#     theta_y = real(ifft(theta_y_hat))
#     psi = real(ifft(psi_hat))
#
#     a_hat = AntiAlias_pad(psix_hat,theta_y_hat)
#     b_hat = AntiAlias_pad(-psiy_hat,theta_x_hat)
#
#     rhs = -ifft(a_hat + b_hat) - c2*ifft(K2.*theta_hat) - c4*ifft(K.^4.*theta_hat)
#     rhs = real(rhs)
#
# end
#
# function w = AntiAlias_trunc(u_hat, v_hat)
#     N = length(u_hat)
#     k1 = floor(N/3)
#     k2 = k1 + (N/2-k1)*2 + 1
#
#     u_hat_t = u_hat
#     v_hat_t = v_hat
#
#     u_hat_t(1:end,k1:k2) = 0
#     u_hat_t(k1:k2,1:end) = 0
#
#     v_hat_t(1:end,k1:k2) = 0
#     v_hat_t(k1:k2,1:end) = 0
#
#     w = real(ifft(u_hat_t)) .* real(ifft(v_hat_t))
# end
#
# function w_hat = AntiAlias_pad(u_hat, v_hat)
#     N = length(u_hat)
#     M = 3*N/2 # 3/2th rule
#     u_hat_pad = zeros(M,M)
#     v_hat_pad = zeros(M,M)
#
#     i1 = N/2
#     i2 = M-N/2+1
#
#     u_hat_pad(1:i1,1:i1) = u_hat(1:i1,1:i1)
#     v_hat_pad(1:i1,1:i1) = v_hat(1:i1,1:i1)
#
#     u_hat_pad(i2:end,1:i1) = u_hat(i1+1:end,1:i1)
#     v_hat_pad(i2:end,1:i1) = v_hat(i1+1:end,1:i1)
#
#     u_hat_pad(1:i1,i2:end) = u_hat(1:i1,i1+1:end)
#     u_hat_pad(1:i1,i2:end) = u_hat(1:i1,i1+1:end)
#
#     u_hat_pad(i2:end,i2:end) = u_hat(i1+1:end,i1+1:end)
#     v_hat_pad(i2:end,i2:end) = v_hat(i1+1:end,i1+1:end)
#
#     u_pad = ifft(u_hat_pad)
#     v_pad = ifft(v_hat_pad)
#     w_pad = u_pad.*v_pad
#     w_pad_hat = fft(w_pad)
#
#     a = cat(1,w_pad_hat(1:i1,1:i1),w_pad_hat(i2:end,1:i1))
#     b = cat(1,w_pad_hat(1:i1,i2:end),w_pad_hat(i2:end,i2:end))
#     w_hat = cat(2,a,b)
# #     w_hat = 3/2*cat(2,a,b) # TOD: REMOVE ABOVE
#
# end
#
# function dissip_hat = get_dissipation(theta)
#
#     global c2 c4 c8 K2 K4 K8
#
#     theta_hat = fft(theta)
#
#     # Operations in the Fourier domain
#     del2_theta_hat = - theta_hat .* K2
#     del2_theta_hat(1,1) = 0
#     del2_theta = real(ifft(del2_theta_hat))
#
#     del4_theta_hat = - theta_hat .* K4
#     del4_theta_hat(1,1) = 0
#     del4_theta = real(ifft(del4_theta_hat))
#
#     del8_theta_hat = - theta_hat .* K8
#     del8_theta_hat(1,1) = 0
#     del8_theta = real(ifft(del8_theta_hat))
#
#     dissip = c2*theta.*del2_theta + c4*theta.*del4_theta + c8*theta.*del8_theta
#     dissip_hat = fft(dissip)
#
# end
