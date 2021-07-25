using Statistics, Printf
using SeawaterPolynomials.TEOS10
using FFTW
using StructArrays
using Plots
using ImageFiltering

include("SQG_model.jl")
#include("/Users/guptam/Dropbox (MIT)/MIT/research/work/muri_project/test_ocean/FloeModel/FloeModel.jl")
#include("/home/guptam/ocean_floes/FloeModel/FloeModel.jl")

# Functions
function gaussian_degrade(fld,fac)
    n = size(fld)[1]
    tmp = [fld fld fld]
    tmp2 = [tmp; tmp; tmp]
    tmp3 = imfilter(tmp2, Kernel.gaussian(fac))
    vect = n+1:2*n
    tmp4 = tmp3[vect,vect]
    return tmp4
end

function get_exp_forcing(x,y,xc,yc,Lx)
    L_scale = 1/(0.03*Lx^2);
    return exp(-L_scale*( (x - xc).^2 + 4*(y - yc).^2));
end

################################################################################
# Main program

function run_sqg_fine_coarse(fac,dt_save,t_tot,Forc_fine)

    # Constants
    g = 9.81
    Ω = 1/(24*3600)
    c2 = 0 # diffusion strength
    c4 = 1e4
    c8 = 0

    # Setting up fine SQG model
    Nx = 32
    Ny = Nx
    dx = 250
    Lx = dx*Nx
    Ly = Lx
    x = LinRange(-dx/2, Lx - (dx/2), Nx)
    y = x'
    kx = (2*π/Lx)*[0:(Nx/2)-1; -Nx/2:-1]
    ky = kx'
    # filter = [ones(1,floor(n/3)) zeros(1,floor(n/3)) ones(1,n - 2*floor(n/3))]
    # kx = kx.*filter
    # ky = ky.*filter
    K = sqrt.(kx.^2 .+ ky.^2)
    K2 = kx.^2 .+ ky.^2
    K4 = kx.^4 .+ ky.^4
    K8 = kx.^8 .+ ky.^8
    fine_SQG = SQG_model(c2,c4,c8,K,K2,K4,kx,ky,Nx)

    # Setting up coarse SQG model
    x_coarse = coarsen_first_dim(x,fac)
    Nx_coarse = Int(Nx/fac)
    Ny_coarse = Nx_coarse
    kx_coarse = (2*π/Lx)*[0:(Nx_coarse/2)-1; -Nx_coarse/2:-1]
    ky_coarse = kx_coarse'
    K_coarse = sqrt.(kx_coarse.^2 .+ ky_coarse.^2)
    K2_coarse = kx_coarse.^2 .+ ky_coarse.^2
    K4_coarse = kx_coarse.^4 .+ ky_coarse.^4
    K8_coarse = kx_coarse.^8 .+ ky_coarse.^8
    coarse_SQG = SQG_model(c2,c4,c8,K_coarse,K2_coarse,K4_coarse,kx_coarse,ky_coarse,Nx_coarse)

    # Initial conditions
    theta_fine = ncread("../data/SQG_noise_start.nc", "theta")[:,:,end]

    # # Forcing term
    # #Forc_fine = 0*ones(Nx,Nx)
    # x_center = Lx/2; y_center = x_center
    # Forc_fine = get_exp_forcing.(x,y,x_center,y_center,Lx); Forc_fine = 5e-7*Forc_fine/maximum(abs.(Forc_fine))

    # Timestepping
    dt = 60 # [s]
    nit = Int(floor(t_tot/dt))
    nsave = Int(floor(t_tot/dt_save))
    t_save = dt_save:dt_save:t_tot
    dit_save = Int(floor(t_tot/dt_save))

    ## Writer
    fname = joinpath("../data","SQG_fine_coarse.nc")
    isfile(fname) && rm(fname)
    nccreate(fname,"theta_coarsened","x_coarse",x_coarse,"y_coarse",x_coarse,"t",t_save)
    nccreate(fname,"theta_fine","x",x,"y",x,"t",t_save)
    nccreate(fname,"P","x_coarse",x_coarse,"y_coarse",x_coarse,"t",t_save)
    theta_fine_all = zeros((Nx,Ny,nsave))
    theta_coarsened_all = zeros((Nx_coarse,Ny_coarse,nsave))
    P_all = zeros((Nx_coarse,Ny_coarse,nsave))

    # Running fine model and getting P
    t = 0
    cnt_save = 1
    for it in 1:nit

        # Running SQG model
        t += dt
        theta_fine = timestep_sqg(fine_SQG,theta_fine,Forc_fine,dt)
        theta_coarsened = coarsen_xy(theta_fine,fac)
        # Note that I am acutually the full RHS here, including diffusive term. Doesn't work as well otherwise
        P = coarsen_xy(get_adv_term(fine_SQG,theta_fine,dt),fac) - get_adv_term(coarse_SQG,theta_coarsened,dt)

        # Saving
        if t%dt_save == 0
            theta_fine_all[:,:,cnt_save] = theta_fine
            theta_coarsened_all[:,:,cnt_save] = theta_coarsened
            P_all[:,:,cnt_save] = P
            cnt_save += 1
        end
    end

    # Saving data
    ncwrite(theta_fine_all,fname,"theta_fine")
    ncwrite(P_all,fname,"P")
    ncwrite(theta_coarsened_all,fname,"theta_coarsened")

    # Making movie
    nsave = size(theta_fine_all)[3]
    anim = @animate for it=1:nsave
        theta_fine_plt = contour(x/1000, x/1000, theta_fine_all[:,:,it]', xlabel="x [km]", ylabel="y [km]",title="theta (fine)",linewidth = 0, fill=true,color=:haline) #,clims=(Tmin, Tmax))
        theta_coarsened_plt = contour(x_coarse/1000, x_coarse/1000, theta_coarsened_all[:,:,it]', xlabel="x [km]", ylabel="y [km]",title="theta (coarsened)",linewidth = 0, fill=true,color=:haline) #,clims=(Tmin, Tmax))
        P_plt = contour(x_coarse/1000, x_coarse/1000,dt*P_all[:,:,it]', xlabel="x [km]", ylabel="y [km]",title="Pxdt",linewidth = 0, fill=true,color=:haline) #,clims=(Tmin, Tmax))
        plot(theta_fine_plt,theta_coarsened_plt,P_plt,layout = (1,3),size=(1200, 300))
    end
    mp4(anim, "../videos/theta_fine_coarse.mp4", fps = 5) # hide
end
