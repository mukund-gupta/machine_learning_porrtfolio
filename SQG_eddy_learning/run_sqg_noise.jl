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

function run_sqg_noise(num)

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

    # Initial conditions
    theta_fine = 1e-1*rand(Nx,Nx)
    Forc_fine = zeros(Nx,Nx)

    # Timestepping
    dt = 60 # [s]
    t_tot = 3600*24*2
    nit = Int(floor(t_tot/dt))

    ## Writer
    fname = "../data/SQG_noise" * string(num) * ".nc"
    isfile(fname) && rm(fname)
    nccreate(fname,"theta","x",x,"y",x)

    # Running fine model and getting P
    t = 0
    cnt_save = 1
    for it in 1:nit
        global t, theta_fine, Forc_fine

        # Running SQG model
        t += dt
        theta_fine = timestep_sqg(fine_SQG,theta_fine,Forc_fine,dt)

        # # Saving
        # if t%dt_save == 0
        #     theta_fine_all[:,:,cnt_save] = theta_fine
        #     theta_coarsened_all[:,:,cnt_save] = theta_coarsened
        #     P_all[:,:,cnt_save] = P
        #     cnt_save += 1
        # end
    end

    # Saving data
    ncwrite(theta_fine,fname,"theta")
    println(fname)

    # theta_fine_plt = contour(x/1000, x/1000, theta_fine', xlabel="x [km]", ylabel="y [km]",title="theta (fine)",linewidth = 0, fill=true,color=:haline)
    # png("../plots/theta_noise2.png")

    # # Making movie
    # nsave = size(theta_fine_all)[3]
    # anim = @animate for it=1:nsave
    #     theta_fine_plt = contour(x/1000, x/1000, theta_fine_all[:,:,it]', xlabel="x [km]", ylabel="y [km]",title="theta (fine)",linewidth = 0, fill=true,color=:haline) #,clims=(Tmin, Tmax))
    #     theta_coarsened_plt = contour(x_coarse/1000, x_coarse/1000, theta_coarsened_all[:,:,it]', xlabel="x [km]", ylabel="y [km]",title="theta (coarsened)",linewidth = 0, fill=true,color=:haline) #,clims=(Tmin, Tmax))
    #     P_plt = contour(x_coarse/1000, x_coarse/1000,dt*P_all[:,:,it]', xlabel="x [km]", ylabel="y [km]",title="Pxdt",linewidth = 0, fill=true,color=:haline) #,clims=(Tmin, Tmax))
    #     plot(theta_fine_plt,theta_coarsened_plt,P_plt,layout = (1,3),size=(1200, 300))
    # end
    # mp4(anim, "../videos/theta_fine_coarse.mp4", fps = 5) # hide

    return theta_fine
end
