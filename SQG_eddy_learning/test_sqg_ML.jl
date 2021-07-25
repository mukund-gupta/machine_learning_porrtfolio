using Statistics, Printf
using SeawaterPolynomials.TEOS10
using FFTW
using StructArrays
using Plots
using ImageFiltering
using BSON
using Flux
using JLD2

include("SQG_model.jl")
#include("/Users/guptam/Dropbox (MIT)/MIT/research/work/muri_project/test_ocean/FloeModel/FloeModel.jl")
include("/home/guptam/ocean_floes/FloeModel/FloeModel.jl")

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

function get_CNN_model()
    CNN_input_size = (16,16,1)
    CNN_output_size = (8,8,8)
    CNN_model = Chain(
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    Conv((3, 3), 16=>8, pad=(1,1), relu),
    Conv((3, 3), 8=>8, pad=(1,1), relu),
    MaxPool((2,2)),
    flatten,
    Dense(prod(CNN_output_size), prod(CNN_input_size) ))
    BSON.@load joinpath("../data/SQG_CNN.bson") params # Loading the saved parameters
    Flux.loadparams!(CNN_model, params) # Loading parameters onto the model
    return CNN_model
end

function get_exp_forcing(x,y,xc,yc,Lx)
    L_scale = 1/(0.03*Lx^2);
    return exp(-L_scale*( (x - xc).^2 + 4*(y - yc).^2));
end

add_singleton_dims(x::Array) = reshape(x, (size(x)...,1,1))

################################################################################

function test_sqg_ML(fac,dt_save,t_tot,Forc_fine)

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
    # Forc_fine = get_exp_forcing.(x,y,x_center,y_center,Lx); Forc_fine = 1e-6*Forc_fine/maximum(abs.(Forc_fine))

    # Timestepping
    dt = 60 # [s]
    nit = Int(floor(t_tot/dt))
    nsave = Int(floor(t_tot/dt_save))
    t_save = dt_save:dt_save:t_tot
    dit_save = Int(floor(t_tot/dt_save))

    # Building the CNN and Setting up with learnt parameters
    CNN_model = get_CNN_model()
    @load "../data/scalings.jld2" scalings

    # Running fine model and comparing P vs P_ML
    t = 0
    cnt_save = 1
    MSE_P_all = zeros(Int(floor(t_tot/dt)))
    for it in 1:nit
        # Running SQG model
        t += dt
        theta_fine = timestep_sqg(fine_SQG,theta_fine,Forc_fine,dt)
        theta_coarsened = coarsen_xy(theta_fine,fac)
        P = coarsen_xy(get_adv_term(fine_SQG,theta_fine,dt),fac) - get_adv_term(coarse_SQG,theta_coarsened,dt)
        P_ML = scalings[3] .+ scalings[4]*reshape(CNN_model(add_singleton_dims( (theta_coarsened .- scalings[1])/scalings[2] )),(Nx_coarse,Nx_coarse))
        MSE_P_all[it] = dropdims((1/Nx^4)*mean( (P - P_ML).^2,dims=(1,2)),dims=(1,2))[1]
    end
    return MSE_P_all
end
