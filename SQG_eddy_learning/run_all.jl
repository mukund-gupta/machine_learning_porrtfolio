using Flux, Statistics
using Flux: logitcrossentropy
using Printf, BSON
using Base.Iterators: partition
using Parameters: @with_kw
using NetCDF
using Random
using JLD2
using Plots.PlotMeasures

include("run_sqg_fine_coarse.jl")
include("learn_sqg.jl")
include("run_sqg_ML.jl")
include("test_sqg_ML.jl")
include("run_sqg_noise.jl")


function get_exp_forcing(x,y,xc,yc,Lx)
    L_scale = 1/(0.03*Lx^2);
    return exp(-L_scale*( (x - xc).^2 + 4*(y - yc).^2));
end

fac = 2
dt = 60
dt_save = 6*3600
t_tot = 3*24*3600
n_epochs = 80
Nx = 32
Ny = Nx
dx = 250
Lx = dx*Nx
Ly = Lx
x = LinRange(-dx/2, Lx - (dx/2), Nx)
y = x'

num = 2 # number corresponding to noise simulation
theta_noise = run_sqg_noise(num);


# Forcing term
Forc_fine = 0*ones(Nx,Nx)
x_center = Lx/2; y_center = x_center
# Forc_fine = get_exp_forcing.(x,y,x_center,y_center,Lx); Forc_fine = 1e-7*Forc_fine/maximum(abs.(Forc_fine))

# Running
run_sqg_fine_coarse(fac,dt_save,t_tot,Forc_fine)
MSE_P = learn_sqg(n_epochs)
MSE_theta_coarse,MSE_theta_ML = run_sqg_ML(fac,dt_save,t_tot,Forc_fine)
MSE_P_add = test_sqg_ML(fac,dt_save,t_tot,Forc_fine)

# Convergence plot for P learnt
MSE_P_plt = plot(xlabel="epochs", ylabel="MSE",title="MSE of P")
plot!(MSE_P,yaxis=:log)
# MSE of theta plot
Nt = length(MSE_theta_coarse)
tt = (1:Nt)*dt_save
MSE_theta_plt = plot(xlabel="t [days]", ylabel="MSE",title="MSE of theta")
plot!(tt/3600/24,MSE_theta_coarse,label="coarse",yaxis=:log)
plot!(tt/3600/24,MSE_theta_ML,label="coarse with ML",yaxis=:log)
# Additional test on P
Nt = length(MSE_P_add)
tt = (1:Nt)*dt
MSE_P_add_plt = plot(xlabel="t [days]", ylabel="MSE",title="MSE of P")
plot!(tt/3600/24,MSE_P_add,yaxis=:log)
# all plots
plot(MSE_P_plt,MSE_theta_plt,MSE_P_add_plt,layout=(1,3),left_margin = 10mm, right_margin = 5mm,top_margin = 5mm, bottom_margin = 5mm,size=(1200,400))
png("../plots/MSE_ML.png")
