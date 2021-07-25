using Flux, Statistics
using Flux: logitcrossentropy
using Printf, BSON
using Base.Iterators: partition
using Parameters: @with_kw
using NetCDF
using Random
using JLD2

# Functions

# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = X[:,:,:,idxs]
    Y_batch = Y[:,idxs]
    return (X_batch, Y_batch)
end

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Function to check if any element is NaN or not
anynan(x) = any(isnan.(x))

####################################################################
# Main program

function learn_sqg(epochs)

    # Loading data
    fname = "../data/SQG_fine_coarse.nc"
    X = ncread(fname,"theta_coarsened")
    y = ncread(fname,"P")
    nimages = size(X)[end]
    nx = size(X)[1]
    X = reshape(X,(nx,nx,1,nimages))
    y = Flux.flatten(y)

    # Standarize the variables to zero mean and unit variance,
    mu_X, sigma_X = mean(X), std(X)
    mu_y, sigma_y = mean(y), std(y)
    X = (X .- mu_X) / sigma_X
    y = (y .- mu_y) / sigma_y
    scalings = [mu_X, sigma_X, mu_y, sigma_y]
    @save "../data/scalings.jld2" scalings

    # Split train/test data
    nimages = size(X)[end]
    ntrain = Int(floor(nimages*0.8))
    idx_all = collect(1:nimages)
    idx_all = idx_all[randperm(length(idx_all))]
    train_idx = idx_all[1:ntrain]
    test_idx = idx_all[ntrain+1:end]
    X_train = X[:,:,:,train_idx]
    y_train = y[:,train_idx]
    X_test = X[:,:,:,test_idx]
    y_test = y[:,test_idx]

    # Making training sets with mini-batches
    batch_size = 5
    mb_idxs = partition(1:ntrain, batch_size)
    train_set = [make_minibatch(X_train, y_train, i) for i in mb_idxs]

    # Building the CNN
    CNN_input_size = (16,16,1)
    CNN_output_size = (8,8,8)
    model = Chain(
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    Conv((3, 3), 16=>8, pad=(1,1), relu),
    Conv((3, 3), 8=>8, pad=(1,1), relu),
    MaxPool((2,2)),
    flatten,
    Dense(prod(CNN_output_size), prod(CNN_input_size) ))

    # Training the CNN
    function loss(X, y)
        return (1/length(y))*mean((model(X) .- y).^2)
        #return logitcrossentropy(model(X), y)
    end
    opt = ADAM(3e-3) # optimization

    @info("Beginning training loop...")
    best_MSE = 1e5
    last_improvement = 0
    MSE_all = zeros((epochs))
    for epoch_idx in 1:epochs
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Terminate on NaN
        if anynan(paramvec(model))
            @error "NaN params"
            break
        end

        # Calculate MSE
        MSE = loss(X_test,y_test)

        @info(@sprintf("[%d]: Test MSE: %.9f", epoch_idx, MSE))
        # # If our accuracy is good enough, quit out.
        # if acc >= 0.999
        #     @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        #     break
        # end

        # If this is the best accuracy we've seen so far, save the model out
        if MSE <= best_MSE
            @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
            BSON.@save joinpath("../data/SQG_CNN.bson") params=cpu.(params(model)) epoch_idx MSE
            best_MSE = MSE
            last_improvement = epoch_idx
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 10
            @warn(" -> We're calling this converged.")
            break
        end

        MSE_all[epoch_idx] = MSE
    end

    return MSE_all

end
