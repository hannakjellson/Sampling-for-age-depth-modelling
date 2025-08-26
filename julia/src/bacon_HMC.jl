include("define_data_and_variables.jl")
using JLD2

function mean_ages(sed_rates, config, data, indices)
    cumulative_sum = vcat(0.0, Base.cumsum(sed_rates))
    return data.theta .- cumulative_sum[indices] .* config.delta_c .- sed_rates[indices] .* (data.age_depths .- config.cs[indices])
end

function energy_function(log_sed_rates, config, data, indices)
    sed_rates = exp.(log_sed_rates)
    prior = sum(-config.a * log_sed_rates .+ config.b .* sed_rates)
    conditional = sum((data.ages .- mean_ages(sed_rates, config, data, indices)) .^ 2 ./ (2 .* data.sigma_y .^ 2))
    return prior + conditional
end

function grad_energy(log_sed_rates, config, data, indices)
    ForwardDiff.gradient(s -> energy_function(s, config, data, indices), log_sed_rates)
end

function run_hmc()
    data = get_data()
    config = get_hmc_config()
    indices = searchsortedlast.(Ref(config.cs), data.age_depths)

    M_inv = Diagonal(1 ./ diag(config.M)) # Works only for a diagonal matrix.
    points = SharedArray{Float64}(config.num_chains, config.num_MH, config.N)
    prog = Progress(config.num_chains * config.num_MH, desc="Running HMC")
    my_lock = ReentrantLock()

    @threads for k in 1:config.num_chains
        variables = rand(Gamma(config.a, 1 / config.b), config.N)
        log_variables = log.(variables)
        mean_acceptance = 0

        for j in 1:config.num_MH
            momentum = rand(MvNormal(zeros(config.N), config.M))
            momentum_init = copy(momentum)
            log_variables_init = copy(log_variables)

            for _ in 1:config.L
                potential_grad = grad_energy(log_variables, config, data, indices)
                momentum .-= (config.dt / 2) .* potential_grad
                log_variables .+= config.dt .* (M_inv * momentum)
                potential_grad = grad_energy(log_variables, config, data, indices)
                momentum .-= (config.dt / 2) .* potential_grad
            end

            logp_new = energy_function(log_variables, config, data, indices)
            logp_old = energy_function(log_variables_init, config, data, indices)
            kinetic_new = 0.5 * dot(momentum, M_inv * momentum)
            kinetic_old = 0.5 * dot(momentum_init, M_inv * momentum_init)

            acceptance_prob = min(1.0, exp(logp_old + kinetic_old - logp_new - kinetic_new))
            mean_acceptance += acceptance_prob
            if rand() < acceptance_prob
                points[k, j, :] = exp.(log_variables)
            else
                points[k, j, :] = exp.(log_variables_init)
            end

            lock(my_lock) do
                next!(prog)
            end
        end
        mean_acceptance /= config.num_MH
        println(mean_acceptance)
    end
    valid_samples = points[:, Int(config.num_MH / 2):end, :]

    file_path = joinpath(@__DIR__, "..", "..", "output", "samples.jld2")
    @save file_path valid_samples
end


function plot_age_depth()
    file_path = joinpath(@__DIR__, "..", "..", "output", "samples.jld2")
    @load file_path valid_samples

    data = get_data()
    config = get_hmc_config()

    # Compute age offsets
    cumulative_sums = cumsum(valid_samples, dims=3) .* config.delta_c
    zeros_vec = zeros(size(valid_samples, 1), size(valid_samples, 2), 1)
    age_offsets = cat(zeros_vec, cumulative_sums; dims=3)
    model_ages = data.theta .- age_offsets

    # Mean model age across chains and samples
    mean_model_age = mean(model_ages, dims=(1, 2)) |> x -> dropdims(x, dims=(1, 2))

    # Set up plot colors
    colors = distinguishable_colors(config.num_chains)

    # # Plot a diagnostic variable (e.g., variable 5 across samples) for each chain
    # p1 = plot(size=(600, 400))
    # for i in 1:config.num_chains
    #     plot!(p1, valid_samples[i, :, 5], label="chain $i", color=colors[i])
    # end
    # display(p1)

    # Plot model age realizations
    p2 = plot(size=(600, 400))
    num_samples = Int(config.num_MH ÷ 2)
    for i in 1:config.num_chains
        for j in 1:num_samples
            plot!(p2, config.cs, vec(model_ages[i, j, :]), color=colors[i], alpha=0.3, lw=0.1, label=false)
        end
    end

    plot!(p2, config.cs, mean_model_age, color=:black, alpha=0.3, lw=1, label="Mean model age")
    scatter!(p2, data.age_depths, data.ages, color=:black, marker=:circle, label="Data", markersize=4)
    xlims!(p2, 0, maximum(config.cs))
    ylims!(p2, 1200, 2000)
    xlabel!("Depth")
    ylabel!("Age")
    display(p2)

    # valid_samples = valid_samples[:, Int(config.num_MH/2):end, :]
    # num_chains, num_samples_per_chain, N = size(valid_samples)

    # plot()
    # for chain_idx in 1:num_chains
    #     chain_samples = reshape(valid_samples[chain_idx, :, :], num_samples_per_chain, N)
    #     model_ages = data.theta .- hcat(zeros(num_samples_per_chain), cumsum(chain_samples, dims=2) .* config.delta_c)
    #     for i in 1:num_samples_per_chain
    #         plot!(config.cs, model_ages[i, :], alpha=0.3, linewidth=1, legend=false, label="", color=chain_idx)
    #     end
    # end

    # mean_model_age = mean(reshape(valid_samples, :, N), dims=1)
    # xlims!(0, maximum(config.cs))
    # ylims!(1200, 2000)
    # plot!(config.cs, mean_model_age[:], color=:black, alpha=0.6, linewidth=2, label="Mean model age")
    # scatter!(data.age_depths, data.ages, label="Data", color=:black)
    # xlabel!("Depth")
    # ylabel!("Age")

    # valid_samples = reshape(valid_samples, :, size(valid_samples, 3))
    # model_ages = data.theta .- hcat(zeros(size(valid_samples, 1)), cumsum(valid_samples, dims=2) .* config.delta_c)
    # mean_model_age = mean(model_ages, dims=1)
    # println(valid_samples)

    # plot()
    # for i in axes(model_ages, 1)
    #     plot!(config.cs, model_ages[i, :], color=:gray, alpha=0.3, linewidth=1, legend=false)
    # end
    # xlims!(0, maximum(config.cs))
    # ylims!(1200, 2000)
    # plot!(config.cs, mean_model_age[:], color=:black, alpha=0.3, linewidth=1, legend=false)
    # scatter!(data.age_depths, data.ages, label="Data", color=:black)
    # xlabel!("Depth")
    # ylabel!("Age")

end