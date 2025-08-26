function read_data(file::String)
    df = CSV.read(file, DataFrame, delim='\t')
    depths = Float64.(df.depth)
    ages = Float64.(df.cal_c14_age)
    true_ages = Float64.(df.true_age)
    sigma = Float64.(df.sigma_age)
    return depths, ages, sigma, true_ages
end

function get_data()
    full_path = joinpath(@__DIR__, "..", "..", "data/Dayu cave.txt")
    depths, ages, sigma_age, true_ages = read_data(full_path)
    depths = depths[2:9]
    ages = ages[2:9]
    sigma_age = sigma_age[2:9]
    return (
        theta=true_ages[1],
        ages=ages,
        age_depths=depths,
        sigma_y=sigma_age,
        num_depths=length(depths)
    )
end

function get_hmc_config()
    N = 50
    H = 100.0
    delta_c = H / N
    cs = range(0, stop=H, length=N + 1)
    return (
        N=N,
        H=H,
        delta_c=delta_c,
        cs=collect(cs),
        dt=0.005,
        M=1.3 * Matrix{Float64}(I, N, N),
        L=500,
        num_MH=2000,
        num_chains=4,
        a=1.5,
        b=0.21
    )
end