# Defining vehicle systems: info needed: Observer matrix, covariance matrix, integration steps, noise

using Distributions  # only required when you're running this file by itself, apart from NGSIM.jl

type VehicleSystem
    H::Matrix{Float64} # observation Jacobian
    R::MvNormal # process noise
    Q::MvNormal # observation noise
    Δt::Float64
    n_integration_steps::Int
    control_noise_accel::Float64
    control_noise_turnrate::Float64

    # the state vector is (x, y, θ, v, δ, a) δ is the control for steering angle a the acceleration
    # NOTE: δ is not exactly the same as presented in Lavalle's Motion Planning book; here δ = (1 / turning radius)
    # tuning the variance!!
    function VehicleSystem(;
        process_noise::Float64 = 0.077,
        observation_noise::Float64 = 16.7,
        control_noise_accel::Float64 = 16.7,
        control_noise_turnrate::Float64 = 0.46
        # control_noise_turnrate::Float64 = 1.0
        )

        Δt = 0.1
        H = [1.0 0.0 0.0 0.0 0.0 0.0;   # state is (x, y, θ, v, w, a), only observing the positions
             0.0 1.0 0.0 0.0 0.0 0.0]
        r = process_noise
        R = MvNormal(diagm([r*0.01, r*0.01, r*0.00001, r*0.1, r*0.1, r*0.1])) # process, TODO: tune this  MvNormal: # multivariate normal distribution with zero mean and covariance C.
        q = observation_noise
        Q = MvNormal(diagm([q, q])) # obs, TODO: tune this   MvNormal: # multivariate normal distribution with zero mean and covariance C.

        n_integration_steps = 10

        new(H, R, Q, Δt, n_integration_steps, control_noise_accel, control_noise_turnrate)
    end
end



draw_proc_noise(ν::VehicleSystem) = rand(ν.R)   # drawing a sample from the random variable R
draw_obs_noise(ν::VehicleSystem) = rand(ν.Q)    # drawing a sample from the random variable Q
get_process_noise_covariance(ν::VehicleSystem) = ν.R.Σ.mat # given VehicleSystem, get the process noise covariance matrix
get_observation_noise_covariance(ν::VehicleSystem) = ν.Q.Σ.mat # given VehicleSystem, get the observation noice covariance matrix

"""
    To derive the covariance of the additional motion noise,
    we first determine the covariance matrix of the noise in control space
    Inputs:
        - ν is the vehicle concrete type
        - u is the control [δ,a]ᵀ
"""

####### change this! 4x4 to account for a and omega
function get_control_noise_in_control_space(ν::VehicleSystem, u::Vector{Float64})
    [ν.control_noise_turnrate 0.0;
     0.0 ν.control_noise_accel]
end

"""
    To derive the covariance of the additional motion noise,
    we transform the covariance of noise in the control space
    by a linear approximation to the derivative of the motion function
    with respect to the motion parameters
    Inputs:
        - ν is the vehicle concrete type
        - u is the control [δ,a]ᵀ
        - x is the state estimate [x,y,θ,v]ᵀ
"""
function get_transform_control_noise_to_state_space(ν::VehicleSystem, u::Vector{Float64}, x::Vector{Float64})
    x, y, θ, v, δ, a = x[1], x[2], x[3], x[4], x[5], x[6]
    b1, b2   = u[1], u[2]
    Δt = ν.Δt
    D = δ + b1
    A = a + b2
    quad = 0.5*A*Δt*Δt + v*Δt # *D + θ

    if abs(D) < 1e-6
        [-quad^2*sin(θ)/2          0.5*Δt*Δt*cos(quad*D+θ);
         quad^2*cos(θ)/2           0.5*Δt*Δt*sin(quad*D+θ);
         quad                          0.5*Δt*Δt*D;
         0.0                              Δt;
         0.0                              0.0;
         0.0                              0.0]
    else
        [1/D^2*(D*quad*cos(quad*D+θ) + sin(θ) - sin(quad*D+θ))                 0.5*Δt*Δt*cos(quad*D+θ);
         1/D^2*(-cos(θ) + cos(quad*D+θ) + D*quad*sin(quad*D+θ))                0.5*Δt*Δt*sin(quad*D+θ);
                        quad                                                   0.5*Δt*Δt*D;
                        0.0                                                       Δt;
                        0.0                                                       0.0;
                        0.0                                                       0.0]
    end
end

"""
    Vehicle dynamics, return the new state
    Inputs:
        - ν is the vehicle concrete type
        - x is the state estimate [x,y,θ,v]ᵀ
        - u is the control [δ,a]ᵀ
"""
function step(ν::VehicleSystem, x::Vector{Float64}, u::Vector{Float64})
    x, y, θ, v, δ, a = x[1], x[2], x[3], x[4], x[5], x[6]
    b1, b2   = u[1], u[2]
    δt = ν.Δt/ν.n_integration_steps
    D = δ + b1
    A = a + b2
    quad = 0.5*A*δt*δt + v*δt # *D + θ


    for i in 1 : ν.n_integration_steps

            if abs(D) < 1e-6 # simulate straight
                x += quad*cos(θ) - quad^2*sin(θ)*D/2   # higher order term probably unnecessary
                y += quad*sin(θ) + quad^2*cos(θ)*D/2
            else # simulate with an arc
                x += 1/D*(sin(quad*D+θ) - sin(θ))             # double checked! I think it is good
                y += 1/D*(cos(θ) - cos(quad*D+θ))
            end

            θ += quad*D
            v += A*δt
            # δ = δ
            # a = a

        end

    [x,y,θ,v,δ,a]
end


"""
    Vehicle observation, returns a saturated observation
    Inputs:
        - ν is the vehicle concrete type
        - x is the state estimate [x,y,θ,v]ᵀ
"""
observe(ν::VehicleSystem, x::Vector{Float64}) = [x[1], x[2]]

"""
    Computes the observation Jacobian (H matrix)
    Inputs:
        - ν is the vehicle concrete type
        - x is the state estimate [x,y,θ,v]ᵀ
"""
compute_observation_jacobian(ν::VehicleSystem, x::Vector{Float64}) = ν.H

"""
    Computes the dynamics Jacobian
    Inputs:
        - ν is the vehicle
        - x is the vehicle state [x,y,θ,v]ᵀ
        - u is the control [δ,a]
"""
function compute_dynamics_jacobian(ν::VehicleSystem, x::Vector{Float64}, u::Vector{Float64})
    x, y, θ, v, δ, a = x[1], x[2], x[3], x[4], x[5], x[6]
    b1, b2   = u[1], u[2]
    Δt = ν.Δt
    D = δ + b1
    A = a + b2
    quad = 0.5*A*Δt*Δt + v*Δt # *D + θ

    if abs(δ + b1) < 1e-6
        # drive straight                                                                                # double checked. I think its ok
        [1.0 0.0   -quad*sin(θ) - quad^2*cos(θ)*D/2   Δt*cos(quad*D+θ)    -quad^2*sin(θ)/2    0.5*Δt*Δt*cos(quad*D+θ);(θ);
         0.0 1.0   quad*cos(θ) - quad^2*sin(θ)*D/2    Δt*sin(quad*D+θ)    quad^2*cos(θ)/2     0.5*Δt*Δt*sin(quad*D+θ);(θ);
         0.0 0.0                 1.0                    Δt*D                   quad                0.5*Δt*Δt*D;
         0.0 0.0                 0.0                    1.0                    0.0                      Δt;
         0.0 0.0                 0.0                    0.0                    1.0                     0.0;
         0.0 0.0                 0.0                    0.0                    0.0                     1.0]
    else                                                                                                # double checked. I think its ok
        # drive in an arc
        [1.0 0.0  1/D*(-cos(θ)+cos(quad*D+θ))     Δt*cos(quad*D+θ)     1/D^2*(quad*D*cos(quad*D+θ)+sin(θ)-sin(quad*D+θ))     0.5*Δt*Δt*cos(quad*D+θ);
         0.0 1.0  1/D*(-sin(θ)+sin(quad*D+θ))     Δt*sin(quad*D+θ)     1/D^2*(-cos(θ)+cos(quad*D+θ)+quad*D*sin(quad*D+θ))    0.5*Δt*Δt*sin(quad*D+θ);
         0.0 0.0            1.0                          Δt*D                                quad                                 0.5*Δt*Δt*D;
         0.0 0.0            0.0                          1.0                                 0.0                                       Δt;
         0.0 0.0            0.0                          0.0                                 1.0                                      0.0;
         0.0 0.0            0.0                          0.0                                 0.0                                      1.0]
    end
end

function EKF(
    ν::VehicleSystem,
    μ::Vector{Float64}, # mean of belief at time t-1
    Σ::Matrix{Float64}, # cov of belief at time t-1
    u::Vector{Float64}, # next applied control
    z::Vector{Float64}, # observation for time t
    )

    G = compute_dynamics_jacobian(ν, μ, u)
    μbar = step(ν, μ, u)    # step outputs [x,y,θ,v]
    R = get_process_noise_covariance(ν)
    M = get_control_noise_in_control_space(ν, u)
    V = get_transform_control_noise_to_state_space(ν, u, μbar)
    # R = V*M*V'
    Σbar = G*Σ*G' + R + V*M*V'
    H = compute_observation_jacobian(ν, μbar)
    K = Σbar * H' / (H*Σbar*H' + get_observation_noise_covariance(ν))  # (K = PH'(HPH'+R)^{-1})
    μ_next = μbar + K*(z - observe(ν, μbar))

    Σ_next = Σbar - K*H*Σbar
    (μ_next, Σ_next)
end

type SimulationResults
    x_arr::Matrix{Float64}
    z_arr::Matrix{Float64}
    u_arr::Matrix{Float64}
    μ_arr::Matrix{Float64}
    Σ_arr::Array{Float64, 3}
end
function simulate(ν::VehicleSystem, nsteps::Int64, x₀::Vector{Float64})
    x_arr = fill(NaN, 6, nsteps+1)
    z_arr = fill(NaN, 2, nsteps)
    u_arr = fill(NaN, 2, nsteps)
    μ_arr = fill(NaN, 6, nsteps+1)
    Σ_arr = fill(NaN, 6, 6, nsteps+1)

    # initial belief
    μ = copy(x₀)
    Σ = copy(ν.R.Σ.mat)

    x_arr[:, 1] = x₀
    μ_arr[:, 1] = μ
    Σ_arr[:, :, 1] = Σ

    x = x₀
    for i in 1 : nsteps

        # move system forward and make observation
        u = [sin(i*0.01), cos(i*0.01)+0.01]                                                 # !!!! WHAT?!?!
        xₚ = step(ν, x, u) + draw_proc_noise(ν)                                             # integration step plus process noise
        z = observe(ν, xₚ) + draw_obs_noise(ν)                                              # observation plus obs noise

        # record trajectories
        x_arr[:,i+1] = xₚ
        z_arr[:,i] = z
        u_arr[:,i] = u

        # apply Kalman filter
        μ_next, Σ_next = EKF(ν, μ, Σ, u, z)
        μ_arr[:,i+1] = μ_next
        Σ_arr[:,:,i+1] = Σ_next

        copy!(x, xₚ)                                                                         # update state for next time step.
        copy!(μ, μ_next)
        copy!(Σ, Σ_next)
    end

    SimulationResults(x_arr, z_arr, u_arr, μ_arr, Σ_arr)
end