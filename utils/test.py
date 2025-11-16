import pymc as pm
import arviz as az
import pickle
import pytensor.tensor as pt
from pytensor.scan import scan
import numpy as np
import matplotlib.pyplot as plt
from pytensor.gradient import disconnected_grad as stopgrad

if __name__ == "__main__":

    data = pickle.load(open("PYMC_input_data/Data2_PYMCinput_CLG.pkl", "rb"))


    with pm.Model(coords = {"participants": np.arange(data["Nparticipants"]), "trials": np.arange(data["Ntrials"]), "getrials": np.arange(data["Nactrials"], data["Ntrials"])}) as model:
        
        r_plus = pt.as_tensor_variable(data["r_plus"], ndim=2)
        r_minus = pt.as_tensor_variable(data["r_minus"], ndim=2)
        k_plus = pt.as_tensor_variable(data["k_plus"], ndim=2)
        k_minus = pt.as_tensor_variable(data["k_minus"], ndim=2)

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        #lambda_sigma = pm.LogNormal("lambda_sigma", mu=np.log(0.3), sigma=0.6)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)

        alpha_mu = pm.Beta("alpha_mu", alpha=1, beta=1)
        #alpha_kappa = pm.LogNormal("alpha_kappa", mu=np.log(4), sigma=0.5)
        alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
        alpha_a = alpha_mu * alpha_kappa
        alpha_b = (1 - alpha_mu) * alpha_kappa
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=3)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)

        # --- anchor θ-center near 5.5, avoid extremes to prevent saturation
        m0 = pm.TruncatedNormal("m0_center_theta", mu=5.5, sigma=0.75, lower=1.2, upper=9.8)

        # inverse of θ = 1 + 9/(1+exp(-x))  →  x = -log(9/(θ-1) - 1)
        w0_mu = pm.Deterministic("w0_mu", -pm.math.log(9.0/(m0 - 1.0) - 1.0))

        # make intercept tight so slope actually controls width
        w0_sigma = pm.HalfNormal("w0_sigma", sigma=0.5)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        # Sigma Means (narrower, stable)
        sigma_mean_learners = pm.Normal("sigma_mean_learners", mu=0.75, sigma=0.2)
        sigma_mean_nonlearners = pm.Normal("sigma_mean_nonlearners", mu=2.25, sigma=0.2)

        # Moderate Scatter (participant-level variance)
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=1.5)
        # Participant-level mean and scatter based on group
        sigma_mu = pm.math.switch(pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners)
        #sigma_sigma = pm.math.switch(pm.math.eq(gp, 0), sigma_scatter_nonlearners, sigma_scatter_learners)

        # Final participant-specific sigmas
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_scatter,
            lower=1e-9,
            upper=3,
            dims="participants"
        )
  
        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants"
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants"
        )

        alpha_1 = pm.Beta(
            "alpha_1",
            alpha=alpha_a, beta=alpha_b, 
            dims="participants"
        )


        # ----- w1 (latent->observed scale) : positive, hierarchical on log scale -----
        # Gamma hierarchy for w1, same spirit, but avoid alpha < 1
        w1_1    = pm.Gamma("w1_1", alpha=w1_a, beta=w1_b, dims="participants")

        #w1_1 = pm.LogNormal("w1_1", mu=0.0, sigma=0.7, dims="participants")  # positive, less crazy tails
        d_sigma = pm.HalfCauchy("d_sigma", beta=2, dims="participants")
        #d_sigma = pm.HalfStudentT("d_sigma", nu=4, sigma=2, dims="participants")
        #sigmazn = pm.math.eq(gp, 0) * sigma1 + pm.math.neq(gp, 0) * sigma0

        # Learning Rate.
        alpha = pm.Deterministic("alpha", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * alpha_1)
        
        # Generalization Rate.
        lambda_ = pm.Deterministic("lambda", pm.math.eq(gp, 0) * 0 + pm.math.eq(gp, 1) * lambda_2 + pm.math.eq(gp, 2) *  lambda_1 + pm.math.eq(gp, 3) *  lambda_1)
        
        # Baseline response.

        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma) 
        #w0 = pm.Normal("w0", mu=w0_mu, sigma=w0_sigma, dims="participants")
        #w0 = pm.Normal("w0", mu=0.0, sigma=3.0, dims="participants")

        # Latent - Observed Scaling.
        w1 = pm.Deterministic("w1", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * w1_1)

        # Initial excitatory associative strength.
        v_plus_init = pt.zeros((1, data["Nparticipants"]), dtype = 'float64')

        # Initial inhibitory associative strength.
        v_minus_init = pt.zeros((1, data["Nparticipants"]), dtype = 'float64')

        # Define the update function for scan.
        def trial_update(
            k_plus_t, r_plus_t, k_minus_t, r_minus_t, v_plus_t, v_minus_t, alpha
        ):
            # Compute trial specific excitatory associative strength.
            v_plus_next = v_plus_t + alpha * (r_plus_t - v_plus_t) * k_plus_t

            # Compute trial specific inhibitory associative strength.
            v_minus_next = v_minus_t + alpha * (r_minus_t - v_minus_t) * k_minus_t
            
            return v_plus_next, v_minus_next

        # Sequential updates across trials.
        [v_plus_seq, v_minus_seq], _ = scan(
            fn=trial_update,
            sequences=[k_plus.T, r_plus.T, k_minus.T, r_minus.T],
            outputs_info=[v_plus_init, v_minus_init],
            non_sequences=alpha[None, :],
        )

        # Excitatory associative strength.
        v_plus = v_plus_seq.squeeze(1).T

        # Inhibitory associative strength.
        v_minus = v_minus_seq.squeeze(1).T

        # Generate missing perceptual distance between CS+ and S.
        d_p_per = pm.TruncatedNormal(
            "d_p_per",
            mu=data["d_p_phy"],
            sigma=d_sigma[:, None],
            lower=0,
            observed=data["d_p_per"]
        )

        # Generate missing perceptual distance between CS- and S.
        d_m_per = pm.TruncatedNormal(
            "d_m_per",
            mu=data["d_m_phy"],
            sigma=d_sigma[:, None],
            lower=0,
            observed=data["d_m_per"]
        )
        
        m_gen = pm.math.ge(gp[:, None], 1)
        m_per = pm.math.eq(gp[:, None], 3)

        # distances chosen by group
        dplus  = m_per * d_p_per  + (1 - m_per) * data["d_p_phy"]
        dminus = m_per * d_m_per  + (1 - m_per) * data["d_m_phy"]

        cond_p = pm.math.gt(stopgrad(v_plus), 0.0)
        cond_m = pm.math.gt(stopgrad(pm.math.abs(v_minus)), 0.0)
        gate_p = cond_p.astype("float64")
        gate_m = cond_m.astype("float64")

        exp_p = pm.math.exp(-lambda_[:, None] * dplus)
        exp_m = pm.math.exp(-lambda_[:, None] * dminus)

        # identical numerically to JAGS: only apply exp when gate==1 *and* gp>=1
        s_plus  = 1.0 + (exp_p - 1.0) * gate_p * m_gen
        s_minus = 1.0 + (exp_m - 1.0) * gate_m * m_gen
                        
        # Generalized associative strength.
        g = v_plus * s_plus + v_minus * s_minus


        # Non-linear transformation (latent - observed scale) with A = 1, K = 10.
        theta = pm.Deterministic("theta", 1 + (10 - 1) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * g))))

        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            observed=data["y"],
            dims=("participants", "trials")
        )
        # Define generalization trials range.
        g_range = np.arange(data["Nactrials"], data["Ntrials"])

        # Prediction for generalization trials.
        y_pre = pm.Normal(
            "y_pre",
            mu=theta[:, g_range],
            sigma=sigma[:, None],
            dims=("participants", "getrials")
        )
        

        prior = pm.sample_prior_predictive(draws=10000)


theta_pp = prior.prior["theta"].values  # (1, draws, P, T)
gp_pp    = prior.prior["gp"].values
print("theta shape:", theta_pp.shape)
print("gp shape:", gp_pp.shape)
mask_learn = (gp_pp[0, :, :] > 0)
gr = np.arange(data["Nactrials"], data["Ntrials"])
theta_gen = theta_pp[0, :, :, gr]
theta_gen_learn = np.where(mask_learn[None, :, :], theta_gen, np.nan)

# per  draw 10/50/90% across ignoring nonlearners
band_low  = np.nanpercentile(theta_gen_learn, 10, axis=(0, 2))
band_med  = np.nanpercentile(theta_gen_learn, 50, axis=(0, 2))
band_high = np.nanpercentile(theta_gen_learn, 90, axis=(0, 2))

print("(median of medians):", np.nanmedian(band_med))
print("10-90% width:", np.nanmedian(band_high - band_low))
parameter_sets = {
    'priors': ["sigma0", "sigma1", "alpha", "lambda", "w1", "w0", "pi"],
    'hyperpriors': ["alpha_mu", "alpha_kappa", "lambda_mu", "lambda_sigma", "w1_a", "w1_b", "w0_mu", "w0_sigma"],
}
sim_y = prior.prior["y_pre"]
for param in parameter_sets["hyperpriors"] + ["sigma0", "sigma1"]:
    samples = prior.prior[param].values.flatten()

    low, high = np.percentile(samples, [1, 99])
    az.plot_dist(prior.prior[param])
    plt.xlim(low, high)
    plt.show()
sim_y_flat = sim_y.values.reshape(-1)
real_y_flat = data["y"][:, 24:180].reshape(-1)
plt.hist(sim_y_flat, bins=50, alpha=0.5, density=True, label="Prior simulated y")
plt.hist(real_y_flat, bins=30, alpha=0.5, density=True, label="real y")
plt.legend()
plt.show()


if __name__ == "__main__":

    data_1 = pickle.load(open("PYMC_input_data/Data1_PYMCinput_CLG.pkl", "rb"))

    with pm.Model(coords = {"participants": np.arange(data_1["Nparticipants"]), "trials": np.arange(data_1["Ntrials"]), "getrials": np.arange(data_1["Nactrials"], data_1["Ntrials"])}) as model:
        
        r = pt.as_tensor_variable(data_1["r"], ndim=2)
        k = pt.as_tensor_variable(data_1["k"], ndim=2)

         # Hyperpriors.
        '''lambda_mu = pm.LogNormal("lambda_mu", mu=np.log(0.04), sigma=0.5)
        lambda_sep = pm.HalfNormal("lambda_sep", sigma=0.01) 
        lambda_sigma_raw = pm.Normal("lambda_sigma_raw", 0.0, 1.0)
        lambda_sigma = pm.Deterministic("lambda_sigma",
                                0.03 * pm.math.log1pexp(lambda_sigma_raw))'''
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        #lambda_sigma = pm.LogNormal("lambda_sigma", mu=np.log(0.3), sigma=0.6)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        '''lambda_sigma = pm.HalfNormal("lambda_sigma", sigma=0.05)
        lambda_mu = pm.TruncatedNormal("lambda_mu", mu=0.06, sigma=0.06, lower=0)'''
        alpha_mu = pm.Beta("alpha_mu", alpha=1, beta=1)
        #alpha_kappa = pm.LogNormal("alpha_kappa", mu=np.log(4), sigma=0.5)
        alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
        alpha_a = alpha_mu * alpha_kappa
        alpha_b = (1 - alpha_mu) * alpha_kappa
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=3)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)

        m0 = pm.TruncatedNormal("m0_center_theta", mu=5.5, sigma=0.75, lower=1.2, upper=9.8)

        w0_mu = pm.Deterministic("w0_mu", -pm.math.log(9.0/(m0 - 1.0) - 1.0))

        w0_sigma = pm.HalfNormal("w0_sigma", sigma=0.5)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        '''# Sigma Means
        sigma_mean_learners = pm.TruncatedNormal("sigma_mean_learners", mu=0.75, sigma=0.1, lower=1e-9, upper=1.5)
        sigma_mean_nonlearners = pm.TruncatedNormal("sigma_mean_nonlearners", mu=2.25, sigma=0.1, lower=1.5, upper=3)

        sigma_scatter_learners = pm.HalfNormal("sigma_scatter_learners", sigma=0.1)
        sigma_scatter_nonlearners = pm.HalfNormal("sigma_scatter_nonlearners", sigma=0.1)

        sigma_mu = pm.math.switch(pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners)
        sigma_sigma = pm.math.switch(pm.math.eq(gp, 0), sigma_scatter_nonlearners, sigma_scatter_learners)

        # Final participant-specific sigmas
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_sigma,
            lower=1e-9,
            upper=3,
            dims="participants"
        )'''
        # Sigma Means
        sigma_mean_learners = pm.Normal("sigma_mean_learners", mu=0.75, sigma=0.2)
        sigma_mean_nonlearners = pm.Normal("sigma_mean_nonlearners", mu=2.25, sigma=0.2)

        #sigma_scatter_nonlearners = pm.HalfStudentT("sigma_scatter_nonlearners", nu=5, sigma=1.5)
        #sigma_scatter_learners = pm.HalfStudentT("sigma_scatter_learners", nu=5, sigma=1.5)
        #sigma_scatter_nonlearners = pm.HalfNormal("sigma_scatter_nonlearners", sigma=1.5)
        #sigma_scatter_learners = pm.HalfNormal("sigma_scatter_learners", sigma=1.5)
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=1.5)
        # Participant-level mean and scatter based on group
        sigma_mu = pm.math.switch(pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners)
        #sigma_sigma = pm.math.switch(pm.math.eq(gp, 0), sigma_scatter_nonlearners, sigma_scatter_learners)

        # Final participant-specific sigmas
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_scatter,
            lower=1e-9,
            upper=3,
            dims="participants"
        )


        '''sigma = pm.Uniform("sigma", lower=[1e-9, 1.5], upper=[1.5, 3], shape=2)'''


        '''
        sigma_mean_learners = pm.Normal("sigma_mean_learners", mu=0.75, sigma=0.2)
        sigma_mean_nonlearners = pm.Normal("sigma_mean_nonlearners", mu=2.25, sigma=0.2)

        #sigma_scatter_nonlearners = pm.HalfStudentT("sigma_scatter_nonlearners", nu=5, sigma=1.5)
        #sigma_scatter_learners = pm.HalfStudentT("sigma_scatter_learners", nu=5, sigma=1.5)
        #sigma_scatter_nonlearners = pm.HalfNormal("sigma_scatter_nonlearners", sigma=1.5)
        #sigma_scatter_learners = pm.HalfNormal("sigma_scatter_learners", sigma=1.5)
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=1.5)
        # Participant-level mean and scatter based on group
        sigma_mu = pm.math.switch(pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners)
        #sigma_sigma = pm.math.switch(pm.math.eq(gp, 0), sigma_scatter_nonlearners, sigma_scatter_learners)

        eps = pm.Normal("eps_sigma", 0, 1, dims="participants")

        # smooth map to (0, 3-ε)
        eps0 = 1e-9
        sigma = pm.Deterministic(
            "sigma",
            (3 - eps0) * pm.math.sigmoid(sigma_mu + sigma_scatter * eps)
        )'''

        '''
        # Response Noise.
        log_sigma_raw = pm.Normal("log_sigma_raw", mu=np.log(1.9), sigma=0.4)

        delta = pm.HalfNormal("delta", sigma=0.30)

        log_sigma = pt.stack([
            log_sigma_raw - delta/2,      # smaller scale
            log_sigma_raw + delta/2       # larger  scale
        ])
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        '''

        '''
        log_sigma0 = pm.Normal("log_sigma0", mu=np.log(2.7), sigma=0.5)
        sigma0     = pm.Deterministic("sigma0", pm.math.exp(log_sigma0))

        log_sigma1 = pm.Normal("log_sigma1", mu=np.log(1.4), sigma=0.5)
        sigma1     = pm.Deterministic("sigma1", pm.math.exp(log_sigma1))
        sigma = pm.Deterministic("sigma", pt.stack([sigma1, sigma0]))'''

        '''
        sigma_mean_learners = pm.TruncatedNormal("sigma_mean_learners", mu=1.0, sigma=0.3, lower=0.5, upper=1.5)
        sigma_mean_nonlearners = pm.TruncatedNormal("sigma_mean_nonlearners", mu=2.0, sigma=0.3, lower=1.5, upper=2.5)
        sigma_sd = pm.HalfNormal("sigma_sd", sigma=0.1)


        sigma = pm.Normal(
            "sigma",
            mu=pt.stack([sigma_mean_nonlearners, sigma_mean_learners]),
            sigma=sigma_sd,
            shape=2
        )'''

        '''sigma = pm.Uniform("sigma", lower=[1e-9, 1.5], upper=[1.5, 3], shape=2)'''
  
        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants"
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants"
        )

        '''L = 0.0052

        lambda_small_mu    = pm.TruncatedNormal("lambda_small_mu", mu=0.0025, sigma=0.0015, lower=0.0, upper=L)
        lambda_small_sigma = pm.HalfNormal("lambda_small_sigma", sigma=0.001)

        lambda_large_mu    = pm.TruncatedNormal("lambda_large_mu", mu=0.08,  sigma=0.03,   lower=L)
        lambda_large_sigma = pm.HalfNormal("lambda_large_sigma", sigma=0.03)

        lambda_2 = pm.TruncatedNormal("lambda_2", mu=lambda_small_mu, sigma=lambda_small_sigma,
                                    lower=0.0, upper=L, dims="participants")
        lambda_1 = pm.TruncatedNormal("lambda_1", mu=lambda_large_mu, sigma=lambda_large_sigma,
                                    lower=L, dims="participants")

        lambda_ = pm.Deterministic(
            "lambda",
            pm.math.eq(gp, 0) * 0.0
            + pm.math.eq(gp, 1) * lambda_2
            + (pm.math.eq(gp, 2) + pm.math.eq(gp, 3)) * lambda_1
        )'''

        '''
        lambda_small_mu  = pm.Normal("lambda_small_mu",  mu=np.log(0.003), sigma=0.5)
        lambda_small_sd  = pm.HalfNormal("lambda_small_sd", sigma=0.4)

        lambda_large_mu  = pm.Normal("lambda_large_mu",  mu=np.log(0.08),  sigma=0.4)
        lambda_large_sd  = pm.HalfNormal("lambda_large_sd", sigma=0.3)

        z_lambda_s = pm.Normal("z_lambda_s", 0.0, 1.0, dims="participants")
        z_lambda_l = pm.Normal("z_lambda_l", 0.0, 1.0, dims="participants")

        lambda_small = pm.Deterministic("lambda_2", pm.math.exp(lambda_small_mu + lambda_small_sd * z_lambda_s))
        lambda_large = pm.Deterministic("lambda_1", pm.math.exp(lambda_large_mu + lambda_large_sd * z_lambda_l))

        lambda_ = pm.Deterministic(
            "lambda",
            pm.math.eq(gp, 0) * 0.0
            + pm.math.eq(gp, 1) * lambda_small
            + (pm.math.eq(gp, 2) + pm.math.eq(gp, 3)) * lambda_large
        )'''
        alpha_1 = pm.Beta(
            "alpha_1",
            alpha=alpha_a, beta=alpha_b, 
            dims="participants"
        )


        '''w1_mu = pm.Normal("w1_mu", mu=np.log(1.5), sigma=0.5)
        w1_sigma = pm.HalfNormal("w1_sigma", sigma=0.5)
        w1raw     = pm.Normal("w1raw", 0.0, 1.0, dims="participants")
        w1_1    = pm.Deterministic("w1_1", pm.math.exp(w1_mu_log + w1_sd_log * w1raw))'''
        '''
        w1_log_mu    = pm.Normal("w1_log_mu", 0.0, 1.0)
        w1_log_sigma = pm.HalfNormal("w1_log_sigma", 0.5)

        w1raw = pm.Normal("w1raw", 0.0, 1.0, dims="participants")
        w1_1   = pm.Deterministic("w1_1", pm.math.exp(w1_log_mu + w1_log_sigma * w1raw))'''
        # Gamma hierarchy for w1, same spirit, but avoid alpha < 1
        w1_1    = pm.Gamma("w1_1", alpha=w1_a, beta=w1_b, dims="participants")

        #w1_1 = pm.LogNormal("w1_1", mu=0.0, sigma=0.7, dims="participants")  # positive, less crazy tails
        d_sigma = pm.HalfCauchy("d_sigma", beta=2, dims="participants")
        #d_sigma = pm.HalfStudentT("d_sigma", nu=4, sigma=2, dims="participants")
        #sigmazn = pm.math.eq(gp, 0) * sigma1 + pm.math.neq(gp, 0) * sigma0

        # Learning Rate.
        alpha = pm.Deterministic("alpha", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * alpha_1)
        
        # Generalization Rate.
        lambda_ = pm.Deterministic("lambda", pm.math.eq(gp, 0) * 0 + pm.math.eq(gp, 1) * lambda_2 + pm.math.eq(gp, 2) *  lambda_1 + pm.math.eq(gp, 3) *  lambda_1)
        
        # Baseline response.
        # ----- w0: hierarchical (centered near 0), non-centered per participant -----
        '''w0_mu = pm.Normal("w0_mu", 0.0, 1.0)
        w0_sigma = pm.HalfNormal("w0_sigma", 0.7)
        w0_1  = pm.Normal("w0_1", 0.0, 1.0, dims="participants")
        w0    = pm.Deterministic("w0", w0_mu + w0_sd * z_w0)'''

        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma) 
        #w0 = pm.Normal("w0", mu=w0_mu, sigma=w0_sigma, dims="participants")
        #w0 = pm.Normal("w0", mu=0.0, sigma=3.0, dims="participants")

        # Latent - Observed Scaling.
        w1 = pm.Deterministic("w1", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * w1_1)
    
        # Initial inhibitory associative strength.
        v_init = pt.zeros((1, data_1["Nparticipants"]), dtype="float64")

        # Define the update function for scan.
        def trial_update(
            k_t, r_t, v_t, alpha, gp
        ):
            # Compute trial specific excitatory associative strength.
            v_next = pm.math.switch(
                pm.math.neq(gp, 0),
                pm.math.switch(
                    pm.math.eq(k_t, 1),
                    v_t + alpha * (r_t - v_t),
                    v_t,
                ),
                0,
            )

            return v_next

        # Sequential updates across trials.
        v_seq, _ = scan(
            fn=trial_update,
            sequences=[k.T, r.T],
            outputs_info=v_init,
            non_sequences=[alpha[None, :], gp[None, :]],
        )

        # Excitatory associative strength.
        v = v_seq.squeeze(1).T

        # Generate missing perceptual distance between CS+ and S.
        d_per = pm.Normal(
            "d_p_per",
            mu=data_1["d_phy"],
            sigma=d_sigma[:, None],
            dims=("participants", "trials")
        )

        m_gen = pm.math.ge(gp[:, None], 1)   # allow generalization for gp ∈ {1,2,3}
        m_per = pm.math.eq(gp[:, None], 3)   # perceptual distances only when gp==3

        # distances chosen by group
        d  = m_per * d_per  + (1 - m_per) * data_1["d_phy"]
        cond = pm.math.gt(stopgrad(v), 0.0)
        gate = cond.astype("float64")

        exp = pm.math.exp(-lambda_[:, None] * d)

        s  = 1.0 + (exp - 1.0) * gate * m_gen

        # Generalized associative strength.
        g = v * s

        # Non-linear transformation (latent - observed scale) with A = 1, K = 10.
        theta = pm.Deterministic("theta", 1 + (9) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * g))))
        
        # Likelihood.
        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            dims=("participants", "trials")
        )
        
        # Define generalization trials range.
        g_range = np.arange(data_1["Nactrials"], data_1["Ntrials"])

        # Prediction for generalization trials.
        y_pre = pm.Normal(
            "y_pre",
            mu=theta[:, g_range],
            sigma=sigma[:, None],
            dims=("participants", "getrials")
        )

        prior_1 = pm.sample_prior_predictive(draws=10000)

theta_pp = prior_1.prior["theta"].values
gp_pp    = prior_1.prior["gp"].values
print("theta shape:", theta_pp.shape)
print("gp shape:", gp_pp.shape)
mask_learn = (gp_pp[0, :, :] > 0)
gr = np.arange(data_1["Nactrials"], data_1["Ntrials"])
theta_gen = theta_pp[0, :, :, gr]
theta_gen_learn = np.where(mask_learn[None, :, :], theta_gen, np.nan)

#10/50/90%
band_low  = np.nanpercentile(theta_gen_learn, 10, axis=(0, 2))
band_med  = np.nanpercentile(theta_gen_learn, 50, axis=(0, 2))
band_high = np.nanpercentile(theta_gen_learn, 90, axis=(0, 2))

print("median of medians:", np.nanmedian(band_med))
print("10-90% width:", np.nanmedian(band_high - band_low))