"""
Module: models_definitions.py

### Context and Source
This module is inspired by the project associated with the study:

"Humans display interindividual differences in the latent mechanisms underlying
fear generalization behaviour."

Contributors: Kenny Yu, Francis Tuerlinckx, Wolf Vanpaemel, Jonas Zaman
Affiliated institutions: KU Leuven
Identifier: DOI 10.17605/OSF.IO/SXJAK

The original R code and the JAGS models, created as part of
the study, are available at the project repository:
https://osf.io/sxjak/

The experimental study utilizes the datasets of two other studies:
experiment 1 simple conditioning (https://osf.io/b4ngs),
exepriment 2 differential conditioning (https://osf.io/t4bzs)
These studies were conducted by one of the contributors: Jonas Zaman.

### About This Module
This Python module primarily converts the original JAGS models used in the study
to PyMC, trying to leverage PyMC's capabilities for
Bayesian inference, thereby ensuring compatibility and reproducibility with
Python-based data analysis workflows.

### Models:
- `model_1v_G2D`: simplified model with single CS and three latent groups
   (overgeneralizers, physical generalizers, perceptual generalizers).
- `model_1v_LG2D`: full model with single CS and four latent groups (non-learners,
   overgeneralizers, physical generalizers, perceptual generalizers).
- `model_1v_LGPHY`: simplified model with single CS and three latent groups
   (non-learners, overgeneralizers, physical generalizers).
- `model_2v_G2D`: simplified model with two CSs and three latent groups
   (overgeneralizers, physical generalizers, perceptual generalizers).
- `model_2v_LG2D`: full model with two CSs and four latent groups (non-learners,
   overgeneralizers, physical generalizers, perceptual generalizers).
- `model_2v_LGPHY`: simplified model with two CSs and three latent groups
   (non-learners, overgeneralizers, physical generalizers).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan

def model_1v_G2D(data):
    """
    Define a PyMC model for single-CS generalization with two distances (G2D).

    This model represents a probabilistic framework for fear generalization
    using one conditioned stimulus (CS+). Participants are grouped into three
    latent categories:
    - Group 1: Overgeneralizers (lambda = 0).
    - Group 2: Physical generalizers (lambda > 0, depends on physical distances).
    - Group 3: Perceptual generalizers (lambda > 0, depends on perceptual distances).

    The model includes the following main parameters:
    - `pi` (0 - 1): Group probabilities for the three latent groups.
    - `lambda` (0 - ∞): Generalization rate.
    - `w0` (-∞ - ∞): Baseline response.
    - `w1` (-∞ - ∞): Latent-observed scaling factor.
    - `sigma` (0 - ∞): Response noise, reflecting variability in observed responses.

    Args:
        data (dict): A dictionary containing:
            - 'd_phy' (ndarray): Physical distances for CS+ for each participant/trial.
            - 'Nparticipants' (int): Number of participants.
            - 'y' (ndarray): Observed responses (US expectancy).

    Returns:
        pm.Model: A PyMC model object representing the statistical structure.
    """
    with pm.Model(
        coords={
            "participants": np.arange(data["Nparticipants"]),
            "getrials": np.arange(data["Nactrials"], data["Ntrials"]),
        }
    ) as model:

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=2)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)
        sigma_a = pm.HalfStudentT("sigma_a", nu=2, sigma=2)
        sigma_b = pm.HalfStudentT("sigma_b", nu=2, sigma=2)
        w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
        w0_sigma = pm.HalfStudentT("w0_sigma", nu=2, sigma=2)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants",
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants",
        )
        d_sigma = pm.HalfCauchy("d_sigma", beta=2, dims="participants")

        # Generalization Rate.
        lambda_ = pm.Deterministic(
            "lambda", pm.math.eq(gp, 0) * lambda_2 + pm.math.neq(gp, 0) * lambda_1
        )

        # Baseline Response
        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma)

        # Latent - Observed Scaling.
        w1 = pm.Gamma("w1", alpha=w1_a, beta=w1_b, dims="participants")

        # Response Noise
        sigma = pm.Gamma("sigma", alpha=sigma_a, beta=sigma_b, dims="participants")

        # Generate missing perceptual distance between CS+ and S.
        d_per = pm.Normal(
            "d_per",
            mu=data["d_phy"],
            sigma=d_sigma[:, None],
            observed=data["d_per"],
            dims=("participants", "getrials"),
        )

        # Perceptual distances only when gp=2.
        m_per = pm.math.eq(gp[:, None], 2)

        # Distances chosen by group.
        d = m_per * d_per + (1 - m_per) * data["d_phy"]

        # Trials stimulus similarity to CS+.
        s = pm.math.exp(-lambda_[:, None] * d)

        # Non-linear transfromation (latent - observed scale) with A = 1, K = 10.
        theta = 1 + (10 - 1) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * s)))

        # Likelihood.
        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            observed=data["y"],
            dims=("participants", "getrials"),
        )

        # Prediction.
        y_pre = pm.Normal(
            "y_pre", mu=theta, sigma=sigma[:, None], dims=("participants", "getrials")
        )

        return model


def model_1v_LG2D(data):
    """
    Define a PyMC model for single-CS learning and generalization with two distances
    (LG2D).

    This model incorporates a learning rate (`alpha`) and groups participants
    into four latent categories:
    - Group 1: Non-learners (alpha = 0, lambda = 0).
    - Group 2: Overgeneralizers (lambda = 0).
    - Group 3: Physical generalizers (lambda > 0, depends on physical distances).
    - Group 4: Perceptual generalizers (lambda > 0, depends on perceptual distances).

    The model includes the following main parameters:
    - `sigma` (0 - ∞): Response noise, reflecting variability in observed responses.
    - `pi` (0 - 1): Group probabilities for the four latent groups.
    - `alpha` (0 - 1): Learning rate.
    - `lambda` (0 - ∞): Generalization rate.
    - `w0` (-∞ - ∞): Baseline response.
    - `w1` (0 - ∞): Latent-observed scaling factor.

    Args:
        data (dict): A dictionary containing:
            - 'd_phy' (ndarray): Physical distances for each participant/trial.
            - 'r' (ndarray): Reinforcement data.
            - 'k' (ndarray): CS indicator data.
            - 'Nparticipants' (int): Number of participants.
            - 'y' (ndarray): Observed responses (US expectancy).
            - 'Nactrials' (int): Number of active trials.
            - 'Ntrials' (int): Total number of trials.

    Returns:
        pm.Model: A PyMC model object representing the statistical structure.
    """
    with pm.Model(
        coords={
            "participants": np.arange(data["Nparticipants"]),
            "trials": np.arange(data["Ntrials"]),
            "getrials": np.arange(data["Nactrials"], data["Ntrials"]),
        }
    ) as model:

        r = pt.as_tensor_variable(data["r"], ndim=2)
        k = pt.as_tensor_variable(data["k"], ndim=2)

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        alpha_mu = pm.Beta("alpha_mu", alpha=1, beta=1)
        alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
        alpha_a = alpha_mu * alpha_kappa
        alpha_b = (1 - alpha_mu) * alpha_kappa
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=2)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)
        w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
        w0_sigma = pm.HalfStudentT("w0_sigma", nu=2, sigma=2)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        # Sigma means.
        sigma_mean_learners = pm.Normal("sigma_mean_learners", mu=0.75, sigma=0.2)
        sigma_mean_nonlearners = pm.Normal("sigma_mean_nonlearners", mu=2.25, sigma=0.2)

        # Participant-level mean based on group.
        sigma_mu = pm.math.switch(
            pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners
        )

        # Participant-level sigma variance.
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=1.5)

        # Final participant-level sigmas.
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_scatter,
            lower=1e-9,
            upper=3,
            dims="participants",
        )

        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants",
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants",
        )
        alpha_1 = pm.Beta("alpha_1", alpha=alpha_a, beta=alpha_b, dims="participants")
        w1_1 = pm.Gamma("w1_1", alpha=w1_a, beta=w1_b, dims="participants")
        d_sigma = pm.HalfCauchy("d_sigma", beta=2, dims="participants")

        # Learning Rate.
        alpha = pm.Deterministic(
            "alpha", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * alpha_1
        )

        # Generalization Rate.
        lambda_ = pm.Deterministic(
            "lambda",
            pm.math.eq(gp, 0) * 0
            + pm.math.eq(gp, 1) * lambda_2
            + pm.math.eq(gp, 2) * lambda_1
            + pm.math.eq(gp, 3) * lambda_1,
        )

        # Baseline response.
        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma)

        # Latent - Observed Scaling.
        w1 = pm.Deterministic("w1", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * w1_1)

        # Initial inhibitory associative strength.
        v_init = pt.zeros((1, data["Nparticipants"]), dtype="float64")

        # Update function for scan.
        def trial_update(k_t, r_t, v_t, alpha):
            # Compute trial specific excitatory associative strength.
            v_next = v_t + alpha * (r_t - v_t) * k_t

            return v_next

        # Sequential updates across trials.
        v_seq, _ = scan(
            fn=trial_update,
            sequences=[k.T, r.T],
            outputs_info=v_init,
            non_sequences=[alpha[None, :]],
        )

        # Excitatory associative strength.
        v = v_seq.squeeze(1).T

        # Generate missing perceptual distance between CS+ and S.
        d_per = pm.Normal(
            "d_p_per",
            mu=data["d_phy"],
            sigma=d_sigma[:, None],
            observed=data["d_per"],
            dims=("participants", "trials"),
        )

        # Allow generalization for gp {1,2,3}.
        m_gen = pm.math.ge(gp[:, None], 1)

        # Perceptual distances used only when gp=3.
        m_per = pm.math.eq(gp[:, None], 3)

        # Distances chosen by group.
        d = m_per * d_per + (1 - m_per) * data["d_phy"]

        cond = pm.math.gt(v, 0.0)
        gate = cond.astype("float64")

        # Trails stimulus similarity to CS+.
        s = 1.0 + (pm.math.exp(-lambda_[:, None] * d) - 1.0) * gate * m_gen

        # Generalized associative strength.
        g = v * s

        # Non-linear transformation (latent - observed scale) with A = 1, K = 10.
        theta = 1 + (10 - 1) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * g)))

        # Likelihood.
        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            observed=data["y"],
            dims=("participants", "trials"),
        )

        # Define generalization trials range.
        g_range = np.arange(data["Nactrials"], data["Ntrials"])

        # Prediction for generalization trials.
        y_pre = pm.Normal(
            "y_pre",
            mu=theta[:, g_range],
            sigma=sigma[:, None],
            dims=("participants", "getrials"),
        )

        return model


def model_1v_LGPHY(data):
    """
    Define a PyMC model for single-CS learning and generalization with physical
    distances (LGPHY).

    This model focuses on generalization along physical dimensions and groups
    participants into three latent categories:
    - Group 1: Non-learners (alpha = 0, lambda = 0).
    - Group 2: Overgeneralizers (lambda = 0).
    - Group 3: Physical generalizers (lambda > 0, depends on physical distances).

    The model includes the following main parameters:
    - `sigma` (0 - ∞): Response noise, reflecting variability in observed responses.
    - `pi` (0 - 1): Group probabilities for the three latent groups.
    - `alpha` (0 - 1): Learning rate.
    - `lambda` (0 - ∞): Generalization rate.
    - `w0` (-∞ - ∞): Baseline response.
    - `w1` (0 - ∞): Latent-observed scaling factor.

    Args:
        data (dict): A dictionary containing:
            - 'd_phy' (ndarray): Physical distances for each participant/trial.
            - 'r' (ndarray): Reinforcement data.
            - 'k' (ndarray): CS indicator data.
            - 'Nparticipants' (int): Number of participants.
            - 'y' (ndarray): Observed responses (US expectancy).
            - 'Nactrials' (int): Number of active trials.
            - 'Ntrials' (int): Total number of trials.

    Returns:
        pm.Model: A PyMC model object representing the statistical structure.
    """
    with pm.Model(
        coords={
            "participants": np.arange(data["Nparticipants"]),
            "trials": np.arange(data["Ntrials"]),
            "getrials": np.arange(data["Nactrials"], data["Ntrials"]),
        }
    ) as model:

        r = pt.as_tensor_variable(data["r"], ndim=2)
        k = pt.as_tensor_variable(data["k"], ndim=2)

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        # lambda_sigma = pm.LogNormal("lambda_sigma", mu=np.log(0.3), sigma=0.6)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        alpha_mu = pm.Beta("alpha_mu", alpha=1, beta=1)
        # alpha_kappa = pm.LogNormal("alpha_kappa", mu=np.log(4), sigma=0.5)
        alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
        alpha_a = alpha_mu * alpha_kappa
        alpha_b = (1 - alpha_mu) * alpha_kappa
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=2)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)
        w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
        w0_sigma = pm.HalfStudentT("w0_sigma", nu=2, sigma=2)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        # Sigma means.
        sigma_mean_learners = pm.Normal("sigma_mean_learners", mu=0.75, sigma=0.2)
        sigma_mean_nonlearners = pm.Normal("sigma_mean_nonlearners", mu=2.25, sigma=0.2)

        # Participant-level mean based on group.
        sigma_mu = pm.math.switch(
            pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners
        )

        # Participant-level sigma variance.
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=1.5)

        # Final participant-level sigmas.
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_scatter,
            lower=1e-9,
            upper=3,
            dims="participants",
        )

        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants",
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants",
        )
        alpha_1 = pm.Beta("alpha_1", alpha=alpha_a, beta=alpha_b, dims="participants")
        w1_1 = pm.Gamma("w1_1", alpha=w1_a, beta=w1_b, dims="participants")

        # Learning Rate.
        alpha = pm.Deterministic(
            "alpha", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * alpha_1
        )

        # Generalization Rate.
        lambda_ = pm.Deterministic(
            "lambda",
            pm.math.eq(gp, 0) * 0
            + pm.math.eq(gp, 1) * lambda_2
            + pm.math.eq(gp, 2) * lambda_1,
        )

        # Baseline response.
        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma)

        # Latent - Observed Scaling.
        w1 = pm.Deterministic("w1", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * w1_1)

        # Initial excitatory associative strength.
        v_init = pt.zeros((1, data["Nparticipants"]), dtype="float64")

        # Define the update function for scan.
        def trial_update(k_t, r_t, v_t, alpha):

            # Compute trial specific excitatory associative strength.
            v_next = v_t + alpha * (r_t - v_t) * k_t

            return v_next

        # Sequential updates across trials.
        v_seq, _ = scan(
            fn=trial_update,
            sequences=[k.T, r.T],
            outputs_info=v_init,
            non_sequences=alpha[None, :],
        )

        # Excitatory associative strength.
        v = v_seq.squeeze(1).T

        # Allow generalization for gp {1,2}.
        m_gen = pm.math.ge(gp[:, None], 1)

        # Block propagation when v=0.
        cond = pm.math.gt(v, 0.0)
        gate = cond.astype("float64")

        # Trails stimulus similarity to CS+.
        s = 1.0 + (pm.math.exp(-lambda_[:, None] * data["d_phy"]) - 1.0) * gate * m_gen

        # Generalized associative strength.
        g = v * s

        # Non-linear transformation (latent - observed scale) with A = 1, K = 10.
        theta = 1 + (10 - 1) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * g)))

        # Likelihood.
        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            observed=data["y"],
            dims=("participants", "trials"),
        )

        # Define generalization trials range.
        g_range = np.arange(data["Nactrials"], data["Ntrials"])

        # Prediction for generalization trials.
        y_pre = pm.Normal(
            "y_pre",
            mu=theta[:, g_range],
            sigma=sigma[:, None],
            dims=("participants", "trials"),
        )

        return model


def model_2v_G2D(data):
    """
    Define a PyMC model for two-CS generalization with two distances (G2D).

    This model considers two conditioned stimuli (CS+ and CS-) and groups
    participants into three latent categories:
    - Group 1: Overgeneralizers (lambda = 0).
    - Group 2: Physical generalizers (lambda > 0, depends on physical distances).
    - Group 3: Perceptual generalizers (lambda > 0, depends on perceptual distances).

    The model includes the following main parameters:
    - `pi` (0 - 1): Group probabilities for the three latent groups.
    - `lambda` (0 - ∞): Generalization rate.
    - `w0` (-∞ - ∞): Baseline response.
    - `w1` (-∞ - ∞): Latent-observed scaling factor.
    - `sigma` (0 - ∞): Response noise, reflecting variability in observed responses.

    Args:
        data (dict): A dictionary containing:
            - 'd_p_phy' (ndarray): Physical distances for CS+.
            - 'd_m_phy' (ndarray): Physical distances for CS-.
            - 'Nparticipants' (int): Number of participants.
            - 'y' (ndarray): Observed responses (US expectancy).

    Returns:
        pm.Model: A PyMC model object representing the statistical structure.
    """

    with pm.Model(
        coords={
            "participants": np.arange(data["Nparticipants"]),
            "getrials": np.arange(data["Nactrials"], data["Ntrials"]),
        }
    ) as model:

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=2)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)
        sigma_a = pm.HalfStudentT("sigma_a", nu=2, sigma=2)
        sigma_b = pm.HalfStudentT("sigma_b", nu=2, sigma=2)
        w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
        w0_sigma = pm.HalfStudentT("w0_sigma", nu=2, sigma=2)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants",
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants",
        )
        d_sigma = pm.HalfCauchy("d_sigma", beta=2, dims="participants")

        # Generalization Rate.
        lambda_ = pm.Deterministic(
            "lambda", pm.math.eq(gp, 0) * lambda_2 + pm.math.neq(gp, 0) * lambda_1
        )

        # Baseline Response
        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma)

        # Latent - Observed Scaling.
        w1 = pm.Gamma("w1", alpha=w1_a, beta=w1_b, dims="participants")

        # Response Noise
        sigma = pm.Gamma("sigma", alpha=sigma_a, beta=sigma_b, dims="participants")

        # Generate missing perceptual distance between CS+ and S.
        d_p_per = pm.Normal(
            "d_p_per",
            mu=data["d_p_phy"],
            sigma=d_sigma[:, None],
            observed=data["d_p_per"],
            dims=("participants", "getrials"),
        )

        # Generate missing perceptual distance between CS- and S.
        d_m_per = pm.Normal(
            "d_m_per",
            mu=data["d_m_phy"],
            sigma=d_sigma[:, None],
            observed=data["d_m_per"],
            dims=("participants", "getrials"),
        )

        # Perceptual distances only when gp=2.
        m_per = pm.math.eq(gp[:, None], 2)

        # Distances chosen by group.
        dplus = m_per * d_p_per + (1 - m_per) * data["d_p_phy"]
        dminus = m_per * d_m_per + (1 - m_per) * data["d_m_phy"]

        # Trials stimulus similarity to CS+.
        s_plus = 1.0 + (pm.math.exp(-lambda_[:, None] * dplus) - 1.0)

        # Trials stimulus similarity to CS-.
        s_minus = 1.0 + (pm.math.exp(-lambda_[:, None] * dminus) - 1.0)

        # Generalized associative strength.
        g = s_plus - s_minus

        # Non-linear transformation (latent - observed scale) with A = 1, K = 10.
        theta = 1 + (10 - 1) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * g)))

        # Likelihood.
        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            observed=data["y"],
            dims=("participants", "getrials"),
        )

        # Prediction.
        y_pre = pm.Normal(
            "y_pre", mu=theta, sigma=sigma[:, None], dims=("participants", "getrials")
        )

        return model


def model_2v_LG2D(data):
    """
    Define a PyMC model for two-CS learning and generalization with two distances
    (LG2D).

    This model incorporates learning rates (`alpha`) for two conditioned stimuli
    (CS+ and CS-) and groups participants into four latent categories:
    - Group 1: Non-learners (alpha = 0).
    - Group 2: Overgeneralizers (lambda = 0).
    - Group 3: Physical generalizers (lambda > 0, depends on physical distances).
    - Group 4: Perceptual generalizers (lambda > 0, depends on perceptual distances).

    The model includes the following main parameters:
    - `sigma` (1E-9 - 3): Response noise, reflecting variability in observed responses.
    - `pi` (0 - 1): Group probabilities for the four latent groups.
    - `alpha` (0 - ∞): Learning rate.
    - `lambda` (0 - ∞): Generalization rate.
    - `w0` (-∞ - ∞): Baseline response.
    - `w1` (0 - ∞): Latent-observed scaling factor.

    Args:
        data (dict): A dictionary containing:
            - 'd_p_phy' (ndarray): Physical distances for CS+.
            - 'd_m_phy' (ndarray): Physical distances for CS-.
            - 'r_plus' (ndarray): Reinforcement data for CS+.
            - 'r_minus' (ndarray): Reinforcement data for CS-.
            - 'k_plus' (ndarray): CS+ indicator data.
            - 'k_minus' (ndarray): CS- indicator data.
            - 'Nparticipants' (int): Number of participants.
            - 'y' (ndarray): Observed responses (US expectancy).
            - 'Nactrials' (int): Number of active trials.
            - 'Ntrials' (int): Total number of trials.

    Returns:
        pm.Model: A PyMC model object representing the statistical structure.
    """

    with pm.Model(
        coords={
            "participants": np.arange(data["Nparticipants"]),
            "trials": np.arange(data["Ntrials"]),
            "getrials": np.arange(data["Nactrials"], data["Ntrials"]),
        }
    ) as model:

        r_plus = pt.as_tensor_variable(data["r_plus"], ndim=2)
        r_minus = pt.as_tensor_variable(data["r_minus"], ndim=2)
        k_plus = pt.as_tensor_variable(data["k_plus"], ndim=2)
        k_minus = pt.as_tensor_variable(data["k_minus"], ndim=2)

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        alpha_mu = pm.Beta("alpha_mu", alpha=1, beta=1)
        alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
        alpha_a = alpha_mu * alpha_kappa
        alpha_b = (1 - alpha_mu) * alpha_kappa
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=2)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)
        w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
        w0_sigma = pm.HalfStudentT("w0_sigma", nu=2, sigma=2)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        # Sigma means.
        sigma_mean_learners = pm.Normal("sigma_mean_learners", mu=0.75, sigma=0.2)
        sigma_mean_nonlearners = pm.Normal("sigma_mean_nonlearners", mu=2.25, sigma=0.2)

        # Participant-level mean based on group.
        sigma_mu = pm.math.switch(
            pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners
        )

        # Participant-level sigma variance.
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=1.5)

        # Final participant-level sigmas.
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_scatter,
            lower=1e-9,
            upper=3,
            dims="participants",
        )

        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants",
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants",
        )
        alpha_1 = pm.Beta("alpha_1", alpha=alpha_a, beta=alpha_b, dims="participants")
        w1_1 = pm.Gamma("w1_1", alpha=w1_a, beta=w1_b, dims="participants")
        d_sigma = pm.HalfCauchy("d_sigma", beta=2, dims="participants")

        # Learning Rate.
        alpha = pm.Deterministic(
            "alpha", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * alpha_1
        )

        # Generalization Rate.
        lambda_ = pm.Deterministic(
            "lambda",
            pm.math.eq(gp, 0) * 0
            + pm.math.eq(gp, 1) * lambda_2
            + pm.math.eq(gp, 2) * lambda_1
            + pm.math.eq(gp, 3) * lambda_1,
        )

        # Baseline response.
        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma)

        # Latent - Observed Scaling.
        w1 = pm.Deterministic("w1", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * w1_1)

        # Initial excitatory associative strength.
        v_plus_init = pt.zeros((1, data["Nparticipants"]), dtype="float64")

        # Initial inhibitory associative strength.
        v_minus_init = pt.zeros((1, data["Nparticipants"]), dtype="float64")

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
            observed=data["d_p_per"],
        )

        # Generate missing perceptual distance between CS- and S.
        d_m_per = pm.TruncatedNormal(
            "d_m_per",
            mu=data["d_m_phy"],
            sigma=d_sigma[:, None],
            lower=0,
            observed=data["d_m_per"],
        )

        # Allow generalization for gp {1,2,3}.
        m_gen = pm.math.ge(gp[:, None], 1)

        # Perceptual distances only when gp=3.
        m_per = pm.math.eq(gp[:, None], 3)

        # Distances chosen by group.
        dplus = m_per * d_p_per + (1 - m_per) * data["d_p_phy"]
        dminus = m_per * d_m_per + (1 - m_per) * data["d_m_phy"]

        # Block propagation when v=0.
        cond_p = pm.math.gt(v_plus, 0.0)
        cond_m = pm.math.gt(pm.math.abs(v_minus), 0.0)
        gate_p = cond_p.astype("float64")
        gate_m = cond_m.astype("float64")

        # Trials stimulus similarity to CS+.
        s_plus = 1.0 + (pm.math.exp(-lambda_[:, None] * dplus) - 1.0) * gate_p * m_gen

        # Trials stimulus similarity to CS-.
        s_minus = 1.0 + (pm.math.exp(-lambda_[:, None] * dminus) - 1.0) * gate_m * m_gen

        # Generalized associative strength.
        g = v_plus * s_plus + v_minus * s_minus

        # Non-linear transformation (latent - observed scale) with A = 1, K = 10.
        theta = 1 + (10 - 1) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * g)))

        # Likelihood.
        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            observed=data["y"],
            dims=("participants", "trials"),
        )

        # Define generalization trials range.
        g_range = np.arange(data["Nactrials"], data["Ntrials"])

        # Prediction for generalization trials.
        y_pre = pm.Normal(
            "y_pre",
            mu=theta[:, g_range],
            sigma=sigma[:, None],
            dims=("participants", "getrials"),
        )

        return model


def model_2v_LGPHY(data):
    """
    Define a PyMC model for two-CS learning and generalization with physical
    distances (LGPHY).

    This model focuses on physical generalization for two conditioned stimuli
    (CS+ and CS-) and groups participants into three latent categories:
    - Group 1: Non-learners (alpha = 0, lambda = 0).
    - Group 2: Overgeneralizers (lambda = 0).
    - Group 3: Physical generalizers (lambda > 0, depends on physical distances).

    The model includes the following main parameters:
    - `sigma` (1E-9 - 3): Response noise, reflecting variability in observed responses.
    - `pi` (0 - 1): Group probabilities for the three latent groups.
    - `alpha` (0 - ∞): Learning rate.
    - `lambda` (0 - ∞): Generalization rate.
    - `w0` (-∞ - ∞): Baseline response.
    - `w1` (0 - ∞): Latent-observed scaling factor.

    Args:
        data (dict): A dictionary containing:
            - 'd_p_phy' (ndarray): Physical distances for CS+.
            - 'd_m_phy' (ndarray): Physical distances for CS-.
            - 'r_plus' (ndarray): Reinforcement data for CS+.
            - 'r_minus' (ndarray): Reinforcement data for CS-.
            - 'k_plus' (ndarray): CS+ indicator data.
            - 'k_minus' (ndarray): CS- indicator data.
            - 'Nparticipants' (int): Number of participants.
            - 'y' (ndarray): Observed responses (US expectancy).
            - 'Nactrials' (int): Number of active trials.
            - 'Ntrials' (int): Total number of trials.

    Returns:
        pm.Model: A PyMC model object representing the statistical structure.
    """

    with pm.Model(
        coords={
            "participants": np.arange(data["Nparticipants"]),
            "trials": np.arange(data["Ntrials"]),
            "getrials": np.arange(data["Nactrials"], data["Ntrials"]),
        }
    ) as model:

        r_plus = pt.as_tensor_variable(data["r_plus"], ndim=2)
        r_minus = pt.as_tensor_variable(data["r_minus"], ndim=2)
        k_plus = pt.as_tensor_variable(data["k_plus"], ndim=2)
        k_minus = pt.as_tensor_variable(data["k_minus"], ndim=2)

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", lower=0, mu=0.1, sigma=1)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        alpha_mu = pm.Beta("alpha_mu", alpha=1, beta=1)
        alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
        alpha_a = alpha_mu * alpha_kappa
        alpha_b = (1 - alpha_mu) * alpha_kappa
        w1_a = pm.HalfStudentT("w1_a", nu=2, sigma=2)
        w1_b = pm.HalfStudentT("w1_b", nu=2, sigma=2)
        w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
        w0_sigma = pm.HalfStudentT("w0_sigma", nu=2, sigma=2)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1]))

        # Latent group indicators.
        gp = pm.Categorical("gp", p=pi, dims="participants")

        # Sigma means.
        sigma_mean_learners = pm.Normal("sigma_mean_learners", mu=0.75, sigma=0.2)
        sigma_mean_nonlearners = pm.Normal("sigma_mean_nonlearners", mu=2.25, sigma=0.2)

        # Participant-level mean based on group.
        sigma_mu = pm.math.switch(
            pm.math.eq(gp, 0), sigma_mean_nonlearners, sigma_mean_learners
        )

        # Participant-level sigma variance.
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=1.5)

        # Final participant-level sigmas.
        sigma = pm.TruncatedNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_scatter,
            lower=1e-9,
            upper=3,
            dims="participants",
        )

        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants",
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants",
        )
        alpha_1 = pm.Beta("alpha_1", alpha=alpha_a, beta=alpha_b, dims="participants")
        w1_1 = pm.Gamma("w1_1", alpha=w1_a, beta=w1_b, dims="participants")

        # Learning Rate.
        alpha = pm.Deterministic(
            "alpha", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * alpha_1
        )

        # Generalization Rate.
        lambda_ = pm.Deterministic(
            "lambda",
            pm.math.eq(gp, 0) * 0
            + pm.math.eq(gp, 1) * lambda_2
            + pm.math.eq(gp, 2) * lambda_1,
        )

        # Baseline response.
        w0_1 = pm.Normal("w0_1", 0, 1, dims="participants")
        w0 = pm.Deterministic("w0", w0_mu + w0_1 * w0_sigma)

        # Latent - Observed Scaling.
        w1 = pm.Deterministic("w1", pm.math.eq(gp, 0) * 0 + pm.math.neq(gp, 0) * w1_1)

        # Initial excitatory associative strength.
        v_plus_init = pt.zeros((1, data["Nparticipants"]), dtype="float64")

        # Initial inhibitory associative strength.
        v_minus_init = pt.zeros((1, data["Nparticipants"]), dtype="float64")

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

        # Allow generalization for gp {1,2}.
        m_gen = pm.math.ge(gp[:, None], 1)

        # Block propagation when v=0.
        cond_p = pm.math.gt(v_plus, 0.0)
        cond_m = pm.math.gt(pm.math.abs(v_minus), 0.0)
        gate_p = cond_p.astype("float64")
        gate_m = cond_m.astype("float64")

        # Trials stimulus similarity to CS+.
        s_plus = (
            1.0
            + (pm.math.exp(-lambda_[:, None] * data["d_p_phy"]) - 1.0) * gate_p * m_gen
        )

        # Trials stimulus similarity to CS-.
        s_minus = (
            1.0
            + (pm.math.exp(-lambda_[:, None] * data["d_m_phy"]) - 1.0) * gate_m * m_gen
        )

        # Generalized associative strength.
        g = v_plus * s_plus + v_minus * s_minus

        # Non-linear transformation (latent - observed scale) with A = 1, K = 10.
        theta = 1 + (10 - 1) / (1 + pm.math.exp(-(w0[:, None] + w1[:, None] * g)))

        # Likelihood.
        y = pm.Normal(
            "y",
            mu=theta,
            sigma=sigma[:, None],
            observed=data["y"],
            dims=("participants", "trials"),
        )

        # Define generalization trials range.
        g_range = np.arange(data["Nactrials"], data["Ntrials"])

        # Prediction for generalization trials.
        y_pre = pm.Normal(
            "y_pre",
            mu=theta[:, g_range],
            sigma=sigma[:, None],
            dims=("participants", "getrials"),
        )

        return model


def marginalized_model(data):
    "WIP prototype (abandoned)"

    data["y_complete"] = np.where(np.isnan(data["y"]), np.nanmean(data["y"]), data["y"])

    coords = {
        "participants": np.arange(data["Nparticipants"]),
        "trials": np.arange(data["Ntrials"]),
        "groups": np.arange(4),
        "getrials": np.arange(data["Nactrials"], data["Ntrials"]),
    }

    with pm.Model(coords=coords) as model:

        r_plus = pt.as_tensor_variable(data["r_plus"], ndim=2)
        r_minus = pt.as_tensor_variable(data["r_minus"], ndim=2)
        k_plus = pt.as_tensor_variable(data["k_plus"], ndim=2)
        k_minus = pt.as_tensor_variable(data["k_minus"], ndim=2)

        # Hyperpriors.
        lambda_mu = pm.TruncatedNormal("lambda_mu", mu=0.1, sigma=1, lower=0)
        lambda_sigma = pm.Uniform("lambda_sigma", lower=1e-9, upper=1)
        alpha_mu = pm.Beta("alpha_mu", alpha=1, beta=1)
        alpha_kappa = pm.Uniform("alpha_kappa", lower=1, upper=10)
        alpha_a = alpha_mu * alpha_kappa
        alpha_b = (1 - alpha_mu) * alpha_kappa
        w1_a = pm.HalfCauchy("w1_a", beta=2)
        w1_b = pm.HalfCauchy("w1_b", beta=2)
        w0_mu = pm.Normal("w0_mu", mu=0, sigma=10)
        w0_sigma = pm.HalfCauchy("w0_sigma", beta=2)

        # Group probability.
        pi = pm.Dirichlet("pi", a=np.array([1, 1, 1, 1]))

        # Response Noise.
        sigma = pm.HalfCauchy("sigma", beta=1.5, shape=2)

        """# global scatter hyper-prior 
        sigma_scatter = pm.HalfNormal("sigma_scatter", sigma=0.10)

        # participant sigma
        sigma_mu =  pm.math.switch(pm.math.eq(gp, 0), pm.math.log(sigma_mean[1]), pm.math.log(sigma_mean[0]))

        sigma = pm.LogNormal(
            "sigma",
            mu=sigma_mu,
            sigma=sigma_scatter,
            dims="participants"
        )"""
        # sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        # Priors.
        lambda_1 = pm.TruncatedNormal(
            "lambda_1",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0.0052,
            dims="participants",
        )
        lambda_2 = pm.TruncatedNormal(
            "lambda_2",
            mu=lambda_mu,
            sigma=lambda_sigma,
            lower=0,
            upper=0.0052,
            dims="participants",
        )
        alpha_1 = pm.Beta("alpha_1", alpha=alpha_a, beta=alpha_b, dims="participants")
        w1_1 = pm.Gamma("w1_1", alpha=w1_a, beta=w1_b, dims="participants")
        # w1_1 = pm.Deterministic("w1_1", w1_1_ + 1.0, dims="participants")
        d_sigma = pm.HalfCauchy("d_sigma", beta=2, dims="participants")
        # inflation = pm.HalfNormal("inflation", sigma=5)
        # d_sigma_phy  = pm.Deterministic("big_sigma", 20 + inflation**2)
        # d_sigma = pm.Deterministic("d_sigma", pm.math.neq(gp, 3)*50 + pm.math.eq(gp, 3)*d_sigma_per)
        w0 = pm.Normal("w0", w0_mu, w0_sigma, dims="participants")

        d_p_per = pm.TruncatedNormal(
            "d_p_per",
            mu=data["d_p_phy"],
            sigma=d_sigma[:, None],
            lower=0,
            observed=data["d_p_per"],
            dims=("participants", "trials"),
        )

        d_m_per = pm.TruncatedNormal(
            "d_m_per",
            mu=data["d_m_phy"],
            sigma=d_sigma[:, None],
            lower=0,
            observed=data["d_m_per"],
            dims=("participants", "trials"),
        )

        # Group-specific parameter arrays
        lambda_g = pt.stack(
            [pt.zeros_like(lambda_1), lambda_2, lambda_1, lambda_1], axis=0
        )
        w1_g = pt.stack([pt.zeros_like(w1_1), w1_1, w1_1, w1_1], axis=0)
        sigma_g = pt.stack(
            [
                pt.ones_like(alpha_1) * sigma[1],
                pt.ones_like(alpha_1) * sigma[0],
                pt.ones_like(alpha_1) * sigma[0],
                pt.ones_like(alpha_1) * sigma[0],
            ],
            axis=0,
        )

        d_g_p = pt.stack(
            [data["d_p_phy"], data["d_p_phy"], data["d_p_phy"], d_p_per], axis=0
        )
        d_g_m = pt.stack(
            [data["d_m_phy"], data["d_m_phy"], data["d_m_phy"], d_m_per], axis=0
        )

        # RL updates
        v_init_plus = pt.zeros((data["Nparticipants"],))
        v_init_minus = pt.zeros((data["Nparticipants"],))

        def step_fn(k_t_p, r_t_p, k_t_m, r_t_m, v_t_p, v_t_m, alpha):
            v_next_p = v_t_p + alpha * (r_t_p - v_t_p) * k_t_p
            v_next_m = v_t_m + alpha * (r_t_m - v_t_m) * k_t_m
            return v_next_p, v_next_m

        [v_plus_seq, v_minus_seq], _ = scan(
            fn=step_fn,
            sequences=[
                k_plus.T,
                r_plus.T,
                k_minus.T,
                r_minus.T,
            ],
            outputs_info=[v_init_plus, v_init_minus],
            non_sequences=alpha_1,
        )

        v_p = v_plus_seq.T
        v_m = v_minus_seq.T
        v_g_p = pt.stack([pt.zeros_like(v_p), v_p, v_p, v_p], axis=0)
        v_g_m = pt.stack([pt.zeros_like(v_m), v_m, v_m, v_m], axis=0)

        s_g_p = pt.where(v_g_p > 0, pt.exp(-lambda_g[..., None] * d_g_p), 1.0)
        s_g_m = pt.where(
            pm.math.abs(v_g_m) > 0, pt.exp(-lambda_g[..., None] * d_g_m), 1.0
        )
        g = v_g_p * s_g_p + v_g_m * s_g_m
        θ_g = 1 + 9 / (1 + pt.exp(-(w0[None, :, None] + w1_g[..., None] * g)))

        # Mixture likelihood
        theta_mix = pt.transpose(θ_g, (1, 2, 0))
        sigma_mix = pt.transpose(sigma_g[..., None], (1, 2, 0))

        y_obs = pm.Mixture(
            "y_obs",
            w=pi[None, None, :],
            comp_dists=pm.Normal.dist(mu=theta_mix, sigma=sigma_mix),
            observed=data["y_complete"],
            dims=("participants", "trials"),
        )

        # Posterior group probabilities
        comp_logp = pm.logp(
            pm.Normal.dist(mu=θ_g, sigma=sigma_g[..., None]), data["y_complete"]
        ).sum(axis=-1)

        log_posterior_gp = pt.log(pi[:, None]) + comp_logp

        posterior_gp_probs = pm.Deterministic(
            "posterior_gp_probs",
            pt.exp(log_posterior_gp - pt.logsumexp(log_posterior_gp, axis=0)),
            dims=("groups", "participants"),
        )

        # Posterior predictive checks
        theta_mix_pre = pt.transpose(θ_g[..., coords["getrials"]], (1, 2, 0))
        sigma_mix_pre = pt.repeat(sigma_mix, len(coords["getrials"]), axis=1)
        y_pre = pm.Mixture(
            "y_pre",
            w=pi[None, None, :],
            comp_dists=pm.Normal.dist(mu=theta_mix_pre, sigma=sigma_mix_pre),
            dims=("participants", "getrials"),
        )

        return model

