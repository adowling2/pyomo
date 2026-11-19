.. _parmest_covariance_section:

Uncertainty Quantification
==========================
The goal of parameter estimation (see :ref:`driversection` Section) is to estimate unknown model parameters
from experimental data. Uncertainty quantification then aims to characterize
how close the parameter estimates are to their true (unknown) values. 
This parameter uncertainty can be computed using four methods in parmest:
covariance matrix, likelihood ratio test, bootstrapping, and leave-N-out.

Covariance Matrix Estimation
----------------------------

The uncertainty in estimated model parameters can be quantified by computing the covariance matrix.
The diagonal of this covariance matrix contains the variance of the estimated parameters, which is used to
calculate their uncertainty. Assuming Gaussian independent and identically distributed measurement errors,
the covariance matrix of the estimated parameters can be computed using the following methods which have been
implemented in parmest.

1. Reduced Hessian Method

    When the objective function is the sum of squared errors (SSE) for homogeneous data, defined as
    :math:`\text{SSE} = \frac{1}{2} \sum_{i = 1}^{n} \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};
    \boldsymbol{\theta})\right)^\text{T} \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};
    \boldsymbol{\theta})\right)`, the covariance matrix is:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = \sigma^2 \left(\frac{\partial^2 \text{SSE}}
        {\partial \boldsymbol{\theta}^2}\right)^{-1}_{\boldsymbol{\theta}
        = \hat{\boldsymbol{\theta}}}

    Similarly, when the objective function is the weighted SSE (WSSE) for heterogeneous data, defined as
    :math:`\text{WSSE} = \frac{1}{2} \sum_{i = 1}^{n} \left(\boldsymbol{y}_{i} -
    \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})\right)^\text{T} \boldsymbol{\Sigma}_{\boldsymbol{y}}^{-1}
    \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})\right)`,
    the covariance matrix is:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = \left(\frac{\partial^2 \text{WSSE}}
        {\partial \boldsymbol{\theta}^2}\right)^{-1}_{\boldsymbol{\theta}
        = \hat{\boldsymbol{\theta}}}

    Where :math:`\boldsymbol{V}_{\boldsymbol{\theta}}` is the covariance matrix of the estimated
    parameters :math:`\hat{\boldsymbol{\theta}} \in \mathbb{R}^p`, :math:`\boldsymbol{y}_{i} \in \mathbb{R}^m` are
    observations of the measured variables, :math:`\boldsymbol{f}` is the model function,
    :math:`\boldsymbol{x}_{i} \in \mathbb{R}^{q}` are the input variables, :math:`n` is the number of experiments,
    :math:`\boldsymbol{\Sigma}_{\boldsymbol{y}}` is the measurement error covariance matrix, and :math:`\sigma^2`
    is the variance of the measurement error. When the standard deviation of the measurement error is not supplied
    by the user, parmest approximates :math:`\sigma^2` as:
    :math:`\hat{\sigma}^2 = \frac{1}{n-p} \sum_{i=1}^{n} \boldsymbol{\varepsilon}_{i}(\boldsymbol{\theta})^{\text{T}}
    \boldsymbol{\varepsilon}_{i}(\boldsymbol{\theta})`, and :math:`\boldsymbol{\varepsilon}_{i} \in \mathbb{R}^m`
    are the residuals between the data and model for experiment :math:`i`.

    In parmest, this method computes the inverse of the Hessian by scaling the
    objective function (SSE or WSSE) with a constant probability factor, :math:`\frac{1}{n}`.

2. Finite Difference Method

    In this method, the covariance matrix, :math:`\boldsymbol{V}_{\boldsymbol{\theta}}`, is
    computed by differentiating the Hessian,
    :math:`\frac{\partial^2 \text{SSE}}{\partial \boldsymbol{\theta}^2}`
    or
    :math:`\frac{\partial^2 \text{WSSE}}{\partial \boldsymbol{\theta}^2}`, and
    applying Gauss-Newton approximation which results in:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = \left(\sum_{i = 1}^n \boldsymbol{G}_{i}^{\text{T}}
        \boldsymbol{\Sigma}_{\boldsymbol{y}}^{-1} \boldsymbol{G}_{i} \right)^{-1}

    where

    .. math::
       \boldsymbol{G}_{i} = \frac{\partial \boldsymbol{f}(\boldsymbol{x}_i;\boldsymbol{\theta})}
        {\partial \boldsymbol{\theta}}

    This method uses central finite difference to compute the Jacobian matrix, :math:`\boldsymbol{G}_{i}`,
    for experiment :math:`i`.

    .. math::
       \boldsymbol{G}_{i}[:,\,k] \approx \frac{\boldsymbol{f}(\boldsymbol{x}_i;\theta_k + \Delta \theta_k)
        \vert_{\hat{\boldsymbol{\theta}}} - \boldsymbol{f}(\boldsymbol{x}_i;\theta_k - \Delta \theta_k)
        \vert_{\hat{\boldsymbol{\theta}}}}{2 \Delta \theta_k} \quad \forall \quad \theta_k \, \in \,
        [\theta_1,\cdots, \theta_p]

3. Automatic Differentiation Method

    Similar to the finite difference method, the covariance matrix is calculated as:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = \left( \sum_{i = 1}^n \boldsymbol{G}_{\text{kaug},\, i}^{\text{T}}
        \boldsymbol{\Sigma}_{\boldsymbol{y}}^{-1} \boldsymbol{G}_{\text{kaug},\, i} \right)^{-1}

    However, this method uses implicit differentiation and the model-optimality or Karush–Kuhn–Tucker (KKT) conditions
    to compute the Jacobian matrix, :math:`\boldsymbol{G}_{\text{kaug},\, i}`, for experiment :math:`i`.

    .. math::
       \boldsymbol{G}_{\text{kaug},\,i} = \frac{\partial \boldsymbol{f}(\boldsymbol{x}_i,\boldsymbol{\theta})}
        {\partial \boldsymbol{\theta}} + \frac{\partial \boldsymbol{f}(\boldsymbol{x}_i,\boldsymbol{\theta})}
        {\partial \boldsymbol{x}_i}\frac{\partial \boldsymbol{x}_i}{\partial \boldsymbol{\theta}}

The covariance matrix calculation is only supported with the built-in objective functions "SSE" or "SSE_weighted".

In parmest, the covariance matrix can be computed after creating the
:class:`~pyomo.contrib.parmest.experiment.Experiment` class,
defining the :class:`~pyomo.contrib.parmest.parmest.Estimator` object,
and estimating the model parameters using :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`
(all these steps were addressed in the :ref:`driversection` Section).

To estimate the covariance matrix, with the default method being "finite_difference", call
the :class:`~pyomo.contrib.parmest.parmest.Estimator.cov_est` function as follows:

.. testsetup:: *
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    # Data
    import pandas as pd
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0],
              [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Create the Experiment class
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment

    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> obj_val, theta_val = pest.theta_est()
    >>> cov = pest.cov_est()

Optionally, one of the three methods; "reduced_hessian", "finite_difference",
and "automatic_differentiation_kaug" can be supplied for the covariance calculation,
e.g.,

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> obj_val, theta_val = pest.theta_est()
    >>> cov_method = "reduced_hessian"
    >>> cov = pest.cov_est(method=cov_method)

Bootstrapping
-------------

Bootstrapping is a non-intrusive method that uses resampling to approximate the variance of the parameter estimates.
By repeatedly fitting the model to resampled datasets, the parameter uncertainty (e.g., variance or covariance)
can be computed without relying on strong distributional assumptions. This method is summarized as follows:

1. Step 0: Define the input data

    Given input data: :math:`[y_1, \dots, y_n]`

2. Step 1: Generate :math:`B` artificial datasets through sampling

    Sample with replacement from the original data to create :math:`B` bootstrap datasets:

    .. math::
        \left\{y_1^{(1)}, \dots, y_n^{(1)} \right\}, \dots, \left\{y_1^{(B)}, \dots, y_n^{(B)} \right\}

3. Step 2: Compute the estimator of the parameters

    Fit the model to each bootstrap dataset to obtain parameter estimates:

    .. math::
        \left\{ \hat{\theta}^{(1)}, \dots, \hat{\theta}^{(B)} \right\}

4. Step 3: Compute the approximate variance of the parameter estimates

    The variability across bootstrap estimates approximates the estimator variance:

    .. math::
        \text{Var}(\hat{\theta}) =
        \frac{1}{B} \sum_{j=1}^{B} \left(\hat{\theta}^{(j)}\right)^2
        -
        \left(
        \frac{1}{B} \sum_{j=1}^{B} \hat{\theta}^{(j)}
        \right)^2

.. note::

    The example code for this method will soon be provided.

Likelihood Ratio Test
---------------------

Use :meth:`~pyomo.contrib.parmest.parmest.Estimator.theta_est` to fit the model,
:meth:`~pyomo.contrib.parmest.parmest.Estimator.objective_at_theta` to evaluate
candidate parameter vectors on the same data, and
:meth:`~pyomo.contrib.parmest.parmest.Estimator.likelihood_ratio_test` to
classify the candidates. The output gives a Boolean column for each
confidence level in ``alphas``. For example, ``0.95`` represents 95% confidence.
With ``return_thresholds=True``, the method also returns the cutoffs in the
same units as the input objective.

Let :math:`Q(\theta)` be the objective returned by parmest, a mean of the
per-experiment objectives over :math:`S` experiments, and :math:`\hat\theta`
be the unrestricted fit. The likelihood-ratio statistic is twice the
difference between the unrestricted and restricted maximized log-likelihoods.
Under regularity conditions it is asymptotically :math:`\chi^2_p`, where
:math:`p` is the number of fitted scalar parameters fixed by the candidate.
This method fixes the full parameter vector. It does not profile subsets of
the fitted parameters.

* For unregularized ``SSE`` with independent Gaussian errors of unknown common
  variance, let :math:`n` be the number of scalar residuals in the objective.
  Profiling the variance gives

  .. math::

     T(\theta) = n\log\!\left(\frac{Q(\theta)}{Q(\hat\theta)}\right).

  Thus the cutoff at confidence level :math:`a` is
  :math:`Q(\hat\theta)\exp(\chi^2_{p,a}/n)`. This calculation requires a
  positive fitted SSE and :math:`n>p`.

* For unregularized ``SSE_weighted`` with independent Gaussian errors and
  known, fixed measurement standard deviations, the summed objective is
  the negative log-likelihood up to a constant. The statistic is

  .. math::

     T(\theta) = 2S\left[Q(\theta)-Q(\hat\theta)\right].

  The cutoff is :math:`Q(\hat\theta) + \chi^2_{p,a}/(2S)`. A shared, known
  standard deviation can be supplied for every measurement to use this
  case.

For custom objectives and the deprecated estimator interface, parmest
retains the historical chi-squared approximation using
:math:`Q(\hat\theta)[1+\chi^2_{p,a}/(S-p)]`. This rule assumes a constant
multiple of SSE with one independent scalar residual per experiment and
unknown common variance. A generic loss or a user-supplied negative
log-likelihood need not have this interpretation. A nonzero L2 penalty is
incompatible with these likelihood-ratio calculations.

The chi-squared approximation assumes an identifiable model, an interior
true parameter, and comparable minimized objectives on the same data. In
small samples, at parameter boundaries, or with weak identifiability, the
reported region may differ from the intended coverage. See
`Wilks (1938) <https://doi.org/10.1214/aoms/1177732360>`_,
`Seber and Wild, Section 5.2.3 <https://doi.org/10.1002/0471725315.ch5>`_,
and `Self and Liang (1987) <https://doi.org/10.1080/01621459.1987.10478472>`_.

Leave-N-Out
-----------

:meth:`~pyomo.contrib.parmest.parmest.Estimator.theta_est_leaveNout` fits each
complement after leaving out ``lNo`` experiments.
:meth:`~pyomo.contrib.parmest.parmest.Estimator.leaveNout_bootstrap_test` fits
the held-out experiments themselves and compares that fit with the bootstrap
cloud from the complementary training set. Both subsets must support their
respective fits. This comparison is a sensitivity diagnostic. In particular,
the held-out estimator and the training estimator can have different
sampling variances, so the membership flags do not give a calibrated
hypothesis test.

A random seed controls the split and training draws together. The training draws retain the existing bootstrap requirement for more distinct experiments than fitted parameters.
