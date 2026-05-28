rm(list = ls())

Prior1<-"functions {
  // Custom log-PDF for your QULL distribution
  real Q_custom_lpdf(real y, real alpha, real mu, real u) {
    real term1;
    real term2;
    real term3;
    real term4;
    real term5;
    real term6;
    real term7;
    real exponent;

    // compute common exponent
    exponent = pow(-log(y), alpha) * pow(-log(mu), -alpha);

    // log(alpha / y)
    term1 = log(alpha);
    term2 = -log(y);

    // (-log y)^(alpha - 1)
    term3 = (alpha - 1) * log(-log(y));

    // (-log mu)^(-alpha)
    term4 = -alpha * log(-log(mu));

    // log(log(1 - log u))
    term5 = log(log(1 - log(u)));

    // [(1 - log u)^( -log mu^-alpha )]^( -log y^alpha )
    term6 = exponent * log(1 - log(u));

    // exp{1 - [...] } -> log(exp{1 - [...]}) = 1 - [...]
    term7 = 1 - pow(1 - log(u), exponent);

    return term1 + term2 + term3 + term4 + term5 + term6 + term7;
  }
  real Q_custom_rng(real alpha, real mu, real u) {
    real p;
    real B;
    real denom_log;
    real numer_log;
    real y;

    // Sample p ~ Uniform(0,1)
    p = uniform_rng(0, 1);

    // Compute intermediate values
    B = -log(mu);
    denom_log = log(1 - log(u));
    numer_log = log(1 - log(p));

    // Compute the quantile
    y = exp( - B * pow(numer_log / denom_log, 1 / alpha) );
    return y;
  }
}

data {
  int<lower=0> n;                 // number of observations
  int<lower=0> M;                 // number of groups
  real<lower=0, upper=1> y[n];    // response
  int<lower=1> K;                 // number of predictors
  matrix[n, K] X;                 // design matrix
  int<lower=1, upper=M> id[n];    // group ID
  vector[K] b0;                    // prior mean for beta
  matrix[K, K] B0;                 // prior covariance for beta
  real<lower=0, upper=1> u;       // fixed u
}

parameters {
  vector[K] beta;                 
  vector<lower=0>[2] sigma_b;     
  matrix[2, M] z_b;               
  cholesky_factor_corr[2] L_b;    
  real<lower=0> alpha;            
}

transformed parameters {
  vector<lower=0, upper=1>[n] mu;
  vector[n] eta_mu;
  matrix[2, M] b;
  
  b = diag_pre_multiply(sigma_b, L_b) * z_b;
  eta_mu = X * beta;
  
  for (i in 1:n)
    mu[i] = inv_logit(eta_mu[i] + b[1, id[i]] + b[2, id[i]] * X[i, 3]);
}

model {
  // Priors
  L_b ~ lkj_corr_cholesky(1.0);
  to_vector(z_b) ~ normal(0, 1);
  beta ~ multi_normal(b0, B0);
  sigma_b ~ normal(0,1);
  alpha ~ exponential(0.5);
  
  // Likelihood
  for (i in 1:n)
    target += Q_custom_lpdf(y[i] | alpha, mu[i], u);
}
generated quantities {
  matrix[2, 2] Rho;
  vector[n] log_lik;
  vector[n] y_sim;

  // Compute correlation matrix
  Rho = multiply_lower_tri_self_transpose(L_b);

  // Compute log-likelihood for observed data
  for (i in 1:n)
    log_lik[i] = Q_custom_lpdf(y[i] | alpha, mu[i], u);

  // Simulate new data using custom RNG
  for (i in 1:n)
    y_sim[i] = Q_custom_rng(alpha, mu[i], 0.5);
}"


Prior2<-"functions {
  // Custom log-PDF for your QULL distribution
  real Q_custom_lpdf(real y, real alpha, real mu, real u) {
    real term1;
    real term2;
    real term3;
    real term4;
    real term5;
    real term6;
    real term7;
    real exponent;

    // compute common exponent
    exponent = pow(-log(y), alpha) * pow(-log(mu), -alpha);

    // log(alpha / y)
    term1 = log(alpha);
    term2 = -log(y);

    // (-log y)^(alpha - 1)
    term3 = (alpha - 1) * log(-log(y));

    // (-log mu)^(-alpha)
    term4 = -alpha * log(-log(mu));

    // log(log(1 - log u))
    term5 = log(log(1 - log(u)));

    // [(1 - log u)^( -log mu^-alpha )]^( -log y^alpha )
    term6 = exponent * log(1 - log(u));

    // exp{1 - [...] } -> log(exp{1 - [...]}) = 1 - [...]
    term7 = 1 - pow(1 - log(u), exponent);

    return term1 + term2 + term3 + term4 + term5 + term6 + term7;
  }
  real Q_custom_rng(real alpha, real mu, real u) {
    real p;
    real B;
    real denom_log;
    real numer_log;
    real y;

    // Sample p ~ Uniform(0,1)
    p = uniform_rng(0, 1);

    // Compute intermediate values
    B = -log(mu);
    denom_log = log(1 - log(u));
    numer_log = log(1 - log(p));

    // Compute the quantile
    y = exp( - B * pow(numer_log / denom_log, 1 / alpha) );
    return y;
  }
}

data {
  int<lower=0> n;                 // number of observations
  int<lower=0> M;                 // number of groups
  real<lower=0, upper=1> y[n];    // response
  int<lower=1> K;                 // number of predictors
  matrix[n, K] X;                 // design matrix
  int<lower=1, upper=M> id[n];    // group ID
  vector[K] b0;                    // prior mean for beta
  matrix[K, K] B0;                 // prior covariance for beta
  real<lower=0, upper=1> u;       // fixed u
}

parameters {
  vector[K] beta;                 
  vector<lower=0>[2] sigma_b;     
  matrix[2, M] z_b;               
  cholesky_factor_corr[2] L_b;    
  real<lower=0> alpha;            
}

transformed parameters {
  vector<lower=0, upper=1>[n] mu;
  vector[n] eta_mu;
  matrix[2, M] b;
  
  b = diag_pre_multiply(sigma_b, L_b) * z_b;
  eta_mu = X * beta;
  
  for (i in 1:n)
    mu[i] = inv_logit(eta_mu[i] + b[1, id[i]] + b[2, id[i]] * X[i, 3]);
}

model {
  // Priors
  L_b ~ lkj_corr_cholesky(2.0);
  to_vector(z_b) ~ normal(0, 1);
  beta ~ multi_normal(b0, B0);
  sigma_b ~ normal(0,1);
  alpha ~ exponential(0.5);
  
  // Likelihood
  for (i in 1:n)
    target += Q_custom_lpdf(y[i] | alpha, mu[i], u);
}
generated quantities {
  matrix[2, 2] Rho;
  vector[n] log_lik;
  vector[n] y_sim;

  // Compute correlation matrix
  Rho = multiply_lower_tri_self_transpose(L_b);

  // Compute log-likelihood for observed data
  for (i in 1:n)
    log_lik[i] = Q_custom_lpdf(y[i] | alpha, mu[i], u);

  // Simulate new data using custom RNG
  for (i in 1:n)
    y_sim[i] = Q_custom_rng(alpha, mu[i], 0.5);
}"

Prior3<-"functions {
 // Custom log-PDF for your QULL distribution
  real Q_custom_lpdf(real y, real alpha, real mu, real u) {
    real term1;
    real term2;
    real term3;
    real term4;
    real term5;
    real term6;
    real term7;
    real exponent;

    // compute common exponent
    exponent = pow(-log(y), alpha) * pow(-log(mu), -alpha);

    // log(alpha / y)
    term1 = log(alpha);
    term2 = -log(y);

    // (-log y)^(alpha - 1)
    term3 = (alpha - 1) * log(-log(y));

    // (-log mu)^(-alpha)
    term4 = -alpha * log(-log(mu));

    // log(log(1 - log u))
    term5 = log(log(1 - log(u)));

    // [(1 - log u)^( -log mu^-alpha )]^( -log y^alpha )
    term6 = exponent * log(1 - log(u));

    // exp{1 - [...] } -> log(exp{1 - [...]}) = 1 - [...]
    term7 = 1 - pow(1 - log(u), exponent);

    return term1 + term2 + term3 + term4 + term5 + term6 + term7;
  }
  real Q_custom_rng(real alpha, real mu, real u) {
    real p;
    real B;
    real denom_log;
    real numer_log;
    real y;

    // Sample p ~ Uniform(0,1)
    p = uniform_rng(0, 1);

    // Compute intermediate values
    B = -log(mu);
    denom_log = log(1 - log(u));
    numer_log = log(1 - log(p));

    // Compute the quantile
    y = exp( - B * pow(numer_log / denom_log, 1 / alpha) );
    return y;
  }
}

data {
  int<lower=0> n;                 // number of observations
  int<lower=0> M;                 // number of groups
  real<lower=0, upper=1> y[n];    // response
  int<lower=1> K;                 // number of predictors
  matrix[n, K] X;                 // design matrix
  int<lower=1, upper=M> id[n];    // group ID
  vector[K] b0;                    // prior mean for beta
  matrix[K, K] B0;                 // prior covariance for beta
  real<lower=0, upper=1> u;       // fixed u
}

parameters {
  vector[K] beta;                 
  vector<lower=0>[2] sigma_b;     
  matrix[2, M] z_b;               
  cholesky_factor_corr[2] L_b;    
  real<lower=0> alpha;            
}

transformed parameters {
  vector<lower=0, upper=1>[n] mu;
  vector[n] eta_mu;
  matrix[2, M] b;
  
  b = diag_pre_multiply(sigma_b, L_b) * z_b;
  eta_mu = X * beta;
  
  for (i in 1:n)
    mu[i] = inv_logit(eta_mu[i] + b[1, id[i]] + b[2, id[i]] * X[i, 3]);
}

model {
  // Priors
  L_b ~ lkj_corr_cholesky(2.0);
  to_vector(z_b) ~ normal(0, 1);
  beta ~ multi_normal(b0, B0);
  sigma_b ~ normal(0,1);
  alpha ~ exponential(1);
  
  // Likelihood
  for (i in 1:n)
    target += Q_custom_lpdf(y[i] | alpha, mu[i], u);
}
generated quantities {
  matrix[2, 2] Rho;
  vector[n] log_lik;
  vector[n] y_sim;

  // Compute correlation matrix
  Rho = multiply_lower_tri_self_transpose(L_b);

  // Compute log-likelihood for observed data
  for (i in 1:n)
    log_lik[i] = Q_custom_lpdf(y[i] | alpha, mu[i], 0.5);

  // Simulate new data using custom RNG
  for (i in 1:n)
    y_sim[i] = Q_custom_rng(alpha, mu[i], 0.5);
}"

Prior4<-"functions {
  // Custom log-PDF for your QULL distribution
  real Q_custom_lpdf(real y, real alpha, real mu, real u) {
    real term1;
    real term2;
    real term3;
    real term4;
    real term5;
    real term6;
    real term7;
    real exponent;

    // compute common exponent
    exponent = pow(-log(y), alpha) * pow(-log(mu), -alpha);

    // log(alpha / y)
    term1 = log(alpha);
    term2 = -log(y);

    // (-log y)^(alpha - 1)
    term3 = (alpha - 1) * log(-log(y));

    // (-log mu)^(-alpha)
    term4 = -alpha * log(-log(mu));

    // log(log(1 - log u))
    term5 = log(log(1 - log(u)));

    // [(1 - log u)^( -log mu^-alpha )]^( -log y^alpha )
    term6 = exponent * log(1 - log(u));

    // exp{1 - [...] } -> log(exp{1 - [...]}) = 1 - [...]
    term7 = 1 - pow(1 - log(u), exponent);

    return term1 + term2 + term3 + term4 + term5 + term6 + term7;
  }
  real Q_custom_rng(real alpha, real mu, real u) {
    real p;
    real B;
    real denom_log;
    real numer_log;
    real y;

    // Sample p ~ Uniform(0,1)
    p = uniform_rng(0, 1);

    // Compute intermediate values
    B = -log(mu);
    denom_log = log(1 - log(u));
    numer_log = log(1 - log(p));

    // Compute the quantile
    y = exp( - B * pow(numer_log / denom_log, 1 / alpha) );
    return y;
  }
}

data {
  int<lower=0> n;                 // number of observations
  int<lower=0> M;                 // number of groups
  real<lower=0, upper=1> y[n];    // response
  int<lower=1> K;                 // number of predictors
  matrix[n, K] X;                 // design matrix
  int<lower=1, upper=M> id[n];    // group ID
  vector[K] b0;                    // prior mean for beta
  matrix[K, K] B0;                 // prior covariance for beta
  real<lower=0, upper=1> u;       // fixed u
}

parameters {
  vector[K] beta;                 
  vector<lower=0>[2] sigma_b;     
  matrix[2, M] z_b;               
  cholesky_factor_corr[2] L_b;    
  real<lower=0> alpha;            
}

transformed parameters {
  vector<lower=0, upper=1>[n] mu;
  vector[n] eta_mu;
  matrix[2, M] b;
  
  b = diag_pre_multiply(sigma_b, L_b) * z_b;
  eta_mu = X * beta;
  
  for (i in 1:n)
    mu[i] = inv_logit(eta_mu[i] + b[1, id[i]] + b[2, id[i]] * X[i, 3]);
}

model {
  // Priors
  L_b ~ lkj_corr_cholesky(1.0);
  to_vector(z_b) ~ normal(0, 1);
  beta ~ multi_normal(b0, B0);
  sigma_b ~ normal(0,1);
  alpha ~ exponential(1);
  
  // Likelihood
  for (i in 1:n)
    target += Q_custom_lpdf(y[i] | alpha, mu[i], u);
}
generated quantities {
  matrix[2, 2] Rho;
  vector[n] log_lik;
  vector[n] y_sim;

  // Compute correlation matrix
  Rho = multiply_lower_tri_self_transpose(L_b);

  // Compute log-likelihood for observed data
  for (i in 1:n)
    log_lik[i] = Q_custom_lpdf(y[i] | alpha, mu[i], u);

  // Simulate new data using custom RNG
  for (i in 1:n)
    y_sim[i] = Q_custom_rng(alpha, mu[i], 0.5);
}"

Prior5<-"functions {
  // Custom log-PDF for your QULL distribution
  real Q_custom_lpdf(real y, real alpha, real mu, real u) {
    real term1;
    real term2;
    real term3;
    real term4;
    real term5;
    real term6;
    real term7;
    real exponent;

    // compute common exponent
    exponent = pow(-log(y), alpha) * pow(-log(mu), -alpha);

    // log(alpha / y)
    term1 = log(alpha);
    term2 = -log(y);

    // (-log y)^(alpha - 1)
    term3 = (alpha - 1) * log(-log(y));

    // (-log mu)^(-alpha)
    term4 = -alpha * log(-log(mu));

    // log(log(1 - log u))
    term5 = log(log(1 - log(u)));

    // [(1 - log u)^( -log mu^-alpha )]^( -log y^alpha )
    term6 = exponent * log(1 - log(u));

    // exp{1 - [...] } -> log(exp{1 - [...]}) = 1 - [...]
    term7 = 1 - pow(1 - log(u), exponent);

    return term1 + term2 + term3 + term4 + term5 + term6 + term7;
  }
  real Q_custom_rng(real alpha, real mu, real u) {
    real p;
    real B;
    real denom_log;
    real numer_log;
    real y;

    // Sample p ~ Uniform(0,1)
    p = uniform_rng(0, 1);

    // Compute intermediate values
    B = -log(mu);
    denom_log = log(1 - log(u));
    numer_log = log(1 - log(p));

    // Compute the quantile
    y = exp( - B * pow(numer_log / denom_log, 1 / alpha) );
    return y;
  }
}

data {
  int<lower=0> n;                 // number of observations
  int<lower=0> M;                 // number of groups
  real<lower=0, upper=1> y[n];    // response
  int<lower=1> K;                 // number of predictors
  matrix[n, K] X;                 // design matrix
  int<lower=1, upper=M> id[n];    // group ID
  vector[K] b0;                    // prior mean for beta
  matrix[K, K] B0;                 // prior covariance for beta
  real<lower=0, upper=1> u;       // fixed u
}

parameters {
  vector[K] beta;                 
  vector<lower=0>[2] sigma_b;     
  matrix[2, M] z_b;               
  cholesky_factor_corr[2] L_b;    
  real<lower=0> alpha;            
}

transformed parameters {
  vector<lower=0, upper=1>[n] mu;
  vector[n] eta_mu;
  matrix[2, M] b;
  
  b = diag_pre_multiply(sigma_b, L_b) * z_b;
  eta_mu = X * beta;
  
  for (i in 1:n)
    mu[i] = inv_logit(eta_mu[i] + b[1, id[i]] + b[2, id[i]] * X[i, 3]);
}

model {
  // Priors
  L_b ~ lkj_corr_cholesky(1);
  to_vector(z_b) ~ normal(0, 1);
  beta ~ multi_normal(b0, B0);
  sigma_b ~ normal(0,2);
  alpha ~ exponential(0.5);
  
  // Likelihood
  for (i in 1:n)
    target += Q_custom_lpdf(y[i] | alpha, mu[i], u);
}
generated quantities {
  matrix[2, 2] Rho;
  vector[n] log_lik;
  vector[n] y_sim;

  // Compute correlation matrix
  Rho = multiply_lower_tri_self_transpose(L_b);

  // Compute log-likelihood for observed data
  for (i in 1:n)
    log_lik[i] = Q_custom_lpdf(y[i] | alpha, mu[i], u);

  // Simulate new data using custom RNG
  for (i in 1:n)
    y_sim[i] = Q_custom_rng(alpha, mu[i], 0.5);
}"


#########################################
#################Simulation Try#############
############################################
library(MASS)     # for mvrnorm
library(rstan)
library(loo)

set.seed(123)

# ---------------------------
# Simulation settings
# ---------------------------
Nsim <- 100
n <- 100
M <- 15
K <- 3
u <- 0.5

# True parameters
beta_true <- c(0.5, -1, 1.5)
sigma_b_true <- c(0.3, 0.5)
rho_true <- 0.2
alpha_true <- 2

Sigma_b_true <- matrix(c(
  sigma_b_true[1]^2, rho_true * sigma_b_true[1] * sigma_b_true[2],
  rho_true * sigma_b_true[1] * sigma_b_true[2], sigma_b_true[2]^2
), nrow = 2, byrow = TRUE)

# ---------------------------
# Custom quantile and random functions
# ---------------------------
Q_custom <- function(p, alpha, mu, u) {
  if (alpha <= 0) stop("alpha must be > 0")
  if (mu <= 0 || mu >= 1) stop("mu must be in (0,1)")
  
  B <- -log(mu)
  denom_log <- log(1 - log(u))
  numer_log <- log(1 - log(p))
  frac <- numer_log / denom_log
  y <- exp(-B * (frac)^(1 / alpha))
  return(y)
}

r_custom <- function(n, alpha, mu, u) {
  p <- runif(n)
  Q_custom(p, alpha, mu, u)
}

# ---------------------------
# Function to simulate data
# ---------------------------
simulate_data <- function(n, M, beta_true, Sigma_b_true, alpha_true, u) {
  X <- cbind(1, rnorm(n), rnorm(n))
  id <- rep(1:M, length.out = n)
  b_mat <- mvrnorm(M, mu = c(0,0), Sigma = Sigma_b_true)
  
  eta <- X %*% beta_true
  mu <- numeric(n)
  y <- numeric(n)
  
  for (i in 1:n) {
    rand_eff <- b_mat[id[i], 1] + b_mat[id[i], 2] * X[i,3]
    mu[i] <- 1 / (1 + exp(-(eta[i] + rand_eff)))
    y[i] <- r_custom(1, alpha_true, mu[i], u)
  }
  
  list(X = X, y = y, id = id, mu = mu)
}

# Compile Stan models
Model1_compiled <- stan_model("Prior1.stan")
Model2_compiled <- stan_model("Prior2.stan")
Model3_compiled <- stan_model("Prior3.stan")
Model4_compiled <- stan_model("Prior4.stan")
Model5_compiled <- stan_model("Prior5.stan")

### List of models##
models <- list(Model1_compiled, Model2_compiled,Model3_compiled,Model4_compiled,Model5_compiled)

# ---------------------------
# Preallocate storage
# ---------------------------
# Pre-allocate storage
LOOIC_mat <- matrix(NA, nrow = Nsim, ncol = length(models))
WAIC_mat  <- matrix(NA, nrow = Nsim, ncol = length(models))
best_model_LOOIC <- numeric(Nsim)
best_model_WAIC  <- numeric(Nsim)
##############
###Simulation##
###############
for (sim in 1:Nsim) {
  cat("Simulation", sim, "of", Nsim, "\n")
  
  # Simulate data
  data_sim <- simulate_data(n, M, beta_true, Sigma_b_true, alpha_true, u)
  stan_data <- list(
    n = n, M = M, y = data_sim$y, K = K, X = data_sim$X, id = data_sim$id,
    b0 = rep(0, K), B0 = diag(100, K), u = u)
  
  inits1 <- function() {
    list(
      beta    = rnorm(K, 0, 0.05),
      sigma_b = runif(2, 0.05, 0.2),
      z_b     = matrix(rnorm(2 * M, 0, 0.05), nrow = 2, ncol = M),
      L_b     = diag(2) + matrix(rnorm(4, 0, 0.01), nrow = 2),
      alpha   = runif(1, 0.5, 1.8)
    )
  }
  
  loo_values <- numeric(length(models))
  waic_values <- numeric(length(models))
  
  for (m in seq_along(models)) {
    warning_occurred <- FALSE
    
    fit <- tryCatch(
      withCallingHandlers(
        sampling(
          models[[m]],
          data = stan_data,
          seed=1234,
          chains = 2,
          iter = 4000,
          init = inits1,
          control = list(adapt_delta = 0.999, max_treedepth = 17),
          refresh = 0
        ),
        warning = function(w) {
          message("⚠️  Warning in sim ", sim, " model ", m, ": ", conditionMessage(w))
          warning_occurred <<- TRUE
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) {
        message("❌  Error in sim ", sim, " model ", m, ": ", conditionMessage(e))
        NULL
      }
    )
    
    if (warning_occurred || is.null(fit)) {
      cat("⏩ Skipping simulation", sim, "model", m, "due to warning or error\n")
      loo_values[m] <- NA
      waic_values[m] <- NA
      next
    }
    
    # Extract log-lik and compute LOO/WAIC
    log_lik <- extract_log_lik(fit, merge_chains = FALSE)
    loo_res <- tryCatch(loo(log_lik, save_psis = TRUE), error = function(e) NULL)
    waic_res <- tryCatch(waic(log_lik), error = function(e) NULL)
    
    # Store estimates for this model
    loo_values[m] <- if(!is.null(loo_res)) loo_res$estimates["looic", "Estimate"] else NA
    waic_values[m] <- if(!is.null(waic_res)) waic_res$estimates["waic", "Estimate"] else NA
  }
  
  # Store results across models for this simulation
  LOOIC_mat[sim, ] <- loo_values
  WAIC_mat[sim, ]  <- waic_values
  
  # Identify best model for this simulation (ignoring NAs)
  if (all(is.na(loo_values))) {
    best_model_LOOIC[sim] <- NA
  } else {
    best_model_LOOIC[sim] <- which.min(loo_values)
  }
   if (all(is.na(waic_values))) {
    best_model_WAIC[sim] <- NA
  } else {
    best_model_WAIC[sim] <- which.min(waic_values)
  }
}
# Number of models
n_models <- length(models)

# Count best model selections for LOOIC
loo_best_counts <- table(factor(best_model_LOOIC, levels = 1:n_models, exclude = NULL))
loo_best_counts

# Count best model selections for WAIC
waic_best_counts <- table(factor(best_model_WAIC, levels = 1:n_models, exclude = NULL))
waic_best_counts

# Optionally, make a summary table
best_model_summary <- data.frame(
  Model = 1:n_models,
  LOOIC_selected = as.integer(loo_best_counts),
  WAIC_selected  = as.integer(waic_best_counts)
)
print(best_model_summary)
# Summarize model selection##
table(LOOIC = best_model_LOOIC, WAIC = best_model_WAIC)
prop.table(table(best_model_LOOIC))  # % of times model 1 or 2 selected by LOOIC
prop.table(table(best_model_WAIC))   # % of times model 1 or 2 selected by WAIC

LOOIC_mat
WAIC_mat
result <- t(apply(LOOIC_mat, 1, function(x) {
  # Replace NA with Inf for comparison
  x2 <- ifelse(is.na(x), Inf, x)
  
  min_val <- min(x2)            # NA-free min
  col_idx <- which.min(x2)      # Column of min
  
  c(min_val, col_idx)
}))

colnames(result) <- c("MinValue", "Column")
result

col_means <- apply(LOOIC_mat, 2, function(x) summary(x, na.rm = TRUE))
col_means
col_means <- apply(WAIC_mat, 2, function(x) summary(x, na.rm = TRUE))
col_means

















# ---------------------------
# Summarize results
# ---------------------------
best_model_table <- data.frame(
  Prior = 1:6,
  LOOIC_selected = tabulate(best_model_LOOIC),
  WAIC_selected = tabulate(best_model_WAIC)
)
LOOIC_mat
WAIC_mat
write.csv(LOOIC_mat, file = "LOOIC_mat.csv", row.names = FALSE)
write.csv(WAIC_mat, file = "WAIC_mat.csv", row.names = FALSE)

LOOIC_summary <- apply(LOOIC_mat, 2, function(x) c(mean=mean(x,na.rm=TRUE), sd=sd(x,na.rm=TRUE)))
WAIC_summary  <- apply(WAIC_mat, 2, function(x) c(mean=mean(x,na.rm=TRUE), sd=sd(x,na.rm=TRUE)))
row_min <- apply(LOOIC_mat, 1, min, na.rm = TRUE)
row_min_col <- apply(LOOIC_mat, 1, which.min)
min_count <- table(row_min_col)
which(!is.finite(LOOIC_mat), arr.ind = TRUE)
# Ensure we are working on the exact matrix
LOOIC_mat_clean <- LOOIC_mat

# Optional: check for extreme values
range(LOOIC_mat_clean)
LOOIC_mat_clean[!is.finite(LOOIC_mat_clean)] <- NA
which(LOOIC_mat > 0, arr.ind = TRUE)
LOOIC_mat[LOOIC_mat > 0]
# ---------------------------
# Save results
# ---------------------------
write.csv(data.frame(Simulation=1:Nsim, LOOIC_mat, WAIC_mat), "LOOIC_WAIC_all_simulations.csv", row.names=FALSE)

write.csv(best_model_table, "BestPriorCounts.csv", row.names=FALSE)
write.csv(data.frame(t(LOOIC_summary)), "LOOIC_summary.csv")
write.csv(data.frame(t(WAIC_summary)), "WAIC_summary.csv")

list(
  BestPriorCounts = best_model_table,
  LOOIC_summary = LOOIC_summary,
  WAIC_summary = WAIC_summary
)





