%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   runmodel_Qmatrix.m
%%   Construct a Q matrix containing all interactions:
%%     1) The Treatment variable is processed separately (whether to fix it as 1 is up to you)
%%     2) Interactions among covariates => Q_kappa_kappa = b
%%     3) Interactions among microbial features => Q_beta_beta = Q_full
%%     4) Interactions between covariates and microbial features => 0 (or another matrix if needed)
%%
%%   Then call the original gibbsgamma.m without any modifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; rng(123);

%% 1) Read data
Treat = csvread('train_treat.csv');       % (n×1) If present
Cov   = csvread('train_cov_two.csv');     % (n×c) c non-microbial covariates
Z     = csvread('train_otu.csv');         % (n×p_micro) p_micro microbial features
Y     = csvread('train_bmi.csv');         % (n×1) response variable

[n, c]       = size(Cov);
[~, p_micro] = size(Z);

%% 2) Construct X: [Treat, Cov, Zlog]
%   If you want Treatment to be in the first column, then ...
Zlog = log(Z + 0.5);
Xnew = [Treat, Cov, Zlog];   % [n×(1 + c + p_micro)]
p_new = 1 + c + p_micro;

%% =========== 3. (Optional) Standardize X & Y ============
X_part = [Cov, Zlog];                       % size: (n x (c + p_micro))
[Xstd_part, muX_part, stdX_part] = zscore(X_part);

% Concatenate original Treatment back to the front
% Xstd => first column is the original Treat, followed by standardized Cov and Zlog
Xstd = [Treat, Xstd_part];

% Manually combine and save the standardization parameters so dimensions match later
% Note: the 1st element corresponds to Treatment, not scaled => mean=0, std=1
stand.mux = [0, muX_part]';     % (p_new×1) column vector
stand.Sx  = [1, stdX_part]';    % (p_new×1) column vector

% If Y also needs standardization, keep the same approach as before
[Ystd, muY, stdY] = zscore(Y);
stand.muy = muY;
stand.Sy  = stdY;

%% 4) Construct T
c_sum = 100;
T_treat_cov = eye(1 + c);                      % treatment + covariates
T_micro     = [eye(p_micro); c_sum*ones(1,p_micro)];
Tblock      = blkdiag(T_treat_cov, T_micro);   % block diagonal
T           = Tblock * diag(1./stand.Sx);      % scale

%% 5) Construct a
a_treat = 0;    % treatment
a_cov   = -5;   % covariates
a_beta  = -7.8; % microbial features
a = [ a_treat, repmat(a_cov,1,c), repmat(a_beta,1,p_micro) ];
% length(a) = p_new

%% 6) Construct Q matrix
%    - Q(1,1) is for Treatment, generally set to 0 (because Treatment is fixed or we do not consider interactions)
%    - Q_{kappa,kappa} = b (interactions among covariates)
%    - Q_{beta,beta} = Q_full (interactions among microbial features)
%    - Q_{kappa,beta} = 0 (not considering interactions between covariates and microbial features)

% step 6.1: Read the microbial Q_full
Q_full = csvread('sortcorr.csv');   % p_micro x p_micro
Q_full(Q_full > 0.9) = 0.9;
Q_full = Q_full + 0.1*eye(p_micro);
Q_inv  = Q_full \ eye(p_micro);  % matrix inverse
Q_inv  = Q_inv - diag(diag(Q_inv));
Q_inv(Q_inv > 1)  = 1;
Q_inv(Q_inv < -1) = -1;

% step 6.2: Construct the Q matrix (p_new x p_new)
Q = zeros(p_new, p_new);

% Hyperparameter b
b_kappa = 2.0;

% a) Fill treatment and others: generally set to 0
% b) Fill covariates among themselves (rows and cols 2..(1+c))
for i = 2:(1+c)
    for j = 2:(1+c)
        if i ~= j
            Q(i,j) = b_kappa;
        end
    end
end

% c) Fill microbial features (rows and cols (2+c)..p_new)
startMicro = 1 + c + 1;
Q(startMicro:end, startMicro:end) = Q_inv;

% d) Covariate-microbial interactions (set to 0 if not considered)
% Q(2:(1+c), startMicro:end) = 0;
% Q(startMicro:end, 2:(1+c)) = 0;

Q = sparse(Q);

%% 7) Run MCMC
nburnin = 20000;
niter   = 5000;
nop     = floor(n/2);
tau     = 1;
nu      = 0;
omega   = 0;
seed    = 2020;
predict = true;
display = true;

% Use the original gibbsgamma.m with no modifications
[gamma_out, betahat_out, MSE_out, nselect_out, Yhat_out] = ...
  gibbsgamma(nburnin, niter, p_new, nop, Ystd, Xstd, T, a, Q, ...
    n, tau, nu, omega, seed, predict, stand, display);

%% =========== 9. Summaries ============
freq = mean(gamma_out((nburnin+1):(nburnin+niter), :), 1);
disp('Variable selection frequency:');
disp(freq);

cutoff = 0.5;
sel_idx = find(freq > cutoff);
disp(['Selected variables freq>', num2str(cutoff), ' =>']);
disp(sel_idx);

%% =========== 10. Final Coefficients ============
betafinal = zeros(1, p_new);
Xri = Xstd(:, sel_idx);
Tri = T(:, sel_idx);
Ari = Xri'*Xri + tau^(-2)*(Tri'*Tri);
invAri = Ari\eye(length(sel_idx));
tembeta = invAri * Xri' * Ystd;

betafinal(sel_idx) = stand.Sy .* (1./stand.Sx(sel_idx)) .* tembeta;
disp('Final Coefficients:');
disp(betafinal);

fprintf('Treatment = %.3f\n', betafinal(1));
for cc = 1 : c
    fprintf('Covariate %d = %.3f\n', cc, betafinal(1+cc));
end

Yhat_std = Xri * tembeta;
Yhat_ori = stand.muy + stand.Sy * Yhat_std;
TrainMSE = mean((Y - Yhat_ori).^2);
fprintf('Train MSE = %.3f\n', TrainMSE);
