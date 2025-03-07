%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   runmodel_bkappa.m: Covariates + Microbes with b * gamma_k^T gamma_k
%%   - 1) Treatment (column 1, always included)
%%   - 2) Covariates (column 2..(1+c)), with b * gamma^T gamma
%%   - 3) Microbes (remaining columns), with Q * gamma^T gamma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; rng(123);

%% =========== 1. Load Data ============

Treat = csvread('train_treat.csv');  % (n×1) Treatment variable
Cov   = csvread('train_cov_two.csv');  % (n×c) c covariates (e.g., c=2)
Z     = csvread('train_otu.csv');  % (n×p_micro) Microbial features
Y     = csvread('train_bmi.csv');  % (n×1) Response variable

[n, c]       = size(Cov);
[~, p_micro] = size(Z);

%% =========== 2. Construct X ============
%    Column 1 = Treatment
%    Columns 2..(1+c) = Covariates
%    Columns (2+c)..(1+c+p_micro) = Microbes
Zlog = log(Z + 0.5);
Xnew = [Treat, Cov, Zlog];
p_new = 1 + c + p_micro;

%% =========== 3. (Optional) Standardize X & Y ============
X_part     = [Cov, Zlog];                 % 大小 (n x (c+p_micro))
[Xstd_part, muX_part, stdX_part] = zscore(X_part);

% 将 Treat 原样拼回前面
% Xstd 就成为了“第一列是原始的 Treat，后面列是标准化后的 Cov、Zlog”
Xstd = [Treat, Xstd_part];

% 手动组合并保存标准化参数，保证维度与后面一致
% 注意：第 1 个元素对应 Treat，不做缩放 => 均值=0, 标准差=1
stand.mux = [0, muX_part]';     % (p_new×1) 列向量
stand.Sx  = [1, stdX_part]';    % (p_new×1) 列向量

% 如果 Y 也需要标准化，可以保持原来的写法
[Ystd, muY, stdY] = zscore(Y);
stand.muy = muY;
stand.Sy  = stdY;

%% =========== 4. Construct T matrix ============
%   - First (1+c) columns => Treatment + Covariates (No sum-to-zero) => identity matrix (1+c)
%   - Microbes => Sum-to-zero constraint => [identity matrix; c * ones(1, p_micro)]
c_sum = 100;
T_treat_cov = eye(1+c);
T_micro     = [eye(p_micro); c_sum * ones(1, p_micro)];

Tblock = blkdiag(T_treat_cov, T_micro);
T = Tblock * diag(1./stand.Sx);  % Scale by 1/std

%% =========== 5. Prior vector a = [a_treat, a_cov, a_beta] ============
%   a(1) = Treatment
%   a(2..(1+c)) = Covariates
%   a((2+c)..end) = Microbes
a_treat  = 100;        % Treatment
a_cov    = -5;       % Covariates
a_beta   = -7.8;     % Microbes

a = [ a_treat,  repmat(a_cov,1,c),  repmat(a_beta,1,p_micro) ];
% length(a) = 1 + c + p_micro

%% =========== 6. Ising matrix Q for microbes (size p_new x p_new) ============
%  Typically, Q is zero for treatment and covariates, with a nonzero block for microbes
Q_full = csvread('sortcorr.csv');  % (p_micro x p_micro)
Q_full(Q_full > 0.9) = 0.9;
Q_full = Q_full + 0.1 * eye(p_micro);
Q_inv  = Q_full \ eye(p_micro);
Q_inv  = Q_inv - diag(diag(Q_inv));
Q_inv(Q_inv > 1)  = 1;
Q_inv(Q_inv < -1) = -1;

Q = zeros(p_new);   % (1 + c + p_micro) x (1 + c + p_micro)
% Assign Q_inv to the microbe section => Rows/cols from (1+c+1) to (1+c+p_micro)
startMicro = 1 + c + 1;
Q(startMicro:end, startMicro:end) = Q_inv;
Q = sparse(Q);

%% =========== 6.1 b_kappa for covariates (Quadratic term) ============
%   b_kappa controls "b * gamma_k^T gamma_k" interactions among covariates
%   cvarIdx => 2..(1+c) (Index range for covariates)
b_kappa = 2.0;   % Example value (can be positive or negative)
cvarIdx = 2 : (1 + c);   % Indexes for covariates

%% =========== 7. MCMC parameters ============
nburnin = 20000;
niter   = 5000;
nop     = floor(n/2);
tau=1; 
nu=0; 
omega=0; 
seed=2020;
predict=true; 
display=true;

%% =========== 8. Call gibbsgamma_bkappa() ============
[gamma_out, betahat_out, MSE_out, nselect_out, Yhat_out] = ...
  gibbsgamma_bkappa(nburnin, niter, p_new, nop, Ystd, Xstd, T, a, Q, ...
    n, tau, nu, omega, seed, predict, stand, display, cvarIdx, b_kappa);

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
Tri= T(:, sel_idx);
Ari= Xri'*Xri + tau^(-2)*(Tri'*Tri);
invAri= Ari\eye(length(sel_idx));
tembeta= invAri * Xri' * Ystd;

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
