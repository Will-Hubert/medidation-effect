%% 0. 初始化环境
clear; clc; rng(2018);

% 打开并行池（如果未打开）
if isempty(gcp('nocreate'))
    parpool('local');  % 可按需调整进程数
end

%% 1. 加载数据
load('structure2_snr10_n100_pnew1000.mat');

%% 2. 构造设计矩阵 X = [T, Cov, Microbes] ，并标准化
[n, p_comp] = size(X_comp);
p_cov   = size(Z_extra, 2);
p_treat = 1;
p       = p_treat + p_cov + p_comp;

X_part = [Z_extra, X_comp];
[Xstd_part, mu_part, std_part] = zscore(X_part);
X_all = [T, Xstd_part];

stand.mux = [0, mu_part];
stand.Sx  = [1, std_part];
[Ystd, stand.muy, stand.Sy] = zscore(Y);

%% 3. 构造惩罚矩阵 Tpen
c = 100;
lambda_vec = [ones(1, p_treat + p_cov), c * ones(1, p_comp)];
Sx_copy = stand.Sx;  % 为避免 parfor 不支持结构体埋点
Tpen = diag(lambda_vec ./ Sx_copy);

%% 4. MCMC 设置
nburnin = 5000;
niter   = 2000;
nop     = floor(n/2);
tau     = 1;
nu      = 0;
omega   = 0;
seed    = 100;

%% 5. 构造结构先驱 Q
Q = 0.002 * (ones(p,p) - eye(p));
offset = p_treat + p_cov;
for i = 180:20:380, for j = (i+20):20:400, Q(offset+i, offset+j) = 4; end, end
for i = 580:20:780, for j = (i+20):20:800, Q(offset+i, offset+j) = 4; end, end
for i = 120:10:150, for j = (i+10):10:160, Q(offset+i, offset+j) = 4; end, end
for i = 820:10:850, for j = (i+10):10:860, Q(offset+i, offset+j) = 4; end, end
Q = (Q + Q') / 2;

%% 6. 网格扫描 a_cov 和 a_micro
cutoff = 0.5;
a_treat = 9999;
a_cov_grid   = -15:5:0;
a_micro_grid = -15:5:0;
n_cov   = length(a_cov_grid);
n_micro = length(a_micro_grid);
n_total = n_cov * n_micro;

TPR = zeros(n_total,1); FPR = zeros(n_total,1); MSE = zeros(n_total,1); NVS = zeros(n_total,1);
TP  = zeros(n_total,1); FP  = zeros(n_total,1); TN  = zeros(n_total,1); FN  = zeros(n_total,1);

%% 7. 循环扫描
parfor idx = 1:n_total
    [i, j] = ind2sub([n_cov, n_micro], idx);
    a_cov   = a_cov_grid(i);
    a_micro = a_micro_grid(j);

    a = [a_treat, repmat(a_cov, 1, p_cov), repmat(a_micro, 1, p_comp)];

    [gamma, betahat, MSE_vec] = gibbsgamma(nburnin, niter, p, nop, Ystd, X_all, Tpen, ...
        a, Q, n, tau, nu, omega, seed, true, stand, true);

    freq = mean(gamma((nburnin+1):(nburnin+niter), :), 1);
    selectindx = freq > cutoff;

    NVS(idx) = sum(selectindx);
    fp = sum((gammatrue'==0) & (selectindx==1));
    tn = sum((gammatrue'==0) & (selectindx==0));
    fn = sum((gammatrue'==1) & (selectindx==0));
    tp = sum((gammatrue'==1) & (selectindx==1));

    FPR(idx) = fp / (fp + tn + eps);
    TPR(idx) = tp / (tp + fn + eps);
    FP(idx)  = fp;
    TN(idx)  = tn;
    FN(idx)  = fn;
    TP(idx)  = tp;
    MSE(idx) = mean(MSE_vec((nburnin+1):(nburnin+niter)));
end

%% 8. 保存结果
figname = 'str2_snr10_minganstable';
pathname = pwd;

save(fullfile(pathname, 'minganstable.mat'), ...
    'a_cov_grid', 'a_micro_grid', 'TPR', 'FPR', 'MSE', 'NVS', 'TP', 'FP', 'TN', 'FN');

[A_cov_mat, A_micro_mat] = meshgrid(a_cov_grid, a_micro_grid);
result_table = table( ...
    A_cov_mat(:), A_micro_mat(:), ...
    TPR, FPR, MSE, TP, FP, TN, FN, NVS, ...
    'VariableNames', {'a_cov','a_micro','TPR','FPR','MSE','TP','FP','TN','FN','NumSelected'});

writetable(result_table, fullfile(pathname, [figname '.csv']));