%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. 初始化环境
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; rng(2018);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. 加载 structure2 数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('structure2_snr10_n100_pnew1000.mat');
% 已包含：X_comp, Z_extra, T, Y, beta_final, gammatrue

[n, p_comp] = size(X_comp);      % p_comp = 1000
p_cov       = size(Z_extra, 2);  % p_cov = 2
p_treat     = 1;                 % Treatment
p           = p_treat + p_cov + p_comp;  % p = 1003

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. 构造设计矩阵 X = [T, Cov, Microbes]，并标准化
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_part = [Z_extra, X_comp];
[Xstd_part, mu_part, std_part] = zscore(X_part);
X_all = [T, Xstd_part];  % 主设计矩阵

stand.mux = [0, mu_part];
stand.Sx  = [1, std_part];

[Ystd, stand.muy, stand.Sy] = zscore(Y);

c_sum = 100;
T_treat_cov = eye(p_treat + p_cov);
T_micro = [ eye(p_comp); c_sum * ones(1, p_comp) ];
Tblock = blkdiag(T_treat_cov, T_micro);
Tpen = Tblock * diag(1 ./ stand.Sx);  % 惩罚矩阵

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. MCMC 参数与结构先验设定
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nburnin = 10000;
niter   = 5000;
nop     = floor(n/2);

sc=100; sl=0; su=3;
ss = linspace(sl, su, sc+1);

a_treat = 9999;
a_cov   = -50;
a_beta  = -16.5;
a = [ a_treat, repmat(a_cov,1,p_cov), repmat(a_beta,1,p_comp) ];

tau = 1;
seed = 100;

Q = zeros(p);  % 初始化结构先验矩阵
b = 2;
for i = 2:(1 + p_cov)
    for j = (i+1):(1 + p_cov)
        Q(i,j) = b;
        Q(j,i) = b;
    end
end

offset = 1 + p_cov;
for i = 180:20:380
    for j = (i+20):20:400
        Q(offset+i, offset+j) = 4;
        Q(offset+j, offset+i) = 4;
    end
end
for i = 580:20:780
    for j = (i+20):20:800
        Q(offset+i, offset+j) = 4;
        Q(offset+j, offset+i) = 4;
    end
end
for i = 120:10:150
    for j = (i+10):10:160
        Q(offset+i, offset+j) = 4;
        Q(offset+j, offset+i) = 4;
    end
end
for i = 820:10:850
    for j = (i+10):10:860
        Q(offset+i, offset+j) = 4;
        Q(offset+j, offset+i) = 4;
    end
end
Q = (Q + Q') / 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. 执行变量选择主循环
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cutoff=0.5;
FP=zeros(1,sc+1); TN=zeros(1,sc+1); FN=zeros(1,sc+1); TP=zeros(1,sc+1);
FPR=zeros(1,sc+1); TPR=zeros(1,sc+1); mse=zeros(1,sc+1);
nvs=zeros(1,sc+1); coef_l2=zeros(1,sc+1);
alpha0=zeros(1,sc+1); beta0=zeros(1,sc+1);

parfor i=1:(sc+1)
    nu=ss(i);
    omega=ss(i);
    tic
    [gamma, betahat, MSE, ~, ~] = gibbsgamma( ...
        nburnin, niter, p, nop, Ystd, X_all, Tpen, ...
        a, Q, n, tau, nu, omega, seed, ...
        true, stand, true);
    toc

    freq = mean(gamma((nburnin+1):(nburnin+niter), :), 1);
    xindx = find(freq > cutoff);

    betafinal = zeros(p,1);
    Xri = X_all(:, xindx);
    Tri = Tpen(:, xindx);
    Ari = Xri' * Xri + tau^(-2) * (Tri' * Tri);
    invAri = Ari \ eye(length(xindx));
    betafinal(xindx) = stand.Sy * diag(1 ./ stand.Sx(xindx)) * invAri * X_all(:,xindx)' * Ystd;

    coef_l2(i) = sqrt(sum((betafinal - beta_final).^2));

    Yhat = X_all * diag(stand.Sx) * betafinal;
    SSE = (Y - Yhat).^2;
    SSE = sum(SSE);
    alpha0(i) = (n + nu)/2;
    beta0(i)  = (SSE + nu * omega)/2;

    selectindx = freq > cutoff;
    nvs(i) = sum(selectindx);

    fp = sum((gammatrue'==0) & (selectindx==1));
    tn = sum((gammatrue'==0) & (selectindx==0));
    fn = sum((gammatrue'==1) & (selectindx==0));
    tp = sum((gammatrue'==1) & (selectindx==1));

    FPR(i) = fp / (fp + tn);
    TPR(i) = tp / (tp + fn);
    FP(i) = fp; TN(i) = tn; FN(i) = fn; TP(i) = tp;
    mse(i) = mean(MSE((nburnin+1):(nburnin+niter)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. 保存 .mat + .csv 结果文件
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figname = 'str2_snr10_mingan';
pathname = pwd;

save('canmingan.mat', 'ss', 'FPR', 'TPR', 'FP', 'TN', 'FN', 'TP', ...
     'mse', 'nvs', 'coef_l2', 'alpha0', 'beta0');

T_result = table(ss', FPR', TPR', FP', TN', FN', TP', mse', nvs', coef_l2', alpha0', beta0', ...
    'VariableNames', {'nu_omega','FPR','TPR','FP','TN','FN','TP','MSE','NumSelected','CoefL2','Alpha0','Beta0'});

writetable(T_result, fullfile(pathname, [figname '_results.csv']));
