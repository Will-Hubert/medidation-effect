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
X_all = [T, Xstd_part];

stand.mux = [0, mu_part];
stand.Sx  = [1, std_part];

[Ystd, stand.muy, stand.Sy] = zscore(Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. 构造惩罚矩阵 Tpen：微生物部分带零和
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c_sum = 100;
T_treat_cov = eye(p_treat + p_cov);
T_micro = [ eye(p_comp); c_sum * ones(1, p_comp) ];
Tblock = blkdiag(T_treat_cov, T_micro);
Tpen = Tblock * diag(1 ./ stand.Sx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. 设置 MCMC 参数与先验向量 a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nburnin = 15000;
niter   = 5000;
nop     = floor(n / 2);

a_treat = 9999;
a_cov   = -7;
a_beta  = -10;
a = [ a_treat, repmat(a_cov,1,p_cov), repmat(a_beta,1,p_comp) ];

tau = 1;
nu  = 0;
omega = 0;
seed = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. 构造结构先验矩阵 Q（协变量之间相关，其他为0）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=0.002*(ones(p,p)-eye(p));

% 协变量之间设相关性（例如 b=2）
b = 2;
for i = 2:(1 + p_cov)
    for j = (i+1):(1 + p_cov)
        Q(i,j) = b;
        Q(j,i) = b;
    end
end

% 微生物模块结构（真变量块）
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

% 🔹 新增：噪声模块结构
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

Q = (Q + Q') / 2;  % 对称化

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6. 运行 Gibbs Gamma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[gamma, betahat, MSE, nselect, Yhat] = gibbsgamma( ...
    nburnin, niter, p, nop, Ystd, X_all, Tpen, ...
    a, Q, n, tau, nu, omega, seed, ...
    true, stand, true);
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 7. 后处理：筛选变量、还原系数、L2误差
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cutoff = 0.5;
freq = mean(gamma((nburnin+1):(nburnin+niter), :), 1);
xindx = find(freq > cutoff);

Xsel = X_all(:, xindx);
Tsel = Tpen(:, xindx);
Ari = Xsel' * Xsel + tau^(-2) * (Tsel' * Tsel);
invAri = Ari \ eye(length(xindx));

betafinal = zeros(1, p);
betafinal(xindx) = stand.Sy * diag(1 ./ stand.Sx(xindx)) * invAri * (Xsel' * Ystd);

coef_l2 = sqrt(sum((betafinal(:) - beta_final).^2));

fprintf('选中变量:'); disp(xindx);
fprintf('选中变量系数和: %.3f\n', sum(betafinal(xindx)));
fprintf('与真值 L2 距离: %.4f\n', coef_l2);

% (可选) 保存结果：writematrix(xindx, 'selected_variables_structure2.csv');