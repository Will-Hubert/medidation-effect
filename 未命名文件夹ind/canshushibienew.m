%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. 清理环境, 设置随机数种子
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; rng(2018);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. 加载 .mat 文件
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('structure1_snr10_n50_p30.mat');  
% 包括:
%  X_comp(50×30), Z_extra(50×2), T(50×1), Y(50×1) 等
% 但我们实际上会用 “T,Z_extra,X_comp” 这三个再拼起来，顺序同上

[n, p_comp] = size(X_comp);  % p_comp=30
p_cov       = size(Z_extra, 2); % p_cov=2
p_treat     = 1;               % 1 列 Treatment
p           = p_treat + p_cov + p_comp;  
% => 1 + 2 + 30 = 33

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. 组装设计矩阵 X_all: [T(第1列), Cov(2列), Microbes(30列)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 先把 Cov + Microbes 合并并做 zscore
X_part = [Z_extra, X_comp];              % (n×(2+30)) = (n×32)
[Xstd_part, mu_part, std_part] = zscore(X_part);

% 再把 T (不标准化) 放在最前列
X_all = [ T, Xstd_part ];   % => (n×(1+32)) = (n×33)

% 存储标准化参数
stand.mux = [0, mu_part];       % 长度 1+32
stand.Sx  = [1, std_part];      % 同上

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. 对 Y 做 zscore
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Ystd, stand.muy, stand.Sy] = zscore(Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. 构造分块惩罚矩阵 (仅微生物部分带零和约束)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c_sum = 100;

% 这里的分块要区分：前 (1 + 2) 列是 T + Cov，无零和约束
% 后 30 列是 Microbes，有零和约束

T_treat_cov = eye(p_treat + p_cov);  % => eye(3)
% Microbe 部分:
T_micro = [ eye(p_comp); c_sum * ones(1, p_comp) ];
% => (30+1)×30 = 31×30

% block diagonal
Tblock = blkdiag(T_treat_cov, T_micro);
% => (3+31)×(3+30) = 34×33

% 再乘以 diag(1./stand.Sx)，考虑标准化缩放
Tpen = Tblock * diag(1 ./ stand.Sx);  % => 34×33

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. 其他 MCMC 参数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nburnin = 15000;
niter   = 5000;
nop     = floor(n/2);  
a0      = -10;
a       = a0 * ones(1, p);

tau     = 1;
nu      = 0;
omega   = 0;
seed    = 100;

% 构造 Q
Q = 0.002 * (ones(p, p) - eye(p));
Q = (Q + Q')/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6. 调用 gibbsgamma() 采样
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[gamma, betahat, MSE, nselect, Yhat] = gibbsgamma( ...
    nburnin, niter, p, nop, Ystd, X_all, Tpen, ...
    a, Q, n, tau, nu, omega, seed, ...
    true, ...   % showMsg
    stand, ...  % 标准化参数
    true);      % standardizeY
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 7. 后续分析：筛选变量、还原系数、计算 L2 距离
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cutoff = 0.5;
freq   = mean(gamma((nburnin + 1):(nburnin + niter), :), 1);
xindx  = find(freq > cutoff);

fprintf('已选中的变量下标:\n');
disp(xindx);

% 后验系数 (在标准化条件下估计)
betafinal = zeros(1, p);

Xsel  = X_all(:, xindx);
Tsel  = Tpen(:, xindx);
Ari   = Xsel'*Xsel + tau^(-2)*(Tsel'*Tsel);
invAri= Ari \ eye(size(Xsel, 2));

% 回到原尺度
betafinal(xindx) = stand.Sy * diag(1./stand.Sx(xindx)) * (invAri * (Xsel' * Ystd));

% 若真值在 mat 中是 beta_final(33×1)，可比较 L2
coef_l2 = sqrt(sum( (betafinal(:) - beta_final).^2 ));

fprintf('选中变量的系数和: %.3f\n', sum(betafinal(xindx)));
fprintf('与真值 L2 距离: %.4f\n', coef_l2);

% (可选) 写出 xindx
writematrix(xindx, 'selected_variables.csv');
