%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. 基本设置
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(2018)

n = 50;         % number of samples
p_comp  = 30;   % number of compositional predictors (microbes)
p_cov   = 2;    % number of extra covariates
p_treat = 1;    % number of treatment variables
p_all   = p_treat + p_cov + p_comp;  % 总维度: 1 + 2 + 30 = 33
snr     = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. 设置哪些变量是真值 [T, Cov, Microbes]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gammatrue = zeros(1, p_all);

gammatrue(1) = 1;    % Treatment
gammatrue(2) = 1;    % Z1 (有效协变量)
gammatrue(3) = 0;    % Z2 (无效协变量)

% 微生物真值位置：从第 4 列开始
microbe_true_index = 3 + [1:3, 6:8];  % shift by 3 for [T, Cov]
gammatrue(microbe_true_index) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. 设置组合成分系数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b_comp = zeros(1, p_comp);
b_comp(1:3) = [1, -0.8, 0.6];
b_comp(6:8) = [-1.5, -0.5, 1.2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. 协变量 & Treatment 系数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_cov   = [1.3; 0];   % Z1 有效，Z2 无效
beta_treat = 1.5;        % Treatment 有效

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. 噪声水平
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigmaX = 1;
sigma  = 1/snr * mean(abs(b_comp(b_comp ~= 0)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. 生成组合成分数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta = zeros(1, p_comp);
theta(1:5) = log(0.5 * p_comp);

[Xorg, beta_comp, epsilon] = gen_comp_simdata_ind( ...
    n, p_comp, gammatrue(4:end), b_comp, theta, sigmaX, sigma);

temp   = exp(2 * Xorg);
Zcomp  = bsxfun(@rdivide, temp, sum(temp, 1));
X_comp = log(Zcomp);  % 微生物 log

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6. 生成协变量 + Treatment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z_extra = randn(n, p_cov);        % 协变量 Z1, Z2
T       = randi([0, 1], n, 1);     % Treatment (二值)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 7. 生成响应变量 Y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y = T * beta_treat + Z_extra * beta_cov + X_comp * beta_comp' + epsilon;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 8. 合成 beta_final 和 gammatrue（顺序一致：T, Cov, Microbes）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_final = [beta_treat; beta_cov; beta_comp(:)];
beta_final = beta_final(:);
gammatrue  = gammatrue(:);

coeffs = table(beta_final, gammatrue, ...
    'VariableNames', {'beta','gammatrue'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 9. 查看信号/噪声
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
signal = mean(abs(b_comp(b_comp ~= 0)));
noise  = sigma;
disp(['signal = ', num2str(signal)])
disp(['noise  = ',  num2str(noise )])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 10. 保存为 CSV 和 .mat 文件
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
type   = 'structure1';
prefix = [type '_snr' num2str(round(snr)) '_n' num2str(n) '_p' num2str(p_comp)];

% 拼接 [T, Cov, Microbes]
X_final = [T, Z_extra, X_comp];  % 最终分析矩阵

% 保存为 CSV
writematrix(X_final, [prefix '_X.csv']);
writematrix(Y,        [prefix '_Y.csv']);
writetable(coeffs,    [prefix '_coeffs.csv']);

% 保存 .mat
save([prefix '.mat'], ...
    'X_final', 'T', 'Z_extra', 'X_comp', 'Y', ...
    'Xorg', 'Zcomp', 'beta_comp', 'b_comp', 'epsilon', 'theta', ...
    'beta_final', 'gammatrue', 'coeffs', ...
    'signal', 'noise');
