%% 1. 设定随机种子，确保可复现
rng(2024);

%% 2. 设定样本数和变量数
n = 100; % 样本数
p_otu = 50;  % 微生物变量数（OTU）
p_cov = 2;    % 额外协变量数
p_treatment = 1; % Treatment 变量数

function r = drchrnd(alpha, n)
    % 生成 n 个 Dirichlet 分布的随机样本
    p = length(alpha);  % 维度
    r = gamrnd(repmat(alpha, n, 1), 1, n, p);  % 生成 Gamma 分布样本
    r = r ./ sum(r, 2);  % 归一化，使得每一行的和为 1
end


%% 3. 生成 X（微生物变量）+ Z（协变量）+ T（Treatment）
% 生成 X，每一行都是一个 Dirichlet 采样，确保行和为 1
alpha = ones(1, p_otu); % 参数向量，所有 OTU 具有相同权重
X = drchrnd(alpha, n);  % 生成 n 行，每行 p_otu 个 OTU 丰度值，总和为 1
Z = randn(n, p_cov);   % 额外协变量（标准正态分布）
T = randi([0, 1], n, p_treatment); % **生成 0/1 二分类变量（治疗变量）**

%% 4. 设定真实的回归系数 beta
beta_otu = [1.2, -0.8, 0.5, zeros(1, p_otu-3)]';  % **前三个微生物变量影响 BMI，其他无影响**
beta_cov = [0.3, 0]';  % **Z1 影响 BMI，Z2 不影响**
beta_treat = 1.5;      % **Treatment 影响 BMI**

%% 5. 生成噪声项（模拟随机误差）
sigma = 0.5; % 设定噪声标准差
epsilon = sigma * randn(n, 1); % 生成噪声

%% 6. 计算 Y（BMI 结果）
Y = X * beta_otu + Z * beta_cov + T * beta_treat + epsilon; % 计算 BMI

%% 7. 查看数据（前 5 行）
disp('Treatment (first 5 samples):'); disp(T(1:5, :));
disp('Covariates (first 5 samples):'); disp(Z(1:5, :));
disp('OTU (first 5 samples):'); disp(X(1:5, :));
disp('BMI (first 5 values):'); disp(Y(1:5));

%% 8. 保存数据到 CSV
writematrix(T, 'train_treat.csv'); % 保存 Treatment 变量
writematrix(Z, 'train_cov_two.csv'); % 保存协变量
writematrix(X, 'train_otu.csv'); % 保存 OTU 变量
writematrix(Y, 'train_bmi.csv'); % 保存 BMI 结果

disp('✅ 模拟数据已生成并保存为 CSV 文件！');

%% 9. 生成 Q 矩阵
% p_otu 已经定义为 50，表示有 50 个 OTU
Q = eye(p_otu);  % 初始化为单位矩阵

% 设定块状结构：
% 前3个 OTU之间相关性较高（0.8）
Q(1:3, 1:3) = 0.8;

% OTU4到OTU6之间中等相关（0.5），如果 p_otu>=6
if p_otu >= 6
    Q(4:6, 4:6) = 0.5;
end

% OTU7到OTU10之间相关性较低（0.3），如果 p_otu>=10
if p_otu >= 10
    Q(7:10, 7:10) = 0.3;
end

% 让剩余 OTU 之间有更低的相关性（0.1）
if p_otu > 10
    for i = 11:p_otu
        for j = 11:p_otu
            if i ~= j
                Q(i, j) = 0.1;
            end
        end
    end
end

% 确保对角线为1
for i = 1:p_otu
    Q(i,i) = 1;
end

% 显示生成的 Q 矩阵
disp('生成的 Q 矩阵为：');
disp(Q);

%% 10. 保存 Q 矩阵到 CSV 文件
writematrix(Q, 'sortcorr.csv');
disp('✅ Q 矩阵已存入 CSV 文件：sortcorr.csv');


%% 1. 设定随机种子，确保可复现
rng(2025);  % 设定不同的随机种子，以区分测试数据

%% 2. 设定样本数和变量数（与训练集保持一致）
n = 100; % 样本数
p_otu = 50;  % 微生物变量数（OTU）
p_cov = 2;    % 额外协变量数
p_treatment = 1; % Treatment 变量数


%% 3. 生成测试数据 X（微生物变量）+ Z（协变量）+ T（Treatment）
% 生成 X，每一行都是一个 Dirichlet 采样，确保行和为 1
alpha = ones(1, p_otu); % 参数向量，所有 OTU 具有相同权重
X_test = drchrnd(alpha, n);  % 生成 n 行，每行 p_otu 个 OTU 丰度值，总和为 1
Z_test = randn(n, p_cov);   % 额外协变量（标准正态分布）
T_test = randi([0, 1], n, p_treatment); % **生成 0/1 二分类变量（治疗变量）**

%% 4. 设定测试集的真实回归系数 beta
beta_otu_test = [1.5, zeros(1, p_otu-1)]';  % **只有第一个 OTU 影响 BMI，其他 OTU 系数为 0**
beta_cov_test = [0.6, -0.4]';  % **两个协变量都影响 BMI**
beta_treat_test = 1.5;  % **Treatment 影响 BMI（与训练集保持一致）**

%% 5. 生成噪声项（模拟随机误差）
sigma = 0.5; % 设定噪声标准差
epsilon_test = sigma * randn(n, 1); % 生成噪声

%% 6. 计算 Y_test（BMI 结果）
Y_test = X_test * beta_otu_test + Z_test * beta_cov_test + T_test * beta_treat_test + epsilon_test; 

%% 7. 查看数据（前 5 行）
disp('Test Treatment (first 5 samples):'); disp(T_test(1:5, :));
disp('Test Covariates (first 5 samples):'); disp(Z_test(1:5, :));
disp('Test OTU (first 5 samples):'); disp(X_test(1:5, :));
disp('Test BMI (first 5 values):'); disp(Y_test(1:5));

%% 8. 保存数据到 CSV
writematrix(T_test, 'test_treat.csv'); % 保存 Treatment 变量
writematrix(Z_test, 'test_cov_two.csv'); % 保存协变量
writematrix(X_test, 'test_otu.csv'); % 保存 OTU 变量
writematrix(Y_test, 'test_bmi.csv'); % 保存 BMI 结果

disp('✅ 测试数据已生成并保存为 CSV 文件！');
