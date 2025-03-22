%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   runmodel_withTreatCov.m
%%   目标：在原始的交叉验证脚本中加入治疗 (Treatment) 和协变量 (Covariates)
%%   并对其与微生物特征 (Z) 做统一建模与变量选择
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; rng(2018);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. 读入数据：治疗、协变量、微生物特征，以及响应变量
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 假设以下是模拟/读取方式，请替换成自己的读法:
% load str2_snr0d1.mat  只包含 Z, Y 时，我们需要额外提供 Treat 和 Cov


Treat = csvread('train_treat.csv'); 
Cov   = csvread('train_cov_two.csv');  % c 列
Z     = csvread('train_otu.csv');  % p_micro 列
Y     = csvread('train_bmi.csv');  % 目标变量
[n, c]       = size(Cov);
[~, p_micro] = size(Z);


% 现在 Z 依然是微生物特征矩阵 (n×p_micro)
% Y 是响应变量 (n×1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. 构造设计矩阵 X：[Treat, Cov, log(Z)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Zlog = log(Z + 0.5);          % 避免 log(0) 问题
p_micro = size(Zlog, 2);      % 重新确认 p_micro
X_part = [Cov, Zlog];         % 将协变量和微生物特征先拼一起

% 对 Cov + Zlog 做标准化
[Xstd_part, muX_part, stdX_part] = zscore(X_part);

% 保留 Treatment 原始值，拼到最前面
X = [Treat, Xstd_part];
p_new = 1 + size(Xstd_part, 2);  % 1 + c + p_micro

% 存储标准化参数 (Treatment 不变 => mean=0, std=1)
stand.mux = [0, muX_part]';   
stand.Sx  = [1, stdX_part]';

% 继续对 Y 做标准化
[Y, stand.muy, stand.Sy] = zscore(Y);

fprintf('New design matrix X: size = %d x %d\n', size(X,1), size(X,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. 构造惩罚矩阵 T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4) Construct T
c_sum = 100;
T_treat_cov = eye(1 + c);                      % treatment + covariates
T_micro     = [eye(p_micro); c_sum*ones(1,p_micro)];
Tblock      = blkdiag(T_treat_cov, T_micro);   % block diagonal
T           = Tblock * diag(1./stand.Sx);      % scale

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. 设置 MCMC 采样参数 / 超参数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nburnin = 10000;
niter   = 5000;
% nop   = floor(n/2);  % 看你的 traintest 里是否需要
a0      = -10;        % 用于初始 a
a       = a0 * ones(1, p_new); 

tau  = 1;
nu   = 0;
omega= 0;
seed = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. 构造先验矩阵 Q 并做相应改动
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 如果原先你的 Q 是针对 p 列，现在需改成对 p_new 列
% 以下展示一种思路：Treatment + Cov 不考虑交互，微生物特征在 Q 中赋值
% 
% 对 p 列做了以下过程：
%     Q=0.002*ones(p,p);
%     ... (一些 for 循环给特定位置设为4) ...
%     Q=(Q+Q')/2;
% 现在需拆分:
p_treat_cov = 1 + c;      % 包含 1 列 Treatment, c 列 Cov
p_beta      = p_micro;    % 微生物特征部分

% 先初始化 Q_new => p_new x p_new
Q_new = zeros(p_new, p_new);

% (a) Q 中 treatment + cov 部分，不考虑交互可设为某个常量 (如 2.0) 或 0
%     例如:
b_kappa = 2.0;  % cov 间的交互强度
% 只对 cov 之间(i~=j)赋值 b_kappa，treatment 与 cov 不交互则设 0
for i = 2 : p_treat_cov
    for j = 2 : p_treat_cov
        if i ~= j
            Q_new(i,j) = b_kappa;
        end
    end
end

% (b) Q 中微生物特征部分 => 沿用你之前 0.002 & 4 赋值方式
%     具体根据你的需求
Qbeta = 0.002 * ones(p_beta, p_beta);

% 原脚本：
% for i=180:20:380; for j=(i+20):20:400; Q(i,j)=4; end; end; ...
% 这些循环针对微生物特征之间的特定索引赋值 => 这里需要注意 index 不同
% 
% 举个例子：如果你想复用原先 i/j 范围，需要先做适配:
%   startBeta = 1 + c + 1;  
%   i' = i - (某个偏移量) 
%   j' = j - (某个偏移量)


for i = 1 : p_beta
    for j = 1 : p_beta
        if i~=j && mod(i,20)==0 && j==i+20
            Qbeta(i,j) = 4; 
        end
    end
end
Qbeta = (Qbeta + Qbeta')/2;

% 将 Qbeta 放到 Q_new 的微生物特征区块
startMicro = p_treat_cov + 1; 
Q_new(startMicro:end, startMicro:end) = Qbeta;

% 最后转成稀疏
Q_new = sparse(Q_new);








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6. 执行交叉验证: 调用 traintest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cutoff  = 0.5;
predict = true;
k       = 10;     % 10 折
pl      = false;  % 不绘图

% 若你有真实 beta 需要比较，请将其扩展到 (1+c+p_micro) 大小
% 这里假设无真实 beta，或你已有相应扩展
beta    = zeros(1, p_new);  % 如果无真实值，可用全0

rn=10;
MSEall=zeros(rn,k);
MSEpreall=zeros(rn,k);

FPR_all=zeros(rn,k);
FNR_all=zeros(rn,k);
FPN_all=zeros(rn,k);
FNN_all=zeros(rn,k);
sensitivity_all=zeros(rn,k);
specificity_all=zeros(rn,k);
precision_all=zeros(rn,k);
accuracy_all=zeros(rn,k);

coef_l1_all=zeros(rn,k);
coef_l2_all=zeros(rn,k);
coef_linf_all=zeros(rn,k);

%%%%%roc curve
doroc=true;
sn=0.05;
if doroc
    rFPR=zeros(rn*k,1/sn+1);
    rTPR=zeros(rn*k,1/sn+1);
    rAUC=zeros(rn,k);
end

fprintf('Start cross-validation...\n');
tic
for repeat=1:rn
    if doroc
               [MSEpre,MSEtest,average,sem,FPR,FNR,FPN,FNN,sensitivity,...
         specificity,precision,accuracy,etafinal,roc] = ...
            traintest(k, nburnin, niter, p_new, Y, X, T, beta, a, Q_new, ...
                      n, tau, nu, omega, seed, predict, cutoff, stand, ...
                      pl, [], [], [], doroc);

        % 保存 ROC 信息
        rFPR(((repeat-1)*k+1):(repeat*k),:) = roc.rFPR;
        rTPR(((repeat-1)*k+1):(repeat*k),:) = roc.rTPR;
        rAUC(repeat,:) = roc.rAUC; 
        
    else
        [MSEpre,MSEtest,average,sem,FPR,FNR,FPN,FNN,sensitivity,specificity,precision,accuracy,etafinal]= ...
            traintest(k,nburnin,niter,p_new,Y,X,T,beta,a,Q_new,n,tau,nu,omega,seed,predict,cutoff,stand,pl,[],[],false);
    end
    
    % 记录结果
    MSEall(repeat,:)=MSEtest;
    MSEpreall(repeat,:)=MSEpre;
    FPR_all(repeat,:)=FPR;
    FNR_all(repeat,:)=FNR;
    FPN_all(repeat,:)=FPN;
    FNN_all(repeat,:)=FNN;
    sensitivity_all(repeat,:)=sensitivity;
    specificity_all(repeat,:)=specificity;
    precision_all(repeat,:)=precision;
    accuracy_all(repeat,:)=accuracy;
    
    coef_l1_all(repeat,:)=sum(abs(etafinal-beta),2);
    coef_l2_all(repeat,:)=sqrt(sum(abs(etafinal-beta).^2,2))';
    coef_linf_all(repeat,:)=max(abs(etafinal-beta),[],2);   
end    
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 7. 汇总结果
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
varCell{1} = MSEall;
varCell{2} = FPR_all;
varCell{3} = FNR_all;
varCell{4} = FPN_all;
varCell{5} = FNN_all;
varCell{6} = sensitivity_all;
varCell{7} = specificity_all;
varCell{8} = precision_all;
varCell{9} = accuracy_all;
varCell{10} = coef_l1_all;
varCell{11} = coef_l2_all;
varCell{12} = coef_linf_all;
varCell{13} = MSEpreall;

meanall = cellfun(@mean2, varCell);
meanstd = cellfun(@std2,  varCell) / sqrt(k*rn);

varCell{14} = [meanall; meanstd];

% 如果进行了 ROC 分析，可再做聚合、保存图形等
if doroc
    pathname = pwd;
    FPRm = mean(rFPR,1);
    TPRm = mean(rTPR,1);
    AUCm = mean2(rAUC);
    fprintf('Mean AUC across all folds & repeats = %.3f\n', AUCm);
    % ... (可绘制 ROC 曲线)
end

filename = 'crosstest_withTreatCov.xlsx';
for i = 1:14
    sheetname = strcat('Sheet', num2str(i));
    writematrix(varCell{i}, filename, 'Sheet', sheetname);
end


disp('Done.');
