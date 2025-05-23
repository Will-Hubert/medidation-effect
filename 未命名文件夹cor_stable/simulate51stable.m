
rng(2018)

load str2_snr10.mat

%standadize data
[X,stand.mux,stand.Sx] = zscore(log(Z));
[Y,stand.muy,stand.Sy] = zscore(Y);

%specify penalty matrix
c=100;
T=[eye(p);c*ones(1,p)]*diag(1./stand.Sx);

%set the number of burn-in steps and iterations
nburnin=10000;
niter=5000;

%initialize gamma and setsort
nop=floor(n/2);

ss=100;
al=-30;
au=0;
a0=al:(au-al)/ss:au;

tau=1;
nu=0;
omega=0;
seed=100;

Q=0.002*(ones(p,p)-eye(p));
for i=180:20:380
    for j=(i+20):20:400
        Q(i,j)=4;
    end
end

for i=580:20:780
    for j=(i+20):20:800
        Q(i,j)=4;
    end
end

for i=445:459
    for j=(i+1):460
        Q(i,j)=4;
    end
end

for i=945:959
     for j=(i+1):960
         Q(i,j)=4;
     end
end

for i=45:59
     for j=(i+1):60
         Q(i,j)=4;
     end
end

Q=(Q+Q')/2;

cutoff=0.5;

FP=zeros(1,ss+1);
TN=zeros(1,ss+1);
FN=zeros(1,ss+1);
TP=zeros(1,ss+1);
FPR=zeros(1,ss+1);
TPR=zeros(1,ss+1);
mse=zeros(1,ss+1);
nvs=zeros(1,ss+1);

parfor i=1:(ss+1)
    a=a0(i)*ones(1,p);
    [gamma,betahat,MSE]=gibbsgamma(nburnin,niter,p,nop,Y, X, T, a, Q, n,tau,nu,omega,seed,true,stand,false);
    
    freq=sum(gamma((nburnin+1):(nburnin+niter),:))/niter;
    selectindx=freq>cutoff;
    nvs(i)=sum(selectindx==1);
    fp=sum((gammatrue'==0) & (selectindx==1));
    tn=sum((gammatrue'==0) & (selectindx==0));
    fn=sum((gammatrue'==1) & (selectindx==0));
    tp=sum((gammatrue'==1) & (selectindx==1));
    FPR(i)= fp/(fp+tn);
    TPR(i)= tp/(tp+fn);
    FP(i)=fp;
    TN(i)=tn;
    FN(i)=fn;
    TP(i)=tp;
    mse(i)=mean(MSE((nburnin+1):(nburnin+niter)));
end

figname='str2_snr1_';
%get the directory of your input files:
pathname = pwd;

save('stable.mat','a0','FPR','TPR','FP','TN','FN','TP','mse')

% 创建结果表格
result_table = table( ...
    a0', ...
    TPR', ...
    FPR', ...
    mse', ...
    TP', ...
    FP', ...
    TN', ...
    FN', ...
    nvs', ...
    'VariableNames', {'a0','TPR','FPR','MSE','TP','FP','TN','FN','NumVariablesSelected'});

% 保存为 CSV 文件
csvfile = fullfile(pathname, [figname 'stable.csv']);
writetable(result_table, csvfile);
