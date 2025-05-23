
rng(2018)

load str1_snr10_n50_p30.mat

%standadize data
[X,stand.mux,stand.Sx] = zscore(log(Z));
[Y,stand.muy,stand.Sy] = zscore(Y);

%specify penalty matrix
c=100;
T=[eye(p);c*ones(1,p)]*diag(1./stand.Sx);

%set the number of burn-in steps and iterations
nburnin=15000;
niter=5000;

%initialize gamma and setsort
nop=floor(n/2);

a0=-10;
a=a0*ones(1,p);

tau=1;
nu=0;
omega=0;
seed=100;

Q=0.002*(ones(p,p)-eye(p));
% for i=1:7
%     for j=(i+1):8
%         Q(i,j)=4;
%     end
% end

Q=(Q+Q')/2;

tic
[gamma,betahat,MSE,nselect,Yhat]=gibbsgamma(nburnin,niter,p,nop,Y, X, T, a, Q, n,tau,nu,omega,seed,true,stand,true);
toc



%%%%%%%%%%%Describe blocks%%%%%%
x=[1:8];                  %#initialize x array
y1=0.6+zeros(1,length(x));                      %#create first curve
y2=0.8+zeros(1,length(x));                   %#create second curve
X1=[x,fliplr(x)];                %#create continuous x value array for plotting
Y1=[y1,fliplr(y2)];              %#create y values for out and then back

% x=[580:20:800];                  %#initialize x array
% y1=0.6+zeros(1,length(x));                      %#create first curve
% y2=0.8+zeros(1,length(x));                   %#create second curve
% X2=[x,fliplr(x)];                %#create continuous x value array for plotting
% Y2=[y1,fliplr(y2)];  %#create y values for out and then back
% 
% x=[445:460];                  %#initialize x array
% y1=0.6+zeros(1,length(x));                      %#create first curve
% y2=0.8+zeros(1,length(x));                   %#create second curve
% X3=[x,fliplr(x)];                %#create continuous x value array for plotting
% Y3=[y1,fliplr(y2)];  %#create y values for out and then back
% 
% 
% x=[945:960];                  %#initialize x array
% y1=0.6+zeros(1,length(x));                      %#create first curve
% y2=0.8+zeros(1,length(x));                   %#create second curve
% X4=[x,fliplr(x)];                %#create continuous x value array for plotting
% Y4=[y1,fliplr(y2)];              %#create y values for out and then back
% 
% x=[45:60];                    %#initialize x array
% y1=0.6+zeros(1,length(x));                      %#create first curve
% y2=0.8+zeros(1,length(x));                   %#create second curve
% X5=[x,fliplr(x)];                %#create continuous x value array for plotting
% Y5=[y1,fliplr(y2)];              %#create y values for out and then back



%%%%%%%%%%%plot figure%%%%%%%%%%%

figname='sim51';

cutoff=0.5;

freq=sum(gamma((nburnin+1):(nburnin+niter),:))/niter;
top=freq(freq>cutoff);
temp=1:p;
xindx=temp(freq>cutoff);

betafinal=zeros(1,p);
Xri=X(:,xindx);
Tri=T(:,xindx);
Ari=Xri'*Xri+tau^(-2)*(Tri'*Tri);
invAri=Ari\eye(size(Xri,2));
betafinal(xindx)=stand.Sy*diag(1./stand.Sx(xindx))*invAri*X(:,xindx)'*Y;
coef_l2=sqrt(sum(abs(betafinal'-beta).^2));

xindx
sum(betafinal(xindx))
coef_l2
writematrix(xindx, fullfile(pathname, 'selected_variables.csv'));
