function [gamma,betahat,MSE,nselect,Yhat] = gibbsgamma_bkappa( ...
    nburnin, niter, p, nop, Y, X, T, a, Q, n, tau, nu, omega, seed, ...
    predict, stand, display, cvarIdx, b_kappa )

% gibbsgamma_bkappa.m
%   - Adds a b_kappa * gamma_k^T gamma_k for covariates in cvarIdx
%   - Microbes in Q (the rest), Treatment=1 forced in

rng(seed);

index = randsample(1:p, nop);
index = sort(index);
nop = length(index);

% init gamma
gamma = zeros(nburnin + niter + 1, p);
gamma(1,1) = 1;             % force treatment
gamma(1, index) = 1;

if isempty(stand)
    invSx = eye(p);
    Sy    = 1;
    Yobs  = Y;
else
    invSx = diag(1./stand.Sx);
    Sy    = stand.Sy;
    Yobs  = Y.*stand.Sy + stand.muy;
end

proposeindx = randsample(1:p, nburnin+niter+1, true);

% init Ari
Xri = X(:, index);
Tri = T(:, index);
Ari = Xri'*Xri + tau^(-2)*(Tri'*Tri);
invAri = Ari \ eye(nop);
Lri = chol(invAri,'lower');

keep.resAi = Y'*Y - Y'*Xri*invAri*Xri'*Y;
keep.sqrtdetinvAi = sum(log(diag(Lri)));

if predict
    betahat = zeros(p, nburnin+niter+1);
    tembeta = invAri * Xri' * Y;
    betahat(index,1) = Sy * invSx(index,index)*tembeta;

    Yhat = zeros(n, nburnin+niter+1);
    Yhat(:,1) = stand.muy + Sy * X(:,index)*tembeta;

    MSE = zeros(1, nburnin+niter+1);
    MSE(1) = 1/n * (Yobs - Yhat(:,1))'*(Yobs - Yhat(:,1));
    nselect = zeros(1, nburnin+niter+1);
    nselect(1) = nop;
end

if display
    disp('Gibbs Sampling with Cov + Micro + b_kappa: Start...');
    k=1;
end

for i = 1 : (nburnin + niter)
    gamma(i+1, :) = gamma(i, :);
    gamma(i+1, 1) = 1;  % treatment forced in

    if predict
        betahat(:, i+1) = zeros(p,1);
    end

    currVar = proposeindx(i);
    flag = any(index==currVar);

    if flag
        % remove
        indxtemp = index(index~=currVar);
        Xri = X(:, indxtemp);
        Xi  = X(:, currVar);
        XIi = [Xri, Xi];

        Tri = T(:, indxtemp);
        Ti  = T(:, currVar);
        TIi = [Tri, Ti];

        invAritemp = invAri;
        tn = length(index);
        seq=1:tn;
        idx = repmat({':'}, ndims(invAri), 1);

        if currVar == max(index)
            idx{1}=seq; idx{2}=seq;
        elseif currVar == min(index)
            idx{1}=[2:tn,1]; idx{2}=[2:tn,1];
        else
            ti=seq(index==currVar);
            idx{1}=[1:(ti-1),(ti+1):tn,ti];
            idx{2}=[1:(ti-1),(ti+1):tn,ti];
        end
        invAitemp=invAri(idx{:});
        invAri=invAitemp(1:(tn-1),1:(tn-1)) - invAitemp(1:(tn-1),tn)*invAitemp(tn,tn)^(-1)*invAitemp(tn,1:(tn-1));

        [F, keep] = BayesFactor(Y,Xri,Xi,XIi,Tri,Ti,TIi,invAri,n,tau,nu,omega,flag,keep);

        if currVar==1
            % treatment: forced
            newgamma=1;
        elseif ismember(currVar, cvarIdx)
            % cov => a + b_kappa
            gammaCov = gamma(i, cvarIdx);  
            sumCov   = sum(gammaCov);  
            minusSelf= gamma(i, currVar);
            pairPart = b_kappa * (sumCov - minusSelf);  % b*(# other cov selected)
            
            linearPart= a(currVar);
            priorVal  = linearPart + pairPart;
            
            pgammai1  = exp(priorVal);
            pcond     = 1/(1 + F^(-1)/pgammai1);
            newgamma  = binornd(1, pcond);
        else
            % microbes => a + Q
            priorVal= a(currVar) + Q(currVar, indxtemp)*gamma(i,indxtemp)';
            pgammai1= exp(priorVal);
            pcond   = 1/(1 + F^(-1)/pgammai1);
            newgamma= binornd(1, pcond);
        end
        gamma(i+1,currVar)= newgamma;

        if (newgamma==0)&&(tn>1)
            index=indxtemp;
            keep.resAi=keep.resAri;
            keep.sqrtdetinvAi=keep.sqrtdetinvAri;
            if predict
                tembeta=invAri*Xri'*Y;
                betahat(index, i+1)=Sy*invSx(index,index)*tembeta;
                Yhat(:,i+1)=stand.muy + Sy*X(:,index)*tembeta;
                MSE(i+1)=1/n*(Yobs-Yhat(:,i+1))'*(Yobs-Yhat(:,i+1));
                nselect(i+1)=tn-1;
            end
        else
            invAri=invAritemp;
            if predict
                betahat(:, i+1)=betahat(:, i);
                Yhat(:, i+1)=Yhat(:, i);
                MSE(i+1)=MSE(i);
                nselect(i+1)=nselect(i);
            end
        end

    else
        % add
        indxtemp = index;
        Xri = X(:, indxtemp);
        Xi  = X(:, currVar);
        XIi = [Xri, Xi];

        Tri = T(:, indxtemp);
        Ti  = T(:, currVar);
        TIi = [Tri, Ti];

        [F, keep, invAi] = BayesFactor(Y,Xri,Xi,XIi,Tri,Ti,TIi,invAri,n,tau,nu,omega,flag,keep);

        if currVar==1
            newgamma=1;
        elseif ismember(currVar, cvarIdx)
            gammaCov= gamma(i, cvarIdx);
            sumCov  = sum(gammaCov);
            minusSelf= gamma(i, currVar);
            pairPart= b_kappa * (sumCov - minusSelf);
            
            linearPart= a(currVar);
            priorVal  = linearPart + pairPart;
            
            pgammai1= exp(priorVal);
            pcond   = 1/(1 + F^(-1)/pgammai1);
            newgamma= binornd(1,pcond);
        else
            priorVal= a(currVar) + Q(currVar, indxtemp)*gamma(i, indxtemp)';
            pgammai1= exp(priorVal);
            pcond   = 1/(1 + F^(-1)/pgammai1);
            newgamma= binornd(1, pcond);
        end
        gamma(i+1,currVar)= newgamma;

        if newgamma==1
            index=sort([indxtemp, currVar]);
            invAri=invAi;
            tn=length(index);
            seq=1:tn;
            idx = repmat({':'}, ndims(invAri), 1);

            if currVar>max(indxtemp)
                idx{1}=seq; idx{2}=seq;
            elseif currVar<min(indxtemp)
                idx{1}=[tn,1:(tn-1)]; idx{2}=[tn,1:(tn-1)];
            else
                ti=seq(index==currVar);
                idx{1}=[1:(ti-1),tn,ti:(tn-1)];
                idx{2}=[1:(ti-1),tn,ti:(tn-1)];
            end
            invAri=invAri(idx{:});

            if predict
                tembeta=invAri * X(:,index)'*Y;
                betahat(index, i+1)=Sy*invSx(index,index)*tembeta;
                Yhat(:, i+1)=stand.muy + Sy*X(:,index)*tembeta;
                MSE(i+1)=1/n*(Yobs-Yhat(:,i+1))'*(Yobs-Yhat(:,i+1));
                nselect(i+1)=tn;
            end
        else
            keep.resAi=keep.resAri;
            keep.sqrtdetinvAi=keep.sqrtdetinvAri;
            if predict
                betahat(:, i+1)=betahat(:, i);
                Yhat(:, i+1)=Yhat(:, i);
                MSE(i+1)=MSE(i);
                nselect(i+1)=nselect(i);
            end
        end
    end

    if display
        if mod(k,500)==0
            disp(['iteration: ', num2str(k)]);
        end
        k=k+1;
    end
end

if display
    disp('Gibbs ends. Frequency of selected variables:');
    sum(gamma(nburnin:(nburnin+niter),:),1)
end
end
