function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve_performance(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
    % iW = 0: exact line
    % iW = 1: BLS with WC
    % iW = 2: BLS with SWC

    % rho: BLS parameter

    % isd = 1: GM
    % isd = 3: BFGS
    % isd = 7: SGM
   
   
    %----------------------------------------------------------------------
    %Training data set
    [Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq); %tr_p number of images

    %----------------------------------------------------------------------
    %Test data set
    te_freq = tr_freq/10; % we lower the number of target numbers
    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, te_freq);

    %----------------------------------------------------------------------
    %Functions
    sig = @(X) 1./(1 + exp(-X));
    y = @(X,w) sig(w'*sig(X));
    L = @(w, Xtr, ytr, la) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+(la*norm(w)^2)/2;
    gL = @(w, Xtr, ytr, la) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;
    %----------------------------------------------------------------------
    %Optimization
    n = size(Xtr,1); % corresponding to the # of pixels, the number of variables of entry information
    w1 = zeros(n,1); %the start is in 0
    if isd==1 || isd==3
        x1 = w1; %the variable we want to minimize is w
        f = @(w) L(w, Xtr, ytr, la); g = @(w) gL(w, Xtr, ytr, la); h= @(w) 0; %no second derivatives methods are applied
        almax = 1; %predefinite starting alfa max value
        almin=0; rho=0; iW=0; %parameters not used as we use isl=3, advanced linear search
        delta=0;    %parameters not used as we use algorithm icg = 1,3,7

        tic % start of elapsed time
        [wk,dk,alk,betak,iWk,Hk,tauk] = uo_solve_performance(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,ils,ialmax,kmaxBLS,epsal);
        tex = toc; %final of elapsed time

        wo = wk(:,end); % the best one is the last iteration

    elseif isd == 7 % Stochastic Gradient Method (SGM)
        rng(sg_seed); %we set the seed

        tic % start of elapsed time
        [wo,wk,dk,alk] = SGM(w1,la,L,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);
        tex = toc; %final of elapsed time
    end
    
    %----------------------------------------------------------------------
    %Results of best wo
    niter = size(wk,2); %store number of iterations
    fo = L(wo, Xtr, ytr, la); %final minimal error of the model
    %----------------------------------------------------------------------
    %Accuracy

    syms y_var_tr y_tr;
    y_var_tr = sym(round(y(Xtr,wo)));
    y_tr = sym(ytr);
    tr_acc = 100/tr_p * sum(kroneckerDelta(y_var_tr,y_tr));

    syms y_var_te y_te;
    y_var_te = sym(round(y(Xte,wo)));
    y_te = sym(yte);
    te_acc = 100/te_q * sum(kroneckerDelta(y_var_te,y_te));

end

