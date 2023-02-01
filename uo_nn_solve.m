function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
    % iW = 0: exact line
    % iW = 1: BLS with WC
    % iW = 2: BLS with SWC

    % rho: BLS parameter

    % isd = 1: GM
    % isd = 3: BFGS
    % isd = 7: SGM

    fprintf('[uo-nn-solve]------------------------------------------------\n');
    fprintf('[uo-nn-solve]----Pattern recognition with neural networks----\n');
    fprintf('[uo-nn-solve] %s\n', datetime);
    fprintf('[uo-nn-solve]------------------------------------------------\n');
   
   
    %----------------------------------------------------------------------
    %Training data set
    
    fprintf('[uo-nn-solve]  Training data set generation:\n');
    [Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq); %tr_p number of images
    fprintf('[uo-nn-solve]      num_target  = %i\n', num_target);
    fprintf('[uo-nn-solve]      tr_freq     = %4.2f\n', tr_freq);
    fprintf('[uo-nn-solve]      tr_p (#obs) = %i\n', tr_p);
    fprintf('[uo-nn-solve]      tr_seed     = %i\n', tr_seed);
    %----------------------------------------------------------------------
    %Test data set
    fprintf('[uo-nn-solve]  Test data set generation:\n');
    te_freq = tr_freq/10; % we lower the number of target numbers
    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, te_freq);
    fprintf('[uo-nn-solve]      te_freq     = %4.2f\n', te_freq);
    fprintf('[uo-nn-solve]      te_q (#obs) = %i\n', te_q);
    fprintf('[uo-nn-solve]      te_seed     = %i\n', te_seed);
    %----------------------------------------------------------------------
    %Functions
    sig = @(X) 1./(1 + exp(-X));
    y = @(X,w) sig(w'*sig(X));
    L = @(w, Xtr, ytr, la) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+(la*norm(w)^2)/2;
    gL = @(w, Xtr, ytr, la) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;
    %----------------------------------------------------------------------
    %Optimization
    fprintf('[uo-nn-solve]  Optimization:\n');
    fprintf('[uo-nn-solve]      L2 reg. lambda  = %4.4f\n', la);

    n = size(Xtr,1); % corresponding to the # of pixels, the number of variables of entry information
    w1 = zeros(n,1); %the start is in 0
    if isd==1 || isd==3
        fprintf('[uo-nn-solve]      epsG= %+3.1e, kmax= %i\n', epsG, kmax);
        fprintf('[uo-nn-solve]      ils= %i, ialmax= %i, kmaxBLS= %i, epsBLS= %+3.1e\n', ils, ialmax, kmaxBLS, epsal);
        fprintf('[uo-nn-solve]      c1= %4.2f, c2= %4.2f, isd= %i\n', c1, c2, isd);

        x1 = w1; %the variable we want to minimize is w
        f = @(w) L(w, Xtr, ytr, la); g = @(w) gL(w, Xtr, ytr, la); h= @(w) 0; %no second derivatives methods are applied
        almax = 1; %predefinite starting alfa max value
        almin=0; rho=0; iW=0; %parameters not used as we use isl=3, advanced linear search
        delta=0;    %parameters not used as we use algorithm icg = 1,3,7

        tic % start of elapsed time
        [wk,dk,alk,betak,iWk,Hk,tauk] = uo_solve(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,ils,ialmax,kmaxBLS,epsal);
        tex = toc; %final of elapsed time

        wo = wk(:,end); % the best one is the last iteration

    elseif isd == 7 % Stochastic Gradient Method (SGM)
        fprintf('[uo-nn-solve]      sg_al0= %4.2f, sg_be=%4.2f, sg_ga=%4.2f\n', sg_al0,sg_be, sg_ga);
        fprintf('[uo-nn-solve]      sg_emax= %i, sg_ebest= %i, isd= %i\n', sg_emax, sg_ebest, isd);
        rng(sg_seed); %we set the seed

        tic % start of elapsed time
        [wo,wk,dk,alk] = SGM(w1,la,L,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);
        tex = toc; %final of elapsed time
        
        logfreq = ceil(size(wk,2)/10);
        %logfreq = 1;
        iWk = []; Hk = []; tauk = [];
        f = @(w) L(w, Xtr, ytr, la); g = @(w) gL(w, Xtr, ytr, la);
        [gk,la1k,kappak] = uo_solve_log(w1,f,g,isd,wk,dk,alk,iWk,Hk,tauk,logfreq);
        
    end
    
    %----------------------------------------------------------------------
    %Results of best wo
    niter = size(wk,2); %store number of iterations
    fo = L(wo, Xtr, ytr, la); %final minimal error of the model
    %We print the solution we have found
    
    fprintf('[uo-nn-solve]      wo=[\n')
    fprintf('[uo-nn-solve]           %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(1:5))
    fprintf('[uo-nn-solve]           %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(6:10))
    fprintf('[uo-nn-solve]           %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(11:15))
    fprintf('[uo-nn-solve]           %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(16:20))
    fprintf('[uo-nn-solve]           %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(21:25))
    fprintf('[uo-nn-solve]           %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(26:30))
    fprintf('[uo-nn-solve]           %+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e\n', wo(31:35))
    fprintf('[uo-nn-solve]         ]\n')
    
    %----------------------------------------------------------------------
    %Accuracy
    fprintf('[uo-nn-solve]  Accuracy.\n');
    
    syms y_var_tr y_tr;
    y_var_tr = sym(round(y(Xtr,wo)));
    y_tr = sym(ytr);
    tr_acc = 100/tr_p * sum(kroneckerDelta(y_var_tr,y_tr));
    fprintf('[uo-nn-solve]  tr_accuracy = %4.1f\n', tr_acc);
    
    syms y_var_te y_te;
    y_var_te = sym(round(y(Xte,wo)));
    y_te = sym(yte);
    te_acc = 100/te_q * sum(kroneckerDelta(y_var_te,y_te));
    fprintf('[uo-nn-solve]  te_accuracy = %4.1f\n', te_acc);
    fprintf('[uo-nn-solve]------------------------------------------------\n');
    %----------------------------------------------------------------------
    %Plot characters
    %{
    figure('Renderer', 'painters', 'Position', [10 10 300 450])
    uo_nn_Xyplot(wo,0,[]);
    
    figure('units','normalized','outerposition',[0 0 1 1])
    uo_nn_Xyplot(Xtr,ytr,wo);
    
    figure('units','normalized','outerposition',[0 0 1 1])
    uo_nn_Xyplot(Xte,yte,wo);
    %}
    %----------------------------------------------------------------------
end

