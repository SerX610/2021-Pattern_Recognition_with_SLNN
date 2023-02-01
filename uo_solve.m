function [xk,dk,alk,iWk,betak,Hk,tauk] = uo_solve(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,ils,ialmax,kmaxBLS,epsal)
    % isd = 1: GM
    % isd = 3: BFGS
    % isd = 7: SGM
    
    
    xk = []; dk = []; alk = []; iWk = []; betak=[]; Hk=[]; tauk=[];
    logfunct = @(xk) ceil(size(xk,2)/10);

    if isd == 1 %Gradient Method (GM)
        [xk,dk,alk,iWk] = GM(x1,f,g,epsG,kmax,almax,almin,rho,c1,c2,iW,ils,ialmax,kmaxBLS,epsal);
        logfreq = logfunct(xk);
        %logfreq = 1;
        [gk,la1k,kappak] = uo_solve_log(x1,f,g,isd,xk,dk,alk,iWk,Hk,tauk,logfreq);
    else % isd == 3 % Quasi-Newton Methods (BFGS)
        [xk,dk,alk,iWk,Hk] = BFGS(x1,f,g,epsG,kmax,almax,almin,rho,c1,c2,iW,ils,ialmax,kmaxBLS,epsal);
        logfreq = logfunct(xk);
        %logfreq = 1;
        [gk,la1k,kappak] = uo_solve_log(x1,f,g,isd,xk,dk,alk,iWk,Hk,tauk,logfreq);

    end
end