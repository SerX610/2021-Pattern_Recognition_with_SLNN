function [xk,dk,alk,iWk,betak,Hk,tauk] = uo_solve_performance(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,ils,ialmax,kmaxBLS,epsal)
    % isd = 1: GM
    % isd = 3: BFGS
    % isd = 7: SGM
    
    
    xk = []; dk = []; alk = []; iWk = []; Hk=[]; betak=[]; tauk =[];

    if isd == 1 %Gradient Method (GM)
        [xk,dk,alk,iWk] = GM(x1,f,g,epsG,kmax,almax,almin,rho,c1,c2,iW,ils,ialmax,kmaxBLS,epsal);
    else % isd == 3 % Quasi-Newton Methods (BFGS)
        [xk,dk,alk,iWk,Hk] = BFGS(x1,f,g,epsG,kmax,almax,almin,rho,c1,c2,iW,ils,ialmax,kmaxBLS,epsal);
    end
end