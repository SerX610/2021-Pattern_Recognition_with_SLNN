function [xk,dk,alk,iWk] = GM(x1,f,g,epsG,kmax,almax,almin,rho,c1,c2,iW,ils,ialmax,kmaxBLS,epsal)
    %to accelerate storing we create a n x kmax vector with NaN values
    xk = NaN(size(x1,1),kmax); dk = NaN(size(x1,1),kmax); alk = NaN(1,kmax); iWk = NaN(1,kmax); 
    xk(:,1)=x1; x=x1; k=0;
    while norm(g(x)) > epsG && k < kmax
            d = -g(x);
            dk(:,k+1)= d; %corresponds to k, but starting position in vector is 1 not 0

            %Exact or inexact line search
            if ils == 1 %exact line search
                [al, iWout] = uo_ELS(x, d, f, g, h, c1, c2);
            elseif ils==2 %inexact line search
                [al, iWout] = uo_BLS(x, d, f, g, almax, almin, rho, c1, c2, iW);
            elseif ils==3 % line search advanced algorithm
                if k~=0 %we take a standard max step lentgh for the first iteration
                    if ialmax == 1
                        almax = al*(g(xk(:,k))'*dk(:,k))/(g(x)'*d); %method 1
                    else %ialmax == 2
                        almax = 2*(f(x)-f(xk(:,k)))/(g(x)'*d); %method 2
                    end 
                end
                [al,iWout] = uo_BLSNW32(f,g,x,d, almax,c1,c2,kmaxBLS,epsal);
            end
            
            % we save the alfa found and the condition that it satifies
            alk(k+1)= al;
            iWk(k+1)= iWout;

            x = x + al*d; 
            xk(:,k+2)= x; % we save the new xk corresponding to k+1

            k = k+1;
    end
    %we delete extra positions
    xk = xk(:,~all(isnan(xk))); dk = dk(:,~all(isnan(dk))); alk = alk(~isnan(alk)); iWk = iWk(~isnan(iWk));
end