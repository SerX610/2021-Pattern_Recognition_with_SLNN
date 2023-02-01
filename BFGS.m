function [xk,dk,alk,iWk,Hk] = BFGS(x1,f,g,epsG,kmax,almax,almin,rho,c1,c2,iW,ils,ialmax,kmaxBLS,epsal)
        %to accelerate storing we create a n x kmax vector with NaN values
        x=x1; k=0; n = size(x,1); H = eye (n);

        xk = NaN(n,kmax); dk = NaN(n,kmax); alk = NaN(1,kmax); iWk = NaN(1,kmax); Hk = NaN(n,n,kmax);

        xk(:,1)=x1; Hk(:,:,1) = H;
        
        while norm(g(x)) > epsG && k < kmax
            d = -H*g(x); % we look direction
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

            alk(k+1)= al;
            iWk(k+1)= iWout;

            %We save the new point
            x = x + al*d;
            xk(:,k+2)= x; % we save the new xk corresponding to k+1
            
            %calculate the next H; k+2 in vect index = k+1 and k+1 in vect index = k
            sk = xk(:,k+2)-xk(:,k+1); yk=g(xk(:,k+2))-g(xk(:,k+1)); pk = 1/(yk'*sk);
            H = (eye (n)-pk*sk*yk')*H*(eye (n)-pk*yk*sk')+pk*sk*sk';
            Hk(:,:,k+1) = H;
            k = k+1;
        end

        xk = xk(:,~all(isnan(xk))); 
        dk = dk(:,~all(isnan(dk))); 
        alk = alk(~isnan(alk)); 
        iWk = iWk(~isnan(iWk)); 
        Hk(:,:,~isnan(Hk(1,1,:)));
end