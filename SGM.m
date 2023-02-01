function [wo,wk,dk,alk] = SGM(w1,la,L,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest)
    p = size(Xtr, 2); %take columns of matrix Xtr, # of realizations of experiments

    m = floor(sg_ga*p); % the length of the minibatch

    sg_ke = round(p/m); %number of iterations per epoch
    sg_kmax = sg_emax*sg_ke; %max number of iterations in total

    %to accelerate storing we create a n x kmax vector with NaN values
    wk = NaN(size(w1,1),sg_kmax); dk = NaN(size(w1,1),sg_kmax); alk = NaN(1,sg_kmax);
    wk(:,1)=w1;
    
    w=w1; e = 0; s = 0; Lte_best = inf; k = 0; %some variables we will need.

    while e <= sg_emax && s < sg_ebest
        P = randperm(p); %permutation indexs

        for i = 0:ceil(p/m-1)

            S = P(i*m+1:(min((i+1)*m,p)));

            d = -gL(w,Xtr(:,S),ytr(S),la);
            dk(:,k+1)= d; % we save the new direction

            sg_al = 0.01*sg_al0;
            sg_k = floor(sg_be*sg_kmax);
            if k <= sg_k
                al = (1-k/sg_k)*sg_al0+k/sg_k*sg_al;
            else
                al = sg_al;
            end
            
            alk(k+1)= al; %corresponds to k, but starting position in vector is 1 not 0

            w = w + al*d;
            wk(:,k+2)= w; %corresponds to k+1, but starting position in vector is 1 not 0

            k = k + 1;
        end

        e = e + 1;
        Lte = L(w,Xte,yte,la);

        if Lte < Lte_best
            Lte_best = Lte;
            wo = w;
            s = 0;
        else
            s = s + 1;
        end
    end

    %we delete extra positions
    wk = wk(:,~all(isnan(wk))); dk = dk(:,~all(isnan(dk))); alk = alk(~isnan(alk));
end