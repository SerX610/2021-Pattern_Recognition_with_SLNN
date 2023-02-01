% [start] Alg. BLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iWout = 0: al does not satisfy any WC
% iWout = 1: al satisfies (WC1)
% iWout = 2: al satisfies WC
% iWout = 3: al satisfies SWC
function [al,iWout] = uo_BLS(x,d,f,g,almax,almin,rho,c1,c2,iW)
al = almax; % We set the starting alfa to max

%We create the conditions
WC1 = @ (al) f(x+al*d) <= f(x)+c1*g(x)'*d*al;
WC2 = @ (al) g(x+al*d)' * d >= c2*g(x)'*d;
SWC2 = @ (al) abs(g(x+al*d)' * d) <= c2*abs(g(x)'*d);
%We stablish WC and SWC
WC = @ (al) (WC1(al) & WC2(al));
SWC = @ (al) (WC1(al) & SWC2(al));

if iW == 1 %WC
    while al > almin
        if WC (al)
            iWout = 2;
            return;
        end
    al = rho * al; % rho must be less than 1 to make alfa decrease
    end

elseif iW==2 %SWC
    while al > almin
        if SWC (al)
            iWout = 3;
            return;
        end
        al = rho * al; % rho must be less than 1 to make alfa decrease
    end
end

if WC1 (al) % We look if the last alfa is accepted by WC1
    iWout = 1;
else
    iWout = 0;
end 

end
% [end] Alg. BLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%