% [start] Alg. ELS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iWout = 0: al does not satisfy any WC
% iWout = 1: al satisfies (WC1)
% iWout = 2: al satisfies WC
% iWout = 3: al satisfies SWC
function [al,iWout] = uo_ELS(x, d, f, g, h, c1, c2)
%{
%We create the conditions
WC1 = @ (al) f(x+al*d) <= f(x)+c1*g(x)'*d*al;
WC2 = @ (al) g(x+al*d)' * d >= c2*g(x)'*d;
SWC2 = @ (al) abs(g(x+al*d)' * d) <= c2*abs(g(x)'*d);
%We stablish WC and SWC
WC = @ (al) (WC1(al) & WC2(al));
SWC = @ (al) (WC1(al) & SWC2(al));
%}

%Exact line search iW=0
% Only can be applied to quadratic functions, because we supose g(x)=Qx-b
Q = h(x);
al = -g(x)'*d/(d'*Q*d);
iWout = -1; % iWOut unknown
%{
if SWC (al)
    iWout = 3;
elseif WC (al)
    iWout = 2;
elseif WC1 (al) % We look if the last alfa is accepted by WC1
    iWout = 1;
else
    iWout = 0;
end 
%}
end
% [end] Alg. ELS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

