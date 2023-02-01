function [gk,la1k,kappak] = uo_solve_log(x1,f,g,isd,xk,dk,alk,iWk,Hk,tauk,logfreq)
niter = size(xk,2); n= size(x1,1);
fk  = []; gk = []; gdk = [];
la1k=zeros(1,niter);     % if NM or MNM: lowest vap of h(xk:,k)); if QNM: lowest vap of Hk(:,:,k).
kappak = zeros(1,niter); % if NM : cond. number of h(xk(:,k))), if +def; if QNM or MNM: cond. number of Hk or Bk resp.
for k = 1:niter
    x = xk(:,k); fk = [fk,f(x)]; gk = [gk,g(x)]; 
    if k < niter
        gdk = [gdk,gk(:,k)'*dk(:,k)];
        if  isd == 3 la1k(k) = min(eig(Hk(:,:,k))); kappak(k) = cond(Hk(:,:,k)); end
    end
end
if niter > 1
    if isd <= 4  tauk(1:niter-1) = 0; end
end
if isd == 3  la1k(niter)     = 0; end

if isd ==1
    fprintf('[uo-nn-solve]      k     g''*d        al iW    ||g||        f\n');
elseif isd ==3
    fprintf('[uo-nn-solve]      k     g''*d        al iW    la(1)    kappa    ||g||        f\n');   
elseif isd ==7
    fprintf('[uo-nn-solve]      k     g''*d        al    ||g||        f\n');  
end
if niter == 1
    krange=[];
elseif niter == 2
        krange =[1];
else
        krange=[1:logfreq:max(2,niter-11),max(3,niter-10):niter-1];
end
for k = krange
%for k = [1:logfreq:max(2,niter-11),max(3,niter-10):niter-1]
    if isd == 1 
        fprintf('[uo-nn-solve]  %6d %+3.1e %+3.2e  %1d %+3.1e %+3.1e\n', k, gdk(k), alk(k), iWk(k), norm(gk(:,k)), fk(k));
    elseif isd == 3
        fprintf('[uo-nn-solve]  %6d %+3.1e %+3.2e  %1d %+3.1e %+3.1e %+3.1e %+3.1e\n', k, gdk(k), alk(k), iWk(k), la1k(k), kappak(k), norm(gk(:,k)), fk(k));      
    elseif isd==7        
        fprintf('[uo-nn-solve]  %6d %+3.1e %+3.2e %+3.1e %+3.1e\n', k, gdk(k), alk(k), norm(gk(:,k)), fk(k));      
    end
end
if isd == 1
    fprintf('[uo-nn-solve]  %6d                       %+3.1e %+3.1e\n', niter, norm(gk(:,niter)), fk(niter));
    fprintf('[uo-nn-solve]      k     g''*d        al iW    ||g||        f\n');
elseif isd == 3
    fprintf('[uo-nn-solve]  %6d                                         %+3.1e %+3.1e\n', niter, norm(gk(:,niter)), fk(niter));
    fprintf('[uo-nn-solve]      k     g''*d        al iW    la(1)    kappa    ||g||        f\n');
elseif isd == 7
    fprintf('[uo-nn-solve]  %6d                    %+3.1e %+3.1e\n', niter, norm(gk(:,niter)), fk(niter));
    fprintf('[uo-nn-solve]      k     g''*d        al    ||g||        f\n');
end
end

