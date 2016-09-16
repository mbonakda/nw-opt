% Author: Matt Bonakdarpour
% Description: 
% - Implements Projected CG method 
% - Nocedal & Wright, Algo 16.2

function [ x_cg ] = subspace_CG(x_c, l, u, G, b)
    N    = size(x_c, 1);
    active_constraint_idx = find((abs(x_c-l) < 1e-3) + (abs(x_c-u) < 1e-3));
    I    = eye(N);
    
    % preconditioning
%     inactv_idx = setdiff(1:length(x_c), active_constraint_idx);
%     Z              = I(inactv_idx,:)';
%     %L             = ichol(G);
%     H              = diag(abs(diag(G)));
%     P_mid          = Z'*H*Z;
%     [L_mid,U_mid]  = lu(P_mid);
    
    A    = I(active_constraint_idx, :);
    b_cg = x_c(active_constraint_idx);
    x_cg  = x_c;
    r    = G*x_cg + b;
    
    v    = (A*A')\(A*r);
    g    = r - A'*v;
    %g    = Z*(L_mid\(U_mid\(Z'*r)));
    d    = -g;
    cg_iter = 1;
    while r'*g >= 1e-5 
        if (d'*G*d <= 0)
            %display('negative curvature condition');
            break
        end
        alpha = (r'*g)/(d'*G*d);
        x_prev = x_cg;
        x_cg   = x_cg + alpha*d;
        if  norm(x_prev - x_cg) < 1e-10
            break
        end
        if any(x_cg < l) || any(x_cg > u)
            %x_cg(x_cg < l) = l(x_cg < l);
            %x_cg(x_cg > u) = u(x_cg >u);
            x_cg = x_prev;
            break
        end
        r_p   = r + alpha*G*d;
        v     = (A*A')\(A*r_p);
        g_p   = r_p - A'*v;
        %g_p    = Z*(L_mid\(U_mid\(Z'*r_p)));
        beta  = (r_p'*g_p)/(r'*g);
        d     = -g_p + beta*d;
        g     = g_p;
        r     = r_p;
        cg_iter = cg_iter+ 1;
    end
end

