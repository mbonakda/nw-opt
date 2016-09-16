% Author: Matt Bonakdarpour
% Description:
% - Implements the Long-Step Path-Following interior-point method for solving:
%       min c'x s.t. Ax=b, x >= 0
% - See Nocedal & Wright, Algorithm 14.2

function [x_out, lambda_out, s_out] = longStepPathFollow(A, b, c)
    
    % starting point
    x1  = A'*((A*A')\b);
    l1  = (A*A')\(A*c);
    s1  = c - A'*l1;
    dx1 = max(-(1.5)*min(x1),0);
    ds1 = max(-(1.5)*min(s1),0);
    
    e   = ones(size(x1,1),1);
    x2  = x1 + dx1*e;
    s2  = s1 + ds1*e;
    dx2 = 0.5*(x2'*s2)/(e'*s2);
    ds2 = 0.5*(x2'*s2)/(e'*x2);
    
    x_out      = x2 + dx2*e;
    lambda_out = l1;
    s_out      = s2 + ds2*e;

    MAX_ITER   = 5000;
    
    for i = 1:MAX_ITER
        X = diag(x_out);
        S = diag(s_out);
        J = [zeros(size(A',1), size(A,2)) A' eye(size(A',1), size(X,2)); ...
            A zeros(size(A, 1), size(A',2)) zeros(size(A,1), size(X,2));  ...
            S zeros(size(S,1), size(A',2)) X];
        r_c = A'*lambda_out + s_out - c;
        r_b = A*x_out - b;
        F_aff   = [r_c; r_b; X*S*ones(size(X,2),1)];
        
        [L,U]    = lu(J);
        aff_soln = -U\(L\F_aff);
        %aff_soln = -J\F_aff;
        x_aff    = aff_soln(1:size(x_out,1));
        l_aff    = aff_soln((size(x_out,1)+1):(size(x_out,1)+size(lambda_out,1)));
        s_aff    = aff_soln((size(x_out,1)+size(lambda_out,1)+1):(size(x_out,1)+size(lambda_out,1)+size(s_out,1)));
        X_aff    = diag(x_aff);
        S_aff    = diag(s_aff);
        
        search_vars =  x_out ./ x_aff;
        alpha_aff_primal = min(1, min(-search_vars(find(search_vars < 0))));
        search_vars =  s_out ./ s_aff;
        alpha_aff_dual = min(1, min(-search_vars(find(search_vars < 0))));
        mu_aff  = ((x_out + alpha_aff_primal*x_aff)'*(s_out + alpha_aff_dual*s_aff))/size(x_out,1);
        mu      = (x_out'*s_out)/size(x_out,1);
        if mu < 0.0001
            display('int point success')
            return
        end
        sigma   = (mu_aff/mu)^3; 
        
        F    = [r_c; r_b; X*S*ones(size(X,2),1) + X_aff*S_aff*ones(size(X_aff,2),1) - sigma*mu*ones(size(X_aff,2),1)];
        %dir  = -J\F;
        dir   = -U\(L\F);
        delta_x    = dir(1:size(x_out,1));
        delta_l    = dir((size(x_out,1)+1):(size(x_out,1)+size(lambda_out,1)));
        delta_s    = dir((size(x_out,1)+size(lambda_out,1)+1):(size(x_out,1)+size(lambda_out,1)+size(s_out,1)));
        
        search_vars =  x_out ./ delta_x;
        alpha_k_primal_max = min(-search_vars(find(search_vars < 0)));
        search_vars =  s_out ./ delta_s;
        alpha_k_dual_max = min(-search_vars(find(search_vars < 0)));
        
        %TODO: increase as k -> inf
        eta_k = 0.95;
        alpha_k_primal = min(1, eta_k*alpha_k_primal_max);
        alpha_k_dual   = min(1, eta_k*alpha_k_dual_max);
        
        x_out      = x_out + alpha_k_primal*delta_x;
        lambda_out = lambda_out + alpha_k_dual*delta_l;
        s_out      = s_out + alpha_k_dual*delta_s;
        
        
    end
end

