% Author: Matt Bonakdarpour
% Description:
% - Implements Trust-region Newton-CG Method (Steihaug)
% - See Algorithm 7.2 in Nocedal & Wright
function [x_out, numCalls, num_cg_iters, gradNorms] = trust_region_newtonCG( x_in, delta_max, fncHandle ) 
    delta     = 1e-2; % initial trust region radius
    eta       = 0.25;
    MAX_ITER  = 1000;
    numCalls  = 0;
    num_cg_iters = 0;
    gradNorms = zeros(MAX_ITER,1);
    [f,g,B]   = fncHandle( x_in, 2 );
    x_out     = x_in;
    for k = 1:MAX_ITER
        
        % compute direction p_k using CG-Steihaug
        [p_k, B_k, f, g, cg_iters] = cg_steihaug( x_out, delta, 1e-8, fncHandle );
        num_cg_iters = num_cg_iters + cg_iters;
        numCalls  = numCalls + 1;
        
        gradNorms(k) = norm(g);
        if gradNorms(k) < 10^-6
            return
        end
        
        % compute relative agreement 
        actual_reduction = f - fncHandle( x_out + p_k, 0);
        numCalls         = numCalls + 1;
        pred_reduction   = -g'*p_k - 0.5*p_k'*B_k*p_k;
        rho_k            = actual_reduction / pred_reduction;
        
        % check model agreement
        if rho_k < 0.25
            delta = 0.25 * delta; % shrink trust region
        elseif rho_k > 0.75 && (norm(p_k) - delta) < 1e-10
            delta = min(2*delta, delta_max); % increase trust region
        end
        if rho_k > eta
            x_out = x_out + p_k;
        end   
    end
    display('Reached max iteration')
end

function [p_k, B, f, g, iter] = cg_steihaug( x_in, delta, e_k, fncHandle )
    
    MAX_ITER        = 100;
    iter            = 0;
    [f,g,B]         = fncHandle( x_in, 2 );

    % CG-Steihaug method
    z_j = zeros(size(x_in,1), 1); r_j = g; d_j = -r_j;
    
    % TODO: think about this
    e_k = norm(r_j)*e_k;
    
    if norm(r_j) < e_k
        display('early exit')
        p_k = z_j;
        return
    end
    
    for iter = 1:MAX_ITER
        % precompute and store
        K = d_j'*B*d_j;
        if K <= 0
            display('negative curvature')
            a   = d_j'*d_j;
            b   = 2*z_j'*d_j;
            c   = z_j'*z_j - delta*delta;
            
            tau1 = (-b + sqrt(b^2 - 4*a*c))/ (2*a);
            tau2 = (-b - sqrt(b^2 - 4*a*c))/ (2*a);
            
            pk_1 = z_j + tau1*d_j;
            pk_2 = z_j + tau2*d_j;
            
            val1 = f + g'*pk_1 + .5*pk_1'*B*p_k1;
            val2 = f + g'*pk_2 + .5*pk_2'*B*p_k2;
            if val1 < val2
                p_k = pk_1;
            else
                p_k = pk_2;
            end
            return
        end
        
        alpha_j = (r_j'*r_j)/K;
        z_next  = z_j + alpha_j*d_j;
        if norm(z_next) >= delta
            a   = d_j'*d_j;
            b   = 2*z_j'*d_j;
            c   = z_j'*z_j - delta*delta;
            tau = (-b + sqrt(b^2 - 4*a*c))/ (2*a);
            p_k = z_j + tau*d_j;
            return
        end
        r_next = r_j + alpha_j*B*d_j;
        if norm(r_next) < e_k
            p_k = z_next;
            return
        end
        
        beta_j = (r_next'*r_next) / (r_j'*r_j);
        d_j    = -r_next + beta_j*d_j;
        z_j    = z_next;
        r_j    = r_next;
    end
    display('CG-Steihaug: max iterations reached')
    
end

