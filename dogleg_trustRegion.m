% Author: Matt Bonakdarpour
% Description:
% - Implements dogleg trust-region optimization
% - The matrix B_k of the quadratic model is obtained with modified symmetric indefinite LDL' factorization
% - See Algorithm 4.1 in Nocedal & Wright

function [x_out, numCalls, numSolves, gradNorms] = dogleg_trustRegion( x_in, delta_max, fncHandle ) 
    delta     = 1e-2; % initial trust region radius
    eta       = 0.25;
    MAX_ITER  = 1000;
    numSolves = 0;
    numCalls  = 0;
    gradNorms = zeros(MAX_ITER,1);
    
    x_out     = x_in;
    for k = 1:MAX_ITER
        
        % compute direction p_k using dogleg method
        [p_k, B_k, f, g] = dogleg( x_out, delta, fncHandle );
        numCalls  = numCalls + 1;
        numSolves = numSolves + 1;
        
        gradNorms(k) = norm(g);
        if gradNorms(k) < 10^-5
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


%          __
%         /  \
%        / ..|\
%       (_\  |_)
%       /  \@'
%      /     \
%  _  /  `   |
% \\/  \  | _\
%  \   /_ || \\_
%   \____)|_) \_)  <-- dogleg
%

function [p_k,B_k,f,g] = dogleg( x_in, delta, fncHandle )
   
    [f,g,H]         =   fncHandle( x_in , 2 );

    % modified symmetric indefinite factorization
    [L,B,P]         =   ldl(H);
    [V,D]           =   eig(full(B));       
    tau             =   zeros(size(D,1),1);
    D               =   diag(D);
    tau(D < delta)  =   delta - D(D < delta);
    V = sparse(V);
    F               =   V*diag(tau)*V';
    % modify Hessian and solve for direction
    B_k                =   H + P*L*(F)*L'*P;
    
    % setup the two relevant vectors for dogleg
    p_u = -(g'*g / (g'*B_k*g))*g;
    p_b = -B_k \ g;
   
    if norm(p_u) >= delta
        p_k =  delta * p_u / norm(p_u);
    elseif norm(p_b) <= delta
        p_k = p_b;
    else
        a   = (p_b - p_u)'*(p_b - p_u);
        b   = 2*p_u'*(p_b - p_u);
        c   = p_u'*p_u - delta^2;
        tau   = (-b + sqrt(b^2 - 4*a*c))/ (2*a);
        p_k = p_u + tau*(p_b - p_u);
    end
    if norm(p_k) - delta > 1e-9 
        return
    end
end
