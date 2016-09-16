% Author: Matt Bonakdarpour
% Description:
% - Reduces linearly constrained optimization problem to unconstrainted problem. 
%       Original problem: min f(x) s.t. Ax = b
% - Solves unconstrained opt with modified newton line search

function [x_out, lambda] = simpleLC_opt(fncHandle, A, b)
    
    % find a solution
    x0 = A\b;
    
    % obtain basis for null space
    K  = null(A);
    
    % optimize 
    [ z_out, numCalls, grad_norms, numSolves ] = modified_Sym_Indef_Fact( zeros(size(K,2),1), fncHandle, K, x0 );
    
    x_out = K*z_out + x0;
    [f,g] = fncHandle(x_out, 1);
    lambda = A'\g;
end


function [ x_out, numCalls, grad_norms, numSolves ] = modified_Sym_Indef_Fact( x_in, fncHandle, K, x0 )
    numCalls        = 0;
    numSolves       = 0;
    NUM_ITER        = 500;
    grad_norms      = zeros(NUM_ITER,1);
    delta           = 10^-8; % TODO: test sqrt(precision)
    x_k             = x_in;
    for k = 1:NUM_ITER
       
        [f,g,H]         = fncHandle(K*x_k + x0,2);
        g               = K'*g;
        H               = K'*H*K;
        % number of function evaluations
        numCalls        =   numCalls + 1;

        % stop condition on norm of gradient
        grad_norms(k) = norm(g);
        if norm(g) < 10^-5
            x_out = x_k;
            return
        end
        
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
        p_k                =   -B_k\g;
        numSolves          = numSolves + 1;
        % backtracking to find sufficient step size
        [alpha_k, nCalls]  =   backtrack(f, g, p_k, x_k, fncHandle, K, x0);
        numCalls           =   numCalls + nCalls;
        x_k                =   x_k + alpha_k*p_k;
    end
    x_out = x_k;
end

function [alpha, numCalls] = backtrack(f_k, g_k, p_k, x_k, fncHandle, K, x0)
    numCalls = 0;
    alpha = 1;
    rho   = 0.9; 
    c     = 0.5;
    f_new = fncHandle(K*(x_k+alpha*p_k)+x0, 0);
    numCalls = numCalls + 1;
    while f_new > f_k + c*alpha*(g_k'*p_k)
        alpha    = rho * alpha;
        f_new    = fncHandle(K*(x_k+alpha*p_k)+x0, 0);
        numCalls = numCalls + 1;
    end
end

