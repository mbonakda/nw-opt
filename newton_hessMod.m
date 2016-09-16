% Author: Matt Bonakdarpour
% Description: 
% - Line Search Newton's Method with Hessian Modificaiton. Uses Armijo backtracking.
% - Hessian modification done by modified symmetric indefinite LDL' factorization
% - See section 3.4 of Nocedal & Wright

function [ x_out, numCalls, grad_norms, numSolves ] = newton_hessMod( x_in, fncHandle )
    numCalls        = 0;
    numSolves       = 0;
    NUM_ITER        = 500;
    grad_norms      = zeros(NUM_ITER,1);
    delta           = 10^-8; % TODO: test sqrt(precision)
    x_k             = x_in;
    for k = 1:NUM_ITER
       
        [f,g,H]         =   fncHandle(x_k,2);
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
        [alpha_k, nCalls]  =   backtrack(f, g, p_k, x_k, fncHandle);
        numCalls           = numCalls + nCalls;
        x_k                =   x_k + alpha_k*p_k;
    end
    x_out = x_k;
end

function [alpha, numCalls] = backtrack(f_k, g_k, p_k, x_k, fncHandle)
    numCalls = 0;
    alpha = 1;
    rho   = 0.9; 
    c     = 0.5;
    f_new = fncHandle(x_k+alpha*p_k, 0);
    numCalls = numCalls + 1;
    while f_new > f_k + c*alpha*(g_k'*p_k)
        alpha    = rho * alpha;
        f_new    = fncHandle(x_k+alpha*p_k, 0);
        numCalls = numCalls + 1;
    end
end

