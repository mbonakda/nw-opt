% Author: Matt Bonakdarpour
% Description: 
% - Implements the Gradient Projection Method for Quadratic Programs
% - uses conjugate gradient iteration to find approximate solution of subproblem
% - Nocedal and Wright, Algo 16.5

function [ x ] = projected_gradient(G, b, l, u)
  
    N    = size(b, 1);
    x_k  = zeros(N,1);
    A    = [eye(N); -eye(N)];
    x_c  = zeros(N,1);
    
    while 1
        
        bound_vec      = [l; -u];
        active         = find(A*x_k - bound_vec < 1e-8);
        not_active     = find(A*x_k - bound_vec > 1e-8);
        lambda         = pinv(A(active,:)')*(G*x_k+b);
        grad1          = G*x_k + b;
        grad2          = G*x_k + b - A(active,:)'*lambda;
     
        if isempty(active) && (norm(grad1) < 1e-5) && all(A(not_active,:)*x_k - bound_vec(not_active) >= 0)
            obj = .5*x_k'*G*x_k + x_k'*b;
            x   = x_k;
            %sprintf('Final objective = %f', obj)
            return
        end
        if ~isempty(active) && isempty(not_active) && (norm(grad2) < 1e-5) && all(lambda >= 0)
            obj = .5*x_k'*G*x_k + x_k'*b;
            x   = x_k;
            %sprintf('Final objective = %f', obj)
            return
        end
        if ~isempty(active) && ~isempty(not_active) && (norm(grad2)<1e-5) && all(lambda >= 0)
            obj = .5*x_k'*G*x_k + x_k'*b;
            x   = x_k;
            %sprintf('Final objective = %f', obj)
            return
        end
        
        % cauchy point
        x_prev_c          = x_c;
        f_before_cauchy   = computeObj(x_k,G,b);
        x_c               = getCauchyPoint(x_k, G, b, l, u);
        f_after_cauchy    = computeObj(x_c, G,b);
        
        if f_after_cauchy - f_before_cauchy > 1e-10
            x_k = x_c;
        end
        
        if(norm(x_prev_c - x_c) < 1e-7)
            x = x_k;
            obj = .5*x_k'*G*x_k + x_k'*b;
            %sprintf('Final objective = %f', obj)
            return
        end
        
        if(norm(x_k - x_c) < 1e-7)
            x = x_k;
            obj = .5*x_k'*G*x_k + x_k'*b;
            %sprintf('Final objective = %f', obj)
            return
        end
        
        x_prev     = x_k;
        actv_dx    = find((abs(x_c-l) < 1e-4) + (abs(x_c-u) < 1e-4));
        inactv_dx  = setdiff(1:length(x_c), actv_dx);

        if ~isempty(actv_dx)% && ~isempty(inactv_dx)
           
            x_k    = subspace_CG(x_c, l, u, G, b);
     
            f_after_cg = computeObj(x_k,G,b);
            if f_after_cg - f_after_cauchy > 1e-10
                x_k  = x_c;
            end
        else
            x_k = x_c;
        end
        
        if(norm(x_prev - x_k) < 1e-10)
            x = x_k;
            obj = .5*x_k'*G*x_k + x_k'*b;
            %sprintf('Final objective = %f', obj)
            return
        end
    end
    display('reached max iter')
    obj = .5*x_k'*G*x_k + x_k'*b;
    sprintf('Final objective = %f', obj)

end


function [obj] = computeObj(x_k,G,b)
    obj=.5*(x_k)'*G*(x_k) + (x_k)'*b;
    return
end
