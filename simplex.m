% Author: Matt Bonakdarpour
% Description:
% - Implements simplex method using the 2-phrase strategy outlined in Nocedal & Wright (Ch. 13)
% - Solves following problem:
%       min c'x s.t. Ax=b, x >= 0

function [ts,xs] = simplex(A, b, c)
   
    simplexStart = tic;
    % construct phase 1 problem
    e = ones(m,1);
    E = zeros(m,m);
    for i = 1:m
        if b(i) >= 0
            E(i,i) = 1;
        else
            E(i,i) = -1;
        end
    end
    
    p1_A  = [A E];
    p1_c  = [zeros(3*m,1); e];
    % initial basic feasible point for phase 1
    p1_x     = [zeros(3*m,1); abs(b)];
    b_idx1    = (3*m + 1):(4*m);
    n_idx1    = 1:3*m;
    [p1_soln, p1_b, p1_n]  = simplex_imp(p1_A, b, p1_c, p1_x, b_idx1, n_idx1);
    
    if any(abs(p1_soln((3*m + 1):4*m)) >= 1e-3)
        display('infeasible')
        return
    end
    
    % phase 2
    if all(p1_b <= 3*m)
        [xs, p2_b,p2_n] = simplex_imp(A, b, c, p1_soln(1:3*m), p1_b, setdiff(1:3*m,p1_b));
    else
        display('uh oh')
        xs = -Inf
    end
    ts = toc(simplexStart);
end

function [x_out, b_idx, n_idx] = simplex_imp(A,b,c, x_in, b_idx, n_idx)
    x_out  = x_in;
    while 1
        B = A(:, b_idx);
        N = A(:, n_idx);
        
        lambda = B'\c(b_idx);
        sN     = c(n_idx) - N'*lambda;
        
        if all(sN >= 0)
            return
        end

        % find entering index
        [m,q] = min(sN);
        d     = B\N(:,q);
        if all(d <= 0)
            display('infeasible')
            x_out = -Inf;
            return
        end

        search_vec      = x_out(b_idx) ./ d;
        pos_entries     = find(d > 0);
        [xq, p]         = min(search_vec(pos_entries));
        x_out(b_idx)    = x_out(b_idx) - (d*xq); 
        x_out(n_idx(q)) = xq;

        removeIdx = b_idx(pos_entries(p));
        addIdx    = n_idx(q);
        b_idx(pos_entries(p))  = addIdx;
        n_idx(q)  = removeIdx;
    end
end
