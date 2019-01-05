% fnc to correct nodes in matrix memory
function [vec, delta] = updateRandomNode(MatrixMem, vec)
    % choose a random node to update
    rand_node = randi(length(vec));
    
    % initialize the delta to the random node value
    delta = vec(rand_node);
    
    % apply the signum function to the vector at randomly selected index
    vec(rand_node) = sign(MatrixMem(rand_node,:) * vec);
    
    % find the number of nodes changed
    delta = abs(delta - vec(rand_node))/2;
    
end