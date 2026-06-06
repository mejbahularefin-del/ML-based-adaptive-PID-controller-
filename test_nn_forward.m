%% test_nn_forward.m
clc; clear;

% Force reload by clearing persistent variables
clear nn_forward;

test_inputs = [ 0    0;
                0.5  0;
               -0.5  0.3;
                0.8 -0.6;
                0   -0.5;
               -0.3  0.4 ];

fprintf('MATLAB neural network forward pass:\n');
fprintf('%-22s  %-10s  %-10s  %-10s\n', 'Input', 'Kp', 'Ki', 'Kd');
fprintf('%s\n', repmat('-', 1, 60));
for k = 1:size(test_inputs,1)
    inp = test_inputs(k,:);
    out = nn_forward(inp);
    fprintf('(%+.2f, %+.2f)            %.6f    %.6f    %.6f\n', ...
            inp(1), inp(2), out(1), out(2), out(3));
end