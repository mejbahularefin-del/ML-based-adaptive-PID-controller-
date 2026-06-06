%% test_nn_3in.m
clc; clear; clear nn_forward_3in;

% (error, d_error, power) inputs
test_inputs = [ 0    0    1.0;     % steady state at 100% power
                0    0    0.5;     % steady state at 50% power
                0.5  0    1.0;
               -0.5  0.3  0.6;
                0.8 -0.6  0.2 ];

fprintf('Test of 3-input neural network:\n');
fprintf('%-25s  %-10s  %-10s  %-10s\n', 'Input (e, de, P)', 'Kp_eff', 'Ki_eff', 'Kd_eff');
fprintf('%s\n', repmat('-', 1, 70));
for k = 1:size(test_inputs,1)
    inp = test_inputs(k,:);
    out = nn_forward_3in(inp);
    fprintf('(%+.2f, %+.2f, %.2f)         %.4f      %.4f      %.4f\n', ...
            inp(1), inp(2), inp(3), out(1), out(2), out(3));
end