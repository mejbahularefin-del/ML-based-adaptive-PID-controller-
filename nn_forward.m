function out = nn_forward(input_vec)
%NN_FORWARD  Evaluates the trained Python neural network in MATLAB.
%   input_vec : 2-element vector [error; d_error]
%   out       : 3-element column vector [Kp; Ki; Kd] in [0,1]
%
%   Weights are loaded once and cached in persistent variables.

    persistent W1 b1 W2 b2 W3 b3 loaded

    if isempty(loaded)
        d  = load('nn_weights.mat');
        W1 = d.fc1_weight;            % 128 x 2
        b1 = d.fc1_bias(:);           % 128 x 1
        W2 = d.fc2_weight;            % 128 x 128
        b2 = d.fc2_bias(:);           % 128 x 1
        W3 = d.fc3_weight;            % 3   x 128
        b3 = d.fc3_bias(:);           % 3   x 1
        loaded = true;
    end

    x  = input_vec(:);
    h1 = tanh(W1 * x  + b1);
    h2 = tanh(W2 * h1 + b2);
    out = 1 ./ (1 + exp(-(W3 * h2 + b3)));
end