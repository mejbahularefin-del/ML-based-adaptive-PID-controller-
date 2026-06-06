function out = nn_forward_3in(input_vec)
%NN_FORWARD_3IN  Evaluates the trained 3-input neural network.
%   input_vec : 3-element vector [error; d_error; power]
%   out       : 3-element column vector [Kp_eff; Ki_eff; Kd_eff]
%               (already denormalised — these are the effective PID gains)

    persistent W1 b1 W2 b2 W3 b3 sKp sKi sKd loaded

    if isempty(loaded)
        d  = load('nn_weights_3in.mat');
        W1 = d.fc1_weight;            % 128 x 3
        b1 = d.fc1_bias(:);           % 128 x 1
        W2 = d.fc2_weight;            % 128 x 128
        b2 = d.fc2_bias(:);           % 128 x 1
        W3 = d.fc3_weight;            % 3   x 128
        b3 = d.fc3_bias(:);           % 3   x 1
        s  = load('scale_factors.mat');
        sKp = s.maxKp;
        sKi = s.maxKi;
        sKd = s.maxKd;
        loaded = true;
    end

    x  = input_vec(:);
    h1 = tanh(W1 * x  + b1);
    h2 = tanh(W2 * h1 + b2);
    y  = 1 ./ (1 + exp(-(W3 * h2 + b3)));   % normalised in [0,1]
    out = [y(1)*sKp; y(2)*sKi; y(3)*sKd];   % denormalised effective gains
end