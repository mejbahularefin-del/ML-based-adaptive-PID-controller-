d = load('nn_weights.mat');
disp(fieldnames(d));
fprintf('fc1_weight size: %dx%d\n', size(d.fc1_weight));
fprintf('fc2_weight size: %dx%d\n', size(d.fc2_weight));
fprintf('fc3_weight size: %dx%d\n', size(d.fc3_weight));
fprintf('fc1_bias size:   %dx%d\n', size(d.fc1_bias));
fprintf('fc3_bias size:   %dx%d\n', size(d.fc3_bias));