%% export_data_for_python_dense.m
% Dense sampling near the origin to reduce Ki error

clc; clear; close all;

%% Load the existing Mamdani FIS
fis = readfis('Fuzzy_PID_Tuner_9x9_Trap');
fprintf('Loaded Mamdani FIS.\n');

%% Coarse grid over full range [-1,1] x [-1,1]
N_coarse = 31;
[E1, DE1] = meshgrid(linspace(-1, 1, N_coarse), linspace(-1, 1, N_coarse));
inputs_coarse = [E1(:), DE1(:)];

%% Fine grid concentrated near the origin (where Ki spike is strongest)
N_fine = 41;                          % you can increase to 61 if needed
range_fine = 0.3;                     % dense sampling in [-0.3, 0.3]
[E2, DE2] = meshgrid(linspace(-range_fine, range_fine, N_fine), ...
                     linspace(-range_fine, range_fine, N_fine));
inputs_fine = [E2(:), DE2(:)];

%% Combine and remove duplicates
inputs = unique([inputs_coarse; inputs_fine], 'rows');
fprintf('Total unique input points: %d\n', size(inputs,1));

%% Evaluate FIS at all points
outputs = evalfis(inputs, fis);       % columns: Kp, Ki, Kd

%% Combine into one big table
data = [inputs, outputs];             % [error, d_error, Kp, Ki, Kd]

%% Save as CSV

csvwrite('training_data_dense.csv', data);

fprintf('Saved %d samples to training_data_dense.csv\n', size(data,1));
fprintf('Columns: error, d_error, Kp, Ki, Kd\n');
fprintf('File location: %s\n', fullfile(pwd, 'training_data_dense.csv'));