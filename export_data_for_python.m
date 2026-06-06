%% export_data_for_python.m
% Samples the Mamdani FIS on a grid and exports to a CSV file
% that Python can read.

clc; clear; close all;

%% Load the existing Mamdani FIS
fis = readfis('Fuzzy_PID_Tuner_9x9_Trap');
fprintf('Loaded Mamdani FIS.\n');

%% Create a 31x31 grid (961 samples — more than enough)
N = 31;
[E, DE] = meshgrid(linspace(-1,1,N), linspace(-1,1,N));
inputs = [E(:), DE(:)];

%% Evaluate FIS (R2018a accepts both argument orders; this one is safe)
outputs = evalfis(inputs, fis);   % size: 961 x 3 (Kp, Ki, Kd)

%% Combine into one big table
data = [inputs, outputs];
% Columns:  error | d_error | Kp | Ki | Kd

%% Save as CSV (no header — Python will know the column order)
csvwrite('training_data.csv', data);

fprintf('Saved %d samples to training_data.csv\n', size(data,1));
fprintf('Columns: error, d_error, Kp, Ki, Kd\n');
fprintf('File location: %s\n', fullfile(pwd, 'training_data.csv'));