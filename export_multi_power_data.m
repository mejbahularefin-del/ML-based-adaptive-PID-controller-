%% export_multi_power_data.m
clc; clear;

fis = readfis('Fuzzy_PID_Tuner_9x9_Trap');
load('pid_table.mat');

N = 21;                                 % grid resolution
[E, DE] = meshgrid(linspace(-1,1,N), linspace(-1,1,N));
flat = [E(:), DE(:)];
fuzzyOut = evalfis(flat, fis);          % (N*N) x 3

% Build the big dataset: sweep error, d_error, and power
P_levels = linspace(0.2, 1.0, 9);       % 9 power samples for smoother fit
allRows = [];
for P = P_levels
    Kp1 = interp1(powerLevels, Kp1_tab, P, 'pchip');
    Ki1 = interp1(powerLevels, Ki1_tab, P, 'pchip');
    Kd1 = interp1(powerLevels, Kd1_tab, P, 'pchip');
    Kp_eff = Kp1 .* fuzzyOut(:,1);
    Ki_eff = Ki1 .* fuzzyOut(:,2);
    Kd_eff = Kd1 .* fuzzyOut(:,3);
    rows = [flat, P*ones(size(flat,1),1), Kp_eff, Ki_eff, Kd_eff];
    allRows = [allRows; rows];
end

% Normalise the targets so they sit in roughly [0,1] for stable training
maxKp = max(allRows(:,4));
maxKi = max(allRows(:,5));
maxKd = max(allRows(:,6));
allRows(:,4) = allRows(:,4) / maxKp;
allRows(:,5) = allRows(:,5) / maxKi;
allRows(:,6) = allRows(:,6) / maxKd;
save('scale_factors.mat','maxKp','maxKi','maxKd');

csvwrite('training_data_multipower.csv', allRows);
fprintf('Saved %d samples to training_data_multipower.csv\n', size(allRows,1));
fprintf('Columns: error, d_error, power, Kp/%g, Ki/%g, Kd/%g\n', maxKp, maxKi, maxKd);