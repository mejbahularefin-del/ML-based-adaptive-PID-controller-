%% step_C1_build_plant_models.m
% Build 5 reactor plant models at power 20%, 40%, 60%, 80%, 100%.
clc; clear; close all;

powerLevels = [0.20 0.40 0.60 0.80 1.00];
num100 = [62500 86190 22740 1683];
den100 = [1 401.4 667.8 392.6 44];

plantModels = cell(length(powerLevels), 1);
for k = 1:length(powerLevels)
    P = powerLevels(k);
    gainScale  = 0.4 + 0.6 * P;
    speedScale = 0.6 + 0.4 * P;
    num = num100 * gainScale;
    den = [1, den100(2:end) ./ speedScale];
    plantModels{k} = tf(num, den);
end
save('plant_models.mat', 'plantModels', 'powerLevels');