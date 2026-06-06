%% powerleveltune.m   (manual gain scheduling, anchored at 100% power)
% At 100% power we use the well-tuned gains the original Mamdani design used.
% At lower power, the plant gain is lower, so the controller needs slightly
% higher gains to maintain similar closed-loop behaviour. The scaling factors
% below are a physically reasonable, smooth interpolation.

clc; clear;

powerLevels = [0.20 0.40 0.60 0.80 1.00];

% Hand-tuned values at full power (your original design)
Kp_base = 22.5703;
Ki_base = 2.5695;
Kd_base = 18.2009;

% Scaling factors (1.0 at full power, grow modestly at lower power)
% Order: 20%, 40%, 60%, 80%, 100%
scale_Kp = [1.35 1.22 1.12 1.05 1.00];
scale_Ki = [1.50 1.30 1.16 1.06 1.00];
scale_Kd = [1.25 1.15 1.08 1.03 1.00];

Kp1_tab = (Kp_base * scale_Kp)';
Ki1_tab = (Ki_base * scale_Ki)';
Kd1_tab = (Kd_base * scale_Kd)';

save('pid_table.mat','powerLevels','Kp1_tab','Ki1_tab','Kd1_tab');

disp(table(powerLevels(:)*100, Kp1_tab, Ki1_tab, Kd1_tab, ...
           'VariableNames',{'Power_pct','Kp1','Ki1','Kd1'}));

%% Visualise
figure('Position',[100 100 1200 400]);
subplot(1,3,1);
plot(powerLevels*100, Kp1_tab, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Power (%)'); ylabel('Kp1'); title('Kp1 vs power'); grid on;

subplot(1,3,2);
plot(powerLevels*100, Ki1_tab, 'rs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Power (%)'); ylabel('Ki1'); title('Ki1 vs power'); grid on;

subplot(1,3,3);
plot(powerLevels*100, Kd1_tab, 'g^-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Power (%)'); ylabel('Kd1'); title('Kd1 vs power'); grid on;