clc
clear
close all

%% Create FIS
fis = newfis('Fuzzy_PID_Tuner_9x9_Trap','mamdani');

%% 9 Membership names
names = {'NB','NM','NS','NVS','Z','PVS','PS','PM','PB'};

%% MFs for INPUTS (Trapezoidal, range [-1, 1])
%  9 centres at -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
%  Top width 0.15, base extends to neighbour centres for 50% overlap
mf_in = [
-1      -1        -0.925   -0.75
-1      -0.825    -0.675   -0.5
-0.75   -0.575    -0.425   -0.25
-0.5    -0.325    -0.175    0
-0.25   -0.075     0.075    0.25
 0       0.175     0.325    0.5
 0.25    0.425     0.575    0.75
 0.5     0.675     0.825    1
 0.75    0.925     1        1
];

%% MFs for OUTPUTS (Trapezoidal, range [0, 1])
%  9 centres at 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1
mf_out = [
0       0         0.0375    0.125
0       0.0875    0.1625    0.25
0.125   0.2125    0.2875    0.375
0.25    0.3375    0.4125    0.5
0.375   0.4625    0.5375    0.625
0.5     0.5875    0.6625    0.75
0.625   0.7125    0.7875    0.875
0.75    0.8375    0.9125    1
0.875   0.9625    1         1
];

%% INPUT 1 : error
fis = addvar(fis,'input','error',[-1 1]);
for i = 1:9
    fis = addmf(fis,'input',1,names{i},'trapmf',mf_in(i,:));
end

%% INPUT 2 : d_error
fis = addvar(fis,'input','d_error',[-1 1]);
for i = 1:9
    fis = addmf(fis,'input',2,names{i},'trapmf',mf_in(i,:));
end

%% OUTPUT 1 : Kp
fis = addvar(fis,'output','Kp',[0 1]);
for i = 1:9
    fis = addmf(fis,'output',1,names{i},'trapmf',mf_out(i,:));
end

%% OUTPUT 2 : Ki
fis = addvar(fis,'output','Ki',[0 1]);
for i = 1:9
    fis = addmf(fis,'output',2,names{i},'trapmf',mf_out(i,:));
end

%% OUTPUT 3 : Kd
fis = addvar(fis,'output','Kd',[0 1]);
for i = 1:9
    fis = addmf(fis,'output',3,names{i},'trapmf',mf_out(i,:));
end

%% 9x9 Rule tables (rows = error: NB..PB, cols = d_error: NB..PB)
%  Values are MF indices: 1=NB, 2=NM, 3=NS, 4=NVS, 5=Z, 6=PVS, 7=PS, 8=PM, 9=PB

% Kp : LARGE when |e| big (aggressive push), SMALL near setpoint.
%      Symmetric about e=Z and de=Z.
Kp_table = [
9 9 8 8 7 8 8 9 9
9 8 8 7 7 7 8 8 9
8 8 7 7 6 7 7 8 8
8 7 7 6 5 6 7 7 8
7 7 6 5 3 5 6 7 7
8 7 7 6 5 6 7 7 8
8 8 7 7 6 7 7 8 8
9 8 8 7 7 7 8 8 9
9 9 8 8 7 8 8 9 9
];

% Ki : LARGE near setpoint (kill steady-state error),
%      SMALL at large |e| (avoid integral windup). Symmetric.
Ki_table = [
1 1 2 2 3 2 2 1 1
1 2 2 3 3 3 2 2 1
2 2 3 3 4 3 3 2 2
2 3 3 4 5 4 3 3 2
3 3 4 5 9 5 4 3 3
2 3 3 4 5 4 3 3 2
2 2 3 3 4 3 3 2 2
1 2 2 3 3 3 2 2 1
1 1 2 2 3 2 2 1 1
];

% Kd : moderate everywhere, smallest at steady state for fine damping.
%      Symmetric.
Kd_table = [
7 7 6 6 5 6 6 7 7
7 6 6 5 5 5 6 6 7
6 6 5 5 4 5 5 6 6
6 5 5 4 4 4 5 5 6
5 5 4 4 3 4 4 5 5
6 5 5 4 4 4 5 5 6
6 6 5 5 4 5 5 6 6
7 6 6 5 5 5 6 6 7
7 7 6 6 5 6 6 7 7
];

%% Sanity check: all three tables must be symmetric about the centre
assert(isequal(Kp_table, flipud(fliplr(Kp_table))), 'Kp_table is not symmetric');
assert(isequal(Ki_table, flipud(fliplr(Ki_table))), 'Ki_table is not symmetric');
assert(isequal(Kd_table, flipud(fliplr(Kd_table))), 'Kd_table is not symmetric');
disp('All rule tables are symmetric — OK.');

%% Build 81 rules
rules = [];
for i = 1:9
    for j = 1:9
        rules = [rules;
                 i j Kp_table(i,j) Ki_table(i,j) Kd_table(i,j) 1 1];
    end
end
fis = addrule(fis,rules);

%% Save FIS so the Simulink Fuzzy Logic Controller block can load it
writefis(fis,'Fuzzy_PID_Tuner_9x9_Trap');

%% Open Fuzzy Editor
fuzzy(fis)

%% Surface views
figure; gensurf(fis,[1 2],1); title('Kp Surface (9x9 Trap) — bowl, min at centre');
xlabel('error'); ylabel('d\_error'); zlabel('Kp')

figure; gensurf(fis,[1 2],2); title('Ki Surface (9x9 Trap) — inverted bowl, max at centre');
xlabel('error'); ylabel('d\_error'); zlabel('Ki')

figure; gensurf(fis,[1 2],3); title('Kd Surface (9x9 Trap) — shallow bowl, min at centre');
xlabel('error'); ylabel('d\_error'); zlabel('Kd')