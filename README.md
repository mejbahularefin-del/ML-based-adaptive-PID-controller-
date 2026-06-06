# ML-based-adaptive-PID-controller-
This repository contains the full implementation of an ML-based adaptive PID controller for nuclear reactor power-level control. The controller combines two mechanisms:
Self-tuning gains
Power-level gain scheduling
ml-adaptive-pid-repo/
│
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies (data processing)
│
├── src/
│   ├── nn_controller.m              ← Neural network forward pass (MATLAB)
│   ├── pid_block.m                  ← PID computation with NN-tuned gains
│   ├── gain_scheduler.m             ← Power-level gain scheduling logic
│   ├── fuzzy_pid_reference.m        ← Fuzzy-PID controller (training data source)
│   ├── train_nn.py                  ← Neural network training script (Python)
│   └── generate_training_data.m     ← Closed-loop data generation from Fuzzy-PID
│
├── simulink_models/
│   ├── reactor_plant.slx            ← Nonlinear reactor plant model
│   ├── fuzzy_pid_closed_loop.slx    ← Fuzzy-PID closed-loop (data generation)
│   └── nn_pid_closed_loop.slx       ← Final NN-based adaptive PID system
│
├── docs/
│   ├── architecture.md              ← Detailed system design explanation
│   ├── training_pipeline.md         ← Data generation → training → deployment steps
│   └── results_analysis.md          ← Performance comparison tables and plots
│
└── results/
    ├── training_loss.png
    ├── step_response_comparison.png
    ├── disturbance_rejection.png
    └── power_sweep_stability.png
