from uqc.physics import GmonSystem, GmonSystemConfig
from uqc.eval import ControlPlan, robustness_metrics
config = GmonSystemConfig()
system = GmonSystem(config=config)  # usa i tuoi stessi parametri --alpha 2.2 --gamma pi/2   --noise-optimized   --iterations 100   --episodes-per-batch 3000   --eval-every 1 --save-every 1   --dt-ns 2   --max-time-ns 240   --runtime-norm-ns 60   --termination-cost 0.5   --num-workers 4   --episodes-per-task 8   --out ./out/trpo_single --train-noise-std 0.0
plan = ControlPlan.load('./runs/trpo_noise_alpha_2.2_gamma_pi2/best_control_plan.npz')
metrics = robustness_metrics(system, plan, sigma_mhz=1.0, num_samples=60)
print('Average fidelity (paper metric):', metrics['average_fidelity'])
print('Average gate fidelity:', metrics['average_gate_fidelity'])
print('Fidelity variance:', metrics['fidelity_variance'])
