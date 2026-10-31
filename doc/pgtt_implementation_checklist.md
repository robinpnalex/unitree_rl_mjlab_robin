# PGTT implementation checklist for Go2 EDU

**Paper:** Ntagkas et al., *PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion*, arXiv:2510.18348. The arXiv page currently identifies v2 (22 July 2026) as the latest revision and says it was accepted at IROS 2026. Use the paper's method/equations as the source of truth; the `Aaryan127/unitree_rl_mjlab` `pgtt` branch is a reference implementation, not the specification.

## How to use this checklist

- Work top to bottom. Do not begin full training until the task builds and the sensor/observation checks pass.
- For values and formulae, record the paper section/table/equation and the chosen implementation in the notes. Do not infer paper hyperparameters from another repository.
- Keep PGTT in a separate Go2 task/config initially so the existing baseline remains available.
- Mark an item complete only after it has been tested, not merely configured.

## 1. Translate the paper into an implementation spec

- [x] Record the paper version used and review the method and experiment setup.
- [ ] Write down the exact per-leg phase definition, phase offsets, phase progression, and phase reset/initialization behavior. (The formula and author-code offsets are recorded below; episode-reset behavior still needs verification.)
- [x] Transcribe the cubic Hermite spline definition for desired foot height, including swing/stance intervals and boundary conditions.
- [ ] Transcribe how local heightmap statistics modify swing height, including the neighborhood, scaling/clamping, and units. (The statistic and addition are known; the exact sampling neighborhood and any clipping need verification.)
- [x] Transcribe the swing-phase contact penalty and its phase mask, sensor signal, and coefficient.
- [x] Record the paper's observation vector, action type, and control frequency. (The exact action scale and every implementation-specific preprocessing detail still need verification.)
- [ ] Record all other reward terms, weights, termination rules, terrain curriculum, and domain randomization used in the reported experiments.
- [x] Distinguish required method components from experiment-specific settings and engineering choices in the reference code.

**Spec notes / paper locations:**

| Topic | Paper section/equation/table | Required definition / value | Repo implementation location |
|---|---|---|---|
| Phase | III-C, Eq. 2; IV-B; authors' `robots/gait.py` | `phi_i(t) = (phi_i,0 + 2*pi*f*t) mod 2*pi`; stance is `[0, 2*pi*p_stance)`, swing is the remainder. Paper says sample `f ~ U[1,3] Hz`. Authors' code uses `p_stance=0.5`, offsets `[0, pi, pi, 0]` (FR, FL, RR, RL), initializes those offsets on reset, and advances phase by `2*pi*f*ctrl_dt`. The paper's frequency is `U[1,3] Hz`; confirm the authors' training override uses that range. | Not implemented yet |
| Foot-height spline | III-E, Eqs. 5-9; Table I | Cubic Hermite interpolation in phase coordinates with zero endpoint tangents. Stance height is `d_b`; swing-up goes from `d_b` to `d_s + delta_H_i`; swing-down returns to `d_b`. `T_stance=2*pi*p_stance`, `T_swing=pi*(1-p_stance)`, `T_peak=pi*(1+p_stance)`. Paper foot-phase reward is `sum_i exp(-((p_des_f,z,i - p_f,z,i)^2 / sigma_f))`, `sigma_f=0.05`, weight `+1.0`. Note: authors' `joystick_base.py` currently computes `exp(-sum_i(error_i^2)/sigma_f)`, which is not algebraically the same; replicate the paper equation unless intentionally reproducing that code variant. | Not implemented yet |
| Terrain adaptation | III-E, III-B; IV-G | Per leg, compute `delta_H_i = H_max,i(h_t) - H_min,i(h_t)` from terrain heights in the world frame and add it to the swing apex. Deployment map is an 11x9 robot-centric grid over 1.1 m x 0.9 m. The paper does not fully specify the exact per-foot sampling neighborhood. Authors' `joystick_base.py` divides the 11x9 map into four quadrants and uses each quadrant's max-minus-min as that leg's height variation; verify orientation/leg-to-quadrant mapping before adopting it. | Current Go2 scanner and `mdp.height_map`; representation differs and still needs comparison |
| Swing contact penalty | III-E, Eq. 9; Table I | `sum_i 1[pi <= phi_i < 2*pi] * c_i`, where `c_i=1` when foot i contacts ground; weight `-0.25`. | Existing `feet_ground_contact` sensor is configured; PGTT reward not implemented |
| Observations/actions | III-D, Eqs. 3-4; IV-G | Actor input order: body angular velocity, local gravity vector, joint angles, joint velocities, `cos(phi)` for four legs, `sin(phi)` for four legs, flattened heightmap, frequency, previous action, velocity command. Critic additionally receives privileged local linear velocity. Action is 12 desired joint angles around stand pose, `q_des=q_stand+k*a`; policy/control rate is 50 Hz; simulation timestep is 0.005 s. Numeric `k` must be matched to the authors' implementation; authors' Go2 config sets `action_scale=0.5`. Authors' Go2 code uses an 11x9 heightmap, with 0.1 m grid spacing. | Existing policy/extero groups; phase and frequency absent |
| Terrain curriculum/randomization | III-F-H; IV-C | WFC stair-like terrain with four difficulty levels. Paper reports step width 0.3-0.45 m, 2-4 steps, level-dependent step heights 1-3, 1-7, 1-10, 1-13 cm. Randomize sensor noise, robot masses/default joint positions/motor gains/friction, and terrain friction. Exact per-parameter distributions need table/code verification. | Existing terrain curriculum/randomization; not yet compared |
| Training/evaluation | IV-B-F | PPO; command `[v_x,v_y,omega_z]`, frequency sampled `U[1,3] Hz`; 5 seeds in reported comparisons. Paper reports success-rate evaluation over 1000 rollouts and disturbance/discrete-obstacle tests. | Existing RSL-RL PPO task; experiment protocol not configured for PGTT |

**Source distinctions:** Paper: https://arxiv.org/abs/2510.18348 (v2). Authors' code: https://github.com/NtagkasAlex/phase_guided_terrain_traversal. Treat paper equations/settings as requirements; use code only to resolve implementation details the paper omits, and record disagreements rather than silently copying them. A concrete discrepancy to preserve: the paper's phase reward is a sum of per-foot exponentials, while the current authors' code exponentiates the sum of squared errors.

## 2. Audit the existing Go2 task and deployment constraints

- [ ] Identify the Go2 model's body, joint, actuator, foot-geom, and sensor names.
- [ ] Verify all four feet are identified consistently; check that sensor regexes select the intended geoms and no leg is silently omitted.
- [ ] Verify contact signals are available per foot and determine whether they represent contact state, force, or another quantity.
- [ ] Verify the height scanner's pose/frame, field of view, grid spacing, ray count, update rate, and treatment of invalid/no-hit rays.
- [ ] Confirm the scanner observes useful terrain ahead of both front and hind feet, not just terrain under the robot.
- [ ] Compare the current height-map encoding with the paper's required representation; document any intentional mismatch.
- [ ] Confirm every actor observation is available on the physical Go2 EDU or has an explicitly justified deployment substitute.
- [ ] Check deployment observation ordering, normalization, units, history, and control rate against the training policy.
- [ ] Identify privileged observations, if any, and ensure they are not accidentally required by the deployed actor.

**Known audit items from the current repo:** Go2 height scanner is connected to actor observations, but uses a compact foot-relative representation that differs from the reference branch's map path. The `shank_thigh_contact` configuration and its use by critic observations need runtime verification. These are checks, not yet confirmed defects.

## 3. Implement a dedicated PGTT Go2 task

- [ ] Add a separate PGTT Go2 task/config; preserve the current velocity task as a baseline.
- [ ] Implement phase as a deterministic gait clock/observation according to the paper, not as an action unless the paper explicitly specifies otherwise.
- [ ] Add the exact per-leg phase features and ensure phase timing matches simulation/control time.
- [ ] Keep the policy action interface consistent with the paper (the abstract describes direct joint-space control); verify action scaling, offsets, and actuator targets.
- [ ] Implement the cubic-Hermite desired-foot-height computation and terrain-adaptive swing-height adjustment from the paper.
- [ ] Implement the swing-only contact penalty with correct per-foot phase masks and contact data.
- [ ] Add the terrain observation representation required by the selected method and confirm its frame/flattening/order.
- [ ] Add or adapt the paper's remaining reward, termination, curriculum, and randomization settings.
- [ ] Keep changes modular so each PGTT reward/observation can be enabled and inspected independently.
- [ ] Add focused tests for phase boundaries, spline continuity/values, terrain-height adjustment, and swing/stance contact masking where the code allows.

## 4. Build and sensor/observation validation

- [ ] Construct the PGTT environment successfully in the configured project environment.
- [ ] Run a minimal CPU/debug environment before GPU-scale training.
- [ ] Print/check observation term names, shapes, order, units, and finite values.
- [ ] Visualize the scan grid and verify it tracks the intended terrain frame as the robot moves/turns.
- [ ] Verify contact output for each foot in a controlled stance/lift test.
- [ ] Log phase, desired foot height, measured foot height, local terrain statistic, contact state, and each reward term.
- [ ] Check rewards are finite and have the expected sign/magnitude in hand-constructed swing and stance cases.
- [ ] Audit for reward exploits: verify body-ground contacts terminate, commanded velocity cannot be ignored for high shaping reward, and tracking/reward terms correlate with task success.
- [ ] Confirm reset behavior does not start with stale phase, sensor, or contact history.
- [ ] Run a short smoke training job and confirm actions, episode resets, and checkpoint saving work.

## 5. Train and evaluate incrementally

- [ ] Establish a baseline run using the existing Go2 velocity task and record its configuration/checkpoint.
- [ ] Train PGTT first on a small environment count/short run to catch configuration and numerical issues.
- [ ] Scale training only after the smoke run and logs look correct; keep seeds and configs for reproducibility.
- [ ] Evaluate on held-out terrain and commands, including stairs/discrete obstacles and disturbances as specified by the paper.
- [ ] Compare success rate, falls, velocity tracking, foot clearance/contact events, and robustness—not reward alone.
- [ ] Run ablations of phase-guided height shaping and swing-contact penalty if replication scope requires it.
- [ ] Compare the results fairly against the baseline with matching terrain, command, and evaluation conditions.

## 6. Go2 EDU sim-to-real readiness

- [ ] Confirm the physical sensor path supplies a heightmap with the same frame, dimensions, ordering, units, and update behavior expected by the policy.
- [ ] Validate LiDAR/elevation-to-heightmap conversion, latency, dropouts, and noise; do not train against an ideal sensor if deployment uses a materially different one without addressing the gap.
- [ ] Verify joint order, signs, offsets, observation normalization, action scaling, and policy/control rates against the robot interface.
- [ ] Test safely in simulation, then in a controlled low-risk real-robot setup with an emergency stop and conservative limits.
- [ ] Check actuator delay, state-estimation error, contact sensing assumptions, and terrain-map alignment before increasing speed or terrain difficulty.

## Current checkpoint

**Implemented but still requiring validation:** a separate PGTT Go2 config, phase/frequency observations, terrain heightmap observation, phase-guided foot-height reward, swing-contact penalty, and paper-inspired shared rewards are present. The task has completed one user-reported 1-iteration CUDA smoke run. That run is not enough to verify episode-level reward behavior or rule out reward exploits. PGTT retains non-foot ground-contact and fall terminations as guardrails. The terrain generator, randomization, and evaluation protocol still differ from the paper or need verification.

**Immediate next action:** run a short multi-environment audit and inspect episode metrics, contact/fall rates, velocity tracking, and per-term reward magnitudes before committing to a long run.
