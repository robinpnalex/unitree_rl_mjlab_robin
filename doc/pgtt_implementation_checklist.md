# PGTT implementation checklist for Go2 EDU

**Paper:** Ntagkas et al., *PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion*, arXiv:2510.18348. The arXiv page currently identifies v2 (22 July 2026) as the latest revision and says it was accepted at IROS 2026. Use the paper's method/equations as the source of truth; the `Aaryan127/unitree_rl_mjlab` `pgtt` branch is a reference implementation, not the specification.

## How to use this checklist

- Work top to bottom. Do not begin full training until the task builds and the sensor/observation checks pass.
- For values and formulae, record the paper section/table/equation and the chosen implementation in the notes. Do not infer paper hyperparameters from another repository.
- Keep PGTT in a separate Go2 task/config initially so the existing baseline remains available.
- Mark an item complete only after it has been tested, not merely configured.

## 1. Translate the paper into an implementation spec

- [ ] Record the paper version used and review the method, experiment setup, and supplementary material if available.
- [ ] Write down the exact per-leg phase definition, phase offsets, phase progression, and phase reset/initialization behavior.
- [ ] Transcribe the cubic Hermite spline definition for desired foot height, including swing/stance intervals, boundary conditions, and parameters.
- [ ] Transcribe how local heightmap statistics modify swing height, including the neighborhood, statistic(s), scaling, clamping, and units.
- [ ] Transcribe the swing-phase contact penalty and its phase mask, sensor signal, and coefficient.
- [ ] Record the full observation vector and ordering, action definition/scaling, and policy update/control frequency.
- [ ] Record all other reward terms, weights, termination rules, terrain curriculum, and domain randomization used in the reported experiments.
- [ ] Distinguish required method components from experiment-specific settings and engineering choices in the reference code.

**Spec notes / paper locations:**

| Topic | Paper section/equation/table | Required definition / value | Repo implementation location |
|---|---|---|---|
| Phase | | | |
| Foot-height spline | | | |
| Terrain adaptation | | | |
| Swing contact penalty | | | |
| Observations/actions | | | |
| Terrain curriculum/randomization | | | |
| Training/evaluation | | | |

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

**Not yet implemented/verified:** paper equations and hyperparameters have not yet been transcribed into this checklist; PGTT-specific rewards and phase features are not implemented in this repo; environment construction/training has not been verified here because the local Python environment is missing `warp`.

**Immediate next action:** fill in Section 1 from the paper, then audit the Go2 model/sensor/observation paths in Section 2 before editing task code.
