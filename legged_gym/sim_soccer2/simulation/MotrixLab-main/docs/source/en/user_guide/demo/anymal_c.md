# ANYmal-C Navigation

## Overview

This document describes in detail the navigation task environment based on the ANYmal-C quadruped robot. This environment is part of the navigation task collection in the MotrixLab project, providing a complete implementation for training quadruped robots to navigate to target positions and orientations using reinforcement learning.

```{video} /_static/videos/anymal_c.mp4
:poster: _static/images/poster/anymal_c.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## Environment Description

The ANYmal-C navigation task environment is built based on the real ANYmal-C quadruped robot, designed to train robots to navigate to specified target positions and orientations on flat terrain. This environment uses the MotrixSim physics engine for simulation, providing high-fidelity dynamic simulation.

### Robot Structure

ANYmal-C is a quadruped robot composed of the following main parts:

-   **Base**: The core torso of the robot, containing sensor modules such as IMU, camera, and lidar
-   **Four Legs**: Each leg contains three joints
    -   HAA (Hip Abduction/Adduction): Hip abduction/adduction joint
    -   HFE (Hip Flexion/Extension): Hip flexion/extension joint
    -   KFE (Knee Flexion/Extension): Knee flexion/extension
-   **Four Feet**: Spherical contact geometries that generate frictional contact with the ground

### Task Objective

The robot needs to complete the following navigation objectives:

1.  **Position Navigation**: Move to the specified target position (XY plane coordinates)
2.  **Orientation Control**: Adjust robot orientation to the target heading angle (yaw angle)
3.  **Stable Stop**: Maintain stable standing after reaching the target, with linear and angular velocities approaching zero

The environment provides visualization markers:

-   **Green Arrow**: Indicates target position and orientation
-   **Green Arrow Above Robot**: Current actual movement direction
-   **Blue Arrow Above Robot**: Desired movement direction

---

## Action Space

The action space is `Box(-1.0, 1.0, (12,), float32)`, representing position control commands applied to 12 joints (offsets relative to the default standing posture).

### Control Mode

The environment uses position control mode. Actions are converted to joint target positions as follows:

```
Target Joint Angle = Default Joint Angle + (Action Value × Action Scale)
```

Where the action scale is specified by the configuration parameter `control_config.action_scale`.

### Action Dimension Details

| Index | Action Description   | Control Range | Joint Name | Joint Type |
| ----- | -------------------- | ------------- | ---------- | ---------- |
| 0     | Left Front Hip HAA   | -1.0 ~ 1.0    | LF_HAA     | hinge      |
| 1     | Left Front Hip HFE   | -1.0 ~ 1.0    | LF_HFE     | hinge      |
| 2     | Left Front Knee KFE  | -1.0 ~ 1.0    | LF_KFE     | hinge      |
| 3     | Right Front Hip HAA  | -1.0 ~ 1.0    | RF_HAA     | hinge      |
| 4     | Right Front Hip HFE  | -1.0 ~ 1.0    | RF_HFE     | hinge      |
| 5     | Right Front Knee KFE | -1.0 ~ 1.0    | RF_KFE     | hinge      |
| 6     | Left Hind Hip HAA    | -1.0 ~ 1.0    | LH_HAA     | hinge      |
| 7     | Left Hind Hip HFE    | -1.0 ~ 1.0    | LH_HFE     | hinge      |
| 8     | Left Hind Knee KFE   | -1.0 ~ 1.0    | LH_KFE     | hinge      |
| 9     | Right Hind Hip HAA   | -1.0 ~ 1.0    | RH_HAA     | hinge      |
| 10    | Right Hind Hip HFE   | -1.0 ~ 1.0    | RH_HFE     | hinge      |
| 11    | Right Hind Knee KFE  | -1.0 ~ 1.0    | RH_KFE     | hinge      |

### PD Control Parameters

The underlying system uses position actuators with PD control parameters defined in the XML file:

-   **kp (Proportional Gain)**: 200
-   **kv (Derivative Gain)**: 1
-   **Torque Limit**: -140 N·m ~ 140 N·m

---

## Observation Space

The observation space is `Box(-inf, inf, (54,), float32)`, containing the robot's proprioceptive information, task-related information, and action history.

### Observation Components

The observation vector consists of the following parts (in order):

1.  **Proprioceptive State (33 dimensions)**

    -   Base Linear Velocity (3 dim): Linear velocity of robot base in world frame [vx, vy, vz]
    -   Angular Velocity (3 dim): Angular velocity from gyroscope [ωx, ωy, ωz]
    -   Projected Gravity (3 dim): Gravity vector projected in robot body frame
    -   Joint Angles (12 dim): 12 joint angle offsets relative to default standing posture
    -   Joint Velocities (12 dim): 12 joint angular velocities

2.  **Action History (12 dimensions)**

    -   Action executed at previous timestep

3.  **Velocity Commands (3 dimensions)**

    -   Desired Linear Velocity XY (2 dim): Desired linear velocity calculated from position error
    -   Desired Angular Velocity Z (1 dim): Desired angular velocity calculated from orientation error

4.  **Task State (6 dimensions)**
    -   Position Error Vector (2 dim): XY plane error vector to target position (normalized)
    -   Orientation Error (1 dim): Angle difference to target orientation (normalized to [-1, 1])
    -   Distance (1 dim): Euclidean distance to target (normalized)
    -   Arrival Flag (1 dim): Whether both position and orientation arrival conditions are satisfied (0 or 1)
    -   Stop Ready Flag (1 dim): Whether stop criteria are met (arrived and angular velocity near zero)

### Observation Details

| Index | Observation Content                        | Min  | Max | Normalization Coefficient | Unit          |
| ----- | ------------------------------------------ | ---- | --- | ------------------------- | ------------- |
| 0-2   | Base Linear Velocity (vx, vy, vz)          | -inf | inf | normalization.lin_vel     | m/s           |
| 3-5   | Angular Velocity (ωx, ωy, ωz)              | -inf | inf | normalization.ang_vel     | rad/s         |
| 6-8   | Projected Gravity (gx, gy, gz)             | -1   | 1   | 1.0                       | Dimensionless |
| 9-20  | Joint Angle Offsets (12 joints)            | -inf | inf | normalization.dof_pos     | rad           |
| 21-32 | Joint Angular Velocities (12 joints)       | -inf | inf | normalization.dof_vel     | rad/s         |
| 33-44 | Previous Action                            | -1   | 1   | 1.0                       | Dimensionless |
| 45-47 | Velocity Commands (vx_cmd, vy_cmd, ωz_cmd) | -inf | inf | commands_scale            | m/s, rad/s    |
| 48-49 | Position Error Vector (Δx, Δy)             | -inf | inf | 1/5.0                     | m             |
| 50    | Orientation Error                          | -1   | 1   | 1/π                       | rad           |
| 51    | Distance to Target                         | 0    | 1   | 1/5.0 (clipped)           | m             |
| 52    | Arrival Flag                               | 0    | 1   | 1.0                       | Boolean       |
| 53    | Stop Ready Flag                            | 0    | 1   | 1.0                       | Boolean       |

### Sensor Information

The environment uses the following sensors to obtain state:

-   **framelinvel** (name: base_linvel): Base linear velocity sensor
-   **gyro** (name: base_gyro): Gyroscope sensor, mounted at IMU site

---

## Reward Function

The reward function uses a composite design, employing different reward strategies based on whether the robot has reached the target.

### Rewards Before Reaching Target

Total Reward = Velocity Tracking Reward + Approach Reward - Penalty Terms

**Main Reward Terms:**

1.  **Linear Velocity Tracking Reward** (Weight: 1.5)

    -   Formula: `1.5 × exp(-||v_xy - v_cmd||² / 0.25)`
    -   Encourages robot to track desired XY plane linear velocity

2.  **Angular Velocity Tracking Reward** (Weight: 0.3)

    -   Formula: `0.3 × exp(-(ωz - ωz_cmd)² / 0.25)`
    -   Encourages robot to track desired yaw angular velocity

3.  **Approach Reward**
    -   Formula: `clip((Historical Minimum Distance - Current Distance) × 4.0, -1.0, 1.0)`
    -   Rewards robot for progress when getting closer to target

**Penalty Terms:**

-   Z-axis Linear Velocity Penalty (Weight: 2.0): `-2.0 × vz²`
-   XY-axis Angular Velocity Penalty (Weight: 0.05): `-0.05 × (ωx² + ωy²)`
-   Torque Penalty (Weight: 0.00001): `-0.00001 × ||τ||²`
-   Action Rate Penalty (Weight: 0.001): `-0.001 × ||Δa||²`

### Rewards After Reaching Target

Total Reward = Stop Reward + First Arrival Reward - Penalty Terms

**Main Reward Terms:**

1.  **Stop Base Reward**

    -   Formula: `2 × [0.8 × exp(-(v_xy/0.2)²) + 1.2 × exp(-(ωz/0.1)⁴)]`
    -   Encourages robot to maintain low velocity and angular velocity after arrival

2.  **Zero Angular Velocity Reward** (Extra Reward: 6.0)

    -   Condition: Arrived at target and |ωz| < 0.05 rad/s
    -   Encourages robot to completely stop rotation

3.  **First Arrival Reward** (One-time: 10.0)
    -   Condition: First time both position and orientation arrival conditions are satisfied
    -   Provides clear signal for reaching target

**Penalty Terms:** (Same as before arrival)

### Termination Condition Penalties

An additional penalty of -20.0 is applied in the following cases:

-   Joint velocity exceeds limit (exceeds `max_dof_vel` configuration value)
-   Joint velocity is NaN or Inf
-   Robot base contacts ground
-   Robot rollover (tilt angle exceeds 75°)

### Arrival Criteria

-   **Position Arrival**: Distance to target < 0.3 meters
-   **Orientation Arrival**: Orientation error < 15°
-   **Complete Arrival**: Both position arrival and orientation arrival conditions satisfied
-   **Stop Ready**: Complete arrival AND |ωz| < 0.05 rad/s

### info Return Content

The info dictionary returned each step contains the following debug information:

-   `pose_commands`: Current target position and orientation [x, y, yaw]
-   `last_actions`: Previous action
-   `current_actions`: Current action
-   `steps`: Current episode step count
-   `ever_reached`: Whether target has ever been reached
-   `min_distance`: Historical minimum distance (used to calculate approach reward)

---

## Initial State

### Robot Initialization

**Position Initialization:**

The robot's initial position in world coordinates is randomly sampled within the range defined by configuration parameter `init_state.pos_randomization_range`:

-   X coordinate: Uniform random sampling in [x_min, x_max]
-   Y coordinate: Uniform random sampling in [y_min, y_max]
-   Z coordinate: Fixed at 0.56 meters (to avoid falling sensation)

**Orientation Initialization:**

-   Robot orientation (quaternion): Initialized to unit quaternion [0, 0, 0, 1], indicating forward orientation
-   No random noise added to quaternion (ensures initial stability)

**Joint Initialization:**

Joint angles are set to default standing posture, defined by configuration parameter `init_state.default_joint_angles`. No random noise added to joint angles (ensures stable standing initially).

**Velocity Initialization:**

All linear and angular velocities are initialized to zero, ensuring robot starts from stationary state.

### Target Generation

**Target Position:**

Target position is generated relative to robot's initial position:

```
Target Position = Robot Initial Position + Random Offset
```

Random offset is sampled within the range defined by configuration parameter `commands.pose_command_range`:

-   X direction offset: [pose_command_range[0], pose_command_range[3]]
-   Y direction offset: [pose_command_range[1], pose_command_range[4]]

**Target Orientation:**

Target orientation (yaw angle) is randomly generated in absolute reference frame:

-   Orientation angle: [pose_command_range[2], pose_command_range[5]]

### Visualization Marker Initialization

-   **Target Marker** (green arrow): Set to target position and orientation
-   **Movement Direction Arrows**: Initialized at 0.76 meters above robot

---

## Episode Termination

### Termination Conditions

The environment terminates an episode when any of the following conditions are met:

1.  **Timeout Termination**

    -   Condition: Episode reaches `max_episode_second` configuration value
    -   Description: Prevents infinite episodes

2.  **Joint Velocity Anomaly**

    -   Condition: Absolute value of any joint velocity exceeds `max_dof_vel`
    -   Condition: Joint velocity is NaN, Inf, or exceeds 1e6
    -   Description: Prevents numerical divergence and physical instability

3.  **Base Contacts Ground**

    -   Condition: Robot base (geometries defined by `terminate_after_contacts_on` configuration parameter) contacts ground
    -   Description: Robot fell or pose failure

4.  **Rollover**
    -   Condition: Robot tilt angle exceeds 75°
    -   Calculation: Tilt angle calculated via projected gravity vector `arctan2(||g_xy||, |g_z|)`
    -   Description: Robot severely rolled over

### Success Conditions

Although the environment does not terminate upon success, task success is defined as:

-   Robot reaches target position and orientation (both position threshold < 0.3m and orientation threshold < 15° satisfied)
-   Robot maintains stable stop (linear velocity < 0.05 m/s, angular velocity < 0.05 rad/s)
