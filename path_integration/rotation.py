import torch
# Assuming s_t1 and s_t2 are your 3D tensors for locations at t1 and t2
s_t1 = torch.rand(3)
s_t2 = torch.rand(3)

def construct_velocity_rotation_tensor(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    # 1. Location at t1 is just s_t1
    location_t1 = vec1

    # 2. Linear velocity at t2 (magnitude of the difference, assuming Δt = 1)
    linear_velocity = (vec2 - vec1).norm()

    # 3. Angular velocity at t2
    direction_t1 = vec1 / vec1.norm()  # Normalize
    direction_t2 = vec2 / vec2.norm()  # Normalize

    # Angle between direction_t1 and direction_t2
    cos_theta = torch.dot(direction_t1, direction_t2)
    angle = torch.acos(cos_theta)  # This is in radians

    # Axis of rotation (normalized)
    axis_of_rotation = torch.cross(direction_t1, direction_t2)
    axis_of_rotation_normalized = axis_of_rotation / axis_of_rotation.norm()

    # Angular velocity (magnitude and direction, assuming Δt = 1)
    angular_velocity = axis_of_rotation_normalized * angle  # Magnitude is the angle in radians
    sine_cosine_rep = torch.tensor([torch.sin(angle), torch.cos(angle) for angle in angular_velocity])
    return torch.cat((location_t1, linear_velocity, angular_velocity))
# 1. Location at t1 is just s_t1
location_t1 = s_t1

# 2. Linear velocity at t2 (magnitude of the difference, assuming Δt = 1)
linear_velocity = (s_t2 - s_t1).norm()

# 3. Angular velocity at t2
direction_t1 = s_t1 / s_t1.norm()  # Normalize
direction_t2 = s_t2 / s_t2.norm()  # Normalize

# Angle between direction_t1 and direction_t2
cos_theta = torch.dot(direction_t1, direction_t2)
angle = torch.acos(cos_theta)  # This is in radians

# Axis of rotation (normalized)
axis_of_rotation = torch.cross(direction_t1, direction_t2)
axis_of_rotation_normalized = axis_of_rotation / axis_of_rotation.norm()

# Angular velocity (magnitude and direction, assuming Δt = 1)
angular_velocity = axis_of_rotation_normalized * angle  # Magnitude is the angle in radians

print("Location at t1:", location_t1)
print("Linear velocity magnitude at t2:", linear_velocity)
print("Angular velocity vector at t2:", angular_velocity)