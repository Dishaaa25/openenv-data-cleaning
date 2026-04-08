def compute_reward(
    old_quality: float,
    new_quality: float,
    action_valid: bool,
    resolved_dependency_correctly: bool,
) -> float:
    if not action_valid:
        return 0.01

    progress = new_quality - old_quality
    ordering_bonus = 0.05 if resolved_dependency_correctly else 0.0
    step_cost = -0.01

    reward = progress + ordering_bonus + step_cost
    return round(max(0.01, min(0.99, reward)), 4)
