class DataCleaningGrader:
    def grade(self, final_state: dict, task_config: dict) -> float:
        issues_fixed = len(final_state["resolved_issues"])
        total_issues = task_config["total_issues"]
        steps_taken = task_config["max_steps"] - final_state["steps_remaining"]
        wrong_actions = sum(1 for action in final_state["action_history"] if action.get("error"))

        correctness = issues_fixed / total_issues if total_issues > 0 else 1.0
        efficiency = max(0, 1 - steps_taken / (2 * total_issues)) if total_issues > 0 else 1.0
        penalty = wrong_actions * 0.05

        score = 0.8 * correctness + 0.2 * efficiency - penalty
        # Phase 2 requires task scores to stay strictly inside (0, 1).
        return round(max(0.01, min(0.99, score)), 2)
