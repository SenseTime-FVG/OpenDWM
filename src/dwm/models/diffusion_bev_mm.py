import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Any, List

class BEVWDiffusion(nn.Module):
    def __init__(self,
                 action_threshold: List[float] = [0.5, 1.0, 1.0]):
        """
        Args:
            action_threshold: List[float] with length 3 - Representing the threshold for turn left or right, speed up, slow down. Other actions are "keep normal".
        """
        super(BEVWDiffusion, self).__init__()
        self.action_threshold = action_threshold
    
    @staticmethod
    def get_action_condition(ego_transform, turn_threshold=0.1, speed_threshold=0.5):
        """
        Determine the actions of the car based on the ego_transform tensor.
        
        Args:
            ego_transform: [batch_size, seq_len, 4, 4] - Representing the transform 
                           from the car coordinate to the world coordinate.
            turn_threshold (float): Threshold for determining left/right turns.
            speed_threshold (float): Threshold for determining speed up/slow down.
        
        Returns:
            list: List of actions for each timestamp.
        """
        bs, sequence_length = ego_transform.shape[:1]
        actions = []

        for i in range(1, sequence_length):
            # Extract the rotation and translation components
            prev_rotation = ego_transform[:, i-1, :3, :3]
            curr_rotation = ego_transform[:, i, :3, :3]
            prev_translation = ego_transform[:, i-1, :3, 3]
            curr_translation = ego_transform[:, i, :3, 3]

            # Calculate the change in rotation (yaw)
            rotation_change = torch.atan2(curr_rotation[:, 1, 0], curr_rotation[:, 0, 0]) - \
                            torch.atan2(prev_rotation[:, 1, 0], prev_rotation[:, 0, 0])
            
            # Calculate the change in position (speed)
            speed_change = torch.norm(curr_translation - prev_translation)

            # Determine the action based on thresholds
            if rotation_change > turn_threshold:
                action = "turn right"
            elif rotation_change < -turn_threshold:
                action = "turn left"
            elif speed_change > speed_threshold:
                action = "speed up"
            elif speed_change < -speed_threshold:
                action = "slow down"
            else:
                action = "keep normal"

            actions.append(action)

        return actions

# Example usage:
if __name__ == "__main__":
    x = torch.randn(32, 3, 6, 32, 32)

    model = BEVWDiffusion()
    # Dummy input
    output = model(x)
    print(output.shape)  # Expected output shape: (32, spatial_dim, temporal_dim, input_dim)


