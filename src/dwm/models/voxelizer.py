import torch


class Voxelizer(torch.nn.Module):
    """Voxelizer for converting Lidar point cloud to image"""

    def __init__(self, x_min, x_max, y_min, y_max, step, z_min, z_max, z_step):
        super().__init__()

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step = step
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step

        self.width = round((self.x_max - self.x_min) / self.step)
        self.height = round((self.y_max - self.y_min) / self.step)
        self.z_depth = round((self.z_max - self.z_min) / self.z_step)
        self.depth = self.z_depth

    def voxelize_single(self, lidar, bev):
        """Voxelize a single lidar sweep into image frame
        Image frame:
        1. Increasing depth indices corresponds to increasing real world z
            values.
        2. Increasing height indices corresponds to decreasing real world y
            values.
        3. Increasing width indices corresponds to increasing real world x
            values.
        Args:
            lidar (torch.Tensor N x 4 or N x 5) x, y, z, intensity, height_to_ground (optional)
            bev (torch.Tensor D x H x W) D = depth, the bird's eye view
                raster to populate
        """

        # 1 & 2. Convert points to tensor index location. Clamp z indices to
        # valid range.
        indices_h = torch.floor((lidar[:, 1] - self.y_min) / self.step).long()
        indices_w = torch.floor((lidar[:, 0] - self.x_min) / self.step).long()
        indices_d = torch.floor(
            (lidar[:, 2] - self.z_min) / self.z_step).long()

        # 3. Remove points out of bound
        valid_mask = ~torch.any(
            torch.stack(
                [
                    indices_h < 0,
                    indices_h >= self.height,
                    indices_w < 0,
                    indices_w >= self.width,
                    indices_d < 0,
                    indices_d >= self.z_depth,
                ]
            ),
            dim=0,
        )
        indices_h = indices_h[valid_mask]
        indices_w = indices_w[valid_mask]
        indices_d = indices_d[valid_mask]
        # 4. Assign indices to 1
        bev[indices_d, indices_h, indices_w] = 1.0

    def forward(self, lidars):
        """Voxelize multiple sweeps in the current vehicle frame into voxels
            in image frame
        Args:
            list(list(tensor)): B * T * tensor[N x 4],
                where B = batch_size, T = 5, N is variable,
                4 = [x, y, z, intensity]
        Returns:
            tensor: [B x D x H x W], B = batch_size, D = T * depth, H = height,
                W = width
        """
        batch_size = len(lidars)
        assert batch_size > 0 and len(lidars[0]) > 0
        num_sweep = len(lidars[0])

        bev = torch.zeros(
            (batch_size, num_sweep, self.depth, self.height, self.width),
            dtype=torch.float,
            device=lidars[0][0][0].device,
        )
        for b in range(batch_size):
            assert len(lidars[b]) == num_sweep
            for i in range(num_sweep):
                self.voxelize_single(lidars[b][i], bev[b][i])

        return bev

    def get_voxel_coordinates(self, downsample_scale = 1):
        x_coord = torch.arange(self.x_min, self.x_max, ((self.x_max - self.x_min) / self.width) / downsample_scale) + self.step / 2
        y_coord = torch.arange(self.y_min, self.y_max, ((self.y_max - self.y_min) / self.height) / downsample_scale) + self.step / 2
        z_coord = torch.arange(self.z_min, self.z_max, ((self.z_max - self.z_min) / self.z_depth) / downsample_scale) + self.z_step / 2
        x_grid, y_grid, z_grid = torch.meshgrid(z_coord, y_coord, x_coord)

        return torch.stack([x_grid, y_grid, z_grid], dim = -1)
