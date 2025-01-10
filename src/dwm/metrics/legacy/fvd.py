import einops
import torch
import torch.distributed
import torchmetrics

# PYTHONPATH includes ${workspaceFolder}/externals/TATS/tats/fvd
import pytorch_i3d

MAX_BATCH = 16


@torch.no_grad()
def get_logits(i3d, videos, device):
    # assert videos.shape[0] % MAX_BATCH == 0
    logits = []
    for i in range(0, videos.shape[0], MAX_BATCH):
        batch = videos[i:i + MAX_BATCH].to(device)
        logits.append(i3d(batch))

    logits = torch.cat(logits, dim=0)
    return logits


def _symmetric_matrix_square_root(mat, eps=1e-10):
    # https://github.com/tensorflow/gan/blob/
    # de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/
    # classifier_metrics.py#L161
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


def trace_sqrt_product(sigma, sigma_v):
    # https://github.com/tensorflow/gan/blob/
    # de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/
    # classifier_metrics.py#L400
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(
        sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''

    # https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2

    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)  # unbiased estimate
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()  # if complex: mt = m.t().conj()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


class FrechetVideoDistance(torchmetrics.Metric):
    def __init__(
        self, inception_3d_checkpoint_path, max_batch_size=8, **kwargs
    ):
        super().__init__(**kwargs)

        self.inception = pytorch_i3d.InceptionI3d()
        self.inception.eval()
        state_dict = torch.load(
            inception_3d_checkpoint_path, map_location="cpu")
        self.inception.load_state_dict(state_dict)

        self.real_features = []
        self.fake_features = []
        self.max_batch_size = max_batch_size

        # this is fixed in fvd
        self.target_resolution = (224, 224)
        self.i3d_min = 10
        self.warned = False

    def update(self, images, real=True):
        """
        Notes:
            1. transform is recommended to be vae transform, this align the
               gt/infer images
        """
        target_resolution = (self.i3d_min, ) + self.target_resolution
        for start in range(0, len(images), self.max_batch_size):
            cur = images[start: start+self.max_batch_size]
            video_length = cur.shape[1]
            cur = einops.rearrange(cur, "b f c h w -> (b f) c h w")
            resized_videos = torch.nn.functional.interpolate(
                cur, size=self.target_resolution, mode="bilinear",
                align_corners=False)
            resized_videos = einops.rearrange(
                resized_videos, "(b f) c h w -> b c f h w", f=video_length)
            if resized_videos.shape[2] < self.i3d_min:
                if not self.warned:
                    # I3D 要求输入维度大于等于 10，否则无法池化
                    if (
                        not torch.distributed.is_initialized() or
                        torch.distributed.get_rank() == 0
                    ):
                        print(
                            "Warning: Current frame number < {}, and we pad "
                            "it to {}.".format(self.i3d_min, self.i3d_min))

                    self.warned = True

                resized_videos = torch.nn.functional.interpolate(
                    resized_videos, size=target_resolution, mode="trilinear",
                    align_corners=False)

            resized_videos = 2. * resized_videos - 1  # [-1, 1]
            feat = get_logits(self.inception, resized_videos, self.device)
            if real == True:
                self.real_features.append(feat)
            else:
                self.fake_features.append(feat)

    def compute(self):
        real = torch.cat(self.real_features, dim=0)
        fake = torch.cat(self.fake_features, dim=0)
        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        if world_size > 1:
            all_real_features = real.new_zeros(
                (len(real)*world_size, ) + real.shape[1:])
            all_fake_features = fake.new_zeros(
                (len(fake)*world_size, ) + fake.shape[1:])
            torch.distributed.all_gather_into_tensor(
                all_real_features, real)
            torch.distributed.all_gather_into_tensor(
                all_fake_features, fake)
            real, fake = all_real_features, all_fake_features

        self.real_features_num_samples = len(real)
        self.fake_features_num_samples = len(fake)
        return frechet_distance(real, fake)

    def reset(self):
        self.real_features.clear()
        self.fake_features.clear()
        self.warned = False
        super().reset()
