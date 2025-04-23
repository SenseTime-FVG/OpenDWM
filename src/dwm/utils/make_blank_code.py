import argparse
import dwm.common
import dwm.pipelines.lidar_vqvae
import json
import pickle
import torch
import torch.utils.data


def create_parser():
    parser = argparse.ArgumentParser(
        description="Make blank code file of the LiDAR codebook checkpoint.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The path of training config file.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of input checkpoint file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path of output blank code file.")
    parser.add_argument(
        "-it", "--iteration", default=100, type=int,
        help="The iteration count from the validation set for blank code.")
    parser.add_argument(
        "-s", "--sample-count", default=16, type=int,
        help="The count of blank code to sample.")
    return parser


def count_code(indices):
    unique_elements, counts = torch.unique(indices, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_elements = unique_elements[sorted_indices]
    sorted_counts = counts[sorted_indices]
    return sorted_elements, sorted_counts


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device(config.get("device", "cpu"))
    dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])

    vq_point_cloud = dwm.common.create_instance_from_config(
        config["pipeline"]["vq_point_cloud"])
    vq_point_cloud.to(device)
    vq_point_cloud.load_state_dict(
        dwm.pipelines.lidar_vqvae.LidarCodebook.load_state(args.input_path),
        strict=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        **dwm.common.instantiate_config(config["validation_dataloader"]))

    iteration = 0
    code_dict = {}
    vq_point_cloud.eval()
    for batch in dataloader:
        with torch.no_grad():
            points = dwm.pipelines.lidar_codebook.LidarCodebook.get_points(
                batch, config["pipeline"]["common_config"], device)

            voxels = vq_point_cloud.voxelizer(points)
            lidar_feats = vq_point_cloud.lidar_encoder(voxels)
            _, _, code_indices = vq_point_cloud.vector_quantizer(
                lidar_feats, vq_point_cloud.code_age,
                vq_point_cloud.code_usage)

            codes, counts = count_code(code_indices)
            for code, count in zip(codes.tolist(), counts.tolist()):
                if code in code_dict:
                    code_dict[code] += count
                else:
                    code_dict[code] = count

        iteration += 1
        if iteration >= args.iteration:
            break

    blank_code = [
        i[0] for i in sorted(
            code_dict.items(),
            key=lambda i: i[1], reverse=True)[:args.sample_count]
    ]
    with open(args.output_path, "wb") as f:
        pickle.dump(blank_code, f)
