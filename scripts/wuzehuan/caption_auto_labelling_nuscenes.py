import argparse
import base64
import dwm.common
import json
import re
import requests
import tqdm


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to auto labels the caption with VQA APIs.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The dataset config.")
    parser.add_argument(
        "-e", "--endpoint-url", type=str,
        default="http://112.111.7.64:10067/cabin_intern/service",
        help="The URL of the VQA API.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The output path to save the dict of image ID and labelled "
        "caption.")
    parser.add_argument("-f", "--range-from", type=int, default=0)
    parser.add_argument("-t", "--range-to", type=int, default=-1)
    return parser


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    json_answer_pattern = re.compile("```json\n(?P<content>(.|\n)*)```")
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset = dwm.common.create_instance_from_config(config["dataset"])
    range_to = len(dataset) if args.range_to == -1 else args.range_to
    print(
        "Dataset count: {}, processing range {} - {}".format(
            len(dataset), args.range_from, range_to))

    result = {}
    for i in tqdm.tqdm(range(args.range_from, range_to)):
        item = dataset.items[i]

        flatten_segments = [
            k
            for j in item["segment"]
            for k in j
        ]

        for j in flatten_segments:
            image_data = dataset.fs.cat_file(j["filename"])
            answers = {}
            for k in config["prompts"]:
                response = requests.post(
                    args.endpoint_url,
                    data={
                        "question": k["question"],
                        "image": base64.b64encode(image_data).decode("utf-8"),
                    }).json()

                if k["key"] is None:
                    json_answer = re\
                        .search(json_answer_pattern, response["answer"])\
                        .group("content")
                    for rk, rv in json.loads(json_answer).items():
                        answers[rk] = rv
                else:
                    answers[k["key"]] = response["answer"]

            result[j["token"]] = answers

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
