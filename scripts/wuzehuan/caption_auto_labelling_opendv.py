import argparse
import base64
import dwm.common
import io
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
        default="http://10.119.27.60:6550/service",
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
        file_path, t, fps = dataset.items[i]
        token = "{}.{:.0f}".format(file_path, t * 1000)
        item = dataset.read_item(file_path, t, fps)

        with io.BytesIO() as f:
            item["images"][0].save(f, format="jpeg")
            image_data = f.getvalue()

        answers = {}
        for k in config["prompts"]:
            response = requests.post(
                args.endpoint_url,
                data={
                    "question": k["question"],
                    "image": base64.b64encode(image_data).decode("utf-8"),
                }).json()

            if k["key"] is None:
                json_answer_search = re.search(
                    json_answer_pattern, response["answer"])
                if json_answer_search is None:
                    answers["time"] = answers["weather"] = \
                        answers["environment"] = answers["objects"] = ""
                    print("Warning: no JSON answer of {}".format(token))
                else:
                    try:
                        json_answer = json_answer_search.group("content")
                        for rk, rv in json.loads(json_answer).items():
                            answers[rk] = rv
                    except:
                        answers["time"] = answers["weather"] = \
                            answers["environment"] = answers["objects"] = ""
                        print("Warning: wrong JSON format of {}".format(token))

            else:
                answers[k["key"]] = response["answer"]

        result[token] = answers

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, **config["dump_options"])
