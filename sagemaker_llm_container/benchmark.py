import os
import time
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel, HuggingFacePredictor
from sagemaker.session import Session
import boto3
import json
import argparse
import yaml

from utils.get_metrics import get_metrics_from_cloudwatch


def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--model_id", default=None, type=str)
    parser.add_argument("--iam_role", default="sagemaker_execution_role", type=str)
    parser.add_argument("--tp_degree", type=int, default=1)
    parser.add_argument("--instance_type", type=str)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--vu", type=int, default=1)
    parser.add_argument("--quantize", choices=["gptq"])
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--endpoint_name", type=str, default=None)
    parser.add_argument("--inference_component", type=str, default=None)

    return parser.parse_args()


def run_benchmark(
    iam_role,
    token,
    model_id,
    instance_type,
    tp_degree,
    vu,
    quantize,
    endpoint_name=None,
    inference_component=None,
    batch_size=None,
    sequence_length=None,
):
    start = time.time()
    iam = boto3.client("iam")
    role = iam.get_role(RoleName=iam_role)["Role"]["Arn"]

    print(f"sagemaker role arn: {role}")
    print(f"token: {token[:10] if token else None}")
    print(f"model id: {model_id}")
    print(f"instance type: {instance_type}")
    print(f"tp_degree: {tp_degree}")
    print(f"vu: {vu}")
    print(f"quantize: {quantize}")
    print(f"endpoint_name: {endpoint_name}")
    print(f"inference_component: {inference_component}")

    if endpoint_name:
        llm = HuggingFacePredictor(
            endpoint_name=endpoint_name, component_name=inference_component
        )
    else:
        # TGI config
        config = {
            "HF_MODEL_ID": model_id,  # model_id from hf.co/models
            "HUGGING_FACE_HUB_TOKEN": token,
        }
        if quantize:
            config["HF_MODEL_QUANTIZE"] = "gptq"
        if batch_size is not None:
            # NeuronX model with static batch_size and sequence_length
            llm_image = get_huggingface_llm_image_uri("huggingface-neuronx", version="0.0.20")
            config["MAX_BATCH_SIZE"] = json.dumps(batch_size)
            config["MAX_INPUT_LENGTH"] = json.dumps(sequence_length // 2)
            config["MAX_TOTAL_TOKENS"] = json.dumps(sequence_length)
            config["HF_BATCH_SIZE"] = json.dumps(batch_size)
            config["HF_SEQUENCE_LENGTH"] = json.dumps(sequence_length)
            config["HF_AUTO_CAST_TYPE"] = "fp16"
            config["HF_NUM_CORES"] = json.dumps(tp_degree)
        else:
            config["MAX_INPUT_LENGTH"] = json.dumps(2048),  # Max length of input text
            config["MAX_TOTAL_TOKENS"] = json.dumps(4096),  # Max length of the generation (including input text)
            config["SM_NUM_GPUS"] = json.dumps(tp_degree)  # Number of GPU used per replica
            # retrieve the llm GPU image uri
            llm_image = get_huggingface_llm_image_uri("huggingface", version="1.0.3")
        print(f"Using {llm_image} for deployment")
        # create HuggingFaceModel
        llm_model = HuggingFaceModel(role=role, image_uri=llm_image, env=config)

        # deploy model to endpoint
        try:
            llm = llm_model.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                container_startup_health_check_timeout=3600,
                volume_size=512,
            )
        except Exception as e:
            print(e)
            print(
                f"Failed to deploy model with config {model_id}, {instance_type}, {tp_degree}, {vu}"
            )
        print(f"Succesfully deployed model with config {model_id}, {instance_type}, {tp_degree}, {vu}")

    # get endpoint region and credentials
    endpoint_region = llm.sagemaker_session._region_name
    credentials = llm.sagemaker_session.boto_session.get_credentials()
    # run benchmark
    try:
        # run warm up inference
        for i in range(2):
            llm.predict({
                "inputs": "This is a sample sentence to warm up the model",
                "parameters": {"max_new_tokens": 50, "top_k": 50, "top_p": 0.95, "do_sample": True},
            })

        # run k6 load test
        time.sleep(10)
        benchmark_start_time = time.time()
        command = "k6 run sagemaker_load.js " \
                  f"-e ENDPOINT_NAME={llm.endpoint_name} " \
                  f"-e REGION={endpoint_region} " \
                  f"-e DO_SAMPLE=0 -e VU={vu} " \
                  f"-e AWS_ACCESS_KEY_ID={credentials.access_key} " \
                  f"-e AWS_SECRET_ACCESS_KEY={credentials.secret_key} "

        if inference_component:
            command += f" -e INFERENCE_COMPONENT={inference_component}"

        os.system(command)

        # Read results to get the actual duration, number of requests, failures
        with open("summary.json") as f:
            results = json.load(f)
            checks = results["root_group"]["checks"][0]
            passes = checks["passes"]
            fails = checks["fails"]
            print(f"Requests completed/failed : {passes}/{fails}")
            duration = results["state"]["testRunDurationMs"] / 1000

        # wait for cloudwatch logs to be populated and ready to read
        print("Waiting for cloudwatch logs to be populated")
        time.sleep(120)
        benchmark_end_time = time.time()

        # get cloudwatch logs
        results = get_metrics_from_cloudwatch(
            endpoint_name=llm.endpoint_name,
            st=int(benchmark_start_time),
            et=int(benchmark_end_time),
            instance_type=instance_type,
            tp_degree=tp_degree,
            vu=vu,
            quantize=quantize,
            model_id=model_id,
            inference_component=inference_component,
            duration=duration,
            num_gen_tokens=500,
        )

        # print results
        basename = f"{model_id.split('/')[-1]}_{instance_type}_tp_{int(tp_degree)}_vu_{int(vu)}"
        if batch_size is not None:
            basename = f"{basename}_bs{batch_size}_seqlen{sequence_length}"
        with open(f"{basename}.json", "w") as f:
            f.write(json.dumps(results, indent=2))
    except Exception as e:
        print(e)

    if not endpoint_name and llm is not None:
        # delete endpoint
        print("Deleting endpoint")
        llm.delete_model()
        llm.delete_endpoint()


if __name__ == "__main__":
    args = parse_args()

    if args.config_file:
        with open(args.config_file, "r") as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)["configs"]

        # run each config for each vu group
        vu_group = [1, 5, 10, 20]

        # run each config for each vu group
        print(f"Running benchmark for {len(configs) * 4} configs")
        print(
            f"Expected time between: {len(configs) * 4 * 18}-{len(configs) * 4 * 27} minutes"
        )

        for config in configs:
            vu_group = config.get("vu_group")
            for vu in vu_group:
                batch_size = config.get("batch_size", None)
                sequence_length = config.get("sequence_length", None)
                if batch_size is not None:
                    if batch_size > vu:
                        print(f"Skipping test for vu {vu} which is < batch size {batch_size}")
                        continue
                run_benchmark(
                    iam_role=args.iam_role,
                    token=args.token,
                    model_id=config["model_id"],
                    instance_type=config["instance_type"],
                    tp_degree=config["tp_degree"],
                    vu=vu,
                    quantize="gptq" if "quantize" in config else None,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )

    else:
        del args.config_file
        run_benchmark(**vars(args))
