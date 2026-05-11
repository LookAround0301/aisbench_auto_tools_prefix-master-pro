from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/mnt/share/q00946761/weights/DeepSeek-V2-Lite-Chat",
        model="glm-5",
        request_rate=0,
        retry=2,
        host_ip="141.61.81.154",
        host_port=7060,
        max_out_len=300,
        batch_size=1,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
