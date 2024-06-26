import argparse
import concurrent.futures

from .rags_utils import *
from .rags_eval import *


def parser_args():
    parser = argparse.ArgumentParser(description="rag evaluation")

    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="./kimi_response_demo.json",
        help="待评估的json文件",
    )
    parser.add_argument(
        "--knowledge_base",
        type=str,
        default="test_eval",
        help="知识库目录, 必须为 knowledge_base 下的根目录, 例如 samples",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="评估结果输出路径, 会输出两个，1个csv、1个json",
    )
    parser.add_argument(
        "--emb_model_path",
        type=str,
        help="使用本地 Embedding, 不传则使用 OpenAI Embedding",
    )
    parser.add_argument(
        "--generate_model_path",
        type=str,
        help="使用本地 LLM, 不传则使用 OpenAI gpt-3.5-turbo-16k",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu")

    parser.add_argument(
        "--kb_top_k",
        type=int,
        default="5",
        help="检索知识库的条数")

    parser.add_argument(
        "--score_threshold",
        type=float,
        default="0.8",
        help="检索阈值, 一般取值范围在0-1, 距离越小相关度越高", )

    parser.add_argument(
        "--is_print",
        action="store_true",
        help="打印")

    parser.add_argument(
        "--max_workers",
        type=int,
        help="是否并行执行")

    return parser.parse_args()


def start_process(args):
    print(colored("[DEBUG] ========== start process ==========", "light_cyan"))
    sss = time.time()

    if args.emb_model_path:
        EMB_MODEL = SentenceTransformer(args.emb_model_path, device=args.device)

    if args.generate_model_path:
        # padding_side="left",
        TOKENIZER = AutoTokenizer.from_pretrained(args.generate_model_path, trust_remote_code=True)
        # TOKENIZER.pad_token = TOKENIZER.eos_token
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(args.generate_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    else:
        # 使用 OpenAI 接口
        TOKENIZER, LLM_MODEL = None, None

    debug_print(content="模型加载完成", color="light_cyan")

    with open(args.eval_data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    debug_print(label="数据加载完成", content=len(data), color="light_cyan")

    count_source = {
        "total_count": len(data),
        "succ_count": 0,
        "err_count": 0,
        "avg_result_score": [],
        "avg_faithfulness": [],
        "avg_cr_q": [],
        "avg_cr_gt": [],
        "avg_ar": [],
        "avg_ac": [],
        "avg_ai": [],
    }

    results = []
    os.makedirs(args.output_path, exist_ok=True)

    if args.max_workers:  # TODO: 多进程在检索知识库的时候，chatchat里会有多线程，会卡主   （!!! 进程池待更新 !!!）
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_item = {executor.submit(process_eval, item, is_print=args.is_print): item for item in data}
            debug_print("len future_to_item", len(future_to_item), color="green")

            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item), desc="Parallel Computing Evaluation Metrics", unit="item",):
                item = future_to_item[future]
                try:
                    item_res = future.result()

                    if not item_res:
                        with lock:
                            count_source["err_count"] += 1
                    else:
                        results.append(item_res)
                        with lock:
                            update_count_source(item_res, count_source)  # 可变-引用传递

                except Exception as e:
                    with lock:
                        count_source["err_count"] += 1
                    exc_traceback = traceback.format_exc()
                    print(
                        f"{item['file_name']} generated an exception: {e}\n{exc_traceback}"
                    )
    else:
        for item in tqdm(data, desc="Calculating evaluation metrics", unit="item"):
            try:
                item_res = process_eval(args, item, tokenizer=TOKENIZER, llm_model=LLM_MODEL, emb_model=EMB_MODEL)

                # item_res 为空，跳过
                if not item_res:
                    count_source["err_count"] += 1
                else:
                    results.append(item_res)
                    update_count_source(item_res, count_source)
                    stream_save_csv(item_res, save_path=args.output_path)   # 流式写入

            except Exception as e:
                count_source["err_count"] += 1
                exc_traceback = traceback.format_exc()
                print(f"[DEBUG] {item['file_name']} generated an exception: {e}\n{exc_traceback}")
                print("[DEBUG] item_res: ", type(item_res), len(item_res))
                # raise Exception(exc_traceback)

    save_csv(results, save_path=args.output_path)
    debug_print("count_source", save_json(count_source, save_path=args.output_path), "green")

    print(colored("[DEBUG] ========== process end {} ==========".format(time.time() - sss), "light_cyan",))


if __name__ == '__main__':
    """
    # data = "./kimi_response_for_51docs_on_13questions_20240523.json"

    cd rag_evaluation

    CUDA_VISIBLE_DEVICES=1,2,3 python -m rag_evaluation \
                                    --eval_data_path /home/wangguisen/projects/Langchain-Chatchat/kimi_response_demo2.json \
                                    --knowledge_base test_eval \
                                    --output_path ./rag_evaluation/output \
                                    --emb_model_path /home/wangguisen/models/bge-large-zh-v1.5 \
                                    --generate_model_path /home/wangguisen/models/Qwen1.5-14B-Chat \
                                    --device cuda \
                                    --kb_top_k 5 \
                                    --score_threshold 0.8

    CUDA_VISIBLE_DEVICES=1,2,3 python -m rag_evaluation \
                                    --eval_data_path /home/wangguisen/projects/Langchain-Chatchat/kimi_response_demo2.json \
                                    --knowledge_base test_eval \
                                    --output_path ./rag_evaluation/output \
                                    --emb_model_path /home/wangguisen/models/bge-large-zh-v1.5 \
                                    --device cuda \
                                    --kb_top_k 5 \
                                    --score_threshold 0.8

    CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m rag_evaluation \
    --eval_data_path /home/wangguisen/projects/Langchain-Chatchat/kimi_response_for_51docs_on_13questions_20240523.json \
    --knowledge_base test_eval \
    --output_path ./rag_evaluation/output \
    --emb_model_path /home/wangguisen/models/bge-large-zh-v1.5 \
    --generate_model_path /home/wangguisen/models/Qwen1.5-14B-Chat \
    --device cuda \
    --kb_top_k 5 \
    --score_threshold 0.8 \
    > ./rag_evaluation/output/run.log 2>&1 &



    """

    # data_process()

    # 初始化知识库
    # init_data_base()

    args = parser_args()

    debug_print("args", args)

    start_process(args)


