"""
用于 RAG 的评估
"""
from dotenv import load_dotenv
load_dotenv()

import os
import json
from multiprocessing import Lock, Manager
import numpy as np
from typing import List, Union, Dict, Any, Tuple
import ast
from retry import retry
import traceback
import logging
import warnings

import torch
from openai import OpenAI, BadRequestError, APITimeoutError
from jinja2 import Template
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pynlpir
from argparse import ArgumentParser

from configs.prompt_config import PROMPT_TEMPLATES
from server.knowledge_base.kb_doc_api import search_docs
from configs import logger

from .rags_utils import (debug_print,
                         name_modify,
                         PUNCTUATION_LST,
                         fix_json,
                         cosine_similarity,
                         cut_sent,
                         fix_lst,
                         PynlpirContextManager,)


# 忽略调用 OpenAI 的log
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# 忽略 langchain 的 INFO
logger.setLevel(logging.WARNING)

warnings.filterwarnings("ignore")


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE_URL"], timeout=30)

lock = Lock()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_print("device", device)


def openai_response(prompt: str) -> str:
    return client.chat.completions.create(
            model=os.environ["OPENAI_MODEL"],
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content


# def local_llm_response(prompt: str,
#                        generate_model_path: str,
#                        tokenizer: AutoTokenizer,
#                        llm_model: AutoModelForCausalLM
#                        ) -> str:

#     model_name = generate_model_path.split("/")[-1]

#     if "chat" in model_name or "Chat" in model_name:

#         if "Qwen1.5-14B" in model_name:
#             messages = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]
#             text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)

#             generated_ids = llm_model.generate(model_inputs.input_ids, max_new_tokens=512)

#             generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
#             response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         else:
#             response, history = llm_model.chat(tokenizer, prompt, history=None)

#     else:
#         inputs = tokenizer(prompt, return_tensors='pt')
#         inputs = inputs.to(llm_model.device)
#         # pred = llm_model.generate(**inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
#         pred = llm_model.generate(**inputs, max_new_tokens=512)
#         response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

#     return response


def local_llm_response(prompt: str,
                       tokenizer: AutoTokenizer,
                       llm_model: AutoModelForCausalLM
                       ) -> str:

    messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    inputs = inputs.to(llm_model.device)

    # "max_length": 1024, "top_k": 50, "top_p": 0.95, "num_return_sequences": 1
    gen_kwargs = {"do_sample": True, "max_new_tokens": 512}
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def get_text_embedding_openai(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(input=[text], model=os.environ["OPENAI_EMEBEDDING_MODEL"])
    return np.array(response.data[0].embedding)


def get_text_embedding(emb_model: Any, text: str | np.ndarray) -> np.ndarray:
    """https://huggingface.co/BAAI/bge-large-zh-v1.5
    BGE模型的相似度分布大约在[0.6,1]。因此，相似度得分大于 0.5 并不表明两个句子相似。
    """
    # with lock:
    if isinstance(emb_model, SentenceTransformer):
        return emb_model.encode(
            text, show_progress_bar=False, normalize_embeddings=False
        )
    else:
        raise ValueError("TODO: Embedding type")


def my_search_docs(query, knowledge_base_name, kb_top_k, score_threshold, rag_eval_lock=lock):
    pass


def process_context(query: str,
                    knowledge_base_name: str,
                    kb_top_k: int,
                    score_threshold: float
                    ) -> Tuple[str, List[str], List[float]]:
    """检索上下文

    Args:
        query (str): query
        knowledge_base_name (str): 知识库名字
        kb_top_k (int): 检索top k
        score_threshold (float): 检索阈值, 一般取值范围在0-1, 距离越小相关度越高。

    Returns:
        Tuple[str, List[str], List[float]]: 检索到的上下文(str)、检索到的上下文-未拼接字符串、相似分数
    """

    docs = search_docs(query, knowledge_base_name, kb_top_k, score_threshold)  # TODO
    # docs = search_docs(query, knowledge_base_name, kb_top_k, score_threshold, rag_eval_lock=lock)   # TODO 多进程会卡住

    context_pages = [doc.page_content for doc in docs]
    context = "\n".join(context_pages)
    scores = [doc.score for doc in docs]

    return context, context_pages, scores


def process_llm_response(query: str,
                         relevant_context: str = None,
                         generate_model_path: str = None,
                         tokenizer: AutoTokenizer = None,
                         llm_model: AutoModelForCausalLM = None,
                         ) -> str:
    """获取大模型的回复
    """
    prompt = query

    if relevant_context:
        prompt = Template(
            PROMPT_TEMPLATES["knowledge_base_chat"]["default"]   # TODO
        ).render(context=relevant_context, question=query)

    if generate_model_path:
        response = local_llm_response(prompt, tokenizer, llm_model)
    else:
        response = openai_response(prompt)

    return response


@retry(exceptions=(BadRequestError, APITimeoutError, json.decoder.JSONDecodeError), tries=2)
def faithfulness_func(
    query: str,
    answer: str,
    relevant_context: str,
    generate_model_path: str = None,
    tokenizer: AutoTokenizer = None,
    llm_model: AutoModelForCausalLM = None,
) -> Tuple[float, Dict[str, str], Dict[str, int]]:
    """忠实度
    衡量生成的答案是否基于给定的上下文

    F = |V| / |S|

    Returns:
        Tuple[float, Dict[str, str], Dict[str, int]]: 忠实度分数、陈述、陈述的置信度
    """
    try:

        prompt = f"""
        给定一个问题和答案，请你从给定答案的每个句子中创建一个或多个不重复的陈述。

        问题：{query}
        答案：{answer}

        结果请以JSON的形式返回：{{"陈述1"：陈述1, "陈述2"：陈述2, ...}}，请严格按照指定格式返回。
        """
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)
        response = fix_json(response)
        debug_response = response

        statements = json.loads(response)

        prompt = f"""
        你的任务是根据给定的上下文判断以下陈述的可信度。具体要求如下：
        1. 对于每个陈述，请你判断它是否可以从给定的上下文信息中推断。
        2. 如果上下文支持该陈述，请返回 1；如果不支持该陈述，请返回 0。
        3. 请以JSON形式返回结果：{{"陈述1"：可信度1或0, "陈述2"：可信度1或0, ...}}，请严格按照指定格式返回，返回结果不要偏离指定格式。

        上下文：{relevant_context}
        陈述：{statements}
        """
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)
        response = fix_json(response)

        response_flag = json.loads(response)

        sum_lst = []
        for v in response_flag.values():
            if isinstance(v, str):
                if "1" in v:
                    v = 1
                else:
                    v = 0
            sum_lst.append(v)

        faithfulness = sum(sum_lst) / len(statements)   # 如果有上下文但没有陈述也走异常
    except Exception as e:
        faithfulness, statements, response_flag = 0, {}, {}
        print("[DEBUG] faithfulness_func exception: ", e)
        print("[DEBUG] response1: \n {}".format(debug_response))
        print("[DEBUG] response2: \n {}".format(response))
        raise e

    return faithfulness, statements, response_flag


@retry(exceptions=(BadRequestError, APITimeoutError, json.decoder.JSONDecodeError), tries=2)
def answer_relevance_func(
    query: str,
    answer: str,
    generate_model_path: str = None,
    tokenizer: AutoTokenizer = None,
    llm_model: AutoModelForCausalLM = None,
    emb_model: Any = None
) -> Tuple[float, List[str], np.ndarray]:
    """答案相关性
    衡量生成的答案与问题之间的相关性

    AR = \frac{1}{n} \sum^n_{i=1} sim(q, q_i)

    Returns:
        Tuple[float, List[str], np.ndarray]: 答案相关性分数、生成的答案、相似分数
    """

    prompt = f"""
    请你为给定的答案生成一个或多个不重复的问题。

    答案：{answer}

    结果请以JSON形式返回：{{"生成问题1": 问题1, "生成问题2": 问题2, ...}}，请严格按照指定格式返回。
    """
    try:
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)
        response = fix_json(response)

        gen_questions = json.loads(response)

        gen_questions = list(gen_questions.values())

        if emb_model:
            sim_sum = np.array([cosine_similarity(get_text_embedding(emb_model, query), get_text_embedding(emb_model, q)) for q in gen_questions])
        else:
            sim_sum = np.array([cosine_similarity(get_text_embedding_openai(client, query), get_text_embedding_openai(client, q))for q in gen_questions])

        ar = np.average(sim_sum)

    except Exception as e:
        ar, gen_questions, sim_sum = 0., [], np.array([])
        print("[DEBUG] answer_relevance_func exception: ", e)
        print("[DEBUG] response: \n {}".format(response))
        raise e

    return ar, gen_questions, sim_sum


@retry(exceptions=(BadRequestError, APITimeoutError, json.decoder.JSONDecodeError), tries=2)
def context_relevance_func(
    query: str,
    relevant_context: str,
    relevant_context_pages: List[str],
    generate_model_path: str = None,
    tokenizer: AutoTokenizer = None,
    llm_model: AutoModelForCausalLM = None,
) -> Tuple[float, float, float, List[str]]:
    """上下文相关性
    衡量检索出来的上下文与问题之间的相关性。理想情况下, 检索到的上下文c(q)应该仅包含回答问题所需要的相关信息，所以通过这个指标来衡量或者说是惩罚包含冗余信息的情况。
    以句子和片段两个维度进行计算。

    $CR_{q1} = \frac{|E_s|}{|C_s|}$
    $CR_{q2} = \frac{|E_f|}{|C_k|}$
    $CR_q = avg(CR_{q1}, CR_{q2})$

    Returns:
        Tuple[float, float, float, List[str]]: 上下文相关性分数、句子维度、片段维度、相关句子
    """

    prompt = f"""
    请你从提供的上下文中提取所有可能有助于回答以下问题的相关句子。具体要求如下：
    1.提取所有不重复的相关句子，且不修改上下文中的任何句子。
    2.请勿直接回答问题，仅提取有助于回答问题的相关句子。
    3.结果请以Python列表形式返回，要求如下：
        - 如果找不到相关句子，或者你认为无法从给定的上下文中回答该问题，请返回: ["信息不足"]
        - 如果提取到相关句子，请返回: [相关句子1, 相关句子2, ...]
    4.请严格按照指定格式返回结果。

    上下文：{relevant_context}
    问题：{query}
    """
    # 去除里面的所有换行，避免影响计算
    relevant_context_pages = [e.replace("\n", "") for e in relevant_context_pages]
    recon_lst = cut_sent("\n".join(relevant_context_pages))  # 分句
    recon_lst = [rlc for rlc in recon_lst if rlc != ""]  # 忽略空串

    try:
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)

        response = fix_lst(response)
        response = ast.literal_eval(response)

        if "信息不足" in response[0] and len(response) == 1:
            return 0., 0., 0., response
        else:
            cr_q1 = len(response) / len(recon_lst)
            e_f = 0
            for e in relevant_context_pages:
                for sub in response:
                    sub = sub.replace("\n", "")  # 去除里面的所有换行，避免影响计算
                    if sub in e:
                        e_f += 1
                        continue
            cr_q2 = e_f / len(relevant_context_pages)

            # TODO: 文档是英文的，这里会存在问题，LLM返回的相关句子数多于上下文数
            if cr_q1 > 1 or cr_q2 > 1:
                print("[DEBUG] len(response): {}, len(recon_lst): {}".format(len(response), len(recon_lst)))
                print("[DEBUG] recon_lst: \n {}".format(recon_lst))
                print("[DEBUG] e_f: {}, len(relevant_context_pages): {}".format(e_f, len(relevant_context_pages)))
                raise ValueError("[DEBUG] cr_q1, cr_q2 分子太大: {}、{}".format(cr_q1, cr_q2))

            cr_q = (cr_q1 + cr_q2) / 2

    except Exception as e:
        cr_q, cr_q1, cr_q2, response = 0., 0., 0., response
        print("[DEBUG] context_relevance_func exception: ", e)
        print("[DEBUG] response: \n {}".format(response))
        raise e

    return cr_q, cr_q1, cr_q2, response


@retry(exceptions=(BadRequestError, APITimeoutError, json.decoder.JSONDecodeError), tries=2)
def context_recall_func(
    relevant_context: str,
    gt: str,
    generate_model_path: str = None,
    tokenizer: AutoTokenizer = None,
    llm_model: AutoModelForCausalLM = None,
) -> Tuple[float, List[str], Dict[str, int]]:
    """上下文召回率
    衡量检索到的上下文与 ground truth 之间的一致程度。
    理论上，ground truth 中的所有的句子都应该归因于检索到的上下文，所以判断 ground truth 中的每个句子是否可以由上下文所推断，就可以计算出上下文对于 ground truth 的召回程度。

    CR_{gt} = \frac{|A_s|}{|G_s|}

    Returns:
        Tuple[float, List[str], Dict[str, int]]: 上下文召回率、陈述、陈述置信度
    """
    try:
        prompt = f"""
        请你根据给定的真实答案，将其分解成1条或多条不重复的陈述。

        真实答案：{gt}

        请以JSON形式返回结果：{{"陈述1": 陈述1, "陈述2": 陈述2,...}}。
        """
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)
        response = fix_json(response)
        debug_response = response

        statements = list(json.loads(response).values())

        prompt = f"""
        请你根据给定的上下文和一些陈述，分析每条陈述，判断该陈述是否可以由上下文验证。具体要求如下：
        1. 如果该陈述可以由上下文验证，请返回 1，否则返回 0。
        2. 判断过程中，请勿修改陈述和上下文的内容。
        3. 结果以 JSON 形式返回：{{"陈述1": 1/0, "陈述2": 1/0, ...}}。
        4. 仅分析每条陈述，结果中不包含上下文的信息。

        上下文：{relevant_context}
        陈述：{statements}
        """
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)
        response = fix_json(response)

        response_flag = json.loads(response)
        cr_gt = sum(response_flag.values()) / len(statements)

    except Exception as e:
        cr_gt, statements, response_flag = 0., [], {}
        print("[DEBUG] context_recall_func exception: ", e)
        print("[DEBUG] response1: \n {}".format(debug_response))
        print("[DEBUG] response2: \n {}".format(response))
        raise e

    return cr_gt, statements, response_flag


@retry(exceptions=(BadRequestError, APITimeoutError, json.decoder.JSONDecodeError), tries=2)
def answer_correctness_func(
    answer: str,
    gt: str,
    ac_weight=[0.5, 0.5],
    generate_model_path: str = None,
    tokenizer: AutoTokenizer = None,
    llm_model: AutoModelForCausalLM = None,
    emb_model=None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """答案正确性
    衡量生成答案的准确性: 事实正确性 + 语义相似性

    AC_1 = F1 = \frac{|TP|}{|TP| + 0.5 \times (|FP| + |FN|)}
    AC_2 = cos\_sim(g_{emb}, a_{emb})
    AC = \alpha \times AC_1 + \beta \times AC_2, \ \ \alpha+\beta = 1

    Args:
        ac_weight (List[float]): 事实正确性与语义相似性的权重

    Returns:
        Tuple[float, float, float, Dict[str, Any]]: 答案正确性分数、事实正确性分数、语义相似性分数、混淆矩阵结果
    """

    # TODO
    prompt = f"""
    给你一个真实答案和一个回答，请你分析回答中的每个陈述，并将其分类到以下类别之一：
    - TP（真正例）：在回答和真实答案中都存在的陈述。
    - FP（假正例）：在回答中存在但在真实答案中未找到的陈述。
    - FN（假负例）：在真实答案中存在但在回答中遗漏的相关陈述。

    需要强调的是：
    1. 首先你需要根据给定的回答，将其分解成1条或多条不重复的陈述，请勿改动回答的内容。
    2. 然后分析每条陈述，将该陈述分类到上述所提到的三个类别当中。
    3. 每个陈述必须准确地分类到一个类别中，不要试图解释真实答案和回答的含义，只需要比较它们中陈述的存在情况。
    4. 在你的返回结果中，请勿改动回答以及真实答案的内容。

    真实答案：{gt}
    回答：{answer}

    请以JSON形式返回，JSON的键为上述的三个类别，值为符合类别的列表，请严格按照指定格式返回结果，格式如下所示：
    {{"TP": [符合TP的陈述], "FP": [符合FP的陈述], "FN": [符合FN陈述]}}
    """
    try:
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)
        response = fix_json(response)

        response = json.loads(response)

        # for k, v in response.items():
        #     print("[DEBUG] ", k, v, len(v[-1]))

        tp, fp, fn = len(response["TP"]), len(response["FP"]), len(response["FN"])
        try:
            ac_1 = tp / (tp + 0.5 * (fp + fn))
        except ZeroDivisionError:
            ac_1 = 0

        if emb_model:
            ac_2 = cosine_similarity(get_text_embedding(emb_model, gt), get_text_embedding(emb_model, answer))
        else:
            ac_2 = cosine_similarity(get_text_embedding_openai(client, gt), get_text_embedding_openai(client, answer))

        ac = np.average([ac_1, ac_2], weights=ac_weight)

    except Exception as e:
        ac, ac_1, ac_2, response = 0., 0., 0., response
        print("[DEBUG] answer_correctness_func exception: ", e)
        print("[DEBUG] response: \n {}".format(response))
        raise e

    return ac, ac_1, ac_2, response


@retry(exceptions=(BadRequestError, APITimeoutError, json.decoder.JSONDecodeError), tries=2)
def content_integrity_func(
    answer: str,
    gt: str,
    generate_model_path: str = None,
    tokenizer: AutoTokenizer = None,
    llm_model: AutoModelForCausalLM = None,
) -> Tuple[float, float, float, List[str], List[str], List[str], List[str], List[str], List[str]]:
    """内容完整性
    以 ground truth 为依据，衡量生成答案的内容是否完整，或者说衡量生成答案的详细程度: 关键词覆盖率 + 信息密度

    coverage = \frac{|I_s|}{|K_S|}
    density = (\frac{|answer\_sentences|}{|gt\_sentences|} + \frac{|answer\_words|}{|gt\_words|} + \frac{|answer\_words ∩ gt\_words|}{|answer\_words ∪ gt\_words|}) / 3
    AI = (coverage + density) / 2

    Returns:
        Tuple[float, float, float, List[str], List[str], List[str], List[str], List[str], List[str]]: 内容完整性分数、关键词覆盖率、信息密度、关键词、匹配的关键词、answer分句、gt分句、answer分词、gt分词
    """

    prompt = f"""
    请分析以下段落，并提取出涵盖其主要内容的所有关键词。具体要求如下：
    1. 关键词应全面覆盖段落的主要信息。
    2. 关键词应不重复，并尽可能简洁明确。
    3. 请勿包含标点符号或多余的修饰词。

    段落：{gt}

    结果请以Python列表的形式返回，列表中的每个元素为你提取的关键词，关键词为字符串类型，格式为：[关键词1, 关键词2, ...]
    """

    try:
        response = process_llm_response(prompt, generate_model_path=generate_model_path, tokenizer=tokenizer, llm_model=llm_model)
        response = fix_lst(response)

        keywards = ast.literal_eval(response)

        # 计算关键词覆盖率
        matched_keywards = [k for k in keywards if k in answer]
        coverage = len(matched_keywards) / len(keywards)

        # 计算信息密度: 句密度、词密度、词召回
        answer_sentences = cut_sent(answer)
        gt_sentences = cut_sent(gt)

        with PynlpirContextManager():
            answer_words = [s for s in pynlpir.segment(answer.strip().replace("\n", "。"), pos_tagging=False) if s not in PUNCTUATION_LST]
            gt_words = [s for s in pynlpir.segment(gt.strip().replace("\n", "。"), pos_tagging=False) if s not in PUNCTUATION_LST]

        # info_density_recal = len(set(answer_words) & set(gt_words)) / len(set(answer_words) | set(gt_words))
        density = (len(answer_sentences) / len(gt_sentences) + len(answer_words) / len(gt_words)) / 2

        ai = (coverage + density) / 2

    except Exception as e:
        ai, coverage, density, keywards, matched_keywards, answer_sentences, gt_sentences, answer_words, gt_words = 0, 0, 0, [], [], [], [], [], []
        exc_traceback = traceback.format_exc()
        print("[DEBUG] content_integrity_func exception: ", exc_traceback)
        print("[DEBUG] response: \n {}".format(response))
        raise e

    return ai, coverage, density, keywards, matched_keywards, answer_sentences, gt_sentences, answer_words, gt_words


def process_eval(args: ArgumentParser,
                 item: Dict[str, Any],
                 tokenizer: AutoTokenizer,
                 llm_model: AutoModelForCausalLM,
                 emb_model: Any,
                 ) -> Dict[str, Any]:
    """并行计算指标

    Returns:
        Dict[str, Any]: result, 其中 flag (1为正常、0为无gt 不参与计算、-1为召回上下文为空)
    """

    file_name = name_modify(item["file_name"])
    query = item["query"]
    kimi_response = item["kimi_response"]
    remark = item["remark"]

    # 如果数据有错误，则不计算
    flag = 0
    if remark:
        return {}

    # 检索上下文
    context, context_pages, _ = process_context(query,
                                                "{}/{}".format(args.knowledge_base, file_name),
                                                kb_top_k=args.kb_top_k,
                                                score_threshold=args.score_threshold)  # TODO

    # LLM generate
    answer = process_llm_response(query,
                                  relevant_context=context,
                                  generate_model_path=args.generate_model_path,
                                  tokenizer=tokenizer,
                                  llm_model=llm_model)  # TODO

    args.is_print and (debug_print("file_name", file_name, color="green"),
        debug_print("query", query, color="green"),
        debug_print("kimi_response", kimi_response, color="green"),
        debug_print("context", context, color="green"),
        debug_print("answer", answer, color="green"),
    )

    # 检索不到上下文
    if context is None or len(context) == 0:
        flag = -1

        faithfulness, faithfulness_statements, faithfulness_flag = 0.0, {}, {}
        cr_q, cr_q1, cr_q2, context_relevance_response = 0.0, 0.0, 0.0, []
        cr_gt, context_recall_statements, context_recall_flag = 0.0, [], {}
    else:
        flag = 1

        # 计算忠实度
        faithfulness, faithfulness_statements, faithfulness_flag = faithfulness_func(query, answer, context, args.generate_model_path, tokenizer, llm_model)

        # 计算上下文相关性
        cr_q, cr_q1, cr_q2, context_relevance_response = context_relevance_func(query, context, context_pages, args.generate_model_path, tokenizer, llm_model)

        # 计算上下文召回率
        cr_gt, context_recall_statements, context_recall_flag = context_recall_func(context, kimi_response, args.generate_model_path, tokenizer, llm_model)

    # 计算答案相关性
    ar, ar_gen_questions, ar_sim_sum = answer_relevance_func(query, answer, args.generate_model_path, tokenizer, llm_model, emb_model)

    # 计算答案正确性
    ac, ac_1, ac_2, answer_correctness_response = answer_correctness_func(answer, kimi_response, [0.4, 0.6], args.generate_model_path, tokenizer, llm_model, emb_model)

    # 计算答案完整性
    ai, coverage, density, keywards, matched_keywards, answer_sentences, gt_sentences, answer_words, gt_words = content_integrity_func(answer, kimi_response, args.generate_model_path, tokenizer, llm_model)

    result_score = faithfulness + ar + cr_q + cr_gt + ac + ai

    args.is_print and (debug_print("faithfulness_statements", faithfulness_statements),
        debug_print("faithfulness_flag", faithfulness_flag),
        debug_print("faithfulness", faithfulness),
        debug_print("ar_gen_questions", ar_gen_questions),
        debug_print("ar_sim_sum", ar_sim_sum),
        debug_print("ar", ar),
        debug_print("context_relevance_response", context_relevance_response),
        debug_print("cr_q", (cr_q, cr_q1, cr_q2)),
        debug_print("context_recall_statements", context_recall_statements),
        debug_print("context_recall_flag", context_recall_flag),
        debug_print("cr_gt", cr_gt),
        debug_print("answer_correctness_response", answer_correctness_response),
        debug_print("ac", (ac, ac_1, ac_2)),
        debug_print("keywards", keywards),
        debug_print("matched_keywards", matched_keywards),
        debug_print("answer_sentences", answer_sentences),
        debug_print("gt_sentences", gt_sentences),
        debug_print("answer_words", answer_words),
        debug_print("gt_words", gt_words),
        debug_print("ai", (ai, coverage, density)),
        debug_print("result_score", result_score, color="green"),
    )

    return {
        "flag": flag,
        "file_name": file_name,
        "query": query,
        "context": context,
        "answer": answer,
        "kimi_response": kimi_response,
        "result_score": result_score,
        "faithfulness": [faithfulness, faithfulness_statements, faithfulness_flag],
        "context_relevance": [cr_q, cr_q1, cr_q2, context_relevance_response],
        "context_recall": [cr_gt, context_recall_statements, context_recall_flag],
        "answer_relevance": [ar, ar_gen_questions, ar_sim_sum],
        "answer_correctness": [ac, ac_1, ac_2, answer_correctness_response],
        "content_integrity": [ai, coverage, density, keywards, matched_keywards, answer_sentences, gt_sentences, answer_words, gt_words]
    }

