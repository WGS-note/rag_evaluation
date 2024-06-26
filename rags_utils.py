"""
rags utlis
"""
import os
import time
import re
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple
import numpy as np
import csv
import json
from tqdm import tqdm

import pynlpir
from termcolor import colored


PUNCTUATION_LST = " 。？！，、；：“”‘’『』（）［］〔〕〈〉《》「」【】……——-—―·.?!,;:'\"()[]{}<>@#$%^&*_+-=~`|\\/"


class PynlpirContextManager:
    def __enter__(self):
        pynlpir.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pynlpir.close()


def data_process():
    """数据格式转换，存入 ./knowledge_base"""

    count = 0
    obj = "./knowledge_base/test_eval"
    path_root = Path("./PDF测试集")

    for path in path_root.iterdir():
        if not path.is_dir():
            continue

        for sou_path in path.iterdir():
            sou_name = sou_path.name.split(".pdf")[0]

            """
            2标准合同/新能源场站并网调度协议示范文本_GF-2021-0513
            7教育培训/第一单元作文导写_2023—2024学年统编版高一语文必修下册
            10小说/未尽的七子之歌_中国海外流失文物回归
            """
            # if " " in sou_name:
            #     sou_name = sou_name.replace("  ", "_").replace(" ", "_")
            #     sou_path = Path(os.path.join(sou_path.parent, sou_name))

            os.system("mkdir {}".format(os.path.join(obj, sou_name)))
            os.system("mkdir {}".format(os.path.join(obj, sou_name, "content")))
            os.system(
                "cp {} {}".format(sou_path, os.path.join(obj, sou_name, "content"))
            )
            count += 1

    print("[DEBUG] count: ", count)


def init_data_base():
    """初始化知识库"""

    sss = time.time()
    count = 0
    print("[DEBUG] 开始初始测试化知识库")

    command = ("CUDA_VISIBLE_DEVICES=3 python init_database.py --recreate-vs --kb-name={}")

    for path in Path("./knowledge_base/test_eval").iterdir():

        os.system(command.format(os.path.join("test_eval", path.name)))

        count += 1
        print("[DEBUG] {} 初始化完成".format(path.name))

    print("[DEBUG] 初始化知识库完成 {} {}".format(time.time() - sss, count))


def name_modify(name: str) -> str:
    wait = ["新能源场站并网调度协议示范文本", "第一单元作文导写", "未尽的七子之歌"]
    if any(ws in name for ws in wait):
        name = name.replace("  ", "_").replace(" ", "_")

    return name.split(".pdf")[0]


def debug_print(label: str="", content: Any="", color=None):
    print(colored(f"[DEBUG] {label}: {content}", color=color))


def fix_json(response: str):

    response = response.strip().replace("'", '"').replace("，", ",")

    def fix_json_sub(response):
        # response = response.replace("```json", "").replace("```", "")
        response = re.findall(r'```json(.*)```', response, flags=re.DOTALL)[0]
        return response

    if response.startswith("```json") and response.endswith("```"):
        response = fix_json_sub(response)

    # if response.startswith("```json") and "```" in response and "解释" in response:
    #     dice_idx = response.find("解")
    #     response = response[:dice_idx].rstrip()
    #     response = fix_json_sub(response)

    elif response.startswith("```json") and "```" in response:
        sidx = response.find("```json") + 7
        eidx = response.find("}\n```") + 1
        response = response[sidx:eidx]

    if "..." in response:
        response = response.replace("...", "")

    # 处理多余引号: "xx'qq'xx"
    response = remove_extra_quotation_marks_dict(response)

    # 去掉尾部逗号
    response = re.sub(r',\s*}', '}', response)
    response = re.sub(r',\s*]\s*}', ']}', response)

    # 去掉注释  // 匹配注释的开始，.* 匹配从 // 到行尾的所有字符。
    response = re.sub(r'//.*', '', response)

    return response


def fix_string(input_str):
    fix_str = ""
    str_lst = input_str.split(",")
    for tmp in str_lst:
        tmp = tmp.strip()
        if tmp.count("'") % 2 != 0:
            tmp = re.sub(r"(?<!^)'(?!$)", "", tmp)
            tmp = tmp.replace("...", "").strip()

        fix_str += ", {}".format(tmp)

    return fix_str[2:]


def remove_extra_quotation_marks(response):
    """去除多余的引号，针对列表
    """
    # response = """['why and how there 'is a contradiction' in this connection.',]"""

    fixed_sss = ""
    in_quote = False

    i = 0
    while i < len(response):
        if response[i] == "'":
            if not in_quote:
                in_quote = True
                fixed_sss += response[i]
            else:
                # 不是尾部
                if i+1 < len(response) and response[i+1] != "," and response[i+1] != "]":
                    fixed_sss += '"'
                else:
                    in_quote = False
                    fixed_sss += response[i]
        else:
            fixed_sss += response[i]
        i += 1

    return fixed_sss


def remove_extra_quotation_marks_dict(response):
    """去掉多余引号，针对字典，针对双引号嵌套
    """
    # e.g. {"陈述5": "文档解释了为何莱布尼茨的物质理论会引发矛盾", "陈述6": "文档批判了个体物质定义中的"完整概念"包含所有谓词观点", "陈述5": "文档解释了为何莱布尼茨的物质理论会引发矛盾"}

    fixed_sss = ""
    in_quote = False
    escape = False

    i = 0
    while i < len(response):
        if response[i] == '"' and not escape:
            if not in_quote:
                in_quote = True
                fixed_sss += response[i]
            else:
                # 检查是否在引号内出现了其他的双引号
                if i + 1 < len(response) and response[i + 1] not in {',', '}', ':', ' ', '\n'}:
                    fixed_sss += "'"
                else:
                    in_quote = False
                    fixed_sss += response[i]
        else:
            # 检查当前字符是否是反斜杠, 确保当前的反斜杠不是转义字符的一部分
            if response[i] == '\\' and not escape:
                escape = True
            else:
                escape = False
            fixed_sss += response[i]
        i += 1

    return fixed_sss


def is_chinese_or_alpha(char):
    """判断一个字符是否是汉字或字母
    """
    if '\u4e00' <= char <= '\u9fff':  # 汉字的Unicode范围
        return True
    if char.isalpha():  # 判断是否为字母
        return True
    return False


def fix_lst(response: str):
    """处理OpenAI返回列表有时不稳定的情况

    Args:
        response: 理论上：字符串类型的列表，但有时不稳定需要处理（只针对字符串的列表）
    """
    response = response.strip()

    replacements = {
        "°": ",",
        '"': "'",
        "“": "'",
        "”": "'",
        "'，": "',",
        "'s": "’s",
        "\n": "",
    }
    for old, new in replacements.items():
        response = response.replace(old, new)

    if response.count('[') == 1 and response.count(']') == 1:

        if response.startswith('[') or response.endswith(']'):
            left_idx, right_idx = response.index('['), response.index(']')
            response = response[left_idx:right_idx+1]

            if response.count("'") % 2 != 0:
                response = fix_string(response)

            # e.g. [xxx]
            if response.count("'") == 0:
                response = response.replace("[", "['").replace("]", "']")

    if "信息不足" in response:
        response = "['信息不足']"

    if response.endswith("..."):
        response = response.replace("...", "]")

    if not response.endswith("]") and "]" in response:
        response = response.split("]")[0] + "]"

    # 字符串里面还有引号 e.g. ['xxx', 'xxx'xxx'xxx', ]
    response = remove_extra_quotation_marks(response)

    # 字符串列表无 ] 结尾 e.g. ['xxx', 'xxx  ['xxx',
    if response.startswith('[') and not response.endswith(']'):
        if response.endswith("',"):
            response += "]"
        elif not response.endswith("'") and is_chinese_or_alpha(response[-1]):
            response += "']"
        else:
            response += "]"

    # 开头和结尾的引号不一致  e.g. ['This information is being prsis of." ] (只对len=1进行处理)
    if response.startswith('[') and response.endswith(']'):
        if response[-2] == " ":
            response = response.replace(" ]", "]")

        if ((response[1] == "'" or response[1] == '"') or (response[-2] == "'" or response[-2] == '"')) and (response[1] != response[-2]):
            response = response.replace("'", "").replace('"', "")
            response = response.replace("[", "['").replace("]", "']")

    return response.strip()


def cosine_similarity(vec1: Union[np.ndarray, List], vec2: Union[np.ndarray, List]) -> np.ndarray:
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    emb_vec1 = vec1 / norm_vec1
    emb_vec2 = vec2 / norm_vec2
    sim = emb_vec1 @ emb_vec2.T
    # # (sim + 1) / 2
    # # edis = np.linalg.norm(vec1 - vec2)
    return sim


def cut_sent(para: str) -> List[str]:
    para = re.sub("([。！？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    sentences = para.split("\n")  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences


def flatten_dict_to_csv(rowitem):
    flat_lst = []
    for key, value in rowitem.items():
        if isinstance(value, list):
            for v in value:
                flat_lst.append(str(v))
        else:
            flat_lst.append(str(value))

    return flat_lst


def save_csv(data, save_path="./output/"):
    """结果写入文件"""

    # 与返回的字段名对应
    columns = ["flag", "file_name", "query", "context", "answer", "kimi_response", "result_score",
        "faithfulness", "faithfulness_statements", "faithfulness_flag",
        "cr_q", "cr_q1", "cr_q2", "context_relevance_response",
        "cr_gt", "context_recall_statements", "context_recall_flag",
        "ar", "ar_gen_questions", "ar_sim_sum",
        "ac", "ac_1", "ac_2", "answer_correctness_response",
        "ai", "coverage", "density", "keywards", "matched_keywards", "answer_sentences", "gt_sentences", "answer_words", "gt_words",]

    with open(os.path.join(save_path, "result.csv"), mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        for item in tqdm(data, desc="Writing to CSV", unit="item"):
            flat_item = flatten_dict_to_csv(item)
            writer.writerow(flat_item)


def stream_save_csv(data, save_path="./output/"):
    """结果流式写入文件"""

    # 与返回的字段名对应
    columns = ["flag", "file_name", "query", "context", "answer", "kimi_response", "result_score",
        "faithfulness", "faithfulness_statements", "faithfulness_flag",
        "cr_q", "cr_q1", "cr_q2", "context_relevance_response",
        "cr_gt", "context_recall_statements", "context_recall_flag",
        "ar", "ar_gen_questions", "ar_sim_sum",
        "ac", "ac_1", "ac_2", "answer_correctness_response",
        "ai", "coverage", "density", "keywards", "matched_keywards", "answer_sentences", "gt_sentences", "answer_words", "gt_words",]

    file_path = os.path.join(save_path, "stream_result.csv")
    file_exists = os.path.isfile(file_path)

    # 使用'a'模式打开文件进行追加
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        # writer = csv.DictWriter(file, fieldnames=columns)
        writer = csv.writer(file)

        # 如果文件不存在则写入表头
        if not file_exists:
            # writer.writeheader()
            writer.writerow(columns)

        writer.writerow(flatten_dict_to_csv(data))


def save_json(count_source, save_path="./output/"):

    for k, v in count_source.items():
        if isinstance(v, list):
            count_source[k] = float(np.mean(v))

    with open(os.path.join(save_path, "result.json"), "w", encoding="utf-8") as f:
        json.dump(count_source, f, ensure_ascii=False, indent=4)

    print("[DEBUG] ", "Write to json completion")

    return count_source


def update_count_source(item_res, count_source):
    count_source["avg_result_score"].append(item_res["result_score"])
    count_source["avg_faithfulness"].append(item_res["faithfulness"][0])
    count_source["avg_cr_q"].append(item_res["context_relevance"][0])
    count_source["avg_cr_gt"].append(item_res["context_recall"][0])
    count_source["avg_ar"].append(item_res["answer_relevance"][0])
    count_source["avg_ac"].append(item_res["answer_correctness"][0])
    count_source["avg_ai"].append(item_res["content_integrity"][0])
    count_source["succ_count"] += 1

