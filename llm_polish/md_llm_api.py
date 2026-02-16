import json
from llm_keywords import ali_api_text_key_001
from llm_api import ali_api_text
from llm_prompt import DEFAULT_TEXT_POLISH_SYSTEM_PROMPT
from llm_local import local_api_text
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="md_llm_api所需参数")

# 添加命令行参数
parser.add_argument("--polish_type", type=str, required=True, help="润色种类")
parser.add_argument("--model_type", type=str, required=True, help="润色模型种类 详见api.py")
parser.add_argument("--report_info_path", type=str, required=True, help="路径：report_info.json")
parser.add_argument("--input_file_path_1", type=str, required=True, help="路径：初版建议 report_jianyi.txt")
parser.add_argument("--input_file_path_2", type=str, required=True, help="路径：初版数值分析 report_abnormal.txt")
parser.add_argument("--output_file_path_1", type=str, required=True, help="路径：润色后建议 report_polished_jianyi.txt")
parser.add_argument("--output_file_path_2", type=str, required=True, help="路径：数值分析 + 润色建议 report_polished_merge.txt")


# 解析参数
args = parser.parse_args()

# 取出参数
polish_type = int(args.polish_type)
model_type = int(args.model_type)
report_info_path = args.report_info_path
input_file_path_1 = args.input_file_path_1
input_file_path_2 = args.input_file_path_2
output_file_path_1 = args.output_file_path_1
output_file_path_2 = args.output_file_path_2


def read_input_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        input_text = file.read().strip()
    return input_text

def merge_txt_files(file1_path, file2_path, output_path):
    def read_file_content(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 若文件为空，直接返回空字符串
            if not lines:
                return ""

            # 如果第一行包含特殊标记，则跳过第一行
            first_line = lines[0].strip()
            if first_line in ("no_need_post_process", "need_post_process"):
                return ''.join(lines[1:])  # 从第二行开始
            else:
                return ''.join(lines)  # 全部内容
        except Exception as e:
            print(f"读取文件时出错 ({path}): {e}")
            return ""

    try:
        content1 = read_file_content(file1_path)
        content2 = read_file_content(file2_path)

        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(content1)
            out.write('\n')  # 可选分隔符
            out.write(content2)

    except Exception as e:
        print(f"合并文件时发生错误: {e}")
        
def polish_jianyi(input_path, output_path, polish_type):

    sys_prompt = "你是一个精通医学语言的专业写作助手，擅长将医学报告润色得更专业、简洁、准确。"
    
    prompt_list = [
            "润色并合理化以下内容,按照【一、饮食调整】，【二、适度运动】,【三、日常生活】，【四、保健品服用】四个方面各润色为一整段信息的流畅的文字,每段文字50字左右,同时在每个方面要避免描述的重复与啰嗦，尽量精简，并分别以【一、饮食调整】，【二、适度运动】,【三、日常生活】，【四、保健品服用】开头。你只需给出润色后的内容，无需任何其它说明",
     ]
    
    # 防止索引越界，默认给最后一个prompt
    if 0 <= polish_type < len(prompt_list):
        prompt = prompt_list[polish_type]
    else:
        prompt = prompt_list[-1]

    # 读取整个文件内容
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 如果文件为空，直接写入空并返回
    if not lines:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    # 获取第一行标记并去除首尾空白
    first_line_flag = lines[0].strip()
    
    # 获取第二行及以后的内容，并将列表重新组合成字符串
    remaining_content = "".join(lines[1:]).strip()

    content_to_write = ""

    if first_line_flag == "need_post_process":
        # 逻辑1：需要润色，调用API润色内容（第二行及以后）
        print(f"检测到 need_post_process，开始润色...")
        model_numebr = 2  # 注意：此处建议使用传入的 model_type 参数，如果必须保持原样则不动
        
        if remaining_content:
            content_to_write = ali_api_text(DEFAULT_TEXT_POLISH_SYSTEM_PROMPT, 
                                            prompt, 
                                            remaining_content, 
                                            model_numebr, 
                                            ali_api_text_key_001,
                                            )
        else:
            content_to_write = ""

    else:
        # 逻辑2：不需要润色，直接存入原始内容（第二行及以后）
        print(f"未检测到 need_post_process，跳过润色。")
        content_to_write = remaining_content

    # 写入结果文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(content_to_write))


if __name__ == '__main__':

    #建议润色
    polish_jianyi(input_file_path_1, output_file_path_1 ,polish_type)

    merge_txt_files(input_file_path_2, output_file_path_1, output_file_path_2)




