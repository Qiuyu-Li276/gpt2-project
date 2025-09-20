import json
import re

def clean_json_file(input_file, output_file):
    """
    读取一个JSON文件，清除其中的控制字符，然后将干净的数据
    保存到一个新的JSON文件中。

    参数:
    input_file (str): 输入的JSON文件名。
    output_file (str): 输出的JSON文件名。
    """
    # 定义用于匹配和移除控制字符的正则表达式
    control_chars = re.compile(r'[\x00-\x1F\x7F]')

    # 定义一个函数来移除字符串中的控制字符
    def remove_control_characters(s):
        return control_chars.sub('', s)

    try:
        # 打开并读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查数据是否是列表（根据您之前的脚本，它应该是一个字符串列表）
        if not isinstance(data, list):
            print("警告：输入的JSON文件不是一个列表。脚本将继续，但这可能不是预期的格式。")

        # 使用列表推导式来清理列表中的每一个字符串
        # 这种方式比递归函数更直接，因为我们知道数据结构是一个简单的列表
        cleaned_data = [remove_control_characters(item) for item in data if isinstance(item, str)]
        
        # 将清理后的数据写入新的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        
        print(f"成功清理了 {len(cleaned_data)} 条新闻。")
        print(f"文件已成功保存为: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file}'。请确保文件名正确并且文件在当前目录下。")
    except json.JSONDecodeError as e:
        print(f"解析JSON时出错: {e}。请检查 '{input_file}' 是否为有效的JSON格式。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

# --- 使用方法 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    input_filename = 'train_sports.json'
    output_filename = 'train_sports_cleaned.json'

    # 调用函数进行清洗
    clean_json_file(input_filename, output_filename)