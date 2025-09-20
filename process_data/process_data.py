import json

def convert_cnews_to_json(input_file, output_file, category='体育'):
    """
    读取cnews.train.txt文件，仅提取指定类别的新闻内容，
    并将其保存为JSON格式。

    参数:
    input_file (str): 输入的cnews.train.txt文件名。
    output_file (str): 输出的.json文件名。
    category (str): 要筛选的新闻类别，默认为'体育'。
    """
    sports_news_contents = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 假设标签和内容之间由制表符(\t)分隔
                parts = line.strip().split('\t', 1)
                
                # 检查这一行是否有两部分，并且第一部分是指定的类别
                if len(parts) == 2 and parts[0] == category:
                    # parts[0] 是标签, parts[1] 是新闻内容
                    sports_news_contents.append(parts[1])

        # 将筛选出的体育新闻内容列表写入到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 使用ensure_ascii=False来确保中文字符在文件中正确显示
            json.dump(sports_news_contents, f, ensure_ascii=False, indent=4)

        print(f"成功筛选并处理了 {len(sports_news_contents)} 条'{category}'新闻。")
        print(f"文件已成功保存为: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file}'。请确保文件名正确并且文件在当前目录下。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 使用方法 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    input_filename = 'cnews.train.txt'
    output_filename = 'train_sports.json'

    # 调用函数进行转换，只提取体育新闻
    convert_cnews_to_json(input_filename, output_filename, category='体育')