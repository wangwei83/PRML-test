import re

# 打开 LaTeX 文件
with open(r'C:\Users\19002\Desktop\PRML-test\new-dev\main.tex', 'r', encoding='utf-8') as file:
    content = file.read()

# 使用正则表达式匹配所有备注内容
comments = re.findall(r'%.+', content)

# 将备注内容写入新文件
with open(r'C:\Users\19002\Desktop\PRML-test\new-dev\main.txt', 'w', encoding='utf-8') as output_file:
    for comment in comments:
        output_file.write(comment + '\n')

print("备注提取完成，已保存到 extracted_comments.txt 文件中。")
