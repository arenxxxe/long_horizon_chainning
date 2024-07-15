import os
import re

def remove_comment(file_path):
    with open (file=file_path,mode="r",encoding="utf-8") as file:
        content = file.read()

    #正则表达式 需要测试的
    patttern=r"(\"\"\".*?\"\"\" | \'\'\'.*?\'\'\' | #.*?&)"
    no_comment_code=re.sub(pattern=patttern,repl=content,flags=re.MULTILINE)

    with open (file=file_path,mode="w",encoding="utf-8") as file:
        content = file.write(no_comment_code)

string1=".py"


def iterate_remove(directory_path):
    for root,_,files in os.walk(directory_path):
        for file in files:
            fpath=os.path.join(root,file)
            if fpath.endswith(".py"):
                remove_comment(fpath)

#建议先用简单的测试文件和相同结构的测试项目文件 或者复制真实文件来测试
directory_path=""
iterate_remove(directory_path)