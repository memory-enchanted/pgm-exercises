"""
将 .py 练习文件转换为 .ipynb Jupyter Notebook。
每个练习 = 独立的 cell（markdown标题 + 代码(定义+调用)）。
原 .py 文件保留不动。
"""
import re
import json


def py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []

    # ---- 1. 顶部文档字符串 -> markdown cell ----
    doc_match = re.match(r'^"""\s*\n(.*?)"""', content, re.DOTALL)
    if doc_match:
        doc_text = doc_match.group(1).strip()
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + '\n' for line in doc_text.split('\n')]
        })
        after_doc = content[doc_match.end():]
    else:
        after_doc = content

    # ---- 2. 找第一个 "# =====\n# 练习" 或 "# =====\n# 综合" 标题 ----
    first_section = re.search(r'\n# =+\n# (练习|综合|运行)', after_doc)
    if first_section:
        preamble = after_doc[:first_section.start()].strip()
        rest = after_doc[first_section.start():]
    else:
        preamble = after_doc.strip()
        rest = ''

    # preamble = imports + 公共辅助函数 (draw_dag / build_student_network)
    if preamble:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [line + '\n' for line in preamble.split('\n')],
            "outputs": [],
            "execution_count": None
        })

    if not rest:
        return _write_notebook(ipynb_path, cells)

    # ---- 3. 按 "# =====\n# 标题\n# =====\n" 分割 ----
    # 每个 section 由 "标题注释块 + 函数定义" 组成
    sections = re.split(r'\n(?=# =+\n# (?:练习|综合|运行))', rest)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # 分离标题注释和代码
        # 标题格式: "# =====\n# 标题文本\n# =====\n"
        header_match = re.match(r'(# =+\n# .*?\n# =+\n)', section, re.DOTALL)
        if header_match:
            header_raw = header_match.group(0)
            code_part = section[header_match.end():].strip()

            # 标题 -> markdown (去掉 # ===== 线，保留中文标题)
            md_lines = []
            for line in header_raw.split('\n'):
                s = line.strip()
                if s.startswith('# ') and not s.startswith('# ='):
                    md_lines.append(s[2:])  # 去掉 "# "
            if md_lines:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [l + '\n' for l in md_lines]
                })
        else:
            code_part = section

        # 切掉末尾的 if __name__ == '__main__' 块（只保留练习代码）
        main_match = re.search(r'\n(?:# =+\n# .*?\n# =+\n)?\s*if\s+__name__\s*==', code_part)
        if main_match:
            code_part = code_part[:main_match.start()].strip()

        if not code_part:
            continue

        # 找函数名用于自动调用
        func_names = re.findall(r'^def\s+(\w+)\s*\(', code_part, re.MULTILINE)
        code_lines = [line + '\n' for line in code_part.split('\n')]

        # 在代码末尾添加函数调用
        if func_names:
            code_lines.append('\n')
            for fname in func_names:
                code_lines.append(f'{fname}()\n')

        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": code_lines,
            "outputs": [],
            "execution_count": None
        })

    _write_notebook(ipynb_path, cells)


def _write_notebook(ipynb_path, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"Created: {ipynb_path}")


if __name__ == '__main__':
    import glob
    import sys

    if len(sys.argv) > 1:
        # 用法: python convert_to_ipynb.py 06_ve_exercises.py
        for py_file in sys.argv[1:]:
            ipynb_file = py_file.rsplit('.', 1)[0] + '.ipynb'
            py_to_ipynb(py_file, ipynb_file)
    else:
        # 自动转换所有 *_exercises.py 文件
        exercise_files = sorted(glob.glob('*_exercises.py'))
        if not exercise_files:
            print("No *_exercises.py files found in current directory.")
            sys.exit(1)
        for py_file in exercise_files:
            ipynb_file = py_file.rsplit('.', 1)[0] + '.ipynb'
            py_to_ipynb(py_file, ipynb_file)
        print(f"\nDone — converted {len(exercise_files)} files.")
