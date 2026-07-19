"""
icon_map.py —— 构建 icon_id -> PNG 文件名 的映射。

真实 AAC pictogram 在 /home/user1/liuduanye/AACTest/AAC/data/images/*.png。
其权威映射来自 dataset_custom.json 的 (word -> filename)。
部分动词 id 在文件名里插入了逗号（build_to -> build_,_to.png），
做一层回退即可覆盖绝大多数。
"""
import os
import json

DATASET_JSON = "/home/user1/liuduanye/AACTest/AAC/data/dataset_custom.json"
IMAGES_DIR = "/home/user1/liuduanye/AACTest/AAC/data/images"


def _verb_comma_fallback(word: str) -> str:
    """want_to -> want_,_to.png 这类逗号插入回退（磁盘文件名在 _ 之后插入 ,_ ）。"""
    if "_to" in word:
        return word.replace("_to", "_,_to") + ".png"
    return word + ".png"


def build_icon_map(dataset_json: str = DATASET_JSON, images_dir: str = IMAGES_DIR):
    """返回 (icon_map, images_dir)。

    icon_map: {icon_id: filename}  —— 仅包含磁盘上真实存在的图片。
    回退顺序：filename 字段 -> word+.png -> word 逗号插入回退。
    """
    icon_map = {}
    try:
        with open(dataset_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[icon_map] Warning: cannot read {dataset_json}: {e}")
        data = []

    # dataset_custom.json 可能是 list 或 {"data": [...]}
    if isinstance(data, dict):
        data = data.get("data", [])

    for rec in data:
        if not isinstance(rec, dict):
            continue
        word = rec.get("word") or rec.get("original_symbol_name")
        fname = rec.get("filename")
        if not word:
            continue
        # 校验磁盘存在；不存在则尝试回退文件名
        candidates = []
        if fname:
            candidates.append(fname)
        candidates.append(word + ".png")
        candidates.append(_verb_comma_fallback(word))
        resolved = None
        for c in candidates:
            if c and os.path.isfile(os.path.join(images_dir, c)):
                resolved = c
                break
        if resolved:
            icon_map[word] = resolved

    return icon_map, images_dir


if __name__ == "__main__":
    m, d = build_icon_map()
    print(f"images_dir={d}")
    print(f"mapped icons with real PNG: {len(m)}")
    for k in ["I", "want_to", "water", "build_to", "go_to"]:
        print(f"  {k!r} -> {m.get(k, '<NONE>')!r}")
