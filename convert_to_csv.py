"""把 evaluate.py 输出的 JSONL 转成 Kaggle 提交的 CSV。

evaluate.py 已经在内部做完多数投票，每个 id 在 JSONL 里仍然只占 1 行，
其中 `response` 字段就是胜出的那一条回答。所以这里直接读 id, response 即可。
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    results = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    # Sanity check
    ids = [r["id"] for r in results]
    if len(ids) != len(set(ids)):
        print(f"⚠️  警告：发现重复 id！原 {len(ids)} 行，去重后 {len(set(ids))} 个 id；保留每个 id 的第一条")
        seen = set()
        deduped = []
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)
        results = deduped

    df = pd.DataFrame([
        {"id": r["id"], "response": r["response"]}
        for r in results
    ])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # pandas 默认对含 ,\n" 的字段会自动加双引号转义，符合标准 CSV
    df.to_csv(args.output, index=False)
    print(f"✅ 已保存 {len(df)} 行（{df['id'].nunique()} 个 id）到: {args.output}")
    print(df.head(2))


if __name__ == "__main__":
    main()