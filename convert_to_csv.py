"""把 evaluate.py 输出的 JSONL 转成 Kaggle 提交的 CSV。"""
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

    df = pd.DataFrame([
        {"id": r["id"], "response": r["response"]}
        for r in results
    ])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"✅ 已保存 {len(df)} 行到: {args.output}")
    print(df.head(2))


if __name__ == "__main__":
    main()