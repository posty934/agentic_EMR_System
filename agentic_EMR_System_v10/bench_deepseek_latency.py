#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用于测试 OpenAI 兼容接口（如 DeepSeek）的端到端延迟。
特点：
1. 直接走 requests，不经过 LangChain，尽量排除框架额外开销
2. 同时测试：
   - 继承当前环境代理
   - 强制禁用代理
3. 支持非流式 / 流式（可测首 token 时间）
"""

import os
import json
import time
import argparse
import statistics
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import requests

load_dotenv()
def normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("LLM_BASE_URL 为空")
    return base_url


def build_chat_url(base_url: str) -> str:
    base_url = normalize_base_url(base_url)
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/chat/completions"


def get_env(name: str, required: bool = True, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def build_session(disable_proxy: bool) -> requests.Session:
    session = requests.Session()
    if disable_proxy:
        session.trust_env = False
        session.proxies = {"http": "", "https": ""}
    return session


def run_once(
    url: str,
    api_key: str,
    model: str,
    timeout: float,
    disable_proxy: bool,
    stream: bool,
    payload_text: str,
) -> Dict[str, Any]:
    session = build_session(disable_proxy=disable_proxy)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个延迟测试助手。"},
            {"role": "user", "content": payload_text},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": stream,
    }

    t0 = time.perf_counter()
    first_token_s = None
    content = ""

    try:
        resp = session.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
        status_code = resp.status_code
        resp.raise_for_status()

        if not stream:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            total_s = time.perf_counter() - t0
            return {
                "ok": True,
                "status_code": status_code,
                "total_s": total_s,
                "first_token_s": None,
                "response_preview": content[:120],
                "error": None,
            }

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("data: "):
                raw_line = raw_line[6:]
            if raw_line.strip() == "[DONE]":
                break
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            delta = item.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece and first_token_s is None:
                first_token_s = time.perf_counter() - t0
            content += piece

        total_s = time.perf_counter() - t0
        return {
            "ok": True,
            "status_code": status_code,
            "total_s": total_s,
            "first_token_s": first_token_s,
            "response_preview": content[:120],
            "error": None,
        }
    except Exception as e:
        total_s = time.perf_counter() - t0
        return {
            "ok": False,
            "status_code": None,
            "total_s": total_s,
            "first_token_s": first_token_s,
            "response_preview": "",
            "error": repr(e),
        }


def summarize(results):
    oks = [x for x in results if x["ok"]]
    if not oks:
        return {"ok_count": 0, "avg_total_s": None, "p50_total_s": None, "avg_first_token_s": None}

    total_list = [x["total_s"] for x in oks]
    first_list = [x["first_token_s"] for x in oks if x["first_token_s"] is not None]

    return {
        "ok_count": len(oks),
        "avg_total_s": statistics.mean(total_list),
        "p50_total_s": statistics.median(total_list),
        "avg_first_token_s": statistics.mean(first_list) if first_list else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="每种模式测试次数")
    parser.add_argument("--timeout", type=float, default=120, help="单次请求超时秒数")
    parser.add_argument("--stream", action="store_true", help="使用流式输出，额外统计首 token 时间")
    parser.add_argument(
        "--text",
        type=str,
        default="请只回复四个字：ping ok。",
        help="测试提示词",
    )
    args = parser.parse_args()

    api_key = get_env("LLM_API_KEY")
    base_url = get_env("LLM_BASE_URL")
    model = get_env("LLM_MODEL_NAME")
    url = build_chat_url(base_url)

    print("=" * 80)
    print("DeepSeek / OpenAI-compatible API 延迟测试")
    print("=" * 80)
    print(f"URL   : {url}")
    print(f"MODEL : {model}")
    print(f"HTTP_PROXY  = {os.getenv('HTTP_PROXY')}")
    print(f"HTTPS_PROXY = {os.getenv('HTTPS_PROXY')}")
    print(f"ALL_PROXY   = {os.getenv('ALL_PROXY')}")
    print(f"STREAM      = {args.stream}")
    print()

    modes = [
        ("inherit_env_proxy", False),
        ("disable_proxy", True),
    ]

    all_mode_results = {}

    for mode_name, disable_proxy in modes:
        print("-" * 80)
        print(f"模式: {mode_name}")
        print("-" * 80)
        mode_results = []

        for i in range(1, args.runs + 1):
            result = run_once(
                url=url,
                api_key=api_key,
                model=model,
                timeout=args.timeout,
                disable_proxy=disable_proxy,
                stream=args.stream,
                payload_text=args.text,
            )
            mode_results.append(result)

            if result["ok"]:
                print(
                    f"[{i}/{args.runs}] ok "
                    f"| total={result['total_s']:.3f}s "
                    f"| first_token={result['first_token_s'] if result['first_token_s'] is None else round(result['first_token_s'], 3)} "
                    f"| preview={result['response_preview']!r}"
                )
            else:
                print(
                    f"[{i}/{args.runs}] fail "
                    f"| total={result['total_s']:.3f}s "
                    f"| error={result['error']}"
                )

        summary = summarize(mode_results)
        all_mode_results[mode_name] = {
            "summary": summary,
            "results": mode_results,
        }

        print("汇总:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print()

    print("=" * 80)
    print("最终对比结论（看 avg_total_s / avg_first_token_s）")
    print("=" * 80)
    print(json.dumps(
        {k: v["summary"] for k, v in all_mode_results.items()},
        ensure_ascii=False,
        indent=2
    ))


if __name__ == "__main__":
    main()
