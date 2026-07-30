from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


JUDGE_PROMPT = """You are evaluating whether an AI assistant successfully solved a user's task using external APIs.

Rate how well the agent's final answer satisfies the user's query with a continuous score between 0.0 and 1.0, where
1.0 = every part of the request is answered with concrete, non-placeholder content grounded in the API observations;
0.5 = the request is partially answered, or answered without grounding some part of it in an observation;
0.0 = the agent gave up, returned no answer, produced only failed calls, or answered a different question.

Judge only the final answer and the trajectory that produced it. Do not reward effort, length, or apologies. Do not penalize an answer for using fewer APIs than were available.

User query: {query}
Available APIs: {tool_names}
Trajectory: {trajectory}
Final answer: {final_answer}
Respond with JSON only:
{{"reason": "one sentence", "score": 0.0}}
"""


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def canonical_tool_set(names: list[str]) -> str:
    return "|".join(sorted(set(names)))


def trajectory_fields(item: dict) -> dict:
    query = str(item.get("query", "")).strip()
    final_answer = str(item.get("final_answer", "")).strip()
    trajectory = item.get("trajectory", "")
    if not isinstance(trajectory, str):
        trajectory = json.dumps(trajectory, ensure_ascii=False)

    conversations = item.get("conversations", [])
    if conversations:
        trajectory = json.dumps(conversations, ensure_ascii=False)
        if not query:
            query = next(
                (
                    str(message.get("value", "")).replace("Begin!", "").strip()
                    for message in conversations
                    if message.get("from") == "user"
                ),
                "",
            )
        if not final_answer:
            for message in reversed(conversations):
                if message.get("from") != "assistant":
                    continue
                value = str(message.get("value", ""))
                if "final_answer" in value:
                    try:
                        start = value.rfind("{")
                        payload = json.loads(value[start:])
                        final_answer = str(payload.get("final_answer", "")).strip()
                    except (ValueError, json.JSONDecodeError):
                        pass
                if final_answer:
                    break

    tools = item.get("selected_tools", item.get("retrieved_tools", item.get("tool_set", [])))
    tools = [str(tool) for tool in tools] if isinstance(tools, list) else []
    query_id = str(item.get("query_id", item.get("id", ""))) or query_hash(query)
    failed = bool(item.get("failed", False) or item.get("exhausted", False))
    failed = failed or not final_answer
    return {
        "query_id": query_id,
        "query": query,
        "final_answer": final_answer,
        "trajectory": trajectory,
        "tools": tools,
        "failed": failed,
    }


def judge(
    fields: dict,
    api_key: str,
    api_base: str,
    model: str,
    retries: int = 3,
) -> Optional[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base or None)
    prompt = JUDGE_PROMPT.format(
        query=fields["query"],
        tool_names=", ".join(fields["tools"]),
        trajectory=fields["trajectory"],
        final_answer=fields["final_answer"],
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=1.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            score = float(max(0.0, min(1.0, result.get("score", 0.0))))
            return {"score": score, "reason": str(result.get("reason", ""))}
        except Exception as exc:
            if attempt + 1 == retries:
                print(f"Judge failed after {retries} attempts: {exc}")
                return None
            time.sleep(2**attempt)
    return None


def score_item(item: dict, api_key: str, api_base: str, model: str) -> tuple[str, str, dict] | None:
    fields = trajectory_fields(item)
    if not fields["query"] or not fields["tools"]:
        return None
    if fields["failed"]:
        result = {"score": 0.0, "reason": "No valid grounded final answer."}
    else:
        result = judge(fields, api_key, api_base, model)
        if result is None:
            return None
    result["selected_tools"] = sorted(set(fields["tools"]))
    return fields["query_id"], canonical_tool_set(fields["tools"]), result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HYSET execution-reward cache")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api_key", default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--api_base", default=os.getenv("JUDGE_API_BASE", ""))
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", ""))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.api_key or not args.model:
        parser.error("JUDGE_API_KEY/--api_key and JUDGE_MODEL/--model are required")
    return args


def main() -> None:
    args = parse_args()
    with open(args.trajectories, encoding="utf-8") as stream:
        items = json.load(stream)
    if args.limit:
        items = items[: args.limit]

    output = Path(args.output)
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        records = payload.get("records", {})
    else:
        records = {}

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_item, item, args.api_key, args.api_base, args.model) for item in items]
        for future in as_completed(futures):
            scored = future.result()
            if scored is None:
                continue
            query_id, set_key, result = scored
            records.setdefault(query_id, {})[set_key] = result
            completed += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            {"format": "hyset-execution-rewards-v1", "records": records},
            stream,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved {completed} new rewards to {output}")


if __name__ == "__main__":
    main()
