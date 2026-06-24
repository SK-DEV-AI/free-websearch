"""Worker subprocess: loads reranker, reads JSON lines from stdin, returns results on stdout."""

import json, sys, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = os.environ.get("RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
_rankers = {}

def _load(model_name: str = MODEL_NAME):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    key = model_name
    if key not in _rankers:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float16 if torch.cuda.is_available() else torch.float32, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
            _rankers[key] = (model, tok, torch.device("cuda"))
        else:
            _rankers[key] = (model, tok, torch.device("cpu"))
    return _rankers[key]

def _rerank(query: str, passages: list, top_k: int = 20) -> list:
    import torch
    model, tok, device = _load()
    texts = [p.get("snippet", "") or p.get("title", "") for p in passages]
    pairs = [[query, t] for t in texts]
    inputs = tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).cpu().tolist()
    if isinstance(scores, (int, float)):
        scores = [scores]
    scored = list(zip(passages, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"score": round(s, 4), **p} for p, s in scored[:top_k]]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
        cmd = msg.get("cmd", "rerank")
        if cmd == "ping":
            json.dump({"ok": True, "model": MODEL_NAME}, sys.stdout)
        elif cmd == "rerank":
            result = _rerank(msg["query"], msg["passages"], msg.get("top_k", 20))
            json.dump({"ok": True, "result": result}, sys.stdout)
        elif cmd == "shutdown":
            json.dump({"ok": True}, sys.stdout)
            break
        else:
            json.dump({"ok": False, "error": f"unknown cmd: {cmd}"}, sys.stdout)
            break
    except Exception as e:
        json.dump({"ok": False, "error": str(e)}, sys.stdout)
    sys.stdout.flush()
