from google import genai
import json, os, re, importlib.util, random, time
from typing import List, Dict, Optional, Tuple
import sys
import datetime
from google.genai import errors as genai_errors
import asyncio, itertools
from dataclasses import dataclass, field
import multiprocessing as mp
import textwrap
import ast

# ====== State ======
@dataclass
class IslandState:
    island_id: int
    results: List[Tuple[int, float, str]] = field(default_factory=list)
    cnt: int = 0

# ====== Logger ======
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ====== Helpers ======
MAX_CODE_CHARS = 200_000
EVAL_TIMEOUT_SEC = 10

def safe_read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return None

def try_extract_heuristic(code_text: Optional[str]) -> Optional[str]:
    if not code_text:
        return None
    try:
        tree = ast.parse(code_text)
    except SyntaxError as e:
        print(f"[WARN] unparsable code skipped: {e}")
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "heuristic":
            lines = code_text.splitlines()
            start = node.lineno - 1
            end = getattr(node, "end_lineno", len(lines))
            return textwrap.dedent("\n".join(lines[start:end]))
    print("[WARN] no def heuristic found")
    return None

def rename_func(def_text: str, new_name: str) -> str:
    return re.sub(r"def\s+heuristic\s*\(", f"def {new_name}(", def_text, count=1)

def validate_api_json(text: str) -> Optional[str]:
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        code = data.get("code")
        if not code or not isinstance(code, str):
            return None
        if len(code) > MAX_CODE_CHARS:
            print(f"[WARN] code too large ({len(code)} chars)")
            return None
        if "def heuristic" not in code:
            print("[WARN] missing def heuristic in payload")
            return None
        return code
    except Exception as e:
        print(f"[WARN] bad JSON: {e}")
        return None

# ====== Prompt builder ======
def build_best_shot_prompt(
    low_code: str, low_score: float,
    high_code: str, high_score: float,
    rand_list: Optional[List[Tuple[str, float]]] = None
) -> str:
    low_def  = try_extract_heuristic(low_code)
    high_def = try_extract_heuristic(high_code)

    if high_def is None and low_def is not None:
        high_def, high_score, low_def, low_score = low_def, low_score, None, float("inf")
    if high_def is None:
        high_def = (
            "def heuristic_v1(item: int, bins: List[Dict[str, object]], capacity: int) -> Optional[int]:\n"
            "    # fallback placeholder\n"
            "    return -1\n"
        )
        high_score = float("inf")


    header = (
        "You are evolving a deterministic 1D bin-packing heuristic.\n"
        "Signature:\n"
        "def heuristic(item: int, bins: List[Dict[str, object]], capacity: int) -> Optional[int]\n"
        "- bins: list of dicts with keys 'load' (int) and 'items' (List[int])\n"
        "- Return a bin index (>=0) or None/-1 to open a new bin.\n"
        "- MUST never exceed capacity; must be deterministic and side-effect free.\n\n"
        "Background:\n"
        "- Bin Packing is NP-hard; goal is to minimize number of bins (i.e. minimize excess space).\n"
        "- Standard greedy heuristics: First Fit, Best Fit, Worst Fit, Next Fit.\n"
        "- Harmonic heuristics partition item sizes into ranges (1/2–1], (1/3–1/2], (1/4–1/3], etc.\n"
        "- Strong strategies often mix greedy rules, tie-breakers, lookahead, or size classification.\n\n"
        "Task:\n"
        f"- Given past implementations, produce a new `heuristic` that beats the current best ({high_score:.4f}).\n"
        "- Concise code is fine, but you may innovate (greedy, harmonic classes, hybrids, etc.).\n\n"
        "Implementation rules:\n"
        "- Start with imports, then define exactly one function `heuristic` as above.\n"
        "- Output ONLY JSON: {{\"code\": \"<full python code>\"}} (no markdown).\n"
        "- Do not include v0/v1/vr; only the final `heuristic`.\n"
    )


    prompt = header
    if low_def is not None:
        prompt += f"\n# === Low (worse) v0 [score={low_score:.4f}] ===\n{rename_func(low_def, 'heuristic_v0')}"
    prompt += f"\n\n# === High (better) v1 [score={high_score:.4f}] ===\n{rename_func(high_def, 'heuristic_v1')}"

    if rand_list:
        for idx, (rc, rs) in enumerate(rand_list):
            rdef = try_extract_heuristic(rc)
            if rdef:
                prompt += f"\n\n# === Random vr{idx} [score={rs:.4f}] ===\n{rename_func(rdef, f'heuristic_vr{idx}')}"
    prompt += "\n\n# === Now produce improved final `heuristic` ===\n"
    return prompt

# ====== IO ======
FOLDER = "generated_programs"
os.makedirs(FOLDER, exist_ok=True)

def write_code(island_id: int, cnt: int, code: str) -> str:
    path = os.path.join(FOLDER, f"generated_program_{island_id}_{cnt}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path

def load_heuristic_from_file(path: str):
    unique_name = f"mod_{os.path.basename(path).replace('.py','')}_{int(time.time()*1e6)}"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "heuristic")

# ====== Model call ======
client = genai.Client(api_key="AIzaSyDVtcHutNv7qP8Qhzcfn0dxOhGUyhS47uA")
API_MAX_CONCURRENCY = 8
api_sem = asyncio.Semaphore(API_MAX_CONCURRENCY)

def model_generate(prompt: str) -> str:
    backoff = 0.6
    models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    random.shuffle(models)
    last_err = None
    for i in range(5):
        model = models[min(i, len(models)-1)]
        print("model used:")
        print(model)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={
                    "response_mime_type": "application/json",
                    "system_instruction": "Return ONLY JSON {\"code\": \"...\"}",
                },
            )
            code = validate_api_json(getattr(resp, "text", "") or "")
            if code:
                return code
            raise RuntimeError("invalid payload")
        except Exception as e:
            last_err = e
            print(f"[WARN] model error: {e}")
            time.sleep(backoff)
            backoff = min(8.0, backoff * 2)
    return "from typing import List, Dict, Optional\ndef heuristic(item:int,bins:List[Dict[str,object]],capacity:int)->Optional[int]:\n    return -1\n"

async def model_generate_async(prompt: str) -> str:
    async with api_sem:
        return await asyncio.to_thread(model_generate, prompt)

# ====== Evaluation ======
def _eval_worker(path: str, out_q: mp.Queue):
    try:
        import evaluate
        heuristic = load_heuristic_from_file(path)
        score = evaluate.evaluate(heuristic, "or1.txt")
        out_q.put(float(score))
    except Exception as e:
        out_q.put(e)

def evaluate_file(path: str) -> float:
    out_q = mp.Queue()
    proc = mp.Process(target=_eval_worker, args=(path, out_q), daemon=True)
    proc.start()
    proc.join(EVAL_TIMEOUT_SEC)
    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        print(f"[WARN] timeout evaluating {path}")
        return float("inf")
    try:
        result = out_q.get_nowait()
        if isinstance(result, Exception):
            print(f"[WARN] eval failed: {result}")
            return float("inf")
        return float(result)
    except Exception as e:
        print(f"[WARN] eval infra error: {e}")
        return float("inf")

async def evaluate_file_async(path: str) -> float:
    return await asyncio.to_thread(evaluate_file, path)

# ====== Selection ======
def select_low_high_rand_from(results_list: List[Tuple[int, float, str]]):
    valid = [(c, s, p) for (c, s, p) in results_list if p and os.path.exists(p)]
    if len(valid) < 2:
        raise RuntimeError("need at least two programs")
    sorted_list = sorted(valid, key=lambda x: x[1])
    best = sorted_list[0]; worst = sorted_list[-1]
    pool = sorted_list[1:-1]
    rand_choices = random.sample(pool, min(3, len(pool))) if pool else []
    low_code = safe_read_file(worst[2]); low_score = worst[1]
    high_code = safe_read_file(best[2]); high_score = best[1]
    rand_codes = [(safe_read_file(r[2]), r[1]) for r in rand_choices if safe_read_file(r[2])]
    return (low_code, low_score), (high_code, high_score), rand_codes

# ====== Iteration ======
async def generate_one_iteration_async(state: IslandState) -> Tuple[int, float]:
    try:
        (low_code, low_score), (high_code, high_score), rand_list = \
            select_low_high_rand_from(state.results)
        prompt = build_best_shot_prompt(low_code, low_score, high_code, high_score, rand_list)
        new_code = await model_generate_async(prompt)
        path = write_code(state.island_id, state.cnt, new_code)
        score = await evaluate_file_async(path)
    except Exception as e:
        print(f"[WARN] island {state.island_id} v{state.cnt} gen failed: {e}")
        path = os.path.join(FOLDER, f"failed_{state.island_id}_{state.cnt}.py")
        with open(path, "w") as f: f.write("# failed\n")
        score = float("inf")
    state.results.append((state.cnt, score, path))
    print(f"[Island {state.island_id} | v{state.cnt}] score={score:.4f} -> {path}")
    state.cnt += 1
    return state.cnt-1, score

# ====== Bootstrap ======
def bootstrap_islands(num_islands: int = 8) -> List[IslandState]:
    init_meta = [
        (0, 6.42, os.path.join(FOLDER, "generated_program_0.py")),
        (1, 5.81, os.path.join(FOLDER, "generated_program_1.py")),
    ]
    init_codes = [(safe_read_file(p), s) for (_, s, p) in init_meta]
    states = []
    for i in range(num_islands):
        st = IslandState(island_id=i)
        for ver, (code, score) in enumerate(init_codes):
            if not code: continue
            path = write_code(i, ver, code)
            st.results.append((ver, float(score), path))
            print(f"[Bootstrap] island={i} v={ver} score={score:.4f} -> {path}")
        st.cnt = len(st.results)
        states.append(st)
    return states

# ====== Cull & refill ======
def cull_and_refill(states: List[IslandState]):
    island_bests = []
    for st in states:
        if st.results:
            br = min(st.results, key=lambda x: x[1])
            island_bests.append((st, br))
        else:
            island_bests.append((st, (None, float("inf"), "")))
    ranked = sorted(island_bests, key=lambda x: x[1][1])
    survivors = [st for st, _ in ranked[:len(states)//2]]
    culled = [st for st, _ in ranked[len(states)//2:]]
    print("\n=== CULL ===")
    print("Survivors:", [s.island_id for s in survivors])
    print("Culled:", [c.island_id for c in culled])
    survivor_best_snippets = []
    for st in survivors:
        cnt, score, path = min(st.results, key=lambda x: x[1])
        survivor_best_snippets.append((st.island_id, safe_read_file(path), score, path))
    for st in culled:
        st.results.clear(); st.cnt = 0
        for seed_idx, (src_id, code_text, seed_score, src_path) in enumerate(survivor_best_snippets):
            if not code_text: continue
            path = write_code(st.island_id, seed_idx, code_text)
            st.results.append((seed_idx, seed_score, path))
            print(f"[Refill] island={st.island_id} <- from {src_id} (score={seed_score:.4f})")

# ====== Summary ======
def island_summary_str(state: IslandState) -> str:
    try:
        best_cnt, best_score, _ = min(state.results, key=lambda x: x[1])
        last_cnt, last_score, _ = state.results[-1]
        delta = last_score - best_score
        return f"Island {state.island_id} | best={best_score:.4f} | last={last_score:.4f} | n={len(state.results)} | Δ={delta:+.4f}"
    except Exception:
        return f"Island {state.island_id} | empty"

def print_summary(states: List[IslandState], round_idx: int):
    print(f"\n===== ROUND {round_idx+1} SUMMARY =====")
    for st in states:
        print(island_summary_str(st))
    all_results = [(st.island_id, *rec) for st in states for rec in st.results if rec]
    if all_results:
        gbest = min(all_results, key=lambda x: x[2])
        print(f"GLOBAL_BEST: island={gbest[0]} v={gbest[1]} score={gbest[2]:.4f} -> {gbest[3]}")

# ====== Main ======
async def main_multi_islands():
    states = bootstrap_islands(num_islands=8)
    TOTAL_ROUNDS = 190
    CHECKPOINT_INTERVAL = 25
    SUMMARY_INTERVAL = 25
    for r in range(TOTAL_ROUNDS):
        tasks = [asyncio.create_task(generate_one_iteration_async(st)) for st in states]
        results_or_exc = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, res in enumerate(results_or_exc):
            if isinstance(res, Exception):
                print(f"[WARN] task failed on island {states[idx].island_id}: {res}")
        if (r+1) % SUMMARY_INTERVAL == 0:
            print_summary(states, r)
        if (r+1) % CHECKPOINT_INTERVAL == 0:
            cull_and_refill(states)
            print_summary(states, r)
    print("\n=== Final Results ===")
    for st in states:
        if st.results:
            best = min(st.results, key=lambda x: x[1])
            print(f"Island {st.island_id}: v{best[0]} score={best[1]:.4f}")
    all_results = [(st.island_id, *rec) for st in states for rec in st.results if rec]
    if all_results:
        gbest = min(all_results, key=lambda x: x[2])
        print(f"\nGLOBAL BEST: island={gbest[0]} v={gbest[1]} score={gbest[2]:.4f}")

if __name__ == "__main__":
    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"log_{timestamp}.txt")
    sys.stdout = Logger(logfile)
    asyncio.run(main_multi_islands())
