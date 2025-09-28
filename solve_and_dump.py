from klotski import load_board_states, astar
from heuristics import blocking_goal_h
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

INPUT_FILE = "klotski_levels_30_20250928_022610.txt"
OUTPUT_FILE = f"klotski_levels_30_{timestamp}_cost.txt"

def dump_block_with_cost(board, state, name, cost):
    """生成单个局面的文本块，并在末尾加 COST"""
    lines = []
    if name:
        lines.append(f"NAME {name}")
    lines.append(f"BOARD {board.W} {board.H}")

    gw, gh = (-1, -1)
    gx, gy = (-1, -1)
    if board.goal_piece is not None:
        gw, gh = board.goal_piece
    if board.goal_pos is not None:
        gx, gy = board.goal_pos
    lines.append(f"GOAL {gw} {gh} {gx} {gy}")

    lines.append(f"PIECES {len(state.pieces)}")
    for p in state.pieces:
        lines.append(f"{p.w} {p.h} {p.x} {p.y}")

    # 最优步数
    lines.append(f"COST {cost if cost is not None else -1}")
    lines.append("")  # 空行分隔
    return "\n".join(lines)


if __name__ == "__main__":
    entries = load_board_states(INPUT_FILE)
    print(f"共读取 {len(entries)} 个局面")

    with open(OUTPUT_FILE, "w") as f:
        f.write(str(len(entries)) + "\n")  # 第一行写条目数

        for idx, (board, state, name) in enumerate(entries):
            print(f"[{idx+1}/{len(entries)}] 处理 {name or '(no name)'}...")

            result = astar(board, state, blocking_goal_h, max_steps=80500000)
            cost = result["cost"]

            print(f"   cost={cost}, expanded={result['expanded']}, time={result['time']:.2f}s")

            block = dump_block_with_cost(board, state, name, cost)
            f.write(block + "\n")

    print(f"✅ 已写入 {OUTPUT_FILE}")
