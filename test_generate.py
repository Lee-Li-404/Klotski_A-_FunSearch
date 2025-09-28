# klotski_solver_general.py
import pygame
import heapq
import time
import random
import colorsys
import os 
from heuristics import trivial_h, blocking_h, corridor_h
from board_state_class import Piece, State, Board
from classic_start import cao_cao_start, standard_start, simple30_start, tiny_start
from graph import GraphLogger
import sys
import datetime

# ---- Graph logging for A* exploration → interactive HTML ----

# ----------------------------
# ASCII 渲染
# ----------------------------
def render_ascii(board, s, labels=None):
    grid = [['.' for _ in range(board.W)] for _ in range(board.H)]
    if labels is None:
        labels = [chr(ord('A') + i) for i in range(len(s.pieces))]
    for idx, p in enumerate(s.pieces):
        ch = labels[idx]
        for dy in range(p.h):
            for dx in range(p.w):
                grid[p.y + dy][p.x + dx] = ch
    return "\n".join("".join(row) for row in grid)

# ----------------------------
# A* 搜索
# ----------------------------
def astar(board, start, h, max_steps=10**6, log_html_path=None):
    t0 = time.time()
    open_heap = []  # (f, g, id, State)
    g_cost = {start: 0}
    parent = {start: None}

    logger = GraphLogger(board)
    counter = 0
    f0 = h(start, board)
    heapq.heappush(open_heap, (f0, 0, counter, start))

    expanded = 0
    goal_state = None

    while open_heap and expanded < max_steps:
        f, g, _, s = heapq.heappop(open_heap)
        expanded += 1
        if (expanded % 10000 == 0):
            print(expanded)
        # ✅ log only expanded node
        logger.add_or_get(s, g=g, f=f, parent_state=parent[s])

        if board.is_goal(s):
            goal_state = s
            break

        for nb in board.neighbors(s):
            ng = g + 1
            if nb not in g_cost or ng < g_cost[nb]:
                g_cost[nb] = ng
                parent[nb] = s
                fnb = ng + h(nb, board)
                counter += 1
                heapq.heappush(open_heap, (fnb, ng, counter, nb))
                # ❌ no logger.add_or_get here anymore

    # reconstruct path if any
    path, cost = None, None
    if goal_state is not None:
        path = []
        cur = goal_state
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        cost = len(path) - 1
        logger.mark_solution_path(goal_state)

    result = {
        "path": path,
        "cost": cost,
        "expanded": expanded,
        "time": time.time() - t0
    }

    if log_html_path is not None:
        logger.write_html(log_html_path)

    return result

# ----------------------------
# GUI 可视化路径 (更通用)
# ----------------------------
def generate_colors(n):
    """生成 n 种可区分的颜色"""
    colors = []
    for i in range(n):
        hue = i / n
        light = 0.6 + 0.2 * random.random()
        sat = 0.6 + 0.3 * random.random()
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

def visualize_path(board, path, cell_size=80, delay=500):
    """
    通用可视化：支持自动播放和手动控制
    - 空格：播放/暂停
    - ← →：前后切换
    - ESC 或 关闭窗口：退出
    """
    pygame.init()
    screen = pygame.display.set_mode((board.W*cell_size, board.H*cell_size))
    clock = pygame.time.Clock()

    colors = generate_colors(len(path[0].pieces))

    step = 0
    running = True
    autoplay = True
    last_update = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    autoplay = not autoplay
                elif event.key == pygame.K_RIGHT:
                    step = min(step + 1, len(path) - 1)
                elif event.key == pygame.K_LEFT:
                    step = max(step - 1, 0)

        now = pygame.time.get_ticks()
        if autoplay and step < len(path)-1 and now - last_update > delay:
            step += 1
            last_update = now

        screen.fill((240, 240, 240))
        state = path[step]

        for idx, p in enumerate(state.pieces):
            rect = pygame.Rect(p.x*cell_size, p.y*cell_size,
                               p.w*cell_size, p.h*cell_size)
            pygame.draw.rect(screen, colors[idx % len(colors)], rect)
            pygame.draw.rect(screen, (0,0,0), rect, 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# 可视化单个状态
def visualize_state(board, state, cell_size=80):
    """
    可视化单个 State，方便检查布局是否正确
    """
    pygame.init()
    screen = pygame.display.set_mode((board.W * cell_size, board.H * cell_size))
    clock = pygame.time.Clock()

    colors = generate_colors(len(state.pieces))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        screen.fill((240, 240, 240))

        for idx, p in enumerate(state.pieces):
            rect = pygame.Rect(p.x * cell_size, p.y * cell_size,
                               p.w * cell_size, p.h * cell_size)
            pygame.draw.rect(screen, colors[idx % len(colors)], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

def visualize_state_with_save(board, state, filename="klotski_levels.txt", name=None, cell_size=80):
    """
    可视化单个 State，带 Save / Discard 按钮。
    - Save: 保存局面并返回 "save"
    - Discard: 不保存并返回 "discard"
    - Esc / 关闭窗口: 返回 None
    """
    pygame.init()
    screen = pygame.display.set_mode((board.W * cell_size, board.H * cell_size + 50))
    clock = pygame.time.Clock()

    colors = generate_colors(len(state.pieces))

    # 按钮区域
    button_save = pygame.Rect(10, board.H * cell_size + 10, 100, 30)
    button_discard = pygame.Rect(120, board.H * cell_size + 10, 100, 30)
    font = pygame.font.SysFont(None, 24)

    result = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_save.collidepoint(event.pos):
                    print("✅ 保存当前局面到", filename)
                    save_board_state_append(filename, board, state, name=name)
                    result = "save"
                    running = False
                elif button_discard.collidepoint(event.pos):
                    print("❌ 丢弃当前局面")
                    result = "discard"
                    running = False

        screen.fill((240, 240, 240))

        # 画棋盘
        for idx, p in enumerate(state.pieces):
            rect = pygame.Rect(p.x * cell_size, p.y * cell_size,
                               p.w * cell_size, p.h * cell_size)
            pygame.draw.rect(screen, colors[idx % len(colors)], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)

        # Save 按钮
        pygame.draw.rect(screen, (180, 255, 180), button_save)
        pygame.draw.rect(screen, (0, 0, 0), button_save, 2)
        text = font.render("Save", True, (0, 0, 0))
        screen.blit(text, (button_save.x + 20, button_save.y + 5))

        # Discard 按钮
        pygame.draw.rect(screen, (255, 180, 180), button_discard)
        pygame.draw.rect(screen, (0, 0, 0), button_discard, 2)
        text = font.render("Discard", True, (0, 0, 0))
        screen.blit(text, (button_discard.x + 10, button_discard.y + 5))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return result
#随机生成
def random_goal_with_blocks(W=4, H=5, hero_size=(2,2), extra_shapes=[(1,2,2),(1,1,2)]):
    # hero 固定在底部出口（默认居中）
    gx = (W - hero_size[0]) // 2
    gy = H - hero_size[1]
    hero = Piece(hero_size[0], hero_size[1], gx, gy)

    pieces = [hero]
    occupied = [[False]*W for _ in range(H)]
    for dy in range(hero.h):
        for dx in range(hero.w):
            occupied[hero.y+dy][hero.x+dx] = True

    # 随机放置其他方块（严格边界 + 重叠检查）
    for (w,h,count) in extra_shapes:
        for _ in range(count):
            placed = False
            tries = 0
            while not placed and tries < 200:
                x = random.randint(0, W-w)
                y = random.randint(0, H-h)
                if x + w > W or y + h > H:
                    tries += 1
                    continue
                overlap = False
                for yy in range(h):
                    for xx in range(w):
                        if occupied[y+yy][x+xx]:
                            overlap = True
                            break
                    if overlap:
                        break
                if not overlap:
                    for yy in range(h):
                        for xx in range(w):
                            occupied[y+yy][x+xx] = True
                    pieces.append(Piece(w,h,x,y))
                    placed = True
                tries += 1

    return State(tuple(pieces)), (gx, gy)

def scramble_from_goal(board, goal_state, steps=50):
    """从终局正向随机走 steps 步，禁止直接回退。"""
    s = goal_state
    prev = None
    for _ in range(steps):
        moves = board.legal_moves(s)
        if not moves:
            break
        if prev is not None:
            inv = (prev[0], -prev[1], -prev[2])  # 撤销上一步的动作
            moves = [m for m in moves if m != inv]
            if not moves:
                break
        mv = random.choice(moves)
        s = board.apply_move(s, mv)  # 注意：正向应用 mv（不再取反）
        prev = mv
    return s

def validate_state(board, state):
    errs = []
    occ = [[-1]*board.W for _ in range(board.H)]
    for idx, p in enumerate(state.pieces):
        if p.x < 0 or p.y < 0 or p.x + p.w > board.W or p.y + p.h > board.H:
            errs.append(f"Piece {idx} OOB: {p}")
            continue
        for dy in range(p.h):
            for dx in range(p.w):
                y = p.y + dy; x = p.x + dx
                if occ[y][x] != -1:
                    errs.append(f"Overlap at {(x,y)} between {occ[y][x]} and {idx}")
                else:
                    occ[y][x] = idx
    return errs

# --------- 序列化 / 反序列化（含 Board 和 State）---------

def _dump_block(board, state, name=None):
    lines = []
    if name:
        lines.append(f"NAME {name}")
    gw, gh = (-1, -1)
    gx, gy = (-1, -1)
    if board.goal_piece is not None:
        gw, gh = board.goal_piece
    if board.goal_pos is not None:
        gx, gy = board.goal_pos
    lines.append(f"BOARD {board.W} {board.H}")
    lines.append(f"GOAL {gw} {gh} {gx} {gy}")
    lines.append(f"PIECES {len(state.pieces)}")
    for p in state.pieces:
        lines.append(f"{p.w} {p.h} {p.x} {p.y}")
    lines.append("")  # 空行分隔
    return "\n".join(lines)

def save_board_state_append(filename, board, state, name="klotski_levels.txt"):
    """
    追加保存一个 (board, state)。若文件不存在则创建并写入 N=1；
    存在则将首行 N+1，并在文件末尾添加一个条目。
    """
    block = _dump_block(board, state, name)
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("1\n")
            f.write(block)
            f.write("\n")
        print(f"新建 {filename} 并保存 1 个条目")
    else:
        # 读出全部，更新首行数量，再整体写回并在末尾追加新块
        with open(filename, "r") as f:
            lines = f.readlines()
        if not lines:
            n = 0
            header = "0\n"
        else:
            try:
                n = int(lines[0].strip())
            except:
                n = 0
            header = str(n+1) + "\n"
        lines[0] = header
        with open(filename, "w") as f:
            f.writelines(lines)
            f.write(block)
            f.write("\n")
        print(f"已追加到 {filename}，现在共有 {n+1} 个条目")

def load_board_states(filename):
    """
    读取文件，返回 [(Board, State, name_or_None), ...]
    可容忍空行与可选 NAME。
    """
    with open(filename, "r") as f:
        raw = [ln.rstrip("\n") for ln in f]

    if not raw:
        return []

    # 读取首行 N（容错：即使不准，也按块解析到 EOF）
    try:
        total = int(raw[0].strip())
    except:
        total = None
    i = 1
    results = []

    def skip_blank(ix):
        while ix < len(raw) and raw[ix].strip() == "":
            ix += 1
        return ix

    while True:
        i = skip_blank(i)
        if i >= len(raw):
            break

        # 可选 NAME
        name = None
        if raw[i].startswith("NAME "):
            name = raw[i][5:].strip()
            i += 1

        # BOARD
        if i >= len(raw) or not raw[i].startswith("BOARD "):
            # 结束或格式异常，跳出
            break
        parts = raw[i].split()
        if len(parts) != 3:
            break
        W, H = int(parts[1]), int(parts[2])
        i += 1

        # GOAL
        if i >= len(raw) or not raw[i].startswith("GOAL "):
            break
        parts = raw[i].split()
        if len(parts) != 5:
            break
        gw, gh, gx, gy = map(int, parts[1:5])
        goal_piece = None if gw < 0 or gh < 0 else (gw, gh)
        goal_pos   = None if gx < 0 or gy < 0 else (gx, gy)
        i += 1

        # PIECES
        if i >= len(raw) or not raw[i].startswith("PIECES "):
            break
        parts = raw[i].split()
        if len(parts) != 2:
            break
        K = int(parts[1])
        i += 1

        pcs = []
        ok = True
        for _ in range(K):
            if i >= len(raw):
                ok = False
                break
            parts = raw[i].split()
            if len(parts) != 4:
                ok = False
                break
            w, h, x, y = map(int, parts)
            pcs.append(Piece(w, h, x, y))
            i += 1
        if not ok:
            break

        # 可选分隔空行
        i = skip_blank(i)

        b = Board(W, H, goal_piece=goal_piece, goal_pos=goal_pos)
        s = State(tuple(pcs))
        results.append((b, s, name))

    print(f"读取 {len(results)} 个条目（文件声明N={total}）")
    return results



# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":

    # 带时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"klotski_run_log_{timestamp}.txt"
    log_file = open(log_filename, "w", encoding="utf-8")

    # 定义一个能同时打印到屏幕和文件的函数
    def log(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file, flush=True)

    width = 4
    height = 5
    saved = 0
    total = 0

    for seed1 in range(8100, 9100, 1):
        log(f"\n===== Seed {seed1} =====")
        random.seed(seed1)
        goal_state, (gx, gy) = random_goal_with_blocks(
            W=width, H=height, hero_size=(2, 2),
            extra_shapes=[(2, 1, 2), (1, 2, 3), (1, 1, 4)]
        )
        board = Board(width, height, goal_piece=(2, 2), goal_pos=(gx, gy))
        start = scramble_from_goal(board, goal_state, steps=842000)

        # === 只跑一遍 A* ===
        result = astar(board, start, corridor_h, max_steps=3500000)
        total += 1

        if result["cost"] is None:
            log(f"❌ Seed {seed1} 无解或未在步数内找到解")
            continue

        log("\nA* 结果：")
        log("cost:", result["cost"])
        log("expanded:", result["expanded"])
        log("time:", result["time"], "s")

        # 步数大于 30 才保存
        if result["cost"] > 30:
            log(f"✅ 保存 Seed {seed1}, cost={result['cost']}")
            save_board_state_append("klotski_levels_30.txt", board, start, name=str(seed1))
            saved += 1
        else:
            log(f"❌ 丢弃 Seed {seed1}, cost={result['cost']} ≤ 30")

    log(f"\n=== 处理完毕: 总共 {total} 个种子，保存 {saved} 个局面 ===")

    log_file.close()
    print(f"\n📄 日志已保存到 {log_filename}")
