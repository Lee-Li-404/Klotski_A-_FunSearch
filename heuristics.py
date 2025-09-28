# ----------------------------
# 启发式函数
# ----------------------------
def trivial_h(s, board):
    # 找到目标方块，计算到目标位置的曼哈顿距离
    for p in s.pieces:
        if (p.w, p.h) == board.goal_piece:
            gx, gy = board.goal_pos
            return abs(p.x - gx) + abs(p.y - gy)
    return 0

def blocking_h(s, board):
    # 找 hero
    hero = None
    for p in s.pieces:
        if (p.w, p.h) == board.goal_piece:
            hero = p
            break
    if hero is None:
        return 0

    gx, gy = board.goal_pos
    dist = max(0, gy - hero.y)  # hero 需要往下的行数

    # 计算出口矩形（hero 要占据的最终位置区域）
    target_rect = (gx, gy, board.goal_piece[0], board.goal_piece[1])

    blocking = 0
    for q in s.pieces:
        if q == hero:
            continue
        # 如果 q 与目标区域重叠，则算阻塞
        if not (q.x + q.w <= target_rect[0] or
                target_rect[0] + target_rect[2] <= q.x or
                q.y + q.h <= target_rect[1] or
                target_rect[1] + target_rect[3] <= q.y):
            blocking += 1

    return dist + blocking

def corridor_h(s, board):
    # 找 hero
    hero = None
    for p in s.pieces:
        if (p.w, p.h) == board.goal_piece:
            hero = p
            break
    if hero is None:
        return 0

    gx, gy = board.goal_pos
    dist = max(0, gy - hero.y)  # hero 要下落的距离

    blocking_rows = 0
    # 检查 hero 下方每一行
    for row in range(hero.y + hero.h, gy + board.goal_piece[1]):
        row_blocked = False
        for q in s.pieces:
            if q == hero:
                continue
            # 判断 q 是否占据 hero 水平范围内的这个 row
            if q.y <= row < q.y + q.h:
                if not (q.x + q.w <= hero.x or q.x >= hero.x + hero.w):
                    row_blocked = True
                    break
        if row_blocked:
            blocking_rows += 1

    return dist + blocking_rows


def admissible_corridor_h(s, board):
    # 找 hero
    hero = None
    for p in s.pieces:
        if (p.w, p.h) == board.goal_piece:
            hero = p
            break
    if hero is None:
        return 0

    gx, gy = board.goal_pos
    dist = max(0, gy - hero.y)  # 至少要下降的行数

    # 找到阻塞方块集合
    blockers = set()
    hero_x1, hero_x2 = hero.x, hero.x + hero.w
    for q in s.pieces:
        if q == hero:
            continue
        # q 是否在 hero 下落路径范围内
        if q.y < gy + board.goal_piece[1] and q.y + q.h > hero.y + hero.h:
            if not (q.x + q.w <= hero_x1 or q.x >= hero_x2):
                blockers.add(q)

    return max(dist, len(blockers))



def blocking_goal_h(s, board):
    hero = None
    for p in s.pieces:
        if (p.w, p.h) == board.goal_piece:
            hero = p
            break
    if hero is None:
        return 0

    gx, gy = board.goal_pos
    dist = max(0, gy - hero.y)

    # 目标区域矩形
    target_rect = (gx, gy, board.goal_piece[0], board.goal_piece[1])

    blocked = False
    for q in s.pieces:
        if q == hero:
            continue
        if not (q.x + q.w <= target_rect[0] or
                target_rect[0] + target_rect[2] <= q.x or
                q.y + q.h <= target_rect[1] or
                target_rect[1] + target_rect[3] <= q.y):
            blocked = True
            break

    return dist + (1 if blocked else 0)


def aggressive_h(s, board):
    # 找 hero
    hero = None
    for p in s.pieces:
        if (p.w, p.h) == board.goal_piece:
            hero = p
            break
    if hero is None:
        return 0

    gx, gy = board.goal_pos
    dist = max(0, gy - hero.y)  # hero 到目标的纵向距离

    hero_x1, hero_x2 = hero.x, hero.x + hero.w
    hero_bottom = hero.y + hero.h

    penalty = 0
    for q in s.pieces:
        if q == hero:
            continue
        # q 是否在 hero 下落路径内
        if q.y < gy + board.goal_piece[1] and q.y + q.h > hero_bottom:
            if not (q.x + q.w <= hero_x1 or q.x >= hero_x2):
                # 基础阻碍
                penalty += 2
                # 如果紧贴 hero 底部 → 更严重
                if q.y == hero_bottom:
                    penalty += 5

    # 计算剩余空格（用来衡量拥挤度）
    occ = sum(p.w * p.h for p in s.pieces)
    empty = board.W * board.H - occ
    crowd_penalty = max(0, 3 - empty) * 2  # 空格 <3 时，额外惩罚

    return dist + penalty + crowd_penalty
