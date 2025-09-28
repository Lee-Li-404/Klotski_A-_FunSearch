from dataclasses import dataclass

# ----------------------------
# 数据结构
# ----------------------------
@dataclass(frozen=True)
class Piece:
    w: int
    h: int
    x: int
    y: int

@dataclass(frozen=True)
class State:
    pieces: tuple
    def key(self):
        return tuple((p.x, p.y) for p in self.pieces)

class Board:
    def __init__(self, W, H, goal_piece=None, goal_pos=None, start_state=None):
        self.W = W
        self.H = H
        self.goal_piece = goal_piece  # (w,h)
        self.goal_pos = goal_pos      # (x,y)
        self.hero_index = None

        # 如果提供了初始状态，就锁定 hero 的索引
        if start_state is not None and goal_piece is not None:
            # 找所有尺寸符合的候选
            candidates = [i for i, p in enumerate(start_state.pieces)
                          if (p.w, p.h) == goal_piece]
            if not candidates:
                self.hero_index = None
            elif len(candidates) == 1:
                self.hero_index = candidates[0]
            else:
                # 如果有多个候选，选一个离 goal_pos 最近的
                if goal_pos is not None:
                    gx, gy = goal_pos
                    self.hero_index = min(
                        candidates,
                        key=lambda i: abs(start_state.pieces[i].x - gx) +
                                      abs(start_state.pieces[i].y - gy)
                    )
                else:
                    self.hero_index = candidates[0]

    def is_goal(self, s):
        if self.goal_piece is None or self.goal_pos is None or self.hero_index is None:
            return False
        hero = s.pieces[self.hero_index]
        return (hero.x, hero.y) == self.goal_pos
