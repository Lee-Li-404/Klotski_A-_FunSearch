from board_state_class import Piece, State, Board
# ----------------------------
# Demo: 标准华容道局面
# ----------------------------
def standard_start():
    board = Board(4, 5, goal_piece=(2,2), goal_pos=(1,3))
    hero  = Piece(2, 2, 1, 0)
    left  = Piece(1, 2, 0, 0)
    right = Piece(1, 2, 3, 0)
    h1    = Piece(2, 1, 1, 2)
    h2    = Piece(2, 1, 1, 3)
    s1    = Piece(1, 1, 0, 3)
    s2    = Piece(1, 1, 3, 3)
    s3    = Piece(1, 1, 1, 4)
    s4    = Piece(1, 1, 2, 4)
    state = State((hero, left, right, h1, h2, s1, s2, s3, s4))
    return board, state

def tiny_start():
    # 3x3 棋盘, 目标是把 2x2 方块从 (0,0) 移到 (0,1)
    board = Board(3, 3, goal_piece=(2,2), goal_pos=(0,1))
    hero  = Piece(2, 2, 0, 0)
    s1  = Piece(1, 1, 0, 2)

    state = State((hero,s1))
    return board, state

def cao_cao_start():
    # 正统“曹操出逃”布局
    board = Board(4, 5, goal_piece=(2,2), goal_pos=(1,3))

    # 曹操（2×2）
    hero = Piece(2, 2, 1, 0)

    # 四个 1×2 竖块
    v1 = Piece(1, 2, 0, 0)
    v2 = Piece(1, 2, 3, 0)
    v3 = Piece(1, 2, 0, 2)
    v4 = Piece(1, 2, 3, 2)

    # 一个 2×1 横块
    h1 = Piece(2, 1, 1, 2)

    # 四个 1×1 小兵
    s1 = Piece(1, 1, 1, 3)
    s2 = Piece(1, 1, 2, 3)
    s3 = Piece(1, 1, 0, 4)
    s4 = Piece(1, 1, 3, 4)

    state = State((hero, v1, v2, v3, v4, h1, s1, s2, s3, s4))
    return board, state

def simple30_start():
    # 一个简化华容道局面，大约 30 步解
    board = Board(4, 5, goal_piece=(2,2), goal_pos=(1,3))

    # 曹操
    hero = Piece(2, 2, 1, 0)

    # 两个竖直块
    v1 = Piece(1, 2, 0, 0)
    v2 = Piece(1, 2, 3, 0)

    # 一个横块
    h1 = Piece(2, 1, 1, 2)

    # 两个小兵
    s1 = Piece(1, 1, 0, 3)
    s2 = Piece(1, 1, 3, 3)

    state = State((hero, v1, v2, h1, s1, s2))
    return board, state

