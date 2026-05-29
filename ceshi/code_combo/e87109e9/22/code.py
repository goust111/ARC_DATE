# Combo variant of e87109e9 (index 22)
# combo_spec: D2/2+D7/2
# Name: 单像素障碍加形状编码（退化）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - 删除原题顶部图例，输入只保留主网格。
#     - 障碍物从大矩形退化为 3-4 个互不重叠的单像素。
#     - 原本 D2/2 想让“障碍物自身形状”编码转向；但 1x1 像素没有宽高差，
#       所以把编码继续退化成像素坐标关系：
#         * 障碍像素满足 row > col -> 顺时针转弯
#         * 障碍像素满足 row < col -> 逆时针转弯
#       row == col 的对角线像素会让规则失去判别力，因此生成器完全禁用。
#   concrete_objects:
#     - 仍保留一个实心青色(8)正方形作为光束源，尺寸 2x2 或 3x3。
#     - 主网格背景为单色，障碍物颜色互异，但颜色不再承载语义。
#     - 障碍物都是单像素，所以“碰撞”来自方形光束下一步的 sz x sz 脚印盖到该像素。
#   behavior_trace:
#     old_logic:
#       - 北/东/南/西四束光先沿直线移动。
#       - 下一步若撞上某色矩形障碍物，就去顶部图例查同色竖条在左还是右，再决定左转或右转。
#     fused_logic:
#       - 四束光仍按原题方式移动，但输入里没有图例，障碍物也只有单像素。
#       - 当下一步脚印第一次盖到某个障碍像素 (r, c) 时，不再看颜色或形状，
#         直接比较该像素的行列号：
#           row > col 视作“横向占优”的退化代理 -> 顺时针
#           row < col 视作“纵向占优”的退化代理 -> 逆时针
#       - 例如同为 1x1 障碍，放在 (14, 6) 会触发顺时针，而放在 (6, 14) 会触发逆时针。
#     difference:
#       - 决策来源从“图例中的颜色查表”变成“障碍像素相对主对角线的位置”。
#       - 决策对象从大矩形退化为单像素，因此真正有用的信息不再是可见形状，而是绝对位置。
#   complexity_calibration:
#     - 保留原题的四向发射、3-4 个障碍、以及恰好一次二次碰撞，避免退化成四条直线。
#     - 每题都强制至少命中一个 row > col 障碍和一个 row < col 障碍；
#       如果所有命中障碍都落在对角线同一侧，人类几乎无法从少量样例区分“全左转/全右转”。
#     - 生成器还要求障碍离主对角线至少 2 格，让“在对角线哪一侧”更容易肉眼看出来。
#   implicit_assumption_review:
#     - 原 main() 依赖“颜色 5 只出现在顶部图例”，因此会先裁掉 legend；这里该前提失效，必须整段删除。
#     - 原 main() 只需要 obstacle color -> turn 的映射；现在必须解析每个障碍像素的绝对坐标。
#     - 原代码默认障碍物是大片矩形，光束脚印撞上后只会看到一种障碍色；
#       1x1 障碍更容易在同一个 sz x sz 脚印里出现多个像素，因此生成器要显式避免这种近距离摆放。
#     - 原 D2 变体默认“每色对应一个非方形矩形”；融合后新前提变成“每色恰有一个像素，且不在 row == col 上”。
#   placement_legality:
#     - 每个障碍像素都必须位于某束初始光线的命中通道内，保证该方向的首碰是可达的。
#     - 二次碰撞链通过几何相容性构造：首碰像素放得足够靠近光源，使转向后的 sz 宽通道能扫到目标像素。
#     - 障碍像素互不重叠，且任意两点不会同时落入同一个 sz x sz 脚印中；最终再用模拟器验算整张图。
#   visual_observability:
#     - 所有障碍外观完全相同，解题者只能依赖位置归纳规则，因此样本必须同时展示主对角线两侧的转向差异。
#     - 由于颜色不编码方向，重复出现的“对角线下方点右转 / 对角线上方点左转”就是核心可见结构。
#   neatness:
#     - 去掉 legend 后画面更干净，只剩光源和稀疏障碍点。
#     - 通过限制障碍间距，避免多像素堆在同一局部区域造成“看不清到底撞了谁”。
#   main_assumptions:
#     - 背景色是全图出现次数最多的颜色。
#     - 全图恰有一个实心青色正方形光源。
#     - 每个障碍颜色只形成一个 4 连通单像素，且其坐标满足 row != col。
#     - 任一步的光束脚印至多覆盖一种障碍颜色，否则输入被视为非法。
#   generator_guarantees:
#     - 障碍颜色互异，且每个障碍都至少被命中一次。
#     - 总碰撞数固定为 obstacle_count + 1，即恰有一次二次碰撞。
#     - 命中障碍中必定同时包含 row > col 和 row < col 两类。
#     - 用新 main() 与独立模拟器做 oracle 双检，确保 solver/generator 一致。


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if hh != ww:
            continue
        if int(cells.shape[0]) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), hh))
    assert len(candidates) == 1, "grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _turn_from_position(r: int, c: int) -> str:
    assert r != c, "diagonal obstacle pixels do not encode a turn"
    return "R" if r > c else "L"


def _infer_obstacle_turns(g: np.ndarray, bg: int, teal: int = 8) -> dict[int, str]:
    """Infer each obstacle color's turn from its unique pixel position."""
    turns: dict[int, str] = {}
    obstacle_colors = sorted(set(map(int, np.unique(g))) - {bg, teal})
    assert obstacle_colors, "at least one obstacle must be present"

    for color in obstacle_colors:
        comps = list(components_mask(g == color, connectivity=4))
        assert len(comps) == 1, "each obstacle color must form one component"
        cells = comps[0]
        assert int(cells.shape[0]) == 1, "each obstacle must be a single pixel"
        r, c = map(int, cells[0])
        turns[color] = _turn_from_position(r, c)
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, turning by single-pixel obstacle position.

    Input structure:
    - A single main grid with one dominant background color, one solid teal (8) square, and 3-4 isolated colored obstacle pixels.
    - There is no legend region, and every obstacle is exactly one cell large.

    Rule:
    - Launch one square-width teal beam in each cardinal direction from the teal square.
    - A beam paints teal only onto background cells as its sz x sz footprint slides.
    - If the next footprint would include an obstacle pixel at (r, c), do not advance into that step; instead compare row and column:
      row > col means turn clockwise, row < col means turn counterclockwise.
    - Continue from the current location after turning, and stop once the next step would leave the grid.

    Key insight: The removed legend and vanished shape cue are both replaced by a positional code:
    isolated pixels below the main diagonal force clockwise turns, while pixels above it force counterclockwise turns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.size, "input grid must be non-empty"

    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[counts.argmax()])
    teal = 8
    sr, sc, sz = _find_teal_square(g, teal=teal)
    obstacle_turn = _infer_obstacle_turns(g, bg=bg, teal=teal)

    out = g.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(teal)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert oc in obstacle_turn, "beam hit an unknown obstacle"
            d = _rot(d, obstacle_turn[oc])

    return out


# --------------- generator ---------------

OBSTACLE_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
BG_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5
OBS_COUNT_4_PROB = 0.65

DIAG_MARGIN = 2
MAX_RETRIES = 400
LAYOUT_TRIES = 90


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Simulate all four beams, returning output and per-beam hit records."""
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(out[r : r + sz, c : c + sz])))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)
            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                paint(r, c)
                continue
            beam_hits.append((d, obs, nr, nc))
            assert obs in turns
            d = _rot(d, turns[obs])
            assert len(beam_hits) <= 2
        hits.append(beam_hits)

    return out, hits


def _first_hit_candidates(
    side: int,
    turn: str,
    sr: int,
    sc: int,
    sz: int,
    h: int,
    w: int,
    *,
    near: bool,
) -> list[tuple[int, int]]:
    """Candidate pixel coordinates that the initial beam from `side` would hit first."""
    coords: list[tuple[int, int]] = []

    if side == _N:
        row_lo = max(0, sr - sz) if near else 0
        row_hi = sr - 1
        rows = range(row_lo, row_hi + 1)
        cols = range(sc, sc + sz)
    elif side == _S:
        row_lo = sr + sz
        row_hi = min(h - 1, sr + 2 * sz - 1) if near else h - 1
        rows = range(row_lo, row_hi + 1)
        cols = range(sc, sc + sz)
    elif side == _W:
        col_lo = max(0, sc - sz) if near else 0
        col_hi = sc - 1
        rows = range(sr, sr + sz)
        cols = range(col_lo, col_hi + 1)
    else:  # _E
        col_lo = sc + sz
        col_hi = min(w - 1, sc + 2 * sz - 1) if near else w - 1
        rows = range(sr, sr + sz)
        cols = range(col_lo, col_hi + 1)

    for r in rows:
        for c in cols:
            if abs(r - c) < DIAG_MARGIN:
                continue
            if _turn_from_position(r, c) != turn:
                continue
            coords.append((r, c))

    random.shuffle(coords)
    return coords


def _post_hit_state(side: int, coord: tuple[int, int], sr: int, sc: int, sz: int) -> tuple[int, int]:
    """Top-left beam state immediately before the obstacle-triggered turn."""
    r, c = coord
    if side == _N:
        return r + 1, sc
    if side == _S:
        return r - sz, sc
    if side == _W:
        return sr, c + 1
    return sr, c - sz


def _chain_compatible(
    src_side: int,
    src_coord: tuple[int, int],
    dst_side: int,
    dst_coord: tuple[int, int],
    sr: int,
    sc: int,
    sz: int,
) -> bool:
    """Return True iff turning at src_coord can make the beam hit dst_coord next."""
    r0, c0 = _post_hit_state(src_side, src_coord, sr, sc, sz)
    rr, cc = dst_coord
    if dst_side == _N:
        return c0 <= cc <= c0 + sz - 1 and rr < r0
    if dst_side == _S:
        return c0 <= cc <= c0 + sz - 1 and rr > r0 + sz - 1
    if dst_side == _W:
        return r0 <= rr <= r0 + sz - 1 and cc < c0
    return r0 <= rr <= r0 + sz - 1 and cc > c0 + sz - 1


def _same_footprint(a: tuple[int, int], b: tuple[int, int], sz: int) -> bool:
    """Conservative spacing rule: two points too close could occupy one beam footprint."""
    return abs(a[0] - b[0]) < sz and abs(a[1] - b[1]) < sz


def _build_base(
    h: int,
    w: int,
    bg: int,
    sr: int,
    sc: int,
    sz: int,
    obstacles: dict[int, tuple[int, int]],
) -> np.ndarray:
    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    for color, (r, c) in obstacles.items():
        base[r, c] = color
    return base


def _choose_present_sides(chain_src: int, chain_dst: int, obstacle_count: int) -> set[int]:
    present = {chain_src, chain_dst}
    others = [s for s in (_N, _E, _S, _W) if s not in present]
    random.shuffle(others)
    while len(present) < obstacle_count:
        present.add(others.pop())
    return present


def _completion_attempt(
    h: int,
    w: int,
    sr: int,
    sc: int,
    sz: int,
    present: set[int],
    fixed_turns: dict[int, str],
    fixed_coords: dict[int, tuple[int, int]],
    all_cands: dict[int, dict[str, list[tuple[int, int]]]],
) -> np.ndarray | None:
    missing_side = next(iter(set(range(4)) - present)) if len(present) == 3 else None
    remaining = [s for s in sorted(present) if s not in fixed_coords]

    for _ in range(LAYOUT_TRIES):
        side_turn = dict(fixed_turns)
        side_coord = dict(fixed_coords)
        used = list(side_coord.values())
        ok = True

        order = sorted(
            remaining,
            key=lambda s: len(all_cands[s]["L"]) + len(all_cands[s]["R"]),
        )

        for side in order:
            turn_options = [t for t in ("L", "R") if all_cands[side][t]]
            if not turn_options:
                ok = False
                break

            desired_missing = {"L", "R"} - set(side_turn.values())
            preferred: list[str] = []
            if desired_missing:
                preferred.extend([t for t in turn_options if t in desired_missing])
            if missing_side is not None:
                preferred.extend(
                    [t for t in turn_options if _rot(side, t) == missing_side and t not in preferred]
                )
            preferred.extend([t for t in turn_options if t not in preferred])

            placed = False
            for turn in preferred:
                coords = [
                    p for p in all_cands[side][turn]
                    if all(not _same_footprint(p, q, sz) for q in used)
                ]
                random.shuffle(coords)
                for coord in coords[:40]:
                    side_turn[side] = turn
                    side_coord[side] = coord
                    used.append(coord)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break

        if not ok or set(side_turn.values()) != {"L", "R"}:
            continue

        obstacle_colors = random.sample(OBSTACLE_COLOR_POOL, len(present))
        side_color = dict(zip(sorted(present), obstacle_colors))
        bg = random.choice([c for c in BG_COLOR_POOL if c not in obstacle_colors])
        color_obstacles = {side_color[s]: side_coord[s] for s in present}
        turns = {side_color[s]: side_turn[s] for s in present}
        base = _build_base(h, w, bg, sr, sc, sz, color_obstacles)

        try:
            inferred = _infer_obstacle_turns(base, bg=bg, teal=TEAL)
            expected, hits = _simulate_paths(base, inferred, bg)
        except AssertionError:
            continue

        hit_counts = [len(beam) for beam in hits]
        hit_colors = {color for beam in hits for _, color, _, _ in beam}
        if sum(hit_counts) != len(present) + 1 or any(c > 2 for c in hit_counts):
            continue
        if set(color_obstacles) - hit_colors:
            continue
        if {inferred[c] for c in hit_colors} != {"L", "R"}:
            continue
        if inferred != turns:
            continue

        try:
            actual = main(base)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue

        return base

    return None


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3

    min_r, max_r = SQUARE_EDGE_GAP, h - sz - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - sz - SQUARE_EDGE_GAP
    if min_r > max_r or min_c > max_c:
        return None

    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    all_cands = {
        side: {
            turn: _first_hit_candidates(side, turn, sr, sc, sz, h, w, near=False)
            for turn in ("L", "R")
        }
        for side in range(4)
    }
    near_cands = {
        side: {
            turn: _first_hit_candidates(side, turn, sr, sc, sz, h, w, near=True)
            for turn in ("L", "R")
        }
        for side in range(4)
    }

    chain_options: list[tuple[int, str, int, str, tuple[int, int], tuple[int, int]]] = []
    for src_side in range(4):
        for src_turn in ("L", "R"):
            dst_side = _rot(src_side, src_turn)
            for src_coord in near_cands[src_side][src_turn]:
                for dst_turn in ("L", "R"):
                    for dst_coord in all_cands[dst_side][dst_turn]:
                        if _same_footprint(src_coord, dst_coord, sz):
                            continue
                        if _chain_compatible(src_side, src_coord, dst_side, dst_coord, sr, sc, sz):
                            chain_options.append(
                                (src_side, src_turn, dst_side, dst_turn, src_coord, dst_coord)
                            )

    if not chain_options:
        return None

    random.shuffle(chain_options)
    for src_side, src_turn, dst_side, dst_turn, src_coord, dst_coord in chain_options[:80]:
        present = _choose_present_sides(src_side, dst_side, obstacle_count)
        fixed_turns = {src_side: src_turn, dst_side: dst_turn}
        fixed_coords = {src_side: src_coord, dst_side: dst_coord}
        base = _completion_attempt(h, w, sr, sc, sz, present, fixed_turns, fixed_coords, all_cands)
        if base is not None:
            return base

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
