# Combo variant of d35bdbdc (index 1)
# combo_spec: D1/2+D5/2
# Name: 反转匹配+跨层转移
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题: 目标精灵 d 用自己的中心色当键，去找 source.border == d.center 的源精灵，
#      然后把 source.center 写回 d.center。
#      例: d(border=2, center=7), s(border=7, center=4) -> 输出 d(border=2, center=4)。
#    - 本融合题严格按规格改成:
#        match: source.center == dest.border
#        write: source.center -> dest.border
#      若 d(border=2, center=7), s(border=9, center=2)，则匹配成立，因为 2 == d.border。
#      但写回时仍然把 2 涂到 d 的边框上，所以保留下来的目标精灵外观不会变化。
#    - 这说明 D1/2 与 D5/2 在本题对象语义上发生了“代数抵消”:
#      匹配键与写入值都等于目标边框色，目标的可见变化坍缩为零；唯一直接可见的输出变化
#      只剩下删除未连到灰色路径的精灵。
#
# b) 复杂度校准
#    - 目标仍然由灰色路径两端决定，保留原题的空间推理负担。
#    - 为了让融合规则不是“样本恰好没用上”的伪退化，生成器强制两件事同时成立:
#      1. 每个目标边框色都恰好有一个非目标 source.center 与之相等，保证融合匹配真实存在。
#      2. 每个目标中心色都恰好指向另一个源精灵的 border，使同一输入在原题规则下会改色；
#         这样可以量化证明“融合输出 != 原题输出”。
#    - 具体做法是固定 2 个源精灵，因为两个目标边框色彼此不同，必须各自有唯一中心色匹配源。
#      目标中心色则用这两个源精灵的 border 色交叉填入，避免与融合匹配重合。
#
# c) 隐式假设审查
#    - 路径端点找目标的几何规则不变，相关代码可以原样沿用。
#    - 原题的查找表是 by_border；融合后必须改成 by_center，且键来自 dest.border。
#    - D5/2 把写入位置从中心改成边框，因此 solver 不能再只写 center_pos，必须能重涂
#      目标精灵的所有非中心像素。
#    - 但由于匹配值与写入值都是 dest.border，本题的重涂在数值上是幂等操作。solver 仍然
#      必须显式执行这一步，而不是偷懒“直接不改”，否则无法保证输入里真的存在符合融合
#      规则的源精灵，也无法用生成器做反退化验证。
#    - 新假设: 每个目标边框色至多对应一个非目标 source.center。生成器通过唯一赋值和
#      干扰项避让来保证。
#
# d) 放置合法性
#    - 继续使用 output-first。输出先放两个目标精灵和灰色路径；输入再额外加入 2 个源精灵
#      和可选 1 个干扰精灵。
#    - 所有精灵仍是 3x3 box 或 plus，且通过 `_overlaps` 保证不重叠、不 4 邻接粘连。
#    - 路径端口仍要求离中心曼哈顿距离 2，且中间格必须是目标边框色；因为目标边框色在输入
#      和输出中都不变，所以端口合法性不会被融合规则破坏。
#
# e) 视觉可观察性
#    - 这个组合按字面实现后，保留下来的目标精灵不会发生可见改色；从纯 I/O 视角看，最直观
#      的规则就是“保留灰路径连接的两个精灵，删除其余精灵”。
#    - 因而这里能做的最佳约束不是强行捏造额外可见效果，而是保证融合匹配真实存在，并且用
#      反事实检查证明同一输入在原题规则下会产生不同输出，避免训练样本退化成“随便删一删”。
#
# f) 可视化整洁度
#    - 不额外叠加标记，也不改变原题的路径/精灵布局骨架；输出仍然只含 2 个目标精灵和灰路径。
#    - 源与干扰精灵都在输入中远离目标与路径，避免颜色重复造成局部视觉糊成一团。
#
# g) 变化的可推断性
#    - 颜色自由度仍限制在 ARC 的 1-9（去掉灰色 5），目标数量固定为 2，源数量固定为 2。
#    - 输入中每个保留目标的边框色都会在某个被删除源精灵的中心再次出现，形成“中心色对边框色”
#      的重复线索；同时目标中心色又交叉复用了两个源的边框色，保证原题反事实输出会不同。
#
# h) 量化反退化指标
#    - fused_matches: 两个目标中有多少个真的满足 `exists unique source.center == dest.border`。
#      目标值固定为 2。
#    - old_rule_disagreement: 同一输入下，原题输出与融合输出在两个目标中心上有多少处不同。
#      目标值固定为 2。
#    - d5_only_disagreement: 同一输入下，D5-only (`source.border == dest.center`, 写边框) 输出与
#      融合输出在两个目标边框上有多少处不同。目标值固定为 2。
#
# i) 新 solver 假设与 generator 保证
#    - solver 假设每个目标边框色最多命中一个非目标 source.center；generator 明确避免重复中心键。
#    - solver 假设目标精灵在输入中仍是合法的双色 bordered sprite；generator 只改颜色不改几何。
#    - solver 假设灰路径端点仍能唯一确定两个目标；generator 保持原题相同的端口与路径生成规则。

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_trail_endpoints(
    g: np.ndarray,
    grey: int,
    sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the 2 endpoints of the grey one-stroke trail under 8-connectivity."""
    h, w = g.shape
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        border_color = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == border_color:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    n = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == n:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
    return path[0], path[-1]


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    This also handles clipped/partial borders near grid edges.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color in (bg, grey) or center_color == grey:
            continue

        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue

        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _ports_for_sprite(
    sprite: dict,
    g: np.ndarray,
    eps: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    A grey endpoint is a port if it is 2 steps from the center and the intermediate cell is border-colored.
    """
    cr, cc = sprite["center_pos"]
    border_color = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == border_color:
            ports.append((er, ec))
    return ports


def _select_destinations(g: np.ndarray, grey: int, sprites: list[dict]) -> list[dict]:
    """Use the two trail endpoints to pick the two destination sprites."""
    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[te1]) <= len(opts[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))

    assert all(d is not None for d in dest), "expected exactly 2 destination sprites from trail endpoints"
    return dest


def _cleanup_to_destinations(
    g: np.ndarray,
    sprites: list[dict],
    dest: list[dict],
    bg: int,
    grey: int,
) -> np.ndarray:
    """Keep only destination components plus grey path."""
    dest_ids = {id(sp) for sp in dest}
    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc]:
                    continue
                if out[nr, nc] == bg or out[nr, nc] == grey:
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))

    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg
    return out


def _repaint_border(out: np.ndarray, sprite: dict, color: int) -> None:
    """Recolor every visible non-center cell of a bordered sprite."""
    center_pos = sprite["center_pos"]
    for r, c in sprite["pts"]:
        if (int(r), int(c)) != center_pos:
            out[r, c] = int(color)


def _fused_rule_output(grid: np.ndarray) -> np.ndarray:
    """Shared implementation for the requested combo behavior."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    dest = _select_destinations(g, grey, sprites)
    dest_ids = {id(sp) for sp in dest}

    by_center: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        by_center.setdefault(int(sp["center"]), []).append(sp)

    out = _cleanup_to_destinations(g, sprites, dest, bg, grey)

    for d in dest:
        want_center = int(d["border"])
        srcs = by_center.get(want_center, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination border should match at most one source center"
        src = srcs[0]
        assert int(src["center"]) == int(d["border"])
        _repaint_border(out, d, int(src["center"]))

    return out


def _original_rule_output(grid: np.ndarray) -> np.ndarray:
    """Reference implementation of the original task, used for anti-degeneration checks."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    dest = _select_destinations(g, grey, sprites)

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = _cleanup_to_destinations(g, sprites, dest, bg, grey)

    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination center should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


def _d5_only_output(grid: np.ndarray) -> np.ndarray:
    """Single-variant D5/2 reference for behavior validation."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    dest = _select_destinations(g, grey, sprites)

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = _cleanup_to_destinations(g, sprites, dest, bg, grey)

    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination center should match at most one source border"
        src = srcs[0]
        _repaint_border(out, d, int(src["center"]))

    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the fused matching structure is real and distinguishable from counterfactual rules."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5
    sprites = _detect_bordered_sprites(g, bg, grey)
    dest = _select_destinations(g, grey, sprites)
    dest_ids = {id(sp) for sp in dest}

    by_center: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        by_center.setdefault(int(sp["center"]), []).append(sp)

    fused_matches = sum(1 for d in dest if len(by_center.get(int(d["border"]), [])) == 1)

    fused = _fused_rule_output(g)
    old = _original_rule_output(g)
    d5 = _d5_only_output(g)

    center_diffs = 0
    border_diffs = 0
    for d in dest:
        dr, dc = d["center_pos"]
        if int(fused[dr, dc]) != int(old[dr, dc]):
            center_diffs += 1
        for r, c in d["pts"]:
            if (int(r), int(c)) == d["center_pos"]:
                continue
            if int(fused[r, c]) != int(d5[r, c]):
                border_diffs += 1
                break

    return {
        "fused_matches": int(fused_matches),
        "old_rule_disagreement": int(center_diffs),
        "d5_only_disagreement": int(border_diffs),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the two path-linked bordered sprites and apply the fused reversed-match, border-write rule.

    Input structure:
    - The grid contains background, a grey one-stroke trail, and several small bordered sprites.
    - Each sprite has one repeated border color and a single center pixel of another color.
    - The two grey trail endpoints touch exactly two destination sprites through 2-step ports; the other
      sprites are candidate sources or distractors.

    Rule:
    - Use the trail endpoints to identify the two destination sprites and delete every non-destination sprite.
    - For each destination, find the unique non-destination source whose center color equals that
      destination's border color.
    - Transfer that source center color onto the destination border, leaving the destination center untouched.
      If no source matches, keep the destination unchanged.

    Key insight:
    - Matching now reads source centers against destination borders, and the transferred layer is also the
      border. In this task family that creates an idempotent repaint, so the solver still has to recover the
      hidden center-to-border correspondence even though the visible kept sprites do not change color.
    """
    return _fused_rule_output(grid)


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
NUM_SOURCES = 2
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return 3x3 patch for a box or plus. Center is at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent background and clipping at edges."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo `_place_sprite` by clearing the patch's non-background cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and int(patch[dr, dc]) != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap or 4-neighbor touching with existing non-bg, non-grey cells."""
    h, w = grid.shape
    patch_cells: set[tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            if int(patch[dr, dc]) == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if int(grid[rr, cc]) != BG:
                    return True
                patch_cells.add((rr, cc))

    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid top-left placements for a 3x3 patch whose center remains on-grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check whether adding a path cell would make the 8-neighbor trail too thick."""
    nbs = [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (r + dr, c + dc) in occupied
    ]
    if len(nbs) > MAX_THICKNESS:
        return True
    for nr, nc in nbs:
        deg = sum(
            1
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (nr + dr, nc + dc) in occupied
        )
        if deg + 1 > MAX_THICKNESS:
            return True
    return False


def _gen_winding_path(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    target_len: int,
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int, bool]) -> int:
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

    for _ in range(target_len + 20):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < GRID_H
                    and 0 <= nc < GRID_W
                    and (nr, nc) not in visited
                    and int(grid[nr, nc]) == BG
                    and not _too_thick(nr, nc, visited)
                ):
                    moves.append((nr, nc, dr != 0 and dc != 0))

        if not moves:
            return None

        dist = abs(cr - end[0]) + abs(cc - end[1])
        budget = target_len - len(path)
        diag = [m for m in moves if m[2]]
        away = [m for m in moves if score(m) > dist]

        if budget > dist + 4:
            if away and random.random() < PATH_WANDER_AWAY_P:
                pool = away
            elif diag and random.random() < PATH_WANDER_DIAG_P:
                pool = diag
            else:
                pool = moves
            chosen = random.choice(pool)
        elif budget > dist + 1:
            pool = diag if diag and random.random() < PATH_WANDER_DIAG_P else moves
            if random.random() < PATH_MODERATE_RANDOM_P:
                chosen = random.choice(pool)
            else:
                chosen = min(pool, key=score)
        else:
            best = sorted(moves, key=score)
            top = [m for m in best if score(m) <= score(best[0]) + 1]
            dtop = [m for m in top if m[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if not _too_thick(end[0], end[1], visited):
            path.append(end)
        else:
            return None

    return path


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    """Check 8-connectivity via BFS."""
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
    return len(seen) == len(cell_set)


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int) -> tuple[int, int] | None:
    """Find a valid endpoint port direction for a sprite."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == border_color and int(grid[er, ec]) == BG:
            return (er, ec)
    return None


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict[str, list[int]] | None:
    """Assign colors so the fused rule is real and counterfactual rules disagree."""
    remaining = list(range(2, num_sprites))
    if len(remaining) < NUM_SOURCES:
        return None

    source_indices = random.sample(remaining, NUM_SOURCES)
    distractor_indices = [idx for idx in remaining if idx not in source_indices]

    # Fusion sources: source.center matches the corresponding destination border.
    source_center_colors = [border_colors[0], border_colors[1]]

    # Counterfactual original/D5 rules use destination.center as the key; cross-wire that key.
    dest_center_colors = [border_colors[source_indices[1]], border_colors[source_indices[0]]]

    distractor_center_colors: list[int] = []
    banned_center_keys = {border_colors[0], border_colors[1]}
    for di in distractor_indices:
        candidates = [c for c in range(1, 10) if c not in {GREY, border_colors[di]} and c not in banned_center_keys]
        if not candidates:
            return None
        distractor_center_colors.append(random.choice(candidates))

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
    }


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    dest_border_colors: list[int],
    dest_center_colors: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    """Place the two destination sprites on the output grid."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions: list[tuple[int, int]] = []
    first_port = None

    for i in range(2):
        patch = _make_sprite_patch(shape_type, dest_border_colors[i], dest_center_colors[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1

            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc, dest_border_colors[i])
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue

            dest_center_positions.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            return None

    return dest_center_positions


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    palette = [c for c in range(1, 10) if c != GREY]
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(palette, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_border_colors = [border_colors[0], border_colors[1]]
    dest_center_colors = colors["dest_center_colors"]
    result = _place_dests(output_grid, shape_type, dest_border_colors, dest_center_colors, min_port_dist)
    if result is None:
        return None
    dest_center_positions = result

    port_cells = []
    for i in range(2):
        cr, cc = dest_center_positions[i]
        port = _find_port_dir(output_grid, cr, cc, dest_border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(output_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if int(output_grid[r, c]) != BG:
            return None
        output_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(output_grid[nr, nc]) == GREY
        )
        if deg > 1:
            return None

    input_grid = output_grid.copy()
    sprites_to_place = [
        (border_colors[si], colors["source_center_colors"][j])
        for j, si in enumerate(colors["source_indices"])
    ] + [
        (border_colors[di], colors["distractor_center_colors"][j])
        for j, di in enumerate(colors["distractor_indices"])
    ]

    all_pos = _sprite_positions()
    for border_color, center_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, border_color, center_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if not _overlaps(input_grid, patch, r0, c0):
                _place_sprite(input_grid, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    try:
        fused_out = main(input_grid)
        if not np.array_equal(fused_out, output_grid):
            return None
        metrics = _behavior_metrics(input_grid)
        if metrics["fused_matches"] != 2:
            return None
        if metrics["old_rule_disagreement"] != 2:
            return None
        if metrics["d5_only_disagreement"] != 2:
            return None
    except Exception:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
