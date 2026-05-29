# Combo variant of d35bdbdc (index 2)
# combo_spec: D1/0+D5/1
# Name: 中心色匹配+双向交换
#
# Design analysis
# a) 具体化与行为追踪:
# - 原题里，灰色路径两端选出 2 个 destination；每个 destination 用自己的中心色去找
#   一个 source，使得 `source.border == dest.center`，然后只把 `source.center` 单向写进
#   destination，source 在输出里被整体删除。
# - D1/0 把匹配键改成 `source.center == dest.center`。如果再把 D5/1 的“中心色互换”
#   直接叠上去，那么一对精灵本来就共享同一个中心色，字面 swap 会变成完全无变化。
# - 因此本融合版把“配对条件”和“交换载荷”拆开：先用同中心色完成配对，再把这对精灵
#   的中心同步改成“对方的边框色”。示例：destination = (border 2, center 7)，
#   source = (border 8, center 7)，输出变成 destination.center = 8，
#   source.center = 2。共享中心色 7 只负责命名配对，不是最终写回的颜色。
# - 这样既保留了 D1 的中心色对中心色匹配，也保留了 D5 的双向、同步、两端都改变。
#
# b) 复杂度校准:
# - 双向交换要求一对一配对；原题允许“一 source 服务两个 destination”的分支在这里会
#   产生顺序歧义，所以生成器固定为 2 个 source、2 个 destination 的一对一结构。
# - 两个配对键颜色都取自“非任何边框色”的颜色池，并且互不相同。这样不会意外满足原题
#   的 `dest.center == source.border`，也避免两个 source 同时命中同一个键。
# - distractor 的中心色也禁止复用这两个配对键，防止中心色匹配出现第二个候选。
#
# c) 隐式假设审查:
# - 原题可以把全部 sprite 都放进 `by_border`，因为 destination 不会用自己的中心色
#   匹配自己。现在按 center 建索引时，如果不排除 destination，它会先匹配到自己。
# - 原题先删掉全部非 destination 再改色；融合后 source 也属于输出的一部分，必须先
#   求出配对，再只删除未参与交换的 sprite。
# - 双向更新不能原地逐个写，否则前一对的写回可能污染后一对读取；因此必须基于输入态
#   先记录所有 pair，再做同步赋值。
# - 新 solver 额外假设：每个 destination 最多命中一个非 destination source，且两个
#   destination 不会复用同一个 source。生成器会显式保证这点。
#
# d) 放置合法性:
# - 仍沿用原题的 10x10 网格、3x3 box/plus 精灵、灰色单笔 8 连通路径、2 步端口规则。
# - 生成器继续使用 output-first：先放 2 个 destination 和灰路径，再放输出态的 source，
#   最后从 output 回退出 input 并补上 distractor。这样路径空间最大，成功率更稳定。
# - 所有 sprite 仍通过 `_overlaps()` 约束为不重叠、且不与其他前景 4 邻接粘连。
#
# e) 视觉可观察性:
# - 人类可见线索是：灰路径只标出哪两个是 destination；输入中每个 destination 和远处
#   一个 source 共享中心色；输出时这对精灵都保留，但共享中心色同时消失，改成彼此的
#   边框色。这个规律能从 2-4 个样例中直接肉眼比较出来。
# - 全过程不使用颜色编号大小、扫描顺序之类不可见因素做选择；歧义只靠空间端口和颜色
#   相等关系解决。
#
# f) 可视化整洁度:
# - 输出只保留 4 个关键对象：2 个 destination、2 个 matched source、灰路径。若总
#   sprite 数为 5，则多出的 1 个 distractor 会被删除，画面不会过满。
# - 交换后的 source 仍是清晰的双色 box/plus，不会像“变纯色”那样过于接近背景逻辑，
#   也不会与 destination 混淆。
#
# g) 变化的可推断性:
# - 两个输入配对键颜色在同一张图里各自重复两次，给出足够的“谁和谁成对”线索。
# - 输出不再保留这些键颜色，而是改成成对对象彼此的边框色；因为边框色本身就在输入里
#   可见，所以这种颜色变化是有结构提示的，不是无提示随机改色。
#
# i) 新隐式假设与生成器对应:
# - `main()` 假设两个 destination 都能通过路径端点唯一识别；生成器沿用原题端口放置。
# - `main()` 假设每个 destination 的中心色至多对应一个 detached source；生成器让
#   两个 key 颜色唯一且 distractor 不复用。
# - `main()` 假设被保留的 source 与 destination 彼此分离；生成器通过 output-first
#   与 `_overlaps()` 保证组件分割稳定。

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT, ROOT / "solvers", ROOT / "arc2_solver_select"):
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.append(extra_str)

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
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
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == b:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    total = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == total:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                traced = dfs(nb, visited | {nb})
                if traced is not None:
                    return [cur] + traced
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
    return path[0], path[-1]


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders (e.g., near the grid edge).
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
        if border_color == bg or border_color == grey or center_color == grey:
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


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint (er,ec) is a port for this sprite if:
      - it's exactly 2 steps away from the sprite center (Manhattan distance 2),
      - the intermediate cell (1 step away from center toward endpoint) is the sprite border color.
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def _select_destinations(
    sprites: list[dict],
    g: np.ndarray,
    trail_eps: set[tuple[int, int]],
    eps_pair: list[tuple[int, int]],
) -> list[dict]:
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[eps_pair[0]]) <= len(opts[eps_pair[1]]) else [1, 0]
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
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"
    return dest


def _match_swap_pairs(sprites: list[dict], dest: list[dict]) -> list[tuple[dict, dict]]:
    """
    Pair each destination with the unique detached source that shares its center color.

    The matched pair keeps both sprites in the output. The visible 'swap' is expressed by
    writing border identities into the two centers: dest.center <- src.border and
    src.center <- dest.border.
    """
    dest_ids = {id(sp) for sp in dest}
    by_center: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        by_center.setdefault(int(sp["center"]), []).append(sp)

    pairs: list[tuple[dict, dict]] = []
    used_sources: set[int] = set()
    for d in dest:
        want_center = int(d["center"])
        srcs = by_center.get(want_center, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination center color should match at most one detached source"
        src = srcs[0]
        sid = id(src)
        assert sid not in used_sources, "combo swap requires one-to-one destination/source matches"
        used_sources.add(sid)
        pairs.append((d, src))
    return pairs


def _prune_unkept_foreground(out: np.ndarray, grey: int, keep_ids: set[int], sprites: list[dict], bg: int) -> None:
    """Delete all unkept sprites and any non-sprite foreground not 4-connected to a kept sprite."""
    for sp in sprites:
        if id(sp) in keep_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for sp in sprites:
        if id(sp) not in keep_ids:
            continue
        for r, c in sp["pts"]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match path-selected destinations to detached sprites by equal center color, then perform a visible two-way exchange.

    Input structure:
    - The grid contains background, one grey (5) one-stroke trail, and several small bordered sprites with one distinct center pixel.
    - The trail endpoints identify exactly two destination sprites through 2-step ports; other sprites are detached sources or distractors.

    Rule:
    - Use the two trail endpoints to select the destination sprites.
    - For each destination, find the unique detached source whose center color equals the destination's center color.
    - Keep the destinations and their matched sources, delete every other sprite, and simultaneously rewrite the paired centers as `dest.center <- source.border` and `source.center <- dest.border`.

    Key insight:
    - The shared center color is only a pairing key. If we swapped those equal colors directly, nothing would change.
    - The observable exchange therefore happens through the visible border identities of the matched pair, while the grey path still tells you which sprites are destinations.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    dest = _select_destinations(sprites, g, trail_eps, eps_pair)
    pairs = _match_swap_pairs(sprites, dest)

    keep_ids = {id(sp) for sp in dest}
    for _d, src in pairs:
        keep_ids.add(id(src))

    out = g.copy()
    _prune_unkept_foreground(out, grey, keep_ids, sprites, bg)

    updates: dict[tuple[int, int], int] = {}
    for d, src in pairs:
        dr, dc = d["center_pos"]
        sr, sc = src["center_pos"]
        updates[(dr, dc)] = int(src["border"])
        updates[(sr, sc)] = int(d["border"])
    for (r, c), color in updates.items():
        out[r, c] = color

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNT = 2
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
    """Place patch cells on grid, skipping transparent (0) and out-of-bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a _place_sprite by zeroing non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != 0:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != 0:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if grid[rr, cc] != 0:
                        return True
                    patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid (r0,c0) for a 3x3 patch where center stays on grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check if adding (r,c) would push any cell's 8-degree above MAX_THICKNESS."""
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


def _gen_winding_path(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int):
    """Generate a winding 8-connected grey path with no 2x2 blocks."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(move: tuple[int, int, bool]) -> int:
        return abs(move[0] - end[0]) + abs(move[1] - end[1])

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
                    and grid[nr, nc] == 0
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
    """Check 8-connectivity of a set of cells via BFS."""
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return len(visited) == len(cell_set)


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int):
    """Find a valid port direction for a sprite centered at (cr,cc) with border color bc."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == border_color and grid[er, ec] == 0:
            return (er, ec)
    return None


def _assign_colors(border_colors: list[int], num_sprites: int):
    """Assign destination/source/distractor roles and all pre/post-swap center colors."""
    if num_sprites < 2 + SOURCE_COUNT:
        return None

    remaining = list(range(2, num_sprites))
    random.shuffle(remaining)
    source_indices = remaining[:SOURCE_COUNT]
    distractor_indices = remaining[SOURCE_COUNT:]
    pair_source_indices = list(source_indices)
    random.shuffle(pair_source_indices)

    key_pool = [c for c in range(1, 10) if c not in {BG, GREY} and c not in border_colors]
    if len(key_pool) < SOURCE_COUNT:
        return None
    pair_keys = random.sample(key_pool, SOURCE_COUNT)

    dest_center_colors = list(pair_keys)
    source_center_by_index = {
        si: pair_keys[i] for i, si in enumerate(pair_source_indices)
    }

    output_dest_centers = [border_colors[si] for si in pair_source_indices]
    output_source_centers = {si: border_colors[i] for i, si in enumerate(pair_source_indices)}

    distractor_center_colors = []
    spare_centers = [c for c in key_pool if c not in pair_keys]
    for di in distractor_indices:
        candidates = [c for c in spare_centers if c != border_colors[di]]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        distractor_center_colors.append(chosen)
        spare_centers = [c for c in spare_centers if c != chosen]

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "pair_source_indices": pair_source_indices,
        "pair_keys": pair_keys,
        "dest_center_colors": dest_center_colors,
        "source_center_by_index": source_center_by_index,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
        "output_source_centers": output_source_centers,
    }


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
):
    """Place 2 destination sprites on the grid. Returns destination center positions or None."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions = []
    first_port = None

    for i in range(2):
        border_color = border_colors[i]
        patch = _make_sprite_patch(shape_type, border_color, output_dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1

            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc, border_color)
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            if i == 1:
                assert first_port is not None
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


def _place_free_sprite(grid: np.ndarray, patch: np.ndarray):
    """Place a sprite anywhere legal; return its center position or None."""
    pool = _sprite_positions()
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(grid, patch, r0, c0):
            continue
        _place_sprite(grid, patch, r0, c0)
        return (r0 + 1, c0 + 1)
    return None


def _gen_attempt():
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
    dest_center_positions = _place_dests(
        output_grid, shape_type, border_colors, colors["output_dest_centers"], min_port_dist
    )
    if dest_center_positions is None:
        return None

    port_cells = []
    for i in range(2):
        cr, cc = dest_center_positions[i]
        port = _find_port_dir(output_grid, cr, cc, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(output_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    source_center_positions: dict[int, tuple[int, int]] = {}
    for si in colors["source_indices"]:
        patch = _make_sprite_patch(shape_type, border_colors[si], colors["output_source_centers"][si])
        pos = _place_free_sprite(output_grid, patch)
        if pos is None:
            return None
        source_center_positions[si] = pos

    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc = dest_center_positions[i]
        input_grid[cr, cc] = colors["dest_center_colors"][i]
    for si in colors["source_indices"]:
        sr, sc = source_center_positions[si]
        input_grid[sr, sc] = colors["source_center_by_index"][si]

    for j, di in enumerate(colors["distractor_indices"]):
        patch = _make_sprite_patch(shape_type, border_colors[di], colors["distractor_center_colors"][j])
        if _place_free_sprite(input_grid, patch) is None:
            return None

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            return input_grid
    except Exception:
        pass
    return None


def generate_input():
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
