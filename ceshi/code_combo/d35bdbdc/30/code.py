# Combo variant of d35bdbdc (index 30)
# combo_spec: D4/3+D5/1
# Name: 所有精灵互为源目标+双向交换

# Design analysis
# a) 具体化与行为追踪:
#    - 原题里，只有灰路径两端接到的 2 个精灵是 destination；其余精灵只当 source 或 distractor。
#      每个 destination 读取自己的 center 颜色，把它当作“去哪里找 source.border”的 lookup key，
#      然后把 source.center 单向复制过来，并删除所有非 destination 精灵。
#    - D4/3 把“destination 集合”扩成所有精灵：每个精灵都必须把自己的 center 当作 lookup key，
#      同时也要作为别人的被查找对象。于是所有精灵天然形成一个 1 出 1 入的有向网络。
#    - D5/1 把“单向复制”改成“交换”。但若仍按原题的局部二元视角理解，在长度 > 2 的颜色环上会冲突：
#      一个精灵既要和前驱交换，又要和后继交换，顺序不同会得到不同结果；若把网络限制成若干 2-cycle，
#      又会退化成“所有中心都变回自身边框色”，看不出全局匹配网络。
#    - 本融合版采用一个显式、可生成且可观察的稳定化解释：
#      1. 所有精灵都参加同一个长度 4 的 lookup 环。
#      2. 先按 D4/3 建立“我指向谁”的全局网络。
#      3. 再把 D5/1 解释为：因为每个被指向精灵也正在参与交换，最终可见结果不是拿到它的原 center，
#         而是拿到它在同一轮交换网络中继续传递后的 center。
#      4. 对单环来说，这等价于“沿 lookup 环同步前进两跳后取 center”。
#    - 具体小例子。若四个精灵边框依次为 A,B,C,D，输入中心依次为 B,C,D,A，
#      则 lookup 环是 A->B->C->D->A。
#      原题若强行改成“所有精灵都单向复制”，输出中心会变成 C,D,A,B（一跳）。
#      本融合版输出中心变成 D,A,B,C（两跳），所有精灵都保留且都发生变化。
#      这迫使解题者理解整个颜色环，而不是只看灰路径两端。
# b) 复杂度校准:
#    - 为避免 2-cycle 退化成“每个中心直接回到自身边框色”，生成器固定使用 4 个精灵并强制单个 4-cycle。
#    - 不再放置 distractor 或额外杂质组件，否则 10x10 画面在保留全部精灵和灰路径后会过满。
#    - 4-cycle 比 5-cycle 更稳：路径和 4 个 3x3 精灵在 10x10 上更容易合法放置，且两跳传播在样例里更清楚。
# c) 隐式假设审查:
#    - 原 main() 假设只有 2 个 destination，且其余精灵可以直接删除；本版完全失效，必须保留全部精灵。
#    - 原 main() 的 border->sprite 查找仍可复用，但现在每个精灵都必须能唯一指向另一个精灵。
#      生成器因此保证所有 center 都是“其他某个精灵的 border”，且 border 全异。
#    - 原 main() 逐个写 destination.center 不会互相影响；本版所有精灵都会变，必须先用输入态记录整张 lookup 环，
#      再同步写回，不能边读边改。
#    - 新 solver 额外依赖 lookup 图是单个长度 4 的环，而不是多个分量、1-cycle 或 2-cycle。
#      生成器通过构造 permutation 并显式验证 cycle 长度来保证这一点。
# d) 放置合法性:
#    - 继续沿用 output-first。先放 2 个带灰路径端口的锚点精灵并生成灰路径，再放另外 2 个普通精灵。
#    - 全部精灵仍通过 `_overlaps()` 避免重叠和 4 邻接粘连，灰路径仍通过 `_too_thick()` 保持单细笔画。
#    - 输入由输出回退而来：仅恢复 4 个精灵的输入 center，边框、形状和灰路径不变。
# e) 视觉可观察性:
#    - 关键线索是：输入里每个中心色都是另一个精灵的边框色，四个精灵串成一个闭环；
#      输出里所有精灵都还在，但中心色不是看一跳得到，而是要沿同一条颜色环再多追一步。
#    - 灰路径仍在画面中，但不再决定“谁保留谁删除”；它只保留原题视觉骨架，真正决定输出的是全局颜色环。
# f) 可视化整洁度:
#    - 固定 4 个同形精灵，避免 5 个对象把 10x10 挤得过满。
#    - 仍只使用一种小精灵形状（box 或 plus）和一条灰色细路径，层次简单，便于肉眼追踪颜色环。
# g) 变化可推断性:
#    - 颜色数量仍小：4 个互异边框色 + 灰色 + 背景。
#    - 所有中心色都直接可见地“指向某个边框色”，解题者可以在 2-3 组样例中发现：
#      输出中心总是来自沿这个可见环多追一步后的对象，而不是来自灰路径端点。
# i) 新 solver 假设与生成器对齐:
#    - 假设 1: 所有被检测到的 bordered sprite 都参与同一个 lookup 环。生成器只生成 4 个 sprite 且无杂质。
#    - 假设 2: 每个 center 唯一匹配某个其他 sprite 的 border。生成器令 4 个边框色全异，并把输入 center 构成 permutation。
#    - 假设 3: 环长度 >= 3，否则两跳传播会退化。生成器固定长度 4 并拒绝任何短环。

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    This also handles clipped/partial borders near the grid edge.
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


def _lookup_cycle(sprites: list[dict]) -> list[dict]:
    """Return the unique full lookup cycle induced by center -> border matching."""
    by_border = {int(sp["border"]): sp for sp in sprites}
    assert len(by_border) == len(sprites), "border colors must be unique"

    nxt: dict[int, dict] = {}
    for sp in sprites:
        target = by_border.get(int(sp["center"]))
        assert target is not None, "every sprite center must name another sprite border"
        assert id(target) != id(sp), "self-loops are not allowed in this combo"
        nxt[id(sp)] = target

    start = sprites[0]
    order = [start]
    seen = {id(start)}
    cur = start
    while True:
        cur = nxt[id(cur)]
        if id(cur) == id(start):
            break
        assert id(cur) not in seen, "expected a single simple cycle"
        order.append(cur)
        seen.add(id(cur))

    assert len(order) == len(sprites), "all sprites must lie on one lookup cycle"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a global two-hop center exchange around the full lookup cycle of bordered sprites.

    Input structure:
    - The grid contains background, one grey (5) winding trail, and exactly four small bordered sprites of one common shape.
    - Every sprite has one border color and one single-pixel center color; border colors are all distinct.
    - Unlike the base task, every sprite center is itself a lookup key: it matches the border color of another sprite, so all four sprites form one directed color cycle.

    Rule:
    - Keep every sprite and preserve the grey trail unchanged.
    - Ignore the old destination/source split: all sprites participate simultaneously.
    - For each sprite, follow the center->border lookup relation twice around the global cycle and copy that second sprite's original center color into the sprite's center.
    - No sprites are deleted in this fused variant.

    Key insight:
    - The grey trail is no longer the selector; the important structure is the full ring formed by visible center-to-border matches.
    - A one-hop read would correspond to the simpler all-target copy variant, but the fused double-sided exchange pushes the color one extra step around the same ring, so solving requires understanding the whole network rather than any local pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) == 4, "expected exactly four bordered sprites in this combo"

    cycle = _lookup_cycle(sprites)
    assert len(cycle) == 4, "combo generator should create a 4-cycle"

    out = g.copy()
    n = len(cycle)
    updates: dict[tuple[int, int], int] = {}
    for i, sp in enumerate(cycle):
        src = cycle[(i + 2) % n]
        r, c = sp["center_pos"]
        updates[(r, c)] = int(src["center"])

    for (r, c), color in updates.items():
        out[r, c] = color
    return out


# --------------- generator (output-first global color-cycle variant) ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNT = 4
INTERIOR_ANCHOR_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500


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
    """Place patch cells on grid, skipping transparent background and clipping at borders."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo _place_sprite by zeroing non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check whether a patch overlaps or 4-touches an existing non-bg, non-grey object."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != BG:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if grid[rr, cc] != BG:
                        return True
                    patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid (r0, c0) placements whose center stays on-grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check if adding (r, c) would make the grey trail too thick under 8-neighbor degree."""
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
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path with no excessive local thickening."""
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
                    and grid[nr, nc] == BG
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
        if _too_thick(end[0], end[1], visited):
            return None
        path.append(end)
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


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int) -> tuple[int, int] | None:
    """Find a legal port cell two steps away from the center through a border pixel."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == int(border_color) and int(grid[er, ec]) == BG:
            return (er, ec)
    return None


def _place_anchor_sprites(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    """Place the two path-touched sprites and return their center positions."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior = random.choices([0, 1], weights=INTERIOR_ANCHOR_WEIGHTS)[0]

    centers = []
    first_port = None
    for i in range(2):
        patch = _make_sprite_patch(shape_type, border_colors[i], output_centers[i])
        pool = list(int_pos if i < num_interior else edge_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc, border_colors[i])
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue
            if i == 1:
                assert first_port is not None
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue
            centers.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break
        if not placed:
            return None
    return centers


def _place_free_sprite(grid: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    """Place a non-anchor sprite anywhere legal; return its center position."""
    pool = _sprite_positions()
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(grid, patch, r0, c0):
            continue
        _place_sprite(grid, patch, r0, c0)
        return (r0 + 1, c0 + 1)
    return None


def _cycle_data(border_colors: list[int]) -> dict[str, object]:
    """Construct a single 4-cycle and the corresponding input/output center colors."""
    cycle = list(range(SPRITE_COUNT))
    random.shuffle(cycle)

    input_centers = [0] * SPRITE_COUNT
    output_centers = [0] * SPRITE_COUNT
    for pos, idx in enumerate(cycle):
        next_idx = cycle[(pos + 1) % SPRITE_COUNT]
        input_centers[idx] = border_colors[next_idx]

    for pos, idx in enumerate(cycle):
        two_hop_idx = cycle[(pos + 2) % SPRITE_COUNT]
        output_centers[idx] = input_centers[two_hop_idx]

    assert len(set(input_centers)) == SPRITE_COUNT
    assert len(set(output_centers)) == SPRITE_COUNT
    for i in range(SPRITE_COUNT):
        assert input_centers[i] != border_colors[i]
        assert output_centers[i] != border_colors[i]

    return {"cycle": cycle, "input_centers": input_centers, "output_centers": output_centers}


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, SPRITE_COUNT)
    cycle_info = _cycle_data(border_colors)
    input_centers = cycle_info["input_centers"]
    output_centers = cycle_info["output_centers"]

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    anchor_centers = _place_anchor_sprites(
        output_grid, shape_type, border_colors, output_centers, min_port_dist
    )
    if anchor_centers is None:
        return None

    ports = []
    for i in range(2):
        cr, cc = anchor_centers[i]
        port = _find_port_dir(output_grid, cr, cc, border_colors[i])
        if port is None:
            return None
        ports.append(port)

    path_cells = _gen_winding_path(output_grid, ports[0], ports[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None
    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    for pr, pc in ports:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    sprite_centers = {0: anchor_centers[0], 1: anchor_centers[1]}
    for i in range(2, SPRITE_COUNT):
        patch = _make_sprite_patch(shape_type, border_colors[i], output_centers[i])
        pos = _place_free_sprite(output_grid, patch)
        if pos is None:
            return None
        sprite_centers[i] = pos

    input_grid = output_grid.copy()
    for i in range(SPRITE_COUNT):
        r, c = sprite_centers[i]
        input_grid[r, c] = int(input_centers[i])

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            return input_grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
