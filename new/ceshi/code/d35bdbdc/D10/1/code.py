# Variant of d35bdbdc
# Dimension: D10 — 精灵数量与配对基数 (multi_step)
# Change: 多个源和多个目标精灵，多对多匹配（每个目标找到一个匹配的源）... → 多源对一目标
#
# Design analysis
# ===============
# spec:
#   concretization:
#     - 保留原题的灰色单笔路径、若干小型带边框精灵、以及“颜色链”这一核心线索。
#     - 唯一目标精灵改成一个更大的 3x4/3x5 或 4x3/5x3 长条框块：
#       外环是同色边框，内部中央一行/一列有 2-3 个槽位。
#     - 输入中，槽位颜色依次写入若干源精灵的 border 颜色；输出中，这些槽位按同一空间顺序
#       替换成对应源精灵的 center 颜色。这样多个源被聚合到同一个目标内部区域。
#     - 灰色路径的两个端点分别接在这个长条目标的两端端口上，因此“灰路选目标”的机制仍在，
#       但两端点共同指向同一个目标，而不是两个不同目标。
#   behavior_trace:
#     - 原题在“根据端点找目标”之后，对每个目标只读 1 个中心像素，查到 1 个源，再把 1 个颜色写回。
#     - 这里改成：端点确定 1 个长条目标；读取该目标 2-3 个槽位中的 lookup 颜色，分别查 source.border；
#       再把查到的 source.center 依序写回这些槽位。差异点在于目标数量从 2 降到 1，但目标内部承载了
#       多个独立 lookup，因此 solver 需要做一次聚合而不是单点替换。
#   complexity_calibration:
#     - 源精灵数量固定为 2 或 3，避免退化回单源情形。
#     - 槽位顺序直接由目标内部的左右/上下空间顺序给出，不依赖颜色 id 排序，肉眼可观察。
#     - 目标只允许水平或垂直两个朝向，分支少且可从 2-4 个样例归纳。
#   old_assumption_audit:
#     - 原 `_detect_bordered_sprites` 假设“组件只有 2 色且稀有色恰好 1 个”，这对新目标不成立；
#       因此目标检测必须单独实现，不能沿用“小精灵只有一个中心点”的前提。
#     - 原 `main()` 默认灰路两个端点各对应一个不同目标，并做 bipartite match；变体中这会错误地把
#       同一长条目标拆成两个目标，所以必须改成“两个端点共同锁定同一目标”。
#     - 原输出更新阶段对每个目标只写 1 个中心像素；若沿用，会漏掉其余槽位，因此要改成逐槽位写回。
#     - 原删除逻辑保留两个目标组件；变体中必须只保留单个长条目标组件。
#   legality:
#     - 目标始终完整放在网格内，并额外保证两侧端口也在网格内，便于灰路连接。
#     - 小精灵仍使用原题 3x3 box/plus，放置时不重叠、不 4 邻接其他非背景对象。
#     - 生成器采用 output-first：先构造带聚合结果的目标与灰路，再回填输入 lookup 槽位，最后放源和干扰项。
#   observability:
#     - 输入里单个大目标的多个槽位 visibly 写着若干源的边框色；输出里同一位置变成这些源的中心色，
#       并且所有源精灵被删掉。聚合行为在视觉上明显区别于原题的“两个独立单点目标”。
#   cleanliness:
#     - 目标固定为一个完整长条矩形，不做裁剪，避免与小精灵的局部残缺形态混淆。
#     - 颜色变化仅发生在目标中央槽位，输出读图干净：灰路保留，单一目标保留，其余对象清空。

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
    Small bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
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


def _try_detect_target(g: np.ndarray, pts: np.ndarray, bg: int, grey: int) -> dict | None:
    """
    Aggregation target = a full 3x4/3x5 or 4x3/5x3 rectangle:
      - outer ring is one border color
      - central row/column holds 2-3 non-border lookup cells
      - the two ports are immediately outside the short-side midpoints
    """
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if (h, w) not in {(3, 4), (3, 5), (4, 3), (5, 3)}:
        return None
    if len(pts) != h * w:
        return None

    bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    comp_cells = {(int(r), int(c)) for r, c in pts}
    if bbox_cells != comp_cells:
        return None

    if h == 3:
        orientation = "horizontal"
        slot_positions = [(r0 + 1, c) for c in range(c0 + 1, c1)]
        ports = [(r0 + 1, c0 - 1), (r0 + 1, c1 + 1)]
    else:
        orientation = "vertical"
        slot_positions = [(r, c0 + 1) for r in range(r0 + 1, r1)]
        ports = [(r0 - 1, c0 + 1), (r1 + 1, c0 + 1)]
    if any(not (0 <= pr < g.shape[0] and 0 <= pc < g.shape[1]) for pr, pc in ports):
        return None

    slot_set = set(slot_positions)
    border_positions = sorted(bbox_cells - slot_set)
    border_color = int(g[border_positions[0]])
    if border_color in (bg, grey):
        return None
    if any(int(g[r, c]) != border_color for r, c in border_positions):
        return None

    slot_colors = [int(g[r, c]) for r, c in slot_positions]
    if len(slot_colors) not in (2, 3):
        return None
    if any(col in (bg, grey, border_color) for col in slot_colors):
        return None

    return {
        "pts": pts,
        "border": border_color,
        "slot_positions": slot_positions,
        "slot_colors": slot_colors,
        "ports": ports,
        "orientation": orientation,
    }


def _detect_targets(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    targets: list[dict] = []
    for pts in comps:
        detected = _try_detect_target(g, pts, bg, grey)
        if detected is not None:
            targets.append(detected)
    return targets


def _find_trail_endpoints(
    g: np.ndarray, grey: int, targets: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two endpoint cells of the grey one-stroke trail."""
    cells = {(int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False)}
    assert cells, "expected grey trail cells"

    target_ports = {
        port
        for target in targets
        for port in target["ports"]
        if port in cells
    }
    if len(target_ports) == 2:
        a, b = sorted(target_ports)
        return a, b

    adj = {
        cell: [nb for nb in neighbors8(*cell) if nb in cells]
        for cell in cells
    }
    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]
    raise AssertionError("could not identify grey trail endpoints")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aggregate several source colors into one path-selected target strip, then remove everything else.

    Input structure:
    - The grid contains a grey (5) one-stroke trail, several small bordered sprites, and exactly one larger rectangular target sprite.
    - The large target has a single border color and 2-3 central slots arranged in one row or one column.
      The grey trail touches that target at both end ports.

    Rule:
    - Use the two grey trail endpoints to identify the single large target they both connect to.
    - Read each target slot's original color as a lookup key: it must match the border color of one small source sprite.
    - Replace every target slot with the matched source sprite's center color, preserving slot order, then delete all non-target sprites.

    Key insight:
    - The target no longer asks for one lookup at one center pixel; instead, its visible slot sequence stores several border-color keys.
      Solving requires aggregating multiple source-center values into that one target region in the same spatial order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    targets = _detect_targets(g, bg, grey)
    assert targets, "expected one aggregation target"
    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected small bordered sprites"

    te1, te2 = _find_trail_endpoints(g, grey, targets)
    trail_eps = {te1, te2}
    dests = [target for target in targets if set(target["ports"]) == trail_eps]
    assert len(dests) == 1, "expected exactly one target touched by both trail endpoints"
    dest = dests[0]

    by_border: dict[int, dict] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), sp)

    out = g.copy()
    keep = {(int(r), int(c)) for r, c in dest["pts"]}
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if out[r, c] in (bg, grey):
                continue
            if (r, c) not in keep:
                out[r, c] = bg

    for (r, c), want_border in zip(dest["slot_positions"], dest["slot_colors"], strict=True):
        src = by_border.get(int(want_border))
        if src is not None:
            out[r, c] = int(src["center"])

    return out


# --------------- generator (c) ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SOURCE_COUNTS = [2, 3]
DISTRACTOR_COUNTS = [0, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500


def _make_small_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus sprite with a single center cell."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _make_target_patch(orientation: str, border_color: int, slot_colors: list[int]) -> np.ndarray:
    """Build the single large target whose center strip has 2-3 slots."""
    n = len(slot_colors)
    if orientation == "horizontal":
        patch = np.full((3, n + 2), border_color, dtype=int)
        for i, color in enumerate(slot_colors, start=1):
            patch[1, i] = color
    else:
        patch = np.full((n + 2, 3), border_color, dtype=int)
        for i, color in enumerate(slot_colors, start=1):
            patch[i, 1] = color
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    for dr in range(h):
        for dc in range(w):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap or 4-touch with existing non-bg, non-grey objects."""
    h, w = patch.shape
    patch_cells = set()
    for dr in range(h):
        for dc in range(w):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < GRID_H and 0 <= cc < GRID_W):
                continue
            if grid[rr, cc] != BG:
                return True
            patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(grid[nr, nc]) not in (BG, GREY):
                return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid top-lefts for a small 3x3 sprite whose center stays on-grid."""
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


def _gen_winding_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int,
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path with no overlap into occupied cells."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(m: tuple[int, int, bool]) -> int:
        return abs(m[0] - end[0]) + abs(m[1] - end[1])

    for _ in range(target_len + 20):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if (nr, nc) in visited or grid[nr, nc] != BG:
                    continue
                if _too_thick(nr, nc, visited):
                    continue
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
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nb in neighbors8(r, c):
            if nb in cell_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(cell_set)


def _target_layout(
    orientation: str, num_slots: int, r0: int, c0: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if orientation == "horizontal":
        width = num_slots + 2
        slots = [(r0 + 1, c0 + i) for i in range(1, width - 1)]
        ports = [(r0 + 1, c0 - 1), (r0 + 1, c0 + width)]
    else:
        height = num_slots + 2
        slots = [(r0 + i, c0 + 1) for i in range(1, height - 1)]
        ports = [(r0 - 1, c0 + 1), (r0 + height, c0 + 1)]
    return slots, ports


def _target_positions(orientation: str, num_slots: int) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        h, w = 3, num_slots + 2
        return [(r0, c0) for r0 in range(0, GRID_H - h + 1) for c0 in range(1, GRID_W - w)]
    h, w = num_slots + 2, 3
    return [(r0, c0) for r0 in range(1, GRID_H - h) for c0 in range(0, GRID_W - w + 1)]


def _assign_colors() -> dict | None:
    num_sources = random.choice(SOURCE_COUNTS)
    num_distractors = random.choice(DISTRACTOR_COUNTS)
    total_borders = 1 + num_sources + num_distractors
    palette = [c for c in range(1, 10) if c != GREY]
    if total_borders > len(palette):
        return None

    borders = random.sample(palette, total_borders)
    target_border = borders[0]
    source_borders = borders[1:1 + num_sources]
    distractor_borders = borders[1 + num_sources:]

    available_centers = [c for c in palette if c not in borders]
    if len(available_centers) < num_sources:
        return None
    source_centers = random.sample(available_centers, num_sources)
    source_map = dict(zip(source_borders, source_centers, strict=True))

    lookup_order = list(source_borders)
    random.shuffle(lookup_order)
    output_slots = [source_map[color] for color in lookup_order]

    distractor_centers: list[int] = []
    for bc in distractor_borders:
        candidates = [c for c in palette if c not in (bc, target_border)]
        distractor_centers.append(random.choice(candidates))

    return {
        "num_sources": num_sources,
        "target_border": target_border,
        "lookup_order": lookup_order,
        "output_slots": output_slots,
        "source_borders": source_borders,
        "source_centers": source_centers,
        "distractor_borders": distractor_borders,
        "distractor_centers": distractor_centers,
    }


def _place_target(output_grid: np.ndarray, orientation: str, colors: dict) -> dict | None:
    slot_colors = colors["output_slots"]
    patch = _make_target_patch(orientation, colors["target_border"], slot_colors)
    positions = _target_positions(orientation, len(slot_colors))
    random.shuffle(positions)

    for r0, c0 in positions:
        if _overlaps(output_grid, patch, r0, c0):
            continue
        _place_patch(output_grid, patch, r0, c0)
        slots, ports = _target_layout(orientation, len(slot_colors), r0, c0)
        if all(0 <= pr < GRID_H and 0 <= pc < GRID_W and output_grid[pr, pc] == BG for pr, pc in ports):
            return {
                "r0": r0,
                "c0": c0,
                "orientation": orientation,
                "slots": slots,
                "ports": ports,
                "patch": patch,
            }
        _erase_patch(output_grid, patch, r0, c0)
    return None


def _place_small_sprites(input_grid: np.ndarray, colors: dict, shape_type: str) -> bool:
    all_pos = _sprite_positions()
    edge_pos = [pos for pos in all_pos if _is_edge(*pos)]
    int_pos = [pos for pos in all_pos if not _is_edge(*pos)]
    sprites_to_place = [
        (bc, cc)
        for bc, cc in zip(colors["source_borders"], colors["source_centers"], strict=True)
    ] + [
        (bc, cc)
        for bc, cc in zip(colors["distractor_borders"], colors["distractor_centers"], strict=True)
    ]
    random.shuffle(sprites_to_place)

    for idx, (border_color, center_color) in enumerate(sprites_to_place):
        patch = _make_small_sprite_patch(shape_type, border_color, center_color)
        pool = edge_pos + int_pos if idx < len(edge_pos) else all_pos
        pool = list(pool)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_patch(input_grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    colors = _assign_colors()
    if colors is None:
        return None

    orientation = random.choice(["horizontal", "vertical"])
    shape_type = random.choice(["box", "plus"])
    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    target = _place_target(output_grid, orientation, colors)
    if target is None:
        return None

    path_cells = _gen_winding_path(output_grid, target["ports"][0], target["ports"][1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    for pr, pc in target["ports"]:
        deg4 = sum(
            1
            for nr, nc in neighbors4(pr, pc)
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg4 > 1:
            return None

    input_grid = output_grid.copy()
    for (r, c), lookup_color in zip(target["slots"], colors["lookup_order"], strict=True):
        input_grid[r, c] = lookup_color

    if not _place_small_sprites(input_grid, colors, shape_type):
        return None

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
