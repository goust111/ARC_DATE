# Combo variant of d35bdbdc (index 38)
# combo_spec: D4/2+D11/2
# Name: 面积判定目标+按面积排序处理
#
# Design analysis
# a) 具体化与行为追踪:
#    - 保留原题的两色 bordered sprite 语法、灰色细路径、10x10 画布；只把“谁是目标”和“谁先处理”都改成面积逻辑。
#    - 生成样例固定使用 3 档可见面积: 9(完整 3x3)、6(边缘裁剪成 2x3/3x2)、4(角落裁剪成 2x2)。
#    - 原题会沿灰路径端点找两个目标；本融合版忽略路径，直接把面积前两名当目标，再按面积从大到小处理。
#    - 两个目标的输入中心色都等于同一个小 source 的边框色，所以它们会竞争同一个 source；
#      面积 9 的目标先拿到 source.center，面积 6 的目标随后因 source 已被占用而保持原中心色不变。
#    - 同一输入若按原题规则，会保留被路径碰到的小精灵；按本规则会保留面积 9 和 6 的两个大精灵，差异可见。
# b) 复杂度校准:
#    - 面积只保留 9/6/4 三档，并强制恰好两个最大者为 9 和 6，避免排序歧义。
#    - 所有真正参与竞争的颜色关系只有一条: 两个目标都查找同一个 source.border；其余小精灵是干扰项但不制造额外匹配。
#    - 不使用颜色编号、坐标奇偶等不可视因素；人类只需看大小与颜色链。
# c) 隐式假设审查:
#    - 原 solver 里 `_find_trail_endpoints()`、`_ports_for_sprite()` 和端点二分匹配都依赖“目标由路径端点决定”，本融合版完全失效，必须删除。
#    - 原来的 `dest.center -> source.border` 颜色传递仍可保留，但 source 集合必须改成“非目标精灵”，否则可能误把另一个目标当 source。
#    - 新增的顺序竞争假设是“同一个 source 最多只能被使用一次”；solver 里显式维护 `used_sources`。
#    - 另一个新假设是“面积排序足够稳定”。生成器通过固定目标面积 9、6，其余全为 4 来保证。
# d) 放置合法性:
#    - 面积 9 目标只放完整内区，面积 6 目标只放边缘非角位置，面积 4 小精灵只放四角，天然不越界且角色层次清晰。
#    - 先占位全部精灵，再在空白处生成灰路径，避免路径穿过未来精灵。
#    - `_overlaps()` 继续禁止精灵重叠或 4 邻接，保证 `components_mask()` 检测稳定。
# e) 视觉可观察性:
#    - 路径故意连接两个面积 4 的小精灵，但输出删除它们，迫使观察者放弃“路径选目标”的旧解释。
#    - 输出只保留面积 9 和 6 两个目标，其中只有更大的那个发生中心色变化，能把“面积也决定处理顺序”直接暴露出来。
# f) 可视化整洁度:
#    - 三档面积分居不同区域类型: 大目标在内部/边缘，小精灵在角落，灰路径穿越中间空白，减少遮挡与层叠。
#    - 所有精灵都保持同一种 box 语法，避免同时改动形状和角色规则。
# g) 变化可推断性:
#    - 除两个目标的输入中心色外，其余中心色都不再等于任何边框色，减少无关颜色链。
#    - 因此样例会稳定呈现“两个较大精灵都指向同一个小 source，但只有面积更大的那个得到转移”的结构线索。

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, paste


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
    Clipped edge cases still qualify as long as this 2-color pattern remains visible.
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
        center_pos = next(((int(r), int(c)) for r, c in pts if int(g[r, c]) == center_color), None)
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
                "area": len(pts),
            }
        )
    return sprites


def _rank_sprites_by_area(sprites: list[dict]) -> list[dict]:
    """Sort by visible area descending; coordinate tie-break is fallback only."""
    return sorted(
        sprites,
        key=lambda sp: (-int(sp["area"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between bordered sprites, with area deciding both role and priority.

    Input structure:
    - The grid contains background 0, a grey (5) winding trail, and several two-color bordered sprites.
    - Sprites can be visibly clipped, so their remaining areas differ; the largest sprites are the important ones and the grey trail may touch only smaller decoys.

    Rule:
    - Rank all bordered sprites by visible area from large to small and treat the two largest sprites as destinations.
    - Process those destinations in that same area order. For each one, look among the smaller sprites for an unused source whose border color matches the destination's original center color.
    - If such a source exists, copy that source's center color into the destination center and mark the source as consumed; later destinations cannot reuse it.
    - Delete every non-destination sprite; keep the grey trail and the two largest destinations.

    Key insight:
    - A single observable dimension, visible area, drives the whole inference chain: it selects which sprites survive and also which surviving sprite gets first claim on a shared color-matching source.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) >= 2, "expected at least two bordered sprites"

    ranked = _rank_sprites_by_area(sprites)
    dest = ranked[:2]
    dest_ids = {id(sp) for sp in dest}

    source_by_border: dict[int, dict] = {}
    for sp in ranked[2:]:
        border = int(sp["border"])
        assert border not in source_by_border, "non-destination border colors should be unique"
        source_by_border[border] = sp

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
            if out[r, c] == bg or out[r, c] == grey or keep[r, c]:
                continue
            keep[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    out[(out != bg) & (out != grey) & (~keep)] = bg

    used_sources: set[int] = set()
    for d in dest:
        src = source_by_border.get(int(d["center"]))
        if src is None or id(src) in used_sources:
            continue
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])
        used_sources.add(id(src))

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500


def _make_box_patch(border_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), border_color, dtype=int)
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and 4-neighbor touching between sprites."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _visible_area(patch: np.ndarray, r0: int, c0: int) -> int:
    area = 0
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                area += 1
    return area


def _all_positions() -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _positions_with_area(area: int) -> list[tuple[int, int]]:
    probe = _make_box_patch(1, 2)
    return [(r0, c0) for r0, c0 in _all_positions() if _visible_area(probe, r0, c0) == area]


FULL_POSITIONS = [p for p in _positions_with_area(9) if 1 <= p[0] <= GRID_H - 4 and 1 <= p[1] <= GRID_W - 4]
MEDIUM_POSITIONS = _positions_with_area(6)
SMALL_POSITIONS = _positions_with_area(4)


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
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
    blocked_grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path through empty cells."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int]) -> int:
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
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if (nr, nc) in visited or blocked_grid[nr, nc] != BG:
                    continue
                if _too_thick(nr, nc, visited):
                    continue
                moves.append((nr, nc, dr != 0 and dc != 0))

        if not moves:
            return None

        dist = abs(cr - end[0]) + abs(cc - end[1])
        budget = target_len - len(path)
        diag = [m for m in moves if m[2]]
        away = [m for m in moves if score((m[0], m[1])) > dist]

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
                chosen = min(pool, key=lambda m: score((m[0], m[1])))
        else:
            best = sorted(moves, key=lambda m: score((m[0], m[1])))
            top = [m for m in best if score((m[0], m[1])) <= score((best[0][0], best[0][1])) + 1]
            dtop = [m for m in top if m[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if blocked_grid[end[0], end[1]] != BG or _too_thick(end[0], end[1], visited):
            return None
        path.append(end)

    return path


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
    return len(seen) == len(cell_set)


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, bc: int) -> tuple[int, int] | None:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == BG:
            return er, ec
    return None


def _center_of(r0: int, c0: int) -> tuple[int, int]:
    return r0 + 1, c0 + 1


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _choose_center_colors(border_colors: list[int], num_distractors: int) -> tuple[int, list[int]] | None:
    pool = [c for c in range(1, 10) if c != GREY and c not in set(border_colors)]
    need = 1 + num_distractors
    if len(pool) < need:
        return None
    picked = random.sample(pool, need)
    return picked[0], picked[1:]


def _build_specs(num_sprites: int, border_colors: list[int]) -> dict[int, dict] | None:
    num_distractors = num_sprites - 3
    colors = _choose_center_colors(border_colors, num_distractors)
    if colors is None:
        return None
    shared_source_center, distractor_centers = colors
    shared_lookup = border_colors[2]
    specs = {
        0: {
            "role": "dest_large",
            "border": border_colors[0],
            "input_center": shared_lookup,
            "output_center": shared_source_center,
            "area": 9,
        },
        1: {
            "role": "dest_medium",
            "border": border_colors[1],
            "input_center": shared_lookup,
            "output_center": shared_lookup,
            "area": 6,
        },
        2: {
            "role": "source_shared",
            "border": shared_lookup,
            "input_center": shared_source_center,
            "area": 4,
        },
    }
    for idx in range(3, num_sprites):
        specs[idx] = {
            "role": "distractor_anchor" if idx == 3 else "distractor",
            "border": border_colors[idx],
            "input_center": distractor_centers[idx - 3],
            "area": 4,
        }
    return specs


def _place_targets(planning_grid: np.ndarray, specs: dict[int, dict]) -> dict[int, tuple[int, int]] | None:
    large_patch = _make_box_patch(specs[0]["border"], specs[0]["input_center"])
    medium_patch = _make_box_patch(specs[1]["border"], specs[1]["input_center"])

    large_pool = list(FULL_POSITIONS)
    medium_pool = list(MEDIUM_POSITIONS)
    random.shuffle(large_pool)
    random.shuffle(medium_pool)

    for r0, c0 in large_pool:
        if _overlaps(planning_grid, large_patch, r0, c0):
            continue
        _place_sprite(planning_grid, large_patch, r0, c0)
        large_center = _center_of(r0, c0)

        placed_medium = None
        for mr0, mc0 in medium_pool:
            if _overlaps(planning_grid, medium_patch, mr0, mc0):
                continue
            if _manhattan(large_center, _center_of(mr0, mc0)) < 4:
                continue
            _place_sprite(planning_grid, medium_patch, mr0, mc0)
            placed_medium = (mr0, mc0)
            break

        if placed_medium is not None:
            return {0: (r0, c0), 1: placed_medium}

        _erase_sprite(planning_grid, large_patch, r0, c0)

    return None


def _place_small_with_port(
    planning_grid: np.ndarray,
    border_color: int,
    center_color: int,
    used_positions: set[tuple[int, int]],
    min_port_dist: int,
    first_port: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    patch = _make_box_patch(border_color, center_color)
    pool = [p for p in SMALL_POSITIONS if p not in used_positions]
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(planning_grid, patch, r0, c0):
            continue
        _place_sprite(planning_grid, patch, r0, c0)
        cr, cc = _center_of(r0, c0)
        port = _find_port_dir(planning_grid, cr, cc, border_color)
        if port is None:
            _erase_sprite(planning_grid, patch, r0, c0)
            continue
        if first_port is not None and _manhattan(port, first_port) < min_port_dist:
            _erase_sprite(planning_grid, patch, r0, c0)
            continue
        used_positions.add((r0, c0))
        return (r0, c0), port
    return None


def _place_extra_small(
    planning_grid: np.ndarray,
    border_color: int,
    center_color: int,
    used_positions: set[tuple[int, int]],
) -> tuple[int, int] | None:
    patch = _make_box_patch(border_color, center_color)
    pool = [p for p in SMALL_POSITIONS if p not in used_positions]
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(planning_grid, patch, r0, c0):
            continue
        _place_sprite(planning_grid, patch, r0, c0)
        used_positions.add((r0, c0))
        return r0, c0
    return None


def _paint_patch(
    grid: np.ndarray, r0: int, c0: int, border_color: int, center_color: int
) -> None:
    _place_sprite(grid, _make_box_patch(border_color, center_color), r0, c0)


def _gen_attempt() -> np.ndarray | None:
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample([c for c in range(1, 10) if c != GREY], num_sprites)
    specs = _build_specs(num_sprites, border_colors)
    if specs is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    planning_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    target_places = _place_targets(planning_grid, specs)
    if target_places is None:
        return None

    used_small_positions: set[tuple[int, int]] = set()
    small_places: dict[int, tuple[int, int]] = {}

    source_result = _place_small_with_port(
        planning_grid,
        specs[2]["border"],
        specs[2]["input_center"],
        used_small_positions,
        min_port_dist=0,
    )
    if source_result is None:
        return None
    small_places[2], port_a = source_result

    anchor_result = _place_small_with_port(
        planning_grid,
        specs[3]["border"],
        specs[3]["input_center"],
        used_small_positions,
        min_port_dist=min_port_dist,
        first_port=port_a,
    )
    if anchor_result is None:
        return None
    small_places[3], port_b = anchor_result

    for idx in range(4, num_sprites):
        placed = _place_extra_small(
            planning_grid,
            specs[idx]["border"],
            specs[idx]["input_center"],
            used_small_positions,
        )
        if placed is None:
            return None
        small_places[idx] = placed

    path_cells = _gen_winding_path(planning_grid, port_a, port_b, target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for idx in (0, 1):
        r0, c0 = target_places[idx]
        _paint_patch(output_grid, r0, c0, specs[idx]["border"], specs[idx]["output_center"])

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    input_grid = output_grid.copy()
    for idx in (0, 1):
        r0, c0 = target_places[idx]
        _erase_sprite(
            input_grid,
            _make_box_patch(specs[idx]["border"], specs[idx]["output_center"]),
            r0,
            c0,
        )
        _paint_patch(input_grid, r0, c0, specs[idx]["border"], specs[idx]["input_center"])

    for idx in range(2, num_sprites):
        r0, c0 = small_places[idx]
        _paint_patch(input_grid, r0, c0, specs[idx]["border"], specs[idx]["input_center"])

    sprites = _detect_bordered_sprites(input_grid, BG, GREY)
    areas = sorted(sp["area"] for sp in sprites)
    expected = [4] * (num_sprites - 2) + [6, 9]
    if areas != expected:
        return None

    ranked = _rank_sprites_by_area(sprites)
    if [int(ranked[0]["area"]), int(ranked[1]["area"])] != [9, 6]:
        return None
    if int(ranked[0]["center"]) != specs[2]["border"] or int(ranked[1]["center"]) != specs[2]["border"]:
        return None

    try:
        out = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(out, output_grid):
        return None

    changed = [sp for sp in _detect_bordered_sprites(out, BG, GREY)]
    changed_ranked = _rank_sprites_by_area(changed)
    if int(changed_ranked[0]["center"]) != specs[2]["input_center"]:
        return None
    if int(changed_ranked[1]["center"]) != specs[2]["border"]:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
