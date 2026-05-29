# Variant of d35bdbdc
# Dimension: D4 — 目标对象选取 (dst)
# Change: 通过灰色路径端点连接的带边框精灵为目标精灵(d)... → 面积较大的精灵为目标
#
# Design analysis
# spec:
#   concrete_visual:
#     - 网格仍是 10x10，保留灰色(5)细路径、若干带边框精灵、背景 0。
#     - 精灵继续使用“单个中心像素 + 其余同色边框”的两色结构，避免引入新的识别语法。
#     - 为了让“面积较大”在像素上可见，两个目标精灵固定为完整可见的 3x3 方框(面积 9)；
#       其余精灵都仍来自同一个 3x3 方框模板，但放在边界上被裁剪成 2x3(面积 6)或 2x2(面积 4)。
#     - 灰色路径仍然存在，但特意连接两个小精灵；输出时路径保留，小精灵全部删除。
#   behavior_trace:
#     - 原题：路径端点连接的两个精灵是目标。例如两个被路径碰到的小边缘框会被保留。
#     - 变体：忽略路径连接，先检测所有 bordered sprite，再按可见面积排序，面积最大的两个完整 3x3 方框保留。
#     - 差异：同一输入上，原规则会跟着路径走；新规则会跟着尺寸走。生成器据此强制让路径去碰小精灵，从而避免退化。
#   complexity_calibration:
#     - 面积档位只保留 9 / 6 / 4 三种，且每题恰好两个面积 9 的精灵，避免排序歧义。
#     - 颜色链仍只使用 1-2 个 source，保持原题的间接颜色传递难度，不把变化扩散到颜色机制本身。
#     - 不使用颜色编号参与排序，也不依赖隐藏计数；解题者肉眼可直接看到“大框留下，小框消失”。
#   implicit_assumption_audit:
#     - 原 solver 中 `_find_trail_endpoints()`、`_ports_for_sprite()`、端点二分匹配都依赖“目标必须连到路径端点”；
#       这些前提在变体中失效，因此目标选取整段必须删除。
#     - 删除非目标精灵、按 `dest.center -> source.border` 做中心色替换的逻辑仍成立，因为 D4 没改颜色链。
#     - `by_border` 仍假设每个边框色唯一；生成器继续保证这一点。
#     - 新 solver 引入的前提是“恰好两个最大面积精灵”。生成器通过让目标固定面积 9、其余固定 < 9 来保证。
#   placement_legality:
#     - 目标精灵放在完整可见位置，避免被裁剪后与小精灵并列。
#     - 非目标精灵只放在裁剪位置，并先全部占位后再走灰路径，保证路径不会穿过未来要放的小精灵。
#     - `_overlaps()` 继续禁止精灵之间重叠或 4 邻接，保证 component 检测稳定。
#   visual_observability:
#     - 输入里灰路径连接小精灵，输出里却保留两个大精灵，这会强迫观察者放弃“路径决定目标”的旧解释。
#     - 保留统一的方框语法，只改变可见面积，避免画面同时变化太多维度而变乱。
#   color_inference:
#     - 颜色变化仍由“目标中心色 = 某 source 的边框色”触发，和原题一致。
#     - 颜色种类有限且边框色唯一，解题者能从剩余大精灵中心色的变化回推出来源。

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
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders near the grid boundary.
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
                "area": len(pts),
            }
        )
    return sprites


def _largest_destinations(sprites: list[dict], k: int = 2) -> list[dict]:
    """Pick the k bordered sprites with the largest visible areas."""
    ranked = sorted(
        sprites,
        key=lambda sp: (-int(sp["area"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])),
    )
    assert len(ranked) >= k, f"expected at least {k} bordered sprites"
    return ranked[:k]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between bordered sprites, keeping only the largest ones.

    Input structure:
    - The grid contains background, a grey (5) winding trail, and several two-color bordered sprites.
    - Two sprites are visibly larger because they occupy more cells; the remaining sprites are clipped smaller versions near the edges.

    Rule:
    - Ignore which sprites the grey trail touches.
    - Rank all bordered sprites by visible area and treat the two largest sprites as destinations.
    - For each destination, find the source sprite whose border color equals the destination's original center color, then copy that source center color into the destination center.
    - Delete every smaller sprite; keep the grey trail and the two large destinations.

    Key insight:
    - The role split is encoded geometrically by sprite size rather than by path endpoints, while the color transfer still uses destination-center to source-border matching.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) >= 2, "expected at least two bordered sprites"

    dest = _largest_destinations(sprites, k=2)
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

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

    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


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


def _full_positions() -> list[tuple[int, int]]:
    patch = _make_box_patch(1, 2)
    return [(r0, c0) for r0, c0 in _all_positions() if _visible_area(patch, r0, c0) == 9]


def _clipped_positions() -> list[tuple[int, int]]:
    patch = _make_box_patch(1, 2)
    return [(r0, c0) for r0, c0 in _all_positions() if _visible_area(patch, r0, c0) in (4, 6)]


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
                if (nr, nc) in visited:
                    continue
                if blocked_grid[nr, nc] != BG:
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
        if blocked_grid[end[0], end[1]] != BG:
            return None
        if _too_thick(end[0], end[1], visited):
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


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    remaining = list(range(2, num_sprites))
    max_sources = min(2, len(remaining))
    choices = [n for n in SOURCE_COUNTS if n <= max_sources]
    if not choices:
        return None
    num_sources = random.choice(choices)
    random.shuffle(remaining)
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        dest_center_colors = [border_colors[source_indices[0]]] * 2
    else:
        dest_center_colors = [border_colors[source_indices[0]], border_colors[source_indices[1]]]

    source_center_colors = []
    for j, si in enumerate(source_indices):
        forbidden = {BG, GREY, border_colors[si]}
        if num_sources == 1:
            forbidden.update(dest_center_colors)
        else:
            forbidden.add(border_colors[j])
        candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        source_center_colors.append(random.choice(candidates))

    distractor_center_colors = []
    for di in distractor_indices:
        candidates = [c for c in range(1, 10) if c not in {GREY, border_colors[di]}]
        distractor_center_colors.append(random.choice(candidates))

    output_dest_centers = (source_center_colors * 2)[:2] if num_sources == 1 else source_center_colors[:2]

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
    }


def _place_destinations(
    planning_grid: np.ndarray, border_colors: list[int], output_dest_centers: list[int]
) -> dict[int, tuple[int, int, np.ndarray]] | None:
    placements: dict[int, tuple[int, int, np.ndarray]] = {}
    pool = _full_positions()
    random.shuffle(pool)
    for idx in range(2):
        patch = _make_box_patch(border_colors[idx], output_dest_centers[idx])
        placed = False
        for r0, c0 in pool:
            if _overlaps(planning_grid, patch, r0, c0):
                continue
            if placements:
                prev = placements[idx - 1]
                prev_center = (prev[0] + 1, prev[1] + 1)
                cur_center = (r0 + 1, c0 + 1)
                if abs(cur_center[0] - prev_center[0]) + abs(cur_center[1] - prev_center[1]) < 4:
                    continue
            _place_sprite(planning_grid, patch, r0, c0)
            placements[idx] = (r0, c0, patch)
            placed = True
            break
        if not placed:
            return None
    return placements


def _place_small_sprite(
    planning_grid: np.ndarray,
    border_color: int,
    center_color: int,
    min_port_dist: int,
    first_port: tuple[int, int] | None = None,
) -> tuple[int, int, np.ndarray, tuple[int, int]] | None:
    patch = _make_box_patch(border_color, center_color)
    pool = _clipped_positions()
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(planning_grid, patch, r0, c0):
            continue
        _place_sprite(planning_grid, patch, r0, c0)
        cr, cc = r0 + 1, c0 + 1
        port = _find_port_dir(planning_grid, cr, cc, border_color)
        if port is None:
            _erase_sprite(planning_grid, patch, r0, c0)
            continue
        if first_port is not None:
            dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
            if dist < min_port_dist:
                _erase_sprite(planning_grid, patch, r0, c0)
                continue
        return r0, c0, patch, port
    return None


def _build_role_specs(border_colors: list[int], colors: dict) -> dict[int, dict]:
    specs = {
        0: {"role": "dest", "border": border_colors[0], "input_center": colors["dest_center_colors"][0],
            "output_center": colors["output_dest_centers"][0]},
        1: {"role": "dest", "border": border_colors[1], "input_center": colors["dest_center_colors"][1],
            "output_center": colors["output_dest_centers"][1]},
    }
    for j, si in enumerate(colors["source_indices"]):
        specs[si] = {
            "role": "source",
            "border": border_colors[si],
            "input_center": colors["source_center_colors"][j],
            "output_center": None,
        }
    for j, di in enumerate(colors["distractor_indices"]):
        specs[di] = {
            "role": "distractor",
            "border": border_colors[di],
            "input_center": colors["distractor_center_colors"][j],
            "output_center": None,
        }
    return specs


def _gen_attempt() -> np.ndarray | None:
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample([c for c in range(1, 10) if c != GREY], num_sprites)
    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    specs = _build_role_specs(border_colors, colors)
    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    planning_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_places = _place_destinations(planning_grid, border_colors, colors["output_dest_centers"])
    if dest_places is None:
        return None

    remaining = [idx for idx in range(num_sprites) if idx not in (0, 1)]
    random.shuffle(remaining)
    anchor_indices = remaining[:2]
    small_places: dict[int, tuple[int, int, np.ndarray]] = {}
    anchor_ports: dict[int, tuple[int, int]] = {}

    first_anchor = _place_small_sprite(
        planning_grid,
        specs[anchor_indices[0]]["border"],
        specs[anchor_indices[0]]["input_center"],
        min_port_dist,
    )
    if first_anchor is None:
        return None
    r0, c0, patch, port = first_anchor
    small_places[anchor_indices[0]] = (r0, c0, patch)
    anchor_ports[anchor_indices[0]] = port

    second_anchor = _place_small_sprite(
        planning_grid,
        specs[anchor_indices[1]]["border"],
        specs[anchor_indices[1]]["input_center"],
        min_port_dist,
        first_port=port,
    )
    if second_anchor is None:
        return None
    r0, c0, patch, port2 = second_anchor
    small_places[anchor_indices[1]] = (r0, c0, patch)
    anchor_ports[anchor_indices[1]] = port2

    for idx in remaining[2:]:
        result = _place_small_sprite(
            planning_grid,
            specs[idx]["border"],
            specs[idx]["input_center"],
            min_port_dist=0,
        )
        if result is None:
            return None
        r0, c0, patch, _port = result
        small_places[idx] = (r0, c0, patch)

    path_cells = _gen_winding_path(
        planning_grid,
        anchor_ports[anchor_indices[0]],
        anchor_ports[anchor_indices[1]],
        target_len,
    )
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for idx in (0, 1):
        r0, c0, _patch = dest_places[idx]
        out_patch = _make_box_patch(specs[idx]["border"], specs[idx]["output_center"])
        _place_sprite(output_grid, out_patch, r0, c0)

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    input_grid = output_grid.copy()
    for idx in (0, 1):
        r0, c0, _patch = dest_places[idx]
        in_patch = _make_box_patch(specs[idx]["border"], specs[idx]["input_center"])
        _erase_sprite(input_grid, _make_box_patch(specs[idx]["border"], specs[idx]["output_center"]), r0, c0)
        _place_sprite(input_grid, in_patch, r0, c0)

    for idx in remaining:
        r0, c0, _patch = small_places[idx]
        in_patch = _make_box_patch(specs[idx]["border"], specs[idx]["input_center"])
        _place_sprite(input_grid, in_patch, r0, c0)

    areas = sorted(sp["area"] for sp in _detect_bordered_sprites(input_grid, BG, GREY))
    if areas[-2:] != [9, 9]:
        return None
    if any(area >= 9 for area in areas[:-2]):
        return None

    try:
        out = main(input_grid)
    except Exception:
        return None
    if np.array_equal(out, output_grid):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
