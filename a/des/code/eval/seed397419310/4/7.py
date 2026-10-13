from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("TL", "TR", "BL", "BR")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _scan_corner_legend(grid: np.ndarray, bg: int, corner: str) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    if corner == "TL":
        col = 0
        rows = list(range(h))
    elif corner == "TR":
        col = w - 1
        rows = list(range(h))
    elif corner == "BL":
        col = 0
        rows = list(range(h - 1, -1, -1))
    else:
        col = w - 1
        rows = list(range(h - 1, -1, -1))

    if int(grid[rows[0], col]) == bg:
        return None

    cells: list[tuple[int, int]] = []
    for r in rows:
        if int(grid[r, col]) == bg:
            break
        cells.append((int(r), int(col)))

    if not cells:
        return None
    if corner in {"BL", "BR"}:
        cells.reverse()
    return cells


def _find_legend(grid: np.ndarray, bg: int) -> tuple[str, list[tuple[int, int]], list[int]]:
    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    for corner in CORNERS:
        cells = _scan_corner_legend(grid, bg, corner)
        if cells is not None:
            candidates.append((corner, cells))

    assert len(candidates) == 1, "expected exactly one corner legend"
    corner, cells = candidates[0]
    colors = [int(grid[r, c]) for r, c in cells]
    return corner, cells, colors


def _parse_tower(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, int]:
    color = int(grid[coords[0]])
    assert all(int(grid[r, c]) == color for r, c in coords), "tower must be monochrome"

    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    assert len(rows) >= 2, "tower height must be at least 2"
    assert rows == list(range(rows[0], rows[-1] + 1)), "pillar rows must be contiguous"
    assert len(cols) == 2, "tower must occupy exactly two adjacent columns"

    row_to_cols = {
        r: sorted(int(c) for rr, c in coords if int(rr) == r)
        for r in rows
    }
    top_r = rows[0]
    top_cols = row_to_cols[top_r]
    assert len(top_cols) == 2 and top_cols[1] == top_cols[0] + 1, "top row must have two adjacent pixels"
    for r in rows[1:]:
        assert len(row_to_cols[r]) == 1, "only the top row may contain the antenna arm"

    pillar_col = row_to_cols[rows[1]][0]
    assert all(row_to_cols[r][0] == pillar_col for r in rows[1:]), "pillar must be vertical"
    assert pillar_col in top_cols, "top row must contain the pillar pixel"
    arm_col = top_cols[1] if top_cols[0] == pillar_col else top_cols[0]
    direction = 1 if arm_col > pillar_col else -1
    height = len(rows)

    assert len(coords) == height + 1, "tower must be a pillar plus one arm pixel"
    return {
        "top_r": int(top_r),
        "pillar_c": int(pillar_col),
        "height": int(height),
        "direction": int(direction),
        "color": int(color),
    }


def _coverage_counts(grid: np.ndarray, bg: int, towers: list[dict[str, int]]) -> np.ndarray:
    h, w = grid.shape
    counts = np.zeros((h, w), dtype=int)
    bg_mask = grid == bg

    for tower in towers:
        top_r = int(tower["top_r"])
        pillar_c = int(tower["pillar_c"])
        height = int(tower["height"])
        direction = int(tower["direction"])

        for dist in range(1, height + 1):
            cc = pillar_c + direction * dist
            if not (0 <= cc < w):
                continue
            r0 = max(0, top_r - dist)
            r1 = min(h - 1, top_r + dist)
            counts[r0 : r1 + 1, cc] += bg_mask[r0 : r1 + 1, cc].astype(int)

    return counts


def _paint_counts(grid: np.ndarray, bg: int, legend_colors: list[int], counts: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    covered = (grid == bg) & (counts > 0)
    if np.any(covered):
        palette = np.asarray(legend_colors, dtype=int)
        capped = np.minimum(counts[covered], len(palette)) - 1
        out[covered] = palette[capped]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    根据角落图例和带方向的信号塔，把背景区域按覆盖次数着色。
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    _legend_corner, legend_cells, legend_colors = _find_legend(g, bg)

    tower_mask = g != bg
    for r, c in legend_cells:
        tower_mask[r, c] = False
    tower_components = _find_components(tower_mask)
    assert tower_components, "expected at least one tower"
    towers = [_parse_tower(g, comp) for comp in tower_components]

    counts = _coverage_counts(g, bg, towers)
    return _paint_counts(g, bg, legend_colors, counts)


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (15, 22)
GRID_WIDTH_RANGE = (18, 28)
TOWER_COUNT_RANGE = (2, 5)
TOWER_HEIGHT_RANGE = (2, 5)
LEGEND_LEN_MAX = 4
COLOR_POOL = tuple(range(10))
PLACEMENT_ATTEMPTS = 200


def _legend_cells(shape: tuple[int, int], corner: str, length: int) -> list[tuple[int, int]]:
    h, w = shape
    if corner == "TL":
        return [(r, 0) for r in range(length)]
    if corner == "TR":
        return [(r, w - 1) for r in range(length)]
    if corner == "BL":
        return [(h - length + r, 0) for r in range(length)]
    return [(h - length + r, w - 1) for r in range(length)]


def _coords_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _tower_cells(top_r: int, pillar_c: int, height: int, direction: int) -> list[tuple[int, int]]:
    cells = [(top_r + dr, pillar_c) for dr in range(height)]
    cells.append((top_r, pillar_c + direction))
    return cells


def _sector_cells(top_r: int, pillar_c: int, height: int, direction: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dist in range(1, height + 1):
        cc = pillar_c + direction * dist
        for rr in range(top_r - dist, top_r + dist + 1):
            cells.append((rr, cc))
    return cells


def _candidate_valid(
    shape: tuple[int, int],
    legend_column: int,
    legend_set: set[tuple[int, int]],
    blocked: np.ndarray,
    top_r: int,
    pillar_c: int,
    height: int,
    direction: int,
    target: tuple[int, int] | None,
) -> bool:
    h, w = shape
    tower = _tower_cells(top_r, pillar_c, height, direction)
    sector = _sector_cells(top_r, pillar_c, height, direction)

    for r, c in tower + sector:
        if not (0 <= r < h and 0 <= c < w):
            return False
    if target is not None and target in tower:
        return False
    if any(blocked[r, c] for r, c in tower):
        return False
    if any(c == legend_column for _r, c in sector):
        return False
    if any((r, c) in legend_set for r, c in sector):
        return False
    return True


def _place_overlap_towers(
    shape: tuple[int, int],
    legend_cells: list[tuple[int, int]],
    heights: list[int],
    directions: list[int],
) -> tuple[list[dict[str, int]], tuple[int, int]] | None:
    h, w = shape
    max_h = max(heights)
    if h < 2 * max_h + 1 or w < 2 * max_h + 4:
        return None

    target_r = random.randint(max_h, h - max_h - 1)
    target_c = random.randint(max_h + 1, w - max_h - 2)
    target = (target_r, target_c)

    legend_set = set(legend_cells)
    legend_column = legend_cells[0][1]
    occupied = _coords_mask(shape, legend_cells)
    blocked = _dilate8(occupied)
    placed: list[dict[str, int] | None] = [None] * len(heights)

    order = sorted(range(len(heights)), key=lambda idx: heights[idx], reverse=True)
    for idx in order:
        height = int(heights[idx])
        direction = int(directions[idx])
        ok = False
        for _ in range(PLACEMENT_ATTEMPTS):
            dist = random.randint(1, height)
            top_lo = max(height, target_r - dist)
            top_hi = min(h - height - 1, target_r + dist)
            if top_lo > top_hi:
                continue
            top_r_i = random.randint(top_lo, top_hi)
            pillar_c = target_c - direction * dist
            if not _candidate_valid(
                shape,
                legend_column,
                legend_set,
                blocked,
                top_r_i,
                pillar_c,
                height,
                direction,
                target,
            ):
                continue

            placed[idx] = {
                "top_r": int(top_r_i),
                "pillar_c": int(pillar_c),
                "height": int(height),
                "direction": int(direction),
            }
            for r, c in _tower_cells(top_r_i, pillar_c, height, direction):
                occupied[r, c] = True
            blocked = _dilate8(occupied)
            ok = True
            break
        if not ok:
            return None

    return [tower for tower in placed if tower is not None], target


def _place_scatter_towers(
    shape: tuple[int, int],
    legend_cells: list[tuple[int, int]],
    heights: list[int],
    directions: list[int],
) -> list[dict[str, int]] | None:
    h, w = shape
    legend_set = set(legend_cells)
    legend_column = legend_cells[0][1]
    occupied = _coords_mask(shape, legend_cells)
    blocked = _dilate8(occupied)
    towers: list[dict[str, int]] = []

    order = sorted(range(len(heights)), key=lambda idx: heights[idx], reverse=True)
    for idx in order:
        height = int(heights[idx])
        direction = int(directions[idx])
        ok = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top_r = random.randint(height, h - height - 1)
            if direction == 1:
                c_lo, c_hi = 1, w - height - 1
            else:
                c_lo, c_hi = height, w - 2
            if c_lo > c_hi:
                continue
            pillar_c = random.randint(c_lo, c_hi)
            if not _candidate_valid(
                shape,
                legend_column,
                legend_set,
                blocked,
                top_r,
                pillar_c,
                height,
                direction,
                None,
            ):
                continue

            towers.append(
                {
                    "top_r": int(top_r),
                    "pillar_c": int(pillar_c),
                    "height": int(height),
                    "direction": int(direction),
                }
            )
            for r, c in _tower_cells(top_r, pillar_c, height, direction):
                occupied[r, c] = True
            blocked = _dilate8(occupied)
            ok = True
            break
        if not ok:
            return None

    return towers


def _render_input(
    shape: tuple[int, int],
    bg: int,
    legend_cells: list[tuple[int, int]],
    legend_colors: list[int],
    towers: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for (r, c), color in zip(legend_cells, legend_colors):
        grid[r, c] = int(color)
    for tower in towers:
        color = int(tower["color"])
        for r, c in _tower_cells(
            int(tower["top_r"]),
            int(tower["pillar_c"]),
            int(tower["height"]),
            int(tower["direction"]),
        ):
            grid[r, c] = color
    return grid


def _sample_mode() -> str:
    roll = random.random()
    if roll < 0.35:
        return "clipped_overlap"
    if roll < 0.7:
        return "plain_overlap"
    return "scatter"


def _sample_tower_count(mode: str) -> int:
    if mode == "clipped_overlap":
        return random.randint(3, TOWER_COUNT_RANGE[1])
    if mode == "plain_overlap":
        return random.randint(2, min(4, TOWER_COUNT_RANGE[1]))
    return random.randint(*TOWER_COUNT_RANGE)


def _sample_legend_len(mode: str, n_towers: int) -> int:
    if mode == "clipped_overlap":
        return random.randint(1, min(LEGEND_LEN_MAX, n_towers - 1))
    if mode == "plain_overlap":
        return random.randint(n_towers, LEGEND_LEN_MAX)
    return random.randint(2, LEGEND_LEN_MAX)


def _sample_directions(n_towers: int, force_both: bool) -> list[int]:
    directions = [random.choice((-1, 1)) for _ in range(n_towers)]
    if force_both and n_towers >= 2 and len(set(directions)) == 1:
        directions[0] *= -1
    return directions


def _sample_colors(n_towers: int, legend_len: int) -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    bg = colors.pop()
    multicolor_towers = random.random() < 0.35

    if multicolor_towers:
        tower_colors = [colors.pop() for _ in range(n_towers)]
    else:
        tower_color = colors.pop()
        tower_colors = [tower_color] * n_towers

    legend_colors = [colors.pop() for _ in range(legend_len)]
    return int(bg), [int(c) for c in tower_colors], [int(c) for c in legend_colors]


def _gen_attempt() -> np.ndarray | None:
    mode = _sample_mode()
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    n_towers = _sample_tower_count(mode)
    legend_len = _sample_legend_len(mode, n_towers)
    corner = random.choice(CORNERS)

    bg, tower_colors, legend_colors = _sample_colors(n_towers, legend_len)
    legend_cells = _legend_cells((h, w), corner, legend_len)
    heights = [random.randint(*TOWER_HEIGHT_RANGE) for _ in range(n_towers)]
    directions = _sample_directions(n_towers, force_both=(mode != "scatter" or random.random() < 0.5))

    if mode == "scatter":
        tower_specs = _place_scatter_towers((h, w), legend_cells, heights, directions)
        if tower_specs is None:
            return None
        target = None
    else:
        placed = _place_overlap_towers((h, w), legend_cells, heights, directions)
        if placed is None:
            return None
        tower_specs, target = placed

    towers = []
    for spec, color in zip(tower_specs, tower_colors):
        spec = dict(spec)
        spec["color"] = int(color)
        towers.append(spec)

    grid = _render_input((h, w), bg, legend_cells, legend_colors, towers)
    oracle = _paint_counts(grid, bg, legend_colors, _coverage_counts(grid, bg, towers))

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, oracle):
        return None
    if np.array_equal(grid, oracle):
        return None
    if _get_bg(grid) != bg:
        return None

    if mode != "scatter":
        assert target is not None
        tr, tc = target
        overlap = int(_coverage_counts(grid, bg, towers)[tr, tc])
        if overlap != n_towers:
            return None
        if mode == "clipped_overlap" and overlap <= legend_len:
            return None
        if mode == "plain_overlap" and overlap > legend_len:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
