import random

from collections import deque
import numpy as np


BG = 0
GREY = 5
AZURE = 8
FLOOR_COLORS = (1, 2, 3, 4, 6, 7, 9)

# --------------- generator config ---------------
GEN_RETRIES = 300
BUILDING_COUNT_CHOICES = (3, 4)
PRIORITY_LEN_RANGE = (4, 6)
INTERIOR_WIDTH_RANGE = (2, 5)
MAX_FLOORS = 6
TOP_PAD_RANGE = (0, 3)
BOTTOM_BLANK_ROWS_RANGE = (1, 3)
SIDE_MARGIN_RANGE = (1, 3)
BETWEEN_GAP_RANGE = (2, 4)
EXTRA_WIDTH_RANGE = (0, 4)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_priority_bar(grid: np.ndarray) -> tuple[int, list[int], dict[int, int]]:
    h, w = grid.shape
    for r in range(h - 1, -1, -1):
        cols = [c for c in range(w) if int(grid[r, c]) != BG]
        if len(cols) < 2:
            continue
        c0, c1 = cols[0], cols[-1]
        values = [int(v) for v in grid[r, c0 : c1 + 1]]
        if any(v == BG for v in values):
            continue
        if any(v in (GREY, AZURE) for v in values):
            continue
        if len(set(values)) != len(values):
            continue
        if c0 > 0 and np.any(grid[r, :c0] != BG):
            continue
        if c1 + 1 < w and np.any(grid[r, c1 + 1 :] != BG):
            continue
        return r, values, {color: idx for idx, color in enumerate(values)}
    raise AssertionError("priority bar not found")


def _extract_building(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int] | None:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width < 3 or height < 3 or height % 2 == 0:
        return None

    floors: list[int] = []
    for r in range(r0, r1 + 1):
        row = [int(v) for v in grid[r, c0 : c1 + 1]]
        if (r - r0) % 2 == 0:
            if any(v != GREY for v in row):
                return None
            continue
        if row[0] != GREY or row[-1] != GREY:
            return None
        color = row[1]
        if color in (BG, GREY, AZURE):
            return None
        if any(v != color for v in row[1:-1]):
            return None
        floors.append(color)
    return floors


def _find_ruler_length(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, _r1, c1 = bbox
    candidates: list[int] = []
    for c in range(c0 + 1, c1):
        length = 0
        rr = r0 - 1
        while rr >= 0 and int(grid[rr, c]) == AZURE:
            length += 1
            rr -= 1
        if length > 0:
            candidates.append(length)
    assert len(candidates) == 1, "each building needs exactly one ruler"
    return candidates[0]


def _find_buildings(grid: np.ndarray, priority_row: int) -> list[dict]:
    buildings: list[dict] = []
    grey_mask = grid == GREY
    for comp in _components(grey_mask):
        bbox = _bbox(comp)
        r0, _c0, r1, _c1 = bbox
        if r1 >= priority_row:
            continue
        floors = _extract_building(grid, bbox)
        if floors is None:
            continue
        buildings.append(
            {
                "bbox": bbox,
                "floors": floors,
                "target": _find_ruler_length(grid, bbox),
            }
        )
    assert buildings, "no buildings found"
    return sorted(buildings, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _draw_building(grid: np.ndarray, r1: int, c0: int, c1: int, floors: list[int]) -> int:
    top = r1 - 2 * len(floors)
    for idx, color in enumerate(floors):
        wall_r = top + 2 * idx
        floor_r = wall_r + 1
        grid[wall_r, c0 : c1 + 1] = GREY
        grid[floor_r, c0] = GREY
        grid[floor_r, c1] = GREY
        grid[floor_r, c0 + 1 : c1] = color
    grid[r1, c0 : c1 + 1] = GREY
    return top


def _transform_floors(floors: list[int], target: int, priority: dict[int, int]) -> list[int]:
    current = len(floors)
    assert current >= 1 and target >= 1
    if current > target:
        keep = sorted(range(current), key=lambda idx: (priority[floors[idx]], idx))[:target]
        keep_set = set(keep)
        return [color for idx, color in enumerate(floors) if idx in keep_set]
    if current < target:
        lowest_rank = max(priority[color] for color in floors)
        assert priority[floors[-1]] == lowest_rank, "bottom floor must be the lowest priority"
        return floors + [floors[-1]] * (target - current)
    return list(floors)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    priority_row, _priority_colors, priority = _find_priority_bar(g)
    buildings = _find_buildings(g, priority_row)

    out = g.copy()
    out[out == AZURE] = BG

    for building in buildings:
        r0, c0, r1, c1 = building["bbox"]
        new_floors = _transform_floors(building["floors"], building["target"], priority)
        new_top = r1 - 2 * len(new_floors)
        clear_top = min(r0, new_top)
        out[clear_top : r1 + 1, c0 : c1 + 1] = BG
        _draw_building(out, r1, c0, c1, new_floors)

    return out


def _sample_branches(rng: random.Random, count: int) -> list[str]:
    branches = ["compress", "equal", "stretch"]
    if count == 4:
        branches.append(rng.choice(branches))
    rng.shuffle(branches)
    return branches


def _sample_counts(rng: random.Random, branch: str) -> tuple[int, int]:
    if branch == "compress":
        current = rng.randint(3, MAX_FLOORS)
        target = rng.randint(1, current - 1)
    elif branch == "stretch":
        current = rng.randint(1, MAX_FLOORS - 2)
        target = rng.randint(current + 1, MAX_FLOORS)
    else:
        current = rng.randint(1, MAX_FLOORS)
        target = current
    return current, target


def _sample_floor_stack(rng: random.Random, priority_colors: list[int], current: int) -> list[int]:
    if current == 1:
        return [rng.choice(priority_colors)]
    bottom_rank = rng.randint(1, len(priority_colors) - 1)
    upper_pool = priority_colors[:bottom_rank]
    upper = [rng.choice(upper_pool) for _ in range(current - 1)]
    if len(upper) > 1:
        rng.shuffle(upper)
    return upper + [priority_colors[bottom_rank]]


def _gen_attempt(rng: random.Random) -> np.ndarray:
    n_buildings = rng.choice(BUILDING_COUNT_CHOICES)
    priority_len = rng.randint(PRIORITY_LEN_RANGE[0], PRIORITY_LEN_RANGE[1])
    priority_colors = rng.sample(list(FLOOR_COLORS), priority_len)
    branches = _sample_branches(rng, n_buildings)

    specs: list[dict] = []
    needs: list[int] = []
    widths: list[int] = []
    for branch in branches:
        current, target = _sample_counts(rng, branch)
        floors = _sample_floor_stack(rng, priority_colors, current)
        interior_width = rng.randint(INTERIOR_WIDTH_RANGE[0], INTERIOR_WIDTH_RANGE[1])
        total_width = interior_width + 2
        widths.append(total_width)
        needs.append(max(2 * target, 2 * current + target))
        specs.append(
            {
                "branch": branch,
                "current": current,
                "target": target,
                "floors": floors,
                "width": total_width,
            }
        )

    top_pad = rng.randint(TOP_PAD_RANGE[0], TOP_PAD_RANGE[1])
    base_bottom = top_pad + max(needs)
    blank_rows = rng.randint(BOTTOM_BLANK_ROWS_RANGE[0], BOTTOM_BLANK_ROWS_RANGE[1])
    height = base_bottom + blank_rows + 2

    left_margin = rng.randint(SIDE_MARGIN_RANGE[0], SIDE_MARGIN_RANGE[1])
    right_margin = rng.randint(SIDE_MARGIN_RANGE[0], SIDE_MARGIN_RANGE[1])
    gaps = [rng.randint(BETWEEN_GAP_RANGE[0], BETWEEN_GAP_RANGE[1]) for _ in range(n_buildings - 1)]
    layout_width = left_margin + sum(widths) + sum(gaps) + right_margin
    min_width = max(layout_width, priority_len + 2)
    width = min_width + rng.randint(EXTRA_WIDTH_RANGE[0], EXTRA_WIDTH_RANGE[1])
    x_shift = rng.randint(0, width - layout_width)

    grid = np.full((height, width), BG, dtype=int)
    cursor = left_margin + x_shift
    for idx, spec in enumerate(specs):
        c0 = cursor
        c1 = c0 + spec["width"] - 1
        top = _draw_building(grid, base_bottom, c0, c1, spec["floors"])
        ruler_c = c0 + 1 + (spec["width"] - 2) // 2
        ruler_top = top - spec["target"]
        if ruler_top < 0:
            return None
        grid[ruler_top:top, ruler_c] = AZURE
        cursor = c1 + 1
        if idx < len(gaps):
            cursor += gaps[idx]

    bar_c0 = rng.randint(0, width - priority_len)
    grid[height - 1, bar_c0 : bar_c0 + priority_len] = priority_colors
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate a valid input")
