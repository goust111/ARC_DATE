import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 500
GRID_MIN = 18
GRID_MAX = 30
PLACEMENT_PAD = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue

            visited[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _is_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if component["area"] != 2 * (h + w) - 4:
        return False

    mask = _component_mask(component)
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    if h > 2 and w > 2 and mask[1:-1, 1:-1].any():
        return False
    return True


def _is_core(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h == w and h in (1, 2) and component["area"] == h * w


def _contains(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return r0 < ir0 and c0 < ic0 and ir1 < r1 and ic1 < c1


def _paint_group(output: np.ndarray, outer_bbox: tuple[int, int, int, int], depth: int, color: int) -> None:
    r0, c0, r1, c1 = outer_bbox
    if r1 - r0 <= 1 or c1 - c0 <= 1:
        return

    if depth % 2 == 1:
        output[r0 + 1 : r1, c0 + 1 : c1] = color
        return

    for r in range(r0 + 1, r1):
        fill_color = color if (r - (r0 + 1)) % 2 == 0 else BG
        output[r, c0 + 1 : c1] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each nested core color to the outermost frame interior.

    Odd frame depth -> solid fill.
    Even frame depth -> horizontal stripes starting at the first interior row.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    components = _find_components(g)
    frames = [comp for comp in components if _is_frame(comp)]
    cores = [comp for comp in components if _is_core(comp)]

    groups: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for core in cores:
        ancestors = [frame for frame in frames if _contains(frame["bbox"], core["bbox"])]
        if not ancestors:
            continue
        ancestors.sort(key=lambda comp: _bbox_area(comp["bbox"]))
        outer_bbox = ancestors[-1]["bbox"]
        candidate = (len(ancestors), int(core["color"]))
        if outer_bbox not in groups or candidate[0] > groups[outer_bbox][0]:
            groups[outer_bbox] = candidate

    for outer_bbox, (depth, color) in groups.items():
        _paint_group(out, outer_bbox, depth, color)

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_depths(rng: random.Random, n_groups: int) -> list[int]:
    depths = [rng.choice((1, 3)), rng.choice((2, 4))]
    while len(depths) < n_groups:
        depths.append(rng.choice((1, 2, 3, 4)))
    rng.shuffle(depths)
    return depths


def _sample_group_spec(rng: random.Random, depth: int, crowded: bool) -> dict | None:
    core_size = rng.choice((1, 2))
    margin_choices = (1, 2) if crowded else (1, 2, 2, 3)
    row_insets = [rng.choice(margin_choices) for _ in range(depth)]
    col_insets = [rng.choice(margin_choices) for _ in range(depth)]

    outer_h = core_size + 2 * sum(row_insets)
    outer_w = core_size + 2 * sum(col_insets)
    if outer_h > 18 or outer_w > 18:
        return None

    colors = rng.sample(COLORS, depth + 1)
    return {
        "depth": depth,
        "core_size": core_size,
        "row_insets": row_insets,
        "col_insets": col_insets,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "colors": colors,
    }


def _place_group(grid: np.ndarray, occupied: np.ndarray, spec: dict, top: int, left: int) -> dict:
    depth = spec["depth"]
    cur_r0 = top
    cur_c0 = left
    cur_r1 = top + spec["outer_h"] - 1
    cur_c1 = left + spec["outer_w"] - 1

    for layer in range(depth):
        _draw_frame(grid, (cur_r0, cur_c0, cur_r1, cur_c1), int(spec["colors"][layer]))
        cur_r0 += int(spec["row_insets"][layer])
        cur_c0 += int(spec["col_insets"][layer])
        cur_r1 -= int(spec["row_insets"][layer])
        cur_c1 -= int(spec["col_insets"][layer])

    core_size = int(spec["core_size"])
    assert cur_r1 - cur_r0 + 1 == core_size
    assert cur_c1 - cur_c0 + 1 == core_size
    grid[cur_r0 : cur_r1 + 1, cur_c0 : cur_c1 + 1] = int(spec["colors"][-1])

    r0 = top
    c0 = left
    r1 = top + spec["outer_h"] - 1
    c1 = left + spec["outer_w"] - 1
    pad_r0 = max(0, r0 - PLACEMENT_PAD)
    pad_c0 = max(0, c0 - PLACEMENT_PAD)
    pad_r1 = min(grid.shape[0] - 1, r1 + PLACEMENT_PAD)
    pad_c1 = min(grid.shape[1] - 1, c1 + PLACEMENT_PAD)
    occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True

    placed = dict(spec)
    placed["outer_bbox"] = (r0, c0, r1, c1)
    placed["core_bbox"] = (cur_r0, cur_c0, cur_r1, cur_c1)
    placed["core_color"] = int(spec["colors"][-1])
    return placed


def _try_place_all(grid: np.ndarray, specs: list[dict], rng: random.Random) -> list[dict] | None:
    occupied = np.zeros_like(grid, dtype=bool)
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        h = int(spec["outer_h"])
        w = int(spec["outer_w"])
        max_top = grid.shape[0] - h
        max_left = grid.shape[1] - w
        if max_top < 0 or max_left < 0:
            return None

        success = False
        for _ in range(250):
            top = rng.randint(0, max_top)
            left = rng.randint(0, max_left)
            r1 = top + h - 1
            c1 = left + w - 1
            pad_r0 = max(0, top - PLACEMENT_PAD)
            pad_c0 = max(0, left - PLACEMENT_PAD)
            pad_r1 = min(grid.shape[0] - 1, r1 + PLACEMENT_PAD)
            pad_c1 = min(grid.shape[1] - 1, c1 + PLACEMENT_PAD)
            if occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                continue
            placed.append(_place_group(grid, occupied, spec, top, left))
            success = True
            break

        if not success:
            return None

    return placed


def _expected_output(grid: np.ndarray, placed_specs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in placed_specs:
        _paint_group(out, spec["outer_bbox"], int(spec["depth"]), int(spec["core_color"]))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_groups = rng.randint(2, 4)
    crowded = n_groups >= 4
    specs: list[dict] = []
    for depth in _sample_depths(rng, n_groups):
        spec = _sample_group_spec(rng, depth, crowded)
        if spec is None:
            return None
        specs.append(spec)

    total_area = sum(spec["outer_h"] * spec["outer_w"] for spec in specs)
    max_h = max(spec["outer_h"] for spec in specs)
    max_w = max(spec["outer_w"] for spec in specs)
    base = max(GRID_MIN, int(np.ceil(np.sqrt(total_area * 2.0))))
    h_lo = max(base, max_h + 2 * PLACEMENT_PAD + 1)
    w_lo = max(base, max_w + 2 * PLACEMENT_PAD + 1)
    if h_lo > GRID_MAX or w_lo > GRID_MAX:
        return None

    h = rng.randint(h_lo, GRID_MAX)
    w = rng.randint(w_lo, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    placed = _try_place_all(grid, specs, rng)
    if placed is None:
        return None

    expected = _expected_output(grid, placed)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None

    odd_present = any(int(spec["depth"]) % 2 == 1 for spec in placed)
    even_present = any(int(spec["depth"]) % 2 == 0 for spec in placed)
    if not (odd_present and even_present):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
