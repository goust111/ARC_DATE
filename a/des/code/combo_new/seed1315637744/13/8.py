import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = grid == color
        adjacency = int(np.count_nonzero(mask[:, 1:] & mask[:, :-1]))
        adjacency += int(np.count_nonzero(mask[1:, :] & mask[:-1, :]))
        if adjacency == 0:
            continue
        pixel_count = int(np.count_nonzero(mask))
        score = (adjacency, pixel_count)
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None, "expected a wall color"
    return int(best_color)


def _label_regions(grid: np.ndarray, wall_color: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    region_id = np.full((h, w), -1, dtype=int)
    regions: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == wall_color or region_id[r, c] != -1:
                continue
            stack = [(r, c)]
            region_id[r, c] = len(regions)
            coords: list[tuple[int, int]] = []
            touches_border = False

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if int(grid[nr, nc]) == wall_color or region_id[nr, nc] != -1:
                        continue
                    region_id[nr, nc] = len(regions)
                    stack.append((nr, nc))

            regions.append({"coords": coords, "closed": not touches_border})

    return region_id, regions


def _winner_color(
    counts: dict[int, int],
    region_seed_rows: dict[int, int],
    region_seed_cols: dict[int, int],
    global_seed_rows: dict[int, int],
    global_seed_cols: dict[int, int],
) -> int:
    best_color = None
    best_key = None
    for color, ray_count in counts.items():
        top_row = int(region_seed_rows.get(color, global_seed_rows[color]))
        left_col = int(region_seed_cols.get(color, global_seed_cols[color]))
        key = (int(ray_count), -top_row, -left_col)
        if best_key is None or key > best_key:
            best_key = key
            best_color = int(color)
    assert best_color is not None
    return int(best_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed rays until grey walls, then fill each enclosed region by the color with the most rays entering it.

    Input structure:
    - The background is the dominant color.
    - Grey wall pixels form horizontal and vertical barriers that partition the canvas into open space and enclosed regions.
    - Non-background, non-grey singleton pixels are colored seeds placed in some enclosed regions.

    Rule:
    - Each seed emits four straight rays of its own color: up, down, left, and right.
    - A ray continues through non-wall cells until it reaches a wall or leaves the grid; it paints only background cells and does not overwrite walls or seed pixels.
    - For every enclosed region, count how many individual rays of each color pass through that region.
    - Fill the region's remaining background cells with the color that has the highest ray count there; if counts tie, choose the color whose seed in that region is highest.

    Key insight:
    - The deciding quantity is the number of rays entering a closed region, not the number of painted pixels, so seed placement near walls changes the winner by changing how many directions actually extend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _majority_color(g)
    wall_color = _infer_wall_color(g, bg)
    region_id, regions = _label_regions(g, wall_color)

    seeds: list[tuple[int, int, int]] = []
    global_seed_rows: dict[int, int] = {}
    global_seed_cols: dict[int, int] = {}
    region_seed_rows: list[dict[int, int]] = [dict() for _ in regions]
    region_seed_cols: list[dict[int, int]] = [dict() for _ in regions]

    for r, c in np.argwhere((g != bg) & (g != wall_color)):
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        seeds.append((rr, cc, color))
        global_seed_rows[color] = min(rr, global_seed_rows.get(color, rr))
        global_seed_cols[color] = min(cc, global_seed_cols.get(color, cc))
        rid = int(region_id[rr, cc])
        if rid >= 0:
            region_seed_rows[rid][color] = min(rr, region_seed_rows[rid].get(color, rr))
            region_seed_cols[rid][color] = min(cc, region_seed_cols[rid].get(color, cc))

    seeds.sort()
    out = g.copy()
    region_counts: list[dict[int, int]] = [defaultdict(int) for _ in regions]

    for sr, sc, color in seeds:
        for dr, dc in DIR4:
            path: list[tuple[int, int]] = []
            rr, cc = int(sr + dr), int(sc + dc)
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != wall_color:
                path.append((int(rr), int(cc)))
                rr += dr
                cc += dc
            if not path:
                continue

            touched_regions: set[int] = set()
            for rr, cc in path:
                rid = int(region_id[rr, cc])
                if rid >= 0 and bool(regions[rid]["closed"]):
                    touched_regions.add(rid)
                if int(g[rr, cc]) == bg:
                    out[rr, cc] = int(color)

            for rid in touched_regions:
                region_counts[rid][int(color)] += 1

    for rid, region in enumerate(regions):
        if not bool(region["closed"]) or not region_counts[rid]:
            continue
        winner = _winner_color(
            region_counts[rid],
            region_seed_rows[rid],
            region_seed_cols[rid],
            global_seed_rows,
            global_seed_cols,
        )
        for rr, cc in region["coords"]:
            if int(out[rr, cc]) == bg:
                out[rr, cc] = int(winner)

    return out


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

REGION_COUNT_RANGE = (2, 4)
REGION_SIZE_RANGE = (3, 5)
MARGIN_RANGE = (1, 3)
GEN_RETRIES = 400


def _build_partition(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    n_row_regions = rng.randint(*REGION_COUNT_RANGE)
    n_col_regions = rng.randint(*REGION_COUNT_RANGE)
    row_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(n_row_regions)]
    col_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(n_col_regions)]

    frame_h = sum(row_sizes) + n_row_regions + 1
    frame_w = sum(col_sizes) + n_col_regions + 1
    top_margin = rng.randint(*MARGIN_RANGE)
    bottom_margin = rng.randint(*MARGIN_RANGE)
    left_margin = rng.randint(*MARGIN_RANGE)
    right_margin = rng.randint(*MARGIN_RANGE)

    h = frame_h + top_margin + bottom_margin
    w = frame_w + left_margin + right_margin
    assert h <= 30 and w <= 30

    grid = np.full((h, w), BG, dtype=int)

    h_lines = [top_margin]
    for size in row_sizes:
        h_lines.append(int(h_lines[-1] + size + 1))
    v_lines = [left_margin]
    for size in col_sizes:
        v_lines.append(int(v_lines[-1] + size + 1))

    for row in h_lines:
        grid[int(row), v_lines[0] : v_lines[-1] + 1] = WALL
    for col in v_lines:
        grid[h_lines[0] : h_lines[-1] + 1, int(col)] = WALL

    regions: list[tuple[int, int, int, int]] = []
    for ri in range(n_row_regions):
        for ci in range(n_col_regions):
            r0 = int(h_lines[ri] + 1)
            r1 = int(h_lines[ri + 1] - 1)
            c0 = int(v_lines[ci] + 1)
            c1 = int(v_lines[ci + 1] - 1)
            regions.append((r0, c0, r1, c1))

    return grid, regions


def _slot_map(region: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = region
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    return {
        "corner_tl": (r0, c0),
        "corner_tr": (r0, c1),
        "corner_bl": (r1, c0),
        "corner_br": (r1, c1),
        "top_mid": (r0, mid_c),
        "bottom_mid": (r1, mid_c),
        "left_mid": (mid_r, c0),
        "right_mid": (mid_r, c1),
        "center": (mid_r, mid_c),
    }


MODE_TEMPLATES: dict[str, list[list[str]]] = {
    "single": [
        ["center"],
        ["top_mid"],
        ["left_mid"],
        ["corner_tl"],
        ["corner_br"],
    ],
    "clear": [
        ["center", "corner_tl"],
        ["center", "top_mid"],
        ["top_mid", "corner_br"],
        ["center", "corner_tl", "corner_br"],
        ["center", "left_mid", "corner_br"],
    ],
    "tie": [
        ["corner_tl", "corner_br"],
        ["corner_tr", "corner_bl"],
        ["top_mid", "bottom_mid"],
    ],
}


def _place_seed_template(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    mode: str,
    rng: random.Random,
) -> bool:
    if mode == "empty":
        return False

    slots = _slot_map(region)
    template = list(rng.choice(MODE_TEMPLATES[mode]))
    positions = [slots[name] for name in template]
    if len(set(positions)) != len(positions):
        return False

    colors = rng.sample(PALETTE, len(positions))
    for color, (r, c) in zip(colors, positions):
        grid[int(r), int(c)] = int(color)
    return True


def _choose_region_modes(rng: random.Random, region_count: int) -> list[str]:
    modes = ["empty"] * region_count
    order = list(range(region_count))
    rng.shuffle(order)

    special_mode = rng.choice(["clear", "tie"])
    modes[order[0]] = special_mode
    if region_count > 1:
        modes[order[1]] = rng.choice(["single", "clear"])

    weights = {
        "empty": 1,
        "single": 3,
        "clear": 3,
        "tie": 2,
    }
    choices = list(weights)
    choice_weights = [weights[name] for name in choices]
    for idx in order[2:]:
        modes[idx] = rng.choices(choices, weights=choice_weights, k=1)[0]

    if all(mode == "empty" for mode in modes):
        modes[order[0]] = "single"
    return modes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid, regions = _build_partition(rng)
    modes = _choose_region_modes(rng, len(regions))

    seeded_regions = 0
    for region, mode in zip(regions, modes):
        seeded_regions += int(_place_seed_template(grid, region, mode, rng))

    if seeded_regions == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    if np.count_nonzero(out != grid) < 6:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
