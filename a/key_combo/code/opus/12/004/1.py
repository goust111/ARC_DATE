import random
from collections import defaultdict

import numpy as np


BG = 0
OBJ_COLOR = 5
LEGEND_WIDTHS = (1, 2, 3, 4)
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend(bottom_row: np.ndarray) -> dict[int, int]:
    row = np.asarray(bottom_row, dtype=int)
    legend: dict[int, int] = {}
    c = 0
    while c < row.shape[0]:
        if int(row[c]) == BG:
            c += 1
            continue
        color = int(row[c])
        start = c
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
        width = c - start
        legend[width] = color
    return legend


def _measure_object(coords: np.ndarray) -> dict[str, int]:
    rows: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        rows[int(r)].append(int(c))

    top = min(rows)
    bottom = max(rows)
    span = bottom - top
    top_left = min(rows[top])
    bottom_left = min(rows[bottom])
    total_shear = bottom_left - top_left
    assert span > 0, "expected at least two object rows"
    assert total_shear > 0, "expected a left-leaning shear"
    assert total_shear % span == 0, "expected an integer shear rate"
    return {
        "top": top,
        "bottom": bottom,
        "span": span,
        "S": total_shear,
        "k": total_shear // span,
    }


def _restored_coords(coords: np.ndarray, bottom: int, shear_rate: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    restored = []
    for r, c in coords:
        rr = int(r)
        cc = int(c) + int(shear_rate) * (int(bottom) - int(r))
        if 0 <= rr < h and 0 <= cc < w:
            restored.append((rr, cc))
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each grey object's total left-shear, recolor it from the legend, and straighten it upright.

    Input structure:
    - The background is black, and the bottom row is a legend of separated horizontal color bars.
    - Each legend bar has width 1, 2, 3, or 4 and supplies the replacement color for that shear amount.
    - Above the legend are several grey 4-connected objects whose upper rows are shifted left relative to their bottom rows.

    Rule:
    - For each grey object, measure S = (leftmost column on its bottom row) - (leftmost column on its top row).
    - Read the legend bar of width S to get the object's target color.
    - Let span = bottom_row - top_row, so the per-row shear rate is S / span; move every pixel at (r, c) to
      (r, c + (bottom_row - r) * (S / span)).
    - Repaint the restored object with the legend color, and when restored objects overlap, keep the one with larger S.
    - Preserve the legend row unchanged.

    Key insight:
    - The measured shear is a code: the same value chooses the color, determines how far to undo the slant, and ranks overlap priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    legend = _parse_legend(g[-1])
    components = _find_components_of_color(g[:-1], OBJ_COLOR)

    out = np.full_like(g, BG)
    layers = []
    for coords in components:
        info = _measure_object(coords)
        color = legend[info["S"]]
        restored = _restored_coords(coords, info["bottom"], info["k"], g.shape)
        layers.append((info["S"], color, restored, _bbox(coords)))

    for _s, color, restored, box in sorted(layers, key=lambda item: (item[0], item[3][0], item[3][1])):
        for r, c in restored:
            out[r, c] = int(color)

    out[-1, :] = g[-1, :]
    return out


def _sample_span(total_shear: int) -> int:
    divisors = [d for d in range(1, total_shear + 1) if total_shear % d == 0]
    smaller = [d for d in divisors if d < total_shear]
    if smaller and random.random() < 0.6:
        return random.choice(smaller)
    return random.choice(divisors)


def _sample_lengths(height: int, shear_rate: int) -> list[int]:
    min_row_len = max(2, shear_rate + 1)
    min_area = max(8, min_row_len * height)
    max_area = min(15, 6 * height)
    target = random.randint(min_area, max_area)
    lengths = [min_row_len] * height
    remaining = target - sum(lengths)
    while remaining > 0:
        idx = random.randrange(height)
        if lengths[idx] < 6:
            lengths[idx] += 1
            remaining -= 1
    if len(set(lengths)) == 1:
        grow = random.randrange(height)
        shrink_choices = [i for i in range(height) if i != grow and lengths[i] > 2]
        if lengths[grow] < 6 and shrink_choices:
            lengths[grow] += 1
            lengths[random.choice(shrink_choices)] -= 1
    return lengths


def _mask_from_lengths(lengths: list[int]) -> np.ndarray:
    height = len(lengths)
    width = max(lengths)
    mask = np.zeros((height, width), dtype=bool)
    for r, length in enumerate(lengths):
        mask[r, :length] = True
    return mask


def _make_legend_row(width: int, legend_colors: list[int]) -> tuple[np.ndarray, dict[int, int]]:
    row = np.zeros(width, dtype=int)
    total = sum(LEGEND_WIDTHS) + (len(LEGEND_WIDTHS) - 1)
    start = random.randint(1, width - total - 1)
    c = start
    mapping = {}
    for run_width, color in zip(LEGEND_WIDTHS, legend_colors):
        row[c : c + run_width] = int(color)
        mapping[run_width] = int(color)
        c += run_width + 1
    return row, mapping


def _render_input(specs: list[dict], h: int, w: int, legend_row: np.ndarray) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    grid[-1, :] = legend_row
    for spec in specs:
        mask = spec["mask"]
        r0 = spec["r0"]
        c0 = spec["c0"]
        shear_rate = spec["k"]
        obj_h = spec["h"]
        for rr in range(obj_h):
            cols = np.where(mask[rr])[0]
            for cc in cols:
                gr = r0 + rr
                gc = c0 + int(cc) - shear_rate * (obj_h - 1 - rr)
                if not (0 <= gr < h - 1 and 0 <= gc < w):
                    return None
                if int(grid[gr, gc]) != BG:
                    return None
                grid[gr, gc] = OBJ_COLOR
    return grid


def _render_expected_output(specs: list[dict], h: int, w: int, legend_row: np.ndarray, legend_map: dict[int, int]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for spec in sorted(specs, key=lambda item: item["S"]):
        color = legend_map[spec["S"]]
        mask = spec["mask"]
        for rr in range(spec["h"]):
            cols = np.where(mask[rr])[0]
            for cc in cols:
                out[spec["r0"] + rr, spec["c0"] + int(cc)] = int(color)
    out[-1, :] = legend_row
    return out


def _collision_stats(specs: list[dict]) -> tuple[int, int]:
    occupancy: dict[tuple[int, int], list[int]] = defaultdict(list)
    for spec in specs:
        mask = spec["mask"]
        for rr in range(spec["h"]):
            cols = np.where(mask[rr])[0]
            for cc in cols:
                occupancy[(spec["r0"] + rr, spec["c0"] + int(cc))].append(spec["S"])
    collision_cells = sum(1 for values in occupancy.values() if len(values) > 1)
    max_depth = max((len(values) for values in occupancy.values()), default=1)
    return collision_cells, max_depth


def _validate_generated_pair(inp: np.ndarray, expected: np.ndarray, specs: list[dict], legend_map: dict[int, int]) -> bool:
    if inp is None:
        return False
    if int(np.count_nonzero(inp[:-1] == OBJ_COLOR)) < 24:
        return False
    if set(np.unique(inp).tolist()) - {BG, OBJ_COLOR} != set(legend_map.values()):
        return False
    if _parse_legend(inp[-1]) != legend_map:
        return False

    components = _find_components_of_color(inp[:-1], OBJ_COLOR)
    if len(components) != len(specs):
        return False
    try:
        measured = sorted(_measure_object(comp)["S"] for comp in components)
    except AssertionError:
        return False
    target = sorted(spec["S"] for spec in specs)
    if measured != target:
        return False

    collisions, max_depth = _collision_stats(specs)
    if not (1 <= collisions <= 8):
        return False
    if max_depth > 2:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(18, 24)

    legend_colors = random.sample(PALETTE, 4)
    legend_row, legend_map = _make_legend_row(w, legend_colors)

    object_count = random.randint(3, 4)
    shear_values = random.sample(list(LEGEND_WIDTHS), object_count)
    random.shuffle(shear_values)

    specs = []
    for total_shear in shear_values:
        span = _sample_span(total_shear)
        height = span + 1
        shear_rate = total_shear // span
        lengths = _sample_lengths(height, shear_rate)
        mask = _mask_from_lengths(lengths)
        specs.append(
            {
                "S": total_shear,
                "span": span,
                "k": shear_rate,
                "h": height,
                "w": int(mask.shape[1]),
                "mask": mask,
            }
        )

    max_w = max(spec["w"] for spec in specs)
    max_h = max(spec["h"] for spec in specs)
    cluster_r = random.randint(1, h - max_h - 3)
    cluster_c = random.randint(max(LEGEND_WIDTHS) + 1, w - max_w - 2)

    for _ in range(120):
        for spec in specs:
            r_lo = 1
            r_hi = h - spec["h"] - 2
            c_lo = spec["S"]
            c_hi = w - spec["w"] - 1
            spec["r0"] = min(max(cluster_r + random.randint(-2, 2), r_lo), r_hi)
            spec["c0"] = min(max(cluster_c + random.randint(-2, 2), c_lo), c_hi)

        inp = _render_input(specs, h, w, legend_row)
        if inp is None:
            continue
        expected = _render_expected_output(specs, h, w, legend_row, legend_map)
        if _validate_generated_pair(inp, expected, specs, legend_map):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
