import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6

GEN_ATTEMPTS = 600


GENERAL_TEMPLATES = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)],
]

HIT_TEMPLATES = [
    [(0, -1), (0, 0), (0, 1)],
    [(-1, 0), (0, 0), (1, 0)],
    [(0, -1), (0, 0), (1, 0)],
    [(0, 0), (1, -1), (1, 0)],
    [(0, -1), (0, 0), (0, 1), (1, 0)],
    [(-1, 0), (0, -1), (0, 0), (0, 1)],
    [(-1, 0), (0, 0), (1, 0), (0, 1)],
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    grey_comps = _components_of_color(grid, GREY)
    candidates = []
    for comp in grey_comps:
        if len(comp) != 3:
            continue
        rows = sorted({r for r, _ in comp})
        cols = sorted(c for _, c in comp)
        if len(rows) != 1 or cols != list(range(cols[0], cols[0] + 3)):
            continue
        center = (rows[0], cols[1])
        controls = []
        for nr in (center[0] - 1, center[0] + 1):
            if _in_bounds(grid.shape, nr, center[1]) and int(grid[nr, center[1]]) == PINK:
                controls.append((nr, center[1]))
        if len(controls) != 1:
            continue
        control = controls[0]
        step = (center[0] - control[0], 0)
        if abs(step[0]) != 1:
            continue
        cells = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[0], cols[2]), control}
        candidates.append((center, step, cells))
    assert len(candidates) == 1, "expected a unique horizontal three-cell marker bar with one control pixel"
    return candidates[0]


def _ray_cells(shape: tuple[int, int], center: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = step
    r, c = center
    out = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        out.append((r, c))
        r += dr
        c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair ray-erased red objects by copying their horizontal mirror locations in a new color.

    Input structure:
    - The grid has black background, a large blue pattern, and several small 4-connected red objects arranged with horizontal reflection symmetry.
    - A unique grey three-cell horizontal marker bar has one adjacent magenta control pixel above or below its center.
    - The marker defines a vertical center ray; red objects may intersect that ray while blue pixels do not.

    Rule:
    - Cast a vertical ray from the bar center in the direction opposite the control pixel.
    - Every red 4-connected object touched by that ray is erased completely to black.
    - For each erased cell, look at its horizontal mirror position across the grid midpoint; if that mirror cell is still red, recolor the erased cell green.
    - Preserve the blue pattern, untouched red objects, and the marker itself.

    Key insight:
    - The ray only tells you which whole red components disappear; the missing pixels are recovered from the surviving mirror-side components, and the recovered region is marked in green rather than restored to red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center, step, _ = _find_marker(g)
    ray = set(_ray_cells(g.shape, center, step))
    red_components = _components_of_color(g, RED)

    hit_components = [comp for comp in red_components if any(cell in ray for cell in comp)]
    assert hit_components, "expected at least one red object on the marker ray"

    out = g.copy()
    erased = []
    for comp in hit_components:
        for r, c in comp:
            out[r, c] = BLACK
            erased.append((r, c))

    h, _ = out.shape
    for r, c in erased:
        mr = h - 1 - r
        if int(out[mr, c]) == RED:
            out[r, c] = GREEN

    return out


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _mirror_coords(coords: list[tuple[int, int]], h: int) -> list[tuple[int, int]]:
    return [(h - 1 - r, c) for r, c in coords]


def _normalize_template(template: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in template)
    min_c = min(c for _, c in template)
    return [(r - min_r, c - min_c) for r, c in template]


def _padding_zone(coords: list[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    zone: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    zone.add((nr, nc))
    return zone


def _sample_top_bands(h: int, ray_col: int, w: int):
    mid = h // 2
    top_limit = mid - 3
    target_count = random.choice([2, 3])
    starts = list(range(1, top_limit - 1))
    random.shuffle(starts)
    bands = []
    for start in starts:
        band_h = random.randint(2, min(4, top_limit - start + 1))
        end = start + band_h - 1
        if any(not (end + 1 < b["r0"] or start - 1 > b["r1"]) for b in bands):
            continue
        left_start = random.randint(1, min(3, ray_col - 3))
        right_end = random.randint(max(ray_col + 3, w - 7), w - 2)
        bands.append(
            {
                "r0": start,
                "r1": end,
                "left_start": left_start,
                "right_end": right_end,
            }
        )
        if len(bands) == target_count:
            break
    if len(bands) < 2:
        return None
    bands.sort(key=lambda b: b["r0"])
    return bands


def _blue_support_from_bands(h: int, w: int, ray_col: int, bands) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for band in bands:
        rows = slice(band["r0"], band["r1"] + 1)
        mask[rows, band["left_start"] : ray_col] = True
        mask[rows, ray_col + 1 : band["right_end"] + 1] = True
        mr0 = h - 1 - band["r1"]
        mr1 = h - 1 - band["r0"]
        rows_m = slice(mr0, mr1 + 1)
        mask[rows_m, band["left_start"] : ray_col] = True
        mask[rows_m, ray_col + 1 : band["right_end"] + 1] = True
    return mask


def _place_hit_pair(
    h: int,
    w: int,
    ray_col: int,
    bands,
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    for _ in range(200):
        template = random.choice(HIT_TEMPLATES)
        min_dr = min(r for r, _ in template)
        max_dr = max(r for r, _ in template)
        band = random.choice(bands)
        low = band["r0"] - min_dr
        high = band["r1"] - max_dr
        if low > high:
            continue
        base_r = random.randint(low, high)
        coords = [(base_r + dr, ray_col + dc) for dr, dc in template]
        if any(not _in_bounds((h, w), r, c) for r, c in coords):
            continue
        if len({(r, c) for r, c in coords}) != len(coords):
            continue
        mirror = _mirror_coords(coords, h)
        both = coords + mirror
        if any(cell in reserved for cell in _padding_zone(both, (h, w), radius=1)):
            continue
        return coords
    return None


def _place_nonhit_pair(
    h: int,
    w: int,
    ray_col: int,
    bands,
    blue_support: np.ndarray,
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    for _ in range(250):
        template = _normalize_template(random.choice(GENERAL_TEMPLATES))
        max_r = max(r for r, _ in template)
        max_c = max(c for _, c in template)
        band = random.choice(bands)
        side = random.choice(["left", "right"])
        row_low = band["r0"]
        row_high = band["r1"] - max_r
        if row_low > row_high:
            continue
        if side == "left":
            col_low = band["left_start"]
            col_high = ray_col - 1 - max_c
        else:
            col_low = ray_col + 1
            col_high = band["right_end"] - max_c
        if col_low > col_high:
            continue
        base_r = random.randint(row_low, row_high)
        base_c = random.randint(col_low, col_high)
        coords = [(base_r + dr, base_c + dc) for dr, dc in template]
        if any(c == ray_col for _, c in coords):
            continue
        if any(not blue_support[r, c] for r, c in coords):
            continue
        mirror = _mirror_coords(coords, h)
        both = coords + mirror
        if any(cell in reserved for cell in _padding_zone(both, (h, w), radius=1)):
            continue
        return coords
    return None


def _build_pair_records():
    h = random.choice([19, 21, 23])
    w = random.choice([17, 19, 21, 23, 25])
    mid = h // 2
    ray_dir = random.choice([-1, 1])
    bar_row = mid - 1 if ray_dir == 1 else mid + 1
    control_row = bar_row - ray_dir
    ray_col = random.randint(5, w - 6)

    bands = _sample_top_bands(h, ray_col, w)
    if bands is None:
        return None
    blue_support = _blue_support_from_bands(h, w, ray_col, bands)

    reserved = set()
    marker_cells = [(bar_row, ray_col - 1), (bar_row, ray_col), (bar_row, ray_col + 1), (control_row, ray_col)]
    reserved |= _padding_zone(marker_cells, (h, w), radius=1)

    total_pairs = random.randint(3, 4)
    hit_pairs = random.choice([1, 2])
    records = []

    for _ in range(hit_pairs):
        top_coords = _place_hit_pair(h, w, ray_col, bands, reserved)
        if top_coords is None:
            return None
        bottom_coords = _mirror_coords(top_coords, h)
        reserved |= _padding_zone(top_coords + bottom_coords, (h, w), radius=1)
        records.append({"top": top_coords, "bottom": bottom_coords, "hit": True})

    for _ in range(total_pairs - hit_pairs):
        top_coords = _place_nonhit_pair(h, w, ray_col, bands, blue_support, reserved)
        if top_coords is None:
            return None
        bottom_coords = _mirror_coords(top_coords, h)
        reserved |= _padding_zone(top_coords + bottom_coords, (h, w), radius=1)
        records.append({"top": top_coords, "bottom": bottom_coords, "hit": False})

    visible_blue = int(blue_support.sum()) - sum(len(rec["top"]) + len(rec["bottom"]) for rec in records)
    blue_ratio = visible_blue / float(h * w)
    if not (0.30 <= blue_ratio <= 0.50):
        return None

    return {
        "h": h,
        "w": w,
        "ray_dir": ray_dir,
        "bar_row": bar_row,
        "control_row": control_row,
        "ray_col": ray_col,
        "bands": bands,
        "blue_support": blue_support,
        "records": records,
    }


def _render_grids(spec) -> tuple[np.ndarray, np.ndarray]:
    h = spec["h"]
    w = spec["w"]
    out = np.zeros((h, w), dtype=int)
    inp = np.zeros((h, w), dtype=int)

    out[spec["blue_support"]] = BLUE
    inp[spec["blue_support"]] = BLUE

    hit_on_top = spec["ray_dir"] == -1
    for rec in spec["records"]:
        if rec["hit"] and hit_on_top:
            _paint_coords(out, rec["top"], GREEN)
            _paint_coords(out, rec["bottom"], RED)
        elif rec["hit"]:
            _paint_coords(out, rec["top"], RED)
            _paint_coords(out, rec["bottom"], GREEN)
        else:
            _paint_coords(out, rec["top"], RED)
            _paint_coords(out, rec["bottom"], RED)
        _paint_coords(inp, rec["top"], RED)
        _paint_coords(inp, rec["bottom"], RED)

    rc = spec["ray_col"]
    br = spec["bar_row"]
    cr = spec["control_row"]
    inp[br, rc - 1 : rc + 2] = GREY
    out[br, rc - 1 : rc + 2] = GREY
    inp[cr, rc] = PINK
    out[cr, rc] = PINK
    return inp, out


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    h, w = inp.shape
    if max(inp.shape) > 30:
        return False
    if set(map(int, np.unique(inp))) - {BLACK, BLUE, RED, GREY, PINK}:
        return False
    if set(map(int, np.unique(out))) - {BLACK, BLUE, RED, GREEN, GREY, PINK}:
        return False

    center, step, marker_cells = _find_marker(inp)
    ray = set(_ray_cells(inp.shape, center, step))
    if any(int(inp[r, c]) == BLUE for r, c in ray):
        return False

    stripped = inp.copy()
    for r, c in marker_cells:
        stripped[r, c] = BLACK
    if not np.array_equal(stripped, np.flipud(stripped)):
        return False

    red_components = _components_of_color(inp, RED)
    if not (6 <= len(red_components) <= 8):
        return False
    hits = [comp for comp in red_components if any(cell in ray for cell in comp)]
    if len(hits) not in (1, 2):
        return False
    for comp in hits:
        mirrored = {(h - 1 - r, c) for r, c in comp}
        if any(cell in ray for cell in mirrored):
            return False

    green_coords = set(map(tuple, np.argwhere(out == GREEN)))
    if not green_coords:
        return False
    if Counter(map(int, out.flatten()))[BLUE] / float(h * w) < 0.28:
        return False

    for r, c in green_coords:
        mr = h - 1 - r
        if int(out[mr, c]) != RED:
            return False

    if not np.array_equal(main(inp), out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        spec = _build_pair_records()
        if spec is None:
            continue
        inp, out = _render_grids(spec)
        if _semantic_check(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
