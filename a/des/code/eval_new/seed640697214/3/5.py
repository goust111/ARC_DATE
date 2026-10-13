from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
CANVAS_FILL = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_RETRIES = 500


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = DIR8 if conn8 else DIR4
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _dominant_color(values: list[int]) -> int:
    counts = Counter(values)
    return min(((-count, color) for color, count in counts.items()))[1]


def _find_canvas(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict[str, int]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for coords in _components(g == color, conn8=False):
            r0, c0, r1, c1 = _bbox(coords)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area != len(coords):
                continue
            if r0 == 0 or c0 == 0:
                continue

            top_map: dict[int, int] = {}
            left_map: dict[int, int] = {}
            valid = True

            for c in range(c0, c1 + 1):
                val = int(g[r0 - 1, c])
                if val in (BG, color):
                    continue
                if val in top_map:
                    valid = False
                    break
                top_map[val] = c - c0
            if not valid or not top_map:
                continue

            for r in range(r0, r1 + 1):
                val = int(g[r, c0 - 1])
                if val in (BG, color):
                    continue
                if val in left_map:
                    valid = False
                    break
                left_map[val] = r - r0
            if not valid or not left_map:
                continue

            if set(top_map) != set(left_map):
                continue

            candidates.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "area": area,
                    "pair_count": len(top_map),
                }
            )

    assert candidates, "expected a solid canvas rectangle with paired edge markers"
    return max(candidates, key=lambda item: (item["area"], item["pair_count"]))


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    canvas = _find_canvas(g)
    r0 = int(canvas["r0"])
    c0 = int(canvas["c0"])
    r1 = int(canvas["r1"])
    c1 = int(canvas["c1"])
    canvas_color = int(canvas["color"])

    row_map: dict[int, int] = {}
    col_map: dict[int, int] = {}

    for c in range(c0, c1 + 1):
        val = int(g[r0 - 1, c])
        if val not in (BG, canvas_color):
            col_map[val] = c - c0

    for r in range(r0, r1 + 1):
        val = int(g[r, c0 - 1])
        if val not in (BG, canvas_color):
            row_map[val] = r - r0

    assert row_map and set(row_map) == set(col_map), "edge markers must form paired row/column keys"

    stamp_mask = g != BG
    stamp_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    stamp_mask[r0 - 1, c0 : c1 + 1] = False
    stamp_mask[r0 : r1 + 1, c0 - 1] = False

    stamps: list[dict[str, object]] = []
    seen_colors: set[int] = set()
    for coords in _components(stamp_mask, conn8=True):
        values = [int(g[r, c]) for r, c in coords]
        color = _dominant_color(values)
        if color not in row_map or color not in col_map:
            continue
        assert color not in seen_colors, "stamp colors must be unique"
        seen_colors.add(color)
        sr0, sc0, sr1, sc1 = _bbox(coords)
        stamps.append(
            {
                "color": color,
                "area": len(coords),
                "bbox": (sr0, sc0, sr1, sc1),
                "pattern": g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy(),
            }
        )

    assert len(stamps) == len(row_map), "expected one stamp per marker pair"
    canvas_h = r1 - r0 + 1
    canvas_w = c1 - c0 + 1
    return canvas_h, canvas_w, canvas_color, row_map, col_map, stamps


def _compose_output(
    canvas_h: int,
    canvas_w: int,
    canvas_color: int,
    row_map: dict[int, int],
    col_map: dict[int, int],
    stamps: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((canvas_h, canvas_w), int(canvas_color), dtype=int)
    ordered = sorted(
        stamps,
        key=lambda stamp: (
            int(stamp["area"]),
            tuple(stamp.get("bbox", (-1, -1, -1, -1))),
        ),
    )

    for stamp in ordered:
        top = int(row_map[int(stamp["color"])])
        left = int(col_map[int(stamp["color"])])
        pattern = np.asarray(stamp["pattern"], dtype=int)
        sh, sw = pattern.shape
        for dr in range(sh):
            for dc in range(sw):
                val = int(pattern[dr, dc])
                if val == BG:
                    continue
                rr = top + dr
                cc = left + dc
                if 0 <= rr < canvas_h and 0 <= cc < canvas_w:
                    out[rr, cc] = val
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the grey canvas by stamping each colored pattern at the row/column intersection indicated by paired edge markers.

    Input structure:
    - The background is black.
    - One large filled rectangle acts as the canvas.
    - Single-pixel colored markers sit directly above the canvas and directly left of the canvas; each marker color appears once on each side.
    - Outside the canvas are several small non-black stamp patterns whose dominant colors match the paired marker colors.

    Rule:
    - Read each top marker as a canvas column and each left marker as a canvas row.
    - For every stamp, use its dominant color to find the matching row marker and column marker, then align the stamp's top-left corner to that intersection inside the canvas.
    - Apply stamps in ascending order of non-black pixel count, treating black stamp cells as transparent; later larger stamps overwrite earlier smaller ones where they overlap.
    - Return only the canvas crop, keeping untouched cells in the original canvas fill color.

    Key insight:
    - The puzzle combines two independent mechanisms: paired edge markers define stamp coordinates, while stamp area determines the occlusion order on the canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    canvas_h, canvas_w, canvas_color, row_map, col_map, stamps = _extract_scene(g)
    return _compose_output(canvas_h, canvas_w, canvas_color, row_map, col_map, stamps)


def _random_shape_mask(rng: random.Random, area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(rng.randrange(5), rng.randrange(5))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        coords = list(cells)
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True
        if area == h * w:
            continue
        return mask
    return None


def _make_stamp_pattern(
    rng: random.Random,
    base_color: int,
    accent_pool: tuple[int, ...],
    area: int,
) -> np.ndarray | None:
    mask = _random_shape_mask(rng, area)
    if mask is None:
        return None

    pattern = np.full(mask.shape, BG, dtype=int)
    pattern[mask] = int(base_color)

    if accent_pool and rng.random() < 0.7:
        max_accents = min(3, area // 4)
        if max_accents > 0:
            accent_count = rng.randint(1, max_accents)
            accent_color = int(rng.choice(accent_pool))
            cells = list(zip(*np.where(mask)))
            rng.shuffle(cells)
            for rr, cc in cells[:accent_count]:
                pattern[rr, cc] = accent_color

    return pattern


def _assign_unique_coords(rng: random.Random, limits: list[int]) -> list[int] | None:
    order = sorted(range(len(limits)), key=lambda idx: limits[idx])
    used: set[int] = set()
    out = [-1] * len(limits)

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        choices = [v for v in range(limits[idx] + 1) if v not in used]
        rng.shuffle(choices)
        for value in choices:
            used.add(value)
            out[idx] = value
            if search(pos + 1):
                return True
            used.remove(value)
            out[idx] = -1
        return False

    if not search(0):
        return None
    return out


def _has_output_overlap(stamps: list[dict[str, object]], rows: list[int], cols: list[int]) -> bool:
    for i, stamp_i in enumerate(stamps):
        hi, wi = np.asarray(stamp_i["pattern"]).shape
        for j in range(i + 1, len(stamps)):
            hj, wj = np.asarray(stamps[j]["pattern"]).shape
            row_overlap = max(rows[i], rows[j]) < min(rows[i] + hi, rows[j] + hj)
            col_overlap = max(cols[i], cols[j]) < min(cols[i] + wi, cols[j] + wj)
            if row_overlap and col_overlap:
                return True
    return False


def _choose_output_positions(
    rng: random.Random,
    stamps: list[dict[str, object]],
    canvas_h: int,
    canvas_w: int,
) -> tuple[list[int], list[int]] | None:
    row_limits = [canvas_h - np.asarray(stamp["pattern"]).shape[0] for stamp in stamps]
    col_limits = [canvas_w - np.asarray(stamp["pattern"]).shape[1] for stamp in stamps]
    if min(row_limits) < 0 or min(col_limits) < 0:
        return None

    for _ in range(400):
        rows = _assign_unique_coords(rng, row_limits)
        cols = _assign_unique_coords(rng, col_limits)
        if rows is None or cols is None:
            continue
        if _has_output_overlap(stamps, rows, cols):
            return rows, cols
    return None


def _mark_blocked(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _place_input_stamps(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    stamps: list[dict[str, object]],
    canvas_r0: int,
    canvas_c0: int,
    canvas_h: int,
    canvas_w: int,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_blocked(blocked, canvas_r0, canvas_c0, canvas_h, canvas_w, pad=1)
    order = list(range(len(stamps)))
    rng.shuffle(order)
    positions: list[tuple[int, int] | None] = [None] * len(stamps)

    for idx in order:
        pattern = np.asarray(stamps[idx]["pattern"], dtype=int)
        sh, sw = pattern.shape
        candidates: list[tuple[int, int]] = []
        for r in range(grid_h - sh + 1):
            for c in range(grid_w - sw + 1):
                if blocked[r : r + sh, c : c + sw].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        pos = rng.choice(candidates)
        positions[idx] = pos
        _mark_blocked(blocked, pos[0], pos[1], sh, sw, pad=1)

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _build_grid(
    grid_h: int,
    grid_w: int,
    canvas_r0: int,
    canvas_c0: int,
    canvas_h: int,
    canvas_w: int,
    stamps: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[canvas_r0 : canvas_r0 + canvas_h, canvas_c0 : canvas_c0 + canvas_w] = CANVAS_FILL

    for stamp in stamps:
        color = int(stamp["color"])
        out_r = int(stamp["out_r"])
        out_c = int(stamp["out_c"])
        grid[canvas_r0 - 1, canvas_c0 + out_c] = color
        grid[canvas_r0 + out_r, canvas_c0 - 1] = color

    for stamp in stamps:
        pattern = np.asarray(stamp["pattern"], dtype=int)
        in_r = int(stamp["in_r"])
        in_c = int(stamp["in_c"])
        sh, sw = pattern.shape
        window = grid[in_r : in_r + sh, in_c : in_c + sw]
        mask = pattern != BG
        window[mask] = pattern[mask]

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stamp_count = rng.randint(2, 4)
    base_colors = rng.sample(STAMP_COLORS, stamp_count)
    accent_pool = tuple(color for color in STAMP_COLORS if color not in base_colors)
    areas = rng.sample(list(range(6, 16)), stamp_count)

    stamps: list[dict[str, object]] = []
    for color, area in zip(base_colors, areas):
        pattern = _make_stamp_pattern(rng, color, accent_pool, area)
        if pattern is None:
            return None
        stamps.append({"color": color, "area": area, "pattern": pattern})

    canvas_h = rng.randint(10, 14)
    canvas_w = rng.randint(10, 14)
    out_positions = _choose_output_positions(rng, stamps, canvas_h, canvas_w)
    if out_positions is None:
        return None
    rows, cols = out_positions
    for stamp, out_r, out_c in zip(stamps, rows, cols):
        stamp["out_r"] = int(out_r)
        stamp["out_c"] = int(out_c)

    grid_h = rng.randint(max(20, canvas_h + 9), 30)
    grid_w = rng.randint(max(20, canvas_w + 9), 30)
    extra_h = grid_h - canvas_h
    extra_w = grid_w - canvas_w
    if extra_h < 8 or extra_w < 8:
        return None
    canvas_r0 = rng.randint(5, extra_h - 3)
    canvas_c0 = rng.randint(5, extra_w - 3)

    in_positions = _place_input_stamps(rng, grid_h, grid_w, stamps, canvas_r0, canvas_c0, canvas_h, canvas_w)
    if in_positions is None or len(in_positions) != len(stamps):
        return None
    for stamp, (in_r, in_c) in zip(stamps, in_positions):
        stamp["in_r"] = int(in_r)
        stamp["in_c"] = int(in_c)

    grid = _build_grid(grid_h, grid_w, canvas_r0, canvas_c0, canvas_h, canvas_w, stamps)
    expected = _compose_output(
        canvas_h=canvas_h,
        canvas_w=canvas_w,
        canvas_color=CANVAS_FILL,
        row_map={int(stamp["color"]): int(stamp["out_r"]) for stamp in stamps},
        col_map={int(stamp["color"]): int(stamp["out_c"]) for stamp in stamps},
        stamps=stamps,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
