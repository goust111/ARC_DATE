from collections import Counter
import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("ul", "ur", "ll", "lr")
GEN_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_rectangular_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        if int(len(pts)) != int(border_area):
            continue

        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border and int(grid[r, c]) != color:
                    ok = False
                    break
                if (not on_border) and int(grid[r, c]) == color:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((color, (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected a unique rectangular frame"
    return candidates[0]


def _seed_quadrant(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = frame_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    top = r < center_r
    left = c < center_c
    if top and left:
        return "ul"
    if top and (not left):
        return "ur"
    if (not top) and left:
        return "ll"
    return "lr"


def _quadrant_dirs(quadrant: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if quadrant == "ul":
        return ((-1, 0), (0, -1))
    if quadrant == "ur":
        return ((-1, 0), (0, 1))
    if quadrant == "ll":
        return ((1, 0), (0, -1))
    return ((1, 0), (0, 1))


def _draw_branches(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
    bg: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    color_counts = Counter(
        int(v) for v in inner.flatten() if int(v) not in (bg, frame_color)
    )

    protected = np.zeros_like(grid, dtype=bool)
    protected[grid == frame_color] = True
    seed_cells: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color in (bg, frame_color):
                continue
            protected[r, c] = True
            seed_cells.append((int(r), int(c), color))

    height, width = grid.shape
    for r, c, color in seed_cells:
        length = int(color_counts[color])
        quadrant = _seed_quadrant(r, c, frame_bbox)
        for dr, dc in _quadrant_dirs(quadrant):
            rr, cc = int(r), int(c)
            painted = 0
            while painted < length:
                rr += int(dr)
                cc += int(dc)
                if not (0 <= rr < height and 0 <= cc < width):
                    break
                if protected[rr, cc]:
                    continue
                out[rr, cc] = color
                painted += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-coded branches outward from seeds inside a gray frame.

    Input structure:
    - The background is black.
    - A one-pixel-thick gray rectangular frame encloses a hollow interior.
    - Colored seed pixels are scattered inside the frame, and the rest starts as background.

    Rule:
    - Count how many times each seed color appears inside the frame.
    - Determine each seed's interior quadrant relative to the frame center:
      upper-left grows up and left, upper-right grows up and right,
      lower-left grows down and left, lower-right grows down and right.
    - For each seed, keep walking in each of its two directions until exactly N
      non-frame, non-seed pixels have been painted in that color, where N is the
      total count of that color inside the frame; the walk passes across the
      gray frame without changing the frame pixels, and stops at the grid edge.

    Key insight:
    - Direction comes from the seed's quadrant, while branch length comes from the
      global multiplicity of that seed color inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    frame_color, frame_bbox = _find_rectangular_frame(g, bg)
    return _draw_branches(g, frame_bbox, frame_color, bg)


def _sample_counts(total: int, parts: int) -> list[int] | None:
    if parts <= 0 or total < parts or total > parts * 5:
        return None
    counts = [1] * parts
    remaining = total - parts
    while remaining > 0:
        idx = random.randrange(parts)
        if counts[idx] >= 5:
            continue
        counts[idx] += 1
        remaining -= 1
    if max(counts) == 1:
        return None
    random.shuffle(counts)
    return counts


def _frame_border_cells(frame_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _validate_layout(
    height: int,
    width: int,
    frame_bbox: tuple[int, int, int, int],
    seeds: list[dict[str, int | str]],
    color_counts: dict[int, int],
) -> bool:
    r0, c0, r1, c1 = frame_bbox
    frame_cells = _frame_border_cells(frame_bbox)
    seed_positions = {(int(seed["r"]), int(seed["c"])) for seed in seeds}
    painted_cells: set[tuple[int, int]] = set()

    for seed in seeds:
        sr = int(seed["r"])
        sc = int(seed["c"])
        color = int(seed["color"])
        length = int(color_counts[color])
        for dr, dc in _quadrant_dirs(str(seed["quad"])):
            rr, cc = sr, sc
            painted = 0
            touched_outside = False
            while painted < length:
                rr += int(dr)
                cc += int(dc)
                if not (0 <= rr < height and 0 <= cc < width):
                    return False
                if (rr, cc) in seed_positions and (rr, cc) != (sr, sc):
                    return False
                if (rr, cc) in frame_cells:
                    continue
                if (rr, cc) in painted_cells:
                    return False
                painted_cells.add((rr, cc))
                if not (r0 < rr < r1 and c0 < cc < c1):
                    touched_outside = True
                painted += 1
            if not touched_outside:
                return False
    return True


def _trace_seed_cells(
    height: int,
    width: int,
    frame_bbox: tuple[int, int, int, int],
    frame_cells: set[tuple[int, int]],
    seed_pos: tuple[int, int],
    quadrant: str,
    length: int,
    other_seed_positions: set[tuple[int, int]],
    blocked_paint: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = frame_bbox
    sr, sc = seed_pos
    traced: list[tuple[int, int]] = []
    traced_set: set[tuple[int, int]] = set()

    for dr, dc in _quadrant_dirs(quadrant):
        rr, cc = sr, sc
        painted = 0
        touched_outside = False
        while painted < length:
            rr += int(dr)
            cc += int(dc)
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            if (rr, cc) in other_seed_positions:
                return None
            if (rr, cc) in frame_cells:
                continue
            if (rr, cc) in blocked_paint or (rr, cc) in traced_set:
                return None
            traced.append((rr, cc))
            traced_set.add((rr, cc))
            if not (r0 < rr < r1 and c0 < cc < c1):
                touched_outside = True
            painted += 1
        if not touched_outside:
            return None
    return traced


def _pick_even_size(low: int, high: int) -> int:
    options = [value for value in range(low, high + 1) if value % 2 == 0]
    return int(random.choice(options))


def _make_quadrant_buckets(frame_bbox: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    mid_r = r0 + 1 + inner_h // 2
    mid_c = c0 + 1 + inner_w // 2
    top_rows = list(range(r0 + 1, mid_r))
    bottom_rows = list(range(mid_r, r1))
    left_cols = list(range(c0 + 1, mid_c))
    right_cols = list(range(mid_c, c1))
    return {
        "ul": [(r, c) for r in top_rows for c in left_cols],
        "ur": [(r, c) for r in top_rows for c in right_cols],
        "ll": [(r, c) for r in bottom_rows for c in left_cols],
        "lr": [(r, c) for r in bottom_rows for c in right_cols],
    }


def _gen_attempt() -> np.ndarray | None:
    total_seeds = random.randint(4, 8)
    num_colors = random.randint(2, min(4, total_seeds - 1))
    counts = _sample_counts(total_seeds, num_colors)
    if counts is None:
        return None
    colors = random.sample(SEED_COLORS, num_colors)
    color_counts = {int(color): int(count) for color, count in zip(colors, counts)}

    inner_h = _pick_even_size(6, 12)
    inner_w = _pick_even_size(6, 12)
    top_margin = random.randint(3, 7)
    bottom_margin = random.randint(max(3, top_margin - 2), min(8, top_margin + 2))
    left_margin = random.randint(3, 7)
    right_margin = random.randint(max(3, left_margin - 2), min(8, left_margin + 2))

    height = top_margin + inner_h + bottom_margin + 2
    width = left_margin + inner_w + right_margin + 2
    if max(height, width) > 30:
        return None

    frame_bbox = (
        int(top_margin),
        int(left_margin),
        int(top_margin + inner_h + 1),
        int(left_margin + inner_w + 1),
    )
    buckets = _make_quadrant_buckets(frame_bbox)
    frame_cells = _frame_border_cells(frame_bbox)

    quadrants = list(QUADRANTS)
    quadrants.extend(random.choice(QUADRANTS) for _ in range(total_seeds - 4))
    random.shuffle(quadrants)

    color_pool: list[int] = []
    for color, count in color_counts.items():
        color_pool.extend([int(color)] * int(count))
    random.shuffle(color_pool)

    specs = [
        {
            "quad": quad,
            "color": int(color),
            "length": int(color_counts[int(color)]),
        }
        for quad, color in zip(quadrants, color_pool)
    ]
    random.shuffle(specs)
    specs.sort(key=lambda spec: -int(spec["length"]))

    used_positions: set[tuple[int, int]] = set()
    painted_cells: set[tuple[int, int]] = set()
    seeds: list[dict[str, int | str]] = []
    for spec in specs:
        quad = str(spec["quad"])
        color = int(spec["color"])
        length = int(spec["length"])
        available = [
            cell
            for cell in buckets[quad]
            if cell not in used_positions and cell not in painted_cells
        ]
        random.shuffle(available)

        placed = False
        for r, c in available:
            traced = _trace_seed_cells(
                height,
                width,
                frame_bbox,
                frame_cells,
                (int(r), int(c)),
                quad,
                length,
                used_positions,
                painted_cells,
            )
            if traced is None:
                continue
            used_positions.add((int(r), int(c)))
            painted_cells.update(traced)
            seeds.append({"r": int(r), "c": int(c), "color": color, "quad": quad})
            placed = True
            break
        if not placed:
            return None

    if not _validate_layout(height, width, frame_bbox, seeds, color_counts):
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    for seed in seeds:
        grid[int(seed["r"]), int(seed["c"])] = int(seed["color"])

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
