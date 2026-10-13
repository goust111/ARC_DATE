import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _diag_markers(box: tuple[int, int, int, int], orientation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = box
    if orientation == "main":
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _in_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _pair_specs(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    specs: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        pts = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        assert len(pts) == 2, "each non-background color must appear exactly twice"
        pts.sort()
        (r_a, c_a), (r_b, c_b) = pts
        box = (min(r_a, r_b), min(c_a, c_b), max(r_a, r_b), max(c_a, c_b))
        specs.append(
            {
                "color": color,
                "p1": (r_a, c_a),
                "p2": (r_b, c_b),
                "box": box,
                "area": _box_area(box),
            }
        )

    assert specs, "expected at least one marker pair"
    return bg, specs


def _line_cells(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = sorted((p1, p2))

    cells: list[tuple[int, int]] = []
    step_c = 1 if c2 >= c1 else -1
    for c in range(c1, c2 + step_c, step_c):
        cells.append((r1, c))

    if r2 > r1:
        for r in range(r1, r2 + 1):
            cells.append((r, c2))

    return list(dict.fromkeys(cells))


def _apply_rules(grid: np.ndarray, descending: bool = True) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    bg, specs = _pair_specs(g)

    def sort_key(spec: dict) -> tuple[int, int, int, int, int]:
        r0, c0, r1, c1 = spec["box"]
        area = int(spec["area"])
        return ((-area) if descending else area, r0, c0, r1, c1)

    ordered = sorted(specs, key=sort_key)
    out = np.full_like(g, bg)
    filled = np.zeros_like(g, dtype=bool)
    rect_count = 0
    line_count = 0

    for spec in ordered:
        color = int(spec["color"])
        r0, c0, r1, c1 = spec["box"]
        if not filled[r0 : r1 + 1, c0 : c1 + 1].any():
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            filled[r0 : r1 + 1, c0 : c1 + 1] = True
            rect_count += 1
        else:
            for r, c in _line_cells(spec["p1"], spec["p2"]):
                out[r, c] = color
            line_count += 1

    for spec in specs:
        color = int(spec["color"])
        for r, c in (spec["p1"], spec["p2"]):
            out[r, c] = color

    return out, rect_count, line_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marker-defined rectangle unless it overlaps an earlier filled rectangle;
    conflicting pairs collapse to an L-shaped connector drawn horizontal first, then vertical.
    """
    return _apply_rules(grid, descending=True)[0]


# --------------- generator ---------------

GRID_RANGE_2 = (20, 24)
GRID_RANGE_3 = (22, 28)
MIN_SLOT_SIZE = 10
BASE_SIZE_RANGE = (6, 8)
LINE_TOP_PAD = (1, 2)
LINE_DEPTH = (1, 2)
LINE_SIDE_PAD = (1, 2)
GEN_RETRIES = 400


def _slots_for_grid(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    split_r = rng.randint(MIN_SLOT_SIZE, h - MIN_SLOT_SIZE)
    split_c = rng.randint(MIN_SLOT_SIZE, w - MIN_SLOT_SIZE)
    return [
        (0, 0, split_r - 1, split_c - 1),
        (0, split_c, split_r - 1, w - 1),
        (split_r, 0, h - 1, split_c - 1),
        (split_r, split_c, h - 1, w - 1),
    ]


def _sample_fill_box(
    slot: tuple[int, int, int, int], used_areas: set[int], rng: random.Random
) -> tuple[int, int, int, int] | None:
    sr0, sc0, sr1, sc1 = slot
    slot_h = sr1 - sr0 + 1
    slot_w = sc1 - sc0 + 1
    max_h = min(BASE_SIZE_RANGE[1], slot_h - 4)
    max_w = min(BASE_SIZE_RANGE[1], slot_w - 4)
    min_h = BASE_SIZE_RANGE[0]
    min_w = BASE_SIZE_RANGE[0]
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(80):
        h = rng.randint(min_h, max_h)
        w = rng.randint(min_w, max_w)
        area = h * w
        if area in used_areas:
            continue

        r0_lo = sr0 + 2
        r0_hi = sr1 - h - 1
        c0_lo = sc0 + 2
        c0_hi = sc1 - w - 1
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        r0 = rng.randint(r0_lo, r0_hi)
        c0 = rng.randint(c0_lo, c0_hi)
        return (r0, c0, r0 + h - 1, c0 + w - 1)

    return None


def _sample_line_pair(
    anchor_box: tuple[int, int, int, int],
    slot: tuple[int, int, int, int],
    min_fill_area: int,
    used_areas: set[int],
    rng: random.Random,
) -> dict | None:
    ar0, ac0, ar1, ac1 = anchor_box
    fill_area = _box_area(anchor_box)
    sr0, sc0, sr1, sc1 = slot

    for _ in range(120):
        top_pad = rng.randint(*LINE_TOP_PAD)
        depth = rng.randint(*LINE_DEPTH)
        left_pad = rng.randint(*LINE_SIDE_PAD)
        right_pad = rng.randint(*LINE_SIDE_PAD)

        br0 = ar0 - top_pad
        bc0 = ac0 - left_pad
        br1 = ar0 + depth
        bc1 = ac1 + right_pad
        if br0 < sr0 or bc0 < sc0 or br1 > sr1 or bc1 > sc1:
            continue

        box = (br0, bc0, br1, bc1)
        area = _box_area(box)
        if area in used_areas or area >= fill_area or area >= min_fill_area:
            continue

        orientation = rng.choice(["main", "anti"])
        p1, p2 = _diag_markers(box, orientation)
        path = _line_cells(p1, p2)
        if any(_in_box(cell, anchor_box) for cell in path):
            continue

        return {"box": box, "p1": p1, "p2": p2, "area": area}

    return None


def _build_grid(
    h: int,
    w: int,
    bg: int,
    colors: list[int],
    pairs: list[dict],
) -> np.ndarray:
    g = np.full((h, w), bg, dtype=int)
    for color, pair in zip(colors, pairs):
        r1, c1 = pair["p1"]
        r2, c2 = pair["p2"]
        g[r1, c1] = color
        g[r2, c2] = color
    return g


def _valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    counts = Counter(map(int, g.reshape(-1)))
    return all(color == bg or count == 2 for color, count in counts.items())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    fill_count = rng.choice([2, 3])
    side_range = GRID_RANGE_2 if fill_count == 2 else GRID_RANGE_3
    h = rng.randint(*side_range)
    w = rng.randint(*side_range)
    slots = _slots_for_grid(h, w, rng)
    chosen_slot_ids = rng.sample(range(4), fill_count)

    used_areas: set[int] = set()
    fill_pairs: list[dict] = []
    for slot_id in chosen_slot_ids:
        box = _sample_fill_box(slots[slot_id], used_areas, rng)
        if box is None:
            return None
        orientation = rng.choice(["main", "anti"])
        p1, p2 = _diag_markers(box, orientation)
        fill_pairs.append(
            {
                "slot_id": slot_id,
                "box": box,
                "p1": p1,
                "p2": p2,
                "area": _box_area(box),
            }
        )
        used_areas.add(_box_area(box))

    min_fill_area = min(pair["area"] for pair in fill_pairs)
    line_pairs: list[dict] = []
    for fill_pair in fill_pairs:
        line_pair = _sample_line_pair(
            fill_pair["box"], slots[fill_pair["slot_id"]], min_fill_area, used_areas, rng
        )
        if line_pair is None:
            return None
        used_areas.add(int(line_pair["area"]))
        line_pairs.append(line_pair)

    pairs = fill_pairs + line_pairs
    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]
    if len(pairs) > len(palette):
        return None
    colors = rng.sample(palette, len(pairs))
    rng.shuffle(pairs)
    grid = _build_grid(h, w, bg, colors, pairs)

    if not _valid_input(grid):
        return None

    desc_out, rects, lines = _apply_rules(grid, descending=True)
    asc_out, _, _ = _apply_rules(grid, descending=False)
    if rects != fill_count or lines != fill_count:
        return None
    if np.array_equal(desc_out, asc_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
