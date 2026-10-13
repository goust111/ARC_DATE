from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BLACK = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MISSING_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}
COMPLEMENT = {"TL": "BR", "BR": "TL", "TR": "BL", "BL": "TR"}
PAIR_TYPES = (("TL", "BR"), ("TR", "BL"))

# --------------- solver ---------------


def _half_up_mean(a: int, b: int) -> int:
    return (a + b + 1) // 2


def _components4(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return out


def _classify_l(component: dict) -> dict:
    color = int(component["color"])
    assert color != GREY, "grey is reserved for the merged corner"

    coords = {(int(r), int(c)) for r, c in component["coords"]}
    assert len(coords) == 3, "every foreground component must contain exactly three cells"

    r0, c0, r1, c1 = component["bbox"]
    assert r1 - r0 == 1 and c1 - c0 == 1, "every component must fit in a 2x2 bbox"

    missing = None
    for name, (dr, dc) in MISSING_OFFSETS.items():
        cell = (r0 + dr, c0 + dc)
        if cell not in coords:
            assert missing is None, "an L must miss exactly one corner"
            missing = name
    assert missing is not None, "expected one missing corner in each 2x2 bbox"

    out = dict(component)
    out["missing"] = missing
    out["order_key"] = (r0, c0)
    out["top_left"] = (r0, c0)
    return out


def _distance_sq(a: dict, b: dict) -> int:
    ar, ac = a["top_left"]
    br, bc = b["top_left"]
    dr = ar - br
    dc = ac - bc
    return dr * dr + dc * dc


def _match_groups(left: list[dict], right: list[dict]) -> list[tuple[dict, dict]]:
    left = sorted(left, key=lambda obj: obj["order_key"])
    right = sorted(right, key=lambda obj: obj["order_key"])
    assert len(left) == len(right), "complementary orientations must occur equally often"
    n = len(left)
    if n == 0:
        return []

    costs = [[_distance_sq(a, b) for b in right] for a in left]

    @lru_cache(maxsize=None)
    def solve(i: int, mask: int) -> tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]:
        if i == n:
            return 0, (), ()

        best = None
        for j in range(n):
            if mask & (1 << j):
                continue
            sub_cost, sub_sig, sub_idx = solve(i + 1, mask | (1 << j))
            cand = (
                costs[i][j] + sub_cost,
                (right[j]["order_key"],) + sub_sig,
                (j,) + sub_idx,
            )
            if best is None or cand[:2] < best[:2]:
                best = cand

        assert best is not None
        return best

    _cost, _sig, indices = solve(0, 0)
    return [(left[i], right[j]) for i, j in enumerate(indices)]


def _draw_output_pair(out: np.ndarray, occupied: np.ndarray, a: dict, b: dict) -> None:
    first, second = sorted((a, b), key=lambda obj: obj["order_key"])
    r0 = _half_up_mean(first["bbox"][0], second["bbox"][0])
    c0 = _half_up_mean(first["bbox"][1], second["bbox"][1])
    h, w = out.shape
    assert 0 <= r0 < h - 1 and 0 <= c0 < w - 1, "merged square must stay inside the grid"

    region = occupied[r0 : r0 + 2, c0 : c0 + 2]
    assert not region.any(), "merged squares must not overlap"
    occupied[r0 : r0 + 2, c0 : c0 + 2] = True

    out[r0 : r0 + 2, c0 : c0 + 2] = int(first["color"])
    dr, dc = MISSING_OFFSETS[first["missing"]]
    out[r0 + dr, c0 + dc] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary L-shaped 2x2-minus-one sprites and collapse each pair into a centered 2x2 square with a grey completion cell.

    Input structure:
    - Background is black.
    - Every foreground object is a monochrome 4-connected L made of three cells inside a 2x2 bbox.
    - Each L is missing exactly one corner: top-left, top-right, bottom-left, or bottom-right.
    - Complementary gaps come in opposite pairs (top-left with bottom-right, top-right with bottom-left), and every L belongs to one nearest complementary pair.

    Rule:
    - Detect all 3-cell L components and classify the corner missing from each 2x2 bbox.
    - Within each complementary orientation class, pair the Ls by nearest bbox-center distance.
    - For each pair, place a 2x2 square at the half-up rounded midpoint of the two L bboxes.
    - Use the earlier member of the pair in reading order as the template: its three occupied cells keep its color, and its missing-corner cell becomes grey.
    - Clear the original L positions to black.

    Key insight:
    - The hidden relation is the complementary missing corner, not the color; once the right Ls are paired, each pair disappears and reappears as one midpoint square that preserves one L's shape plus a grey completion corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = [_classify_l(comp) for comp in _components4(g)]
    assert components and len(components) % 2 == 0, "expected an even number of L components"

    matched_pairs: list[tuple[dict, dict]] = []
    used = 0
    for left_missing, right_missing in PAIR_TYPES:
        left = [obj for obj in components if obj["missing"] == left_missing]
        right = [obj for obj in components if obj["missing"] == right_missing]
        matched_pairs.extend(_match_groups(left, right))
        used += len(left) + len(right)
    assert used == len(components), "every component must have a complementary orientation"

    out = np.full_like(g, BLACK)
    occupied = np.zeros(g.shape, dtype=bool)
    for a, b in matched_pairs:
        _draw_output_pair(out, occupied, a, b)
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 300
PAIR_COUNT_CHOICES = (2, 3, 4)
REGION_SIZE = 7
REGION_GAP_CHOICES = (1, 2)
LOCAL_MIDPOINT_CHOICES = (2, 3)
LAYOUT_CHOICES = {
    2: ((1, 2), (2, 1), (2, 2)),
    3: ((1, 3), (3, 1), (2, 2)),
    4: ((2, 2),),
}


def _local_pair_positions(mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    mr = random.choice(LOCAL_MIDPOINT_CHOICES)
    mc = random.choice(LOCAL_MIDPOINT_CHOICES)

    if mode == "vertical":
        r1, r2 = mr - 1, mr + 1
        if random.random() < 0.5:
            dc = random.choice((0, 1))
            c1, c2 = mc - dc, mc + dc
        else:
            dc = random.choice((0, 1))
            c1, c2 = mc - dc, mc + dc - 1
        return (r1, c1), (r2, c2)

    c1, c2 = mc - 1, mc + 1
    row_modes = (
        (mr, mr),
        (mr - 1, mr + 1),
        (mr - 1, mr),
    )
    r1, r2 = random.choice(row_modes)
    return (r1, c1), (r2, c2)


def _draw_l(grid: np.ndarray, top_left: tuple[int, int], missing: str, color: int) -> None:
    r0, c0 = top_left
    for name, (dr, dc) in MISSING_OFFSETS.items():
        if name == missing:
            continue
        grid[r0 + dr, c0 + dc] = color


def _expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for spec in specs:
        r0, c0 = spec["target_top_left"]
        assert not occupied[r0 : r0 + 2, c0 : c0 + 2].any()
        occupied[r0 : r0 + 2, c0 : c0 + 2] = True
        out[r0 : r0 + 2, c0 : c0 + 2] = int(spec["first_color"])
        dr, dc = MISSING_OFFSETS[spec["first_missing"]]
        out[r0 + dr, c0 + dc] = GREY
    return out


def _sample_layout(pair_count: int, gap: int) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]]:
    layout = random.choice(LAYOUT_CHOICES[pair_count])
    rows, cols = layout
    stride = REGION_SIZE + gap
    core_h = rows * REGION_SIZE + (rows - 1) * gap
    core_w = cols * REGION_SIZE + (cols - 1) * gap

    top = random.randint(0, min(2, 30 - core_h))
    left = random.randint(0, min(2, 30 - core_w))
    bottom = random.randint(0, min(2, 30 - core_h - top))
    right = random.randint(0, min(2, 30 - core_w - left))

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen_slots = random.sample(slots, pair_count)
    origins = [(top + r * stride, left + c * stride) for r, c in chosen_slots]
    shape = (core_h + top + bottom, core_w + left + right)
    return shape[0], shape[1], origins, shape


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    gap = random.choice(REGION_GAP_CHOICES)
    height, width, origins, shape = _sample_layout(pair_count, gap)
    assert shape == (height, width)

    palette = random.sample(COLORS, 2 * pair_count)
    grid = np.full(shape, BLACK, dtype=int)
    specs: list[dict] = []

    for i, origin in enumerate(origins):
        pair_type = random.choice(PAIR_TYPES)
        first_missing = random.choice(pair_type)
        second_missing = COMPLEMENT[first_missing]
        first_color = palette[2 * i]
        second_color = palette[2 * i + 1]
        mode = random.choice(("vertical", "horizontal"))

        first_local, second_local = _local_pair_positions(mode)
        first_top_left = (origin[0] + first_local[0], origin[1] + first_local[1])
        second_top_left = (origin[0] + second_local[0], origin[1] + second_local[1])
        target_top_left = (
            _half_up_mean(first_top_left[0], second_top_left[0]),
            _half_up_mean(first_top_left[1], second_top_left[1]),
        )

        _draw_l(grid, first_top_left, first_missing, first_color)
        _draw_l(grid, second_top_left, second_missing, second_color)
        specs.append(
            {
                "first_missing": first_missing,
                "first_color": first_color,
                "target_top_left": target_top_left,
            }
        )

    expected = _expected_from_specs(shape, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid L-pair midpoint inputs")
