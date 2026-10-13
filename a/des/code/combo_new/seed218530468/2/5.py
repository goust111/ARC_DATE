from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_VECS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
COLOR_PRIORITY = [2, 1, 3, 4, 7, 6, 8, 9, 5]
COLOR_RANK = {color: rank for rank, color in enumerate(COLOR_PRIORITY)}

MIN_PAIRS = 1
MAX_PAIRS = 4
MIN_GRID = 12
MAX_GRID = 30
MIN_STRAIGHT_GAP = 2
MIN_L_LEG = 2
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 180


def _tight_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple((int(r), int(c)) for r, c in coords)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in CARDINALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _notch_candidates(mask: np.ndarray) -> list[tuple[tuple[int, int], str]]:
    h, w = mask.shape
    candidates: list[tuple[tuple[int, int], str]] = []

    for c in range(1, w - 1):
        if not mask[0, c] and mask[0, c - 1] and mask[0, c + 1]:
            candidates.append(((0, c), "U"))
        if not mask[h - 1, c] and mask[h - 1, c - 1] and mask[h - 1, c + 1]:
            candidates.append(((h - 1, c), "D"))

    for r in range(1, h - 1):
        if not mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]:
            candidates.append(((r, 0), "L"))
        if not mask[r, w - 1] and mask[r - 1, w - 1] and mask[r + 1, w - 1]:
            candidates.append(((r, w - 1), "R"))

    return candidates


def _build_variant_library() -> list[dict]:
    variants: list[dict] = []
    for h in range(2, 5):
        for w in range(2, 5):
            cell_count = h * w
            for bits in range(1, 1 << cell_count):
                area = bits.bit_count()
                if not (3 <= area <= 8):
                    continue

                mask = np.zeros((h, w), dtype=bool)
                for idx in range(cell_count):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True

                if not mask[0, :].any() or not mask[h - 1, :].any():
                    continue
                if not mask[:, 0].any() or not mask[:, w - 1].any():
                    continue
                if not _connected(mask):
                    continue

                candidates = _notch_candidates(mask)
                if len(candidates) != 1:
                    continue

                (nr, nc), direction = candidates[0]
                filled = mask.copy()
                filled[nr, nc] = True
                if not _connected(filled):
                    continue

                variants.append(
                    {
                        "mask": mask,
                        "cells": tuple((int(r), int(c)) for r, c in np.argwhere(mask)),
                        "h": h,
                        "w": w,
                        "area": area,
                        "notch": (int(nr), int(nc)),
                        "dir": direction,
                        "sig": _tight_signature(mask),
                        "base_sig": _tight_signature(filled),
                    }
                )
    assert variants, "variant library must not be empty"
    return variants


VARIANTS = _build_variant_library()
VARIANTS_BY_BASE: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
for variant in VARIANTS:
    VARIANTS_BY_BASE[variant["base_sig"]].append(variant)

ALL_BASE_KEYS = list(VARIANTS_BY_BASE.keys())
L_BASE_KEYS = [
    base_sig
    for base_sig, variants in VARIANTS_BY_BASE.items()
    if any(v["dir"] in {"L", "R"} for v in variants) and any(v["dir"] in {"U", "D"} for v in variants)
]


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            coords: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _analyze_component(component: dict) -> dict:
    coords = component["coords"]
    color = int(component["color"])
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    candidates = _notch_candidates(mask)
    assert len(candidates) == 1, "each object must have exactly one flanked boundary notch"
    (nr, nc), direction = candidates[0]
    filled = mask.copy()
    filled[nr, nc] = True

    return {
        "color": color,
        "coords": tuple(coords),
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "sig": _tight_signature(mask),
        "base_sig": _tight_signature(filled),
        "notch_local": (int(nr), int(nc)),
        "notch_global": (int(r0 + nr), int(c0 + nc)),
        "dir": direction,
    }


def _ray_cells(start: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIR_VECS[direction]
    h, w = shape
    r, c = start
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _prefix_to(cells: list[tuple[int, int]], stop: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cell in cells:
        out.append(cell)
        if cell == stop:
            return out
    raise AssertionError("stop cell must lie on ray")


def _bridge_cells(a: dict, b: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    ray_a = _ray_cells(a["notch_global"], a["dir"], shape)
    ray_b = _ray_cells(b["notch_global"], b["dir"], shape)
    ray_b_set = set(ray_b)

    meeting = None
    for cell in ray_a:
        if cell in ray_b_set:
            meeting = cell
            break
    assert meeting is not None, "paired notch rays must intersect"

    return set(_prefix_to(ray_a, meeting)) | set(_prefix_to(ray_b, meeting))


def _bridge_color(color_a: int, color_b: int) -> int:
    assert color_a in COLOR_RANK and color_b in COLOR_RANK
    return color_a if COLOR_RANK[color_a] < COLOR_RANK[color_b] else color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each color-mismatched pair of single-notch motifs by extending rays from their notches until the rays meet.

    Input structure:
    - The background is black.
    - The foreground consists of several small monochrome 4-connected motifs, each using 3 to 8 pixels.
    - Every motif has exactly one one-cell notch on its bbox boundary; if that notch is filled back in, the resulting silhouette appears exactly twice, with two different colors.

    Rule:
    - Find each motif's unique flanked boundary notch and use the notch-filled silhouette to pair the motifs.
    - For each pair, choose the bridge color by the fixed visible priority red > blue > green > yellow > orange > magenta > azure > maroon > grey.
    - Starting from the two notch cells, cast one straight cardinal ray in each notch's outward direction and color the shortest prefixes up to their first intersection, producing an L-shaped or straight bridge.
    - Keep the original motifs unchanged.

    Key insight:
    - The missing boundary cell serves two roles at once: filling it normalizes the pairing shape, while its border side tells you exactly where each bridge ray starts and which way it travels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components and len(components) % 2 == 0, "expected an even number of foreground objects"

    analyzed = [_analyze_component(component) for component in components]
    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for item in analyzed:
        groups[item["base_sig"]].append(item)

    assert groups and all(len(items) == 2 for items in groups.values()), "pairing by filled silhouette must be unambiguous"

    out = g.copy()
    h, w = g.shape
    for pair in groups.values():
        a, b = pair
        color = _bridge_color(a["color"], b["color"])
        bridge = _bridge_cells(a, b, (h, w))
        for r, c in bridge:
            assert int(g[r, c]) == BLACK, "bridge path must stay off existing motifs"
            assert int(out[r, c]) in (BLACK, color), "bridges must not overlap each other"
            out[r, c] = color

    return out


def _shifted_cells(variant: dict, origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in variant["cells"]}


def _make_item(variant: dict, origin: tuple[int, int], color: int) -> dict:
    nr, nc = variant["notch"]
    r0, c0 = origin
    return {
        "color": color,
        "base_sig": variant["base_sig"],
        "dir": variant["dir"],
        "notch_global": (r0 + nr, c0 + nc),
        "cells": _shifted_cells(variant, origin),
        "bbox": (r0, c0, r0 + variant["h"] - 1, c0 + variant["w"] - 1),
    }


def _bbox_of(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded_bbox(cells: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = _bbox_of(cells)
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _in_bounds(origin: tuple[int, int], variant: dict, shape: tuple[int, int]) -> bool:
    r0, c0 = origin
    h, w = shape
    return 0 <= r0 and 0 <= c0 and r0 + variant["h"] <= h and c0 + variant["w"] <= w


def _branch_plan(pair_count: int) -> list[str]:
    if pair_count == 1:
        return [random.choice(("straight", "l"))]

    l_count = pair_count // 2
    if pair_count % 2 == 1 and random.random() < 0.5:
        l_count += 1
    plan = ["l"] * l_count + ["straight"] * (pair_count - l_count)
    random.shuffle(plan)
    return plan


def _choose_variants(base_sig: tuple[tuple[int, int], ...], branch: str) -> tuple[dict, dict]:
    variants = VARIANTS_BY_BASE[base_sig]
    if branch == "l":
        horizontals = [v for v in variants if v["dir"] in {"L", "R"}]
        verticals = [v for v in variants if v["dir"] in {"U", "D"}]
        assert horizontals and verticals
        return random.choice(horizontals), random.choice(verticals)

    same_axis_pairs = [
        (a, b)
        for a in variants
        for b in variants
        if (a["dir"] in {"L", "R"} and b["dir"] in {"L", "R"})
        or (a["dir"] in {"U", "D"} and b["dir"] in {"U", "D"})
    ]
    if same_axis_pairs and random.random() < 0.55:
        return random.choice(same_axis_pairs)
    chosen = random.choice(variants)
    return chosen, chosen


def _straight_positions(
    variant_a: dict,
    variant_b: dict,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    dir_a = variant_a["dir"]
    dir_b = variant_b["dir"]

    if dir_a in {"L", "R"} and dir_b in {"L", "R"}:
        feasible_rows = range(
            max(variant_a["notch"][0], variant_b["notch"][0]),
            min(h - variant_a["h"] + variant_a["notch"][0], h - variant_b["h"] + variant_b["notch"][0]) + 1,
        )
        feasible_rows = list(feasible_rows)
        if not feasible_rows:
            return None
        row = random.choice(feasible_rows)

        a_lo = variant_a["notch"][1]
        a_hi = w - variant_a["w"] + variant_a["notch"][1]
        b_lo = variant_b["notch"][1]
        b_hi = w - variant_b["w"] + variant_b["notch"][1]

        if dir_a == "R":
            left_lo, left_hi = a_lo, a_hi
            right_lo, right_hi = b_lo, b_hi
            lo_left = left_lo
            hi_left = min(left_hi, right_hi - MIN_STRAIGHT_GAP)
            if hi_left < lo_left:
                return None
            left = random.randint(lo_left, hi_left)
            right_lo2 = max(right_lo, left + MIN_STRAIGHT_GAP)
            if right_lo2 > right_hi:
                return None
            right = random.randint(right_lo2, right_hi)
            notch_a = (row, left)
            notch_b = (row, right)
        else:
            left_lo, left_hi = b_lo, b_hi
            right_lo, right_hi = a_lo, a_hi
            lo_left = left_lo
            hi_left = min(left_hi, right_hi - MIN_STRAIGHT_GAP)
            if hi_left < lo_left:
                return None
            left = random.randint(lo_left, hi_left)
            right_lo2 = max(right_lo, left + MIN_STRAIGHT_GAP)
            if right_lo2 > right_hi:
                return None
            right = random.randint(right_lo2, right_hi)
            notch_b = (row, left)
            notch_a = (row, right)

        origin_a = (notch_a[0] - variant_a["notch"][0], notch_a[1] - variant_a["notch"][1])
        origin_b = (notch_b[0] - variant_b["notch"][0], notch_b[1] - variant_b["notch"][1])
        if _in_bounds(origin_a, variant_a, shape) and _in_bounds(origin_b, variant_b, shape):
            return origin_a, origin_b
        return None

    if dir_a in {"U", "D"} and dir_b in {"U", "D"}:
        feasible_cols = range(
            max(variant_a["notch"][1], variant_b["notch"][1]),
            min(w - variant_a["w"] + variant_a["notch"][1], w - variant_b["w"] + variant_b["notch"][1]) + 1,
        )
        feasible_cols = list(feasible_cols)
        if not feasible_cols:
            return None
        col = random.choice(feasible_cols)

        a_lo = variant_a["notch"][0]
        a_hi = h - variant_a["h"] + variant_a["notch"][0]
        b_lo = variant_b["notch"][0]
        b_hi = h - variant_b["h"] + variant_b["notch"][0]

        if dir_a == "D":
            top_lo, top_hi = a_lo, a_hi
            bottom_lo, bottom_hi = b_lo, b_hi
            lo_top = top_lo
            hi_top = min(top_hi, bottom_hi - MIN_STRAIGHT_GAP)
            if hi_top < lo_top:
                return None
            top = random.randint(lo_top, hi_top)
            bottom_lo2 = max(bottom_lo, top + MIN_STRAIGHT_GAP)
            if bottom_lo2 > bottom_hi:
                return None
            bottom = random.randint(bottom_lo2, bottom_hi)
            notch_a = (top, col)
            notch_b = (bottom, col)
        else:
            top_lo, top_hi = b_lo, b_hi
            bottom_lo, bottom_hi = a_lo, a_hi
            lo_top = top_lo
            hi_top = min(top_hi, bottom_hi - MIN_STRAIGHT_GAP)
            if hi_top < lo_top:
                return None
            top = random.randint(lo_top, hi_top)
            bottom_lo2 = max(bottom_lo, top + MIN_STRAIGHT_GAP)
            if bottom_lo2 > bottom_hi:
                return None
            bottom = random.randint(bottom_lo2, bottom_hi)
            notch_b = (top, col)
            notch_a = (bottom, col)

        origin_a = (notch_a[0] - variant_a["notch"][0], notch_a[1] - variant_a["notch"][1])
        origin_b = (notch_b[0] - variant_b["notch"][0], notch_b[1] - variant_b["notch"][1])
        if _in_bounds(origin_a, variant_a, shape) and _in_bounds(origin_b, variant_b, shape):
            return origin_a, origin_b
        return None

    return None


def _l_positions(
    variant_h: dict,
    variant_v: dict,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape

    row_lo = variant_h["notch"][0]
    row_hi = h - variant_h["h"] + variant_h["notch"][0]
    col_lo = variant_v["notch"][1]
    col_hi = w - variant_v["w"] + variant_v["notch"][1]
    if row_lo > row_hi or col_lo > col_hi:
        return None

    row = random.randint(row_lo, row_hi)
    col = random.randint(col_lo, col_hi)

    h_col_lo = variant_h["notch"][1]
    h_col_hi = w - variant_h["w"] + variant_h["notch"][1]
    v_row_lo = variant_v["notch"][0]
    v_row_hi = h - variant_v["h"] + variant_v["notch"][0]

    if variant_h["dir"] == "R":
        notch_h_col_hi = min(h_col_hi, col - MIN_L_LEG)
        if h_col_lo > notch_h_col_hi:
            return None
        notch_h_col = random.randint(h_col_lo, notch_h_col_hi)
    else:
        notch_h_col_lo = max(h_col_lo, col + MIN_L_LEG)
        if notch_h_col_lo > h_col_hi:
            return None
        notch_h_col = random.randint(notch_h_col_lo, h_col_hi)

    if variant_v["dir"] == "D":
        notch_v_row_hi = min(v_row_hi, row - MIN_L_LEG)
        if v_row_lo > notch_v_row_hi:
            return None
        notch_v_row = random.randint(v_row_lo, notch_v_row_hi)
    else:
        notch_v_row_lo = max(v_row_lo, row + MIN_L_LEG)
        if notch_v_row_lo > v_row_hi:
            return None
        notch_v_row = random.randint(notch_v_row_lo, v_row_hi)

    origin_h = (row - variant_h["notch"][0], notch_h_col - variant_h["notch"][1])
    origin_v = (notch_v_row - variant_v["notch"][0], col - variant_v["notch"][1])
    if _in_bounds(origin_h, variant_h, shape) and _in_bounds(origin_v, variant_v, shape):
        return origin_h, origin_v
    return None


def _pair_spec(
    branch: str,
    base_sig: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
) -> dict | None:
    variant_a, variant_b = _choose_variants(base_sig, branch)
    color_a, color_b = random.sample(COLOR_PRIORITY, 2)

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        if branch == "straight":
            origins = _straight_positions(variant_a, variant_b, shape)
        else:
            origins = _l_positions(variant_a, variant_b, shape)
        if origins is None:
            continue

        origin_a, origin_b = origins
        item_a = _make_item(variant_a, origin_a, color_a)
        item_b = _make_item(variant_b, origin_b, color_b)
        if item_a["cells"] & item_b["cells"]:
            continue

        bridge = _bridge_cells(item_a, item_b, shape)
        object_cells = item_a["cells"] | item_b["cells"]
        if bridge & object_cells:
            continue

        if branch == "straight":
            rows = {r for r, _c in bridge}
            cols = {c for _r, c in bridge}
            if not (len(rows) == 1 or len(cols) == 1):
                continue
        else:
            rows = {r for r, _c in bridge}
            cols = {c for _r, c in bridge}
            if len(rows) == 1 or len(cols) == 1:
                continue

        all_cells = object_cells | bridge
        return {
            "items": (item_a, item_b),
            "bridge": bridge,
            "bridge_color": _bridge_color(color_a, color_b),
            "branch": branch,
            "box": _expanded_bbox(all_cells, shape),
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(MIN_PAIRS, MAX_PAIRS)
    h = random.randint(max(MIN_GRID, 8 + pair_count * 4), MAX_GRID)
    w = random.randint(max(MIN_GRID, 8 + pair_count * 4), MAX_GRID)
    shape = (h, w)

    branches = _branch_plan(pair_count)
    used_bases: set[tuple[tuple[int, int], ...]] = set()
    pairs: list[dict] = []

    for branch in branches:
        pool = [key for key in (L_BASE_KEYS if branch == "l" else ALL_BASE_KEYS) if key not in used_bases]
        if not pool:
            return None

        random.shuffle(pool)
        placed = None
        for base_sig in pool[:80]:
            spec = _pair_spec(branch, base_sig, shape)
            if spec is None:
                continue
            if any(_boxes_overlap(spec["box"], other["box"]) for other in pairs):
                continue
            placed = spec
            used_bases.add(base_sig)
            pairs.append(spec)
            break
        if placed is None:
            return None

    grid = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)
    for spec in pairs:
        for item in spec["items"]:
            for r, c in item["cells"]:
                assert grid[r, c] == BLACK
                grid[r, c] = item["color"]
                expected[r, c] = item["color"]
        for r, c in spec["bridge"]:
            expected[r, c] = spec["bridge_color"]

    got = main(grid)
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-pair ARC input")
