from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
BG = 0
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CORNER_TYPES = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
MAX_SIDE = 30
OUTER_MARGIN = 1
INTER_BAND_GAP = 1
GENERATION_RETRIES = 400


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    colors = sorted(set(map(int, grid.flatten())) - {bg, GREY})
    for color in colors:
        for coords in components_mask(grid == color):
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "cells": frozenset(map(tuple, coords.tolist())),
                }
            )
    return objects


def _vertical_hit(grid: np.ndarray, obj: dict, marker: Coord, bg: int) -> tuple[str, int] | None:
    h, _w = grid.shape
    r, c = marker
    r0, c0, r1, c1 = obj["bbox"]
    if not (c0 <= c <= c1):
        return None

    if r < r0:
        side = "top"
        rr = r + 1
        gap = 0
        while rr < h:
            value = int(grid[rr, c])
            if value == bg:
                gap += 1
                rr += 1
                continue
            if gap >= 1 and (rr, c) in obj["cells"]:
                return side, gap
            return None

    if r > r1:
        side = "bottom"
        rr = r - 1
        gap = 0
        while rr >= 0:
            value = int(grid[rr, c])
            if value == bg:
                gap += 1
                rr -= 1
                continue
            if gap >= 1 and (rr, c) in obj["cells"]:
                return side, gap
            return None

    return None


def _horizontal_hit(grid: np.ndarray, obj: dict, marker: Coord, bg: int) -> tuple[str, int] | None:
    _h, w = grid.shape
    r, c = marker
    r0, c0, r1, c1 = obj["bbox"]
    if not (r0 <= r <= r1):
        return None

    if c < c0:
        side = "left"
        cc = c + 1
        gap = 0
        while cc < w:
            value = int(grid[r, cc])
            if value == bg:
                gap += 1
                cc += 1
                continue
            if gap >= 1 and (r, cc) in obj["cells"]:
                return side, gap
            return None

    if c > c1:
        side = "right"
        cc = c - 1
        gap = 0
        while cc >= 0:
            value = int(grid[r, cc])
            if value == bg:
                gap += 1
                cc -= 1
                continue
            if gap >= 1 and (r, cc) in obj["cells"]:
                return side, gap
            return None

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase a corner rectangle from each object using the two orthogonal grey gap markers that face it.

    Input structure:
    - The grid has a dominant background, several monochrome 4-connected solid objects, and isolated grey pixels.
    - For each object, one grey marker lies directly above or below its bbox span and one lies directly left or right of its bbox span, with only background in the gap.
    - Extra grey singletons may appear as noise and do not face any object.

    Rule:
    - Pair each object with its unique vertical grey marker and unique horizontal grey marker by straight-line facing.
    - Let `a` be the background gap from the vertical marker to the object and `b` the corresponding horizontal gap.
    - In that object's own bbox, erase the `a x b` corner rectangle closest to both markers, then remove every grey pixel.

    Key insight:
    - Each marker provides only one side length and one side of the target corner; the missing notch is determined only by combining the two orthogonal gaps on the same object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = _find_objects(g, bg)
    assert objects, "expected at least one non-grey object"

    grey_markers = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    assert grey_markers, "expected grey markers"

    assignments = []
    used_markers: list[Coord] = []
    for obj in objects:
        vertical_hits = []
        horizontal_hits = []
        for marker in grey_markers:
            v_hit = _vertical_hit(g, obj, marker, bg)
            if v_hit is not None:
                vertical_hits.append((marker, *v_hit))
            h_hit = _horizontal_hit(g, obj, marker, bg)
            if h_hit is not None:
                horizontal_hits.append((marker, *h_hit))

        assert len(vertical_hits) == 1, "each object must have exactly one vertical marker"
        assert len(horizontal_hits) == 1, "each object must have exactly one horizontal marker"

        v_marker, v_side, v_gap = vertical_hits[0]
        h_marker, h_side, h_gap = horizontal_hits[0]
        assert v_marker != h_marker, "the two markers must be distinct"
        used_markers.extend([v_marker, h_marker])
        assignments.append((obj, v_side, int(v_gap), h_side, int(h_gap)))

    assert len(set(used_markers)) == len(used_markers), "markers must not be shared across objects"

    out = g.copy()
    for obj, v_side, v_gap, h_side, h_gap in assignments:
        r0, c0, r1, c1 = obj["bbox"]
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        assert 1 <= v_gap <= obj_h, "vertical gap must fit the object's height"
        assert 1 <= h_gap <= obj_w, "horizontal gap must fit the object's width"

        if v_side == "top":
            rr = slice(r0, r0 + v_gap)
        else:
            rr = slice(r1 - v_gap + 1, r1 + 1)

        if h_side == "left":
            cc = slice(c0, c0 + h_gap)
        else:
            cc = slice(c1 - h_gap + 1, c1 + 1)

        out[rr, cc] = bg

    out[g == GREY] = bg
    return out


def _share_one_axis_only(v_gaps: list[int], h_gaps: list[int]) -> bool:
    for i in range(len(v_gaps)):
        for j in range(i + 1, len(v_gaps)):
            same_v = v_gaps[i] == v_gaps[j] and h_gaps[i] != h_gaps[j]
            same_h = h_gaps[i] == h_gaps[j] and v_gaps[i] != v_gaps[j]
            if same_v:
                break
        else:
            continue
        break
    else:
        return False

    for i in range(len(v_gaps)):
        for j in range(i + 1, len(v_gaps)):
            if h_gaps[i] == h_gaps[j] and v_gaps[i] != v_gaps[j]:
                return True
    return False


def _sample_specs() -> list[dict] | None:
    n = random.choices([2, 3], weights=[1, 4], k=1)[0]
    colors = random.sample(list(OBJECT_COLORS), n)
    corners = random.sample(list(CORNER_TYPES), n)

    if n == 3:
        choices = [
            (1, 1, 2),
            (1, 2, 1),
            (2, 1, 1),
            (1, 2, 2),
            (2, 1, 2),
            (2, 2, 1),
        ]
        for _ in range(20):
            v_gaps = list(random.choice(choices))
            h_gaps = list(random.choice(choices))
            if _share_one_axis_only(v_gaps, h_gaps):
                break
        else:
            return None
        sizes = [(random.randint(3, 4), random.randint(3, 4)) for _ in range(n)]
    else:
        v_gaps = random.sample([1, 2], 2)
        h_gaps = random.sample([1, 2], 2)
        sizes = [(random.randint(3, 5), random.randint(3, 5)) for _ in range(n)]

    specs = []
    for idx in range(n):
        height, width = sizes[idx]
        v_side, h_side = corners[idx]
        specs.append(
            {
                "color": colors[idx],
                "height": height,
                "width": width,
                "v_gap": int(v_gaps[idx]),
                "h_gap": int(h_gaps[idx]),
                "v_side": v_side,
                "h_side": h_side,
                "marker_row_offset": random.randrange(height),
                "marker_col_offset": random.randrange(width),
            }
        )
    return specs


def _band_size(size: int, gap: int) -> int:
    return int(size + gap + 2)


def _distribute(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _layout_axis(band_sizes: list[int]) -> tuple[int, dict[int, int]] | None:
    n = len(band_sizes)
    order = random.sample(range(n), n)
    required = (
        2 * OUTER_MARGIN
        + sum(band_sizes[idx] for idx in order)
        + INTER_BAND_GAP * (n - 1)
    )
    if required > MAX_SIDE:
        return None

    total = random.randint(required, MAX_SIDE)
    base_gaps = [OUTER_MARGIN] + [INTER_BAND_GAP] * (n - 1) + [OUTER_MARGIN]
    extras = _distribute(total - required, n + 1)
    gap_sizes = [base_gaps[i] + extras[i] for i in range(n + 1)]

    starts: dict[int, int] = {}
    cur = gap_sizes[0]
    for pos, idx in enumerate(order):
        starts[idx] = cur
        cur += band_sizes[idx] + gap_sizes[pos + 1]
    return total, starts


def _place_specs(specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    row_sizes = [_band_size(spec["height"], spec["v_gap"]) for spec in specs]
    col_sizes = [_band_size(spec["width"], spec["h_gap"]) for spec in specs]
    row_layout = _layout_axis(row_sizes)
    col_layout = _layout_axis(col_sizes)
    if row_layout is None or col_layout is None:
        return None

    height, row_starts = row_layout
    width, col_starts = col_layout
    grid = np.full((height, width), BG, dtype=int)

    placed_specs: list[dict] = []
    for idx, spec in enumerate(specs):
        top_space = spec["v_gap"] + 1 if spec["v_side"] == "top" else 1
        left_space = spec["h_gap"] + 1 if spec["h_side"] == "left" else 1
        r0 = row_starts[idx] + top_space
        c0 = col_starts[idx] + left_space
        r1 = r0 + spec["height"] - 1
        c1 = c0 + spec["width"] - 1

        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]

        if spec["v_side"] == "top":
            v_marker = (r0 - spec["v_gap"] - 1, c0 + spec["marker_col_offset"])
        else:
            v_marker = (r1 + spec["v_gap"] + 1, c0 + spec["marker_col_offset"])

        if spec["h_side"] == "left":
            h_marker = (r0 + spec["marker_row_offset"], c0 - spec["h_gap"] - 1)
        else:
            h_marker = (r0 + spec["marker_row_offset"], c1 + spec["h_gap"] + 1)

        if v_marker == h_marker:
            return None

        if grid[v_marker] != BG or grid[h_marker] != BG:
            return None
        grid[v_marker] = GREY
        grid[h_marker] = GREY

        placed = dict(spec)
        placed["bbox"] = (r0, c0, r1, c1)
        placed["v_marker"] = v_marker
        placed["h_marker"] = h_marker
        placed_specs.append(placed)

    return grid, placed_specs


def _chebyshev_far(cell: Coord, others: set[Coord]) -> bool:
    r, c = cell
    for rr, cc in others:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _place_noise(grid: np.ndarray, specs: list[dict]) -> list[Coord] | None:
    h, w = grid.shape
    forbidden_rows = set()
    forbidden_cols = set()
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(grid != BG)}

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        forbidden_rows.update(range(r0, r1 + 1))
        forbidden_cols.update(range(c0, c1 + 1))

    candidates = [
        (r, c)
        for r in range(h)
        if r not in forbidden_rows
        for c in range(w)
        if c not in forbidden_cols and grid[r, c] == BG and _chebyshev_far((r, c), occupied)
    ]
    random.shuffle(candidates)
    chosen: list[Coord] = []
    occupied_plus = set(occupied)
    for cell in candidates:
        if _chebyshev_far(cell, occupied_plus):
            chosen.append(cell)
            occupied_plus.add(cell)
    if not chosen:
        return None

    noise_count = random.randint(1, min(3, len(chosen)))
    return chosen[:noise_count]


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    out[out == GREY] = BG
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        if spec["v_side"] == "top":
            rr = slice(r0, r0 + spec["v_gap"])
        else:
            rr = slice(r1 - spec["v_gap"] + 1, r1 + 1)
        if spec["h_side"] == "left":
            cc = slice(c0, c0 + spec["h_gap"])
        else:
            cc = slice(c1 - spec["h_gap"] + 1, c1 + 1)
        out[rr, cc] = BG
    return out


def _valid_instance(inp: np.ndarray, specs: list[dict], noise: list[Coord]) -> bool:
    values, counts = np.unique(inp, return_counts=True)
    count_by_color = {int(v): int(n) for v, n in zip(values, counts)}
    bg_count = count_by_color.get(BG, 0)
    if bg_count != max(count_by_color.values()):
        return False
    if list(count_by_color.values()).count(bg_count) != 1:
        return False

    row_spans = []
    col_spans = []
    v_gaps = []
    h_gaps = []
    corners = set()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        row_spans.append((r0, r1))
        col_spans.append((c0, c1))
        v_gaps.append(spec["v_gap"])
        h_gaps.append(spec["h_gap"])
        corners.add((spec["v_side"], spec["h_side"]))

    for i in range(len(row_spans)):
        for j in range(i + 1, len(row_spans)):
            if not (row_spans[i][1] < row_spans[j][0] or row_spans[j][1] < row_spans[i][0]):
                return False
            if not (col_spans[i][1] < col_spans[j][0] or col_spans[j][1] < col_spans[i][0]):
                return False

    if len(corners) != len(specs):
        return False
    if len(specs) == 3 and not _share_one_axis_only(v_gaps, h_gaps):
        return False

    for r, c in noise:
        if any(r0 <= r <= r1 for r0, r1 in row_spans):
            return False
        if any(c0 <= c <= c1 for c0, c1 in col_spans):
            return False

    expected = _expected_output(inp, specs)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    placed = _place_specs(specs)
    if placed is None:
        return None
    grid, placed_specs = placed

    noise = _place_noise(grid, placed_specs)
    if noise is None:
        return None
    for r, c in noise:
        grid[r, c] = GREY

    return grid if _valid_instance(grid, placed_specs, noise) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input for puzzle 036/5")
