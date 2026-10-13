from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3, 4)
PATTERN_COLOR = 7

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[Coord]:
    h, w = shape
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _components_from_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, (h, w), connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_rect_border(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _clear_rect_border(grid: np.ndarray, bbox: BBox) -> None:
    _draw_rect_border(grid, bbox, BG)


def _rectangles_by_color(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g) if int(v) not in (BG, PATTERN_COLOR)]
    assert 1 <= len(colors) <= len(FRAME_COLORS), "expected 1-4 corner colors"

    rects: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 4, "each frame color must appear exactly four times"
        rows = sorted(set(map(int, coords[:, 0])))
        cols = sorted(set(map(int, coords[:, 1])))
        assert len(rows) == 2 and len(cols) == 2, "four pixels must define two rows and two columns"
        expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, "same-color pixels must be rectangle corners"
        rects.append({"color": color, "bbox": (rows[0], cols[0], rows[1], cols[1])})
    return rects


def _interior_pattern(grid: np.ndarray, bbox: BBox) -> dict | None:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.any(interior != BG):
        return None

    non_bg = interior != BG
    assert set(map(int, np.unique(interior[non_bg]))) == {PATTERN_COLOR}, "interior motif must be orange"
    comps = _components_from_mask(non_bg, connectivity=8)
    assert len(comps) == 1, "each rectangle may contain at most one orange component"
    local_bbox = _bbox_from_coords(comps[0])
    lr0, lc0, lr1, lc1 = local_bbox
    return {
        "bbox_local": local_bbox,
        "bbox_global": (r0 + 1 + lr0, c0 + 1 + lc0, r0 + 1 + lr1, c0 + 1 + lc1),
        "coords_local": comps[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color corner quartet into a rectangle, then either fill an enclosed incomplete orange motif or erase the empty frame.

    Input structure:
    - The grid has a black background with 3-4 distinct corner colors chosen from blue, red, green, and yellow.
    - For each such color, exactly four singleton pixels appear and they are the four corners of one axis-aligned rectangle.
    - Some rectangles contain one small 8-connected orange motif strictly inside the rectangle; other rectangles have all-black interiors.

    Rule:
    - Group the non-orange, non-black pixels by color and interpret each group of four as the corners of one rectangle.
    - Draw that rectangle's one-pixel border in the same color.
    - If the rectangle interior contains an orange component, find its bounding box and recolor every black cell inside that box to the rectangle color while leaving the orange cells unchanged.
    - If the rectangle interior contains no non-black pixels, erase the entire just-drawn border back to black.

    Key insight:
    - The isolated same-color corner pixels are not independent objects: they define hidden rectangles whose existence both controls whether a frame survives and supplies the color used to complete the interior motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for rect in _rectangles_by_color(g):
        color = int(rect["color"])
        bbox = rect["bbox"]
        _draw_rect_border(out, bbox, color)

        pattern = _interior_pattern(g, bbox)
        if pattern is None:
            _clear_rect_border(out, bbox)
            continue

        pr0, pc0, pr1, pc1 = pattern["bbox_global"]
        region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        region[region == BG] = color

    return out


# --------------- generator (o5x) ---------------

GEN_MAX_ATTEMPTS = 500
GEN_MARGIN = 1
GEN_GAP = 1


def _separated(a: BBox, b: BBox, gap: int = GEN_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _build_pattern_catalog() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for h in (2, 3):
        for w in (2, 3):
            area = h * w
            for k in range(3, min(6, area - 1) + 1):
                for cells in itertools.combinations(range(area), k):
                    mask = np.zeros((h, w), dtype=bool)
                    mask.flat[list(cells)] = True
                    if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
                        continue
                    if len(_components_from_mask(mask, connectivity=8)) != 1:
                        continue
                    masks.append(mask.astype(int))
    return masks


PATTERN_MASKS = _build_pattern_catalog()


def _sample_grid_shape(rect_count: int) -> tuple[int, int]:
    if rect_count == 4:
        return random.randint(21, 24), random.randint(21, 24)
    return random.randint(18, 23), random.randint(18, 23)


def _sample_rect_size(has_pattern: bool) -> tuple[int, int]:
    if has_pattern:
        return random.randint(7, 9), random.randint(7, 9)
    return random.randint(5, 8), random.randint(5, 8)


def _place_rectangles(specs: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    placed: list[dict] = []

    order = sorted(range(len(specs)), key=lambda i: specs[i]["size"][0] * specs[i]["size"][1], reverse=True)
    for idx in order:
        rh, rw = specs[idx]["size"]
        if rh + 2 * GEN_MARGIN > h or rw + 2 * GEN_MARGIN > w:
            return None

        candidates: list[BBox] = []
        for r0 in range(GEN_MARGIN, h - rh - GEN_MARGIN + 1):
            for c0 in range(GEN_MARGIN, w - rw - GEN_MARGIN + 1):
                bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if all(_separated(bbox, prev["bbox"]) for prev in placed):
                    candidates.append(bbox)
        if not candidates:
            return None

        bbox = random.choice(candidates)
        spec = dict(specs[idx])
        spec["bbox"] = bbox
        placed.append(spec)

    return placed


def _stamp_corners(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0] = color
    grid[r0, c1] = color
    grid[r1, c0] = color
    grid[r1, c1] = color


def _place_pattern(grid: np.ndarray, bbox: BBox) -> dict:
    r0, c0, r1, c1 = bbox
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1
    candidates = [mask for mask in PATTERN_MASKS if mask.shape[0] + 4 <= rh and mask.shape[1] + 4 <= rw]
    assert candidates, "patterned rectangles must be large enough"

    mask = random.choice(candidates)
    ph, pw = mask.shape
    top = random.randint(r0 + 2, r1 - ph - 1)
    left = random.randint(c0 + 2, c1 - pw - 1)
    view = grid[top : top + ph, left : left + pw]
    view[mask.astype(bool)] = PATTERN_COLOR

    coords = np.argwhere(mask.astype(bool))
    return {
        "pattern_bbox": (top, left, top + ph - 1, left + pw - 1),
        "pattern_coords": [(top + int(rr), left + int(cc)) for rr, cc in coords],
    }


def _expected_from_metadata(inp: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for rect in rects:
        color = int(rect["color"])
        bbox = rect["bbox"]
        _draw_rect_border(out, bbox, color)
        if rect["has_pattern"]:
            pr0, pc0, pr1, pc1 = rect["pattern_bbox"]
            region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
            region[region == BG] = color
        else:
            _clear_rect_border(out, bbox)
    return out


def _validate_generated_input(inp: np.ndarray, rects: list[dict]) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert any(rect["has_pattern"] for rect in rects)
    assert any(not rect["has_pattern"] for rect in rects)

    analyzed = sorted(_rectangles_by_color(inp), key=lambda x: x["color"])
    expected = sorted(({"color": r["color"], "bbox": r["bbox"]} for r in rects), key=lambda x: x["color"])
    assert analyzed == expected, "corner grouping must be unique"

    orange_mask = inp == PATTERN_COLOR
    covered_orange = np.zeros_like(orange_mask, dtype=bool)

    for i, rect in enumerate(rects):
        bbox = rect["bbox"]
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5
        for other in rects[i + 1 :]:
            assert _separated(bbox, other["bbox"]), "rectangles must not overlap or touch"

        interior = inp[r0 + 1 : r1, c0 + 1 : c1]
        if rect["has_pattern"]:
            info = _interior_pattern(inp, bbox)
            assert info is not None
            pr0, pc0, pr1, pc1 = info["bbox_global"]
            assert pr0 >= r0 + 2 and pc0 >= c0 + 2 and pr1 <= r1 - 2 and pc1 <= c1 - 2
            assert pr1 - pr0 + 1 <= 3 and pc1 - pc0 + 1 <= 3
            assert 3 <= len(rect["pattern_coords"]) <= 6
            motif = inp[pr0 : pr1 + 1, pc0 : pc1 + 1]
            assert np.any(motif == BG), "motif bbox must be incomplete"
            covered_orange[pr0 : pr1 + 1, pc0 : pc1 + 1] |= motif == PATTERN_COLOR
        else:
            assert np.all(interior == BG), "empty rectangles must have no interior motif"

    assert np.array_equal(covered_orange, orange_mask), "all orange pixels must belong to some rectangle motif"


def _validate_generated_output(inp: np.ndarray, out: np.ndarray, rects: list[dict]) -> None:
    for rect in rects:
        color = int(rect["color"])
        r0, c0, r1, c1 = rect["bbox"]
        border_mask = np.zeros_like(inp, dtype=bool)
        border_mask[r0, c0 : c1 + 1] = True
        border_mask[r1, c0 : c1 + 1] = True
        border_mask[r0 : r1 + 1, c0] = True
        border_mask[r0 : r1 + 1, c1] = True

        if rect["has_pattern"]:
            assert np.all(out[border_mask] == color), "patterned frame must survive"
            pr0, pc0, pr1, pc1 = rect["pattern_bbox"]
            region_in = inp[pr0 : pr1 + 1, pc0 : pc1 + 1]
            region_out = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
            assert np.all(region_out[region_in == PATTERN_COLOR] == PATTERN_COLOR), "orange motif must remain"
            assert np.all(region_out[region_in == BG] == color), "missing motif cells must be filled with frame color"
            assert np.any(region_in == BG), "fill branch must do real work"
        else:
            assert np.all(out[border_mask] == BG), "empty frame must be erased"


def _generate_case() -> tuple[np.ndarray, list[dict]] | None:
    rect_count = random.randint(3, 4)
    pattern_count = random.randint(1, rect_count - 1)
    has_patterns = [True] * pattern_count + [False] * (rect_count - pattern_count)
    random.shuffle(has_patterns)

    specs = []
    colors = random.sample(FRAME_COLORS, rect_count)
    for color, has_pattern in zip(colors, has_patterns):
        specs.append({"color": color, "has_pattern": has_pattern, "size": _sample_rect_size(has_pattern)})

    rects = _place_rectangles(specs, _sample_grid_shape(rect_count))
    if rects is None:
        return None

    h = max(rect["bbox"][2] for rect in rects) + GEN_MARGIN + 1
    w = max(rect["bbox"][3] for rect in rects) + GEN_MARGIN + 1
    inp = np.zeros((h, w), dtype=int)

    for rect in rects:
        _stamp_corners(inp, rect["bbox"], int(rect["color"]))
        if rect["has_pattern"]:
            rect.update(_place_pattern(inp, rect["bbox"]))

    return inp, rects


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        candidate = _generate_case()
        if candidate is None:
            continue

        inp, rects = candidate
        try:
            _validate_generated_input(inp, rects)
            expected = _expected_from_metadata(inp, rects)
            actual = main(inp)
            assert np.array_equal(actual, expected), "solver output must match metadata oracle"
            _validate_generated_output(inp, actual, rects)
            return inp
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid puzzle instance")
