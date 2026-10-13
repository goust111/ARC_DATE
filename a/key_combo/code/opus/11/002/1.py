from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 1
INNER = 2
INDICATOR_POOL = (3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    out[0, :] = True
    out[h - 1, :] = True
    out[:, 0] = True
    out[:, w - 1] = True
    return out


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_box = _bbox(g == FRAME)
    inner_box = _bbox(g == INNER)
    assert frame_box is not None, "missing frame"
    assert inner_box is not None, "missing inner rectangle"

    r0, c0, r1, c1 = frame_box
    ir0, ic0, ir1, ic1 = inner_box
    assert r0 < ir0 <= ir1 < r1, "inner rectangle must be strictly inside the frame bbox vertically"
    assert c0 < ic0 <= ic1 < c1, "inner rectangle must be strictly inside the frame bbox horizontally"

    frame_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(frame_patch.shape)
    assert np.all(frame_patch[border] == FRAME), "frame border must be complete"
    assert not np.any((g == FRAME) & ~_mask_from_box(g.shape, frame_box, border_only=True)), "frame color must appear only on the border"

    inner_patch = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    assert np.all(inner_patch == INNER), "inner rectangle must be solid"

    indicator_colors = sorted(int(color) for color in np.unique(g) if color not in (BG, FRAME, INNER))
    assert 1 <= len(indicator_colors) <= 3, "expected 1 to 3 indicator pixels"

    indicators = []
    seen_sides = set()
    for color in indicator_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), "each indicator color must occupy exactly one cell"
        r, c = map(int, coords[0])
        side_hits = []
        if r == r0 - 1 and c0 <= c <= c1 and g[r + 1, c] == FRAME:
            side_hits.append("top")
        if r == r1 + 1 and c0 <= c <= c1 and g[r - 1, c] == FRAME:
            side_hits.append("bottom")
        if c == c0 - 1 and r0 <= r <= r1 and g[r, c + 1] == FRAME:
            side_hits.append("left")
        if c == c1 + 1 and r0 <= r <= r1 and g[r, c - 1] == FRAME:
            side_hits.append("right")
        assert len(side_hits) == 1, "each indicator must be 4-neighbor attached to exactly one frame side"
        side = side_hits[0]
        assert side not in seen_sides, "at most one indicator per side"
        seen_sides.add(side)
        indicators.append({"color": color, "pos": (r, c), "side": side})

    structure_mask = (g == FRAME) | (g == INNER)
    structure_coords = [tuple(map(int, rc)) for rc in np.argwhere(structure_mask)]
    return {
        "grid": g,
        "frame_box": frame_box,
        "inner_box": inner_box,
        "inner_h": int(ir1 - ir0 + 1),
        "inner_w": int(ic1 - ic0 + 1),
        "indicators": sorted(indicators, key=lambda item: item["pos"]),
        "structure_coords": structure_coords,
    }


def _mask_from_box(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    *,
    border_only: bool = False,
) -> np.ndarray:
    r0, c0, r1, c1 = box
    out = np.zeros(shape, dtype=bool)
    if border_only:
        out[r0, c0 : c1 + 1] = True
        out[r1, c0 : c1 + 1] = True
        out[r0 : r1 + 1, c0] = True
        out[r0 : r1 + 1, c1] = True
    else:
        out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _mirror_cell(side: str, box: tuple[int, int, int, int], r: int, c: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return int(2 * r0 - 1 - r), int(c)
    if side == "bottom":
        return int(2 * r1 + 1 - r), int(c)
    if side == "left":
        return int(r), int(2 * c0 - 1 - c)
    if side == "right":
        return int(r), int(2 * c1 + 1 - c)
    raise ValueError(f"unknown side: {side}")


def _visible_mirror_cells(parsed: dict, side: str) -> list[tuple[int, int]]:
    g = parsed["grid"]
    cells = []
    for r, c in parsed["structure_coords"]:
        rr, cc = _mirror_cell(side, parsed["frame_box"], r, c)
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the framed rectangle outward from each indicator and let those copies clip a red concentric expansion.

    Input structure:
    - The background is black.
    - A blue 1-pixel rectangular frame surrounds a solid red rectangle that is strictly inside the frame bbox.
    - One to three single-pixel indicators of distinct nonzero colors sit 4-neighbor outside distinct frame sides.

    Rule:
    - For each indicator, identify the touched frame side and mirror the original blue-frame-plus-red-inner structure across that side's outside edge so the copy lands outside the frame.
    - Recolor every mirrored structure pixel to the indicator's color and draw multiple copies in raster-scan order, keeping the original structure and indicators unchanged.
    - Expand the original frame bbox by the inner rectangle height vertically and the inner rectangle width horizontally, clipped to the grid, and fill only background cells in that expansion ring with red.
    - Mirrored copies act as blockers: any expansion cell already occupied by a mirrored copy is not filled.

    Key insight:
    - The indicator pixels do double duty: they choose a reflection axis and simultaneously create colored blocker copies that must be drawn before evaluating which expansion cells remain fillable.
    """
    parsed = _parse_grid(grid)
    g = parsed["grid"]
    out = g.copy()

    for indicator in parsed["indicators"]:
        color = int(indicator["color"])
        for rr, cc in _visible_mirror_cells(parsed, indicator["side"]):
            out[rr, cc] = color

    structure_mask = (g == FRAME) | (g == INNER)
    out[structure_mask] = g[structure_mask]

    r0, c0, r1, c1 = parsed["frame_box"]
    inner_h = parsed["inner_h"]
    inner_w = parsed["inner_w"]
    er0 = max(0, r0 - inner_h)
    ec0 = max(0, c0 - inner_w)
    er1 = min(g.shape[0] - 1, r1 + inner_h)
    ec1 = min(g.shape[1] - 1, c1 + inner_w)

    expansion_mask = np.zeros(g.shape, dtype=bool)
    expansion_mask[er0 : er1 + 1, ec0 : ec1 + 1] = True
    expansion_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    out[expansion_mask & (out == BG)] = INNER
    return out


def _make_case() -> np.ndarray:
    inner_h = random.randint(2, 5)
    inner_w = random.randint(2, 5)
    top_gap = random.randint(1, 3)
    bottom_gap = random.randint(1, 3)
    left_gap = random.randint(1, 3)
    right_gap = random.randint(1, 3)
    outer_h = inner_h + top_gap + bottom_gap
    outer_w = inner_w + left_gap + right_gap
    max_inner = max(inner_h, inner_w)

    indicator_count = random.choices([1, 2, 3], weights=[1, 4, 4], k=1)[0]
    sides = random.sample(SIDES, indicator_count)
    safe_side = random.choice(sides)

    margins = {}
    for side in SIDES:
        base_lo = max_inner + 1
        if side in ("top", "bottom"):
            full_need = outer_h
        else:
            full_need = outer_w

        if side == safe_side:
            lo = full_need
            hi = full_need + 3
        elif side in sides:
            lo = base_lo
            hi = min(full_need - 1, base_lo + 3)
            if hi < lo:
                hi = lo
        else:
            lo = base_lo
            hi = base_lo + 4
        margins[side] = random.randint(lo, hi)

    H = margins["top"] + outer_h + margins["bottom"]
    W = margins["left"] + outer_w + margins["right"]
    if H > 30 or W > 30:
        raise ValueError("sampled grid exceeds ARC size limit")

    r0 = margins["top"]
    c0 = margins["left"]
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    ir0 = r0 + top_gap
    ic0 = c0 + left_gap
    ir1 = ir0 + inner_h - 1
    ic1 = ic0 + inner_w - 1

    grid = np.zeros((H, W), dtype=int)
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = INNER

    indicator_colors = random.sample(list(INDICATOR_POOL), indicator_count)
    for side, color in zip(sides, indicator_colors):
        if side == "top":
            pos = (r0 - 1, random.randint(c0, c1))
        elif side == "bottom":
            pos = (r1 + 1, random.randint(c0, c1))
        elif side == "left":
            pos = (random.randint(r0, r1), c0 - 1)
        else:
            pos = (random.randint(r0, r1), c1 + 1)
        grid[pos] = int(color)

    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_grid(grid)
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    mirror_colors = [indicator["color"] for indicator in parsed["indicators"]]
    for color in mirror_colors:
        if int((out == color).sum()) <= 1:
            return False

    r0, c0, r1, c1 = parsed["frame_box"]
    er0 = max(0, r0 - parsed["inner_h"])
    ec0 = max(0, c0 - parsed["inner_w"])
    er1 = min(grid.shape[0] - 1, r1 + parsed["inner_h"])
    ec1 = min(grid.shape[1] - 1, c1 + parsed["inner_w"])
    expansion_mask = np.zeros(grid.shape, dtype=bool)
    expansion_mask[er0 : er1 + 1, ec0 : ec1 + 1] = True
    expansion_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    blocker_count = int((expansion_mask & np.isin(out, mirror_colors)).sum())
    fill_count = int((expansion_mask & (out == INNER)).sum())
    if blocker_count < 2 or fill_count < 4:
        return False

    copied_any = False
    copied_inside_ring = False
    for indicator in parsed["indicators"]:
        visible = _visible_mirror_cells(parsed, indicator["side"])
        if visible:
            copied_any = True
        if any(expansion_mask[r, c] for r, c in visible):
            copied_inside_ring = True
    if not copied_any or not copied_inside_ring:
        return False

    if len(parsed["indicators"]) >= 2:
        directions = {indicator["side"] for indicator in parsed["indicators"]}
        if directions <= {"top", "bottom"} or directions <= {"left", "right"}:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _make_case()
        except ValueError:
            continue
        if _semantic_ok(grid):
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
