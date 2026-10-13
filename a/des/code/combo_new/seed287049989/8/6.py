from __future__ import annotations

import random

import numpy as np


BG = 0
CENTER_COLOR = 5
RING_COUNT = 3
OFFSET_OPTIONS = (-1, 0, 1)
SIDES = ("top", "bottom", "left", "right")
VERTICAL_SIDES = {"top", "bottom"}
HORIZONTAL_SIDES = {"left", "right"}
PALETTE = tuple(range(10))
MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _find_center_block(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if int(coords.shape[0]) != 9:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((color, (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected exactly one solid 3x3 center block"
    return candidates[0]


def _parse_ring(
    grid: np.ndarray, color: int, center_r: int, center_c: int
) -> dict[str, int | tuple[int, int, int, int] | tuple[int, int]]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "ring color must be present"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "ring bbox must have an integer center"
    assert (r0 + r1) // 2 == center_r and (c0 + c1) // 2 == center_c, "rings must be concentric"

    perimeter = set(_perimeter_cells(r0, c0, r1, c1))
    actual = {tuple(map(int, rc)) for rc in coords}
    assert actual <= perimeter, "ring color must occupy only its rectangle perimeter"

    missing = list(perimeter - actual)
    assert len(missing) == 1, "each ring must have exactly one 1-pixel gap"
    assert len(actual) == len(perimeter) - 1, "ring perimeter must be complete except for the gap"
    gap_r, gap_c = missing[0]

    if gap_r == r0 and c0 < gap_c < c1:
        side = "top"
        offset = gap_c - center_c
    elif gap_r == r1 and c0 < gap_c < c1:
        side = "bottom"
        offset = gap_c - center_c
    elif gap_c == c0 and r0 < gap_r < r1:
        side = "left"
        offset = gap_r - center_r
    elif gap_c == c1 and r0 < gap_r < r1:
        side = "right"
        offset = gap_r - center_r
    else:
        raise AssertionError("gap must lie on a non-corner perimeter cell")

    assert offset in OFFSET_OPTIONS, "gap must stay near the side midpoint"
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap": (int(gap_r), int(gap_c)),
        "side": side,
        "offset": int(offset),
    }


def _line_hits_gap(ray_side: str, ray_offset: int, gap_side: str, gap_offset: int) -> bool:
    if ray_side in VERTICAL_SIDES:
        return gap_side in VERTICAL_SIDES and ray_offset == gap_offset
    return gap_side in HORIZONTAL_SIDES and ray_offset == gap_offset


def _center_hit_token(side: str, offset: int) -> tuple[int, int]:
    idx = offset + 1
    assert 0 <= idx < 3
    if side == "top":
        return (0, idx)
    if side == "bottom":
        return (2, idx)
    if side == "left":
        return (idx, 0)
    if side == "right":
        return (idx, 2)
    raise AssertionError("unknown side")


def _center_hit_cell(center_bbox: tuple[int, int, int, int], side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = center_bbox
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    if side == "top":
        return (r0, cc + offset)
    if side == "bottom":
        return (r1, cc + offset)
    if side == "left":
        return (cr + offset, c0)
    if side == "right":
        return (cr + offset, c1)
    raise AssertionError("unknown side")


def _trace_rays(
    rings: list[dict[str, int | tuple[int, int, int, int] | tuple[int, int]]],
    center_bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], list[str]]:
    hits: list[tuple[int, int]] = []
    events: list[str] = []
    for i, ring in enumerate(rings):
        side = str(ring["side"])
        offset = int(ring["offset"])
        for inner in rings[i + 1 :]:
            inner_side = str(inner["side"])
            inner_offset = int(inner["offset"])
            if _line_hits_gap(side, offset, inner_side, inner_offset):
                events.append("pass")
            else:
                events.append("turn")
                side = inner_side
                offset = inner_offset
        hits.append(_center_hit_cell(center_bbox, side, offset))
    return hits, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one inward ray from each gapped concentric rectangle ring and color the hit cells on the 3x3 core.

    Input structure:
    - Black is the background. Three differently colored 1-pixel-thick rectangular rings share one center and surround a solid 3x3 core block.
    - Each ring has exactly one 1-pixel gap on one of its four sides, with the gap placed near that side's midpoint.
    - The 3x3 core is a single color distinct from the three ring colors and remains in place.

    Rule:
    - Process the rings from outermost to innermost. Each ring emits one ray from its own gap toward the center, using that side's inward normal direction.
    - When the ray reaches a deeper ring, check whether that ring's gap lies on the current straight ray line. If it does, the ray continues straight through. If not, the deeper ring blocks the ray and redirects it so that the ray now heads inward from that deeper ring's own gap.
    - After all deeper rings are handled, mark the reached cell on the 3x3 core border with the emitting ring's color. Keep every ring pixel and every gap unchanged.

    Key insight:
    - A ring gap serves two roles: it launches that ring's own ray, and it also acts as the redirection portal that outer rays must use whenever they cannot pass straight through the current line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    center_color, center_bbox = _find_center_block(g, bg)
    r0, c0, r1, c1 = center_bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    ring_colors = [int(color) for color in np.unique(g) if int(color) not in (bg, center_color)]
    assert len(ring_colors) == RING_COUNT, "expected exactly three ring colors"

    rings = [_parse_ring(g, color, center_r, center_c) for color in ring_colors]
    rings.sort(
        key=lambda ring: (
            int(ring["bbox"][2]) - int(ring["bbox"][0]) + 1,
            int(ring["bbox"][3]) - int(ring["bbox"][1]) + 1,
        ),
        reverse=True,
    )

    out = g.copy()
    hits, _events = _trace_rays(rings, center_bbox)
    for ring, hit in zip(rings, hits, strict=False):
        hr, hc = hit
        assert r0 <= hr <= r1 and c0 <= hc <= c1, "ray must terminate on the 3x3 core"
        out[hr, hc] = int(ring["color"])
    return out


def _balanced_split(total_extra: int) -> tuple[int, int]:
    assert total_extra >= 2
    mid = total_extra // 2
    lo = max(1, mid - 1)
    hi = min(total_extra - 1, mid + 1)
    first = random.randint(lo, hi)
    return first, total_extra - first


def _gap_cell(center_r: int, center_c: int, half_h: int, half_w: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return (center_r - half_h, center_c + offset)
    if side == "bottom":
        return (center_r + half_h, center_c + offset)
    if side == "left":
        return (center_r + offset, center_c - half_w)
    if side == "right":
        return (center_r + offset, center_c + half_w)
    raise AssertionError("unknown side")


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = BG


def _simulate_layout(layout: tuple[tuple[str, int], ...]) -> tuple[list[tuple[int, int]], list[str]]:
    rings = [{"side": side, "offset": offset} for side, offset in layout]
    hits: list[tuple[int, int]] = []
    events: list[str] = []
    for i, ring in enumerate(rings):
        side = str(ring["side"])
        offset = int(ring["offset"])
        for inner in rings[i + 1 :]:
            inner_side = str(inner["side"])
            inner_offset = int(inner["offset"])
            if _line_hits_gap(side, offset, inner_side, inner_offset):
                events.append("pass")
            else:
                events.append("turn")
                side = inner_side
                offset = inner_offset
        hits.append(_center_hit_token(side, offset))
    return hits, events


def _build_valid_layouts() -> tuple[tuple[tuple[str, int], ...], ...]:
    states = [(side, offset) for side in SIDES for offset in OFFSET_OPTIONS]
    layouts: list[tuple[tuple[str, int], ...]] = []
    for outer in states:
        for middle in states:
            for inner in states:
                layout = (outer, middle, inner)
                all_vertical = all(side in VERTICAL_SIDES and offset == outer[1] for side, offset in layout)
                all_horizontal = all(side in HORIZONTAL_SIDES and offset == outer[1] for side, offset in layout)
                if all_vertical or all_horizontal:
                    continue
                hits, events = _simulate_layout(layout)
                if "pass" not in events or "turn" not in events:
                    continue
                if len(set(hits)) < 2:
                    continue
                layouts.append(layout)
    assert layouts, "generator needs at least one non-degenerate gap layout"
    return tuple(layouts)


VALID_LAYOUTS = _build_valid_layouts()


def _gen_attempt() -> np.ndarray | None:
    inner_half_h = random.randint(2, 4)
    inner_half_w = random.randint(2, 4)
    middle_gap_h = random.randint(2, 4)
    middle_gap_w = random.randint(2, 4)
    outer_gap_h = random.randint(2, 4)
    outer_gap_w = random.randint(2, 4)

    inner_h = inner_half_h
    inner_w = inner_half_w
    middle_h = inner_h + middle_gap_h
    middle_w = inner_w + middle_gap_w
    outer_h = middle_h + outer_gap_h
    outer_w = middle_w + outer_gap_w

    base_h = 2 * outer_h + 1
    base_w = 2 * outer_w + 1
    if base_h > 28 or base_w > 28:
        return None

    extra_h = random.randint(2, 30 - base_h)
    extra_w = random.randint(2, 30 - base_w)
    top_margin, bottom_margin = _balanced_split(extra_h)
    left_margin, right_margin = _balanced_split(extra_w)

    H = base_h + extra_h
    W = base_w + extra_w
    center_r = top_margin + outer_h
    center_c = left_margin + outer_w

    layout = random.choice(VALID_LAYOUTS)
    ring_colors = random.sample([color for color in PALETTE if color not in (BG, CENTER_COLOR)], RING_COUNT)
    half_sizes = [(outer_h, outer_w), (middle_h, middle_w), (inner_h, inner_w)]

    grid = np.full((H, W), BG, dtype=int)
    grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = CENTER_COLOR

    for color, (half_h, half_w), (side, offset) in zip(ring_colors, half_sizes, layout, strict=False):
        bbox = (center_r - half_h, center_c - half_w, center_r + half_h, center_c + half_w)
        gap = _gap_cell(center_r, center_c, half_h, half_w, side, offset)
        _draw_ring(grid, bbox, int(color), gap)

    out = main(grid)
    center_patch = out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2]
    if len(set(map(tuple, np.argwhere(center_patch != CENTER_COLOR).tolist()))) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring ray puzzle")
