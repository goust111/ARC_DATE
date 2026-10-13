from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GRAY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300
PLACEMENT_RETRIES = 160


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_from_start(mask: np.ndarray, start: tuple[int, int], connectivity: int = 4) -> list[tuple[int, int]]:
    h, w = mask.shape
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    seen = {start}
    q = deque([start])
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return comp


def _mode_non_bg(grid: np.ndarray) -> int | None:
    vals = [int(v) for v in grid.ravel() if int(v) != BG]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_ring(border: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(border)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False, (r0, c0, r1, c1)
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(border) == expected, (r0, c0, r1, c1)


def _analyze_regions(grid: np.ndarray) -> tuple[int | None, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gray = _mode_non_bg(g)
    if gray is None:
        return None, []

    regions: list[dict[str, object]] = []
    for interior in _components(g == BG, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in interior):
            continue

        border_seeds = set()
        for r, c in interior:
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != BG:
                    border_seeds.add((nr, nc))
        if not border_seeds:
            continue

        border = _component_from_start(g != BG, next(iter(border_seeds)), connectivity=4)
        marker_cells = [(r, c) for r, c in border if int(g[r, c]) != int(gray)]
        assert len(marker_cells) == 1, "each border must contain exactly one colored marker"
        marker = marker_cells[0]
        is_rect, box = _is_rectangular_ring(border)
        regions.append(
            {
                "interior": interior,
                "border": border,
                "marker": marker,
                "marker_color": int(g[marker]),
                "is_rectangle": is_rect,
                "bbox": box,
            }
        )
    return gray, regions


def _checker_anchor(marker: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[int, int]:
    mr, mc = marker
    r0, c0, r1, c1 = box
    ar, ac = mr, mc
    if mr == r0:
        ar += 1
    elif mr == r1:
        ar -= 1
    if mc == c0:
        ac += 1
    elif mc == c1:
        ac -= 1
    return ar, ac


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black region according to whether its enclosing border is rectangular.

    Input structure:
    - The background is black.
    - Several closed regions are enclosed by gray one-pixel borders, and each border contains exactly one non-gray marker pixel.
    - Some borders are perfect rectangular rings, while others are irregular closed outlines.

    Rule:
    - Find every black region that is enclosed by a border and recover the marker color from that border.
    - If the border is exactly a rectangular ring, fill the region with a checkerboard of marker color and black, phased so the inward-adjacent cell from the marker starts in the marker color.
    - If the border is not rectangular, flood-fill the entire enclosed black region with the marker color.
    - Leave the gray border pixels and the marker pixels unchanged.

    Key insight:
    - The same marker serves two roles: it identifies the region's fill color, and the border geometry decides whether the interior uses a checkerboard or a uniform flood fill.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _gray, regions = _analyze_regions(g)

    for region in regions:
        interior = region["interior"]
        marker = region["marker"]
        marker_color = int(region["marker_color"])
        if bool(region["is_rectangle"]):
            anchor_r, anchor_c = _checker_anchor(marker, region["bbox"])
            for r, c in interior:
                out[r, c] = marker_color if (r - anchor_r + c - anchor_c) % 2 == 0 else BG
        else:
            for r, c in interior:
                out[r, c] = marker_color
    return out


def _region_boundary(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not region[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not region[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _make_rect_patch(marker_color: int, rng: random.Random) -> np.ndarray:
    inner_h = rng.randint(2, 6)
    inner_w = rng.randint(2, 6)
    h, w = inner_h + 2, inner_w + 2
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = GRAY
    patch[-1, :] = GRAY
    patch[:, 0] = GRAY
    patch[:, -1] = GRAY
    marker = rng.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
    patch[marker] = marker_color
    return patch


def _make_irregular_patch(marker_color: int, rng: random.Random) -> np.ndarray:
    base_h = rng.randint(5, 8)
    base_w = rng.randint(5, 8)
    sides = ["top", "bottom", "left", "right"]
    n_protrusions = 1 + int(rng.random() < 0.45)
    chosen_sides = rng.sample(sides, n_protrusions)

    ext = {side: 0 for side in sides}
    specs: list[tuple[str, int, int, int]] = []
    for side in chosen_sides:
        outward = rng.randint(1, 3)
        ext[side] = outward
        if side in ("top", "bottom"):
            span = rng.randint(2, base_w - 1)
            start = rng.randint(0, base_w - span)
        else:
            span = rng.randint(2, base_h - 1)
            start = rng.randint(0, base_h - span)
        specs.append((side, outward, start, span))

    h = base_h + ext["top"] + ext["bottom"]
    w = base_w + ext["left"] + ext["right"]
    region = np.zeros((h, w), dtype=bool)
    br0 = ext["top"]
    bc0 = ext["left"]
    region[br0 : br0 + base_h, bc0 : bc0 + base_w] = True

    for side, outward, start, span in specs:
        if side == "top":
            region[0:outward, bc0 + start : bc0 + start + span] = True
        elif side == "bottom":
            region[br0 + base_h : br0 + base_h + outward, bc0 + start : bc0 + start + span] = True
        elif side == "left":
            region[br0 + start : br0 + start + span, 0:outward] = True
        else:
            region[br0 + start : br0 + start + span, bc0 + base_w : bc0 + base_w + outward] = True

    boundary = _region_boundary(region)
    interior = region & ~boundary
    assert interior.any(), "irregular region must have an interior"

    patch = np.zeros((h, w), dtype=int)
    patch[boundary] = GRAY
    coords = list(map(tuple, np.argwhere(boundary)))
    patch[rng.choice(coords)] = marker_color
    return patch


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randrange(1, h - ph)
        c0 = rng.randrange(1, w - pw)
        rr0 = r0 - 1
        cc0 = c0 - 1
        rr1 = r0 + ph + 1
        cc1 = c0 + pw + 1
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != BG, patch, grid[r0 : r0 + ph, c0 : c0 + pw])
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice([2, 3, 4])
    kinds = ["rect", "irregular"] + [rng.choice(["rect", "irregular"]) for _ in range(count - 2)]
    rng.shuffle(kinds)
    colors = rng.sample(MARKER_COLORS, count)

    patches: list[np.ndarray] = []
    for kind, color in zip(kinds, colors):
        if kind == "rect":
            patches.append(_make_rect_patch(color, rng))
        else:
            patches.append(_make_irregular_patch(color, rng))

    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    min_h = max(max_h + 3, 12 + 4 * count)
    min_w = max(max_w + 3, 12 + 4 * count)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(count), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        if not _place_patch(grid, reserved, patches[idx], rng):
            return None

    try:
        _gray, regions = _analyze_regions(grid)
    except AssertionError:
        return None

    if len(regions) != count:
        return None
    rect_count = sum(1 for region in regions if bool(region["is_rectangle"]))
    if rect_count == 0 or rect_count == len(regions):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after retries")
