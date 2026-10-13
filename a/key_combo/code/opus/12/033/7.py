from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
VERTICAL_AXIS = 1
HORIZONTAL_AXIS = 2
TEMPLATE_COLOR = 9
CONTROL_COLORS = (3, 4, 5, 6, 7, 8)
MAX_ATTEMPTS = 500
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BASE_TEMPLATES = (
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
)


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _transform_offsets(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    rotate_k: int,
    reflect: bool,
) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    for _ in range(rotate_k % 4):
        h = max(r for r, _ in pts) + 1
        pts = [(c, h - 1 - r) for r, c in pts]
    if reflect:
        w = max(c for _, c in pts) + 1
        pts = [(r, w - 1 - c) for r, c in pts]
    return _normalize(pts)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _corner_info(grid: np.ndarray) -> tuple[str, tuple[int, int], int, int]:
    h, w = grid.shape
    candidates = {
        "tl": (0, 0, 0, 1, 1, 0),
        "tr": (0, w - 1, 0, w - 2, 1, w - 1),
        "bl": (h - 1, 0, h - 1, 1, h - 2, 0),
        "br": (h - 1, w - 1, h - 1, w - 2, h - 2, w - 1),
    }
    for corner, (cr, cc, hr, hc, vr, vc) in candidates.items():
        if int(grid[cr, cc]) != TEMPLATE_COLOR:
            continue
        hcol = int(grid[hr, hc])
        vcol = int(grid[vr, vc])
        if hcol in (BG, TEMPLATE_COLOR) or vcol in (BG, TEMPLATE_COLOR) or hcol == vcol:
            continue
        axis_row = 0 if cr == 0 else h - 1
        axis_col = 0 if cc == 0 else w - 1
        row_ok = np.all(
            (grid[axis_row, :] == hcol)
            | (np.arange(w) == axis_col)
        )
        col_ok = np.all(
            (grid[:, axis_col] == vcol)
            | (np.arange(h) == axis_row)
        )
        if row_ok and col_ok:
            return corner, (cr, cc), axis_row, axis_col
    raise AssertionError("failed to locate the axis corner")


def _extract_scene(grid: np.ndarray) -> tuple[tuple[tuple[int, int], ...], list[tuple[int, int, int]], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    _corner, corner, axis_row, axis_col = _corner_info(g)

    maroon_mask = g == TEMPLATE_COLOR
    maroon_mask[corner] = False
    template_comps = [comp for comp in _components(maroon_mask) if len(comp) > 1]
    assert len(template_comps) == 1, "expected one template component"
    template_offsets = _normalize(template_comps[0])

    interior_mask = g != BG
    interior_mask[axis_row, :] = False
    interior_mask[:, axis_col] = False
    for r, c in template_comps[0]:
        interior_mask[r, c] = False

    controls: list[tuple[int, int, int]] = []
    for comp in _components(interior_mask):
        assert len(comp) == 1, "controls must be isolated single pixels"
        r, c = comp[0]
        controls.append((r, c, int(g[r, c])))

    assert 3 <= len(controls) <= 5, "expected 3-5 controls"
    distances = [abs(r - corner[0]) + abs(c - corner[1]) for r, c, _ in controls]
    assert len(set(distances)) == len(distances), "control distances must be unique"
    return template_offsets, controls, corner


def _render_ordered(
    shape: tuple[int, int],
    template_offsets: tuple[tuple[int, int], ...],
    ordered_controls: list[tuple[int, int, int]],
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for r0, c0, color in ordered_controls:
        for dr, dc in template_offsets:
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored copies of the maroon template at each control pixel, with distance to the axis corner deciding which overlap stays visible.

    Input structure:
    - The grid has black background, one full horizontal edge line and one full vertical edge line meeting at a corner maroon pixel.
    - A single 4-connected maroon template object with area greater than one sits inside the grid away from the axes.
    - Three to five isolated non-axis control pixels of distinct non-maroon colors appear elsewhere in the interior.

    Rule:
    - Use the maroon corner pixel as the coordinate origin and measure each control pixel's Manhattan distance to it.
    - Record the template shape as offsets from the template bbox top-left, then place a recolored copy with that same top-left anchor at every control pixel.
    - Draw copies from farthest control to nearest control so nearer copies overwrite farther ones in overlaps, clip off-grid parts, and erase the original axes, corner, and template.

    Key insight:
    The edge lines disappear in the output, but they provide the only ordering signal: without recovering the corner and computing each control's distance to it, overlap colors cannot be explained.
    """
    template_offsets, controls, corner = _extract_scene(grid)
    ordered = sorted(
        controls,
        key=lambda item: abs(item[0] - corner[0]) + abs(item[1] - corner[1]),
        reverse=True,
    )
    return _render_ordered(np.asarray(grid).shape, template_offsets, ordered)


def _grid_transform(grid: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return grid
    if corner == "tr":
        return np.fliplr(grid)
    if corner == "bl":
        return np.flipud(grid)
    if corner == "br":
        return np.flipud(np.fliplr(grid))
    raise ValueError("unknown corner")


def _copy_cells(
    origin: tuple[int, int],
    template_offsets: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0 = origin
    cells = set()
    for dr, dc in template_offsets:
        rr, cc = r0 + dr, c0 + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
    return cells


def _distance_rank(points: list[tuple[int, int]], corner: tuple[int, int]) -> list[int]:
    return sorted(range(len(points)), key=lambda i: abs(points[i][0] - corner[0]) + abs(points[i][1] - corner[1]), reverse=True)


def _semantic_ok(
    shape: tuple[int, int],
    template_offsets: tuple[tuple[int, int], ...],
    controls: list[tuple[int, int, int]],
    corner: tuple[int, int],
    force_clip: bool,
) -> bool:
    points = [(r, c) for r, c, _ in controls]
    dists = [abs(r - corner[0]) + abs(c - corner[1]) for r, c, _ in controls]
    if len(set(dists)) != len(dists):
        return False

    if any(abs(r - corner[0]) <= 1 and abs(c - corner[1]) <= 1 for r, c in points):
        return False

    if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) < 2 for i, a in enumerate(points) for b in points[i + 1 :]):
        return False

    rank = _distance_rank(points, corner)
    by_color = sorted(range(len(controls)), key=lambda i: controls[i][2], reverse=True)
    by_row_major = sorted(range(len(controls)), key=lambda i: (controls[i][0], controls[i][1]), reverse=True)
    if rank == by_color or rank == by_row_major:
        return False

    cover_map: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    clipped = 0
    for idx, (r, c, _color) in enumerate(controls):
        cells = _copy_cells((r, c), template_offsets, shape)
        if not cells:
            return False
        if len(cells) < len(template_offsets):
            clipped += 1
        for cell in cells:
            cover_map[cell].append(idx)

    overlap_pixels = {cell: ids for cell, ids in cover_map.items() if len(ids) >= 2}
    if not overlap_pixels:
        return False

    far_to_near = _render_ordered(
        shape,
        template_offsets,
        [controls[i] for i in rank],
    )
    near_to_far = _render_ordered(
        shape,
        template_offsets,
        [controls[i] for i in reversed(rank)],
    )
    if np.array_equal(far_to_near, near_to_far):
        return False

    informative_overlap = False
    for ids in overlap_pixels.values():
        nearest = min(ids, key=lambda i: dists[i])
        farthest = max(ids, key=lambda i: dists[i])
        nr, nc = next(iter(_copy_cells((controls[nearest][0], controls[nearest][1]), template_offsets, shape) & _copy_cells((controls[farthest][0], controls[farthest][1]), template_offsets, shape)))
        if int(far_to_near[nr, nc]) == controls[nearest][2] and controls[nearest][2] != controls[farthest][2]:
            informative_overlap = True
            break
    if not informative_overlap:
        return False

    if force_clip and clipped == 0:
        return False
    return True


def _choose_template() -> tuple[tuple[tuple[int, int], ...], int, int]:
    base = random.choice(BASE_TEMPLATES)
    pts = _transform_offsets(base, random.randrange(4), random.random() < 0.5)
    _, _, r1, c1 = _bbox(list(pts))
    return pts, r1 + 1, c1 + 1


def _choose_controls(
    h: int,
    w: int,
    template_offsets: tuple[tuple[int, int], ...],
    template_box: tuple[int, int, int, int],
    count: int,
    force_clip: bool,
) -> list[tuple[int, int, int]] | None:
    tr0, tc0, tr1, tc1 = template_box
    th = tr1 - tr0 + 1
    tw = tc1 - tc0 + 1

    for _ in range(160):
        region_h = min(max(th + 2, 4), 6)
        region_w = min(max(tw + 2, 4), 6)
        min_r = max(tr1 + 3, 2)
        min_c = max(tc1 + 3, 2)
        if force_clip:
            anchor_r = random.randint(max(min_r, h - region_h - 1), h - 2)
            anchor_c = random.randint(max(min_c, w - region_w - 1), w - 2)
        else:
            max_anchor_r = max(min_r, h - region_h - th - 1)
            max_anchor_c = max(min_c, w - region_w - tw - 1)
            if max_anchor_r < min_r or max_anchor_c < min_c:
                continue
            anchor_r = random.randint(min_r, max_anchor_r)
            anchor_c = random.randint(min_c, max_anchor_c)

        candidates = [
            (anchor_r + dr, anchor_c + dc)
            for dr in range(region_h)
            for dc in range(region_w)
            if anchor_r + dr < h and anchor_c + dc < w
        ]
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            if any(abs(r - rr) + abs(c - cc) < 2 for rr, cc in chosen):
                continue
            if abs(r - tr0) <= 2 and abs(c - tc0) <= 2:
                continue
            chosen.append((r, c))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            continue

        colors = random.sample(CONTROL_COLORS, count)
        controls = [(r, c, color) for (r, c), color in zip(chosen, colors)]
        if _semantic_ok((h, w), template_offsets, controls, (0, 0), force_clip):
            return controls
    return None


def _build_canonical() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    count = random.randint(3, 5)
    force_clip = random.random() < 0.45

    template_offsets, th, tw = _choose_template()
    if h <= th + 8 or w <= tw + 8:
        return None
    temp_r = random.randint(2, 4)
    temp_c = random.randint(2, 4)
    template_box = (temp_r, temp_c, temp_r + th - 1, temp_c + tw - 1)

    controls = _choose_controls(h, w, template_offsets, template_box, count, force_clip)
    if controls is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = HORIZONTAL_AXIS
    grid[:, 0] = VERTICAL_AXIS
    grid[0, 0] = TEMPLATE_COLOR

    for dr, dc in template_offsets:
        grid[temp_r + dr, temp_c + dc] = TEMPLATE_COLOR
    for r, c, color in controls:
        grid[r, c] = color

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not _semantic_ok(grid.shape, template_offsets, controls, (0, 0), force_clip):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        canonical = _build_canonical()
        if canonical is None:
            continue
        corner = random.choice(("tl", "tr", "bl", "br"))
        grid = _grid_transform(canonical, corner)
        try:
            out = main(grid)
            parsed_template, parsed_controls, parsed_corner = _extract_scene(grid)
        except Exception:
            continue
        if not _semantic_ok(grid.shape, parsed_template, parsed_controls, parsed_corner, False):
            # Clip is already validated in canonical space; after flips we only need non-degenerate overlaps.
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
