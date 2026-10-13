from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
STAMP_SIZE = 3
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((int(cr), int(cc)))
            for dr, dc in DIRS4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _scaled_stamp(mask: np.ndarray) -> np.ndarray:
    """Downscale a bbox-normalized mask by binning source pixel centers into a 3x3 grid."""
    h, w = mask.shape
    out = np.zeros((STAMP_SIZE, STAMP_SIZE), dtype=bool)
    for r, c in np.argwhere(mask):
        tr = min(STAMP_SIZE - 1, int((float(r) + 0.5) * STAMP_SIZE / h))
        tc = min(STAMP_SIZE - 1, int((float(c) + 0.5) * STAMP_SIZE / w))
        out[tr, tc] = True
    assert bool(np.any(out)), "scaled stamp must contain at least one pixel"
    return out


def _paint_stamp(out: np.ndarray, center: tuple[int, int], stamp: np.ndarray, color: int, bg: int) -> None:
    cr, cc = center
    for dr in range(STAMP_SIZE):
        for dc in range(STAMP_SIZE):
            if not bool(stamp[dr, dc]):
                continue
            rr = int(cr + dr - 1)
            cc2 = int(cc + dc - 1)
            if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                assert int(out[rr, cc2]) in {bg, color}, "stamps of different colors must not overlap"
                out[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each monochrome template and stamp its 3x3 normalized miniature at same-color singleton markers.

    Input structure:
    - The grid has a background color plus one or more foreground colors.
    - For each foreground color there is exactly one larger 4-connected template component and
      one or more isolated single-pixel markers of that same color.

    Rule:
    - Convert each template component into a tight bbox boolean mask.
    - Downscale that mask to a 3x3 stamp.
    - Erase the original template.
    - For every same-color singleton marker, draw the 3x3 stamp centered on that marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    out = np.full_like(g, bg)

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _components_of_color(g, color)
        templates = [comp for comp in comps if len(comp) > 1]
        markers = [comp[0] for comp in comps if len(comp) == 1]
        assert len(templates) == 1, "each foreground color must have exactly one template component"
        assert markers, "each foreground color must have at least one singleton marker"

        stamp = _scaled_stamp(_component_mask(templates[0]))
        for marker in markers:
            _paint_stamp(out, marker, stamp, color, bg)

    return out


def _neighbors4_cell(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_path(start: tuple[int, int], goal: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = start
    gr, gc = goal
    cells = {(r, c)}
    while (r, c) != (gr, gc):
        options: list[tuple[int, int]] = []
        if r < gr:
            options.append((r + 1, c))
        elif r > gr:
            options.append((r - 1, c))
        if c < gc:
            options.append((r, c + 1))
        elif c > gc:
            options.append((r, c - 1))
        r, c = random.choice(options)
        cells.add((r, c))
    return cells


def _blob_mask(h: int, w: int) -> np.ndarray:
    anchor = (random.randint(0, h - 1), random.randint(0, w - 1))
    goals = [
        (0, random.randint(0, w - 1)),
        (h - 1, random.randint(0, w - 1)),
        (random.randint(0, h - 1), 0),
        (random.randint(0, h - 1), w - 1),
    ]
    cells = {anchor}
    for goal in goals:
        cells |= _random_path(anchor, goal)

    total = h * w
    lo = max(len(cells), int(np.ceil(total * 0.35)))
    hi = max(lo, min(total - 1, int(np.floor(total * 0.72))))
    target = random.randint(lo, hi)

    frontier = {
        nb
        for cell in cells
        for nb in _neighbors4_cell(cell, h, w)
        if nb not in cells
    }
    while len(cells) < target and frontier:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        cells.add(pick)
        for nb in _neighbors4_cell(pick, h, w):
            if nb not in cells:
                frontier.add(nb)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    for _ in range(random.randint(0, 2)):
        if random.choice((False, True)) and h >= 5:
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - 2)
            c1 = random.randint(c0, w - 2)
            mask[r, c0 : c1 + 1] = True
        elif w >= 5:
            c = random.randint(1, w - 2)
            r0 = random.randint(1, h - 2)
            r1 = random.randint(r0, h - 2)
            mask[r0 : r1 + 1, c] = True
    return mask


def _cross_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    rr = random.randint(0, h - 1)
    cc = random.randint(0, w - 1)
    mask[rr, :] = True
    mask[:, cc] = True
    if h >= 5 and random.choice((False, True)):
        extra_r = min(h - 1, max(0, rr + random.choice((-1, 1))))
        mask[extra_r, :] = True
    if w >= 5 and random.choice((False, True)):
        extra_c = min(w - 1, max(0, cc + random.choice((-1, 1))))
        mask[:, extra_c] = True
    return mask


def _random_template_mask() -> np.ndarray:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    family = random.choice(("blob", "frame", "cross"))
    if family == "blob":
        return _blob_mask(h, w)
    if family == "frame":
        return _frame_mask(h, w)
    return _cross_mask(h, w)


def _reserve_box(occ: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(occ.shape[0], r0 + h + margin)
    cc1 = min(occ.shape[1], c0 + w + margin)
    occ[rr0:rr1, cc0:cc1] = True


def _free_template_positions(occ: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    H, W = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + h + 1
            cc1 = c0 + w + 1
            if not bool(np.any(occ[rr0:rr1, cc0:cc1])):
                positions.append((r0, c0))
    return positions


def _free_marker_centers(occ: np.ndarray) -> list[tuple[int, int]]:
    H, W = occ.shape
    centers: list[tuple[int, int]] = []
    for r in range(2, H - 2):
        for c in range(2, W - 2):
            if not bool(np.any(occ[r - 2 : r + 3, c - 2 : c + 3])):
                centers.append((r, c))
    return centers


def _render_input(
    shape: tuple[int, int],
    bg: int,
    templates: list[tuple[int, np.ndarray, tuple[int, int]]],
    markers: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, mask, (r0, c0) in templates:
        h, w = mask.shape
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[mask] = color
    for color, centers in markers.items():
        for r, c in centers:
            grid[r, c] = color
    return grid


def _expected_output(shape: tuple[int, int], bg: int, stamps: dict[int, np.ndarray], markers: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color, centers in markers.items():
        for center in centers:
            for dr in range(STAMP_SIZE):
                for dc in range(STAMP_SIZE):
                    if stamps[color][dr, dc]:
                        rr = int(center[0] + dr - 1)
                        cc = int(center[1] + dc - 1)
                        out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(1, 3)
    marker_counts = [random.randint(1, 2) for _ in range(n_colors)]
    if random.choice((False, True)):
        marker_counts[random.randrange(n_colors)] += 1
    total_markers = sum(marker_counts)

    base = 12 + 2 * n_colors + 2 * total_markers
    low = max(15, base - 3)
    high = min(30, base + 4)
    if low > high:
        return None
    H = random.randint(low, high)
    W = random.randint(low, high)

    bg = random.randrange(10)
    colors = random.sample([c for c in range(10) if c != bg], n_colors)

    occ = np.zeros((H, W), dtype=bool)
    templates_raw = [(color, _random_template_mask(), count) for color, count in zip(colors, marker_counts, strict=False)]
    templates_raw.sort(key=lambda item: int(np.sum(item[1])), reverse=True)

    templates: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    stamps: dict[int, np.ndarray] = {}
    for color, mask, _count in templates_raw:
        h, w = mask.shape
        positions = _free_template_positions(occ, h, w)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        templates.append((color, mask, (r0, c0)))
        stamps[color] = _scaled_stamp(mask)
        _reserve_box(occ, r0, c0, h, w, margin=1)

    markers: dict[int, list[tuple[int, int]]] = {color: [] for color in colors}
    counts_by_color = dict(zip(colors, marker_counts, strict=False))
    for color in colors:
        for _ in range(counts_by_color[color]):
            centers = _free_marker_centers(occ)
            if not centers:
                return None
            center = random.choice(centers)
            markers[color].append(center)
            _reserve_box(occ, center[0], center[1], 1, 1, margin=2)

    inp = _render_input((H, W), bg, templates, markers)
    if _background_color(inp) != bg:
        return None

    expected = _expected_output((H, W), bg, stamps, markers)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stamp-miniature ARC inputs")
