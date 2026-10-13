from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIR_ORDER = ("up", "right", "down", "left")
DIR_STEPS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _outline_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _square_outline_coords(top: int, left: int, size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    bottom = top + size - 1
    right = left + size - 1
    for c in range(left, right + 1):
        coords.append((top, c))
        if bottom != top:
            coords.append((bottom, c))
    for r in range(top + 1, bottom):
        coords.append((r, left))
        if right != left:
            coords.append((r, right))
    return coords


def _stamp_bbox_from_grey(grey_r: int, grey_c: int, size: int, direction: str) -> tuple[int, int, int, int]:
    if direction == "up":
        top = grey_r - size - 1
        left = grey_c - size // 2
    elif direction == "down":
        top = grey_r + 2
        left = grey_c - size // 2
    elif direction == "left":
        top = grey_r - size // 2
        left = grey_c - size - 1
    else:
        top = grey_r - size // 2
        left = grey_c + 2
    return top, left, top + size - 1, left + size - 1


def _stamp_region_in_bounds(
    h: int, w: int, top: int, left: int, bottom: int, right: int
) -> list[tuple[int, int]]:
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h - 1, bottom)
    c1 = min(w - 1, right)
    if r0 > r1 or c0 > c1:
        return []
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = _outline_mask(h)
        if border.shape != (h, h) and h != w:
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
        else:
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
        if len(comp) != int(border.sum()):
            continue
        if np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1] == MAROON, border):
            candidates.append((h * w, (r0, c0, r1, c1)))
    assert candidates, "expected a maroon legend frame"
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _legend_sizes(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> set[int]:
    r0, c0, r1, c1 = legend_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    sizes: set[int] = set()
    for comp in _components(interior == GREY):
        rr0, cc0, rr1, cc1 = _bbox(comp)
        h = rr1 - rr0 + 1
        w = cc1 - cc0 + 1
        if h != w or h < 3:
            continue
        border = _outline_mask(h)
        if len(comp) != int(border.sum()):
            continue
        if np.array_equal(interior[rr0 : rr1 + 1, cc0 : cc1 + 1] == GREY, border):
            sizes.add(h)
    assert sizes, "legend must contain grey outline samples"
    return sizes


def _direction_from_grey(coords: list[tuple[int, int]], grey: tuple[int, int]) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    gr, gc = grey
    choices: list[str] = []
    if gr == r0:
        choices.append("up")
    if gc == c1:
        choices.append("right")
    if gr == r1:
        choices.append("down")
    if gc == c0:
        choices.append("left")
    if len(choices) == 1:
        return choices[0]

    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = gr - center_r
    dc = gc - center_c
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each colored shape's pixels, match that count to a legend square size,
    then stamp a same-sized hollow square one cell away from the grey arrow cell
    in the arrow's outward direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    legend_bbox = _find_legend_bbox(g)
    sizes = _legend_sizes(g, legend_bbox)

    outside_legend = np.ones((h, w), dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    outside_legend[lr0 : lr1 + 1, lc0 : lc1 + 1] = False
    object_mask = (g != BG) & outside_legend

    out = g.copy()
    for comp in _components(object_mask):
        values = {int(g[r, c]) for r, c in comp}
        assert GREY in values, "each emitted shape must contain a grey marker"
        body_colors = values - {GREY}
        assert len(body_colors) == 1, "each emitted shape must have exactly one body color"
        color = next(iter(body_colors))

        grey_cells = [(r, c) for r, c in comp if int(g[r, c]) == GREY]
        assert len(grey_cells) == 1, "each shape must contain exactly one grey pixel"
        grey = grey_cells[0]

        size = len(comp)
        assert size in sizes, "shape area must match one legend stamp size"
        direction = _direction_from_grey(comp, grey)
        top, left, bottom, right = _stamp_bbox_from_grey(grey[0], grey[1], size, direction)

        for r, c in _square_outline_coords(top, left, size):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = int(color)
    return out


def _make_legend_patch(rng: np.random.Generator) -> np.ndarray:
    top_row = [3, 4, 5]
    bottom_row = [6, 7]
    rng.shuffle(top_row)
    rng.shuffle(bottom_row)
    rows = [top_row, bottom_row]
    if bool(rng.integers(0, 2)):
        rows.reverse()

    row_heights = [max(row) for row in rows]
    inner_h = row_heights[0] + 1 + row_heights[1]
    inner_w = max(sum(row) + len(row) - 1 for row in rows)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = MAROON
    patch[-1, :] = MAROON
    patch[:, 0] = MAROON
    patch[:, -1] = MAROON

    base_r = 1
    for row_sizes, row_h in zip(rows, row_heights, strict=True):
        c = 1
        align_bottom = bool(rng.integers(0, 2))
        for size in row_sizes:
            top = base_r + (row_h - size if align_bottom else 0)
            left = c
            for rr, cc in _square_outline_coords(top, left, size):
                patch[rr, cc] = GREY
            c += size + 1
        base_r += row_h + 1
    return patch


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]], margin: int = 1) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(area: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < area:
        pick = list(frontier)[int(rng.integers(0, len(frontier)))]
        frontier.remove(pick)
        cells.add(pick)
        pr, pc = pick
        for dr, dc in NEIGHBORS4:
            nxt = (pr + dr, pc + dc)
            if nxt not in cells:
                frontier.add(nxt)
    return _normalize_cells(cells)


def _marker_candidates(shape: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(shape)
    out: list[tuple[int, int]] = []
    for r, c in shape:
        if r not in (r0, r1) and c not in (c0, c1):
            continue
        if _direction_from_grey(shape, (r, c)) == direction:
            out.append((r, c))
    return out


def _make_shape_with_marker(
    area: int, direction: str, rng: np.random.Generator, attempts: int = 300
) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(attempts):
        coords = _random_shape(area, rng)
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if area > 3 and (h < 2 or w < 2):
            continue
        candidates = _marker_candidates(coords, direction)
        if not candidates:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        grey = candidates[int(rng.integers(0, len(candidates)))]
        return mask, grey
    return None


def _placement_candidates(
    blocked: np.ndarray,
    mask: np.ndarray,
    grey_local: tuple[int, int],
    area: int,
    direction: str,
    require_clipped: bool,
) -> list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]]:
    h, w = blocked.shape
    sh, sw = mask.shape
    local_cells = [(r, c) for r, c in np.argwhere(mask)]
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for top0 in range(h - sh + 1):
        for left0 in range(w - sw + 1):
            shape_cells = [(top0 + r, left0 + c) for r, c in local_cells]
            if any(blocked[r, c] for r, c in shape_cells):
                continue

            grey_r = top0 + grey_local[0]
            grey_c = left0 + grey_local[1]
            top, left, bottom, right = _stamp_bbox_from_grey(grey_r, grey_c, area, direction)

            fully_in = top >= 0 and left >= 0 and bottom < h and right < w
            if require_clipped:
                if direction == "up":
                    clipped_ok = top < 0 and left >= 0 and right < w and bottom < h
                elif direction == "down":
                    clipped_ok = bottom >= h and left >= 0 and right < w and top >= 0
                elif direction == "left":
                    clipped_ok = left < 0 and top >= 0 and bottom < h and right < w
                else:
                    clipped_ok = right >= w and top >= 0 and bottom < h and left >= 0
                if not clipped_ok:
                    continue
            elif not fully_in:
                continue

            stamp_region = _stamp_region_in_bounds(h, w, top, left, bottom, right)
            if not stamp_region or any(blocked[r, c] for r, c in stamp_region):
                continue

            outline = [(r, c) for r, c in _square_outline_coords(top, left, area) if 0 <= r < h and 0 <= c < w]
            if require_clipped and len(outline) < 2 * area - 1:
                continue
            candidates.append((top0, left0, shape_cells, stamp_region))
    return candidates


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = np.random.default_rng()
        legend = _make_legend_patch(rng)
        lh, lw = legend.shape

        h = int(rng.integers(max(28, lh + 8), 37))
        w = int(rng.integers(max(28, lw + 8), 37))
        grid = np.full((h, w), BG, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        corner = int(rng.integers(0, 4))
        if corner == 0:
            legend_top, legend_left = 1, 1
        elif corner == 1:
            legend_top, legend_left = 1, w - lw - 1
        elif corner == 2:
            legend_top, legend_left = h - lh - 1, 1
        else:
            legend_top, legend_left = h - lh - 1, w - lw - 1
        _place_patch(grid, legend, legend_top, legend_left)
        legend_cells = [
            (legend_top + r, legend_left + c)
            for r in range(lh)
            for c in range(lw)
        ]
        _mark_blocked(blocked, legend_cells, margin=1)

        n_shapes = int(rng.integers(3, 5))
        areas = rng.choice(np.arange(3, 8), size=n_shapes, replace=False).tolist()
        directions = [DIR_ORDER[i] for i in rng.choice(np.arange(4), size=n_shapes, replace=False)]
        colors = rng.choice(np.array(PALETTE), size=n_shapes, replace=True).tolist()
        if n_shapes > 1 and len(set(colors)) == 1:
            colors[-1] = int(PALETTE[(PALETTE.index(colors[-1]) + 1) % len(PALETTE)])

        clipped_flags = [False] * n_shapes
        if bool(rng.integers(0, 2)):
            clipped_flags[int(rng.integers(0, n_shapes))] = True

        specs = list(zip(areas, directions, colors, clipped_flags, strict=True))
        specs.sort(key=lambda item: (not item[3], -item[0]))

        ok = True
        for area, direction, color, need_clipped in specs:
            built = _make_shape_with_marker(int(area), direction, rng)
            if built is None:
                ok = False
                break
            mask, grey_local = built
            placements = _placement_candidates(blocked, mask, grey_local, int(area), direction, bool(need_clipped))
            if not placements:
                ok = False
                break

            pick = placements[int(rng.integers(0, len(placements)))]
            top0, left0, shape_cells, stamp_region = pick
            for r, c in shape_cells:
                grid[r, c] = int(color)
            grid[top0 + grey_local[0], left0 + grey_local[1]] = GREY
            _mark_blocked(blocked, shape_cells + stamp_region, margin=1)

        if ok:
            return grid

    raise RuntimeError("failed to generate a valid input")
