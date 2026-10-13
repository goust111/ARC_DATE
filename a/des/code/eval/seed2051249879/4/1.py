from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = _trim_mask(mask.astype(bool))
    variants: list[tuple[tuple[int, int], bytes]] = []
    for flip in (False, True):
        base = np.flipud(mask) if flip else mask
        for k in range(4):
            variant = np.rot90(base, k)
            variants.append((variant.shape, variant.tobytes()))
    return min(variants)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for flip in (False, True):
        base = np.flipud(mask) if flip else mask
        for k in range(4):
            variant = np.rot90(base, k)
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _infer_palette(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], dict[tuple[tuple[int, int], bytes], int], int]:
    best = None
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            if not _touches_boundary(comp, g.shape):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            inner_comps = [
                pts for pts in _components(sub != color) if not _touches_boundary(pts, sub.shape)
            ]
            if len(inner_comps) < 2:
                continue

            color_sets: list[set[int]] = []
            ok = True
            for pts in inner_comps:
                vals = {int(sub[rr, cc]) for rr, cc in pts}
                if len(vals) != 2:
                    ok = False
                    break
                color_sets.append(vals)
            if not ok:
                continue

            common = set.intersection(*color_sets)
            if len(common) != 1:
                continue
            grey = next(iter(common))

            mapping: dict[tuple[tuple[int, int], bytes], int] = {}
            seen_fills: set[int] = set()
            for pts in inner_comps:
                vals = {int(sub[rr, cc]) for rr, cc in pts}
                fill_choices = vals - {grey}
                if len(fill_choices) != 1:
                    ok = False
                    break
                fill = next(iter(fill_choices))
                if fill in seen_fills:
                    ok = False
                    break
                seen_fills.add(fill)

                cell_mask = np.zeros(sub.shape, dtype=bool)
                cell_mask[pts[:, 0], pts[:, 1]] = True
                grey_mask = cell_mask & (sub == grey)
                grey_comps = _components(grey_mask)
                if len(grey_comps) != 1:
                    ok = False
                    break
                grey_coords = grey_comps[0]
                gr0, gc0, gr1, gc1 = _bbox(grey_coords)
                shape_mask = np.zeros((gr1 - gr0 + 1, gc1 - gc0 + 1), dtype=bool)
                shape_mask[grey_coords[:, 0] - gr0, grey_coords[:, 1] - gc0] = True
                mapping[_canonical_key(shape_mask)] = fill

            if not ok or len(mapping) != len(inner_comps):
                continue

            score = (len(inner_comps), (r1 - r0 + 1) * (c1 - c0 + 1))
            if best is None or score > best[0]:
                best = (score, color, (r0, c0, r1, c1), mapping, grey)

    assert best is not None, "palette region not found"
    _, border_color, bbox, mapping, grey = best
    return int(border_color), bbox, mapping, int(grey)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the grey island shape in each closed fence to the palette template of the same
    shape (up to rotation/flip), then fill that fence's background interior with the
    palette cell's color while leaving the grey island and palette unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    fence_color, palette_bbox, palette_map, grey = _infer_palette(g, bg)
    pr0, pc0, pr1, pc1 = palette_bbox

    out = g.copy()
    for comp in _components(g == fence_color):
        r0, c0, r1, c1 = _bbox(comp)
        if (r0, c0, r1, c1) == (pr0, pc0, pr1, pc1):
            continue
        if _touches_boundary(comp, g.shape):
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        candidate_regions = []
        for pts in _components(sub != fence_color):
            if _touches_boundary(pts, sub.shape):
                continue
            vals = {int(sub[rr, cc]) for rr, cc in pts}
            if grey in vals:
                candidate_regions.append(pts)
        assert len(candidate_regions) == 1, "expected one enclosed island region per fence"

        region = candidate_regions[0]
        region_mask = np.zeros(sub.shape, dtype=bool)
        region_mask[region[:, 0], region[:, 1]] = True
        grey_mask = region_mask & (sub == grey)
        grey_comps = _components(grey_mask)
        assert len(grey_comps) == 1, "expected one grey island per fence"
        island = grey_comps[0]
        ir0, ic0, ir1, ic1 = _bbox(island)
        island_mask = np.zeros((ir1 - ir0 + 1, ic1 - ic0 + 1), dtype=bool)
        island_mask[island[:, 0] - ir0, island[:, 1] - ic0] = True

        fill = palette_map[_canonical_key(island_mask)]
        to_fill = region_mask & (sub == bg)
        out[r0 : r1 + 1, c0 : c1 + 1][to_fill] = fill

    return out


_SHAPE_LIBRARY_RAW = [
    [[1, 1]],
    [[1, 1, 1]],
    [[1, 0], [1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 0], [0, 1, 1]],
    [[1, 0], [1, 0], [1, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
]


def _build_shape_library() -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for raw in _SHAPE_LIBRARY_RAW:
        mask = np.asarray(raw, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            out.append(mask)
    return out


_SHAPES = _build_shape_library()
_RNG = random.Random()
_GRID_MIN = 28
_GRID_MAX = 34
_FENCE_PADDING = 1
_MAX_ATTEMPTS = 400


def _sample_colors(count: int) -> list[int]:
    return _RNG.sample(list(range(10)), count)


def _palette_dims(cell_h: int, cell_w: int, count: int, horizontal: bool) -> tuple[int, int]:
    if horizontal:
        return cell_h + 2, count * cell_w + count + 1
    return count * cell_h + count + 1, cell_w + 2


def _palette_origin(h: int, w: int, ph: int, pw: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - pw
    if corner == "bl":
        return h - ph, 0
    return h - ph, w - pw


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _choose_embed_start(span: int, obj: int) -> int:
    slack = span - obj
    if slack <= 2:
        return 1 if slack >= 1 else 0
    return _RNG.randint(1, slack - 1)


def _draw_palette(
    grid: np.ndarray,
    border_color: int,
    grey: int,
    shapes: list[np.ndarray],
    fill_colors: list[int],
    horizontal: bool,
    corner: str,
) -> tuple[int, int, int, int]:
    count = len(shapes)
    cell_h = max(mask.shape[0] for mask in shapes) + 2
    cell_w = max(mask.shape[1] for mask in shapes) + 2
    ph, pw = _palette_dims(cell_h, cell_w, count, horizontal)
    r0, c0 = _palette_origin(grid.shape[0], grid.shape[1], ph, pw, corner)

    grid[r0 : r0 + ph, c0 : c0 + pw] = border_color
    for idx, (shape, fill) in enumerate(zip(shapes, fill_colors, strict=False)):
        if horizontal:
            sr0 = r0 + 1
            sc0 = c0 + 1 + idx * (cell_w + 1)
        else:
            sr0 = r0 + 1 + idx * (cell_h + 1)
            sc0 = c0 + 1

        grid[sr0 : sr0 + cell_h, sc0 : sc0 + cell_w] = fill
        pr = _choose_embed_start(cell_h, shape.shape[0])
        pc = _choose_embed_start(cell_w, shape.shape[1])
        patch = grid[sr0 : sr0 + cell_h, sc0 : sc0 + cell_w]
        patch[pr : pr + shape.shape[0], pc : pc + shape.shape[1]][shape] = grey

    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 1 or c0 < 1 or r1 >= blocked.shape[0] - 1 or c1 >= blocked.shape[1] - 1:
        return False
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _place_fence(
    grid: np.ndarray,
    blocked: np.ndarray,
    border_color: int,
    grey: int,
    shape: np.ndarray,
    fill_color: int,
) -> tuple[int, int, int, int, int] | None:
    shape_h, shape_w = shape.shape
    inner_h = shape_h + _RNG.randint(2, 4)
    inner_w = shape_w + _RNG.randint(2, 4)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    for _ in range(150):
        r0 = _RNG.randint(1, grid.shape[0] - outer_h - 1)
        c0 = _RNG.randint(1, grid.shape[1] - outer_w - 1)
        if not _can_place(blocked, r0, c0, outer_h, outer_w):
            continue

        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color

        ir = r0 + 1 + _choose_embed_start(inner_h, shape_h)
        ic = c0 + 1 + _choose_embed_start(inner_w, shape_w)
        grid[ir : ir + shape_h, ic : ic + shape_w][shape] = grey

        _mark_blocked(blocked, r0, c0, r1, c1, _FENCE_PADDING)
        return r0, c0, r1, c1, fill_color

    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    count = _RNG.choice([3, 4])
    base_shapes = _RNG.sample(_SHAPES, count)
    palette_shapes = [_RNG.choice(_all_orientations(shape)) for shape in base_shapes]
    fence_shapes = [_RNG.choice(_all_orientations(shape)) for shape in base_shapes]

    bg, border_color, grey, *fill_colors = _sample_colors(count + 3)
    horizontal = _RNG.choice([False, True])
    corner = _RNG.choice(["tl", "tr", "bl", "br"])

    cell_h = max(mask.shape[0] for mask in palette_shapes) + 2
    cell_w = max(mask.shape[1] for mask in palette_shapes) + 2
    ph, pw = _palette_dims(cell_h, cell_w, count, horizontal)
    h = _RNG.randint(max(_GRID_MIN, ph + 2), max(_GRID_MIN, ph + 2) + (_GRID_MAX - _GRID_MIN))
    w = _RNG.randint(max(_GRID_MIN, pw + 2), max(_GRID_MIN, pw + 2) + (_GRID_MAX - _GRID_MIN))

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    pr0, pc0, pr1, pc1 = _draw_palette(
        grid, border_color, grey, palette_shapes, fill_colors, horizontal, corner
    )
    _mark_blocked(blocked, pr0, pc0, pr1, pc1, _FENCE_PADDING)

    fences: list[tuple[int, int, int, int, int]] = []
    order = sorted(
        range(count),
        key=lambda i: fence_shapes[i].shape[0] * fence_shapes[i].shape[1],
        reverse=True,
    )
    for idx in order:
        placement = _place_fence(
            grid,
            blocked,
            border_color,
            grey,
            fence_shapes[idx],
            fill_colors[idx],
        )
        if placement is None:
            raise RuntimeError("fence placement failed")
        fences.append(placement)

    expected = grid.copy()
    for r0, c0, r1, c1, fill_color in fences:
        inside = expected[r0 + 1 : r1, c0 + 1 : c1]
        inside[inside == bg] = fill_color

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        try:
            grid, expected = _generate_case()
        except RuntimeError:
            continue

        out = main(grid)
        if not np.array_equal(out, expected):
            continue
        if np.array_equal(out, grid):
            continue
        if len(np.unique(grid)) < 6:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
