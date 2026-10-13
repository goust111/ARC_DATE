from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, bbox_coords, components_mask, get_bg


MirrorSpec = tuple[str, int, int, int]  # (axis, low_line, high_line, mirror_color)


def _find_parallel_mirrors(grid: np.ndarray, bg: int) -> MirrorSpec:
    candidates: list[MirrorSpec] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue

        full_rows = np.flatnonzero(np.all(grid == color, axis=1))
        if len(full_rows) == 2 and int(full_rows[1]) - int(full_rows[0]) >= 2:
            candidates.append(("horizontal", int(full_rows[0]), int(full_rows[1]), color))

        full_cols = np.flatnonzero(np.all(grid == color, axis=0))
        if len(full_cols) == 2 and int(full_cols[1]) - int(full_cols[0]) >= 2:
            candidates.append(("vertical", int(full_cols[0]), int(full_cols[1]), color))

    assert len(candidates) == 1, "expected exactly one pair of full parallel mirror lines"
    return candidates[0]


def _mirror_line_mask(shape: tuple[int, int], spec: MirrorSpec) -> np.ndarray:
    axis, lo, hi, _color = spec
    mask = np.zeros(shape, dtype=bool)
    if axis == "vertical":
        mask[:, lo] = True
        mask[:, hi] = True
    else:
        mask[lo, :] = True
        mask[hi, :] = True
    return mask


def _component_is_in_corridor(coords: np.ndarray, axis: str, lo: int, hi: int) -> bool:
    values = coords[:, 1] if axis == "vertical" else coords[:, 0]
    inside = bool(np.all((values > lo) & (values < hi)))
    outside = bool(np.all((values < lo) | (values > hi)))
    assert inside or outside, "objects must lie wholly inside or outside the mirror corridor"
    return inside


def _corridor_components(grid: np.ndarray, spec: MirrorSpec, bg: int) -> list[np.ndarray]:
    axis, lo, hi, _mirror = spec
    fg_mask = grid != int(bg)
    fg_mask[_mirror_line_mask(grid.shape, spec)] = False
    corridor: list[np.ndarray] = []
    for coords in components_mask(fg_mask, connectivity=4):
        if _component_is_in_corridor(coords, axis, lo, hi):
            corridor.append(coords)
    assert corridor, "expected at least one corridor object"
    return corridor


def _paint_full_cascade(out: np.ndarray, src: np.ndarray, coords: np.ndarray, spec: MirrorSpec) -> None:
    axis, lo, hi, _mirror = spec
    H, W = out.shape
    span = hi - lo
    limit = max(H, W) + 2

    if axis == "vertical":
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            color = int(src[r, c])
            for k in range(-limit, limit + 1):
                cc = c + 2 * k * span
                if 0 <= cc < W:
                    out[r, cc] = color
                cc = 2 * lo - c + 2 * k * span
                if 0 <= cc < W:
                    out[r, cc] = color
    else:
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            color = int(src[r, c])
            for k in range(-limit, limit + 1):
                rr = r + 2 * k * span
                if 0 <= rr < H:
                    out[rr, c] = color
                rr = 2 * lo - r + 2 * k * span
                if 0 <= rr < H:
                    out[rr, c] = color


def _paint_direct_reflections(out: np.ndarray, src: np.ndarray, coords: np.ndarray, spec: MirrorSpec) -> None:
    axis, lo, hi, _mirror = spec
    H, W = out.shape
    if axis == "vertical":
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            color = int(src[r, c])
            for cc in (2 * lo - c, 2 * hi - c):
                if 0 <= cc < W:
                    out[r, cc] = color
    else:
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            color = int(src[r, c])
            for rr in (2 * lo - r, 2 * hi - r):
                if 0 <= rr < H:
                    out[rr, c] = color


def _has_visible_even_copy(coords: np.ndarray, spec: MirrorSpec, shape: tuple[int, int]) -> bool:
    axis, lo, hi, _mirror = spec
    H, W = shape
    span = hi - lo
    limit = max(H, W) + 2
    if axis == "vertical":
        for _r, c_raw in coords:
            c = int(c_raw)
            for k in range(-limit, limit + 1):
                if k == 0:
                    continue
                cc = c + 2 * k * span
                if 0 <= cc < W:
                    return True
    else:
        for r_raw, _c in coords:
            r = int(r_raw)
            for k in range(-limit, limit + 1):
                if k == 0:
                    continue
                rr = r + 2 * k * span
                if 0 <= rr < H:
                    return True
    return False


def _has_visible_odd_copy(coords: np.ndarray, spec: MirrorSpec, shape: tuple[int, int]) -> bool:
    axis, lo, hi, _mirror = spec
    H, W = shape
    span = hi - lo
    limit = max(H, W) + 2
    if axis == "vertical":
        for _r, c_raw in coords:
            c = int(c_raw)
            for k in range(-limit, limit + 1):
                cc = 2 * lo - c + 2 * k * span
                if 0 <= cc < W:
                    return True
    else:
        for r_raw, _c in coords:
            r = int(r_raw)
            for k in range(-limit, limit + 1):
                rr = 2 * lo - r + 2 * k * span
                if 0 <= rr < H:
                    return True
    return False


def _component_has_partial_clip(coords: np.ndarray, spec: MirrorSpec, shape: tuple[int, int]) -> bool:
    axis, lo, hi, _mirror = spec
    r0, c0, r1, c1 = bbox_coords(coords)
    span = hi - lo
    H, W = shape
    limit = max(H, W) + 2

    if axis == "vertical":
        width = c1 - c0 + 1
        starts = (c0, 2 * lo - c1)
        for base in starts:
            for k in range(-limit, limit + 1):
                start = base + 2 * k * span
                end = start + width - 1
                intersects = start < W and end >= 0
                fully_inside = 0 <= start and end < W
                if intersects and not fully_inside:
                    return True
    else:
        height = r1 - r0 + 1
        starts = (r0, 2 * lo - r1)
        for base in starts:
            for k in range(-limit, limit + 1):
                start = base + 2 * k * span
                end = start + height - 1
                intersects = start < H and end >= 0
                fully_inside = 0 <= start and end < H
                if intersects and not fully_inside:
                    return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade every corridor object through the two parallel mirrors until the reflected hallway fills the canvas.

    Input structure:
    - The background contains exactly two full parallel grey mirror lines, either two rows or two columns.
    - One or more colored objects lie strictly between the mirrors inside the corridor.
    - Additional objects may lie outside the corridor, and no object touches a mirror line.

    Rule:
    - Keep the mirror lines and every outside-corridor object unchanged.
    - For each corridor object, repeatedly alternate reflections across the two mirrors, producing the entire infinite image family.
    - Paint every in-bounds reflected copy in its original colors, clipping any part that falls outside the canvas.

    Key insight:
    - Two parallel mirrors generate two interleaved sequences: odd reflection copies are flipped, while even reflection copies are translated by twice the mirror spacing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected one ARC grid"

    bg = int(get_bg(g))
    spec = _find_parallel_mirrors(g, bg)
    corridor = _corridor_components(g, spec, bg)

    out = g.copy()
    for coords in corridor:
        _paint_full_cascade(out, g, coords, spec)
    return out


GEN_RETRIES = 500
GEN_AXIS_CHOICES = ("vertical", "horizontal")
GEN_SIZE_RANGE = (16, 28)
GEN_CORRIDOR_COUNT_RANGE = (1, 3)
GEN_OUTSIDE_COUNT_RANGE = (1, 3)
GEN_OBJECT_SIZE_RANGE = (2, 4)


def _rand_comp(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _pack_intervals(length: int, sizes: list[int], min_gap: int = 1) -> list[int] | None:
    need = sum(sizes) + min_gap * max(0, len(sizes) - 1)
    if need > length:
        return None
    extras = _rand_comp(length - need, len(sizes) + 1)
    starts: list[int] = []
    cursor = extras[0]
    for idx, size in enumerate(sizes):
        starts.append(cursor)
        cursor += size
        if idx + 1 < len(sizes):
            cursor += min_gap + extras[idx + 1]
    return starts


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    selected = {seed}
    frontier = {seed}

    while len(selected) < area:
        if not frontier:
            frontier = set(selected)
        r, c = random.choice(tuple(frontier))
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in selected:
                nbrs.append((rr, cc))
        if not nbrs:
            frontier.discard((r, c))
            continue
        cell = random.choice(nbrs)
        selected.add(cell)
        frontier.add(cell)

    for r, c in selected:
        mask[r, c] = True
    return mask


def _make_patch(axis: str, max_cross: int, palette: list[int], require_asymmetry: bool) -> np.ndarray | None:
    cross_cap = min(4, int(max_cross))
    if cross_cap < 2:
        return None

    for _ in range(80):
        if axis == "vertical":
            h = random.randint(*GEN_OBJECT_SIZE_RANGE)
            w = random.randint(2, cross_cap)
        else:
            h = random.randint(2, cross_cap)
            w = random.randint(*GEN_OBJECT_SIZE_RANGE)

        max_area = min(h * w - 1, 6)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _random_connected_mask(h, w, area)
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]

        if require_asymmetry:
            flipped = np.fliplr(mask) if axis == "vertical" else np.flipud(mask)
            if np.array_equal(mask, flipped):
                continue

        patch = np.zeros(mask.shape, dtype=int)
        base_color = random.choice(palette)
        patch[mask] = base_color

        if mask.sum() >= 4 and len(palette) >= 2 and random.random() < 0.4:
            accent_color = random.choice([c for c in palette if c != base_color])
            cells = [tuple(map(int, x)) for x in np.argwhere(mask)]
            accent_count = random.randint(1, min(2, len(cells) - 1))
            for rr, cc in random.sample(cells, accent_count):
                patch[rr, cc] = accent_color

        return patch
    return None


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = patch != 0
    grid_view = grid[r0 : r0 + h, c0 : c0 + w]
    grid_view[region] = patch[region]


def _place_inside_patch(grid: np.ndarray, patch: np.ndarray, spec: MirrorSpec, anchor: int, clip_bias: bool) -> bool:
    axis, lo, hi, _mirror = spec
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []

    if axis == "vertical":
        for c0 in range(lo + 1, hi - w + 1):
            candidates.append((anchor, c0))
        if clip_bias:
            biased = [
                pos
                for pos in candidates
                if _component_has_partial_clip(_patch_coords(patch, pos[0], pos[1]), spec, grid.shape)
            ]
            if biased:
                candidates = biased
    else:
        for r0 in range(lo + 1, hi - h + 1):
            candidates.append((r0, anchor))
        if clip_bias:
            biased = [
                pos
                for pos in candidates
                if _component_has_partial_clip(_patch_coords(patch, pos[0], pos[1]), spec, grid.shape)
            ]
            if biased:
                candidates = biased

    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    _paint_patch(grid, patch, r0, c0)
    return True


def _patch_coords(patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(patch != 0).copy()
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _outside_candidates(spec: MirrorSpec, patch: np.ndarray, shape: tuple[int, int], anchor: int) -> list[tuple[int, int]]:
    axis, lo, hi, _mirror = spec
    H, W = shape
    h, w = patch.shape
    positions: list[tuple[int, int]] = []

    if axis == "vertical":
        for c0 in range(0, lo - w + 1):
            positions.append((anchor, c0))
        for c0 in range(hi + 1, W - w + 1):
            positions.append((anchor, c0))
    else:
        for r0 in range(0, lo - h + 1):
            positions.append((r0, anchor))
        for r0 in range(hi + 1, H - h + 1):
            positions.append((r0, anchor))
    return positions


def _has_extra_if_cascaded(patch: np.ndarray, pos: tuple[int, int], spec: MirrorSpec, shape: tuple[int, int]) -> bool:
    coords = _patch_coords(patch, pos[0], pos[1])
    axis, lo, hi, _mirror = spec
    H, W = shape
    span = hi - lo
    limit = max(H, W) + 2
    original = {tuple(map(int, rc)) for rc in coords}

    if axis == "vertical":
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            for k in range(-limit, limit + 1):
                for cc in (c + 2 * k * span, 2 * lo - c + 2 * k * span):
                    if 0 <= cc < W and (r, int(cc)) not in original:
                        return True
    else:
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            for k in range(-limit, limit + 1):
                for rr in (r + 2 * k * span, 2 * lo - r + 2 * k * span):
                    if 0 <= rr < H and (int(rr), c) not in original:
                        return True
    return False


def _build_scene() -> np.ndarray | None:
    axis = random.choice(GEN_AXIS_CHOICES)
    bg = random.choice([Color.BLACK, Color.ORANGE, Color.YELLOW, Color.TEAL])
    if bg == Color.GREY:
        bg = Color.BLACK
    palette = [c for c in Color.NOT_BLACK if c not in {bg, Color.GREY}]

    span = random.randint(4, 7)
    if axis == "vertical":
        lo_min = span - 1
        lo_max = min(7, GEN_SIZE_RANGE[1] - 3 * span - 1)
        if lo_min > lo_max:
            return None
        lo = random.randint(lo_min, lo_max)
        min_w = lo + 3 * span + 1
        W = random.randint(max(GEN_SIZE_RANGE[0], min_w), GEN_SIZE_RANGE[1])
        H = random.randint(*GEN_SIZE_RANGE)
        hi = lo + span
        corridor_cross = hi - lo - 1
        outside_cross = max(lo, W - hi - 1)
        invariant_len = H
    else:
        lo_min = span - 1
        lo_max = min(7, GEN_SIZE_RANGE[1] - 3 * span - 1)
        if lo_min > lo_max:
            return None
        lo = random.randint(lo_min, lo_max)
        min_h = lo + 3 * span + 1
        H = random.randint(max(GEN_SIZE_RANGE[0], min_h), GEN_SIZE_RANGE[1])
        W = random.randint(*GEN_SIZE_RANGE)
        hi = lo + span
        corridor_cross = hi - lo - 1
        outside_cross = max(lo, H - hi - 1)
        invariant_len = W

    if corridor_cross < 2 or outside_cross < 2:
        return None

    spec: MirrorSpec = (axis, lo, hi, Color.GREY)
    grid = np.full((H, W), int(bg), dtype=int)
    line_mask = _mirror_line_mask(grid.shape, spec)
    grid[line_mask] = int(Color.GREY)

    corridor_count = random.randint(*GEN_CORRIDOR_COUNT_RANGE)
    outside_count = random.randint(*GEN_OUTSIDE_COUNT_RANGE)
    force_clip = random.random() < 0.55

    corridor_patches: list[np.ndarray] = []
    for _ in range(corridor_count):
        patch = _make_patch(axis, corridor_cross, palette, require_asymmetry=True)
        if patch is None:
            return None
        corridor_patches.append(patch)

    outside_patches: list[np.ndarray] = []
    for _ in range(outside_count):
        patch = _make_patch(axis, outside_cross, palette, require_asymmetry=False)
        if patch is None:
            return None
        outside_patches.append(patch)

    sizes = []
    for patch in corridor_patches + outside_patches:
        sizes.append(int(patch.shape[0] if axis == "vertical" else patch.shape[1]))
    starts = _pack_intervals(invariant_len, sizes, min_gap=1)
    if starts is None:
        return None

    corridor_starts = starts[:corridor_count]
    outside_starts = starts[corridor_count:]

    clip_index = random.randrange(corridor_count) if force_clip else -1
    for idx, (patch, anchor) in enumerate(zip(corridor_patches, corridor_starts)):
        if not _place_inside_patch(grid, patch, spec, anchor, clip_bias=(idx == clip_index)):
            return None

    misleading_placed = False
    for idx, (patch, anchor) in enumerate(zip(outside_patches, outside_starts)):
        positions = _outside_candidates(spec, patch, grid.shape, anchor)
        if not positions:
            return None
        random.shuffle(positions)
        if not misleading_placed:
            positions = [pos for pos in positions if _has_extra_if_cascaded(patch, pos, spec, grid.shape)] or positions
        r0, c0 = positions[0]
        _paint_patch(grid, patch, r0, c0)
        if _has_extra_if_cascaded(patch, (r0, c0), spec, grid.shape):
            misleading_placed = True

    if not misleading_placed:
        return None

    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    bg = int(get_bg(grid))
    spec = _find_parallel_mirrors(grid, bg)
    axis, lo, hi, _mirror = spec
    line_mask = _mirror_line_mask(grid.shape, spec)

    comps = components_mask((grid != bg) & ~line_mask, connectivity=4)
    corridor = [coords for coords in comps if _component_is_in_corridor(coords, axis, lo, hi)]
    outside = [coords for coords in comps if not _component_is_in_corridor(coords, axis, lo, hi)]
    if not corridor or not outside:
        return False

    out = main(grid)

    expected = grid.copy()
    for coords in corridor:
        _paint_full_cascade(expected, grid, coords, spec)
    if not np.array_equal(out, expected):
        return False

    outside_mask = np.zeros_like(grid, dtype=bool)
    for coords in outside:
        outside_mask[coords[:, 0], coords[:, 1]] = True
    if not np.array_equal(out[outside_mask], grid[outside_mask]):
        return False

    if not any(_has_visible_even_copy(coords, spec, grid.shape) for coords in corridor):
        return False
    if not any(_has_visible_odd_copy(coords, spec, grid.shape) for coords in corridor):
        return False

    direct = grid.copy()
    for coords in corridor:
        _paint_direct_reflections(direct, grid, coords, spec)
    if np.array_equal(out, direct):
        return False

    wrong = grid.copy()
    for coords in comps:
        _paint_full_cascade(wrong, grid, coords, spec)
    if np.array_equal(out, wrong):
        return False

    has_clip = any(_component_has_partial_clip(coords, spec, grid.shape) for coords in corridor)
    if has_clip and np.array_equal(out, grid):
        return False

    if axis == "vertical":
        if not np.all(grid[:, lo] == Color.GREY) or not np.all(grid[:, hi] == Color.GREY):
            return False
    else:
        if not np.all(grid[lo, :] == Color.GREY) or not np.all(grid[hi, :] == Color.GREY):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_scene()
        if grid is not None and _validate_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid double-mirror corridor puzzle")
