from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _paste_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    r1 = int(r0) + h
    c1 = int(c0) + w
    assert 0 <= r0 and 0 <= c0 and r1 <= dst.shape[0] and c1 <= dst.shape[1]
    dst[r0:r1, c0:c1][mask] = int(color)


def _mirror_expand(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return np.block([[g, np.fliplr(g)], [np.flipud(g), np.flipud(np.fliplr(g))]])


def _extract_sprite_and_markers(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)
    multi = [comp for comp in comps if int(comp["area"]) > 1]
    singles = [comp for comp in comps if int(comp["area"]) == 1]

    assert len(multi) == 1, "expected exactly one sprite component"
    assert 1 <= len(singles) <= 2, "expected one or two marker pixels"

    sprite = multi[0]
    sprite_color = int(sprite["color"])
    marker_color = int(singles[0]["color"])
    assert all(int(comp["color"]) == marker_color for comp in singles), "markers must share one color"
    assert marker_color != sprite_color, "marker and sprite colors must differ"

    sprite_patch = _crop_bbox(grid, sprite["bbox"])
    sprite_mask = sprite_patch == sprite_color
    markers = [tuple(map(int, comp["coords"][0])) for comp in singles]
    markers.sort()
    return sprite_mask, markers, marker_color, bg


def _stamp_all(
    out: np.ndarray,
    sprite_mask: np.ndarray,
    markers: list[tuple[int, int]],
    marker_color: int,
    *,
    use_quadrant_mirrors: bool,
    anchor_to_filled_cell: bool,
) -> np.ndarray:
    h, w = out.shape[0] // 2, out.shape[1] // 2
    if use_quadrant_mirrors:
        masks = [
            sprite_mask,
            np.fliplr(sprite_mask),
            np.flipud(sprite_mask),
            np.flipud(np.fliplr(sprite_mask)),
        ]
    else:
        masks = [sprite_mask, sprite_mask, sprite_mask, sprite_mask]

    if anchor_to_filled_cell:
        filled_r, filled_c = map(int, np.argwhere(sprite_mask)[0])
    else:
        filled_r = filled_c = 0

    quadrant_positions = [
        lambda r, c: (r, c),
        lambda r, c: (r, 2 * w - 1 - c),
        lambda r, c: (2 * h - 1 - r, c),
        lambda r, c: (2 * h - 1 - r, 2 * w - 1 - c),
    ]

    for q, place_fn in enumerate(quadrant_positions):
        mask = masks[q]
        for r, c in markers:
            rr, cc = place_fn(int(r), int(c))
            _paste_mask(out, mask, rr - filled_r, cc - filled_c, marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-expand the grid, then stamp the original sprite at every mirrored marker position without flipping the stamp.

    Input structure:
    - A small black-background grid contains one monochrome asymmetric sprite and one or two singleton marker pixels.
    - The sprite is a single 4-connected component with a small bounding box.
    - All markers share one color different from the sprite color and are separated from the sprite.

    Rule:
    - Expand the whole input to a 2x2 mirror layout: top-left original, top-right left-right flip,
      bottom-left top-bottom flip, bottom-right both flips.
    - The marker pixels are mirrored with the rest of the grid, creating one copy per quadrant.
    - At each mirrored marker position, align the sprite bounding box's top-left corner to that marker
      and overwrite the sprite mask in the marker color.
    - The stamped sprite always keeps its original orientation; only the marker positions are mirrored.

    Key insight:
    - The background pattern becomes four-way symmetric, but the stamp breaks that symmetry because
      the copied sprite does not inherit the quadrant's flip direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    sprite_mask, markers, marker_color, _bg = _extract_sprite_and_markers(g)
    out = _mirror_expand(g)
    return _stamp_all(
        out,
        sprite_mask,
        markers,
        marker_color,
        use_quadrant_mirrors=False,
        anchor_to_filled_cell=False,
    )


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    target = {tuple(map(int, p)) for p in coords}
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nbr in target and nbr not in seen:
                stack.append(nbr)
    return len(seen) == len(target)


def _build_shape_library() -> tuple[list[np.ndarray], list[np.ndarray]]:
    corner_filled: list[np.ndarray] = []
    corner_empty: list[np.ndarray] = []
    for h, w in ((2, 3), (2, 4), (3, 3), (3, 4)):
        cell_count = h * w
        for bits in range(1 << cell_count):
            area = bits.bit_count()
            if not (3 <= area <= 6):
                continue
            mask = np.zeros((h, w), dtype=bool)
            for k in range(cell_count):
                if bits & (1 << k):
                    mask[k // w, k % w] = True
            if not _mask_connected(mask):
                continue
            if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                continue
            if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
                continue
            if np.array_equal(mask, np.flipud(np.fliplr(mask))):
                continue
            target = corner_filled if bool(mask[0, 0]) else corner_empty
            target.append(mask.copy())
    assert corner_filled and corner_empty
    return corner_filled, corner_empty


_CORNER_FILLED_SHAPES, _CORNER_EMPTY_SHAPES = _build_shape_library()


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sprite_rect(mask: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    rows, cols = np.argwhere(mask).T
    return int(top + rows.min()), int(left + cols.min()), int(top + rows.max()), int(left + cols.max())


def _chebyshev_far_enough(point: tuple[int, int], cells: list[tuple[int, int]], gap: int) -> bool:
    r, c = point
    return all(max(abs(r - rr), abs(c - cc)) > gap for rr, cc in cells)


def _marker_candidates(h: int, w: int, sprite_mask: np.ndarray, sprite_top: int, sprite_left: int) -> list[tuple[int, int]]:
    sprite_cells = [tuple(map(int, cell)) for cell in np.argwhere(sprite_mask)]
    sprite_cells = [(sprite_top + r, sprite_left + c) for r, c in sprite_cells]
    out: list[tuple[int, int]] = []
    stamp_h, stamp_w = sprite_mask.shape
    for r in range(h):
        for c in range(w):
            if r < stamp_h - 1 or c < stamp_w - 1:
                continue
            if any((r == rr and c == cc) for rr, cc in sprite_cells):
                continue
            if not _chebyshev_far_enough((r, c), sprite_cells, gap=1):
                continue
            out.append((r, c))
    return out


def _render_input(
    h: int,
    w: int,
    sprite_mask: np.ndarray,
    sprite_top: int,
    sprite_left: int,
    sprite_color: int,
    markers: list[tuple[int, int]],
    marker_color: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    patch = np.zeros_like(sprite_mask, dtype=int)
    patch[sprite_mask] = int(sprite_color)
    _paste_mask(grid, sprite_mask, sprite_top, sprite_left, sprite_color)
    for r, c in markers:
        grid[r, c] = int(marker_color)
    return grid


def _wrong_flipped_stamp(grid: np.ndarray) -> np.ndarray:
    sprite_mask, markers, marker_color, _bg = _extract_sprite_and_markers(grid)
    return _stamp_all(
        _mirror_expand(grid),
        sprite_mask,
        markers,
        marker_color,
        use_quadrant_mirrors=True,
        anchor_to_filled_cell=False,
    )


def _wrong_filled_anchor(grid: np.ndarray) -> np.ndarray:
    sprite_mask, markers, marker_color, _bg = _extract_sprite_and_markers(grid)
    return _stamp_all(
        _mirror_expand(grid),
        sprite_mask,
        markers,
        marker_color,
        use_quadrant_mirrors=False,
        anchor_to_filled_cell=True,
    )


def _sample_markers(candidates: list[tuple[int, int]], sprite_mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    stamp_h, stamp_w = sprite_mask.shape
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    rects: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        rect = (r, c, r + stamp_h - 1, c + stamp_w - 1)
        if all(_rect_separated(rect, prev, gap=0) for prev in rects) and all(
            max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen
        ):
            chosen.append((r, c))
            rects.append(rect)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _attempt_generate() -> np.ndarray | None:
    use_corner_empty = random.random() < 0.5
    shape_pool = _CORNER_EMPTY_SHAPES if use_corner_empty else _CORNER_FILLED_SHAPES
    sprite_mask = random.choice(shape_pool)
    stamp_h, stamp_w = sprite_mask.shape

    min_h = max(5, stamp_h + 2)
    min_w = max(6, stamp_w + 2)
    if min_h > 7 or min_w > 7:
        return None
    h = random.randint(min_h, 7)
    w = random.randint(min_w, 7)

    sprite_color, marker_color = random.sample(range(1, 10), 2)
    marker_count = 2 if random.random() < 0.6 else 1

    placements = [(r, c) for r in range(h - stamp_h + 1) for c in range(w - stamp_w + 1)]
    random.shuffle(placements)
    for sprite_top, sprite_left in placements:
        candidates = _marker_candidates(h, w, sprite_mask, sprite_top, sprite_left)
        markers = _sample_markers(candidates, sprite_mask, marker_count)
        if markers is None:
            continue

        grid = _render_input(h, w, sprite_mask, sprite_top, sprite_left, sprite_color, markers, marker_color)
        correct = main(grid)
        if np.array_equal(correct, _mirror_expand(grid)):
            continue
        if np.array_equal(correct, _wrong_flipped_stamp(grid)):
            continue
        if use_corner_empty and np.array_equal(correct, _wrong_filled_anchor(grid)):
            continue

        out_colors = set(map(int, np.unique(correct)))
        if len(out_colors) < 3:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-stamp input")
