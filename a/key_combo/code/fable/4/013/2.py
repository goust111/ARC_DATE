from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors8(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _bbox_from_mask(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
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
                for nr, nc in _neighbors8(mask.shape, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(coords),
                }
            )
    return comps


def _extract_tile_and_control(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fg_mask = g != bg
    comps = _find_components(fg_mask)
    assert comps, "expected foreground objects"

    best_idx = None
    best_score = None
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted({int(v) for v in patch.flat if int(v) != bg})
        h, w = patch.shape
        full = int(np.count_nonzero(patch == bg)) == 0
        score = (len(colors) >= 2, full, h == w, len(colors), h * w)
        if best_score is None or score > best_score:
            best_idx = idx
            best_score = score

    assert best_idx is not None and best_score is not None
    assert best_score[:3] == (True, True, True), "missing solid multicolor square tile"

    tile_comp = comps[best_idx]
    tile_mask = np.zeros_like(fg_mask, dtype=bool)
    for r, c in tile_comp["coords"]:
        tile_mask[r, c] = True

    tr0, tc0, tr1, tc1 = tile_comp["bbox"]
    tile = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    control_mask_global = fg_mask & ~tile_mask
    assert np.any(control_mask_global), "missing control mask"
    control_bbox = _bbox_from_mask(control_mask_global)
    cr0, cc0, cr1, cc1 = control_bbox
    control_patch = control_mask_global[cr0 : cr1 + 1, cc0 : cc1 + 1].copy()
    control_colors = {int(v) for v in g[control_mask_global]}
    assert len(control_colors) == 1, "control mask must be monochrome"
    assert np.any(~control_patch), "control mask must contain background holes"

    return bg, tile, control_patch


def _oriented_tile(tile: np.ndarray, ctrl_r: int, ctrl_c: int, ctrl_h: int, ctrl_w: int) -> np.ndarray:
    out = tile
    if ctrl_r < (ctrl_h - 1) / 2.0:
        out = np.flipud(out)
    if ctrl_c < (ctrl_w - 1) / 2.0:
        out = np.fliplr(out)
    return out


def _stamp_output(tile: np.ndarray, control_mask: np.ndarray, bg: int) -> np.ndarray:
    side_h, side_w = tile.shape
    assert side_h == side_w, "tile must be square"
    ctrl_h, ctrl_w = control_mask.shape
    out = np.full((ctrl_h * side_h, ctrl_w * side_w), int(bg), dtype=int)
    for r, c in np.argwhere(control_mask):
        stamp = _oriented_tile(tile, int(r), int(c), ctrl_h, ctrl_w)
        rr = int(r) * side_h
        cc = int(c) * side_w
        out[rr : rr + side_h, cc : cc + side_w] = stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale a monochrome control mask by stamping a square tile that flips away from the mask center.

    Input structure:
    - A background grid contains exactly one monochrome rectangular control mask with background holes.
    - The other foreground object is a solid multicolor square tile with no background cells inside it.
    - The tile drawn in the input is the unflipped standard orientation for the mask's lower-right side.

    Rule:
    - Crop the control mask to its bounding box and treat each filled cell as one output block position.
    - Output size is `(control_rows * tile_side, control_cols * tile_side)`, with blank blocks where the control mask has holes.
    - Before stamping a tile copy, flip it horizontally when the control cell lies left of the mask center column and flip it vertically when the control cell lies above the mask center row.

    Key insight:
    - This is not a uniform tile enlargement: each stamped copy is reoriented by the control cell's direction from the control mask center, so the pattern appears to turn outward from the middle.
    """
    bg, tile, control_mask = _extract_tile_and_control(np.asarray(grid, dtype=int))
    return _stamp_output(tile, control_mask, bg)


def _make_asymmetric_tile(side: int, palette: list[int]) -> np.ndarray:
    n_colors = random.randint(3, min(4, len(palette), side * side))
    colors = random.sample(palette, n_colors)
    while True:
        tile = np.array(random.choices(colors, k=side * side), dtype=int).reshape(side, side)
        for pos, color in zip(random.sample(range(side * side), n_colors), colors):
            tile.flat[pos] = color
        variants = {
            tile.tobytes(),
            np.fliplr(tile).tobytes(),
            np.flipud(tile).tobytes(),
            np.flipud(np.fliplr(tile)).tobytes(),
        }
        if len(variants) == 4:
            return tile


def _control_size_options(tile_side: int) -> list[int]:
    return [size for size in (3, 5, 7, 9) if size * tile_side <= 30]


def _make_control_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    hole_count = random.randint(1, len(interior))
    for r, c in random.sample(interior, hole_count):
        mask[r, c] = False
    return mask


def _boxes_far_apart(box_a: BBox, box_b: BBox, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    others = np.delete(counts, int(bg))
    return int(counts[int(bg)]) > (int(others.max()) if others.size else 0)


def _place_objects(tile: np.ndarray, control_mask: np.ndarray, control_color: int, bg: int) -> np.ndarray | None:
    th, tw = tile.shape
    ch, cw = control_mask.shape
    min_h = max(12, max(th, ch) + 4)
    min_w = max(12, max(tw, cw) + 4)
    max_h = 30
    max_w = 30
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    for _ in range(200):
        cr = random.randint(1, h - ch - 1)
        cc = random.randint(1, w - cw - 1)
        tr = random.randint(1, h - th - 1)
        tc = random.randint(1, w - tw - 1)
        control_box = (cr, cc, cr + ch - 1, cc + cw - 1)
        tile_box = (tr, tc, tr + th - 1, tc + tw - 1)
        if not _boxes_far_apart(control_box, tile_box, pad=1):
            continue

        grid = np.full((h, w), int(bg), dtype=int)
        grid[tr : tr + th, tc : tc + tw] = tile
        control_view = grid[cr : cr + ch, cc : cc + cw]
        control_view[control_mask] = int(control_color)
        return grid
    return None


def _make_sample() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randint(0, 9)
    usable = [c for c in range(10) if c != bg]
    control_color = random.choice(usable)
    tile_palette = [c for c in usable if c != control_color]

    tile_side = random.randint(3, 5)
    tile = _make_asymmetric_tile(tile_side, tile_palette)

    size_options = _control_size_options(tile_side)
    ctrl_h = random.choice(size_options)
    ctrl_w = random.choice(size_options)
    control_mask = _make_control_mask(ctrl_h, ctrl_w)

    grid = _place_objects(tile, control_mask, control_color, bg)
    if grid is None:
        return None

    out = _stamp_output(tile, control_mask, bg)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not _background_is_dominant(grid, bg):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(300):
        sample = _make_sample()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input for puzzle 2")
