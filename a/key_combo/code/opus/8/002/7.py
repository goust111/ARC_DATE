from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, components_mask, get_bg, paste


def _find_row_band(grid: np.ndarray, bg: int) -> tuple[int, int, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    r = 0
    while r < h:
        nz = np.flatnonzero(grid[r] != bg)
        if nz.size > 0 and int(nz[0]) == 0 and int(nz[-1]) == w - 1:
            start = r
            r += 1
            while r < h:
                nz = np.flatnonzero(grid[r] != bg)
                if nz.size == 0 or int(nz[0]) != 0 or int(nz[-1]) != w - 1:
                    break
                r += 1
            candidates.append((start, r - 1))
        else:
            r += 1

    assert candidates, "missing full-width row band"
    candidates.sort(
        key=lambda seg: (
            seg[1] - seg[0] + 1,
            int(np.count_nonzero(grid[seg[0] : seg[1] + 1] != bg)),
        ),
        reverse=True,
    )
    r0, r1 = candidates[0]
    tile = grid[r0 : r1 + 1].copy()
    assert len(set(map(int, np.unique(tile))) - {bg}) >= 2, "row band needs at least two colors"
    return r0, r1, tile


def _extract_sprite(grid: np.ndarray, color: int) -> tuple[int, int, np.ndarray]:
    mask = grid == int(color)
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1, "expected exactly one red sprite"
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = map(int, box)
    return r0, c0, grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tile_layer(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    band_r0, band_r1, tile = _find_row_band(grid, bg)
    tile_h = tile.shape[0]
    row_indices = (np.arange(grid.shape[0]) - band_r0) % tile_h
    return tile[row_indices].copy(), (band_r0, band_r1), tile


def _stamp_layer(
    shape: tuple[int, int],
    bg: int,
    sprite_patch: np.ndarray,
    sprite_origin: tuple[int, int],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    sprite_r, sprite_c = sprite_origin
    paste(out, sprite_patch, sprite_r, sprite_c, transparent=bg)

    ph, pw = sprite_patch.shape
    center_r = ph // 2
    center_c = pw // 2
    for marker_r, marker_c in markers:
        paste(out, sprite_patch, marker_r - center_r, marker_c - center_c, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the horizontal row band vertically, stamp the red sprite at every gray marker, and flip layer priority based on the marker count.

    Input structure:
    - The grid has a black background, one full-width horizontal row band, one small red 4-connected sprite, and several gray marker pixels.
    - The row band occupies one contiguous block of 2-3 rows and spans the whole canvas width.
    - Gray markers sit outside the band rows, and the red sprite's tight bounding box is at most 3x3.

    Rule:
    - Detect the full-width row band and repeat it upward and downward to fill the whole canvas with the same row-phase alignment.
    - Extract the red sprite's tight bounding box, treating black cells inside that box as transparent.
    - Build a stamp layer by keeping the original red sprite and pasting a copy centered on each gray marker, then removing the gray markers.
    - If the number of gray markers K is at most 3, draw the stamp layer over the tiled band layer; if K is greater than 3, let the tiled band's non-black pixels cover the stamp layer.
    - Finally redraw the original sprite so it always remains visible.

    Key insight:
    - The marker count does not move any object; it only flips which of the two independently built layers is visually on top.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    tiled, _band_rows, _tile = _tile_layer(g, bg)
    sprite_r, sprite_c, sprite_patch = _extract_sprite(g, Color.RED)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.GREY)]
    assert markers, "expected gray markers"

    stamped = _stamp_layer(g.shape, bg, sprite_patch, (sprite_r, sprite_c), markers)

    if len(markers) <= 3:
        out = tiled.copy()
        paste(out, stamped, 0, 0, transparent=bg)
    else:
        out = stamped.copy()
        mask = tiled != bg
        out[mask] = tiled[mask]

    paste(out, sprite_patch, sprite_r, sprite_c, transparent=bg)
    return out


_SPRITE_PATCHES = [
    np.array([[Color.RED, Color.RED, Color.RED]], dtype=int),
    np.array([[Color.RED], [Color.RED], [Color.RED]], dtype=int),
    np.array(
        [
            [Color.BLACK, Color.RED, Color.BLACK],
            [Color.RED, Color.RED, Color.RED],
            [Color.BLACK, Color.RED, Color.BLACK],
        ],
        dtype=int,
    ),
    np.array(
        [
            [Color.RED, Color.RED, Color.RED],
            [Color.BLACK, Color.RED, Color.BLACK],
            [Color.BLACK, Color.RED, Color.BLACK],
        ],
        dtype=int,
    ),
    np.array(
        [
            [Color.RED, Color.RED, Color.BLACK],
            [Color.BLACK, Color.RED, Color.BLACK],
            [Color.BLACK, Color.RED, Color.RED],
        ],
        dtype=int,
    ),
]


def _rotations(patch: np.ndarray) -> list[np.ndarray]:
    seen: set[bytes] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(patch, k).copy()
        key = rot.tobytes() + str(rot.shape).encode()
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


_ALL_PATCHES: list[np.ndarray] = []
for _patch in _SPRITE_PATCHES:
    _ALL_PATCHES.extend(_rotations(_patch))


def _random_band_row(width: int) -> np.ndarray | None:
    period = random.randint(3, min(6, width))
    for _ in range(50):
        motif = [random.choice([Color.BLACK, Color.BLUE, Color.YELLOW]) for _ in range(period)]
        if Color.BLUE not in motif or Color.YELLOW not in motif:
            continue
        row = np.array([motif[c % period] for c in range(width)], dtype=int)
        density = float(np.count_nonzero(row)) / float(width)
        if row[0] == Color.BLACK or row[-1] == Color.BLACK:
            continue
        if not (0.45 <= density <= 0.85):
            continue
        return row
    return None


def _make_band_tile(height: int, width: int) -> np.ndarray | None:
    for _ in range(100):
        rows = []
        for _row_index in range(height):
            row = _random_band_row(width)
            if row is None:
                break
            if rows and np.array_equal(row, rows[-1]):
                row = np.roll(row, random.randint(1, width - 1))
            rows.append(row)
        if len(rows) != height:
            continue
        tile = np.asarray(rows, dtype=int)
        if Color.BLACK not in tile:
            continue
        colors = set(map(int, np.unique(tile))) - {Color.BLACK}
        if len(colors) < 2:
            continue
        return tile
    return None


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _sample_marker_centers(
    height: int,
    width: int,
    band_rows: tuple[int, int],
    sprite_patch: np.ndarray,
    sprite_bbox: tuple[int, int, int, int],
    marker_count: int,
) -> list[tuple[int, int]] | None:
    ph, pw = sprite_patch.shape
    center_r = ph // 2
    center_c = pw // 2
    band_r0, band_r1 = band_rows

    candidates: list[tuple[int, int]] = []
    for r in range(center_r, height - (ph - 1 - center_r)):
        if band_r0 <= r <= band_r1:
            continue
        for c in range(center_c, width - (pw - 1 - center_c)):
            if c == 0 or c == width - 1:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    used_boxes = [sprite_bbox]
    for r, c in candidates:
        box = (r - center_r, c - center_c, r - center_r + ph - 1, c - center_c + pw - 1)
        if any(_bbox_overlap(box, other, pad=1) for other in used_boxes):
            continue
        chosen.append((r, c))
        used_boxes.append(box)
        if len(chosen) == marker_count:
            return chosen
    return None


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(14, 19)
    width = random.randint(13, 19)
    grid = np.zeros((height, width), dtype=int)

    band_h = random.choice([2, 3])
    band_r0 = random.randint(4, height - band_h - 4)
    tile = _make_band_tile(band_h, width)
    if tile is None:
        return None
    grid[band_r0 : band_r0 + band_h] = tile

    sprite_patch = random.choice(_ALL_PATCHES)
    ph, pw = sprite_patch.shape
    place_top = random.random() < 0.5
    if place_top:
        sprite_r_choices = list(range(1, band_r0 - ph))
    else:
        sprite_r_choices = list(range(band_r0 + band_h + 1, height - ph))
    if not sprite_r_choices:
        return None
    sprite_r = random.choice(sprite_r_choices)
    sprite_c = random.randint(1, width - pw - 1)
    paste(grid, sprite_patch, sprite_r, sprite_c, transparent=Color.BLACK)
    sprite_bbox = (sprite_r, sprite_c, sprite_r + ph - 1, sprite_c + pw - 1)

    if random.random() < 0.5:
        marker_count = random.randint(2, 3)
    else:
        marker_count = random.randint(4, 6)

    markers = _sample_marker_centers(
        height,
        width,
        (band_r0, band_r0 + band_h - 1),
        sprite_patch,
        sprite_bbox,
        marker_count,
    )
    if markers is None:
        return None
    for r, c in markers:
        grid[r, c] = Color.GREY

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if len(np.argwhere(grid == Color.GREY)) != marker_count:
        return None
    if len(np.argwhere(grid == Color.RED)) != int(np.count_nonzero(sprite_patch)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
