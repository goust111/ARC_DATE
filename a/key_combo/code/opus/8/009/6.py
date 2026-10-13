from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _find_tile_and_control(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg = _get_bg(grid)
    comps = _components8(grid != bg)

    tile_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        colors = set(map(int, np.unique(patch)))
        if len(colors) < 2:
            continue
        tile_candidates.append((coords, patch.copy()))

    valid: list[tuple[np.ndarray, np.ndarray]] = []
    for coords, patch in tile_candidates:
        tile_mask = np.zeros_like(grid, dtype=bool)
        tile_mask[coords[:, 0], coords[:, 1]] = True
        control_fg = (grid != bg) & (~tile_mask)
        control_box = _bbox(control_fg)
        if control_box is None:
            continue
        control = _crop(grid, control_box)
        if not set(map(int, np.unique(control))).issubset({0, 1, 2}):
            continue
        if np.count_nonzero(control == 1) < 1 or np.count_nonzero(control == 2) < 1:
            continue
        valid.append((patch, control))

    assert len(valid) == 1, "expected a unique multicolor square tile"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a control grid into a tiled output where each control color chooses a tile transform.

    Input structure:
    - The background is black.
    - One separated object is a fully occupied multicolor square tile.
    - The other separated region is a rectangular control grid whose cells use only black(0),
      blue(1), and red(2), with black holes allowed inside its bbox.

    Rule:
    - Identify the square tile and the control grid.
    - For each control cell: 0 leaves the corresponding output block black, 1 stamps the tile
      unchanged, and 2 stamps the tile transposed across its main diagonal.
    - The output shape is `(control_h * tile_h, control_w * tile_w)`.

    Key insight:
    - The control grid is not a binary mask: each non-background color encodes which spatial
      transform of the same source tile should be stamped into that enlarged block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile, control = _find_tile_and_control(g)
    th, tw = tile.shape
    ch, cw = control.shape

    out = np.zeros((ch * th, cw * tw), dtype=int)
    transposed = tile.T.copy()
    for r in range(ch):
        for c in range(cw):
            cell = int(control[r, c])
            if cell == 1:
                _paste(out, tile, r * th, c * tw)
            elif cell == 2:
                _paste(out, transposed, r * th, c * tw)
    return out


def _make_tile(rng: np.random.Generator) -> np.ndarray | None:
    size = int(rng.integers(3, 5))
    n_colors = int(rng.integers(2, 4))
    palette = rng.choice(np.arange(3, 10, dtype=int), size=n_colors, replace=False)
    tile = rng.choice(palette, size=(size, size), replace=True).astype(int)

    for color in palette:
        if not np.any(tile == int(color)):
            rr = int(rng.integers(0, size))
            cc = int(rng.integers(0, size))
            tile[rr, cc] = int(color)

    if len(np.unique(tile)) < 2:
        return None
    if np.array_equal(tile, tile.T):
        return None
    return tile


def _make_control(rng: np.random.Generator, max_side: int) -> np.ndarray | None:
    h = int(rng.integers(3, max_side + 1))
    w = int(rng.integers(3, max_side + 1))
    control = rng.choice(np.array([0, 1, 2], dtype=int), size=(h, w), p=[0.28, 0.36, 0.36])

    if np.count_nonzero(control == 0) < 1:
        return None
    if np.count_nonzero(control == 1) < 2 or np.count_nonzero(control == 2) < 2:
        return None
    if not np.any(control[0, :] != 0):
        return None
    if not np.any(control[-1, :] != 0):
        return None
    if not np.any(control[:, 0] != 0):
        return None
    if not np.any(control[:, -1] != 0):
        return None
    return control


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _make_input(tile: np.ndarray, control: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    th, tw = tile.shape
    ch, cw = control.shape
    min_h = max(th, ch) + 4
    min_w = max(tw, cw) + 4
    canvas_h = int(rng.integers(min_h, 19))
    canvas_w = int(rng.integers(min_w, 19))
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    gap = 2
    for _ in range(200):
        tr = int(rng.integers(1, canvas_h - th))
        tc = int(rng.integers(1, canvas_w - tw))
        cr = int(rng.integers(1, canvas_h - ch))
        cc = int(rng.integers(1, canvas_w - cw))
        tile_box = (tr, tc, tr + th - 1, tc + tw - 1)
        control_box = (cr, cc, cr + ch - 1, cc + cw - 1)
        if _boxes_separated(tile_box, control_box, gap):
            _paste(grid, tile, tr, tc)
            _paste(grid, control, cr, cc)
            return grid
    return None


def _expected_output(tile: np.ndarray, control: np.ndarray) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = control.shape
    out = np.zeros((ch * th, cw * tw), dtype=int)
    tile_t = tile.T.copy()
    for r in range(ch):
        for c in range(cw):
            if control[r, c] == 1:
                _paste(out, tile, r * th, c * tw)
            elif control[r, c] == 2:
                _paste(out, tile_t, r * th, c * tw)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    tile = _make_tile(rng)
    if tile is None:
        return None

    size = tile.shape[0]
    max_side = min(6, 30 // size)
    control = _make_control(rng, max_side)
    if control is None:
        return None

    if max(control.shape[0] * size, control.shape[1] * size) > 30:
        return None

    grid = _make_input(tile, control, rng)
    if grid is None:
        return None

    expected = _expected_output(tile, control)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
