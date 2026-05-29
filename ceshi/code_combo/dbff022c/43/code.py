# Combo variant of dbff022c (index 43)
# combo_spec: D6/2+D10/0
# Name: 精灵裁剪排列输出 + 严格一一对应

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original dbff022c keeps the whole canvas and only fills enclosed holes in place.
#    Here the same palette still tells us which border color maps to which fill color,
#    but after filling we discard the original canvas layout and output only the filled
#    sprites, each cropped to its own bbox and packed in palette-entry order. Example:
#    if the palette border row is [4, 7, 2], and the three scattered hollow sprites use
#    border colors 7, 2, 4 respectively, the original task would return the same full
#    canvas with three holes filled. The fused task instead fills all three, crops the
#    4-sprite, then the 7-sprite, then the 2-sprite, and packs those crops into a fresh
#    strip with 1-cell gaps and a 1-cell outer border.
# b) Complexity calibration:
#    The generator keeps 2-4 sprites and 2-4 palette entries. That is enough repeated
#    structure for humans to see palette-order packing, but not so many colors or objects
#    that the correspondence becomes noisy. Fill colors are chosen disjoint from border
#    colors to keep the visual cue clean.
# c) Broken original assumptions:
#    - "mapping may have extras" no longer holds; main now asserts that every palette key
#      has a matching sprite crop.
#    - "output shape equals input shape" no longer holds; main always repacks crops.
#    - "sprite detection can ignore order" no longer holds; palette reading order now
#      controls output order.
#    The generator enforces those assumptions by creating exactly one sprite per key.
# d) Placement legality:
#    Input placement still uses padded occupancy, so sprites do not overlap each other or
#    the palette, and enclosed holes never touch the outer grid boundary.
# e) Visual observability:
#    The packed output is visibly smaller and reorganized, so D6/2 is obvious. Palette
#    order matters because input positions are random but output order follows the legend.
# f) Visual cleanliness:
#    Output packing uses only one level of structure: cropped sprite, 1-cell gap, 1-cell
#    border. No overlays or extra markers are added.
# g) Inferability of color changes:
#    Every border color in the palette appears on exactly one sprite, and every such
#    sprite is filled. Because there are no distractor sprites or unused keys, the solver
#    only needs to infer border->fill mapping plus palette-order packing.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top", "bottom", "left", "right"}
      - for top/bottom: start/end are column indices [start, end)
      - for left/right: start/end are row indices [start, end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """Ensure palette colors do not leak into adjacent same-colored sprite cells."""
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_horizontal(side: str, outer_row: np.ndarray, inner_row: np.ndarray) -> None:
        nonlocal best
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer_row[c0:c1]
                seg_inner = inner_row[c0:c1]
                outer_cols = set(map(int, np.unique(seg_outer)))
                inner_cols = set(map(int, np.unique(seg_inner)))
                outer_cols.discard(bg)
                inner_cols.discard(bg)
                if len(outer_cols) < 2 or len(inner_cols) < 2:
                    continue
                union_cols = outer_cols | inner_cols
                if len(union_cols) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated(side, c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(border_keys), len(union_cols), -(c1 - c0), nonbg)
                cand = (score, side, c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_vertical(side: str, outer_col: np.ndarray, inner_col: np.ndarray) -> None:
        nonlocal best
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer_col[r0:r1]
                seg_inner = inner_col[r0:r1]
                outer_cols = set(map(int, np.unique(seg_outer)))
                inner_cols = set(map(int, np.unique(seg_inner)))
                outer_cols.discard(bg)
                inner_cols.discard(bg)
                if len(outer_cols) < 2 or len(inner_cols) < 2:
                    continue
                union_cols = outer_cols | inner_cols
                if len(union_cols) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated(side, r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(border_keys), len(union_cols), -(r1 - r0), nonbg)
                cand = (score, side, r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    if h >= 2:
        consider_horizontal("top", g[0, :], g[1, :])
        consider_horizontal("bottom", g[h - 1, :], g[h - 2, :])
    if w >= 2:
        consider_vertical("left", g[:, 0], g[:, 1])
        consider_vertical("right", g[:, w - 1], g[:, w - 2])

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _ordered_palette_entries(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, int]], np.ndarray, str]:
    """Return ordered palette entries, palette mask, and palette side."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    entries: list[tuple[int, int]] = []
    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        border = int(b)
        fill = int(f)
        if border == bg:
            continue
        entries.append((border, fill))

    assert entries, "_ordered_palette_entries requires at least one palette entry"
    return entries, pal_mask, side


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Fill each enclosed bg region by the mapped border color that most strongly surrounds it.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(
            r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts
        )
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if pal_mask[rr, cc]:
                    continue
                v = int(out[rr, cc])
                if v == bg:
                    continue
                adj_counts[v] = adj_counts.get(v, 0) + 1

        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in mapping:
                continue
            if cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        out[tuple(np.array(pts).T)] = int(mapping[border])


def _crop_sprite_by_border(
    filled: np.ndarray, pal_mask: np.ndarray, border_color: int
) -> np.ndarray:
    """Crop one filled sprite using the bbox of its border-color cells."""
    mask = (filled == int(border_color)) & (~pal_mask)
    coords = np.argwhere(mask)
    assert coords.size > 0, f"Missing sprite for border color {border_color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return filled[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _pack_crops(
    crops: list[np.ndarray], bg: int, vertical: bool
) -> np.ndarray:
    """Pack crops into a 1-cell-bordered strip with 1-cell gaps."""
    assert crops, "_pack_crops requires at least one crop"
    if vertical:
        out_h = 2 + sum(int(c.shape[0]) for c in crops) + (len(crops) - 1)
        out_w = 2 + max(int(c.shape[1]) for c in crops)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        r = 1
        for crop in crops:
            h, w = crop.shape
            out[r : r + h, 1 : 1 + w] = crop
            r += h + 1
        return out

    out_h = 2 + max(int(c.shape[0]) for c in crops)
    out_w = 2 + sum(int(c.shape[1]) for c in crops) + (len(crops) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    c = 1
    for crop in crops:
        h, w = crop.shape
        out[1 : 1 + h, c : c + w] = crop
        c += w + 1
    return out


def _packed_output_from_filled(
    filled: np.ndarray,
    bg: int,
    entries: list[tuple[int, int]],
    pal_mask: np.ndarray,
    side: str,
) -> np.ndarray:
    """Create the fused packed output from a filled full-canvas state."""
    crops = [_crop_sprite_by_border(filled, pal_mask, border) for border, _ in entries]
    vertical = side in {"left", "right"}
    return _pack_crops(crops, bg, vertical=vertical)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the edge palette, fill every matched hollow sprite, then repack only those sprites.

    Input structure:
    - The background is the dominant outer-border color.
    - A 2xN or Nx2 palette sits on one edge; the outer line lists unique sprite border colors
      in reading order and the inner line lists their fill colors.
    - Elsewhere are exactly N hollow single-color sprites, one per palette entry, scattered on
      the canvas with empty interiors.

    Rule:
    - Detect the palette and interpret it as an ordered border->fill mapping.
    - Fill each enclosed background region using the mapped fill color of its surrounding border.
    - For each palette entry in order, locate the corresponding sprite, crop its bbox from the
      filled canvas, and pack those crops into a new strip with 1-cell gaps and a 1-cell border.

    Key insight: The palette does two jobs at once: it tells you which fill color each border
    should receive, and it also tells you the order in which the independently cropped sprites
    must appear in the new output layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    entries, pal_mask, side = _ordered_palette_entries(g, bg)
    mapping = {border: fill for border, fill in entries}
    assert len(mapping) == len(entries), "Palette border keys must be unique"

    filled = g.copy()
    _fill_enclosed_bg_regions(filled, bg, mapping, pal_mask)
    return _packed_output_from_filled(filled, bg, entries, pal_mask, side)


# --------------- generator ---------------

_GRID_H_MIN, _GRID_H_MAX = 10, 16
_GRID_W_MIN, _GRID_W_MAX = 10, 16

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_SPRITE_H_MIN, _SPRITE_H_MAX = 3, 6
_SPRITE_W_MIN, _SPRITE_W_MAX = 3, 5
_SPRITE_DIM_SUM_CAP = 9
_SPRITE_CELL_MAX = 15

_P_HORIZ_WALL = 0.35
_P_VERT_WALL = 0.35
_P_BORDER_REMOVAL = 0.35
_P_DIAMOND = 0.2

_MAX_RETRIES = 400
_MAX_OUTPUT_SIDE = 30


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components inside the sprite mask that do not touch the bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped hollow sprite."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a random rotation and optional mirror to a boolean sprite mask."""
    k = rng.randint(0, 3)
    out = np.rot90(mask, k)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole."""
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        length = rng.randint(1, w - 2)
        start = rng.randint(1, w - length - 1)
        sprite[br, start : start + length] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        length = rng.randint(1, h - 2)
        start = rng.randint(1, h - length - 1)
        sprite[start : start + length, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for rr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[rr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite):
        return None
    if int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return _random_transform(sprite, rng)


def _choose_colors(
    rng: random.Random, n: int
) -> tuple[list[int], list[int], dict[int, int]]:
    """Choose disjoint border and fill colors."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n]
    fill_values = colors[n : 2 * n]
    mapping = dict(zip(border_keys, fill_values))
    return border_keys, fill_values, mapping


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the palette block for the chosen edge."""
    n = len(border_keys)
    if side in {"top", "bottom"}:
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer, :] = border_keys
        pal[inner, :] = fill_values
        return pal

    pal = np.zeros((n, 2), dtype=int)
    outer, inner = (0, 1) if side == "left" else (1, 0)
    pal[:, outer] = border_keys
    pal[:, inner] = fill_values
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> None:
    """Place the palette on the requested edge and reserve a padded occupancy box."""
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pw)
    elif side == "bottom":
        r0, c0 = h - ph, rng.randint(0, w - pw)
    elif side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    else:
        r0, c0 = rng.randint(0, h - ph), w - pw

    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements with 1-cell spacing and with holes away from the canvas boundary."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hole in holes
                for hr, hc in hole
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    border_color: int,
    fill_color: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render one sprite onto the grid."""
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = int(border_color)
    if fill_color is None:
        return
    for hole in holes:
        for hr, hc in hole:
            grid[r0 + int(hr), c0 + int(hc)] = int(fill_color)


def _sprite_patch(
    mask: np.ndarray, border_color: int, fill_color: int, bg: int
) -> np.ndarray:
    """Create the filled cropped patch for one sprite."""
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(border_color)
    for hole in _get_enclosed_holes(mask):
        for hr, hc in hole:
            patch[int(hr), int(hc)] = int(fill_color)
    return patch


def _random_sprite(rng: random.Random) -> np.ndarray | None:
    """Sample one hollow sprite shape."""
    h = rng.randint(_SPRITE_H_MIN, _SPRITE_H_MAX)
    w = rng.randint(_SPRITE_W_MIN, _SPRITE_W_MAX)
    if h + w > _SPRITE_DIM_SUM_CAP:
        w = max(_SPRITE_W_MIN, _SPRITE_DIM_SUM_CAP - h)
    if h == 3 and w == 3 and rng.random() < _P_DIAMOND:
        return _random_transform(_make_diamond(), rng)
    return _make_hollow_sprite(h, w, rng)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_n = min(_PALETTE_N_MAX, h if side in {"left", "right"} else w)
    if max_n < _PALETTE_N_MIN:
        return None
    n = rng.randint(_PALETTE_N_MIN, max_n)

    border_keys, fill_values, mapping = _choose_colors(rng, n)
    entries = list(zip(border_keys, fill_values, strict=False))

    shapes: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, ...], ...]] = set()
    for _ in range(n):
        mask = _random_sprite(rng)
        if mask is None:
            return None
        sig = tuple(tuple(int(v) for v in row) for row in mask.astype(int))
        signatures.add(sig)
        shapes.append(mask)
    if n >= 3 and len(signatures) < 2:
        return None

    filled_crops = [
        _sprite_patch(mask, border, fill, bg)
        for mask, (border, fill) in zip(shapes, entries, strict=False)
    ]
    expected = _pack_crops(filled_crops, bg, vertical=side in {"left", "right"})
    if max(expected.shape) > _MAX_OUTPUT_SIDE:
        return None

    palette = _build_palette(side, border_keys, fill_values)
    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, palette, side, rng)

    placed: list[tuple[int, int, np.ndarray]] = []
    order = list(range(n))
    rng.shuffle(order)
    for idx in order:
        mask = shapes[idx]
        border = border_keys[idx]
        fill = mapping[border]
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, border, fill, holes)
        placed.append((r0, c0, mask))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    inp = grid.copy()
    for r0, c0, mask in placed:
        for hole in _get_enclosed_holes(mask):
            for hr, hc in hole:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
