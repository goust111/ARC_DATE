# Combo variant of dbff022c (index 37)
# combo_spec: D1/1+D8/2
# Name: 色值排序映射 + 外接矩形填充

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the fused variant:
# - Concrete trace: the original task reads an explicit 2xN/Nx2 edge palette and then fills only
#   enclosed holes. Here there is no palette at all. If the visible border colors are [1,2,3,4],
#   the implicit mapping is 1->4, 2->1, 3->2, 4->3. A sprite with border color 3 therefore fills
#   with 2. If that sprite is a 5x5 frame with a missing top-edge cell, the old logic would leave
#   the shape unchanged because the background leaks out; the new logic still fills every non-border
#   cell in the 5x5 bbox, including the missing top-edge cell.
# - Complexity calibration: using exactly four recurring border colors gives repeated support for
#   the hidden ranking rule without flooding the grid with too many colors. At least two sprites per
#   sample have border gaps, so D8/2 is visually active instead of collapsing back to hole filling.
# - Broken assumptions from the original solver: there is no edge legend to detect, no palette mask
#   to exclude, and no need to identify enclosed background components. The new solver instead assumes
#   that each non-background color marks a separated monochrome sprite whose full bbox can be claimed.
# - Generator guarantees for those assumptions: every sprite gets a unique border color, sprites are
#   separated by at least one background cell after bbox expansion, and nothing except that sprite's
#   background lives inside its bbox. This makes bbox painting unambiguous.
# - Observability and cleanliness: the same small color family recurs across samples, so the cyclic
#   border->fill mapping can be inferred from examples. Reserving each full bbox keeps later fills
#   from touching unrelated sprites, even when a border gap causes fill to appear on the bbox edge.


_GRID_H_MIN, _GRID_H_MAX = 9, 13
_GRID_W_MIN, _GRID_W_MAX = 11, 15
_NUM_SPRITES = 4
_BORDER_COLORS = (1, 2, 3, 4)
_SPRITE_H_CHOICES = (4, 5)
_SPRITE_W_CHOICES = (4, 5)
_GAP_SPRITES_MIN = 2
_SECOND_GAP_PROB = 0.35
_INTERNAL_BAR_PROB = 0.35
_MAX_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _coords_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sorted_color_mapping(g: np.ndarray, bg: int) -> dict[int, int]:
    colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    if len(colors) < 2:
        return {}
    return {col: colors[(i - 1) % len(colors)] for i, col in enumerate(colors)}


def _sprite_components(g: np.ndarray, bg: int) -> list[tuple[int, np.ndarray, tuple[int, int, int, int]]]:
    sprites: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    for col in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for pts in components_mask(g == col, connectivity=4):
            sprites.append((col, pts, _coords_bbox(pts)))
    return sprites


def _fill_bbox_non_border(out: np.ndarray, bbox: tuple[int, int, int, int], border: int, fill: int) -> None:
    r0, c0, r1, c1 = bbox
    box = out[r0 : r1 + 1, c0 : c1 + 1]
    box[box != border] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow-looking sprite by ranking the border colors and then filling its whole bbox remainder.

    Input structure:
    - The boundary is mostly background, with several separated monochrome frame-like sprites inside the grid.
    - There is no explicit edge legend; the only non-background colors are the sprites' border colors.
    - Inside each sprite's bounding box, the non-border cells are background gaps or interior cells.

    Rule:
    - Collect all non-background colors in the input and sort them by color id.
    - Map each border color to the previous color in that sorted list, wrapping the smallest color to the largest.
    - For every sprite component, take its bounding box and paint every cell inside that box that is not already the border color with the mapped fill color.

    Key insight:
    - The border->fill correspondence is implicit: it comes from the global ordering of the visible colors rather than from a drawn palette.
    - The filled area is not the topological hole set; it is the entire bounding-box remainder, so border openings still get painted.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping = _sorted_color_mapping(g, bg)
    if not mapping:
        return g.copy()

    out = g.copy()
    for border, pts, bbox in _sprite_components(g, bg):
        if border not in mapping:
            continue
        _fill_bbox_non_border(out, bbox, border, int(mapping[border]))
    return out


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _edge_gap_cells(mask: np.ndarray, side: str, gap_len: int, start: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        return [(0, c) for c in range(start, start + gap_len)]
    if side == "bottom":
        return [(h - 1, c) for c in range(start, start + gap_len)]
    if side == "left":
        return [(r, 0) for r in range(start, start + gap_len)]
    return [(r, w - 1) for r in range(start, start + gap_len)]


def _carve_gap(mask: np.ndarray, rng: random.Random) -> bool:
    h, w = mask.shape
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides:
        span = w if side in ("top", "bottom") else h
        inner = span - 2
        if inner <= 0:
            continue
        max_len = min(2, inner)
        lengths = list(range(max_len, 0, -1))
        for gap_len in lengths:
            max_start = span - 1 - gap_len
            starts = list(range(1, max_start + 1))
            rng.shuffle(starts)
            for start in starts:
                cells = _edge_gap_cells(mask, side, gap_len, start)
                trial = mask.copy()
                for r, c in cells:
                    trial[r, c] = False
                if len(components_mask(trial, connectivity=4)) == 1:
                    mask[:, :] = trial
                    return True
    return False


def _add_internal_bar(mask: np.ndarray, rng: random.Random) -> None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return
    if rng.random() < 0.5:
        c = rng.randint(1, w - 2)
        mask[1 : h - 1, c] = True
    else:
        r = rng.randint(1, h - 2)
        mask[r, 1 : w - 1] = True


def _has_boundary_gap(mask: np.ndarray) -> bool:
    return bool(
        np.any(~mask[0, :])
        or np.any(~mask[-1, :])
        or np.any(~mask[:, 0])
        or np.any(~mask[:, -1])
    )


def _make_sprite(rng: random.Random, force_gap: bool) -> np.ndarray | None:
    h = rng.choice(_SPRITE_H_CHOICES)
    w = rng.choice(_SPRITE_W_CHOICES)
    mask = _frame_mask(h, w)
    if rng.random() < _INTERNAL_BAR_PROB:
        _add_internal_bar(mask, rng)

    gap_goal = 1 if force_gap else 0
    if rng.random() < _SECOND_GAP_PROB:
        gap_goal += 1
    for _ in range(gap_goal):
        _carve_gap(mask, rng)

    if force_gap and not _has_boundary_gap(mask):
        return None
    if not np.any(~mask):
        return None
    return mask


def _find_valid_positions(H: int, W: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, H - sh):
        for c0 in range(1, W - sw):
            if occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite_pair(
    inp: np.ndarray,
    out: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    border: int,
    fill: int,
) -> None:
    sh, sw = mask.shape
    out_box = out[r0 : r0 + sh, c0 : c0 + sw]
    inp_box = inp[r0 : r0 + sh, c0 : c0 + sw]
    out_box[:, :] = fill
    inp_box[:, :] = 0
    out_box[mask] = border
    inp_box[mask] = border


def _expected_mapping(colors: tuple[int, ...]) -> dict[int, int]:
    ordered = sorted(int(c) for c in colors)
    return {col: ordered[(i - 1) % len(ordered)] for i, col in enumerate(ordered)}


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    mapping = _expected_mapping(_BORDER_COLORS)

    force_gap_flags = [True] * _GAP_SPRITES_MIN + [False] * (_NUM_SPRITES - _GAP_SPRITES_MIN)
    rng.shuffle(force_gap_flags)
    shapes: list[np.ndarray] = []
    for force_gap in force_gap_flags:
        mask = _make_sprite(rng, force_gap)
        if mask is None:
            return None
        shapes.append(mask)

    order = list(range(_NUM_SPRITES))
    rng.shuffle(order)
    colors = [_BORDER_COLORS[i] for i in order]
    rng.shuffle(shapes)

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    for mask, border in zip(shapes, colors, strict=False):
        positions = _find_valid_positions(H, W, occ, mask)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite_pair(inp, out, r0, c0, mask, int(border), int(mapping[int(border)]))
        sh, sw = mask.shape
        occ[r0 : r0 + sh, c0 : c0 + sw] = True

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
