# Combo variant of bf45cf4b (index 16)
# combo_spec: D10/1+D14/0
# Name: 边框编码放大倍率
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and small trace:
#    Original:
#      - control = monochrome mask with holes
#      - tile size s is also the scale factor
#      - every filled control cell becomes one s x s copy of the tile
#    Fused:
#      - control keeps the same basic role, but some filled pixels on its top border
#        are recolored as marker pixels
#      - marker count k (2, 3, or 4) is the true scale factor
#      - markers still count as filled mask pixels; they annotate the sprite rather
#        than creating a separate frame
#      - the source tile is adapted to k x k before stamping:
#          * if tile is bigger than k, keep its top-left k x k crop
#          * if tile is smaller than k, periodically repeat it and crop to k x k
#    Example:
#      - control bbox mask row starts with [marker, marker, body, body]
#      - marker count = 2, so each filled mask cell becomes a 2 x 2 stamp
#      - if the tile is 3 x 3, solver uses only its top-left 2 x 2 corner
#      - if the tile is 2 x 2 and k = 4, solver repeats it to a 4 x 4 checker-like
#        stamp before placement
#
# b) Complexity calibration:
#    - k is limited to {2, 3, 4}, so the code is visible from a few examples.
#    - tile size is chosen independently from {2, 3, 4, 5} and forced to differ
#      from k, preventing collapse back to the original task.
#    - marker pixels form one contiguous run on the top border; this is strong enough
#      for human detection and restrictive enough for reliable parsing.
#
# c) Original assumptions that break:
#    - "control is monochrome" is false: it now has body color + marker color.
#    - "more colors means tile" is false: both components can be multicolor.
#    - "tile.shape determines output scale" is false: the decoded marker count does.
#    - "mask = cells of one chosen control color" is false: markers also count as
#      filled mask cells because they are only annotations on the sprite border.
#
# d) Placement legality:
#    - generator places exactly two disconnected 8-connected components with >= 2
#      background cells of separation and at least a 1-cell outer margin.
#
# e/f/g) Visual observability and neatness:
#    - the control sprite has a full rectangular border, so the marked top-edge run
#      is easy to notice and interpret as an encoded parameter.
#    - tile colors are disjoint from control colors.
#    - only visible spatial cues are used: border position, contiguity, and count.
#
# New solver assumptions guaranteed by the generator:
#    - exactly one component has exactly two non-bg colors, a fully filled rectangle
#      border, and a contiguous top-edge marker run of length 2..4
#    - the other component is the square tile
#    - the control sprite contains at least one bg hole inside its bbox

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


def _fit_tile(tile: np.ndarray, scale: int) -> np.ndarray:
    """Adapt a source tile to an exact scale x scale stamp by repeat-then-crop."""
    k = int(scale)
    th, tw = map(int, tile.shape)
    reps_r = max(1, (k + th - 1) // th)
    reps_c = max(1, (k + tw - 1) // tw)
    tiled = np.tile(tile, (reps_r, reps_c))
    return tiled[:k, :k].copy()


def _is_contiguous_run(cols: np.ndarray) -> bool:
    if cols.size == 0:
        return False
    return bool(np.array_equal(cols, np.arange(int(cols[0]), int(cols[0]) + cols.size)))


def _extract_control(patch: np.ndarray, bg: int) -> dict | None:
    """
    Recognize the modified control sprite.

    Required structure:
    - exactly two foreground colors: body + markers
    - full rectangular border is filled
    - markers appear only on the top border and form one contiguous run
    - marker count is 2..4 and encodes the output stamp size
    - interior contains only body or bg, with at least one bg hole
    """
    h, w = map(int, patch.shape)
    if h < 5 or w < 5:
        return None

    cols = colors_present(patch, exclude={bg})
    if len(cols) != 2:
        return None

    inner = patch[1:-1, 1:-1]
    if inner.size == 0 or not np.any(inner == int(bg)):
        return None

    for body in cols:
        marker = next(col for col in cols if col != body)
        marker_mask = patch == int(marker)
        if not marker_mask.any():
            continue
        if np.any(marker_mask[1:, :]):
            continue

        marker_cols = np.flatnonzero(marker_mask[0])
        k = int(marker_cols.size)
        if k not in (2, 3, 4) or not _is_contiguous_run(marker_cols):
            continue

        if np.any(patch[0, marker_cols] != int(marker)):
            continue
        body_top = np.ones(w, dtype=bool)
        body_top[marker_cols] = False
        if np.any(patch[0, body_top] != int(body)):
            continue
        if np.any(patch[-1, :] != int(body)):
            continue
        if np.any(patch[1:-1, 0] != int(body)):
            continue
        if np.any(patch[1:-1, -1] != int(body)):
            continue
        if not np.all(np.isin(inner, [int(bg), int(body)])):
            continue

        mask = patch != int(bg)
        return {
            "body": int(body),
            "marker": int(marker),
            "scale": k,
            "mask": mask,
        }

    return None


def _decompose(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        infos.append({"patch": patch, "control": _extract_control(patch, bg)})

    ctrl_candidates = [idx for idx, info in enumerate(infos) if info["control"] is not None]
    assert len(ctrl_candidates) == 1

    ctrl_idx = ctrl_candidates[0]
    tile_idx = 1 - ctrl_idx
    ctrl = infos[ctrl_idx]["control"]
    tile = infos[tile_idx]["patch"]
    return tile, ctrl["mask"], int(ctrl["scale"]), bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top-border marker count from the control sprite, adapt the tile to
    that size, and stamp it at every filled mask cell of the control bbox.
    """
    tile, mask, scale, bg = _decompose(grid)
    stamp = _fit_tile(tile, scale)
    ch, cw = map(int, mask.shape)
    sh, sw = map(int, stamp.shape)
    out = np.full((ch * sh, cw * sw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, stamp, r * sh, c * sw)
    return out


def _dihedral_orbits(size: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    s = int(size)
    for r in range(s):
        for c in range(s):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (c, s - 1 - r),
                (s - 1 - r, s - 1 - c),
                (s - 1 - c, r),
                (r, s - 1 - c),
                (s - 1 - r, c),
                (c, r),
                (s - 1 - c, s - 1 - r),
            }
            cells = sorted(orbit)
            seen.update(cells)
            out.append(cells)
    return out


def _make_tile(tile_colors: list[int], size: int, rng: np.random.Generator) -> np.ndarray:
    chosen_n = int(rng.integers(2, min(4, len(tile_colors) + 1)))
    chosen = [int(x) for x in rng.choice(tile_colors, size=chosen_n, replace=False)]
    tile = np.full((size, size), int(chosen[0]), dtype=int)

    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)
    for i, cells in enumerate(orbits):
        color = int(chosen[i % len(chosen)])
        for r, c in cells:
            tile[r, c] = color

    if len(set(int(x) for x in tile.flatten())) < 2:
        tile[0, 0] = int(chosen[-1])
    return tile


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _make_control_mask(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(5, 7))
    w = int(rng.integers(5, 8))
    if h == w and rng.random() < 0.7:
        w = 7 if w < 7 else 5

    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    area = h * w
    min_fill = int(mask.sum())
    max_fill = min(area - 1, int(np.floor(0.72 * area)))
    if max_fill < min_fill:
        max_fill = min_fill
    target = int(rng.integers(min_fill, max_fill + 1))

    while int(mask.sum()) < target:
        frontier = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        pick = frontier[int(rng.integers(0, len(frontier)))]
        mask[pick] = True

    return mask


def _make_control_patch(
    bg: int,
    body: int,
    marker: int,
    scale: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    while True:
        mask = _make_control_mask(rng)
        h, w = map(int, mask.shape)
        if w < scale:
            continue

        patch = np.full((h, w), int(bg), dtype=int)
        patch[mask] = int(body)
        start = int(rng.integers(0, w - scale + 1))
        patch[0, start : start + scale] = int(marker)

        info = _extract_control(patch, bg)
        if info is None:
            continue
        return patch, info["mask"]


def _compose_output(tile: np.ndarray, mask: np.ndarray, scale: int, bg: int) -> np.ndarray:
    stamp = _fit_tile(tile, scale)
    sh, sw = map(int, stamp.shape)
    ch, cw = map(int, mask.shape)
    out = np.full((ch * sh, cw * sw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, stamp, r * sh, c * sw)
    return out


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray | None:
    horizontal = bool(rng.integers(0, 2))
    gap = 2
    border = 1

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        min_h = max(lh, rh) + 2 * border
        min_w = lw + rw + gap + 2 * border
        if min_h > 18 or min_w > 18:
            return None
        H = int(rng.integers(min_h, 19))
        W = int(rng.integers(min_w, 19))
        r_left = int(rng.integers(border, H - border - lh + 1))
        r_right = int(rng.integers(border, H - border - rh + 1))
        c_left = int(rng.integers(border, W - border - (lw + rw + gap) + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - border - rw + 1))
        if left_obj is tile:
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if rng.random() < 0.5:
            top_obj, bot_obj = tile, ctrl
        else:
            top_obj, bot_obj = ctrl, tile
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        min_h = th + bh + gap + 2 * border
        min_w = max(tw, bw) + 2 * border
        if min_h > 18 or min_w > 18:
            return None
        H = int(rng.integers(min_h, 19))
        W = int(rng.integers(min_w, 19))
        c_top = int(rng.integers(border, W - border - tw + 1))
        c_bot = int(rng.integers(border, W - border - bw + 1))
        r_top = int(rng.integers(border, H - border - (th + bh + gap) + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - border - bh + 1))
        if top_obj is tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    body = int(palette[1])
    marker = int(palette[2])
    tile_colors = [int(x) for x in palette[3:]]

    scale = int(rng.choice(np.array([2, 3, 4], dtype=int)))
    tile_size_choices = [s for s in (2, 3, 4, 5) if s != scale]
    tile_size = int(rng.choice(np.array(tile_size_choices, dtype=int)))
    tile = _make_tile(tile_colors, tile_size, rng)
    if _extract_control(tile, bg) is not None:
        return None

    ctrl, mask = _make_control_patch(bg, body, marker, scale, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    if inp is None:
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    infos = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        infos.append(_extract_control(patch, bg) is not None)
    if sum(infos) != 1:
        return None

    expected = _compose_output(tile, mask, scale, bg)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid fused input")
