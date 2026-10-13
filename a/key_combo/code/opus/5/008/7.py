from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, paste


BG = 0
FRAME = 5
SEP = 6
PALETTE = [1, 2, 3, 4, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_RETRIES = 400


def _is_unit_sprite(patch: np.ndarray) -> bool:
    if patch.shape != (3, 3):
        return False
    colors = sorted(int(v) for v in np.unique(patch) if int(v) != BG)
    if len(colors) != 2:
        return False
    outer = int(patch[0, 0])
    if outer == BG or int(patch[1, 1]) == outer:
        return False
    perimeter = [
        patch[0, 0],
        patch[0, 1],
        patch[0, 2],
        patch[1, 0],
        patch[1, 2],
        patch[2, 0],
        patch[2, 1],
        patch[2, 2],
    ]
    return all(int(v) == outer for v in perimeter)


def _find_separator_col(grid: np.ndarray) -> int:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(sep_cols) == 1, "expected exactly one full-height separator column"
    return int(sep_cols[0])


def _parse_bars(canvas: np.ndarray) -> list[dict]:
    bars: list[dict] = []
    for r in range(canvas.shape[0]):
        nz = np.flatnonzero(canvas[r] != BG)
        if len(nz) == 0:
            continue
        c0, c1 = int(nz[0]), int(nz[-1])
        assert len(nz) == c1 - c0 + 1, "bar row must be one contiguous segment"
        seg = canvas[r, c0 : c1 + 1]
        assert len(seg) >= 3, "bar must have border-fill-border structure"
        border = int(seg[0])
        fill = int(seg[1])
        assert border != BG and fill != BG and border != fill
        assert int(seg[-1]) == border
        assert np.all(seg[1:-1] == fill), "bar interior must be a single fill color"
        bars.append({"r": int(r), "c0": c0, "c1": c1, "fill": fill, "border": border})
    assert 2 <= len(bars) <= 3, "expected 2-3 horizontal bars"
    return bars


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sep_c = _find_separator_col(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]

    comps = find_components(left, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    sprite_candidates = []
    frame_candidates = []
    for comp in comps:
        patch = crop_bbox(left, comp["bbox"])
        colors = {int(v) for v in np.unique(patch) if int(v) != BG}
        if _is_unit_sprite(patch):
            sprite_candidates.append(
                {
                    "bbox": tuple(int(v) for v in comp["bbox"]),
                    "patch": patch,
                    "outer": int(patch[0, 0]),
                    "inner": int(patch[1, 1]),
                }
            )
        elif colors == {FRAME}:
            frame_candidates.append({"bbox": tuple(int(v) for v in comp["bbox"]), "patch": patch})

    assert len(sprite_candidates) == 1, "expected exactly one 3x3 two-tone sprite"
    assert len(frame_candidates) == 1, "expected exactly one frame template"

    sprite = sprite_candidates[0]
    frame_patch = frame_candidates[0]["patch"]
    active = frame_patch == FRAME
    assert active.any(), "frame template must contain active cells"
    frame_comps = find_components(
        active.astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[1],
        return_format="dict",
    )
    assert len(frame_comps) == 1, "active frame cells must be 4-connected"

    bars = _parse_bars(right)
    matching = [bar for bar in bars if int(bar["fill"]) == int(sprite["outer"])]
    assert len(matching) == 1, "expected exactly one color-matching bar"
    return {
        "grid": g,
        "sep_c": sep_c,
        "left": left,
        "right": right,
        "sprite": sprite["patch"],
        "outer": int(sprite["outer"]),
        "inner": int(sprite["inner"]),
        "frame_active": active,
        "bars": bars,
        "match_bar": matching[0],
    }


def _tile_sprite(sprite: np.ndarray, active: np.ndarray) -> np.ndarray:
    fh, fw = active.shape
    out = np.zeros((fh * sprite.shape[0], fw * sprite.shape[1]), dtype=int)
    for r in range(fh):
        for c in range(fw):
            if bool(active[r, c]):
                paste(out, sprite, r * sprite.shape[0], c * sprite.shape[1], transparent=BG)
    return out


def _recolor_composite(comp: np.ndarray, outer: int, inner: int, new_outer: int, new_inner: int) -> np.ndarray:
    out = np.array(comp, dtype=int, copy=True)
    out[comp == int(outer)] = int(new_outer)
    out[comp == int(inner)] = int(new_inner)
    return out


def _dock_patch(canvas_shape: tuple[int, int], patch: np.ndarray, bar: dict) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    bar_w = int(bar["c1"] - bar["c0"] + 1)
    place_r = int(bar["r"] - patch.shape[0])
    place_c = int(bar["c0"] + (bar_w - patch.shape[1]) // 2)
    assert place_r >= 0 and place_c >= 0 and place_c + patch.shape[1] <= canvas_shape[1]
    paste(out, patch, place_r, place_c, transparent=BG)
    return out


def _solve_scene(scene: dict) -> np.ndarray:
    comp = _tile_sprite(scene["sprite"], scene["frame_active"])
    transformed = _recolor_composite(
        comp,
        scene["outer"],
        scene["inner"],
        scene["match_bar"]["border"],
        scene["match_bar"]["fill"],
    )
    return _dock_patch(scene["right"].shape, transformed, scene["match_bar"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a two-tone unit sprite into a grey frame, dock the enlarged composite above the matching color bar, and swap its colors from the bar code.

    Input structure:
    - Black background with one full-height magenta separator column splitting a left panel from a right canvas.
    - The left panel contains a 3x3 two-tone unit sprite on top and a grey active/inactive frame template below it.
    - The right canvas contains 2-3 one-row horizontal bars near the bottom; each bar has a fill color in the middle and a different border color at both ends.

    Rule:
    - Expand the frame by replacing every active grey cell with one copy of the 3x3 unit sprite, preserving the sprite's outer and inner colors.
    - Find the unique bar whose fill color equals the unit sprite's outer color.
    - Recolor the enlarged composite so sprite outer cells become that bar's border color and sprite inner cells become that bar's fill color.
    - Output only the right-canvas columns, with the recolored composite horizontally centered on the matching bar and placed so its bottom row sits exactly one row above the bar.

    Key insight:
    - The sprite's outer color is only a lookup key: it identifies the target bar, then gets inverted into that bar's border color after the frame tiling has determined the final composite shape.
    """
    return _solve_scene(_parse_scene(grid))


def _neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_connected_cells(h: int, w: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for cell in cells:
            frontier.update(_neighbors(cell, h, w))
        frontier.difference_update(cells)
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _random_frame_mask(h: int, w: int) -> np.ndarray | None:
    full = h * w
    lo = max(3, h + w - 2)
    hi = full - 1
    if lo > hi:
        return None
    for _ in range(120):
        area = random.randint(lo, hi)
        cells = _random_connected_cells(h, w, area)
        if cells is None:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if min(rs) != 0 or max(rs) != h - 1 or min(cs) != 0 or max(cs) != w - 1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask.all():
            continue
        return mask
    return None


def _sprite_patch(outer: int, inner: int) -> np.ndarray:
    patch = np.full((3, 3), int(outer), dtype=int)
    patch[1, 1] = int(inner)
    return patch


def _build_input(spec: dict) -> np.ndarray:
    grid = np.zeros((spec["H"], spec["W"]), dtype=int)
    grid[:, spec["sep_c"]] = SEP

    sprite = _sprite_patch(spec["outer"], spec["inner"])
    paste(grid[:, : spec["sep_c"]], sprite, spec["sprite_r"], spec["sprite_c"], transparent=BG)

    frame_patch = np.where(spec["frame_mask"], FRAME, BG).astype(int)
    paste(grid[:, : spec["sep_c"]], frame_patch, spec["frame_r"], spec["frame_c"], transparent=BG)

    for row, fill, border in zip(spec["bar_rows"], spec["fills"], spec["borders"]):
        grid[row, spec["sep_c"] + 1 :] = fill
        grid[row, spec["sep_c"] + 1] = border
        grid[row, -1] = border
    return grid


def _wrong_shortcuts_differ(scene: dict, expected: np.ndarray) -> None:
    match_bar = scene["match_bar"]
    swapped_sprite = _recolor_composite(
        scene["sprite"],
        scene["outer"],
        scene["inner"],
        match_bar["border"],
        match_bar["fill"],
    )
    wrong_single = _dock_patch(scene["right"].shape, swapped_sprite, match_bar)
    assert not np.array_equal(expected, wrong_single), "frame tiling must matter"

    full_rect = _tile_sprite(scene["sprite"], np.ones_like(scene["frame_active"], dtype=bool))
    wrong_rect = _dock_patch(
        scene["right"].shape,
        _recolor_composite(full_rect, scene["outer"], scene["inner"], match_bar["border"], match_bar["fill"]),
        match_bar,
    )
    assert not np.array_equal(expected, wrong_rect), "inactive frame cells must matter"

    unswapped = _dock_patch(scene["right"].shape, _tile_sprite(scene["sprite"], scene["frame_active"]), match_bar)
    assert not np.array_equal(expected, unswapped), "color swap must matter"


def _make_spec() -> dict | None:
    fh = random.randint(2, 4)
    fw = random.randint(2, 4)
    frame_mask = _random_frame_mask(fh, fw)
    if frame_mask is None:
        return None

    comp_h = fh * 3
    comp_w = fw * 3

    left_w = random.randint(max(5, fw + 2), min(7, fw + 3))
    max_right_w = 30 - left_w - 1
    min_right_w = max(11, comp_w + 2)
    right_choices = [w for w in range(min_right_w, max_right_w + 1) if w % 2 == comp_w % 2]
    if not right_choices:
        return None
    right_w = random.choice(right_choices)

    min_h = max(comp_h + 3, 18)
    max_h = min(30, comp_h + 10)
    if min_h > max_h:
        return None
    H = random.randint(min_h, max_h)
    W = left_w + 1 + right_w
    sep_c = left_w

    bar_count = random.randint(2, 3)
    band_h = random.randint(max(3, bar_count), 4)
    band_rows = list(range(H - band_h, H))
    bar_rows = sorted(random.sample(band_rows, bar_count))

    gap_after_sprite = random.randint(1, 3)
    sprite_r = random.randint(1, 2)
    sprite_c = random.randint(1, left_w - 4)
    frame_r_min = sprite_r + 3 + gap_after_sprite
    frame_r_max = min(H - fh - 1, bar_rows[0] - fh - 2)
    if frame_r_min > frame_r_max:
        return None
    frame_r = random.randint(frame_r_min, frame_r_max)
    frame_c = random.randint(1, left_w - fw)

    needed = 2 * bar_count + 1
    if needed > len(PALETTE):
        return None
    colors = random.sample(PALETTE, needed)
    outer = colors[0]
    inner = colors[1]
    other_fills = colors[2 : 2 + bar_count - 1]
    borders = colors[2 + bar_count - 1 :]
    match_idx = random.randrange(bar_count)
    fills = []
    other_it = iter(other_fills)
    for idx in range(bar_count):
        fills.append(outer if idx == match_idx else next(other_it))

    return {
        "H": H,
        "W": W,
        "sep_c": sep_c,
        "right_w": right_w,
        "frame_mask": frame_mask,
        "outer": outer,
        "inner": inner,
        "bar_rows": bar_rows,
        "fills": fills,
        "borders": borders,
        "match_idx": match_idx,
        "sprite_r": sprite_r,
        "sprite_c": sprite_c,
        "frame_r": frame_r,
        "frame_c": frame_c,
    }


def _validate_generated(grid: np.ndarray, spec: dict) -> None:
    scene = _parse_scene(grid)
    assert scene["right"].shape == (spec["H"], spec["right_w"])
    assert scene["outer"] == int(spec["outer"])
    assert scene["inner"] == int(spec["inner"])
    assert np.array_equal(scene["frame_active"], spec["frame_mask"])
    assert [bar["r"] for bar in scene["bars"]] == spec["bar_rows"]
    assert [bar["fill"] for bar in scene["bars"]] == spec["fills"]
    assert [bar["border"] for bar in scene["bars"]] == spec["borders"]

    expected = _solve_scene(scene)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert actual.shape == (spec["H"], spec["right_w"])
    assert np.any(actual != BG)
    _wrong_shortcuts_differ(scene, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        spec = _make_spec()
        if spec is None:
            continue
        grid = _build_input(spec)
        try:
            _validate_generated(grid, spec)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
