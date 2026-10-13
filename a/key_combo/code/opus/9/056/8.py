from __future__ import annotations

import random
import sys

import numpy as np

_TOOLBOX_DIR = "/home/lixinyu/snap/new/solvers"
if _TOOLBOX_DIR not in sys.path:
    sys.path.append(_TOOLBOX_DIR)

from arc_utils import Color, find_components, get_bg, paste


BG = Color.BLACK
RED = Color.RED
GREEN = Color.GREEN
CYAN = Color.TEAL

SPRITE_MASK = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)
SPRITE_H, SPRITE_W = map(int, SPRITE_MASK.shape)
INDICATOR_LOCAL = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400


def _sprite_patch(direction: str) -> np.ndarray:
    patch = np.zeros((SPRITE_H, SPRITE_W), dtype=int)
    patch[SPRITE_MASK] = GREEN
    rr, cc = INDICATOR_LOCAL[direction]
    patch[rr, cc] = RED
    return patch


def _cyan_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = CYAN
    return patch


def _direction_from_red(local_red: tuple[int, int], shape: tuple[int, int]) -> str:
    rr, cc = map(int, local_red)
    h, w = map(int, shape)
    candidates: list[str] = []
    if rr == 0:
        candidates.append("up")
    if rr == h - 1:
        candidates.append("down")
    if cc == 0:
        candidates.append("left")
    if cc == w - 1:
        candidates.append("right")
    assert len(candidates) == 1, "indicator must sit on exactly one bbox edge"
    return candidates[0]


def _extract_originals(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    comps = find_components(
        grid,
        bg=get_bg(grid),
        colors=[RED, GREEN],
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert comps, "expected at least one original sprite"

    originals: list[dict] = []
    ref_mask: np.ndarray | None = None
    for comp in sorted(comps, key=lambda item: tuple(map(int, item["bbox"]))):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != BG
        reds = np.argwhere(patch == RED)
        assert reds.shape == (1, 2), "each original sprite must contain exactly one red indicator"
        if ref_mask is None:
            ref_mask = mask.copy()
        else:
            assert mask.shape == ref_mask.shape and np.array_equal(mask, ref_mask), "all sprites must share one mask"
        local_red = tuple(map(int, reds[0]))
        originals.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top_left": (r0, c0),
                "mask": mask,
                "direction": _direction_from_red(local_red, patch.shape),
                "red_local": local_red,
                "red_global": (r0 + local_red[0], c0 + local_red[1]),
            }
        )

    assert ref_mask is not None
    return originals, ref_mask


def _paint_extension(out: np.ndarray, blockers: np.ndarray, original: dict) -> dict:
    r0, c0, r1, c1 = map(int, original["bbox"])
    red_r, red_c = map(int, original["red_global"])
    direction = str(original["direction"])
    band: set[tuple[int, int]] = set()
    red_line: set[tuple[int, int]] = set()
    stopped_by = "boundary"

    if direction == "up":
        for rr in range(r0 - 1, -1, -1):
            if np.any(blockers[rr, c0 : c1 + 1]):
                stopped_by = "cyan"
                break
            out[rr, c0 : c1 + 1] = GREEN
            out[rr, red_c] = RED
            for cc in range(c0, c1 + 1):
                band.add((rr, cc))
            red_line.add((rr, red_c))
    elif direction == "down":
        for rr in range(r1 + 1, out.shape[0]):
            if np.any(blockers[rr, c0 : c1 + 1]):
                stopped_by = "cyan"
                break
            out[rr, c0 : c1 + 1] = GREEN
            out[rr, red_c] = RED
            for cc in range(c0, c1 + 1):
                band.add((rr, cc))
            red_line.add((rr, red_c))
    elif direction == "left":
        for cc in range(c0 - 1, -1, -1):
            if np.any(blockers[r0 : r1 + 1, cc]):
                stopped_by = "cyan"
                break
            out[r0 : r1 + 1, cc] = GREEN
            out[red_r, cc] = RED
            for rr in range(r0, r1 + 1):
                band.add((rr, cc))
            red_line.add((red_r, cc))
    else:
        assert direction == "right"
        for cc in range(c1 + 1, out.shape[1]):
            if np.any(blockers[r0 : r1 + 1, cc]):
                stopped_by = "cyan"
                break
            out[r0 : r1 + 1, cc] = GREEN
            out[red_r, cc] = RED
            for rr in range(r0, r1 + 1):
                band.add((rr, cc))
            red_line.add((red_r, cc))

    return {
        "direction": direction,
        "band": band,
        "red_line": red_line,
        "stop_reason": stopped_by,
    }


def _solve(grid: np.ndarray, *, fill_completion: bool = True, return_debug: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    originals, sprite_mask = _extract_originals(g)
    rows = sorted({int(item["top_left"][0]) for item in originals})
    cols = sorted({int(item["top_left"][1]) for item in originals})
    assert len(rows) >= 2 and len(cols) >= 2, "expected a row-column lattice"

    occupied = {tuple(map(int, item["top_left"])) for item in originals}
    assert any(all((r, c) in occupied for c in cols) for r in rows), "expected a complete lattice row"
    assert any(all((r, c) in occupied for r in rows) for c in cols), "expected a complete lattice column"

    out = g.copy()
    blockers = np.zeros(g.shape, dtype=bool)
    missing = [(r, c) for r in rows for c in cols if (r, c) not in occupied]
    assert missing, "expected at least one missing lattice slot"

    if fill_completion:
        fill_patch = _cyan_patch(sprite_mask)
        for r, c in missing:
            target = out[r : r + SPRITE_H, c : c + SPRITE_W]
            assert target.shape == (SPRITE_H, SPRITE_W)
            assert np.all(target == BG), "missing lattice slots must be empty before completion"
            paste(out, fill_patch, r, c, transparent=BG)
            blockers[r : r + SPRITE_H, c : c + SPRITE_W] |= sprite_mask

    extensions = [_paint_extension(out, blockers, original) for original in originals]

    if return_debug:
        return out, {
            "originals": originals,
            "rows": rows,
            "cols": cols,
            "missing": missing,
            "extensions": extensions,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing sprite lattice in cyan, then extend only the original sprites until a cyan completion or the canvas edge stops them.

    Input structure:
    - The background is black, and the foreground consists of identical 4-connected green sprites arranged on a rectangular lattice.
    - Each original sprite contains exactly one red pixel that replaces one sprite cell on a non-corner side of the sprite bbox.
    - Some lattice positions are empty, while at least one full lattice row and one full lattice column are already occupied.

    Rule:
    - Treat each red pixel as part of its sprite when recovering the shared sprite mask and the lattice top-left coordinates.
    - Stamp a cyan copy of that mask into every missing row-column lattice slot, without any red indicator pixel.
    - For every original sprite, use the red pixel side to choose an extension direction, then grow a green band from the corresponding sprite edge.
    - Overdraw the indicator row or column inside that band with red, and stop before the next band step would hit any cyan completion cell or leave the grid.

    Key insight:
    - The same lattice information serves two roles: first it reveals where cyan copies must be completed, then those completed copies become the blockers that determine where each original extension must stop.
    """
    return _solve(grid, fill_completion=True, return_debug=False)


def _sample_case() -> np.ndarray | None:
    active_arm = random.choice(["row", "col"])
    row_side = random.choice(["top", "bottom"])
    col_side = random.choice(["left", "right"])

    nrows = random.randint(2, 4)
    ncols = random.randint(2, 4)
    if active_arm == "row":
        ncols = max(ncols, 3)
    else:
        nrows = max(nrows, 3)

    stride = random.randint(5, 6)
    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)

    height = top_margin + bottom_margin + SPRITE_H + stride * (nrows - 1)
    width = left_margin + right_margin + SPRITE_W + stride * (ncols - 1)
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    row_positions = [top_margin + stride * i for i in range(nrows)]
    col_positions = [left_margin + stride * j for j in range(ncols)]

    anchor_row = 0 if row_side == "top" else nrows - 1
    anchor_col = 0 if col_side == "left" else ncols - 1
    row_in = "down" if row_side == "top" else "up"
    row_out = "up" if row_side == "top" else "down"
    col_in = "right" if col_side == "left" else "left"
    col_out = "left" if col_side == "left" else "right"
    corner_dir = random.choice([row_out, col_out])

    occupied: list[tuple[int, int, str]] = []
    for i in range(nrows):
        for j in range(ncols):
            if i != anchor_row and j != anchor_col:
                continue
            if i == anchor_row and j == anchor_col:
                direction = corner_dir
            elif i == anchor_row:
                direction = row_in if active_arm == "row" else row_out
            else:
                direction = col_out if active_arm == "row" else col_in
            occupied.append((i, j, direction))

    random.shuffle(occupied)
    for i, j, direction in occupied:
        patch = _sprite_patch(direction)
        paste(grid, patch, row_positions[i], col_positions[j], transparent=BG)
    return grid


def _valid_case(grid: np.ndarray) -> bool:
    try:
        solved, debug = _solve(grid, fill_completion=True, return_debug=True)
        no_completion, debug_no_completion = _solve(grid, fill_completion=False, return_debug=True)
    except AssertionError:
        return False

    if np.array_equal(solved, grid) or np.array_equal(solved, no_completion):
        return False

    missing_count = len(debug["missing"])
    if missing_count < 1:
        return False

    cyan_components = find_components(
        solved,
        bg=BG,
        colors=[CYAN],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if len(cyan_components) != missing_count:
        return False

    with_ext = debug["extensions"]
    without_ext = debug_no_completion["extensions"]
    if len(with_ext) != len(without_ext):
        return False

    shortened = 0
    cyan_stops = 0
    boundary_stops = 0
    painted_sets = [set(item["band"]) for item in with_ext]
    for i in range(len(painted_sets)):
        for j in range(i + 1, len(painted_sets)):
            if painted_sets[i] & painted_sets[j]:
                return False

    for item, baseline in zip(with_ext, without_ext):
        if len(item["band"]) == 0:
            return False
        if item["stop_reason"] == "cyan":
            cyan_stops += 1
        else:
            boundary_stops += 1
        if len(item["band"]) < len(baseline["band"]):
            shortened += 1

    if cyan_stops < 2 or boundary_stops < 1 or shortened < 2:
        return False
    if int(np.count_nonzero(solved == CYAN)) <= 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_case()
        if grid is not None and _valid_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
