from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, is_contiguous, paste, random_free_location


BG = Color.BLACK
FRAME = Color.TEAL
SPRITE_COLORS = (Color.BLUE, Color.RED, Color.YELLOW, Color.ORANGE)
CAVITY_SIZE = 4
CELL_STRIDE = 5
GEN_RETRIES = 400

CORNER_TO_ROT_CW = {
    (0, 0): 0,
    (0, CAVITY_SIZE - 1): 1,
    (CAVITY_SIZE - 1, CAVITY_SIZE - 1): 2,
    (CAVITY_SIZE - 1, 0): 3,
}


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns))


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(arr != int(bg))
    if coords.size == 0:
        return arr[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_cavities(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cavities: list[tuple[int, int]] = []
    for r0 in range(1, h - CAVITY_SIZE):
        for c0 in range(1, w - CAVITY_SIZE):
            if not np.all(g[r0 - 1, c0 - 1 : c0 + CAVITY_SIZE + 1] == FRAME):
                continue
            if not np.all(g[r0 + CAVITY_SIZE, c0 - 1 : c0 + CAVITY_SIZE + 1] == FRAME):
                continue
            if not np.all(g[r0 - 1 : r0 + CAVITY_SIZE + 1, c0 - 1] == FRAME):
                continue
            if not np.all(g[r0 - 1 : r0 + CAVITY_SIZE + 1, c0 + CAVITY_SIZE] == FRAME):
                continue
            if np.any(g[r0 : r0 + CAVITY_SIZE, c0 : c0 + CAVITY_SIZE] == FRAME):
                continue
            cavities.append((int(r0), int(c0)))

    cavities = sorted(set(cavities))
    assert len(cavities) in {4, 6}, "expected a 2x2 or 2x3 lattice of 4x4 cavities"
    rows = sorted({r for r, _ in cavities})
    cols = sorted({c for _, c in cavities})
    assert len(rows) == 2 and len(cols) in {2, 3}, "expected 2 cavity rows and 2 or 3 columns"
    assert rows[1] - rows[0] == CELL_STRIDE, "cavity rows must be regularly spaced"
    for i in range(len(cols) - 1):
        assert cols[i + 1] - cols[i] == CELL_STRIDE, "cavity cols must be regularly spaced"
    return cavities


def _extract_marker(inner: np.ndarray) -> tuple[int, int, tuple[int, int]] | None:
    nz = np.argwhere(inner != BG)
    if nz.size == 0:
        return None
    assert nz.shape[0] == 1, "each cavity may contain at most one marker pixel"
    r, c = map(int, nz[0])
    assert (r, c) in CORNER_TO_ROT_CW, "marker must sit in a cavity corner"
    color = int(inner[r, c])
    assert color in SPRITE_COLORS, "marker color must match one of the external sprite colors"
    return color, int(CORNER_TO_ROT_CW[(r, c)]), (r, c)


def _extract_external_sprite(grid: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    sprite_comps = [comp for comp in comps if int(comp["area"]) >= 4]
    assert len(sprite_comps) == 1, "each marker color must have exactly one matching external sprite"
    coords = sprite_comps[0]["coords"]
    r0, c0, r1, c1 = sprite_comps[0]["bbox"]
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    assert sprite.shape[0] <= CAVITY_SIZE and sprite.shape[1] <= CAVITY_SIZE
    return coords, sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each matched external sprite by the angle encoded by its cavity marker, then dock it into that cavity.

    Input structure:
    - A black background contains one teal lattice frame forming a 2x2 or 2x3 array of enclosed 4x4 cavities.
    - Some cavities contain a single colored marker pixel in one of the four cavity corners.
    - Outside the lattice frame, there is exactly one 4-connected monochrome sprite for each marker color.

    Rule:
    - Find every 4x4 cavity enclosed by the teal frame.
    - For each marked cavity, use the marker color to select the matching external sprite and use the marker corner to choose the clockwise rotation: upper-left 0, upper-right 90, lower-right 180, lower-left 270.
    - Erase the marker, erase the original external sprite, rotate that sprite, and paste it into the cavity with top-left alignment while treating black cells in the sprite bbox as transparent.

    Key insight:
    - The marker pixel carries two independent signals at once: its color identifies which sprite to move, and its corner position encodes the sprite's rotation before docking.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    tasks: list[tuple[int, int, int, int, tuple[int, int]]] = []
    for r0, c0 in _find_cavities(g):
        inner = g[r0 : r0 + CAVITY_SIZE, c0 : c0 + CAVITY_SIZE]
        marker = _extract_marker(inner)
        if marker is None:
            continue
        color, rot_cw, local_pos = marker
        tasks.append((int(r0), int(c0), int(color), int(rot_cw), local_pos))

    for r0, c0, _color, _rot_cw, (mr, mc) in tasks:
        out[r0 + mr, c0 + mc] = BG

    for r0, c0, color, rot_cw, _local_pos in tasks:
        coords, sprite = _extract_external_sprite(g, color)
        out[coords[:, 0], coords[:, 1]] = BG
        rotated = _rotate_cw(sprite, rot_cw)
        paste(out, rotated, r0, c0, transparent=BG, clip=False)

    return out


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    rotations: list[np.ndarray] = []
    for k in range(4):
        rotated = _crop_non_bg(_rotate_cw(mask.astype(int), k), bg=0).astype(bool)
        if not any(np.array_equal(rotated, old) for old in rotations):
            rotations.append(rotated)
    return rotations


def _random_sprite_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        target = rng.randint(4, min(8, h * w))
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True

        frontier = set()
        while int(mask.sum()) < target:
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(tuple(frontier))
            frontier.remove((nr, nc))
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue

        cropped = _crop_non_bg(mask.astype(int), bg=0).astype(bool)
        if cropped.shape[0] > CAVITY_SIZE or cropped.shape[1] > CAVITY_SIZE:
            continue
        if int(cropped.sum()) < 4 or int(cropped.sum()) > 8:
            continue
        if not is_contiguous(cropped.astype(int), background=0, connectivity=4):
            continue
        if len(_unique_rotations(cropped)) != 4:
            continue
        return cropped

    raise AssertionError("failed to generate a non-degenerate sprite")


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _draw_lattice(grid: np.ndarray, top: int, left: int, rows: int, cols: int) -> list[tuple[int, int]]:
    for rr in range(rows + 1):
        grid[top + rr * CELL_STRIDE, left : left + cols * CELL_STRIDE + 1] = FRAME
    for cc in range(cols + 1):
        grid[top : top + rows * CELL_STRIDE + 1, left + cc * CELL_STRIDE] = FRAME
    return [
        (top + 1 + rr * CELL_STRIDE, left + 1 + cc * CELL_STRIDE)
        for rr in range(rows)
        for cc in range(cols)
    ]


def _try_generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    rows = 2
    cols = 3 if rng.random() < 0.6 else 2
    frame_h = rows * CELL_STRIDE + 1
    frame_w = cols * CELL_STRIDE + 1

    h = rng.randint(max(20, frame_h + 4), min(28, frame_h + 10))
    w = rng.randint(max(20, frame_w + 4), min(28, frame_w + 10))
    top = rng.randint(1, h - frame_h - 1)
    left = rng.randint(1, w - frame_w - 1)

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    cavities = _draw_lattice(inp, top, left, rows, cols)
    _draw_lattice(out, top, left, rows, cols)

    max_marks = min(4, len(cavities))
    mark_count = rng.choices([2, 3, 4][: max_marks - 1], weights=[2, 3, 5][: max_marks - 1], k=1)[0]
    marked_cavities = rng.sample(cavities, mark_count)
    colors = rng.sample(list(SPRITE_COLORS), mark_count)
    corners = rng.sample(list(CORNER_TO_ROT_CW.keys()), mark_count)

    placement_canvas = np.full((h, w), BG, dtype=int)
    placement_canvas[top : top + frame_h, left : left + frame_w] = FRAME

    for (r0, c0), color, corner in zip(marked_cavities, colors, corners):
        marker_r, marker_c = corner
        inp[r0 + marker_r, c0 + marker_c] = int(color)

        sprite_mask = _random_sprite_mask(rng)
        sprite = _colorize(sprite_mask, color)
        input_rot = rng.choice([k for k in range(4) if k != CORNER_TO_ROT_CW[corner]])
        scattered = _rotate_cw(sprite, input_rot)
        out_sprite = _rotate_cw(scattered, CORNER_TO_ROT_CW[corner])
        paste(out, out_sprite, r0, c0, transparent=BG, clip=False)
        try:
            sr, sc = random_free_location(
                placement_canvas,
                scattered,
                transparent=BG,
                border=0,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, scattered, sr, sc, transparent=BG, clip=False)
        paste(placement_canvas, scattered, sr, sc, transparent=BG, clip=False)

    if len(_find_cavities(inp)) != len(cavities):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _try_generate_case(rng)
        if case is None:
            continue
        inp, _out = case
        return inp
    raise AssertionError("failed to generate a valid input for puzzle 4")
