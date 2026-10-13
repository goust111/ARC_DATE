from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds


BLUE = Color.BLUE
RED = Color.RED
ORANGE = Color.ORANGE
TEAL = Color.TEAL
BG = Color.BLACK

GRID_ROWS = (16, 22)
GRID_COLS = (16, 22)
RECT_COUNT = (3, 5)
INPUT_MARGIN = 2
INPUT_GAP = 2
GEN_ATTEMPTS = 500

INTERIOR_SIZES = [(ih, iw) for ih in range(1, 6) for iw in range(1, 6) if 2 <= ih * iw <= 15]
EVEN_SIZES = [size for size in INTERIOR_SIZES if (size[0] * size[1]) % 2 == 0]
ODD_SIZES = [size for size in INTERIOR_SIZES if (size[0] * size[1]) % 2 == 1]
CORNERS = [
    ((0, 0), (-1, -1)),
    ((0, 1), (-1, 1)),
    ((1, 0), (1, -1)),
    ((1, 1), (1, 1)),
]


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(bbox: tuple[int, int, int, int], gap: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - gap), max(0, c0 - gap), min(h - 1, r1 + gap), min(w - 1, c1 + gap)


def _draw_frame(patch: np.ndarray, frame_color: int, fill_color: int | None = None) -> np.ndarray:
    out = patch.copy()
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    if fill_color is not None and out.shape[0] > 2 and out.shape[1] > 2:
        out[1:-1, 1:-1] = fill_color
    return out


def _extract_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    allowed = {BG, BLUE, TEAL}
    colors = set(map(int, np.unique(g)))
    assert colors <= allowed, "input should only contain black, blue, and teal"

    fg_mask = np.isin(g, [BLUE, TEAL])
    comps = components_mask(fg_mask, connectivity=4)
    assert comps, "expected at least one hollow rectangle"

    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    used_teal = 0

    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 3 and w >= 3, "frame must have non-empty border and interior"

        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        assert np.array_equal(np.isin(patch, [BLUE, TEAL]), border_mask), "component must be a 1-pixel rectangular frame"

        teal_coords = np.argwhere(patch == TEAL)
        assert len(teal_coords) == 1, "each frame must contain exactly one teal control pixel"
        used_teal += 1
        teal_r, teal_c = map(int, teal_coords[0])
        assert (teal_r, teal_c) in {
            (0, 0),
            (0, w - 1),
            (h - 1, 0),
            (h - 1, w - 1),
        }, "teal control pixel must sit on a rectangle corner"

        blue_mask = patch == BLUE
        assert blue_mask.sum() == border_mask.sum() - 1, "all non-control border pixels must be blue"

        corner_dr = -1 if teal_r == 0 else 1
        corner_dc = -1 if teal_c == 0 else 1
        interior_area = (h - 2) * (w - 2)
        assert 2 <= interior_area <= 15, "interior area must match the puzzle family"

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "move_dr": int(corner_dr if interior_area % 2 == 0 else -corner_dr),
                "move_dc": int(corner_dc if interior_area % 2 == 0 else -corner_dc),
                "fill": int(RED if interior_area % 2 == 0 else ORANGE),
                "area": int(interior_area),
            }
        )

    assert used_teal == int((g == TEAL).sum()), "every teal pixel must belong to exactly one frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each teal-cornered hollow blue rectangle diagonally based on interior-area parity, then fill it.

    Input structure:
    - A black background contains several one-cell-thick blue hollow rectangles.
    - Each rectangle has exactly one corner cell replaced by a teal control pixel.
    - Rectangle interiors are black, and different rectangles are separated from one another.

    Rule:
    - Treat the teal corner as a directional marker only; the completed rectangle border is fully blue.
    - Compute each rectangle's interior area.
    - If the area is even, move the whole rectangle one cell diagonally toward the teal corner and fill its interior red.
    - If the area is odd, move the whole rectangle one cell diagonally away from the teal corner and fill its interior orange.
    - Erase the original rectangle footprint, including the original teal pixel.

    Key insight:
    - Interior-area parity controls two outputs at once: the fill color and whether the diagonal shift goes toward or away from the marked corner.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    out = np.full_like(g, BG)
    occupied = np.zeros(g.shape, dtype=bool)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        dr = int(frame["move_dr"])
        dc = int(frame["move_dc"])
        fill = int(frame["fill"])

        tr0, tc0 = r0 + dr, c0 + dc
        tr1, tc1 = r1 + dr, c1 + dc
        assert in_bounds(g.shape, tr0, tc0) and in_bounds(g.shape, tr1, tc1), "shifted frame must stay in bounds"
        assert not occupied[tr0 : tr1 + 1, tc0 : tc1 + 1].any(), "shifted frames must not overlap"

        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        patch = _draw_frame(patch, BLUE, fill)
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = patch
        occupied[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    return out


def _choose_parities(k: int) -> list[int]:
    parities = [0, 1]
    while len(parities) < k:
        parities.append(random.randint(0, 1))
    random.shuffle(parities)
    return parities


def _choose_corners(k: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    for _ in range(50):
        corners = [random.choice(CORNERS) for _ in range(k)]
        if len({corner[1] for corner in corners}) >= 2:
            return corners
    return [random.choice(CORNERS) for _ in range(k)]


def _sample_objects(k: int) -> list[dict[str, int | tuple[int, int]]]:
    parities = _choose_parities(k)
    corners = _choose_corners(k)

    objs = []
    for parity, corner in zip(parities, corners):
        ih, iw = random.choice(EVEN_SIZES if parity == 0 else ODD_SIZES)
        (_, _), corner_sign = corner
        move = corner_sign if parity == 0 else (-corner_sign[0], -corner_sign[1])
        objs.append(
            {
                "ih": ih,
                "iw": iw,
                "oh": ih + 2,
                "ow": iw + 2,
                "parity": parity,
                "corner": corner[0],
                "corner_sign": corner_sign,
                "move": move,
                "area": ih * iw,
            }
        )

    objs.sort(key=lambda obj: (obj["oh"] * obj["ow"], obj["area"]), reverse=True)
    return objs


def _candidate_positions(h: int, w: int, oh: int, ow: int) -> list[tuple[int, int]]:
    rows = range(INPUT_MARGIN, h - oh - INPUT_MARGIN + 1)
    cols = range(INPUT_MARGIN, w - ow - INPUT_MARGIN + 1)
    positions = [(r, c) for r in rows for c in cols]
    random.shuffle(positions)
    return positions


def _place_objects(shape: tuple[int, int], objects: list[dict[str, int | tuple[int, int]]]) -> list[dict[str, int | tuple[int, int]]] | None:
    h, w = shape
    placed: list[dict[str, int | tuple[int, int]]] = []

    for obj in objects:
        oh = int(obj["oh"])
        ow = int(obj["ow"])
        move_dr, move_dc = obj["move"]
        candidates = _candidate_positions(h, w, oh, ow)
        chosen = None

        for r0, c0 in candidates:
            inp_bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            out_bbox = (r0 + move_dr, c0 + move_dc, r0 + move_dr + oh - 1, c0 + move_dc + ow - 1)

            if not (in_bounds(shape, out_bbox[0], out_bbox[1]) and in_bounds(shape, out_bbox[2], out_bbox[3])):
                continue

            expanded_inp = _expand_bbox(inp_bbox, INPUT_GAP, shape)
            collision = False
            for other in placed:
                if _bbox_intersects(expanded_inp, other["inp_bbox"]):
                    collision = True
                    break
                if _bbox_intersects(out_bbox, other["out_bbox"]):
                    collision = True
                    break
            if collision:
                continue

            chosen = dict(obj)
            chosen["inp_bbox"] = inp_bbox
            chosen["out_bbox"] = out_bbox
            placed.append(chosen)
            break

        if chosen is None:
            return None

    return placed


def _render_input(shape: tuple[int, int], objects: list[dict[str, int | tuple[int, int]]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["inp_bbox"]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        patch = _draw_frame(patch, BLUE)
        corner_r = 0 if obj["corner"][0] == 0 else patch.shape[0] - 1
        corner_c = 0 if obj["corner"][1] == 0 else patch.shape[1] - 1
        patch[corner_r, corner_c] = TEAL
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return grid


def _valid_example(inp: np.ndarray, expected_count: int) -> bool:
    try:
        frames = _extract_frames(inp)
        if len(frames) != expected_count:
            return False
        parities = {frame["area"] % 2 for frame in frames}
        if parities != {0, 1}:
            return False
        out = main(inp)
        if np.array_equal(inp, out):
            return False
        if np.any(out == TEAL):
            return False
        if not np.any(out == RED) or not np.any(out == ORANGE):
            return False
        blue_components = components_mask(out == BLUE, connectivity=4)
        if len(blue_components) != expected_count:
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    k = random.randint(*RECT_COUNT)
    objects = _sample_objects(k)
    placed = _place_objects((h, w), objects)
    if placed is None:
        return None

    grid = _render_input((h, w), placed)
    return grid if _valid_example(grid, k) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
