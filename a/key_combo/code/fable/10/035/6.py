from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds, neighbors4


Coord = tuple[int, int]
Box = tuple[int, int, int, int]

YELLOW = Color.YELLOW
RED = Color.RED
BACKGROUND = Color.BLACK
BODY_COLORS = (Color.BLUE, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
DISTRACTOR_COLORS = (Color.BLUE, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON, YELLOW, RED)

GRID_MIN = 14
GRID_MAX = 24
STAMP_CELL_COUNT = (5, 8)
STAMP_BBOX_MAX = 4
COPY_COUNT = (3, 4)
DISTRACTOR_COUNT = (1, 4)
MAX_ATTEMPTS = 400
GAP = 1


def _coords_of_color(arr: np.ndarray, color: int) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(arr == int(color))]


def _anchor_vector(yellow: Coord, red: Coord) -> Coord:
    return int(red[0] - yellow[0]), int(red[1] - yellow[1])


def _normalize_shape(shape: set[Coord]) -> tuple[Coord, ...]:
    r0 = min(r for r, _c in shape)
    c0 = min(c for _r, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _build_patch(shape: tuple[Coord, ...], yellow: Coord, red: Coord, body_color: int) -> np.ndarray:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    patch = np.zeros((max(rows) + 1, max(cols) + 1), dtype=int)
    for r, c in shape:
        patch[r, c] = int(body_color)
    patch[yellow] = int(YELLOW)
    patch[red] = int(RED)
    return patch


def _rotation_count(patch: np.ndarray) -> int:
    return len({np.rot90(patch, k).tobytes() for k in range(4)})


def _random_stamp_patch() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        target_size = random.randint(*STAMP_CELL_COUNT)
        shape: set[Coord] = {(0, 0)}
        while len(shape) < target_size:
            frontier = sorted({nb for cell in shape for nb in neighbors4(*cell)} - shape)
            shape.add(random.choice(frontier))
            shape = set(_normalize_shape(shape))

        norm = _normalize_shape(shape)
        rows = [r for r, _c in norm]
        cols = [c for _r, c in norm]
        h = max(rows) + 1
        w = max(cols) + 1
        if h < 2 or w < 2 or h > STAMP_BBOX_MAX or w > STAMP_BBOX_MAX:
            continue

        pairs: list[tuple[Coord, Coord]] = []
        cell_set = set(norm)
        for cell in norm:
            for nb in neighbors4(*cell):
                if nb in cell_set and cell < nb:
                    pairs.append((cell, nb))
                    pairs.append((nb, cell))
        random.shuffle(pairs)
        if not pairs:
            continue

        body_color = random.choice(BODY_COLORS)
        for yellow, red in pairs:
            patch = _build_patch(norm, yellow, red, body_color)
            if _rotation_count(patch) == 4:
                return np.rot90(patch, random.randrange(4)).copy()
    raise RuntimeError("failed to generate a non-degenerate stamp patch")


def _stamp_box(top: Coord, patch: np.ndarray) -> Box:
    r, c = top
    h, w = patch.shape
    return int(r), int(c), int(r + h - 1), int(c + w - 1)


def _boxes_far_apart(a: Box, b: Box, gap: int = GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _placement_candidates(shape: tuple[int, int], patch: np.ndarray, placed: list[Box]) -> list[Coord]:
    H, W = shape
    h, w = patch.shape
    candidates = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            box = _stamp_box((r, c), patch)
            if all(_boxes_far_apart(box, other) for other in placed):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _paste_nonzero(grid: np.ndarray, patch: np.ndarray, top: Coord) -> None:
    r0, c0 = top
    for dr, dc in np.argwhere(patch != 0):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        grid[rr, cc] = int(patch[int(dr), int(dc)])


def _source_and_pairs(grid: np.ndarray) -> tuple[np.ndarray, Coord, Coord, list[tuple[Coord, Coord]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=4)

    source_candidates: list[tuple[np.ndarray, Coord, Coord]] = []
    pair_components: list[tuple[Coord, Coord]] = []

    for coords in comps:
        pts = [tuple(map(int, rc)) for rc in coords]
        colors = [int(g[r, c]) for r, c in pts]
        yellows = [(r, c) for r, c in pts if int(g[r, c]) == YELLOW]
        reds = [(r, c) for r, c in pts if int(g[r, c]) == RED]

        if len(pts) == 2 and len(yellows) == 1 and len(reds) == 1:
            y = yellows[0]
            r = reds[0]
            if abs(y[0] - r[0]) + abs(y[1] - r[1]) == 1:
                pair_components.append((y, r))
            continue

        if len(pts) < 3 or len(yellows) != 1 or len(reds) != 1:
            continue

        source_colors = set(colors)
        if len(source_colors) != 3:
            continue

        y = yellows[0]
        r = reds[0]
        if abs(y[0] - r[0]) + abs(y[1] - r[1]) != 1:
            continue

        r0, c0, r1, c1 = bbox_coords(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for rr, cc in pts:
            patch[rr - r0, cc - c0] = int(g[rr, cc])
        source_candidates.append((patch, (y[0] - r0, y[1] - c0), (r[0] - r0, r[1] - c0)))

    assert len(source_candidates) == 1, "expected exactly one stamp sprite"
    assert pair_components, "expected at least one external anchor pair"

    patch, local_y, local_r = source_candidates[0]
    assert _rotation_count(patch) == 4, "stamp rotations must be distinguishable"
    return patch, local_y, local_r, pair_components


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the stamp so its yellow-to-red arrow matches each anchor pair, then stamp it with yellow aligned.

    Input structure:
    - The background dominates the grid.
    - One multicolor stamp sprite contains exactly one yellow cell and one red cell that are orthogonally adjacent; all remaining stamp cells use one body color.
    - Elsewhere there are several isolated 2-cell yellow-red anchor pairs plus a few isolated distractor pixels that are not part of any pair.

    Rule:
    - Keep the original stamp in place.
    - For each isolated anchor pair, rotate the whole stamp without reflection until the internal yellow->red direction matches that pair.
    - Translate the rotated stamp so its yellow anchor lands on the pair's yellow cell, then stamp every non-background sprite cell there, including the two anchor cells.

    Key insight:
    - The scattered yellow-red pairs are directional instructions: the arrow gives the rotation, and the yellow endpoint gives the placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    source_patch, _local_y, _local_r, pair_components = _source_and_pairs(g)
    out = g.copy()

    for target_y, target_r in pair_components:
        target_vec = _anchor_vector(target_y, target_r)
        matches: list[tuple[np.ndarray, Coord]] = []
        for k in range(4):
            rotated = np.rot90(source_patch, k)
            rotated_y = _coords_of_color(rotated, YELLOW)
            rotated_r = _coords_of_color(rotated, RED)
            assert len(rotated_y) == 1 and len(rotated_r) == 1
            if _anchor_vector(rotated_y[0], rotated_r[0]) == target_vec:
                matches.append((rotated, rotated_y[0]))
        assert len(matches) == 1, "anchor direction must determine a unique rotation"

        rotated, rotated_y = matches[0]
        top = (target_y[0] - rotated_y[0], target_y[1] - rotated_y[1])
        target_pair_cells = {target_y, target_r}

        for rr, cc in np.argwhere(rotated != bg):
            gr = top[0] + int(rr)
            gc = top[1] + int(cc)
            assert in_bounds(out.shape, gr, gc), "stamped copy leaves the grid"
            value = int(rotated[int(rr), int(cc)])
            if (gr, gc) in target_pair_cells:
                assert int(out[gr, gc]) == value, "anchor cells must already match"
            else:
                assert int(out[gr, gc]) == bg, "copies must not overlap other content"
                out[gr, gc] = value

    return out


def _shortcut_no_rotation(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    source_patch, local_y, _local_r, pair_components = _source_and_pairs(g)
    out = g.copy()
    for target_y, target_r in pair_components:
        top = (target_y[0] - local_y[0], target_y[1] - local_y[1])
        target_pair_cells = {target_y, target_r}
        for rr, cc in np.argwhere(source_patch != bg):
            gr = top[0] + int(rr)
            gc = top[1] + int(cc)
            value = int(source_patch[int(rr), int(cc)])
            if in_bounds(out.shape, gr, gc):
                if (gr, gc) in target_pair_cells:
                    out[gr, gc] = value
                elif int(out[gr, gc]) == bg:
                    out[gr, gc] = value
    return out


def _occupied_cells(patch: np.ndarray, top: Coord) -> set[Coord]:
    r0, c0 = top
    return {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(patch != 0)}


def _pair_cells(patch: np.ndarray, top: Coord) -> tuple[Coord, Coord]:
    yellow = _coords_of_color(patch, YELLOW)[0]
    red = _coords_of_color(patch, RED)[0]
    return (top[0] + yellow[0], top[1] + yellow[1]), (top[0] + red[0], top[1] + red[1])


def _valid_distractor_cells(shape: tuple[int, int], forbidden: set[Coord]) -> list[Coord]:
    H, W = shape
    cells: list[Coord] = []
    for r in range(H):
        for c in range(W):
            if (r, c) in forbidden:
                continue
            neighbors = {(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
            if neighbors & forbidden:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    stamp = _random_stamp_patch()
    copy_count = random.randint(*COPY_COUNT)
    directions = random.sample(range(4), k=copy_count)

    placed_boxes: list[Box] = []
    source_top_candidates = _placement_candidates((H, W), stamp, placed_boxes)
    if not source_top_candidates:
        raise RuntimeError("no room for source stamp")
    source_top = source_top_candidates[0]
    placed_boxes.append(_stamp_box(source_top, stamp))

    copies: list[tuple[np.ndarray, Coord]] = []
    for direction in directions:
        rotated = np.rot90(stamp, direction)
        candidates = _placement_candidates((H, W), rotated, placed_boxes)
        if not candidates:
            raise RuntimeError("no room for copy placement")
        top = candidates[0]
        placed_boxes.append(_stamp_box(top, rotated))
        copies.append((rotated, top))

    inp = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    _paste_nonzero(inp, stamp, source_top)
    _paste_nonzero(out, stamp, source_top)

    forbidden = _occupied_cells(stamp, source_top)
    pair_vectors: set[Coord] = set()
    for rotated, top in copies:
        y_cell, r_cell = _pair_cells(rotated, top)
        inp[y_cell] = int(YELLOW)
        inp[r_cell] = int(RED)
        _paste_nonzero(out, rotated, top)
        forbidden |= _occupied_cells(rotated, top)
        pair_vectors.add(_anchor_vector(y_cell, r_cell))

    if len(pair_vectors) < 2:
        raise RuntimeError("need multiple orientations to make rotation necessary")

    distractor_positions = _valid_distractor_cells(inp.shape, forbidden)
    distractor_count = random.randint(*DISTRACTOR_COUNT)
    if len(distractor_positions) < distractor_count:
        raise RuntimeError("not enough room for distractors")

    for pos in distractor_positions[:distractor_count]:
        inp[pos] = int(random.choice(DISTRACTOR_COLORS))
        out[pos] = int(inp[pos])
        forbidden.add(pos)

    return inp, out


def _validate_generated(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    try:
        source_patch, _local_y, _local_r, pairs = _source_and_pairs(inp)
    except Exception:
        return False

    if _rotation_count(source_patch) != 4:
        return False
    if len(pairs) < 3:
        return False
    if len({_anchor_vector(y, r) for y, r in pairs}) < 2:
        return False
    if np.array_equal(_shortcut_no_rotation(inp), out):
        return False

    bg = int(get_bg(inp))
    comps = components_mask(inp != bg, connectivity=4)
    pair_count = 0
    source_count = 0
    for coords in comps:
        pts = [tuple(map(int, rc)) for rc in coords]
        colors = {int(inp[r, c]) for r, c in pts}
        if len(pts) == 2 and colors == {YELLOW, RED}:
            pair_count += 1
        elif len(pts) >= 3 and YELLOW in colors and RED in colors:
            source_count += 1
        elif len(pts) != 1:
            return False
    return pair_count == len(pairs) and source_count == 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, out = _generate_case()
        except RuntimeError:
            continue
        if _validate_generated(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid directed-anchor stamp puzzle")
