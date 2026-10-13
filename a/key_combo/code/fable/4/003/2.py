from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg


BG = Color.BLACK
BODY = Color.GREY
KEY_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
MAX_GENERATION_ATTEMPTS = 16

Coord = tuple[int, int]
Shape = tuple[frozenset[Coord], int, int]


def _background(g: np.ndarray) -> int:
    corners = np.array([g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]], dtype=int)
    vals, counts = np.unique(corners, return_counts=True)
    corner_bg = int(vals[int(np.argmax(counts))])
    if int(counts.max()) >= 3 and corner_bg != BODY:
        return corner_bg
    bg = int(get_bg(g))
    assert bg != BODY, "background must be distinct from the gray body"
    return bg


def _rotate_cell(r: int, c: int, center2: tuple[int, int], turns: int) -> Coord:
    rr = int(r)
    cc = int(c)
    sr, sc = map(int, center2)
    for _ in range(int(turns) % 4):
        nr2 = sr + 2 * cc - sc
        nc2 = sc - 2 * rr + sr
        assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation center must map cells to cells"
        rr = nr2 // 2
        cc = nc2 // 2
    return int(rr), int(cc)


def _rotate_cells(cells: set[Coord] | frozenset[Coord], center2: tuple[int, int], turns: int) -> set[Coord]:
    return {_rotate_cell(r, c, center2, turns) for r, c in cells}


def _normalize_shape(cells: set[Coord] | frozenset[Coord] | list[Coord]) -> Shape:
    assert cells, "shape must be non-empty"
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    norm = frozenset((int(r) - r0, int(c) - c0) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _shape_from_coords(coords: np.ndarray) -> Shape:
    return _normalize_shape([(int(r), int(c)) for r, c in coords])


def _rotate_shape(shape: Shape, turns: int) -> Shape:
    coords, h, w = shape
    out = coords
    hh = int(h)
    ww = int(w)
    for _ in range(int(turns) % 4):
        out = frozenset((c, hh - 1 - r) for r, c in out)
        hh, ww = ww, hh
    return out, hh, ww


def _shape_variants(shape: Shape) -> set[frozenset[Coord]]:
    return {_rotate_shape(shape, turns)[0] for turns in range(4)}


def _mask_components(mask: np.ndarray) -> list[dict]:
    tmp = np.zeros(mask.shape, dtype=int)
    tmp[mask] = 1
    return find_components(tmp, bg=0, connectivity=4, monochromatic=True, colors=[1])


def _body_center(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    gray = np.argwhere(g == BODY)
    assert gray.size > 0, "expected a gray body"
    r0, c0, r1, c1 = bbox_coords(gray)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h >= 7, "the visible gray body must have a square rotational bbox"
    return (int(r0 + r1), int(c0 + c1)), (int(r0), int(c0), int(r1), int(c1))


def _infer_gap_components(g: np.ndarray, bg: int, center2: tuple[int, int]) -> tuple[np.ndarray, list[dict]]:
    h, w = map(int, g.shape)
    complete = np.zeros(g.shape, dtype=bool)

    for r, c in np.argwhere(g == BODY):
        for turns in range(4):
            rr, cc = _rotate_cell(int(r), int(c), center2, turns)
            assert 0 <= rr < h and 0 <= cc < w, "rotated body pixel leaves the grid"
            complete[rr, cc] = True

    colored_keys = (g != bg) & (g != BODY)
    assert not np.any(complete & colored_keys), "key sprites must be outside the body"

    missing = complete & (g == bg)
    assert np.any(missing), "expected at least one missing body gap"
    gap_components = _mask_components(missing)
    assert gap_components, "expected connected gap components"

    for gap in gap_components:
        cells = {(int(r), int(c)) for r, c in gap["coords"]}
        for turns in (1, 2, 3):
            for rr, cc in _rotate_cells(cells, center2, turns):
                assert g[rr, cc] == BODY, "every gap must have three intact gray counterparts"

    return complete, gap_components


def _key_components(g: np.ndarray, bg: int, complete: np.ndarray) -> list[dict]:
    key_colors = [int(c) for c in np.unique(g) if int(c) not in {int(bg), BODY}]
    keys = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=key_colors)
    assert keys, "expected colored key sprites"
    for key in keys:
        coords = key["coords"]
        assert not np.any(complete[coords[:, 0], coords[:, 1]]), "keys must sit outside the inferred body"
    return sorted(keys, key=lambda obj: tuple(map(int, obj["bbox"])))


def _compatible(key_shape: Shape, gap_shape: Shape) -> bool:
    target = gap_shape[0]
    return any(_rotate_shape(key_shape, turns)[0] == target for turns in range(4))


def _unique_matching(gaps: list[dict], keys: list[dict]) -> tuple[int, ...]:
    n = len(gaps)
    assert len(keys) == n, "the number of keys must equal the number of gaps"

    gap_shapes = [_shape_from_coords(gap["coords"]) for gap in gaps]
    key_shapes = [_shape_from_coords(key["coords"]) for key in keys]
    compat = [[_compatible(key_shapes[j], gap_shapes[i]) for j in range(n)] for i in range(n)]

    solutions: list[tuple[int, ...]] = []

    def search(row: int, used: set[int], current: list[int]) -> None:
        if len(solutions) > 1:
            return
        if row == n:
            solutions.append(tuple(current))
            return
        for col in range(n):
            if col in used or not compat[row][col]:
                continue
            used.add(col)
            current.append(col)
            search(row + 1, used, current)
            current.pop()
            used.remove(col)

    search(0, set(), [])
    assert len(solutions) == 1, "key-to-gap shape matching must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use rotated gray counterparts to locate missing body gaps, then fill each gap with its matching key color.

    Input structure:
    - The background surrounds one large gray body whose completed cells are 90-degree rotationally symmetric around the body's center.
    - Some body cells are missing as background gaps; for every gap cell, the other three quarter-turn positions are intact gray cells.
    - Outside the body are monochrome colored key sprites, each shaped like one gap under one of the four quarter-turn rotations.
    - Fully symmetric background voids may exist inside the gray body's bbox and are not gaps.

    Rule:
    - Infer the body's center from the gray bbox and rotate every visible gray pixel through the four quarter-turn positions.
    - The inferred body cells that are currently background are the true gaps; symmetric voids remain background because no rotated gray pixel implies them.
    - Match each outside key sprite to exactly one gap by comparing all four rotations of the key shape.
    - Erase all outside keys, fill every matched gap with its key's color, and leave all other body pixels gray.

    Key insight:
    - The missing shape is defined by the three gray rotational counterparts, while the colored key supplies only the color and a rotated verification of that shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    center2, _bbox = _body_center(g)
    complete, gaps = _infer_gap_components(g, bg, center2)
    gaps = sorted(gaps, key=lambda obj: tuple(map(int, obj["bbox"])))
    keys = _key_components(g, bg, complete)
    matching = _unique_matching(gaps, keys)

    out = g.copy()
    out[(g != bg) & (g != BODY)] = bg
    for gap_idx, key_idx in enumerate(matching):
        coords = gaps[gap_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(keys[key_idx]["color"])
    return out


# --------------- generator ---------------


def _build_shape_pool() -> dict[int, tuple[Shape, ...]]:
    """Enumerate small connected polyominoes once instead of rejection-sampling them."""
    current = {frozenset({(0, 0)})}
    pools: dict[int, tuple[Shape, ...]] = {}
    for area in range(2, 7):
        expanded: set[frozenset[Coord]] = set()
        for coords in current:
            for r, c in coords:
                for dr, dc in NEIGHBORS4:
                    if (r + dr, c + dc) in coords:
                        continue
                    candidate = set(coords)
                    candidate.add((r + dr, c + dc))
                    norm, h, w = _normalize_shape(candidate)
                    if h <= 4 and w <= 4:
                        expanded.add(norm)
        current = expanded
        if area >= 3:
            shapes = []
            for coords in sorted(current, key=lambda cells: tuple(sorted(cells))):
                assert len(coords) == area
                shape = _normalize_shape(coords)
                if shape[1] >= 2 and shape[2] >= 2 and len(_shape_variants(shape)) == 4:
                    shapes.append(shape)
            pools[area] = tuple(shapes)
    assert all(pools[area] for area in range(3, 7))
    return pools


GAP_SHAPES_BY_AREA = _build_shape_pool()


def _in_square(cells: set[Coord], box: tuple[int, int, int, int], margin: int = 0) -> bool:
    r0, c0, r1, c1 = box
    return all(r0 + margin <= r <= r1 - margin and c0 + margin <= c <= c1 - margin for r, c in cells)


def _orbit_union(cells: set[Coord], center2: tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for turns in range(4):
        out.update(_rotate_cells(cells, center2, turns))
    return out


def _add_symmetric_voids(body_mask: np.ndarray, center2: tuple[int, int]) -> np.ndarray:
    """Add one of several tiny, guaranteed-safe symmetric void patterns."""
    out = body_mask.copy()
    center = (center2[0] // 2, center2[1] // 2)
    style = random.randrange(4)
    if style in (1, 3):
        out[center] = False
    if style in (2, 3):
        seed = {(center[0] - 2, center[1] - 1)}
        for r, c in _orbit_union(seed, center2):
            out[r, c] = False
    return out


def _adjacent_to(cells: set[Coord], other: set[Coord]) -> bool:
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) in other:
                return True
    return False


def _surrounded_by_body(cells: set[Coord], body_mask: np.ndarray) -> bool:
    h, w = body_mask.shape
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w and body_mask[rr, cc]):
                return False
    return True


def _gap_candidates(
    shape: Shape,
    body_mask: np.ndarray,
    center2: tuple[int, int],
    box: tuple[int, int, int, int],
) -> list[dict]:
    """Return every valid orbit representative for a gap shape."""
    r0, c0, r1, c1 = box
    candidates: list[dict] = []
    for turns in range(4):
        oriented = _rotate_shape(shape, turns)
        coords, h, w = oriented
        for ar in range(r0 + 2, r1 - h):
            for ac in range(c0 + 2, c1 - w):
                cells = {(ar + r, ac + c) for r, c in coords}
                rotations = [_rotate_cells(cells, center2, k) for k in range(4)]
                if any(not all(body_mask[r, c] for r, c in rotated) for rotated in rotations):
                    continue
                if len(set().union(*rotations)) != 4 * len(cells):
                    continue
                if not _surrounded_by_body(cells, body_mask):
                    continue
                candidates.append({"cells": cells, "shape": oriented, "rotations": rotations})
    random.shuffle(candidates)
    return candidates


def _place_gaps(
    body_mask: np.ndarray,
    center2: tuple[int, int],
    box: tuple[int, int, int, int],
    gap_count: int,
) -> list[dict] | None:
    """Choose separated gaps from finite candidate lists with a small bounded retry."""
    areas = random.sample(range(3, 7), gap_count)
    shapes = [random.choice(GAP_SHAPES_BY_AREA[area]) for area in areas]
    candidate_lists = [_gap_candidates(shape, body_mask, center2, box) for shape in shapes]
    if any(not candidates for candidates in candidate_lists):
        return None

    for _ in range(8):
        missing: set[Coord] = set()
        gaps: list[dict] = []
        for candidates in candidate_lists:
            random.shuffle(candidates)
            chosen = None
            for candidate in candidates:
                cells = candidate["cells"]
                rotations = candidate["rotations"]
                if any(rotated & missing for rotated in rotations):
                    continue
                if _adjacent_to(cells, missing):
                    continue
                chosen = candidate
                break
            if chosen is None:
                break
            gaps.append({"cells": chosen["cells"], "shape": chosen["shape"]})
            missing.update(chosen["cells"])
        if len(gaps) == gap_count:
            return gaps
    return None


def _shape_patch(shape: Shape, color: int) -> tuple[np.ndarray, list[Coord]]:
    coords, h, w = shape
    patch = np.full((h, w), BG, dtype=int)
    for r, c in coords:
        patch[r, c] = int(color)
    return patch, sorted(coords)


def _can_place_key(grid: np.ndarray, body_mask: np.ndarray, patch_cells: list[Coord], r0: int, c0: int) -> bool:
    h, w = grid.shape
    translated = {(int(r0 + r), int(c0 + c)) for r, c in patch_cells}
    for r, c in translated:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG or body_mask[r, c]:
            return False
    for r, c in translated:
        for dr, dc in NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in translated:
                continue
            if body_mask[rr, cc] or grid[rr, cc] != BG:
                return False
    return True


def _place_keys(grid: np.ndarray, body_mask: np.ndarray, gaps: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    out = grid.copy()
    colors = random.sample(KEY_COLORS, len(gaps))
    order = list(range(len(gaps)))
    random.shuffle(order)
    forced_rotated = random.choice(order)
    key_infos: list[dict] = []

    for gap_idx in order:
        turns = random.randrange(4)
        if gap_idx == forced_rotated and turns == 0:
            turns = random.randint(1, 3)
        key_shape = _rotate_shape(gaps[gap_idx]["shape"], turns)
        color = colors[gap_idx]
        patch, patch_cells = _shape_patch(key_shape, color)
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(out.shape[0] - ph + 1) for c in range(out.shape[1] - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if not _can_place_key(out, body_mask, patch_cells, r0, c0):
                continue
            for dr, dc in patch_cells:
                out[r0 + dr, c0 + dc] = int(color)
            key_infos.append({"gap_idx": gap_idx, "color": int(color), "turns": int(turns)})
            placed = True
            break
        if not placed:
            return None
    return out, key_infos


def _gen_attempt() -> np.ndarray | None:
    gap_count = random.randint(2, 4)
    side = random.choice((13, 15)) if gap_count < 4 else 15
    height = random.randint(max(25, side + 10), 30)
    width = random.randint(max(25, side + 10), 30)
    top = random.randint(5, height - side - 5)
    left = random.randint(5, width - side - 5)
    box = (top, left, top + side - 1, left + side - 1)
    center = (top + side // 2, left + side // 2)
    center2 = (2 * center[0], 2 * center[1])

    body_mask = np.zeros((height, width), dtype=bool)
    body_mask[top : top + side, left : left + side] = True
    body_mask = _add_symmetric_voids(body_mask, center2)

    gaps = _place_gaps(body_mask, center2, box, gap_count)
    if gaps is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[body_mask] = BODY
    for gap in gaps:
        for r, c in gap["cells"]:
            grid[r, c] = BG

    placed = _place_keys(grid, body_mask, gaps)
    if placed is None:
        return None
    grid, _key_infos = placed
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric key-gap input")
