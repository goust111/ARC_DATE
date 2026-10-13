from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
ANCHOR = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUADRANT_ORDER = ("UL", "UR", "LL", "LR")
ROTATION_K = {"UL": 0, "UR": 1, "LL": 2, "LR": -1}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_from_coords(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_dicts(grid: np.ndarray, forbidden: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) in forbidden:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) not in forbidden:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"coords": coords, "bbox": _bbox_from_coords(coords), "sprite": _crop_from_coords(grid, coords)})
    return comps


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == ANCHOR)
    assert coords.shape == (4, 2), "expected one teal 2x2 anchor"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (int(r1 - r0), int(c1 - c0)) == (1, 1), "anchor must be a 2x2 square"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR), "anchor must be solid"
    return int(r0), int(c0)


def _translate(coords: list[tuple[int, int]], dr: int, dc: int, step: int) -> list[tuple[int, int]]:
    return [(r + dr * step, c + dc * step) for r, c in coords]


def _touches_anchor(coords: list[tuple[int, int]], anchor_cells: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in anchor_cells:
                return True
    return False


def _touches_other(coords: list[tuple[int, int]], other_cells: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in other_cells:
                return True
    return False


def _slide_candidates(
    coords: list[tuple[int, int]],
    anchor_cells: set[tuple[int, int]],
    obstacle_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[dict]:
    h, w = shape
    candidates = []
    for dr, dc in DIRS4:
        for step in range(1, max(h, w) + 1):
            moved = _translate(coords, dr, dc, step)
            moved_set = set(moved)
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in moved):
                break
            if moved_set & obstacle_cells:
                break
            if _touches_other(moved, obstacle_cells - anchor_cells):
                break
            if _touches_anchor(moved, anchor_cells):
                candidates.append({"direction": (dr, dc), "step": step, "coords": moved})
                break
    return candidates


def _quadrant_of(coords: list[tuple[int, int]], anchor_r0: int, anchor_c0: int) -> str:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    top = (r0 + r1) < (2 * anchor_r0 + 1)
    left = (c0 + c1) < (2 * anchor_c0 + 1)
    if top and left:
        return "UL"
    if top and not left:
        return "UR"
    if not top and left:
        return "LL"
    return "LR"


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide four matching sprites into a teal anchor, then rotate them into a pinwheel.

    Input structure:
    - The grid is black except for one central teal 2x2 anchor and four identical square multicolor sprites.
    - There is one sprite in each broad quadrant around the anchor, and each sprite is aligned so a straight cardinal slide can reach the anchor.
    - The slide paths are clear, and each sprite has exactly one direction that leads to first contact with the anchor.

    Rule:
    - For each sprite, slide it straight in its unique valid direction until it first becomes 4-neighbor adjacent to the teal anchor.
    - Remove the teal anchor.
    - Determine the landed sprite's quadrant relative to the former anchor center.
    - Build a tight 2x2 pinwheel from the four landed sprites: upper-left keeps its orientation, upper-right rotates 90 degrees counterclockwise, lower-left rotates 180 degrees, and lower-right rotates 90 degrees clockwise.

    Key insight:
    - The teal 2x2 block is both the collision target for the sliding stage and the coordinate center for the rotation stage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_r0, anchor_c0 = _find_anchor(g)
    anchor_cells = {(anchor_r0, anchor_c0), (anchor_r0, anchor_c0 + 1), (anchor_r0 + 1, anchor_c0), (anchor_r0 + 1, anchor_c0 + 1)}
    comps = _component_dicts(g, {BLACK, ANCHOR})
    assert len(comps) == 4, "expected four sprites"

    sprite = comps[0]["sprite"]
    side = int(sprite.shape[0])
    assert sprite.shape[0] == sprite.shape[1], "expected square sprites"
    assert all(np.array_equal(comp["sprite"], sprite) for comp in comps), "expected identical sprites"

    quadrant_to_sprite: dict[str, np.ndarray] = {}
    all_cells = set()
    for comp in comps:
        all_cells.update(comp["coords"])

    for comp in comps:
        coord_set = set(comp["coords"])
        candidates = _slide_candidates(
            comp["coords"],
            anchor_cells=anchor_cells,
            obstacle_cells=anchor_cells | (all_cells - coord_set),
            shape=g.shape,
        )
        assert len(candidates) == 1, "each sprite must have exactly one valid slide"
        landed = candidates[0]["coords"]
        quadrant = _quadrant_of(landed, anchor_r0, anchor_c0)
        assert quadrant not in quadrant_to_sprite, "landed quadrants must be unique"
        quadrant_to_sprite[quadrant] = comp["sprite"]

    assert set(quadrant_to_sprite) == set(QUADRANT_ORDER), "expected one landed sprite per quadrant"

    out = np.zeros((2 * side, 2 * side), dtype=int)
    for quadrant in QUADRANT_ORDER:
        rotated = np.rot90(quadrant_to_sprite[quadrant], k=ROTATION_K[quadrant])
        if quadrant == "UL":
            _paste(out, rotated, 0, 0)
        elif quadrant == "UR":
            _paste(out, rotated, 0, side)
        elif quadrant == "LL":
            _paste(out, rotated, side, 0)
        else:
            _paste(out, rotated, side, side)
    return out


GEN_MAX_ATTEMPTS = 500
GEN_SIDES = (3, 5)
GEN_GRID_SIZES = tuple(size for size in range(18, 29, 2))
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]


def _make_sprite(side: int) -> np.ndarray | None:
    for _ in range(100):
        color_count = random.choice((2, 3))
        palette = random.sample(GEN_COLORS, color_count)
        sprite = np.array([[random.choice(palette) for _ in range(side)] for _ in range(side)], dtype=int)
        if len(set(map(int, sprite.flat))) < 2:
            continue
        if any(np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3)):
            continue
        return sprite
    return None


def _orth_choices_top(side: int, anchor_r0: int) -> list[int]:
    return [anchor_r0 - side + offset for offset in (1, 2) if (anchor_r0 - side + offset) + (side - 1) < anchor_r0 + 1]


def _orth_choices_bottom(side: int, anchor_r0: int) -> list[int]:
    return [anchor_r0 + offset for offset in (0, 1) if (anchor_r0 + offset) > anchor_r0]


def _orth_choices_left(side: int, anchor_c0: int) -> list[int]:
    return [anchor_c0 - side + offset for offset in (1, 2) if (anchor_c0 - side + offset) + (side - 1) < anchor_c0 + 1]


def _orth_choices_right(side: int, anchor_c0: int) -> list[int]:
    return [anchor_c0 + offset for offset in (0, 1) if (anchor_c0 + offset) > anchor_c0]


def _component_for_quadrant(
    quadrant: str,
    mode: str,
    side: int,
    anchor_r0: int,
    anchor_c0: int,
    gap: int,
) -> tuple[int, int]:
    if quadrant == "UL":
        if mode == "H":
            r = random.choice(_orth_choices_top(side, anchor_r0))
            c_final = anchor_c0 - side
            return r, c_final - gap
        c = random.choice(_orth_choices_left(side, anchor_c0))
        r_final = anchor_r0 - side
        return r_final - gap, c
    if quadrant == "UR":
        if mode == "H":
            r = random.choice(_orth_choices_top(side, anchor_r0))
            c_final = anchor_c0 + 2
            return r, c_final + gap
        c = random.choice(_orth_choices_right(side, anchor_c0))
        r_final = anchor_r0 - side
        return r_final - gap, c
    if quadrant == "LL":
        if mode == "H":
            r = random.choice(_orth_choices_bottom(side, anchor_r0))
            c_final = anchor_c0 - side
            return r, c_final - gap
        c = random.choice(_orth_choices_left(side, anchor_c0))
        r_final = anchor_r0 + 2
        return r_final + gap, c
    if mode == "H":
        r = random.choice(_orth_choices_bottom(side, anchor_r0))
        c_final = anchor_c0 + 2
        return r, c_final + gap
    c = random.choice(_orth_choices_right(side, anchor_c0))
    r_final = anchor_r0 + 2
    return r_final + gap, c


def _in_bounds(top: int, left: int, side: int, size: int) -> bool:
    return 0 <= top and 0 <= left and top + side <= size and left + side <= size


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0)


def _sprite_bbox(top: int, left: int, side: int) -> tuple[int, int, int, int]:
    return top, left, top + side - 1, left + side - 1


def _render_instance(size: int, sprite: np.ndarray, placements: dict[str, tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    anchor_r0 = size // 2 - 1
    anchor_c0 = size // 2 - 1
    grid[anchor_r0 : anchor_r0 + 2, anchor_c0 : anchor_c0 + 2] = ANCHOR
    side = int(sprite.shape[0])
    for top, left in placements.values():
        grid[top : top + side, left : left + side] = sprite
    return grid


def _oracle_ok(grid: np.ndarray, sprite: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    side = int(sprite.shape[0])
    if out.shape != (2 * side, 2 * side):
        return False
    expected = np.zeros((2 * side, 2 * side), dtype=int)
    expected[:side, :side] = sprite
    expected[:side, side:] = np.rot90(sprite, 1)
    expected[side:, :side] = np.rot90(sprite, 2)
    expected[side:, side:] = np.rot90(sprite, -1)
    if not np.array_equal(out, expected):
        return False

    anchor_r0, anchor_c0 = _find_anchor(grid)
    comps = _component_dicts(grid, {BLACK, ANCHOR})
    if len(comps) != 4 or not all(np.array_equal(comp["sprite"], sprite) for comp in comps):
        return False

    anchor_cells = {(anchor_r0, anchor_c0), (anchor_r0, anchor_c0 + 1), (anchor_r0 + 1, anchor_c0), (anchor_r0 + 1, anchor_c0 + 1)}
    all_cells = set()
    for comp in comps:
        all_cells.update(comp["coords"])

    quadrants = set()
    mode_signatures = set()
    for comp in comps:
        coord_set = set(comp["coords"])
        candidates = _slide_candidates(comp["coords"], anchor_cells, anchor_cells | (all_cells - coord_set), grid.shape)
        if len(candidates) != 1:
            return False
        dr, dc = candidates[0]["direction"]
        mode_signatures.add("V" if dr != 0 else "H")
        quadrants.add(_quadrant_of(candidates[0]["coords"], anchor_r0, anchor_c0))
    return quadrants == set(QUADRANT_ORDER) and mode_signatures == {"H", "V"}


def _attempt_generate() -> np.ndarray | None:
    side = random.choice(GEN_SIDES)
    sprite = _make_sprite(side)
    if sprite is None:
        return None

    size = random.choice([n for n in GEN_GRID_SIZES if n >= 2 * side + 8])
    anchor_r0 = size // 2 - 1
    anchor_c0 = size // 2 - 1

    modes = {quadrant: random.choice(("H", "V")) for quadrant in QUADRANT_ORDER}
    if len(set(modes.values())) == 1:
        modes[random.choice(QUADRANT_ORDER)] = "H" if next(iter(modes.values())) == "V" else "V"

    placements: dict[str, tuple[int, int]] = {}
    bboxes: list[tuple[int, int, int, int]] = []
    for quadrant in QUADRANT_ORDER:
        candidates: list[tuple[int, int]] = []
        for gap in range(1, size):
            top, left = _component_for_quadrant(quadrant, modes[quadrant], side, anchor_r0, anchor_c0, gap)
            if _in_bounds(top, left, side, size):
                bbox = _sprite_bbox(top, left, side)
                if not any(_bboxes_overlap(bbox, other, padding=1) for other in bboxes):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[quadrant] = (top, left)
        bboxes.append(_sprite_bbox(top, left, side))

    grid = _render_instance(size, sprite, placements)
    if not _oracle_ok(grid, sprite):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-pinwheel input")
