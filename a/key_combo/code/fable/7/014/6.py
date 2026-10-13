"""ARC task 6: reunite complementary jagged fragments."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N,2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _valid_destinations(
    grid: np.ndarray,
    large: set[tuple[int, int]],
    small: set[tuple[int, int]],
    background: int,
) -> list[tuple[int, int]]:
    """Enumerate translations in the stated full-grid coordinate system."""
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = _bbox(small)
    answers = []
    for dr in range(-sr0, h - sr1):
        for dc in range(-sc0, w - sc1):
            if dr == 0 and dc == 0:
                continue
            moved = {(r + dr, c + dc) for r, c in small}
            if moved & large:
                continue
            # The destination must be empty in the input.  Thus another colored
            # object cannot be used as an accidental part of the rectangle.
            if any(int(grid[r, c]) != background for r, c in moved):
                continue
            if _is_solid_rectangle(large | moved):
                answers.append((dr, dc))
    return answers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each smaller same-color jagged fragment into its larger mate and recolor the moved fragment grey.

    Input structure:
    - A black background contains two separate 4-connected fragments of each foreground color.
    - In every color pair the fragments have unequal areas and complementary jagged edges; neither fragment's bbox is the completed rectangle.
    - Exactly one in-bounds translation of the smaller fragment avoids all input objects and makes its union with the larger fragment a solid rectangle.

    Rule:
    - For each color, leave the larger fragment fixed, erase the smaller fragment from its original position, and translate it by that unique displacement.
    - Paint every translated cell grey; preserve all fixed fragments and all remaining background cells.

    Key insight:
    - Pair by visible color, then recover the displacement from the global constraint that the two exact masks must interlock without overlap into a hole-free solid rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"
    assert int(np.count_nonzero(g == BACKGROUND)) > g.size // 2, "expected black dominant background"
    assert GREY not in set(map(int, np.unique(g))), "grey is reserved for moved fragments"

    colors = [int(x) for x in np.unique(g) if int(x) != BACKGROUND]
    assert colors, "expected at least one fragment pair"
    moves: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []

    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form exactly two fragments"
        assert len(comps[0]) != len(comps[1]), "the smaller fragment must be unique"
        comps.sort(key=len, reverse=True)
        large = {tuple(map(int, p)) for p in comps[0]}
        small = {tuple(map(int, p)) for p in comps[1]}
        candidates = _valid_destinations(g, large, small, BACKGROUND)
        assert len(candidates) == 1, "each pair must have one unambiguous rectangular interlock"
        dr, dc = candidates[0]
        moved = {(r + dr, c + dc) for r, c in small}
        moves.append((small, moved))

    destinations: set[tuple[int, int]] = set()
    for _, moved in moves:
        assert not destinations & moved, "moved fragments may not collide"
        destinations |= moved

    out = g.copy()
    for small, _ in moves:
        for r, c in small:
            out[r, c] = BACKGROUND
    for _, moved in moves:
        for r, c in moved:
            out[r, c] = GREY
    return out


def _jagged_partition() -> tuple[np.ndarray, np.ndarray, str]:
    """Split a rectangle along a varying seam; both sides remain connected."""
    for _ in range(200):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        orientation = random.choice(("vertical", "horizontal"))
        seam_len, depth = (h, w) if orientation == "vertical" else (w, h)
        cuts = [random.randint(1, depth - 1)]
        for _ in range(1, seam_len):
            choices = [x for x in range(1, depth) if abs(x - cuts[-1]) <= 2]
            cuts.append(random.choice(choices))
        # Three depths and a reversal make a visibly jagged, non-monotone seam.
        diffs = np.diff(cuts)
        if len(set(cuts)) < 3 or not (np.any(diffs > 0) and np.any(diffs < 0)):
            continue
        a = np.zeros((h, w), dtype=bool)
        if orientation == "vertical":
            for r, cut in enumerate(cuts):
                a[r, :cut] = True
        else:
            for c, cut in enumerate(cuts):
                a[:cut, c] = True
        b = ~a
        if int(a.sum()) == int(b.sum()):
            continue
        if len(_components(a)) == len(_components(b)) == 1:
            return a, b, orientation
    raise RuntimeError("failed to make a non-degenerate jagged partition")


def _expanded(box: tuple[int, int, int, int], h: int, w: int, pad: int = 1):
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _free_with_padding(occupied: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _intrinsic_offsets(large: set[tuple[int, int]], small: set[tuple[int, int]]):
    """All solid-union offsets without canvas or distractor restrictions."""
    lr0, lc0, lr1, lc1 = _bbox(large)
    sr0, sc0, sr1, sc1 = _bbox(small)
    found = []
    # Rectangle intervals cannot have a gap in either axis.
    for dr in range(lr0 - sr1 - 1, lr1 - sr0 + 2):
        for dc in range(lc0 - sc1 - 1, lc1 - sc0 + 2):
            moved = {(r + dr, c + dc) for r, c in small}
            if not (moved & large) and _is_solid_rectangle(large | moved):
                found.append((dr, dc))
    return found


def _attempt_generate() -> np.ndarray | None:
    h, w = random.randint(22, 30), random.randint(22, 30)
    pair_count = random.randint(2, 3)
    colors = random.sample(PALETTE, pair_count)
    specs = []
    orientations = []
    for color in colors:
        a, b, orientation = _jagged_partition()
        if a.sum() > b.sum():
            large_mask, small_mask = a, b
        else:
            large_mask, small_mask = b, a
        specs.append({"color": color, "large": large_mask, "small": small_mask})
        orientations.append(orientation)
    # Every generated grid demonstrates both seam orientations.
    if len(set(orientations)) < 2:
        return None

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    occupied = np.zeros((h, w), dtype=bool)
    records = []

    # Place the completed-rectangle sites first and reserve their entire area.
    for spec in specs:
        rh, rw = spec["large"].shape
        choices = [(r, c) for r in range(1, h - rh) for c in range(1, w - rw)]
        random.shuffle(choices)
        placed = None
        for r0, c0 in choices:
            er0, ec0, er1, ec1 = _expanded((r0, c0, r0 + rh - 1, c0 + rw - 1), h, w)
            if not reserved[er0 : er1 + 1, ec0 : ec1 + 1].any():
                placed = (r0, c0)
                reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True
                break
        if placed is None:
            return None
        r0, c0 = placed
        large_local = {tuple(map(int, p)) for p in np.argwhere(spec["large"])}
        small_local = {tuple(map(int, p)) for p in np.argwhere(spec["small"])}
        large = {(r0 + r, c0 + c) for r, c in large_local}
        dest = {(r0 + r, c0 + c) for r, c in small_local}
        for r, c in large:
            grid[r, c] = spec["color"]
            occupied[r, c] = True
        records.append({"spec": spec, "large": large, "dest": dest, "rect_origin": (r0, c0)})

    # Put each small piece elsewhere, outside every reserved target rectangle.
    random.shuffle(records)
    for rec in records:
        local_raw = np.argwhere(rec["spec"]["small"])
        mr, mc = local_raw.min(axis=0)
        local = {(int(r - mr), int(c - mc)) for r, c in local_raw}
        ph = max(r for r, _ in local) + 1
        pw = max(c for _, c in local) + 1
        choices = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
        random.shuffle(choices)
        source = None
        for r0, c0 in choices:
            cells = {(r0 + r, c0 + c) for r, c in local}
            if any(reserved[r, c] for r, c in cells):
                continue
            if _free_with_padding(occupied, cells):
                source = cells
                break
        if source is None:
            return None
        rec["source"] = source
        for r, c in source:
            grid[r, c] = rec["spec"]["color"]
            occupied[r, c] = True

    # At least one small piece is spatially closer to a wrong-color large piece,
    # making proximity an explicit counterexample to the visible color pairing.
    centers = []
    for rec in records:
        rr = [r for r, _ in rec["large"]]
        cc = [c for _, c in rec["large"]]
        centers.append((sum(rr) / len(rr), sum(cc) / len(cc)))
    interleaved = False
    for i, rec in enumerate(records):
        sr = sum(r for r, _ in rec["source"]) / len(rec["source"])
        sc = sum(c for _, c in rec["source"]) / len(rec["source"])
        distances = [abs(sr - r) + abs(sc - c) for r, c in centers]
        if min(range(len(distances)), key=distances.__getitem__) != i:
            interleaved = True
    if not interleaved:
        return None

    # Geometry-only uniqueness: distractors must not manufacture uniqueness.
    for rec in records:
        large = rec["large"]
        source = rec["source"]
        # Compare sets because arbitrary set iteration cannot directly identify dc.
        offsets = _intrinsic_offsets(large, source)
        matching = [(dr, dc) for dr, dc in offsets if {(r + dr, c + dc) for r, c in source} == rec["dest"]]
        if len(offsets) != 1 or len(matching) != 1:
            return None

    expected_grid = grid.copy()
    for rec in records:
        for r, c in rec["source"]:
            expected_grid[r, c] = BACKGROUND
        for r, c in rec["dest"]:
            expected_grid[r, c] = GREY
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected_grid):
        return None

    # Semantic anti-shortcut checks: grey extends outside the stationary piece's
    # bbox, so filling that bbox cannot reproduce the answer; every source moved.
    for rec in records:
        lb = _bbox(rec["large"])
        if not any(not (lb[0] <= r <= lb[2] and lb[1] <= c <= lb[3]) for r, c in rec["dest"]):
            return None
        if rec["source"] == rec["dest"]:
            return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked inputs with two or three color pairs."""
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid broken-mirror input")
