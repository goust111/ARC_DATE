from __future__ import annotations

import hashlib
import itertools
import random
from collections import deque

import numpy as np


BG = 0
LINE = 1
SEP = 2
MASK = 7
SIDE = 11
MAX_ATTEMPTS = 600

_SEEN_OUTPUT_SIGS: set[str] = set()


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _split_halves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    separator_cols = np.flatnonzero(np.all(grid == SEP, axis=0))
    assert len(separator_cols) == 1, "expected one full-height red separator column"
    c = int(separator_cols[0])
    left = grid[:, :c]
    right = grid[:, c + 1 :]
    assert left.shape == right.shape and left.shape == (SIDE, SIDE), "expected two aligned 11x11 halves"
    assert set(map(int, np.unique(left))) <= {BG, LINE}, "left half must contain only black and blue"
    assert set(map(int, np.unique(right))) <= {BG, MASK}, "right half must contain only black and orange"
    return left, right


def _full_bar_positions(line_mask: np.ndarray, box: BBox) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = box
    patch = line_mask[r0 : r1 + 1, c0 : c1 + 1]
    horizontal = [r0 + i for i in range(patch.shape[0]) if np.all(patch[i, :])]
    vertical = [c0 + j for j in range(patch.shape[1]) if np.all(patch[:, j])]

    assert r0 in horizontal and r1 in horizontal, "line drawing must contain top and bottom boundary bars"
    assert c0 in vertical and c1 in vertical, "line drawing must contain left and right boundary bars"
    assert len(horizontal) >= 2 and len(vertical) >= 2

    recon = np.zeros_like(patch, dtype=bool)
    recon[[r - r0 for r in horizontal], :] = True
    recon[:, [c - c0 for c in vertical]] = True
    assert np.array_equal(patch, recon), "line drawing must be composed only of full rows and full columns"
    return horizontal, vertical


def _classify_control(coord: Coord, box: BBox) -> str | None:
    r, c = coord
    r0, c0, r1, c1 = box
    hits: list[str] = []
    if r0 <= r <= r1 and c < c0:
        hits.append("left")
    if r0 <= r <= r1 and c > c1:
        hits.append("right")
    if c0 <= c <= c1 and r < r0:
        hits.append("top")
    if c0 <= c <= c1 and r > r1:
        hits.append("bottom")
    assert len(hits) <= 1, "control points must lie on exactly one side extension line"
    return hits[0] if hits else None


def _stretch_rows_cols(
    rows: list[int], cols: list[int], old_box: BBox, controls: dict[str, Coord]
) -> tuple[list[int], list[int], BBox]:
    old_r0, old_c0, old_r1, old_c1 = old_box
    new_r0 = controls["top"][0] if "top" in controls else old_r0
    new_r1 = controls["bottom"][0] if "bottom" in controls else old_r1
    new_c0 = controls["left"][1] if "left" in controls else old_c0
    new_c1 = controls["right"][1] if "right" in controls else old_c1
    assert 0 <= new_r0 <= old_r0 <= old_r1 <= new_r1 < SIDE
    assert 0 <= new_c0 <= old_c0 <= old_c1 <= new_c1 < SIDE

    new_rows = [new_r0 if r == old_r0 else new_r1 if r == old_r1 else r for r in rows]
    new_cols = [new_c0 if c == old_c0 else new_c1 if c == old_c1 else c for c in cols]
    assert len(new_rows) == len(set(new_rows))
    assert len(new_cols) == len(set(new_cols))
    return new_rows, new_cols, (new_r0, new_c0, new_r1, new_c1)


def _render_line(shape: tuple[int, int], box: BBox, rows: list[int], cols: list[int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    out = np.zeros(shape, dtype=int)
    for r in rows:
        out[r, c0 : c1 + 1] = LINE
    for c in cols:
        out[r0 : r1 + 1, c] = LINE
    return out


def _solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    line_mask = left == LINE
    assert np.count_nonzero(line_mask) > 0, "expected a blue line drawing"
    assert len(_components4(line_mask)) == 1, "line drawing must be 4-connected"

    old_box = _bbox(line_mask)
    assert old_box is not None
    rows, cols = _full_bar_positions(line_mask, old_box)

    diff_coords = [tuple(map(int, rc)) for rc in np.argwhere((right == MASK) & (~line_mask))]
    assert len(diff_coords) >= 2, "expected at least two difference controls"

    controls: dict[str, Coord] = {}
    for coord in diff_coords:
        side = _classify_control(coord, old_box)
        assert side is not None, "difference pixels must lie on a bbox side extension line"
        assert side not in controls, "expected at most one control per side"
        controls[side] = coord
    assert len(controls) == len(diff_coords), "all difference pixels must become side controls"

    new_rows, new_cols, new_box = _stretch_rows_cols(rows, cols, old_box, controls)
    return _render_line(left.shape, new_box, new_rows, new_cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the right-half mask-minus-line difference pixels as stretch controls for the left rectilinear line drawing.

    Input structure:
    - An 11x23 grid has black background, a full-height red divider column, an 11x11 blue rectilinear line drawing on the left, and an aligned 11x11 orange mask on the right.
    - The blue drawing is a single connected union of full horizontal bars and full vertical bars inside its own bounding box.
    - Some orange mask pixels overlap blue line pixels under alignment, while the remaining orange pixels lie on side-extension lines outside the blue bbox.

    Rule:
    - Split at the red divider and align the two 11x11 halves cell by cell.
    - Compute the set difference `mask - line`: orange pixels that do not overlap blue become the control points, with at most one control per side and at least two distinct sides.
    - Move each controlled outer boundary bar of the blue line drawing to the control point's row or column, keep interior bars fixed, and redraw the full horizontal and vertical bars across the stretched bounding box.

    Key insight:
    - The orange mask is not used directly: only the non-overlapping part of the aligned mask encodes which sides expand and where the moved outer bars land.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (SIDE, SIDE * 2 + 1), "expected an 11x23 grid"
    left, right = _split_halves(g)
    return _solve(left, right)


def _all_line_coords(box: BBox, rows: list[int], cols: list[int]) -> list[Coord]:
    r0, c0, r1, c1 = box
    pts: set[Coord] = set()
    for r in rows:
        for c in range(c0, c1 + 1):
            pts.add((r, c))
    for c in cols:
        for r in range(r0, r1 + 1):
            pts.add((r, c))
    return sorted(pts)


def _choose_bar_positions(length: int, with_interior: bool) -> list[int]:
    assert length >= 4
    if not with_interior:
        return [0, length - 1]
    return [0, random.randint(1, length - 2), length - 1]


def _choose_side_subset(available: dict[str, int]) -> tuple[str, ...]:
    candidates: list[tuple[str, ...]] = []
    sides = [side for side, room in available.items() if room > 0]
    for size in (2, 3, 4):
        for subset in itertools.combinations(sides, size):
            candidates.append(tuple(subset))
    assert candidates

    def weight(subset: tuple[str, ...]) -> int:
        score = 1
        if "top" in subset:
            score += 2
        if len(subset) in (2, 3):
            score += 2
        return score

    weighted: list[tuple[str, ...]] = []
    for subset in candidates:
        weighted.extend([subset] * weight(subset))
    return random.choice(weighted)


def _side_boundary_point(side: str, old_box: BBox) -> Coord:
    return random.choice(_side_boundary_coords(side, old_box))


def _side_boundary_coords(side: str, old_box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = old_box
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(side)


def _pick_overlaps(
    old_box: BBox, rows: list[int], cols: list[int], controlled_sides: tuple[str, ...]
) -> set[Coord]:
    all_line = _all_line_coords(old_box, rows, cols)
    overlaps: set[Coord] = set()
    uncontrolled = [side for side in ("top", "bottom", "left", "right") if side not in controlled_sides]
    if uncontrolled:
        overlaps.add(_side_boundary_point(random.choice(uncontrolled), old_box))
    for side in controlled_sides:
        if random.random() < 0.75:
            overlaps.add(_side_boundary_point(side, old_box))

    interior_rows = [r for r in rows if r not in (old_box[0], old_box[2])]
    interior_cols = [c for c in cols if c not in (old_box[1], old_box[3])]
    for r in interior_rows:
        overlaps.add((r, random.randint(old_box[1], old_box[3])))
    for c in interior_cols:
        overlaps.add((random.randint(old_box[0], old_box[2]), c))

    target_count = random.randint(max(2, len(overlaps)), min(8, len(all_line)))
    while len(overlaps) < target_count:
        overlaps.add(random.choice(all_line))
    return overlaps


def _make_example() -> tuple[np.ndarray, np.ndarray, dict]:
    old_h = random.randint(4, 8)
    old_w = random.randint(4, 8)
    old_r0 = random.randint(0, SIDE - old_h)
    old_c0 = random.randint(0, SIDE - old_w)
    old_r1 = old_r0 + old_h - 1
    old_c1 = old_c0 + old_w - 1
    old_box = (old_r0, old_c0, old_r1, old_c1)

    available = {
        "top": old_r0,
        "bottom": SIDE - 1 - old_r1,
        "left": old_c0,
        "right": SIDE - 1 - old_c1,
    }
    if sum(room > 0 for room in available.values()) < 2:
        raise ValueError("not enough expandable sides")

    with_h_interior = random.random() < 0.5
    with_v_interior = random.random() < 0.5
    if not (with_h_interior or with_v_interior):
        if random.random() < 0.5:
            with_h_interior = True
        else:
            with_v_interior = True

    row_offsets = _choose_bar_positions(old_h, with_h_interior)
    col_offsets = _choose_bar_positions(old_w, with_v_interior)
    rows = [old_r0 + off for off in row_offsets]
    cols = [old_c0 + off for off in col_offsets]
    old_line = _render_line((SIDE, SIDE), old_box, rows, cols)

    controlled_sides = _choose_side_subset(available)
    controls: dict[str, Coord] = {}
    for side in controlled_sides:
        if side == "top":
            controls[side] = (old_r0 - random.randint(1, available[side]), random.randint(old_c0, old_c1))
        elif side == "bottom":
            controls[side] = (old_r1 + random.randint(1, available[side]), random.randint(old_c0, old_c1))
        elif side == "left":
            controls[side] = (random.randint(old_r0, old_r1), old_c0 - random.randint(1, available[side]))
        else:
            controls[side] = (random.randint(old_r0, old_r1), old_c1 + random.randint(1, available[side]))

    new_rows, new_cols, new_box = _stretch_rows_cols(rows, cols, old_box, controls)
    expected = _render_line((SIDE, SIDE), new_box, new_rows, new_cols)

    overlaps = _pick_overlaps(old_box, rows, cols, controlled_sides)
    right = np.zeros((SIDE, SIDE), dtype=int)
    for r, c in controls.values():
        right[r, c] = MASK
    for r, c in overlaps:
        right[r, c] = MASK

    left = old_line.copy()
    grid = np.concatenate([left, np.full((SIDE, 1), SEP, dtype=int), right], axis=1)
    meta = {
        "old_box": old_box,
        "rows": rows,
        "cols": cols,
        "controls": controls,
        "controlled_sides": controlled_sides,
        "expected": expected,
        "overlaps": overlaps,
    }
    return grid, expected, meta


def _validate_example(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    left, right = _split_halves(grid)
    line_mask = left == LINE
    diff = (right == MASK) & (~line_mask)
    overlap = (right == MASK) & line_mask
    if int(np.count_nonzero(diff)) != len(meta["controls"]):
        return False
    if int(np.count_nonzero(overlap)) < 2:
        return False

    # Reject examples where every orange pixel is a control-like outer point.
    if np.count_nonzero(right == MASK) == np.count_nonzero(diff):
        return False

    rows = meta["rows"]
    cols = meta["cols"]
    if len(rows) == 2 and len(cols) == 2:
        return False

    uncontrolled = [side for side in ("top", "bottom", "left", "right") if side not in meta["controlled_sides"]]
    if uncontrolled:
        old_box = meta["old_box"]
        found = False
        for side in uncontrolled:
            if any(right[r, c] == MASK and left[r, c] == LINE for r, c in _side_boundary_coords(side, old_box)):
                found = True
                break
        if not found:
            return False

    out_sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
    if out_sig in _SEEN_OUTPUT_SIGS:
        return False
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected, meta = _make_example()
        except ValueError:
            continue
        if _validate_example(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
