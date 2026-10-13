from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
MARKER_COLORS = {RED, BLUE, GREEN}
DEFAULT_LEGEND = {RED: "H", BLUE: "V", GREEN: "R"}
OPS = ("H", "V", "R")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _find_rectangles(g: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for comp in _components(g == GRAY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
            continue
        perim = _perimeter_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        if not comp_set <= perim:
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == GRAY):
            continue

        marks = [(r, c, int(g[r, c])) for r, c in perim if int(g[r, c]) in MARKER_COLORS]
        if len(marks) != 1:
            continue
        marker = marks[0]
        allowed = {GRAY, marker[2]}
        if any(int(g[r, c]) not in allowed for r, c in perim):
            continue
        if len(comp_set) != len(perim) - 1:
            continue
        rectangles.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "marker": (int(marker[0]), int(marker[1]), int(marker[2])),
            }
        )
    return sorted(rectangles, key=lambda item: item["bbox"])


def _op_patch(seed: np.ndarray, op: str, row_diff: bool, col_diff: bool) -> np.ndarray:
    if op == "H":
        return np.flipud(seed) if row_diff else seed.copy()
    if op == "V":
        return np.fliplr(seed) if col_diff else seed.copy()
    if op == "R":
        patch = seed
        if row_diff:
            patch = np.flipud(patch)
        if col_diff:
            patch = np.fliplr(patch)
        return patch.copy()
    raise AssertionError(f"unknown operation {op!r}")


def _tile_from_seed(seed: np.ndarray, seed_qr: int, seed_qc: int, op: str) -> np.ndarray:
    qh, qw = seed.shape
    out = np.full((qh * 2, qw * 2), BG, dtype=int)
    for qr in (0, 1):
        for qc in (0, 1):
            patch = _op_patch(seed, op, qr != seed_qr, qc != seed_qc)
            out[qr * qh : (qr + 1) * qh, qc * qw : (qc + 1) * qw] = patch
    return out


def _sample_for_op(op: str) -> np.ndarray:
    base = np.array([[4, 6], [7, 9]], dtype=int)
    return _tile_from_seed(base, 0, 0, op)


def _classify_sample(block: np.ndarray) -> str | None:
    if block.shape != (4, 4):
        return None
    if np.any(block == BG) or np.any(block == GRAY):
        return None
    if any(np.any(block == color) for color in MARKER_COLORS):
        return None

    base = block[:2, :2]
    matches = [op for op in OPS if np.array_equal(block, _tile_from_seed(base, 0, 0, op))]
    return matches[0] if len(matches) == 1 else None


def _read_legend(g: np.ndarray, rectangles: list[dict]) -> dict[int, str]:
    h, w = g.shape
    blocked = np.zeros(g.shape, dtype=bool)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    mapping: dict[int, str] = {}
    marker_points = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere(~blocked)
        if int(g[r, c]) in MARKER_COLORS
    ]
    for mr, mc, color in marker_points:
        found: set[str] = set()
        for r0 in range(mr - 4, mr + 2):
            for c0 in range(mc - 4, mc + 2):
                if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
                    continue
                if r0 <= mr < r0 + 4 and c0 <= mc < c0 + 4:
                    continue
                if np.any(blocked[r0 : r0 + 4, c0 : c0 + 4]):
                    continue
                adjacent = (
                    (r0 <= mr < r0 + 4 and mc in (c0 - 1, c0 + 4))
                    or (c0 <= mc < c0 + 4 and mr in (r0 - 1, r0 + 4))
                )
                if not adjacent:
                    continue
                op = _classify_sample(g[r0 : r0 + 4, c0 : c0 + 4])
                if op is not None:
                    found.add(op)
        if len(found) == 1:
            mapping[color] = found.pop()

    if set(mapping) == MARKER_COLORS and set(mapping.values()) == set(OPS):
        return mapping
    return DEFAULT_LEGEND.copy()


def _seed_quadrant(inner: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = inner.shape
    assert h % 2 == 0 and w % 2 == 0, "rectangle interior must split into four equal quadrants"
    qh, qw = h // 2, w // 2
    counts = []
    for qr in (0, 1):
        for qc in (0, 1):
            quad = inner[qr * qh : (qr + 1) * qh, qc * qw : (qc + 1) * qw]
            counts.append((int(np.count_nonzero(quad != BG)), qr, qc))
    counts.sort(reverse=True)
    assert counts[0][0] > 0, "each region needs a nonempty seed pattern"
    assert len(counts) == 4 and counts[1][0] == 0, "only one quadrant may contain the seed"
    _count, qr, qc = counts[0]
    return inner[qr * qh : (qr + 1) * qh, qc * qw : (qc + 1) * qw].copy(), int(qr), int(qc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray-bounded rectangle by tiling its corner seed according to the marker decoded from the edge legend.

    Input structure:
    - The background is black, and color 5 draws several one-pixel rectangular closed boundaries.
    - Each rectangle has exactly one red, blue, or green marker pixel replacing a boundary cell.
    - Inside each rectangle, exactly one quadrant contains a small colored seed pattern; the other three quadrants are black.
    - A side-edge legend shows three 4x4 symmetry samples, each adjacent to the marker color for that operation.

    Rule:
    - Read the legend to map marker colors to horizontal-band, vertical-band, or center/diagonal tiling.
    - For every rectangle, keep the gray boundary and its marker fixed, find the occupied seed quadrant, and copy reflected versions into the other quadrants.
    - The horizontal operation mirrors the seed across the horizontal midline and repeats that column; the vertical operation mirrors across the vertical midline and repeats that row.
    - The center/diagonal operation places the seed, its horizontal and vertical flips, and its 180-degree rotation across the four quadrants.

    Key insight: The boundary marker selects a symmetry rule through the separate legend, while the actual content to repeat is the corner seed inside each rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    rectangles = _find_rectangles(g)
    assert rectangles, "expected at least one gray-bounded rectangle"
    legend = _read_legend(g, rectangles)

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        marker_color = rect["marker"][2]
        assert marker_color in legend, "rectangle marker must appear in the legend"
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed, seed_qr, seed_qc = _seed_quadrant(inner)
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_from_seed(seed, seed_qr, seed_qc, legend[marker_color])

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
RECT_COUNT_CHOICES = (3, 3, 3, 4, 5)
SEED_COLORS = (4, 6, 7, 8, 9)
OP_TO_MARKER = {"H": RED, "V": BLUE, "R": GREEN}


def _paint_legend(grid: np.ndarray, occ: np.ndarray, side: str) -> None:
    h, w = grid.shape
    items = list(OPS)
    random.shuffle(items)

    if side in ("top", "bottom"):
        total_w = len(items) * 5 + (len(items) - 1)
        start_c = random.randint(0, w - total_w)
        r0 = 0 if side == "top" else h - 4
        for i, op in enumerate(items):
            c0 = start_c + i * 6
            grid[r0 : r0 + 4, c0 : c0 + 4] = _sample_for_op(op)
            grid[r0 + random.choice((1, 2)), c0 + 4] = OP_TO_MARKER[op]
        if side == "top":
            occ[:5, :] = True
        else:
            occ[h - 5 :, :] = True
    else:
        total_h = len(items) * 5 + (len(items) - 1)
        start_r = random.randint(0, h - total_h)
        c0 = 0 if side == "left" else w - 4
        for i, op in enumerate(items):
            r0 = start_r + i * 6
            grid[r0 : r0 + 4, c0 : c0 + 4] = _sample_for_op(op)
            marker_c = c0 + 4 if side == "left" else c0 - 1
            grid[r0 + random.choice((1, 2)), marker_c] = OP_TO_MARKER[op]
        if side == "left":
            occ[:, :5] = True
        else:
            occ[:, w - 5 :] = True


def _random_seed(qh: int, qw: int) -> np.ndarray | None:
    color_count = random.randint(2, min(4, len(SEED_COLORS)))
    colors = random.sample(SEED_COLORS, color_count)
    seed = np.array([[random.choice(colors) for _ in range(qw)] for _ in range(qh)], dtype=int)
    if len(set(map(int, seed.reshape(-1)))) < 2:
        return None
    if np.array_equal(seed, np.flipud(seed)):
        return None
    if np.array_equal(seed, np.fliplr(seed)):
        return None
    if np.array_equal(seed, np.rot90(seed, 2)):
        return None
    return seed


def _can_place(occ: np.ndarray, r0: int, c0: int, rh: int, rw: int) -> bool:
    h, w = occ.shape
    if r0 < 0 or c0 < 0 or r0 + rh > h or c0 + rw > w:
        return False
    r_start = max(0, r0 - 1)
    r_end = min(h, r0 + rh + 1)
    c_start = max(0, c0 - 1)
    c_end = min(w, c0 + rw + 1)
    return not bool(np.any(occ[r_start:r_end, c_start:c_end]))


def _draw_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    qh: int,
    qw: int,
    op: str,
) -> np.ndarray:
    rh, rw = 2 * qh + 2, 2 * qw + 2
    r1, c1 = r0 + rh - 1, c0 + rw - 1
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY

    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        mr, mc = r0, random.randint(c0 + 1, c1 - 1)
    elif side == "bottom":
        mr, mc = r1, random.randint(c0 + 1, c1 - 1)
    elif side == "left":
        mr, mc = random.randint(r0 + 1, r1 - 1), c0
    else:
        mr, mc = random.randint(r0 + 1, r1 - 1), c1
    grid[mr, mc] = OP_TO_MARKER[op]

    seed = None
    for _ in range(30):
        seed = _random_seed(qh, qw)
        if seed is not None:
            break
    if seed is None:
        raise RuntimeError("failed to build an asymmetric seed")

    seed_qr, seed_qc = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
    ir0 = r0 + 1 + seed_qr * qh
    ic0 = c0 + 1 + seed_qc * qw
    grid[ir0 : ir0 + qh, ic0 : ic0 + qw] = seed
    expected_inner = _tile_from_seed(seed, seed_qr, seed_qc, op)
    return expected_inner


def _place_rectangles(grid: np.ndarray, occ: np.ndarray, count: int) -> np.ndarray | None:
    expected = grid.copy()
    ops = list(OPS)
    while len(ops) < count:
        ops.append(random.choice(OPS))
    random.shuffle(ops)

    for op in ops:
        placed = False
        for _ in range(120):
            if count >= 5:
                qh = random.randint(2, 3)
                qw = random.randint(2, 3)
            elif count == 4:
                qh = random.randint(2, 3)
                qw = random.randint(2, 4)
            else:
                qh = random.randint(2, 4)
                qw = random.randint(2, 5)
            rh, rw = 2 * qh + 2, 2 * qw + 2
            r0 = random.randint(0, grid.shape[0] - rh)
            c0 = random.randint(0, grid.shape[1] - rw)
            if not _can_place(occ, r0, c0, rh, rw):
                continue

            before = grid.copy()
            inner = _draw_rectangle(grid, r0, c0, qh, qw, op)
            expected[r0 : r0 + rh, c0 : c0 + rw] = grid[r0 : r0 + rh, c0 : c0 + rw]
            expected[r0 + 1 : r0 + rh - 1, c0 + 1 : c0 + rw - 1] = inner
            occ[max(0, r0 - 1) : min(grid.shape[0], r0 + rh + 1), max(0, c0 - 1) : min(grid.shape[1], c0 + rw + 1)] = True
            placed = True
            break
            grid[:] = before
        if not placed:
            return None
    return expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(24, 30)
    width = random.randint(24, 30)
    side = random.choice(("top", "bottom", "left", "right"))
    grid = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)
    _paint_legend(grid, occ, side)

    count = random.choice(RECT_COUNT_CHOICES)
    expected = _place_rectangles(grid, occ, count)
    if expected is None:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
