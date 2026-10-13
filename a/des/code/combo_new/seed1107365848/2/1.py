import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OBJECT_COUNTS = (1, 2, 2, 3, 3, 4)
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _frame_from_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    comp_set = set(comp)
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp_set):
        return None

    border = _border_coords(r0, c0, r1, c1)
    border_set = set(border)
    if len(comp_set) != len(border_set) - 1:
        return None

    missing = [(r, c) for r, c in border if int(grid[r, c]) == BLACK]
    if len(missing) != 1:
        return None
    gap = missing[0]

    for r, c in border:
        if (r, c) == gap:
            continue
        if int(grid[r, c]) != GREY:
            return None

    side_hits = {
        "up": gap[0] == r0 and c0 < gap[1] < c1,
        "down": gap[0] == r1 and c0 < gap[1] < c1,
        "left": gap[1] == c0 and r0 < gap[0] < r1,
        "right": gap[1] == c1 and r0 < gap[0] < r1,
    }
    side_names = [name for name, ok in side_hits.items() if ok]
    if len(side_names) != 1:
        return None
    side = side_names[0]
    dr, dc = DIRS[side]

    gate = (gap[0] + dr, gap[1] + dc)
    if not _in_bounds(grid.shape, gate[0], gate[1]):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.any(inner == GREY):
        return None
    marker_rel = np.argwhere((inner != BLACK) & (inner != GREY))
    if marker_rel.shape[0] != 1:
        return None
    mr, mc = map(int, marker_rel[0])
    marker = (r0 + 1 + mr, c0 + 1 + mc)
    color = int(grid[marker])
    if color in (BLACK, GREY):
        return None
    if int(grid[gate]) != color:
        return None

    if dr != 0 and marker[1] != gap[1]:
        return None
    if dc != 0 and marker[0] != gap[0]:
        return None

    dist = abs(marker[0] - gap[0]) + abs(marker[1] - gap[1])
    if dist < 1:
        return None

    inside_ray: list[tuple[int, int]] = []
    for step in range(1, dist + 1):
        rr = marker[0] + dr * step
        cc = marker[1] + dc * step
        if not _in_bounds(grid.shape, rr, cc):
            return None
        if step < dist and int(grid[rr, cc]) != BLACK:
            return None
        inside_ray.append((rr, cc))

    outside_ray: list[tuple[int, int]] = []
    for step in range(dist + 2, 2 * dist + 2):
        rr = marker[0] + dr * step
        cc = marker[1] + dc * step
        if not _in_bounds(grid.shape, rr, cc):
            return None
        if int(grid[rr, cc]) != BLACK:
            return None
        outside_ray.append((rr, cc))

    return {
        "bbox": (r0, c0, r1, c1),
        "marker": marker,
        "gap": gap,
        "gate": gate,
        "color": color,
        "inside_ray": inside_ray,
        "outside_ray": outside_ray,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a grey escape ray from each inner marker through its frame gap and continue the same distance outside.

    Input structure:
    - The background is black.
    - One or more grey one-cell-thick rectangular frames appear, each with exactly one
      single-cell gap on one side.
    - Each frame encloses black interior cells plus one isolated colored marker pixel.
    - The cell immediately outside the gap is a same-colored clue pixel.

    Rule:
    - For each gapped frame, the gap side fixes a cardinal direction.
    - The inner marker is aligned with the gap; let D be the straight Manhattan distance
      from the marker to that gap.
    - Draw grey cells from the marker toward the gap until the gap is filled.
    - Remove the outside clue pixel to black, then continue the grey ray beyond it for
      another D cells.
    - Keep the colored marker itself unchanged.

    Key insight:
    The outside same-colored pixel only reveals where the border gap is; the real rule is
    to mirror the marker-to-gap distance outside the frame after erasing that clue pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames: list[dict] = []
    for comp in _components(g == GREY):
        frame = _frame_from_component(g, comp)
        if frame is not None:
            frames.append(frame)

    assert frames, "expected at least one valid gapped grey frame"

    out = g.copy()
    for frame in sorted(frames, key=lambda item: item["bbox"]):
        out[frame["gate"]] = BLACK
        for rr, cc in frame["inside_ray"]:
            out[rr, cc] = GREY
        for rr, cc in frame["outside_ray"]:
            out[rr, cc] = GREY
        out[frame["marker"]] = frame["color"]
    return out


def _sample_object_count() -> int:
    return random.choice(OBJECT_COUNTS)


def _sample_grid_shape(n_objects: int) -> tuple[int, int]:
    side_lo = max(10, 6 + 4 * n_objects)
    side_hi = min(30, side_lo + 8)
    return random.randint(side_lo, side_hi), random.randint(side_lo, side_hi)


def _candidate_spec(
    shape: tuple[int, int],
    reserved: np.ndarray,
    color: int,
    direction: str,
) -> dict | None:
    h, w = shape
    max_inner_h = min(7, h - 4)
    max_inner_w = min(7, w - 4)
    if max_inner_h < 2 or max_inner_w < 2:
        return None

    dr, dc = DIRS[direction]
    for _ in range(500):
        inner_h = random.randint(2, max_inner_h)
        inner_w = random.randint(2, max_inner_w)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if direction == "up":
            dist = random.randint(1, inner_h)
            offset = random.randint(0, inner_w - 1)
            marker_local = (dist, offset + 1)
            gap_local = (0, offset + 1)
            r_lo, r_hi = dist + 1, h - outer_h
            c_lo, c_hi = 0, w - outer_w
        elif direction == "down":
            dist = random.randint(1, inner_h)
            offset = random.randint(0, inner_w - 1)
            marker_local = (inner_h - dist + 1, offset + 1)
            gap_local = (outer_h - 1, offset + 1)
            r_lo, r_hi = 0, h - outer_h - dist - 1
            c_lo, c_hi = 0, w - outer_w
        elif direction == "left":
            dist = random.randint(1, inner_w)
            offset = random.randint(0, inner_h - 1)
            marker_local = (offset + 1, dist)
            gap_local = (offset + 1, 0)
            r_lo, r_hi = 0, h - outer_h
            c_lo, c_hi = dist + 1, w - outer_w
        else:
            dist = random.randint(1, inner_w)
            offset = random.randint(0, inner_h - 1)
            marker_local = (offset + 1, inner_w - dist + 1)
            gap_local = (offset + 1, outer_w - 1)
            r_lo, r_hi = 0, h - outer_h
            c_lo, c_hi = 0, w - outer_w - dist - 1

        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        marker = (r0 + marker_local[0], c0 + marker_local[1])
        gap = (r0 + gap_local[0], c0 + gap_local[1])
        gate = (gap[0] + dr, gap[1] + dc)
        outside_ray = [
            (marker[0] + dr * step, marker[1] + dc * step)
            for step in range(dist + 2, 2 * dist + 2)
        ]
        if not _in_bounds(shape, gate[0], gate[1]):
            continue
        if any(not _in_bounds(shape, rr, cc) for rr, cc in outside_ray):
            continue

        reserve_core = {
            (rr, cc)
            for rr in range(r0, r1 + 1)
            for cc in range(c0, c1 + 1)
        }
        reserve_core.add(gate)
        reserve_core.update(outside_ray)
        if any(reserved[rr, cc] for rr, cc in reserve_core):
            continue

        border = set(_border_coords(r0, c0, r1, c1))
        border.remove(gap)
        inside_ray = [
            (marker[0] + dr * step, marker[1] + dc * step)
            for step in range(1, dist + 1)
        ]

        return {
            "bbox": (r0, c0, r1, c1),
            "color": color,
            "marker": marker,
            "gap": gap,
            "gate": gate,
            "border": border,
            "inside_ray": inside_ray,
            "outside_ray": outside_ray,
            "reserve": _expand_cells(reserve_core, shape),
        }

    return None


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        for rr, cc in spec["border"]:
            grid[rr, cc] = GREY
        grid[spec["marker"]] = spec["color"]
        grid[spec["gate"]] = spec["color"]
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        out[spec["gate"]] = BLACK
        for rr, cc in spec["inside_ray"]:
            out[rr, cc] = GREY
        for rr, cc in spec["outside_ray"]:
            out[rr, cc] = GREY
        out[spec["marker"]] = spec["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = _sample_object_count()
    shape = _sample_grid_shape(n_objects)
    colors = random.sample(MARKER_COLORS, n_objects)
    directions = random.sample(tuple(DIRS.keys()), n_objects)

    reserved = np.zeros(shape, dtype=bool)
    specs: list[dict] = []
    for color, direction in zip(colors, directions):
        spec = _candidate_spec(shape, reserved, color, direction)
        if spec is None:
            return None
        for rr, cc in spec["reserve"]:
            reserved[rr, cc] = True
        specs.append(spec)

    inp = _render_input(shape, specs)
    expected = _render_expected(inp, specs)

    try:
        actual = main(inp)
    except Exception:
        return None

    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
