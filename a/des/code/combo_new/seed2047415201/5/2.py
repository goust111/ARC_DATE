from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
MAROON = 9
RING_COLORS = tuple(range(1, 9))

SIDES = ("top", "bottom", "left", "right")
INWARD = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}


@dataclass(frozen=True)
class Ring:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    gap: tuple[int, int]
    inward: tuple[int, int]
    cells: frozenset[tuple[int, int]]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _component(g: np.ndarray, start_r: int, start_c: int, seen: np.ndarray) -> set[tuple[int, int]]:
    color = int(g[start_r, start_c])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    cells: set[tuple[int, int]] = set()
    h, w = g.shape

    while stack:
        r, c = stack.pop()
        cells.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return cells


def _ring_from_component(g: np.ndarray, cells: set[tuple[int, int]]) -> Ring:
    colors = {int(g[r, c]) for r, c in cells}
    assert len(colors) == 1, "each frame component must be monochrome"
    color = colors.pop()

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frames must have room for an interior"

    perimeter = _perimeter_cells(r0, c0, r1, c1)
    assert cells <= perimeter, "frame components must lie on a rectangle perimeter"
    missing = perimeter - cells
    assert len(missing) == 1, "each rectangular frame must have exactly one gap"
    gap = next(iter(missing))
    gr, gc = gap

    side_hits = []
    if gr == r0 and c0 < gc < c1:
        side_hits.append("top")
    if gr == r1 and c0 < gc < c1:
        side_hits.append("bottom")
    if gc == c0 and r0 < gr < r1:
        side_hits.append("left")
    if gc == c1 and r0 < gr < r1:
        side_hits.append("right")
    assert len(side_hits) == 1, "the gap must be on exactly one non-corner side"
    side = side_hits[0]

    return Ring(
        color=color,
        r0=r0,
        c0=c0,
        r1=r1,
        c1=c1,
        gap=gap,
        inward=INWARD[side],
        cells=frozenset(cells),
    )


def _find_rings(g: np.ndarray) -> list[Ring]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rings: list[Ring] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BLACK:
                continue
            cells = _component(g, r, c, seen)
            rings.append(_ring_from_component(g, cells))
    assert rings, "expected at least one gapped rectangular frame"
    return rings


def _span2(pos: int, limit: int) -> tuple[int, int]:
    assert limit >= 2
    start = min(max(0, int(pos)), limit - 2)
    return start, start + 2


def _marker_box(last_r: int, last_c: int, dr: int, dc: int, h: int, w: int) -> tuple[int, int, int, int]:
    if dr < 0:
        c0, c1 = _span2(last_c, w)
        return 0, c0, 2, c1
    if dr > 0:
        c0, c1 = _span2(last_c, w)
        return h - 2, c0, h, c1
    if dc < 0:
        r0, r1 = _span2(last_r, h)
        return r0, 0, r1, 2
    r0, r1 = _span2(last_r, h)
    return r0, w - 2, r1, w


def _trace_ray(g: np.ndarray, ring: Ring) -> tuple[str, tuple[tuple[int, int], ...], tuple[int, int, int, int] | None]:
    h, w = g.shape
    dr, dc = ring.inward
    r, c = ring.gap
    last_r, last_c = r, c
    path: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w:
        value = int(g[r, c])
        if value == BLACK:
            path.append((r, c))
        elif (r, c) not in ring.cells:
            return "hit", tuple(path), None
        last_r, last_c = r, c
        r += dr
        c += dc

    return "marker", tuple(path), _marker_box(last_r, last_c, dr, dc, h, w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot colored rays inward through the side gaps of hollow rectangular frames.

    Input structure:
    - The background is black. Each foreground object is a one-pixel-thick monochrome
      rectangular ring with exactly one non-corner missing boundary cell.
    - Rings do not overlap; some smaller rings may be nested inside larger rings.

    Rule:
    - For each ring, the missing boundary cell gives the entry gap and the ray travels
      inward, opposite the side's outward opening direction.
    - Paint every black cell traversed by that ray with the source ring color. The
      source ring's own boundary is not a blocker for its own ray.
    - If the ray reaches another ring boundary, it stops before coloring that boundary.
      If it reaches the canvas edge without hitting another ring, draw a 2x2 maroon
      marker block at the exit edge.

    Key insight:
    The gap is an entrance into the frame, not an exit; nested frames matter because
    they are the other objects that can stop a larger frame's incoming ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    rings = _find_rings(g)
    out = g.copy()
    markers: list[tuple[int, int, int, int]] = []

    for ring in rings:
        _status, path, marker = _trace_ray(g, ring)
        for r, c in path:
            out[r, c] = ring.color
        if marker is not None:
            markers.append(marker)

    for r0, c0, r1, c1 in markers:
        out[r0:r1, c0:c1] = MAROON

    return out


def _make_spec(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
    side: str,
    line: int | None = None,
) -> dict:
    if side == "top":
        gap = (r0, int(line) if line is not None else random.randint(c0 + 1, c1 - 1))
    elif side == "bottom":
        gap = (r1, int(line) if line is not None else random.randint(c0 + 1, c1 - 1))
    elif side == "left":
        gap = (int(line) if line is not None else random.randint(r0 + 1, r1 - 1), c0)
    else:
        gap = (int(line) if line is not None else random.randint(r0 + 1, r1 - 1), c1)
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": int(color), "side": side, "gap": gap}


def _draw_specs(h: int, w: int, specs: list[dict]) -> np.ndarray:
    g = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["r0"], spec["c0"], spec["r1"], spec["c1"]
        color = spec["color"]
        g[r0, c0 : c1 + 1] = color
        g[r1, c0 : c1 + 1] = color
        g[r0 : r1 + 1, c0] = color
        g[r0 : r1 + 1, c1] = color
        gr, gc = spec["gap"]
        g[gr, gc] = BLACK
    return g


def _bbox(spec: dict) -> tuple[int, int, int, int]:
    return spec["r0"], spec["c0"], spec["r1"], spec["c1"]


def _disjoint_with_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _nested_pair(h: int, w: int, color_outer: int, color_inner: int) -> list[dict] | None:
    for _ in range(120):
        max_oh = min(18, h - 4)
        max_ow = min(18, w - 4)
        if max_oh < 11 or max_ow < 11:
            return None
        oh = random.randint(11, max_oh)
        ow = random.randint(11, max_ow)
        r0 = random.randint(2, h - oh - 2)
        c0 = random.randint(2, w - ow - 2)
        r1 = r0 + oh - 1
        c1 = c0 + ow - 1

        side = random.choice(SIDES)
        vertical = side in ("top", "bottom")
        line = random.randint(c0 + 3, c1 - 3) if vertical else random.randint(r0 + 3, r1 - 3)

        max_ih = min(7, oh - 4)
        max_iw = min(7, ow - 4)
        if max_ih < 4 or max_iw < 4:
            continue
        ih = random.randint(4, max_ih)
        iw = random.randint(4, max_iw)

        if vertical:
            ic0_min = max(c0 + 2, line - iw + 2)
            ic0_max = min(c1 - iw - 1, line - 1)
            ir0_min = r0 + 2
            ir0_max = r1 - ih - 1
            if ic0_min > ic0_max or ir0_min > ir0_max:
                continue
            ic0 = random.randint(ic0_min, ic0_max)
            ir0 = random.randint(ir0_min, ir0_max)
            inner_side = random.choice(("left", "right"))
        else:
            ir0_min = max(r0 + 2, line - ih + 2)
            ir0_max = min(r1 - ih - 1, line - 1)
            ic0_min = c0 + 2
            ic0_max = c1 - iw - 1
            if ir0_min > ir0_max or ic0_min > ic0_max:
                continue
            ir0 = random.randint(ir0_min, ir0_max)
            ic0 = random.randint(ic0_min, ic0_max)
            inner_side = random.choice(("top", "bottom"))

        ir1 = ir0 + ih - 1
        ic1 = ic0 + iw - 1
        outer = _make_spec(r0, c0, r1, c1, color_outer, side, line)
        inner = _make_spec(ir0, ic0, ir1, ic1, color_inner, inner_side)
        return [outer, inner]
    return None


def _random_disjoint_ring(h: int, w: int, color: int, specs: list[dict]) -> dict | None:
    for _ in range(160):
        rh = random.randint(4, min(8, h - 4))
        rw = random.randint(4, min(8, w - 4))
        r0 = random.randint(2, h - rh - 2)
        c0 = random.randint(2, w - rw - 2)
        spec = _make_spec(r0, c0, r0 + rh - 1, c0 + rw - 1, color, random.choice(SIDES))
        box = _bbox(spec)
        if all(_disjoint_with_gap(box, _bbox(existing), gap=1) for existing in specs):
            return spec
    return None


def _trace_all(g: np.ndarray) -> list[tuple[Ring, str, tuple[tuple[int, int], ...], tuple[int, int, int, int] | None]]:
    return [(ring, *_trace_ray(g, ring)) for ring in _find_rings(g)]


def _candidate_is_marker(g: np.ndarray, candidate: dict) -> bool:
    for ring, status, _path, _marker in _trace_all(g):
        if (
            ring.color == candidate["color"]
            and (ring.r0, ring.c0, ring.r1, ring.c1) == _bbox(candidate)
        ):
            return status == "marker"
    return False


def _generated_clear(
    g: np.ndarray,
    traces: list[tuple[Ring, str, tuple[tuple[int, int], ...], tuple[int, int, int, int] | None]],
) -> bool:
    path_owner: dict[tuple[int, int], int] = {}
    marker_owner: dict[tuple[int, int], int] = {}
    ring_perimeters = [
        _perimeter_cells(ring.r0, ring.c0, ring.r1, ring.c1)
        for ring, _status, _path, _marker in traces
    ]

    for i, (ring, _status, path, _marker) in enumerate(traces):
        for cell in path:
            for j, perimeter in enumerate(ring_perimeters):
                if i != j and cell in perimeter:
                    return False
            if cell in path_owner:
                return False
            path_owner[cell] = i

    for i, (_ring, _status, _path, marker) in enumerate(traces):
        if marker is None:
            continue
        r0, c0, r1, c1 = marker
        if np.any(g[r0:r1, c0:c1] != BLACK):
            return False
        for r in range(r0, r1):
            for c in range(c0, c1):
                cell = (r, c)
                if cell in path_owner and path_owner[cell] != i:
                    return False
                if cell in marker_owner and marker_owner[cell] != i:
                    return False
                marker_owner[cell] = i
    return True


def _add_marker_ring(h: int, w: int, specs: list[dict], color: int) -> dict | None:
    for _ in range(120):
        candidate = _random_disjoint_ring(h, w, color, specs)
        if candidate is None:
            return None
        grid = _draw_specs(h, w, specs + [candidate])
        try:
            traces = _trace_all(grid)
        except AssertionError:
            continue
        if len(traces) != len(specs) + 1:
            continue
        if not _candidate_is_marker(grid, candidate):
            continue
        if not _generated_clear(grid, traces):
            continue
        return candidate
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    colors = list(RING_COLORS)
    random.shuffle(colors)

    specs: list[dict] = []
    color_idx = 0
    use_nested = random.random() < 0.5
    if use_nested:
        pair = _nested_pair(h, w, colors[color_idx], colors[color_idx + 1])
        if pair is None:
            return None
        specs.extend(pair)
        color_idx += 2

    extra_count = random.randint(1, 2 if use_nested else 3)
    for _ in range(extra_count):
        if color_idx >= len(colors):
            break
        ring = _add_marker_ring(h, w, specs, colors[color_idx])
        if ring is None:
            return None
        specs.append(ring)
        color_idx += 1

    grid = _draw_specs(h, w, specs)
    try:
        traces = _trace_all(grid)
        out = main(grid)
    except AssertionError:
        return None

    if len(traces) != len(specs):
        return None
    hit_count = sum(1 for _ring, status, _path, _marker in traces if status == "hit")
    marker_count = sum(1 for _ring, status, _path, _marker in traces if status == "marker")
    if use_nested and hit_count == 0:
        return None
    if not use_nested and hit_count != 0:
        return None
    if marker_count == 0:
        return None
    if not _generated_clear(grid, traces):
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid gapped-frame ray input")
