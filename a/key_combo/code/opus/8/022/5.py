from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds


BG = Color.BLACK
DIVIDER = Color.GREY
MOVER = Color.ORANGE
LEGEND = np.array([[Color.RED, Color.BLUE], [Color.GREEN, Color.YELLOW]], dtype=int)

GEN_RETRIES = 400


def _divider_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.ravel()))
    foreground = [color for color in counts if color != bg]
    assert foreground, "expected divider and legend colors"
    best = max(counts[color] for color in foreground)
    winners = [color for color in foreground if counts[color] == best]
    assert len(winners) == 1, "expected one dominant divider color"
    return int(winners[0])


def _find_scene(g: np.ndarray) -> tuple[int, int, np.ndarray]:
    bg = get_bg(g)
    divider = _divider_color(g, bg)
    h, w = g.shape

    rows = [r for r in range(h) if np.all(g[r, :] == divider)]
    cols = [c for c in range(w) if np.all(g[:, c] == divider)]
    assert len(rows) == 1 and len(cols) == 1, "expected one full horizontal and one full vertical divider"
    hr = rows[0]
    vc = cols[0]

    legend_mask = (g != bg) & (g != divider) & (g != MOVER)
    boxes: list[tuple[int, int, int, int]] = []
    for coords in components_mask(legend_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        if coords.shape[0] != 4:
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        if r0 in {0, h - 2} and c0 in {0, w - 2}:
            boxes.append((r0, c0, r1, c1))
    assert len(boxes) == 1, "expected one 2x2 corner legend"
    r0, c0, r1, c1 = boxes[0]
    legend = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all(legend != bg) and np.all(legend != divider) and np.all(legend != MOVER)
    return hr, vc, legend


def _quadrant_color(legend: np.ndarray, hr: int, vc: int, r: int, c: int, dc: int) -> int:
    qr = 0 if r < hr else 1
    if c < vc:
        qc = 0
    elif c > vc:
        qc = 1
    else:
        qc = 0 if dc < 0 else 1
    return int(legend[qr, qc])


def _ray_cells(base: np.ndarray, hr: int, vc: int, legend: np.ndarray, start: tuple[int, int], dc: int) -> list[tuple[int, int, int]]:
    h, w = base.shape
    r, c = start
    rr = r - 1
    cc = c + dc
    cells: list[tuple[int, int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        cell = int(base[rr, cc])
        if cc == vc and rr != hr and cell == DIVIDER:
            rr -= 1
            cc += dc
            continue
        if cell != BG:
            break
        cells.append((rr, cc, _quadrant_color(legend, hr, vc, rr, cc, dc)))
        rr -= 1
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each orange pixel downward and emit upward diagonal splashes whose colors are chosen from the quadrant legend as the rays travel.

    Input structure:
    - The grid has a black background, one full gray horizontal divider row, one full gray vertical divider column, and a single 2x2 corner legend.
    - The legend gives the quadrant colors in top-left, top-right, bottom-left, bottom-right order.
    - Two or three isolated orange moving pixels lie above the horizontal divider and never occupy the vertical divider column.

    Rule:
    - Erase every orange source pixel and redraw it one row lower in the same column.
    - If the moved pixel now sits immediately above the horizontal gray divider, launch one up-left and one up-right diagonal ray from that landing pixel.
    - Each ray advances one step at a time; black cells are painted with the legend color of the ray's current quadrant, while the vertical divider itself stays gray and is treated as a transparent boundary crossing.
    - A ray stops before leaving the grid or before entering any non-black cell other than that transparent vertical-divider crossing; keep the divider and legend unchanged.

    Key insight:
    - Splash colors are not fixed by the orange source: they are looked up dynamically from the 2x2 legend using the ray's current position, so a single ray can change color after crossing the vertical quadrant boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    hr, vc, legend = _find_scene(g)

    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == MOVER)]
    assert 1 <= len(sources) <= 3, "expected a small set of moving pixels"
    assert all(r < hr and c != vc for r, c in sources), "moving pixels must be above the trigger row and off the divider column"

    out = g.copy()
    out[g == MOVER] = BG

    moved: list[tuple[int, int]] = []
    for r, c in sources:
        nr = r + 1
        assert in_bounds(g.shape, nr, c), "moved pixel left the grid"
        assert nr != hr, "moving pixel cannot land inside the divider row"
        assert out[nr, c] == BG, "landing cell must be empty"
        out[nr, c] = MOVER
        moved.append((nr, c))

    base = out.copy()
    for nr, c in moved:
        if nr + 1 != hr:
            continue
        for dc in (-1, 1):
            for rr, cc, color in _ray_cells(base, hr, vc, legend, (nr, c), dc):
                out[rr, cc] = color

    return out


def _planned_ray_trace(hr: int, vc: int, start: tuple[int, int], dc: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    rr = r - 1
    cc = c + dc
    trace: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        if cc != vc:
            trace.append((rr, cc))
        rr -= 1
        cc += dc
    return trace


def _choose_nontrigger_column(
    h: int,
    w: int,
    hr: int,
    vc: int,
    used_sources: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> tuple[int, int] | None:
    rows = [r for r in range(max(1, hr - 4), hr - 2)]
    if not rows:
        return None
    candidates: list[tuple[int, int]] = []
    for r in rows:
        for c in range(w):
            moved = (r + 1, c)
            src = (r, c)
            if c == vc:
                continue
            if src in used_sources or moved in blocked or src in blocked:
                continue
            if c < 2 and r < 3:
                continue
            candidates.append(src)
    if not candidates:
        return None
    return random.choice(candidates)


def _parse_input(g: np.ndarray) -> tuple[int, int, np.ndarray, list[tuple[int, int]]]:
    hr, vc, legend = _find_scene(g)
    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == MOVER)]
    return hr, vc, legend, sources


def _validate_input_structure(g: np.ndarray) -> None:
    hr, vc, legend, sources = _parse_input(g)
    assert np.array_equal(legend, LEGEND), "generator should use the fixed quadrant legend"
    assert 2 <= len(sources) <= 3, "generator must create multiple movers"
    assert all(hr - 4 <= r <= hr - 2 for r, _c in sources), "sources must lie 2-4 rows above the divider"
    assert any(r == hr - 2 for r, _c in sources), "at least one source must trigger"
    assert any(r < hr - 2 for r, _c in sources), "at least one source must remain non-triggering"
    traces: list[set[tuple[int, int]]] = []
    for r, c in sources:
        if r != hr - 2:
            continue
        landing = (r + 1, c)
        trace = set(_planned_ray_trace(hr, vc, landing, -1, g.shape) + _planned_ray_trace(hr, vc, landing, 1, g.shape))
        assert trace, "triggered pixels should have visible rays"
        for other in sources:
            assert other not in trace, "rays must not pass through other moving pixels"
        traces.append(trace)
    for i, left in enumerate(traces):
        for right in traces[i + 1 :]:
            assert left.isdisjoint(right), "distinct splash rays should not overlap"


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    hr, vc, legend, sources = _parse_input(inp)
    assert np.array_equal(inp[:2, :2], legend), "legend must remain in the corner"
    assert np.array_equal(out[:2, :2], legend), "legend changed unexpectedly"
    assert np.all(out[hr, :] == DIVIDER) and np.all(out[:, vc] == DIVIDER), "divider changed unexpectedly"

    base = inp.copy()
    base[base == MOVER] = BG
    for r, c in sources:
        base[r + 1, c] = MOVER

    triggered_cross = 0
    for r, c in sources:
        assert out[r, c] == BG, "source cell should be cleared"
        moved = (r + 1, c)
        assert out[moved] == MOVER, "moved pixel missing"
        if r != hr - 2:
            continue
        left = _ray_cells(base, hr, vc, legend, moved, -1)
        right = _ray_cells(base, hr, vc, legend, moved, 1)
        painted_colors = {color for _rr, _cc, color in left + right}
        assert painted_colors, "triggered mover should paint at least one ray cell"
        if any(cc < vc for _rr, cc, _color in right) and any(cc > vc for _rr, cc, _color in right):
            triggered_cross += 1
        if any(cc > vc for _rr, cc, _color in left) and any(cc < vc for _rr, cc, _color in left):
            triggered_cross += 1
        for rr, cc, color in left + right:
            assert out[rr, cc] == color, "ray color mismatch"
    assert triggered_cross >= 1, "at least one triggered ray should cross the vertical divider"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    hr = random.randint(6, h - 4)
    vc = random.randint(4, w - 5)

    grid = np.full((h, w), BG, dtype=int)
    grid[hr, :] = DIVIDER
    grid[:, vc] = DIVIDER
    grid[:2, :2] = LEGEND

    trigger_count = 1 if random.random() < 0.45 else 2
    left_cols = list(range(max(2, vc - 3), vc))
    right_cols = list(range(vc + 1, min(w - 2, vc + 4)))
    if not left_cols or not right_cols:
        return None

    sources: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()

    if trigger_count == 2:
        left_c = random.choice(left_cols)
        right_c = random.choice(right_cols)
        sources.extend([(hr - 2, left_c), (hr - 2, right_c)])
    else:
        side = random.choice(("left", "right"))
        col = random.choice(left_cols if side == "left" else right_cols)
        sources.append((hr - 2, col))

    for r, c in sources:
        landing = (r + 1, c)
        blocked.add(landing)
        blocked.update(_planned_ray_trace(hr, vc, landing, -1, grid.shape))
        blocked.update(_planned_ray_trace(hr, vc, landing, 1, grid.shape))

    nontrigger = _choose_nontrigger_column(h, w, hr, vc, set(sources), blocked)
    if nontrigger is None:
        return None
    sources.append(nontrigger)

    if len({c for _r, c in sources}) != len(sources):
        return None

    for r, c in sources:
        if grid[r, c] != BG or c == vc:
            return None
        grid[r, c] = MOVER

    try:
        _validate_input_structure(grid)
        out = main(grid)
        _validate_output_semantics(grid, out)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
