from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _shortest_cycle(seq: list[int]) -> tuple[int, ...]:
    n = len(seq)
    assert n >= 2, "legend strip is too short"
    for period in range(2, n):
        if all(seq[i] == seq[i % period] for i in range(n)):
            cycle = tuple(int(x) for x in seq[:period])
            assert len(set(cycle)) == period, "cycle colors must be distinct within one period"
            assert n > period, "legend strip must show more than one period"
            return cycle
    raise AssertionError("legend strip must contain a repeated cycle")


def _parse_strip(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if coords.shape[0] != h * w:
        return None
    if h == 1:
        seq = [int(x) for x in grid[r0, c0 : c1 + 1]]
        orientation = "H"
    elif w == 1:
        seq = [int(x) for x in grid[r0 : r1 + 1, c0]]
        orientation = "V"
    else:
        return None
    if bg in seq or len(set(seq)) < 2:
        return None
    cycle = _shortest_cycle(seq)
    return {
        "bbox": (r0, c0, r1, c1),
        "orientation": orientation,
        "sequence": tuple(seq),
        "cycle": cycle,
    }


def _parse_structure(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or coords.shape[0] != h * w:
        return None

    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if np.any(sub == bg):
        return None

    colors = sorted(set(map(int, sub.ravel())))
    if len(colors) != 2:
        return None

    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    inner = ~border

    border_colors = set(map(int, sub[border]))
    inner_colors = set(map(int, sub[inner]))
    if len(border_colors) != 1 or len(inner_colors) != 1:
        return None

    frame_color = int(next(iter(border_colors)))
    inner_color = int(next(iter(inner_colors)))
    if frame_color == inner_color:
        return None

    return {
        "outer_bbox": (r0, c0, r1, c1),
        "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "frame_color": frame_color,
        "inner_color": inner_color,
    }


def _lock_cycle(strip_cycle: tuple[int, ...], inner_color: int, frame_color: int) -> tuple[int, ...]:
    period = len(strip_cycle)
    assert period >= 3, "cycle direction must be disambiguated by at least three colors"
    idx = strip_cycle.index(inner_color)
    next_ok = strip_cycle[(idx + 1) % period] == frame_color
    prev_ok = strip_cycle[(idx - 1) % period] == frame_color
    assert next_ok ^ prev_ok, "inner-to-frame step must lock exactly one direction"
    step = 1 if next_ok else -1
    return tuple(int(strip_cycle[(idx + step * k) % period]) for k in range(period))


def _expanded_mask(shape: tuple[int, int], inner_bbox: tuple[int, int, int, int], layer: int) -> np.ndarray:
    h, w = map(int, shape)
    r0, c0, r1, c1 = inner_bbox
    rr0 = max(0, int(r0) - int(layer))
    cc0 = max(0, int(c0) - int(layer))
    rr1 = min(h - 1, int(r1) + int(layer))
    cc1 = min(w - 1, int(c1) + int(layer))
    mask = np.zeros((h, w), dtype=bool)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _render_rings(grid: np.ndarray, inner_bbox: tuple[int, int, int, int], locked_cycle: tuple[int, ...], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    prev = _expanded_mask(out.shape, inner_bbox, 1)
    for layer in range(2, out.shape[0] + out.shape[1] + 3):
        curr = _expanded_mask(out.shape, inner_bbox, layer)
        add = curr & ~prev
        if not np.any(add):
            break
        out[add & (out == bg)] = int(locked_cycle[layer % len(locked_cycle)])
        prev = curr
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    comps = _components(g, bg)
    assert len(comps) == 2, "expected exactly one legend strip and one frame structure"

    strips: list[dict] = []
    structures: list[dict] = []
    for comp in comps:
        strip = _parse_strip(g, comp, bg)
        if strip is not None:
            strips.append(strip)
        structure = _parse_structure(g, comp, bg)
        if structure is not None:
            structures.append(structure)

    assert len(strips) == 1, "expected one one-pixel legend strip"
    assert len(structures) == 1, "expected one two-color rectangular frame"

    strip = strips[0]
    structure = structures[0]
    locked_cycle = _lock_cycle(strip["cycle"], structure["inner_color"], structure["frame_color"])
    return {
        "bg": bg,
        "strip": strip,
        "structure": structure,
        "locked_cycle": locked_cycle,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue clipped concentric calendar rings from a two-color frame using the phase locked by a sampled legend strip.

    Input structure:
    - The background is dominant.
    - One foreground object is a one-pixel-wide straight legend strip whose colors are a repeated consecutive sample from a cycle.
    - The other foreground object is a filled rectangle made of a thickness-1 outer frame around a solid inner rectangle.
    - The inner color and frame color both belong to the cycle and are adjacent within it.

    Rule:
    - Extract the shortest repeating color cycle from the legend strip.
    - Use the inner-to-frame step to choose the only cycle direction where inner color is layer 0 and frame color is layer 1.
    - Starting from layer 2, expand the inner rectangle outward by one cell per layer; each visible clipped layer is the newly covered area between consecutive expanded rectangles.
    - Color those layers by successive cycle colors, keep the original strip unchanged, and stop when another expansion adds no visible cells.

    Key insight:
    - The strip reveals the periodic palette but not its orientation; the frame itself fixes the phase and direction because outer frame must be exactly one step after the inner core.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _render_rings(grid, info["structure"]["inner_bbox"], info["locked_cycle"], info["bg"])


# --------------- generator ---------------

BG_COLOR = 0
GRID_MIN = 14
GRID_MAX = 30
PERIOD_MIN = 3
PERIOD_MAX = 6
INNER_H_MIN = 1
INNER_H_MAX = 4
INNER_W_MIN = 1
INNER_W_MAX = 5
MAX_ATTEMPTS = 500


def _randint(lo: int, hi: int) -> int:
    return int(random.randint(int(lo), int(hi)))


def _biased_origin(limit: int, span: int) -> int:
    hi = limit - span
    assert hi >= 0
    if hi == 0:
        return 0
    if random.random() < 0.6:
        edge = _randint(0, min(2, hi))
        return edge if random.random() < 0.5 else hi - edge
    return _randint(0, hi)


def _strip_options(
    shape: tuple[int, int],
    outer_bbox: tuple[int, int, int, int],
    orientation: str,
    strip_len: int,
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = outer_bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    options: list[tuple[int, int]] = []
    if orientation == "H":
        for row in range(h):
            for col in range(w - strip_len + 1):
                if rr0 <= row <= rr1 and not (col + strip_len - 1 < cc0 or col > cc1):
                    continue
                options.append((row, col))
    else:
        for row in range(h - strip_len + 1):
            for col in range(w):
                if cc0 <= col <= cc1 and not (row + strip_len - 1 < rr0 or row > rr1):
                    continue
                options.append((row, col))
    return options


def _build_input(
    shape: tuple[int, int],
    cycle: tuple[int, ...],
    inner_idx: int,
    inner_bbox: tuple[int, int, int, int],
    strip_orientation: str,
    strip_start: tuple[int, int],
    strip_dir: int,
    strip_phase: int,
    strip_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    ir0, ic0, ir1, ic1 = inner_bbox
    inner_color = int(cycle[inner_idx])
    frame_color = int(cycle[(inner_idx + 1) % len(cycle)])
    grid[ir0 - 1 : ir1 + 2, ic0 - 1 : ic1 + 2] = frame_color
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = inner_color

    sr, sc = strip_start
    for i in range(strip_len):
        color = int(cycle[(strip_phase + strip_dir * i) % len(cycle)])
        if strip_orientation == "H":
            grid[sr, sc + i] = color
        else:
            grid[sr + i, sc] = color

    locked = tuple(int(cycle[(inner_idx + k) % len(cycle)]) for k in range(len(cycle)))
    expected = _render_rings(grid, inner_bbox, locked, BG_COLOR)
    return grid, expected


def _validate_constructed(
    inp: np.ndarray,
    expected: np.ndarray,
    cycle: tuple[int, ...],
    inner_idx: int,
    inner_bbox: tuple[int, int, int, int],
) -> bool:
    if np.array_equal(inp, expected):
        return False

    info = _analyze(inp)
    locked = info["locked_cycle"]
    want_locked = tuple(int(cycle[(inner_idx + k) % len(cycle)]) for k in range(len(cycle)))
    if locked != want_locked:
        return False
    if info["structure"]["inner_bbox"] != inner_bbox:
        return False

    strip_cycle = info["strip"]["cycle"]
    if len(strip_cycle) != len(cycle):
        return False

    inner_color = int(cycle[inner_idx])
    frame_color = int(cycle[(inner_idx + 1) % len(cycle)])
    pos = strip_cycle.index(inner_color)
    alt_locked = tuple(int(strip_cycle[(pos - k) % len(strip_cycle)]) for k in range(len(strip_cycle)))
    if alt_locked[1] == frame_color:
        return False
    if locked[2] == inner_color or locked[2] == frame_color:
        return False

    outer_mask = _expanded_mask(inp.shape, inner_bbox, 1)
    layer2_mask = _expanded_mask(inp.shape, inner_bbox, 2) & ~outer_mask
    if not np.any(layer2_mask):
        return False
    if not np.any(expected[layer2_mask] == locked[2]):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt() -> np.ndarray | None:
    h = _randint(GRID_MIN, GRID_MAX)
    w = _randint(GRID_MIN, GRID_MAX)

    period = _randint(PERIOD_MIN, PERIOD_MAX)
    cycle = tuple(int(x) for x in random.sample(range(1, 10), period))
    inner_idx = _randint(0, period - 1)

    inner_h = _randint(INNER_H_MIN, INNER_H_MAX)
    inner_w = _randint(INNER_W_MIN, INNER_W_MAX)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if outer_h > h - 2 or outer_w > w - 2:
        return None

    top = _biased_origin(h, outer_h)
    left = _biased_origin(w, outer_w)
    inner_bbox = (top + 1, left + 1, top + inner_h, left + inner_w)
    outer_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)

    strip_orientation = "H" if random.random() < 0.5 else "V"
    max_len = (w if strip_orientation == "H" else h) - 1
    lo = period + 2
    hi = min(max_len, period * 3 + 2)
    if lo > hi:
        return None
    strip_len = _randint(lo, hi)
    options = _strip_options((h, w), outer_bbox, strip_orientation, strip_len)
    if not options:
        return None

    # Bias a fraction of strips toward edges for clearer clipped examples.
    if random.random() < 0.6:
        edge_opts = []
        if strip_orientation == "H":
            for row, col in options:
                if row in (0, h - 1) or col == 0 or col + strip_len == w:
                    edge_opts.append((row, col))
        else:
            for row, col in options:
                if col in (0, w - 1) or row == 0 or row + strip_len == h:
                    edge_opts.append((row, col))
        if edge_opts:
            options = edge_opts

    strip_start = random.choice(options)
    strip_dir = 1 if random.random() < 0.5 else -1
    strip_phase = _randint(0, period - 1)

    inp, expected = _build_input(
        (h, w),
        cycle,
        inner_idx,
        inner_bbox,
        strip_orientation,
        strip_start,
        strip_dir,
        strip_phase,
        strip_len,
    )
    if not _validate_constructed(inp, expected, cycle, inner_idx, inner_bbox):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
