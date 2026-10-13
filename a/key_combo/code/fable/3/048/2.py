from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
RED = 2
OBJECT_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _source_axis(g: np.ndarray) -> tuple[str, int]:
    src = np.argwhere(g == RED)
    assert len(src) == 1, "expected exactly one red edge source pixel"
    r, c = map(int, src[0])
    h, w = g.shape
    on_top_bottom = r == 0 or r == h - 1
    on_left_right = c == 0 or c == w - 1
    assert on_top_bottom ^ on_left_right, "red source must lie on exactly one non-corner edge"
    if on_left_right:
        return "row", r
    return "col", c


def _components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color in (BACKGROUND, RED) or seen[sr, sc]:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            comps.append({"color": color, "coords": frozenset(coords)})

    comps.sort(key=lambda comp: min(comp["coords"]))
    return comps


def _shift(coords: frozenset[tuple[int, int]], dr: int, dc: int, shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    moved = []
    for r, c in coords:
        rr, cc = r + dr, c + dc
        assert 0 <= rr < h and 0 <= cc < w, "shifted object must remain in bounds"
        moved.append((rr, cc))
    return frozenset(moved)


def _center_projection(coords: frozenset[tuple[int, int]], dr: int, dc: int) -> float:
    return sum(r * dr + c * dc for r, c in coords) / len(coords)


def _ahead(
    mover: frozenset[tuple[int, int]],
    other: frozenset[tuple[int, int]],
    dr: int,
    dc: int,
) -> bool:
    return _center_projection(other, dr, dc) >= _center_projection(mover, dr, dc)


def _resolve_pushes(units: list[dict], initial_movers: list[tuple[int, tuple[int, int]]], shape: tuple[int, int]) -> None:
    q: deque[tuple[int, tuple[int, int]]] = deque(initial_movers)
    steps = 0
    max_steps = shape[0] * shape[1] * max(1, len(units))

    while q:
        mover_idx, (dr, dc) = q.popleft()
        while True:
            mover_coords = units[mover_idx]["coords"]
            hits = []
            for idx, unit in enumerate(units):
                if idx == mover_idx:
                    continue
                if mover_coords & unit["coords"]:
                    assert _ahead(mover_coords, unit["coords"], dr, dc), "overlap must be in the push direction"
                    hits.append(idx)
            if not hits:
                break

            hits.sort(key=lambda idx: (_center_projection(units[idx]["coords"], dr, dc), idx), reverse=True)
            for hit_idx in hits:
                old_vec = units[hit_idx]["vec"]
                assert old_vec is None or old_vec == (dr, dc), "an object cannot be pushed in two directions"
                units[hit_idx]["coords"] = _shift(units[hit_idx]["coords"], dr, dc, shape)
                units[hit_idx]["vec"] = (dr, dc)
                q.append((hit_idx, (dr, dc)))
                steps += 1
                assert steps <= max_steps, "push chain did not settle"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split objects struck by an edge-born ray and propagate the resulting push chain.

    Input structure:
    - A black background contains same-color 4-connected solid foreground objects.
    - Exactly one red source pixel lies on a non-corner edge; its row or column is the full ray path.
    - Some objects cross the ray, while other intact objects may sit directly in a sliding half's path.

    Rule:
    - For a horizontal ray, erase object pixels on the ray row, move pixels above it up one cell, and move pixels below it down one cell.
    - For a vertical ray, erase object pixels on the ray column, move pixels left of it left one cell, and move pixels right of it right one cell.
    - Whenever a moved half or pushed object overlaps another object, push the hit object one cell in that same direction, repeating until overlaps disappear.
    - Paint the entire ray row or column red at the end.

    Key insight:
    - The red edge pixel does not mark a pre-existing crack; it creates a full-line fracture whose separated halves can shove a chain of distant objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert 0 < max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    axis, line = _source_axis(g)
    units: list[dict] = []
    movers: list[tuple[int, tuple[int, int]]] = []
    fractured = 0

    for comp in _components(g):
        color = comp["color"]
        coords = comp["coords"]

        if axis == "row":
            on_ray = {p for p in coords if p[0] == line}
            if on_ray:
                above = frozenset(p for p in coords if p[0] < line)
                below = frozenset(p for p in coords if p[0] > line)
                assert above and below, "ray-struck objects must have pixels on both sides of the row"
                fractured += 1
                for half, vec in ((above, (-1, 0)), (below, (1, 0))):
                    shifted = _shift(half, vec[0], vec[1], g.shape)
                    units.append({"color": color, "coords": shifted, "vec": vec})
                    movers.append((len(units) - 1, vec))
            else:
                units.append({"color": color, "coords": coords, "vec": None})
        else:
            on_ray = {p for p in coords if p[1] == line}
            if on_ray:
                left = frozenset(p for p in coords if p[1] < line)
                right = frozenset(p for p in coords if p[1] > line)
                assert left and right, "ray-struck objects must have pixels on both sides of the column"
                fractured += 1
                for half, vec in ((left, (0, -1)), (right, (0, 1))):
                    shifted = _shift(half, vec[0], vec[1], g.shape)
                    units.append({"color": color, "coords": shifted, "vec": vec})
                    movers.append((len(units) - 1, vec))
            else:
                units.append({"color": color, "coords": coords, "vec": None})

    assert fractured > 0, "expected at least one ray-struck object"
    _resolve_pushes(units, movers, g.shape)

    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            assert not (units[i]["coords"] & units[j]["coords"]), "push result must have no overlaps"

    out = np.zeros_like(g)
    for unit in units:
        for r, c in unit["coords"]:
            if axis == "row":
                assert r != line, "objects must clear the final ray row"
            else:
                assert c != line, "objects must clear the final ray column"
            assert out[r, c] == BACKGROUND, "objects must occupy unique cells"
            out[r, c] = unit["color"]

    if axis == "row":
        out[line, :] = RED
    else:
        out[:, line] = RED
    return out


# --------------- generator ---------------

_GEN_RETRIES = 500
_GEN_CALL_COUNT = 0


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _paint_rect(g: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    assert np.all(g[r0 : r0 + h, c0 : c0 + w] == BACKGROUND), "generated objects cannot overlap"
    g[r0 : r0 + h, c0 : c0 + w] = int(color)


def _segments_clear(segments: list[tuple[int, int]], c0: int, c1: int, gap: int = 2) -> bool:
    return all(c1 + gap < a or b + gap < c0 for a, b in segments)


def _choose_segments(width: int, widths: list[int]) -> list[tuple[int, int]] | None:
    segments: list[tuple[int, int]] = []
    for w in widths:
        options = [
            (c, c + w - 1)
            for c in range(2, width - w - 1)
            if _segments_clear(segments, c, c + w - 1)
        ]
        if not options:
            return None
        segment = random.choice(options)
        segments.append(segment)
    return segments


def _clear_with_margin(g: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> bool:
    hgt, wid = g.shape
    if r0 < 0 or c0 < 0 or r0 + h > hgt or c0 + w > wid:
        return False
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(hgt, r0 + h + margin)
    cc1 = min(wid, c0 + w + margin)
    return bool(np.all(g[rr0:rr1, cc0:cc1] == BACKGROUND))


def _place_decoy(g: np.ndarray, line: int, color: int) -> bool:
    hgt, wid = g.shape
    sides = ["above", "below"]
    random.shuffle(sides)
    for side in sides:
        for _ in range(80):
            h = random.randint(1, 3)
            w = random.randint(2, 4)
            if side == "above":
                if line - h - 3 < 0:
                    continue
                r0 = random.randint(1, line - h - 2)
            else:
                if line + 3 > hgt - h:
                    continue
                r0 = random.randint(line + 2, hgt - h - 1)
            c0 = random.randint(1, wid - w - 2)
            if _clear_with_margin(g, r0, c0, h, w, margin=1):
                _paint_rect(g, r0, c0, h, w, color)
                return True
    return False


def _semantic_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        axis, line = _source_axis(inp)
    except AssertionError:
        return False

    if axis == "row":
        if not np.all(out[line, :] == RED):
            return False
    else:
        if not np.all(out[:, line] == RED):
            return False

    comps = _components(inp)
    fractured = 0
    pushed = 0
    unchanged = 0
    for comp in comps:
        coords = comp["coords"]
        color = comp["color"]
        out_coords = frozenset(map(tuple, np.argwhere(out == color)))
        if axis == "row":
            crosses = any(r == line for r, _c in coords)
            if crosses:
                if any(r < line for r, _c in coords) and any(r > line for r, _c in coords):
                    fractured += 1
                continue
            if all(r < line for r, _c in coords):
                vec = (-1, 0)
            elif all(r > line for r, _c in coords):
                vec = (1, 0)
            else:
                return False
        else:
            crosses = any(c == line for _r, c in coords)
            if crosses:
                if any(c < line for _r, c in coords) and any(c > line for _r, c in coords):
                    fractured += 1
                continue
            if all(c < line for _r, c in coords):
                vec = (0, -1)
            elif all(c > line for _r, c in coords):
                vec = (0, 1)
            else:
                return False

        if out_coords == coords:
            unchanged += 1
        else:
            for k in range(1, 5):
                shifted = frozenset((r + vec[0] * k, c + vec[1] * k) for r, c in coords)
                if out_coords == shifted:
                    pushed += 1
                    break

    return fractured >= 1 and pushed >= 2 and unchanged >= 1 and not np.array_equal(inp, out)


def _build_horizontal(call_index: int) -> np.ndarray | None:
    colors = OBJECT_COLORS[:]
    random.shuffle(colors)

    primary_w = random.randint(2, 5)
    primary_top = random.randint(1, 3)
    primary_bottom = random.randint(1, 3)
    force_up_chain = call_index % 3 != 1
    up_chain = 2 if force_up_chain else random.choice([0, 1])
    down_chain = 2 if not force_up_chain else random.choice([0, 1, 2])
    if up_chain < 2 and down_chain < 2:
        up_chain = 2

    use_second = random.random() < 0.55
    widths = [primary_w]
    second_top = second_bottom = 0
    if use_second:
        widths.append(random.randint(2, 4))
        second_top = random.randint(1, 2)
        second_bottom = random.randint(1, 2)

    top_need = max(primary_top + up_chain + 1, second_top + 1)
    bottom_need = max(primary_bottom + down_chain + 1, second_bottom + 1)
    min_h = max(14, top_need + bottom_need + 3)
    if min_h > 28:
        return None
    hgt = random.randint(min_h, 28)
    line = random.randint(top_need, hgt - bottom_need - 1)

    min_w = max(14, sum(widths) + 2 * len(widths) + 5)
    wid = random.randint(min_w, 28)
    segments = _choose_segments(wid, widths)
    if segments is None:
        return None

    g = np.zeros((hgt, wid), dtype=int)
    g[line, random.choice([0, wid - 1])] = RED

    color_idx = 0
    c0, c1 = segments[0]
    _paint_rect(g, line - primary_top, c0, primary_top + 1 + primary_bottom, primary_w, colors[color_idx])
    color_idx += 1

    if up_chain:
        block_w = random.randint(2, primary_w)
        bc0 = random.randint(c0, c1 - block_w + 1)
        for k in range(1, up_chain + 1):
            _paint_rect(g, line - primary_top - k, bc0, 1, block_w, colors[color_idx])
            color_idx += 1

    if down_chain:
        block_w = random.randint(2, primary_w)
        bc0 = random.randint(c0, c1 - block_w + 1)
        for k in range(1, down_chain + 1):
            _paint_rect(g, line + primary_bottom + k, bc0, 1, block_w, colors[color_idx])
            color_idx += 1

    if use_second:
        c0, c1 = segments[1]
        _paint_rect(
            g,
            line - second_top,
            c0,
            second_top + 1 + second_bottom,
            c1 - c0 + 1,
            colors[color_idx],
        )
        color_idx += 1

    decoy_count = random.randint(1, min(2, len(colors) - color_idx))
    for _ in range(decoy_count):
        if not _place_decoy(g, line, colors[color_idx]):
            return None
        color_idx += 1

    return g


def _gen_attempt(call_index: int, vertical: bool) -> np.ndarray | None:
    grid = _build_horizontal(call_index)
    if grid is None:
        return None
    if vertical:
        grid = grid.T.copy()
    try:
        out = main(grid)
    except AssertionError:
        return None
    if not _semantic_ok(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    preferred_vertical = bool(call_index % 2)

    for attempt in range(_GEN_RETRIES):
        vertical = preferred_vertical if attempt < _GEN_RETRIES // 2 else bool(random.getrandbits(1))
        grid = _gen_attempt(call_index, vertical)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-fracture push-chain input")
