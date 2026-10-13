from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    """Return connected components of a boolean mask as (N,2) coord arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = N4 if connectivity == 4 else N8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _segment_cells(top_r: int, top_c: int, length: int, direction: int) -> list[tuple[int, int]]:
    return [(top_r + i, top_c + direction * i) for i in range(length)]


def _extract_slopes(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    """Parse every grey 8-connected component as one diagonal slope segment."""
    slopes = []
    cell_to_segment: dict[tuple[int, int], int] = {}
    for coords in _components(grid == GREY, connectivity=8):
        assert len(coords) >= 2, "each slope must have length >= 2"
        r0, c0, r1, c1 = _bbox(coords)
        order = coords[np.argsort(coords[:, 0])]
        rows = order[:, 0].tolist()
        cols = order[:, 1].tolist()
        assert rows == list(range(r0, r1 + 1)), "slope rows must be consecutive"
        deltas = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        assert all(d == deltas[0] for d in deltas), "slope must keep one diagonal direction"
        assert deltas[0] in (-1, 1), "slope must be a 45-degree diagonal"
        assert len(set(cols)) == len(cols), "slope columns must be unique"

        direction = int(deltas[0])
        top = (int(rows[0]), int(cols[0]))
        slope = {
            "id": len(slopes),
            "coords": {tuple(map(int, rc)) for rc in order.tolist()},
            "top": top,
            "length": int(len(coords)),
            "dir": direction,
        }
        slopes.append(slope)
        for rc in slope["coords"]:
            assert rc not in cell_to_segment, "slopes must not overlap"
            cell_to_segment[rc] = slope["id"]
    assert slopes, "expected at least one grey slope"
    return slopes, cell_to_segment


def _extract_pixels(grid: np.ndarray) -> list[dict]:
    """Return all non-grey, non-black singleton pixels."""
    mask = (grid != BLACK) & (grid != GREY)
    pixels = []
    for coords in _components(mask, connectivity=4):
        assert len(coords) == 1, "every moving object must be an isolated single pixel"
        r, c = map(int, coords[0])
        pixels.append({"pos": (r, c), "color": int(grid[r, c])})
    assert pixels, "expected at least one colored pixel"
    return pixels


def _simulate(
    grid: np.ndarray,
    *,
    bottom_up: bool = True,
    enable_slopes: bool = True,
    allow_multi_segment: bool = True,
    top_entry_only: bool = False,
    with_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """
    Simulate the falling pixels.

    A pixel occupies air cells while moving. It is deflected exactly when the cell
    directly below it is grey, then slides one row down and one column toward that
    slope's low end. Settled pixels affect only the final vertical landing phase.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    slopes, cell_to_segment = _extract_slopes(g)
    pixels = _extract_pixels(g)

    order = sorted(
        pixels,
        key=lambda item: (-item["pos"][0], item["pos"][1]) if bottom_up else (item["pos"][0], item["pos"][1]),
    )
    out = g.copy()
    out[(out != BLACK) & (out != GREY)] = BLACK

    settled: dict[tuple[int, int], int] = {}
    traces = []
    for item in order:
        r, c = item["pos"]
        touched: list[int] = []

        while True:
            below = (r + 1, c)
            slope_id = cell_to_segment.get(below) if enable_slopes and r + 1 < h else None
            if slope_id is not None:
                slope = slopes[slope_id]
                if top_entry_only and below != slope["top"]:
                    slope_id = None
                elif not allow_multi_segment and touched and slope_id not in touched:
                    slope_id = None

            if slope_id is not None:
                if slope_id not in touched:
                    touched.append(int(slope_id))
                r += 1
                c += slopes[slope_id]["dir"]
                assert _in_bounds(g.shape, r, c), "slope slide must stay in bounds"
                continue

            if r == h - 1 or (r + 1, c) in settled:
                assert g[r, c] != GREY, "settled pixels cannot overwrite slope cells"
                settled[(r, c)] = item["color"]
                traces.append(
                    {
                        "start": item["pos"],
                        "final": (r, c),
                        "color": item["color"],
                        "segments": touched[:],
                    }
                )
                break

            r += 1

    for (r, c), color in settled.items():
        out[r, c] = int(color)
    return (out, traces) if with_trace else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Deflect each falling pixel across every grey ramp it meets, then let the accumulated path settle into a stack.

    Input structure:
    - Black background with several differently colored isolated single pixels in the upper area.
    - The middle area contains one or more non-touching grey 45-degree ramp segments.
    - The bottom edge is the ground; there is no separate baseline row.

    Rule:
    - Process the colored pixels from lower initial row to higher initial row.
    - Each pixel falls vertically until the cell directly below it is grey; then it slides one row down and one column toward that ramp's low end.
    - After it passes the low end, it resumes vertical falling, and the same path can be deflected again by later ramps.
    - The pixel finally stops on the ground or on top of an already settled pixel; colors stay unchanged and the grey ramps remain unchanged.

    Key insight:
    - The final column is determined by the full cascade of encountered ramps, not just the first ramp below the start, and stacking depends on processing lower starting pixels first.
    """
    return _simulate(grid)


def _variant_outputs(grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "correct": _simulate(grid),
        "vertical_only": _simulate(grid, enable_slopes=False),
        "first_segment_only": _simulate(grid, allow_multi_segment=False),
        "top_down": _simulate(grid, bottom_up=False),
        "top_entry_only": _simulate(grid, top_entry_only=True),
    }


def _rows_unique(points: Iterable[tuple[int, int]]) -> bool:
    rows = [r for r, _ in points]
    return len(rows) == len(set(rows))


def _reserve_tops(grid: np.ndarray, starts: list[tuple[int, int]], colors: list[int]) -> None:
    for (r, c), color in zip(starts, colors, strict=True):
        grid[r, c] = color


def _segment_margin_ok(occupied: set[tuple[int, int]], cells: Iterable[tuple[int, int]]) -> bool:
    """Require Chebyshev distance >= 2 between distinct grey segments."""
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _place_chain(
    h: int,
    w: int,
    n_chain: int,
) -> tuple[list[dict], int] | None:
    lengths = [random.randint(2, 4) for _ in range(n_chain)]
    first_dir = random.choice([-1, 1])
    dirs = [first_dir if i % 2 == 0 else -first_dir for i in range(n_chain)]
    gaps = [random.randint(1, 2) for _ in range(n_chain - 1)]

    rel_cells = []
    tops = []
    cur_top_r = 0
    cur_top_c = 0
    for i, (length, direction) in enumerate(zip(lengths, dirs, strict=True)):
        tops.append((cur_top_r, cur_top_c))
        rel_cells.extend(_segment_cells(cur_top_r, cur_top_c, length, direction))
        if i < n_chain - 1:
            cur_top_r += length + gaps[i]
            cur_top_c += direction * length

    min_r = min(r for r, _ in rel_cells)
    max_r = max(r for r, _ in rel_cells)
    min_c = min(c for _, c in rel_cells)
    max_c = max(c for _, c in rel_cells)
    top_margin = random.randint(5, 7)
    landing_margin = random.randint(4, 6)
    if max_r - min_r + 1 + top_margin + landing_margin > h:
        return None

    row_shift = top_margin - min_r
    low_needed = max_c - min_c + 1
    if low_needed > w - 4:
        return None
    col_shift = random.randint(2 - min_c, w - 3 - max_c)

    chain = []
    for i, (length, direction) in enumerate(zip(lengths, dirs, strict=True)):
        top_r, top_c = tops[i]
        chain.append(
            {
                "top_r": top_r + row_shift,
                "top_c": top_c + col_shift,
                "length": length,
                "dir": direction,
            }
        )
    return chain, int(chain[0]["top_c"])


def _try_place_extra(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    forbidden_cols: set[int],
) -> dict | None:
    for _ in range(80):
        length = random.randint(3, 5)
        direction = random.choice([-1, 1])
        top_r = random.randint(7, h - 8)
        if direction == 1:
            top_c = random.randint(1, w - length - 2)
        else:
            top_c = random.randint(length, w - 2)
        cells = _segment_cells(top_r, top_c, length, direction)
        cols = {c for _, c in cells}
        if cols & forbidden_cols:
            continue
        if not _segment_margin_ok(occupied, cells):
            continue
        return {"top_r": top_r, "top_c": top_c, "length": length, "dir": direction}
    return None


def _draw_slopes(grid: np.ndarray, segments: list[dict]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for seg in segments:
        cells = _segment_cells(seg["top_r"], seg["top_c"], seg["length"], seg["dir"])
        assert _segment_margin_ok(occupied, cells), "segments must stay separated"
        for r, c in cells:
            grid[r, c] = GREY
            occupied.add((r, c))
    return occupied


def _pixel_layout(chain: list[dict], extra: dict, h: int) -> list[tuple[int, int]]:
    """Build fixed semantic roles with distinct rows."""
    chain_entry = chain[0]["top_c"]
    second_entry = chain[1]["top_c"]
    same_path_rows = [0, 2]
    third_row = 4
    partial_k = random.randint(1, extra["length"] - 1)
    partial_col = extra["top_c"] + extra["dir"] * partial_k
    partial_row = min(6, extra["top_r"] + partial_k - 2)
    if partial_row in {0, 2, 4}:
        partial_row = 6
    starts = [
        (same_path_rows[0], chain_entry),
        (same_path_rows[1], chain_entry),
        (third_row, second_entry),
        (partial_row, partial_col),
    ]

    if random.random() < 0.5:
        free_cols = [c for c in range(1, chain_entry) if c not in {chain_entry, second_entry, partial_col}]
        if not free_cols:
            free_cols = [c for c in range(second_entry + 1, max(second_entry + 2, partial_col + 2))]
        extra_row = 8
        if extra_row < h - 2:
            starts.append((extra_row, random.choice(free_cols)))
    assert _rows_unique(starts)
    return starts


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        correct, traces = _simulate(grid, with_trace=True)
    except AssertionError:
        return False

    variants = _variant_outputs(grid)
    if not np.array_equal(correct, variants["correct"]):
        return False
    if any(np.array_equal(correct, variants[name]) for name in ("vertical_only", "first_segment_only", "top_down", "top_entry_only")):
        return False

    if max(len(t["segments"]) for t in traces) < 2:
        return False

    final_cols = [t["final"][1] for t in traces]
    if len(final_cols) == len(set(final_cols)):
        return False

    starts = [tuple(t["start"]) for t in traces]
    if not _rows_unique(starts):
        return False

    slopes, _ = _extract_slopes(grid)
    if len({seg["dir"] for seg in slopes}) < 2:
        return False

    out = correct
    if np.any((grid == GREY) != (out == GREY)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    n_chain = random.choice([2, 3])
    placed = _place_chain(h, w, n_chain)
    if placed is None:
        return None
    chain, chain_entry = placed

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = _draw_slopes(grid, chain)
    chain_cols = {c for seg in chain for _, c in _segment_cells(seg["top_r"], seg["top_c"], seg["length"], seg["dir"])}
    extra = _try_place_extra(h, w, occupied, forbidden_cols=chain_cols | {chain_entry})
    if extra is None:
        return None
    occupied = _draw_slopes(grid, [extra])

    starts = _pixel_layout(chain, extra, h)
    if any(not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK for r, c in starts):
        return None
    if any(r >= chain[0]["top_r"] - 1 for r, c in starts[:2]):
        return None
    colors = random.sample(COLOR_POOL, len(starts))
    _reserve_tops(grid, starts, colors)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid slope-cascade sample")
