# Combo variant of 2b83f449 (index 18)
# combo_spec: D5/2+D13/0
# Name: 门数编码令牌数+图例行指定令牌行

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import paste

"""
Design analysis for the fused variant.

1. Concrete behavior and a small trace
   Base task: every non-bottom pipe row has two edge tokens, one on each side.
   D5/2 changes the source multiplicity: a pipe row contributes as many tokens as
   the number of gates in the wall directly below it, and the tokens sit at the
   left edge as a consecutive run.
   D13/0 changes the existence rule: an extra legend row tells which pipe rows
   are token-bearing. In the fused task the legend chooses "which rows", then the
   gate count chooses "how many" inside those chosen rows.

   Example sketch with 4 pipe rows (P0..P3) and 3 wall rows (W0..W2):
   - legend markers: [ON, OFF, ON, OFF]
   - gate counts below rows: W0=2, W1=3, W2=1
   Then the input sources become:
   - P0 gets 2 tokens at cols 0,1
   - P1 gets none, even though W1 has 3 gates
   - P2 gets 1 token at col 0
   - P3 is bottom, so none
   The output still follows the original gate-opening + falling + endpoint-packing
   mechanics, but only for those reconstructed sources.

2. Complexity calibration
   The fusion needs both controls to be visible. If every row with gates were
   always active, the legend would be decorative. If every active row had the
   same gate count, the counting signal would be weak. The generator therefore
   enforces:
   - at least one inactive row whose wall still has >=1 gate
   - at least two active rows
   - at least two distinct positive gate-count values among active rows
   This gives repeated evidence for both halves of the rule without making the
   grid visually noisy.

3. Hidden assumptions broken by the fusion
   The base solver assumed:
   - the rare non-gate color is the token color
   - every token pixel in pipe rows is an actual source
   - every non-bottom pipe row contributes sources
   These fail once a legend row introduces new colors and once only selected rows
   may carry tokens. The new solver therefore uses fixed task colors, strips the
   legend row explicitly, counts gates row-by-row, decodes legend markers, and
   reconstructs the intended sources from metadata instead of trusting arbitrary
   token placement.

4. Placement legality
   Tokens are only generated on rows where the leftmost source cells remain open
   after gate rotation; this avoids placing a source on top of a future vertical
   post. Holes are kept away from the needed source cells by filtering active-row
   choices, and the original single-exit repair remains in place so falling paths
   stay unambiguous.

5. Visual observability and cleanliness
   The legend is a separate top row with one discrete marker per pipe row, laid
   out left-to-right in the same order as pipe rows top-to-bottom. Using sparse
   markers avoids clutter and prevents ambiguity with the pipe network itself.
   Only two legend colors are used: one for active rows and one for inactive rows.

6. New solver assumptions and generator guarantees
   The solver assumes:
   - the top row is the legend row
   - legend markers sit at deterministic slot columns
   - active rows use left-anchored consecutive sources whose count equals the
     number of gates in the wall below
   The generator guarantees those conditions and also guarantees at least one
   active token traverses >=3 wall levels so the task remains non-trivial.
"""

WALL = 0
LEGEND_ON = 1
LEGEND_OFF = 2
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _legend_slots(width: int, n_pipe_rows: int) -> list[int]:
    return [((2 * idx + 1) * width) // (2 * n_pipe_rows) for idx in range(n_pipe_rows)]


def _parse_gate_centers(core: np.ndarray) -> list[list[int]]:
    height, width = core.shape
    n_walls = (height - 1) // 2
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        for c in range(1, width - 1):
            if (
                int(core[wr, c - 1]) == GATE
                and int(core[wr, c]) == GATE
                and int(core[wr, c + 1]) == GATE
            ):
                gates[wall_idx].append(c)
        gates[wall_idx] = sorted(set(gates[wall_idx]))
    return gates


def _decode_active_rows(legend: np.ndarray, width: int, n_pipe_rows: int) -> list[bool]:
    markers = legend[0]
    active = []
    for col in _legend_slots(width, n_pipe_rows):
        active.append(int(markers[col]) == LEGEND_ON)
    return active


def _open_gates(core: np.ndarray, gates: list[list[int]]) -> np.ndarray:
    out = np.array(core, dtype=int, copy=True)
    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center in centers:
            out[wr, center - 1 : center + 2] = BG
            out[wr - 1, center] = OPEN
            out[wr, center] = OPEN
            out[wr + 1, center] = OPEN
    out[out == TOKEN] = BG
    return out


def _segments_in_row(grid: np.ndarray, r: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != BG:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == BG:
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exit_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> int | None:
    if r + 2 >= grid.shape[0]:
        return None
    a, b = seg
    exits = [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]
    if len(exits) != 1:
        return None
    return exits[0]


def _entry_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> int | None:
    if r < 2:
        return None
    a, b = seg
    entries = [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]
    if len(entries) != 1:
        return None
    return entries[0]


def _fallback_token_cols(core: np.ndarray, r: int, count: int) -> list[int]:
    return [int(c) for c in np.flatnonzero(core[r] == TOKEN)[:count]]


def _build_sources(core: np.ndarray, opened: np.ndarray, gates: list[list[int]], active_rows: list[bool]) -> list[dict[str, int | str]]:
    width = core.shape[1]
    n_walls = len(gates)
    sources: list[dict[str, int | str]] = []
    for pipe_idx in range(n_walls):
        if pipe_idx >= len(active_rows) or not active_rows[pipe_idx]:
            continue
        count = len(gates[pipe_idx])
        if count <= 0:
            continue
        r = 2 * pipe_idx
        cols = list(range(min(count, width)))
        if any(int(opened[r, c]) != BG for c in cols):
            cols = _fallback_token_cols(core, r, count)
        for c in cols:
            if 0 <= c < width and int(opened[r, c]) == BG:
                sources.append({"r": r, "c": c, "pipe_idx": pipe_idx, "side": "left"})
    return sources


def _simulate_from_sources(opened: np.ndarray, sources: list[dict[str, int | str]]) -> tuple[np.ndarray, list[int]]:
    out = np.array(opened, dtype=int, copy=True)
    records: list[dict[str, int | tuple[int, int] | str]] = []
    depths: list[int] = []

    for source in sources:
        r = int(source["r"])
        c = int(source["c"])
        depth = 0
        while True:
            seg = _find_segment(opened, r, c)
            if seg is None:
                break
            exit_col = _exit_below(opened, r, seg)
            if exit_col is None:
                break
            r += 2
            c = exit_col
            depth += 1
        final_seg = _find_segment(opened, r, c)
        if final_seg is None:
            final_seg = (c, c)
        records.append(
            {
                "r": r,
                "c": c,
                "seg": final_seg,
                "side": str(source["side"]),
                "pipe_idx": int(source["pipe_idx"]),
            }
        )
        depths.append(depth)

    buckets: dict[tuple[int, tuple[int, int], int], list[dict[str, int | tuple[int, int] | str]]] = defaultdict(list)
    for rec in records:
        r = int(rec["r"])
        seg = rec["seg"]
        assert isinstance(seg, tuple)
        entry = _entry_above(opened, r, seg)
        a, b = seg
        if entry is not None:
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if rec["side"] == "left" else b
        buckets[(r, seg, pack_end)].append(rec)

    for (r, seg, pack_end), recs in buckets.items():
        a, b = seg
        recs.sort(key=lambda rec: (int(rec["pipe_idx"]), int(rec["c"])))
        if pack_end == a:
            for idx, _rec in enumerate(recs):
                if a + idx <= b:
                    out[r, a + idx] = TOKEN
        else:
            for idx, _rec in enumerate(recs):
                if b - idx >= a:
                    out[r, b - idx] = TOKEN

    return out, depths


def _solve_core(grid: np.ndarray) -> np.ndarray:
    full = np.array(grid, dtype=int)
    legend = full[:1]
    core = full[1:]
    n_pipe_rows = (core.shape[0] + 1) // 2
    gates = _parse_gate_centers(core)
    active_rows = _decode_active_rows(legend, core.shape[1], n_pipe_rows)
    opened = _open_gates(core, gates)
    sources = _build_sources(core, opened, gates, active_rows)
    out, _depths = _simulate_from_sources(opened, sources)
    return np.vstack([legend, out])


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(np.array(grid, dtype=int))


def _build_opened_grid(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for c in centers:
            grid[wr, c - 1] = BG
            grid[wr, c] = OPEN
            grid[wr, c + 1] = BG
            grid[wr - 1, c] = OPEN
            grid[wr + 1, c] = OPEN

    return grid


def _build_legend(width: int, active_rows: list[bool]) -> np.ndarray:
    legend = np.full((1, width), WALL, dtype=int)
    for pipe_idx, col in enumerate(_legend_slots(width, len(active_rows))):
        legend[0, col] = LEGEND_ON if active_rows[pipe_idx] else LEGEND_OFF
    return legend


def _build_input(gates: list[list[int]], holes: list[list[int]], width: int, active_rows: list[bool]) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    gate_patch = np.full((1, 3), GATE, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for c in centers:
            paste(grid, gate_patch, wr, c - 1)

    for pipe_idx in range(n_walls):
        if not active_rows[pipe_idx]:
            continue
        count = len(gates[pipe_idx])
        for c in range(count):
            grid[2 * pipe_idx, c] = TOKEN

    legend = _build_legend(width, active_rows)
    return np.vstack([legend, grid])


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r):
            above = _entries_above(grid, r, seg)
            below = _exits_below(grid, r, seg)
            if len(above) > 1 or len(below) > 1:
                issues.append((r, seg, above, below))
    return issues


def _separator_column(grid: np.ndarray, r: int, left: int, right: int) -> int | None:
    choices = [c for c in range(left + 1, right) if int(grid[r, c]) == BG]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


def _repair_holes(gates: list[list[int]], holes: list[list[int]], width: int) -> list[list[int]] | None:
    hole_sets = [set(row) for row in holes]

    while True:
        grid = _build_opened_grid(gates, [sorted(row) for row in hole_sets], width)
        issues = _find_issues(grid)
        if not issues:
            return [sorted(row) for row in hole_sets]

        r, _seg, above, below = issues[0]
        pipe_idx = r // 2
        if len(hole_sets[pipe_idx]) >= 2:
            return None

        events = sorted([(c, "up") for c in above] + [(c, "down") for c in below])
        seen_up = False
        seen_down = False
        prev_col: int | None = None
        new_cols: list[int] = []

        for col, kind in events:
            duplicate = (kind == "up" and seen_up) or (kind == "down" and seen_down)
            if duplicate:
                if prev_col is None:
                    return None
                sep = _separator_column(grid, r, prev_col, col)
                if sep is None:
                    return None
                new_cols.append(sep)
                seen_up = False
                seen_down = False

            if kind == "up":
                seen_up = True
            else:
                seen_down = True
            prev_col = col

        if not new_cols:
            return None

        for c in new_cols:
            hole_sets[pipe_idx].add(c)
            if len(hole_sets[pipe_idx]) > 2:
                return None

        if sum(len(row) for row in hole_sets) > 4:
            return None


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    candidates: list[int] = []
    for c in range(2, width - 2):
        if c in gates[row]:
            continue
        if any(abs(c - other) < 5 for other in gates[row]):
            continue

        ok = True
        for adj in (row - 1, row + 1):
            if 0 <= adj < len(gates) and any(abs(c - other) < 2 for other in gates[adj]):
                ok = False
                break
        if ok:
            candidates.append(c)
    return candidates


def _drop_depth(opened: np.ndarray, start_r: int, start_c: int) -> int:
    depth = 0
    r = start_r
    c = start_c
    while True:
        seg = _find_segment(opened, r, c)
        if seg is None:
            return depth
        exit_col = _exit_below(opened, r, seg)
        if exit_col is None:
            return depth
        r += 2
        c = exit_col
        depth += 1


def _choose_active_rows(gates: list[list[int]], opened: np.ndarray, rng: np.random.Generator) -> list[bool] | None:
    n_walls = len(gates)
    positive_rows = []
    row_depths: dict[int, int] = {}

    for pipe_idx in range(n_walls):
        count = len(gates[pipe_idx])
        if count <= 0:
            continue
        r = 2 * pipe_idx
        if any(int(opened[r, c]) != BG for c in range(count)):
            continue
        positive_rows.append(pipe_idx)
        row_depths[pipe_idx] = _drop_depth(opened, r, 0)

    if len(positive_rows) < 3:
        return None

    for _ in range(80):
        order = list(rng.permutation(positive_rows))
        size = int(rng.integers(2, len(positive_rows)))
        chosen = set(order[:size])
        if len(chosen) < 2:
            continue
        if not any(row_depths[row] >= 3 for row in chosen):
            continue
        if not any(row not in chosen for row in positive_rows):
            continue
        active_counts = {len(gates[row]) for row in chosen}
        if len(active_counts) < 2:
            continue
        flags = [False] * (n_walls + 1)
        for row in chosen:
            flags[row] = True
        return flags

    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(5, 10))
    width = int(rng.integers(16, 20))

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = int(rng.integers(3, min(6, n_walls) + 1))
    spine_start = int(rng.integers(0, n_walls - spine_len + 1))

    prev_center: int | None = None
    go_left = bool(rng.integers(2))
    for row in range(spine_start, spine_start + spine_len):
        if prev_center is None:
            options = list(range(3, width - 3))
        elif go_left:
            options = [c for c in range(3, prev_center - 1)]
        else:
            options = [c for c in range(prev_center + 2, width - 2)]
        if not options:
            return None
        center = int(rng.choice(options))
        gates[row] = [center]
        prev_center = center
        go_left = not go_left

    holes = _repair_holes(gates, holes, width)
    if holes is None:
        return None

    total_gates = sum(len(row) for row in gates)
    target_total = int(rng.integers(max(6, total_gates), 13))

    while total_gates < target_total:
        candidates: list[tuple[int, int]] = []
        for row in range(n_walls):
            if len(gates[row]) >= 3:
                continue
            for center in _candidate_centers(gates, row, width):
                candidates.append((row, center))

        if not candidates:
            break

        order = rng.permutation(len(candidates))
        added = False
        for idx in order:
            row, center = candidates[int(idx)]
            new_gates = [list(cols) for cols in gates]
            new_gates[row] = sorted(new_gates[row] + [center])
            new_holes = _repair_holes(new_gates, holes, width)
            if new_holes is None:
                continue
            gates = new_gates
            holes = new_holes
            total_gates += 1
            added = True
            break

        if not added:
            break

    if not (6 <= total_gates <= 12):
        return None

    opened = _build_opened_grid(gates, holes, width)
    if _find_issues(opened):
        return None

    active_rows = _choose_active_rows(gates, opened, rng)
    if active_rows is None:
        return None
    active_rows[-1] = False

    inp = _build_input(gates, holes, width, active_rows)
    expected = _solve_core(inp)
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo 2b83f449/18")
