from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WATER = 1
MARKER = 3
SHELF = 5


def _parse_model(grid: np.ndarray) -> tuple[list[int], list[dict[str, object]]]:
    """Extract top-row source columns and grey shelf segments with marker counts."""
    g = np.array(grid, dtype=int)
    H, W = g.shape

    source_cols = [int(c) for c in range(W) if int(g[0, c]) == WATER]

    shelves: list[dict[str, object]] = []
    sid = 0
    for r in range(H):
        c = 0
        while c < W:
            if int(g[r, c]) != SHELF:
                c += 1
                continue
            c0 = int(c)
            while c + 1 < W and int(g[r, c + 1]) == SHELF:
                c += 1
            c1 = int(c)
            marker_count = 0
            marker_cols: list[int] = []
            if r > 0:
                for mc in range(c0, c1 + 1):
                    if int(g[r - 1, mc]) == MARKER:
                        marker_count += 1
                        marker_cols.append(int(mc))
            shelves.append(
                {
                    "id": sid,
                    "row": int(r),
                    "c0": c0,
                    "c1": c1,
                    "marker_count": int(marker_count),
                    "marker_cols": marker_cols,
                }
            )
            sid += 1
            c += 1
    shelves.sort(key=lambda sh: (int(sh["row"]), int(sh["c0"])))
    return source_cols, shelves


def _simulate_model(
    shape: tuple[int, int],
    source_cols: list[int],
    shelves: list[dict[str, object]],
    parity_override: dict[int, int] | None = None,
) -> tuple[np.ndarray, set[int], set[int]]:
    """Simulate the waterfall. Odd parity shelves are permeable; even parity shelves overflow."""
    H, W = map(int, shape)
    shelf_by_row = {int(sh["row"]): sh for sh in shelves}

    water_mask = np.zeros((H, W), dtype=bool)
    hit_ids: set[int] = set()
    bottom_cols: set[int] = set()
    seen_starts: set[tuple[int, int]] = set()
    q = deque((0, int(c)) for c in source_cols if 0 <= int(c) < W)

    while q:
        r, c = q.popleft()
        state = (int(r), int(c))
        if state in seen_starts:
            continue
        seen_starts.add(state)

        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            water_mask[rr, cc] = True

        while True:
            nr = rr + 1
            if nr >= H:
                bottom_cols.add(cc)
                break

            shelf = shelf_by_row.get(nr)
            if shelf is not None and int(shelf["c0"]) <= cc <= int(shelf["c1"]):
                sid = int(shelf["id"])
                hit_ids.add(sid)
                marker_count = int(shelf["marker_count"])
                parity = marker_count % 2
                if parity_override is not None and sid in parity_override:
                    parity = int(parity_override[sid]) % 2

                if parity == 1:
                    rr = nr
                    continue

                spread_row = nr - 1
                left_over = int(shelf["c0"]) - 1
                right_over = int(shelf["c1"]) + 1
                lo = max(0, left_over)
                hi = min(W - 1, right_over)
                water_mask[spread_row, lo : hi + 1] = True
                if 0 <= left_over < W:
                    q.append((spread_row, left_over))
                if 0 <= right_over < W and right_over != left_over:
                    q.append((spread_row, right_over))
                break

            rr = nr
            water_mask[rr, cc] = True

    return water_mask, hit_ids, bottom_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate parity-gated waterfall cascades through and around grey shelves.

    Input structure:
    - Black background contains 1-2 blue source clusters on the top row.
    - Grey foreground objects are one-cell-high horizontal shelf segments on distinct rows.
    - Green marker pixels sit exactly one row above a shelf, within that shelf's span.

    Rule:
    - For each shelf, count the green markers directly above it across the shelf span.
    - If that count is odd, the shelf is permeable: falling water continues straight through it.
    - If that count is even, the shelf blocks: water spreads across the shelf top and then resumes falling from one cell outside each end.
    - Water starts from every blue top-row source cell, flows downward, merges naturally when paths coincide, paints only background cells blue, and stops at the bottom edge.

    Key insight:
    - The green markers are not decoration: their odd/even count toggles each shelf between pass-through and splitter, so counting changes the entire cascade topology.
    """
    g = np.array(grid, dtype=int)
    source_cols, shelves = _parse_model(g)
    water_mask, _hit_ids, _bottom_cols = _simulate_model(g.shape, source_cols, shelves)
    out = g.copy()
    out[(g == BG) & water_mask] = WATER
    return out


def _choose_rows(H: int, n_shelves: int) -> list[int] | None:
    """Choose distinct shelf rows with enough vertical spacing for marker rows."""
    for _ in range(200):
        rows: list[int] = []
        for i in range(n_shelves):
            min_row = 3 if i == 0 else rows[-1] + 3
            max_row = (H - 2) - 3 * (n_shelves - i - 1)
            if min_row > max_row:
                break
            rows.append(random.randint(min_row, max_row))
        if len(rows) == n_shelves:
            return rows
    return None


def _choose_source_clusters(W: int, n_sources: int) -> list[tuple[int, int]] | None:
    """Choose 1-2 top-row source spans of width 1-3 with visible separation."""
    widths = [random.randint(1, 3) for _ in range(n_sources)]
    if n_sources == 1:
        w = widths[0]
        s = random.randint(0, W - w)
        return [(s, s + w - 1)]

    for _ in range(200):
        spans: list[tuple[int, int]] = []
        occupied = np.zeros(W, dtype=bool)
        ok = True
        for w in widths:
            starts = []
            for s in range(0, W - w + 1):
                e = s + w - 1
                left = max(0, s - 3)
                right = min(W, e + 4)
                if occupied[left:right].any():
                    continue
                starts.append(s)
            if not starts:
                ok = False
                break
            s = random.choice(starts)
            e = s + w - 1
            spans.append((s, e))
            occupied[s : e + 1] = True
        if ok:
            spans.sort()
            return spans
    return None


def _span_candidates(
    W: int,
    required_cols: set[int],
    forbidden_cols: set[int],
    *,
    require_internal_overflow: bool,
) -> list[tuple[int, int]]:
    """Enumerate valid shelf spans of length 4..10."""
    candidates: list[tuple[int, int]] = []
    max_len = min(10, W)
    for length in range(4, max_len + 1):
        for c0 in range(0, W - length + 1):
            c1 = c0 + length - 1
            if require_internal_overflow and (c0 == 0 or c1 == W - 1):
                continue
            if not required_cols or not any(c0 <= c <= c1 for c in required_cols):
                continue
            if any(c0 <= c <= c1 for c in forbidden_cols):
                continue
            candidates.append((c0, c1))
    return candidates


def _marker_count_for_parity(parity: int) -> int:
    choices = [1, 3] if parity % 2 == 1 else [2, 4]
    return int(random.choice(choices))


def _build_grid(
    H: int,
    W: int,
    source_spans: list[tuple[int, int]],
    shelves: list[dict[str, object]],
) -> np.ndarray:
    g = np.zeros((H, W), dtype=int)
    for c0, c1 in source_spans:
        g[0, c0 : c1 + 1] = WATER
    for sh in shelves:
        r = int(sh["row"])
        c0 = int(sh["c0"])
        c1 = int(sh["c1"])
        g[r, c0 : c1 + 1] = SHELF
        for mc in sh["marker_cols"]:
            g[r - 1, int(mc)] = MARKER
    return g


def _cluster_widths_from_grid(grid: np.ndarray) -> list[int]:
    widths: list[int] = []
    W = grid.shape[1]
    c = 0
    while c < W:
        if int(grid[0, c]) != WATER:
            c += 1
            continue
        c0 = c
        while c + 1 < W and int(grid[0, c + 1]) == WATER:
            c += 1
        widths.append(c - c0 + 1)
        c += 1
    return widths


def _override_all(shelves: list[dict[str, object]], parity: int) -> dict[int, int]:
    return {int(sh["id"]): int(parity) for sh in shelves}


def _validate_grid(grid: np.ndarray) -> bool:
    """Check explicit puzzle constraints plus non-degenerate parity semantics."""
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    H, W = grid.shape
    if max(H, W) > 30:
        return False
    if np.any(~np.isin(grid, [BG, WATER, MARKER, SHELF])):
        return False

    widths = _cluster_widths_from_grid(grid)
    if len(widths) not in {1, 2} or any(w < 1 or w > 3 for w in widths):
        return False
    if np.any(grid[1:, :] == WATER):
        return False

    source_cols, shelves = _parse_model(grid)
    if len(shelves) < 4 or len(shelves) > 6:
        return False

    shelf_rows = [int(sh["row"]) for sh in shelves]
    if len(set(shelf_rows)) != len(shelf_rows):
        return False
    if any(r <= 1 for r in shelf_rows):
        return False
    if any(b - a + 1 < 4 or b - a + 1 > 10 for a, b in [(int(sh["c0"]), int(sh["c1"])) for sh in shelves]):
        return False
    if any(b - a < 3 for a, b in zip(shelf_rows, shelf_rows[1:])):
        return False

    parities = []
    for sh in shelves:
        marker_cols = list(sh["marker_cols"])
        if len(marker_cols) < 1 or len(marker_cols) > 4:
            return False
        if any(not (int(sh["c0"]) <= int(c) <= int(sh["c1"])) for c in marker_cols):
            return False
        parities.append(len(marker_cols) % 2)
    if 0 not in parities or 1 not in parities:
        return False

    actual_mask, hit_ids, actual_bottom = _simulate_model(grid.shape, source_cols, shelves)
    if not actual_bottom:
        return False
    hit_parities = {int(shelves[sid]["marker_count"]) % 2 for sid in hit_ids}
    if hit_parities != {0, 1}:
        return False

    all_block_mask, _hb, block_bottom = _simulate_model(
        grid.shape, source_cols, shelves, parity_override=_override_all(shelves, 0)
    )
    all_pass_mask, _hp, pass_bottom = _simulate_model(
        grid.shape, source_cols, shelves, parity_override=_override_all(shelves, 1)
    )
    if np.array_equal(actual_mask, all_block_mask) or np.array_equal(actual_mask, all_pass_mask):
        return False
    if actual_bottom == block_bottom or actual_bottom == pass_bottom:
        return False

    actual_out = main(grid)
    if np.array_equal(actual_out, grid):
        return False
    if not np.array_equal(actual_out[grid == SHELF], grid[grid == SHELF]):
        return False
    if not np.array_equal(actual_out[grid == MARKER], grid[grid == MARKER]):
        return False
    if not np.all((actual_out == grid) | ((grid == BG) & (actual_out == WATER))):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(17, 26)
    W = random.randint(18, 26)
    n_shelves = random.randint(4, 6)
    n_sources = random.choice([1, 2]) if W >= 12 else 1

    rows = _choose_rows(H, n_shelves)
    if rows is None:
        return None
    source_spans = _choose_source_clusters(W, n_sources)
    if source_spans is None:
        return None

    source_cols = sorted({c for c0, c1 in source_spans for c in range(c0, c1 + 1)})
    active_cols = set(source_cols)
    active_shelves = random.randint(3, min(4, n_shelves))

    shelves: list[dict[str, object]] = []
    for i in range(active_shelves):
        row = rows[i]
        parity = 0 if i == 0 else 1 if i == 1 else random.randint(0, 1)
        required_cols = set(active_cols)
        if not required_cols:
            return None

        # Keep the first blocking shelf internal so it visibly splits into two branches.
        candidates = _span_candidates(
            W,
            required_cols=required_cols,
            forbidden_cols=set(),
            require_internal_overflow=(parity == 0 and i == 0),
        )
        if not candidates:
            return None
        c0, c1 = random.choice(candidates)

        hit_cols = {c for c in active_cols if c0 <= c <= c1}
        if not hit_cols:
            return None

        shelf = {"id": len(shelves), "row": row, "c0": c0, "c1": c1, "parity": parity}
        shelves.append(shelf)

        if parity == 0:
            active_cols -= hit_cols
            if c0 - 1 >= 0:
                active_cols.add(c0 - 1)
            if c1 + 1 < W:
                active_cols.add(c1 + 1)

    final_active_cols = set(active_cols)
    for i in range(active_shelves, n_shelves):
        row = rows[i]
        parity = random.randint(0, 1)
        prefer_edge = random.random() < 0.5
        candidates = _span_candidates(
            W,
            required_cols=set(range(W)) if not prefer_edge else ({0} | {W - 1}),
            forbidden_cols=final_active_cols,
            require_internal_overflow=False,
        )
        if prefer_edge:
            candidates = [span for span in candidates if span[0] == 0 or span[1] == W - 1]
        if not candidates:
            candidates = _span_candidates(
                W,
                required_cols=set(range(W)),
                forbidden_cols=final_active_cols,
                require_internal_overflow=False,
            )
        if not candidates:
            return None
        c0, c1 = random.choice(candidates)
        shelves.append({"id": len(shelves), "row": row, "c0": c0, "c1": c1, "parity": parity})

    for sh in shelves:
        count = _marker_count_for_parity(int(sh["parity"]))
        span_cols = list(range(int(sh["c0"]), int(sh["c1"]) + 1))
        sh["marker_cols"] = sorted(random.sample(span_cols, count))
        sh["marker_count"] = count

    grid = _build_grid(H, W, source_spans, shelves)
    if _validate_grid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-gated waterfall task")
