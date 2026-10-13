import numpy as np


BG = 0
LINE_COLORS = (1, 2, 3, 4, 6, 8)
GRID_MIN = 13
GRID_MAX = 22
AXIS_GAP = 3
COUNT_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
MAX_GENERATION_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idxs[0])
    for raw in idxs[1:]:
        idx = int(raw)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _containing_run(runs: list[tuple[int, int]], lo: int, hi: int) -> tuple[int, int]:
    matches = [(start, end) for start, end in runs if start <= lo and hi <= end]
    assert len(matches) == 1, "expected exactly one containing run"
    return matches[0]


def _extract_line(grid: np.ndarray, color: int) -> dict[str, int | str]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "every line color must remain visible"

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if len(rows) == 1 and len(cols) == 1:
        r = int(rows[0])
        c = int(cols[0])
        row_run = _containing_run(_runs_1d(grid[r, :] != BG), c, c)
        col_run = _containing_run(_runs_1d(grid[:, c] != BG), r, r)
        horizontal_len = row_run[1] - row_run[0] + 1
        vertical_len = col_run[1] - col_run[0] + 1
        orient = "h" if horizontal_len >= vertical_len else "v"
    elif len(rows) == 1:
        orient = "h"
    else:
        assert len(cols) == 1, "line color must stay on one row or one column"
        orient = "v"

    if orient == "h":
        r = int(rows[0])
        lo = int(cols.min())
        hi = int(cols.max())
        c0, c1 = _containing_run(_runs_1d(grid[r, :] != BG), lo, hi)
        return {
            "color": int(color),
            "orient": "h",
            "row": r,
            "c0": c0,
            "c1": c1,
            "length": c1 - c0 + 1,
        }

    c = int(cols[0])
    lo = int(rows.min())
    hi = int(rows.max())
    r0, r1 = _containing_run(_runs_1d(grid[:, c] != BG), lo, hi)
    return {
        "color": int(color),
        "orient": "v",
        "col": c,
        "r0": r0,
        "r1": r1,
        "length": r1 - r0 + 1,
    }


def _sort_key(line: dict[str, int | str]) -> tuple[int, int, int, int, int]:
    axis = int(line["row"]) if line["orient"] == "h" else int(line["col"])
    orient_rank = 0 if line["orient"] == "h" else 1
    # Valid puzzle instances would already be separated by wins. Degree/length/axis
    # only stabilize ranks when equal wins are unavoidable.
    return (
        -int(line["wins"]),
        -int(line["degree"]),
        -int(line["length"]),
        orient_rank,
        axis,
    )


def _analyze(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != BG)
    assert 4 <= len(colors) <= 6, "expected 4 to 6 line colors"

    lines = [_extract_line(g, color) for color in colors]
    for line in lines:
        line["wins"] = 0
        line["degree"] = 0

    horizontals = [line for line in lines if line["orient"] == "h"]
    verticals = [line for line in lines if line["orient"] == "v"]
    assert horizontals and verticals, "expected both horizontal and vertical lines"

    for hline in horizontals:
        hr = int(hline["row"])
        hc0 = int(hline["c0"])
        hc1 = int(hline["c1"])
        for vline in verticals:
            vc = int(vline["col"])
            vr0 = int(vline["r0"])
            vr1 = int(vline["r1"])
            if not (hc0 <= vc <= hc1 and vr0 <= hr <= vr1):
                continue
            hline["degree"] = int(hline["degree"]) + 1
            vline["degree"] = int(vline["degree"]) + 1
            visible = int(g[hr, vc])
            if visible == int(hline["color"]):
                hline["wins"] = int(hline["wins"]) + 1
            else:
                assert visible == int(vline["color"]), "intersection must show one line color"
                vline["wins"] = int(vline["wins"]) + 1

    return lines


def _render_output(lines: list[dict[str, int | str]]) -> np.ndarray:
    ranked = sorted(lines, key=_sort_key)
    width = max(int(line["length"]) for line in ranked)
    out = np.full((len(ranked), width), BG, dtype=int)
    for r, line in enumerate(ranked):
        out[r, : int(line["length"])] = int(line["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort visible lines by crossing wins and redraw them as left-aligned bars."""
    return _render_output(_analyze(np.asarray(grid, dtype=int)))


def _sample_spaced_positions(limit: int, count: int) -> list[int] | None:
    candidates = np.arange(2, limit - 2, dtype=int)
    if candidates.size < count:
        return None
    for _ in range(100):
        picks = np.sort(_RNG.choice(candidates, size=count, replace=False))
        if np.all(np.diff(picks) >= AXIS_GAP):
            return [int(v) for v in picks]
    return None


def _sample_intervals(line_count: int, other_count: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for _ in range(line_count):
        start = int(_RNG.integers(0, other_count))
        end = int(_RNG.integers(start, other_count))
        intervals.append((start, end))
    return intervals


def _edge_list(
    h_intervals: list[tuple[int, int]],
    v_intervals: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    h_count = len(h_intervals)
    v_count = len(v_intervals)
    edges: list[tuple[int, int]] = []
    degree_h = [0] * h_count
    degree_v = [0] * v_count
    for hi, (hj0, hj1) in enumerate(h_intervals):
        for vi, (vi0, vi1) in enumerate(v_intervals):
            if hj0 <= vi <= hj1 and vi0 <= hi <= vi1:
                edges.append((hi, vi))
                degree_h[hi] += 1
                degree_v[vi] += 1
    return edges, degree_h, degree_v


def _find_assignment(
    h_count: int,
    v_count: int,
    edges: list[tuple[int, int]],
    degree_h: list[int],
    degree_v: list[int],
) -> tuple[dict[tuple[int, int], str], list[int], list[int]] | None:
    total_masks = 1 << len(edges)
    order = _RNG.permutation(total_masks)
    for raw_mask in order:
        mask = int(raw_mask)
        wins_h = [0] * h_count
        wins_v = [0] * v_count
        for bit, (hi, vi) in enumerate(edges):
            if (mask >> bit) & 1:
                wins_h[hi] += 1
            else:
                wins_v[vi] += 1

        score_pairs = [(wins_h[i], degree_h[i]) for i in range(h_count)]
        score_pairs.extend((wins_v[i], degree_v[i]) for i in range(v_count))
        if len(set(score_pairs)) != h_count + v_count:
            continue

        winners: dict[tuple[int, int], str] = {}
        for bit, edge in enumerate(edges):
            winners[edge] = "h" if (mask >> bit) & 1 else "v"
        return winners, wins_h, wins_v
    return None


def _build_spec() -> tuple[np.ndarray, np.ndarray] | None:
    height = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    width = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    h_count, v_count = COUNT_CHOICES[int(_RNG.integers(0, len(COUNT_CHOICES)))]

    rows = _sample_spaced_positions(height, h_count)
    cols = _sample_spaced_positions(width, v_count)
    if rows is None or cols is None:
        return None

    for _ in range(200):
        h_intervals = _sample_intervals(h_count, v_count)
        v_intervals = _sample_intervals(v_count, h_count)
        edges, degree_h, degree_v = _edge_list(h_intervals, v_intervals)
        if not edges or min(degree_h) == 0 or min(degree_v) == 0:
            continue

        solved = _find_assignment(h_count, v_count, edges, degree_h, degree_v)
        if solved is None:
            continue
        winners, wins_h, wins_v = solved

        colors = [int(c) for c in _RNG.choice(LINE_COLORS, size=h_count + v_count, replace=False)]
        lines: list[dict[str, int | str]] = []
        horizontals: list[dict[str, int | str]] = []
        verticals: list[dict[str, int | str]] = []

        for hi, (j0, j1) in enumerate(h_intervals):
            line = {
                "color": colors[hi],
                "orient": "h",
                "row": rows[hi],
                "c0": cols[j0] - 1,
                "c1": cols[j1] + 1,
                "length": cols[j1] - cols[j0] + 3,
                "wins": wins_h[hi],
                "degree": degree_h[hi],
            }
            horizontals.append(line)
            lines.append(line)

        for vi, (i0, i1) in enumerate(v_intervals):
            line = {
                "color": colors[h_count + vi],
                "orient": "v",
                "col": cols[vi],
                "r0": rows[i0] - 1,
                "r1": rows[i1] + 1,
                "length": rows[i1] - rows[i0] + 3,
                "wins": wins_v[vi],
                "degree": degree_v[vi],
            }
            verticals.append(line)
            lines.append(line)

        if any(int(line["length"]) - int(line["degree"]) < 2 for line in lines):
            continue

        score_pairs = [(int(line["wins"]), int(line["degree"])) for line in lines]
        if len(set(score_pairs)) != len(lines):
            continue

        grid = np.full((height, width), BG, dtype=int)
        for line in horizontals:
            r = int(line["row"])
            grid[r, int(line["c0"]) : int(line["c1"]) + 1] = int(line["color"])
        for line in verticals:
            c = int(line["col"])
            grid[int(line["r0"]) : int(line["r1"]) + 1, c] = int(line["color"])

        for hi, vi in edges:
            hline = horizontals[hi]
            vline = verticals[vi]
            winner = hline if winners[(hi, vi)] == "h" else vline
            grid[int(hline["row"]), int(vline["col"])] = int(winner["color"])

        expected = _render_output(lines)
        if not np.array_equal(main(grid), expected):
            continue
        return grid, expected

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_spec()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid input")
