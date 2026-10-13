import numpy as np
from collections import Counter


BG = 0
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GRID_HEIGHT_RANGE = (11, 18)
GRID_WIDTH_RANGE = (16, 25)
COLOR_COUNT_RANGE = (3, 5)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    return sorted(set(map(int, grid.ravel())) - set(map(int, exclude)))


def _ring_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _color_stats(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    stats = []
    for color in _colors_present(g, exclude={bg}):
        mask = g == color
        rows = np.where(mask.any(axis=1))[0]
        assert 1 <= len(rows) <= 2
        raw = int(mask.sum())
        gaps = 0
        for r in rows:
            row = g[int(r)]
            for c in range(1, g.shape[1] - 1):
                if row[c] == bg and row[c - 1] == color and row[c + 1] == color:
                    gaps += 1
        stats.append(
            {
                "color": int(color),
                "rows": tuple(map(int, rows)),
                "top": int(rows.min()),
                "row_count": int(len(rows)),
                "raw": raw,
                "gaps": gaps,
                "filled": raw + gaps,
                "max_row_span": max(int(np.where(g[r] == color)[0][-1] - np.where(g[r] == color)[0][0] + 1) for r in rows),
            }
        )
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """补齐每种颜色虚线中的单像素间隙后，按补填覆盖面积重排为同心方环。

    Input structure:
    - A black background contains 3-5 non-background colors.
    - Each color occupies one or two dedicated horizontal rows as dashed segments.
    - Dashed rows consist of colored runs separated by one-cell black gaps; colors do not share rows.

    Rule:
    - For each color, inspect every black cell on its occupied rows and count it as a gap when its left and right neighbors are that color.
    - Define the filled coverage area as original pixel count plus the number of such gap cells.
    - Sort colors by decreasing filled coverage area, breaking ties by the smallest occupied row index.
    - Output a square of side 2n-1 whose concentric square rings use the sorted colors from outside to inside.

    Key insight:
    - The ranking is determined by the virtually gap-filled horizontal coverage, not by the visible colored pixel count, so denser gaps can reverse the nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    stats = _color_stats(g)
    assert 3 <= len(stats) <= 5
    filled_areas = [item["filled"] for item in stats]
    assert len(filled_areas) == len(set(filled_areas))

    ordered = sorted(stats, key=lambda item: (-item["filled"], item["top"]))
    return _ring_output([item["color"] for item in ordered])


def _build_row_pattern(span: int, run_count: int, rng: np.random.Generator) -> np.ndarray:
    min_span = 2 * run_count - 1
    assert span >= min_span
    runs = np.ones(run_count, dtype=int)
    for _ in range(span - min_span):
        runs[int(rng.integers(0, run_count))] += 1

    row = np.zeros(span, dtype=int)
    c = 0
    for i, run_len in enumerate(runs):
        row[c : c + run_len] = 1
        c += run_len
        if i + 1 < run_count:
            c += 1
    return row


def _gap_bounds(row_count: int, filled_total: int) -> tuple[int, int]:
    return int(row_count), int((filled_total - row_count) // 2)


def _choose_row_counts(color_count: int, rng: np.random.Generator) -> list[int]:
    row_counts = [1 + int(rng.random() < 0.45) for _ in range(color_count)]
    if color_count >= 2 and 1 not in row_counts:
        row_counts[int(rng.integers(0, color_count))] = 1
    if color_count >= 2 and 2 not in row_counts:
        row_counts[int(rng.integers(0, color_count))] = 2
    return row_counts


def _sample_filled_totals(row_counts: list[int], rng: np.random.Generator) -> list[int] | None:
    totals = []
    for row_count in row_counts:
        lo, hi = (5, 12) if row_count == 1 else (10, 20)
        totals.append(int(rng.integers(lo, hi + 1)))
    if len(set(totals)) != len(totals):
        return None
    if 1 in row_counts and 2 in row_counts:
        if not any(
            row_counts[i] < row_counts[j] and totals[i] > totals[j]
            for i in range(len(row_counts))
            for j in range(len(row_counts))
        ):
            return None
    return totals


def _choose_gap_totals(row_counts: list[int], filled_totals: list[int], rng: np.random.Generator) -> list[int] | None:
    bounds = [_gap_bounds(row_count, filled) for row_count, filled in zip(row_counts, filled_totals)]
    candidate_pairs = [
        (i, j)
        for i in range(len(filled_totals))
        for j in range(len(filled_totals))
        if filled_totals[i] > filled_totals[j]
        and bounds[i][1] >= bounds[j][0] + (filled_totals[i] - filled_totals[j]) + 1
    ]
    if not candidate_pairs:
        return None

    for _ in range(80):
        i, j = candidate_pairs[int(rng.integers(0, len(candidate_pairs)))]
        gaps = [None] * len(filled_totals)
        low_j, high_j = bounds[j]
        gaps[j] = int(rng.integers(low_j, high_j + 1))
        low_i, high_i = bounds[i]
        need_i = max(low_i, gaps[j] + (filled_totals[i] - filled_totals[j]) + 1)
        if need_i > high_i:
            continue
        gaps[i] = int(rng.integers(need_i, high_i + 1))

        for k, (low_k, high_k) in enumerate(bounds):
            if gaps[k] is None:
                gaps[k] = int(rng.integers(low_k, high_k + 1))

        raws = [filled - gap for filled, gap in zip(filled_totals, gaps)]
        fill_order = sorted(range(len(filled_totals)), key=lambda idx: -filled_totals[idx])
        raw_order = sorted(range(len(raws)), key=lambda idx: -raws[idx])
        if len(set(raws)) != len(raws):
            continue
        if fill_order == raw_order:
            continue
        if 1 in row_counts and 2 in row_counts:
            if not any(
                row_counts[a] > row_counts[b] and filled_totals[a] < filled_totals[b]
                for a in range(len(row_counts))
                for b in range(len(row_counts))
            ):
                continue
        return [int(x) for x in gaps]
    return None


def _split_two_row_totals(
    filled_total: int, gap_total: int, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(50):
        gap1 = int(rng.integers(1, gap_total))
        gap2 = gap_total - gap1
        min_span1 = 2 * gap1 + 1
        min_span2 = 2 * gap2 + 1
        low1 = max(min_span1, filled_total - 12)
        high1 = min(12, filled_total - min_span2)
        if low1 > high1:
            continue
        span1 = int(rng.integers(low1, high1 + 1))
        span2 = filled_total - span1
        if min_span2 <= span2 <= 12:
            return (span1, gap1), (span2, gap2)
    return None


def _sample_color_specs(
    color_count: int, width: int, rng: np.random.Generator
) -> list[dict] | None:
    row_counts = _choose_row_counts(color_count, rng)
    filled_totals = _sample_filled_totals(row_counts, rng)
    if filled_totals is None:
        return None
    gap_totals = _choose_gap_totals(row_counts, filled_totals, rng)
    if gap_totals is None:
        return None

    colors = rng.choice(PALETTE, size=color_count, replace=False).tolist()
    rng.shuffle(colors)
    specs = []
    for color, row_count, filled_total, gap_total in zip(colors, row_counts, filled_totals, gap_totals):
        rows = []
        max_row_span = 0
        if row_count == 1:
            if filled_total > min(12, width - 2):
                return None
            pattern = _build_row_pattern(int(filled_total), int(gap_total) + 1, rng)
            rows.append({"span": int(filled_total), "pattern": pattern, "gaps": int(gap_total)})
            max_row_span = int(filled_total)
        else:
            split = _split_two_row_totals(int(filled_total), int(gap_total), rng)
            if split is None:
                return None
            for span, gaps in split:
                if span > min(12, width - 2):
                    return None
                pattern = _build_row_pattern(int(span), int(gaps) + 1, rng)
                rows.append({"span": int(span), "pattern": pattern, "gaps": int(gaps)})
                max_row_span = max(max_row_span, int(span))
        specs.append(
            {
                "color": int(color),
                "row_count": int(row_count),
                "rows": rows,
                "raw": int(filled_total - gap_total),
                "filled": int(filled_total),
                "max_row_span": int(max_row_span),
            }
        )

    fill_order = [item["color"] for item in sorted(specs, key=lambda item: (-item["filled"], item["color"]))]
    raw_order = [item["color"] for item in sorted(specs, key=lambda item: (-item["raw"], item["color"]))]
    span_order = [item["color"] for item in sorted(specs, key=lambda item: (-item["max_row_span"], item["color"]))]
    if fill_order == raw_order or fill_order == span_order:
        return None

    return specs


def _place_rows(grid: np.ndarray, specs: list[dict], rng: np.random.Generator) -> list[dict] | None:
    h, w = map(int, grid.shape)
    row_counts = [spec["row_count"] for spec in specs]
    min_needed = sum(row_counts) + (len(specs) - 1)
    if min_needed > h:
        return None

    slack = h - min_needed
    extras = rng.multinomial(slack, [1 / (len(specs) + 1)] * (len(specs) + 1))
    row = int(extras[0])
    placed = []
    for idx, spec in enumerate(specs):
        spec = {
            "color": spec["color"],
            "row_count": spec["row_count"],
            "raw": spec["raw"],
            "filled": spec["filled"],
            "max_row_span": spec["max_row_span"],
            "rows": [],
            "top": row,
        }
        for row_spec in specs[idx]["rows"]:
            span = int(row_spec["span"])
            if w - span - 1 <= 1:
                start = 1
            else:
                start = int(rng.integers(1, w - span))
            rr = row + len(spec["rows"])
            grid[rr, start : start + span] = row_spec["pattern"] * spec["color"]
            spec["rows"].append({"r": rr, "start": start, "span": span})
        placed.append(spec)
        row += spec["row_count"]
        if idx + 1 < len(specs):
            row += 1 + int(extras[idx + 1])
    return placed


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _get_bg(grid) != BG:
        return False

    stats = _color_stats(grid)
    if not (3 <= len(stats) <= 5):
        return False
    if len({item["raw"] for item in stats}) != len(stats):
        return False
    if len({item["filled"] for item in stats}) != len(stats):
        return False

    by_color = {item["color"]: item for item in stats}
    for color, item in by_color.items():
        for r in item["rows"]:
            row = grid[r]
            if set(map(int, np.unique(row))) - {BG, color}:
                return False
            cols = np.where(row == color)[0]
            if cols.size < 2:
                return False

    if not any(
        a["raw"] > b["raw"] and a["filled"] < b["filled"] for a in stats for b in stats
    ):
        return False
    if 1 in {item["row_count"] for item in stats} and 2 in {item["row_count"] for item in stats}:
        if not any(
            a["row_count"] > b["row_count"] and a["filled"] < b["filled"] for a in stats for b in stats
        ):
            return False

    fill_order = [item["color"] for item in sorted(stats, key=lambda item: (-item["filled"], item["top"]))]
    raw_order = [item["color"] for item in sorted(stats, key=lambda item: (-item["raw"], item["top"]))]
    top_order = [item["color"] for item in sorted(stats, key=lambda item: item["top"])]
    span_order = [item["color"] for item in sorted(stats, key=lambda item: (-item["max_row_span"], item["top"]))]
    if fill_order == raw_order or fill_order == top_order or fill_order == span_order:
        return False

    expected = _ring_output(fill_order)
    return np.array_equal(main(grid), expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    color_count = int(rng.integers(COLOR_COUNT_RANGE[0], COLOR_COUNT_RANGE[1] + 1))

    specs = _sample_color_specs(color_count, width, rng)
    if specs is None:
        return None

    rng.shuffle(specs)
    grid = np.zeros((height, width), dtype=int)
    placed = _place_rows(grid, specs, rng)
    if placed is None:
        return None

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
