import itertools
import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
FENCE = 9
ENDPOINT_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _fence_at(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return 0 <= r < h and 0 <= c < w and int(grid[r, c]) == FENCE


def _is_top_left(grid: np.ndarray, r: int, c: int) -> bool:
    return (
        _fence_at(grid, r, c)
        and _fence_at(grid, r, c + 1)
        and _fence_at(grid, r + 1, c)
        and not _fence_at(grid, r - 1, c)
        and not _fence_at(grid, r, c - 1)
    )


def _is_top_right(grid: np.ndarray, r: int, c: int) -> bool:
    return (
        _fence_at(grid, r, c)
        and _fence_at(grid, r, c - 1)
        and _fence_at(grid, r + 1, c)
        and not _fence_at(grid, r - 1, c)
        and not _fence_at(grid, r, c + 1)
    )


def _is_bottom_left(grid: np.ndarray, r: int, c: int) -> bool:
    return (
        _fence_at(grid, r, c)
        and _fence_at(grid, r, c + 1)
        and _fence_at(grid, r - 1, c)
        and not _fence_at(grid, r + 1, c)
        and not _fence_at(grid, r, c - 1)
    )


def _is_bottom_right(grid: np.ndarray, r: int, c: int) -> bool:
    return (
        _fence_at(grid, r, c)
        and _fence_at(grid, r, c - 1)
        and _fence_at(grid, r - 1, c)
        and not _fence_at(grid, r + 1, c)
        and not _fence_at(grid, r, c + 1)
    )


def _validate_fence(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> dict[str, object] | None:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    if not (_is_top_left(grid, r0, c0) and _is_top_right(grid, r0, c1)):
        return None
    if not (_is_bottom_left(grid, r1, c0) and _is_bottom_right(grid, r1, c1)):
        return None

    border_points = []
    for c in range(c0 + 1, c1):
        border_points.append((r0, c))
        border_points.append((r1, c))
    for r in range(r0 + 1, r1):
        border_points.append((r, c0))
        border_points.append((r, c1))

    for r, c in border_points:
        val = int(grid[r, c])
        if val not in (BG, FENCE):
            return None

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return None

    top: list[tuple[int, int]] = []
    bottom: list[tuple[int, int]] = []
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []

    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) == BG:
            if int(grid[r0, c - 1]) != FENCE or int(grid[r0, c + 1]) != FENCE:
                return None
            top.append((r0, c))
        if int(grid[r1, c]) == BG:
            if int(grid[r1, c - 1]) != FENCE or int(grid[r1, c + 1]) != FENCE:
                return None
            bottom.append((r1, c))

    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == BG:
            if int(grid[r - 1, c0]) != FENCE or int(grid[r + 1, c0]) != FENCE:
                return None
            left.append((r, c0))
        if int(grid[r, c1]) == BG:
            if int(grid[r - 1, c1]) != FENCE or int(grid[r + 1, c1]) != FENCE:
                return None
            right.append((r, c1))

    if len(top) != len(bottom) or len(left) != len(right):
        return None
    if not top and not left:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def _find_fences(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    top_right = {(r, c) for r in range(h) for c in range(w) if _is_top_right(grid, r, c)}
    bottom_left = {(r, c) for r in range(h) for c in range(w) if _is_bottom_left(grid, r, c)}
    bottom_right = {(r, c) for r in range(h) for c in range(w) if _is_bottom_right(grid, r, c)}

    fences: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for r0 in range(h):
        for c0 in range(w):
            if not _is_top_left(grid, r0, c0):
                continue
            cols = [c for rr, c in top_right if rr == r0 and c > c0]
            rows = [r for r, cc in bottom_left if cc == c0 and r > r0]
            for c1 in cols:
                for r1 in rows:
                    if (r1, c1) not in bottom_right:
                        continue
                    info = _validate_fence(grid, r0, c0, r1, c1)
                    if info is not None:
                        fences[info["bbox"]] = info
    result = sorted(fences.values(), key=lambda d: tuple(d["bbox"]))
    assert 2 <= len(result) <= 4, "expected 2-4 fences"
    return result


def _gap_pair_map(fences: list[dict[str, object]]) -> dict[tuple[int, int], tuple[int, int]]:
    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for fence in fences:
        top = list(fence["top"])
        bottom = list(fence["bottom"])
        left = list(fence["left"])
        right = list(fence["right"])
        assert len(top) == len(bottom)
        assert len(left) == len(right)
        for a, b in zip(top, bottom):
            pairs[a] = b
            pairs[b] = a
        for a, b in zip(left, right):
            pairs[a] = b
            pairs[b] = a
    return pairs


def _path_segments(grid: np.ndarray) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for cells in _components_mask(grid == PATH):
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        assert len(rows) == 1 or len(cols) == 1, "path segment must be straight"
        cell_set = set(cells)
        if len(cells) == 1:
            endpoints = cells[:]
        else:
            endpoints = []
            for r, c in cells:
                degree = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
                if degree == 1:
                    endpoints.append((r, c))
            assert len(endpoints) == 2, "non-trivial segment must have two ends"
        segments.append({"cells": cells, "endpoints": endpoints})
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey path chain by tracing it through rectangular maroon fences.

    Gaps on opposite sides of the same fence pair by order: top-to-bottom by
    left-to-right index, and left-to-right by top-to-bottom index.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fences = _find_fences(g)
    gap_pairs = _gap_pair_map(fences)

    seeds = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1)) and int(g[r, c]) not in (BG, PATH, FENCE)
    }
    assert seeds, "expected colored edge seeds"

    segments = _path_segments(g)
    attachment_to_segment: dict[tuple[int, int], int] = {}
    segment_attachments: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}

    for seg_id, seg in enumerate(segments):
        attachments: list[tuple[int, int]] = []
        if len(seg["cells"]) == 1:
            r, c = seg["cells"][0]
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and ((nr, nc) in seeds or (nr, nc) in gap_pairs):
                    attachments.append((nr, nc))
        else:
            for r, c in seg["endpoints"]:
                hits = []
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and ((nr, nc) in seeds or (nr, nc) in gap_pairs):
                        hits.append((nr, nc))
                assert len(hits) == 1, "segment endpoint must attach to one seed or gap"
                attachments.extend(hits)

        attachments = list(dict.fromkeys(attachments))
        assert len(attachments) == 2, "each segment must connect exactly two attachments"
        a0, a1 = attachments
        assert a0 not in attachment_to_segment and a1 not in attachment_to_segment
        attachment_to_segment[a0] = seg_id
        attachment_to_segment[a1] = seg_id
        segment_attachments[seg_id] = (a0, a1)

    out = g.copy()
    segment_colors: dict[int, int] = {}

    for seed in sorted(seeds):
        color = int(g[seed])
        current = seed
        seen = {seed}
        for _ in range(len(segments) + len(gap_pairs) + len(seeds) + 1):
            seg_id = attachment_to_segment.get(current)
            assert seg_id is not None, "broken chain: missing segment at attachment"
            if seg_id in segment_colors:
                assert segment_colors[seg_id] == color
            else:
                segment_colors[seg_id] = color
                for r, c in segments[seg_id]["cells"]:
                    out[r, c] = color

            a0, a1 = segment_attachments[seg_id]
            other = a1 if current == a0 else a0
            if other in seeds:
                assert int(g[other]) == color, "chain endpoints must share a color"
                break

            paired = gap_pairs[other]
            assert paired not in seen, "unexpected loop while tracing path"
            seen.add(paired)
            current = paired
        else:
            raise AssertionError("chain trace did not terminate")

    assert not np.any((g == PATH) & (out == PATH)), "all grey segments must be recolored"
    return out


def _spaced_choices(pool: list[int], count: int, min_sep: int = 2) -> list[tuple[int, ...]]:
    if count == 0:
        return [()]
    out = []
    for combo in itertools.combinations(pool, count):
        if all(combo[i + 1] - combo[i] >= min_sep for i in range(len(combo) - 1)):
            out.append(combo)
    return out


def _choose_rows_or_cols(
    rng: random.Random, pool: list[int], min_count: int, max_count: int, min_sep: int = 2
) -> list[int] | None:
    feasible: list[tuple[int, ...]] = []
    for count in range(min_count, max_count + 1):
        feasible.extend(_spaced_choices(pool, count, min_sep=min_sep))
    if not feasible:
        return None
    return list(rng.choice(feasible))


def _draw_fence(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, rows: list[int], cols: list[int]) -> None:
    grid[r0, c0 : c1 + 1] = FENCE
    grid[r1, c0 : c1 + 1] = FENCE
    grid[r0 : r1 + 1, c0] = FENCE
    grid[r0 : r1 + 1, c1] = FENCE
    for r in rows:
        grid[r, c0] = BG
        grid[r, c1] = BG
    for c in cols:
        grid[r0, c] = BG
        grid[r1, c] = BG


def _paint_horizontal(
    inp: np.ndarray,
    out: np.ndarray,
    row: int,
    spans: list[tuple[int, int]],
    color: int,
) -> None:
    inp[row, 0] = color
    inp[row, -1] = color
    out[row, 0] = color
    out[row, -1] = color

    prev = 0
    for left_gap, right_gap in spans:
        if left_gap - prev > 1:
            inp[row, prev + 1 : left_gap] = PATH
            out[row, prev + 1 : left_gap] = color
        prev = right_gap
    if inp.shape[1] - 1 - prev > 1:
        inp[row, prev + 1 : -1] = PATH
        out[row, prev + 1 : -1] = color


def _paint_vertical(
    inp: np.ndarray,
    out: np.ndarray,
    col: int,
    spans: list[tuple[int, int]],
    color: int,
) -> None:
    inp[0, col] = color
    inp[-1, col] = color
    out[0, col] = color
    out[-1, col] = color

    prev = 0
    for top_gap, bottom_gap in spans:
        if top_gap - prev > 1:
            inp[prev + 1 : top_gap, col] = PATH
            out[prev + 1 : top_gap, col] = color
        prev = bottom_gap
    if inp.shape[0] - 1 - prev > 1:
        inp[prev + 1 : -1, col] = PATH
        out[prev + 1 : -1, col] = color


def _pick_extra_axis_positions(
    rng: random.Random,
    per_fence_pool: list[list[int]],
    total_count: int,
) -> list[list[int]] | None:
    chosen = [[] for _ in per_fence_pool]
    for _ in range(total_count):
        options = []
        for idx, pool in enumerate(per_fence_pool):
            for val in pool:
                if all(abs(val - prev) >= 2 for prev in chosen[idx]):
                    options.append((idx, val))
        if not options:
            return None
        idx, val = rng.choice(options)
        chosen[idx].append(val)
        chosen[idx].sort()
    return chosen


def _build_row_mode(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_fences = rng.randint(2, 4)
    if n_fences == 4:
        width_range = (5, 6)
        gap_range = (2, 3)
        margin_range = (2, 3)
    elif n_fences == 3:
        width_range = (5, 7)
        gap_range = (2, 4)
        margin_range = (2, 4)
    else:
        width_range = (5, 8)
        gap_range = (2, 5)
        margin_range = (2, 5)

    fence_h = rng.randint(5, 8)
    widths = [rng.randint(*width_range) for _ in range(n_fences)]
    gaps = [rng.randint(*gap_range) for _ in range(n_fences - 1)]
    left_margin = rng.randint(*margin_range)
    right_margin = rng.randint(*margin_range)
    top_margin = rng.randint(2, 5)
    bottom_margin = rng.randint(2, 5)

    h = top_margin + fence_h + bottom_margin
    w = left_margin + sum(widths) + sum(gaps) + right_margin
    r0 = top_margin
    r1 = r0 + fence_h - 1

    safe_rows = list(range(r0 + 2, r1 - 1))
    horiz_rows = _choose_rows_or_cols(rng, safe_rows, 1, min(3, len(safe_rows)))
    if not horiz_rows:
        return None

    bboxes: list[tuple[int, int, int, int]] = []
    c = left_margin
    for i, width in enumerate(widths):
        bboxes.append((r0, c, r1, c + width - 1))
        c += width
        if i + 1 < n_fences:
            c += gaps[i]

    extra_capacity = len(ENDPOINT_COLORS) - len(horiz_rows)
    want_vertical = extra_capacity > 0 and rng.random() < 0.6
    total_vertical = 0
    extra_cols = [[] for _ in range(n_fences)]
    if want_vertical:
        max_vertical = min(extra_capacity, 4)
        total_vertical = rng.randint(1, max_vertical)
        pools = [list(range(c0 + 2, c1 - 1)) for _, c0, _, c1 in bboxes]
        picked = _pick_extra_axis_positions(rng, pools, total_vertical)
        if picked is None or sum(len(cols) for cols in picked) != total_vertical:
            return None
        extra_cols = picked

    colors = list(ENDPOINT_COLORS)
    rng.shuffle(colors)
    if len(horiz_rows) + total_vertical > len(colors):
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = inp.copy()

    for idx, (rr0, cc0, rr1, cc1) in enumerate(bboxes):
        _draw_fence(inp, rr0, cc0, rr1, cc1, horiz_rows, extra_cols[idx])
        _draw_fence(out, rr0, cc0, rr1, cc1, horiz_rows, extra_cols[idx])

    fence_spans = [(c0, c1) for _, c0, _, c1 in bboxes]
    for row in horiz_rows:
        color = colors.pop()
        _paint_horizontal(inp, out, row, fence_spans, color)

    for idx, cols in enumerate(extra_cols):
        rr0, _, rr1, _ = bboxes[idx]
        spans = [(rr0, rr1)]
        for col in cols:
            color = colors.pop()
            _paint_vertical(inp, out, col, spans, color)

    return inp, out


def _build_col_mode(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_fences = rng.randint(2, 4)
    if n_fences == 4:
        height_range = (5, 6)
        gap_range = (2, 3)
        margin_range = (2, 3)
    elif n_fences == 3:
        height_range = (5, 7)
        gap_range = (2, 4)
        margin_range = (2, 4)
    else:
        height_range = (5, 8)
        gap_range = (2, 5)
        margin_range = (2, 5)

    fence_w = rng.randint(5, 8)
    heights = [rng.randint(*height_range) for _ in range(n_fences)]
    gaps = [rng.randint(*gap_range) for _ in range(n_fences - 1)]
    top_margin = rng.randint(*margin_range)
    bottom_margin = rng.randint(*margin_range)
    left_margin = rng.randint(2, 5)
    right_margin = rng.randint(2, 5)

    h = top_margin + sum(heights) + sum(gaps) + bottom_margin
    w = left_margin + fence_w + right_margin
    c0 = left_margin
    c1 = c0 + fence_w - 1

    safe_cols = list(range(c0 + 2, c1 - 1))
    vert_cols = _choose_rows_or_cols(rng, safe_cols, 1, min(3, len(safe_cols)))
    if not vert_cols:
        return None

    bboxes: list[tuple[int, int, int, int]] = []
    r = top_margin
    for i, height in enumerate(heights):
        bboxes.append((r, c0, r + height - 1, c1))
        r += height
        if i + 1 < n_fences:
            r += gaps[i]

    extra_capacity = len(ENDPOINT_COLORS) - len(vert_cols)
    want_horizontal = extra_capacity > 0 and rng.random() < 0.6
    total_horizontal = 0
    extra_rows = [[] for _ in range(n_fences)]
    if want_horizontal:
        max_horizontal = min(extra_capacity, 4)
        total_horizontal = rng.randint(1, max_horizontal)
        pools = [list(range(r0 + 2, r1 - 1)) for r0, _, r1, _ in bboxes]
        picked = _pick_extra_axis_positions(rng, pools, total_horizontal)
        if picked is None or sum(len(rows) for rows in picked) != total_horizontal:
            return None
        extra_rows = picked

    colors = list(ENDPOINT_COLORS)
    rng.shuffle(colors)
    if len(vert_cols) + total_horizontal > len(colors):
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = inp.copy()

    for idx, (rr0, cc0, rr1, cc1) in enumerate(bboxes):
        _draw_fence(inp, rr0, cc0, rr1, cc1, extra_rows[idx], vert_cols)
        _draw_fence(out, rr0, cc0, rr1, cc1, extra_rows[idx], vert_cols)

    fence_spans = [(r0, r1) for r0, _, r1, _ in bboxes]
    for col in vert_cols:
        color = colors.pop()
        _paint_vertical(inp, out, col, fence_spans, color)

    for idx, rows in enumerate(extra_rows):
        _, cc0, _, cc1 = bboxes[idx]
        spans = [(cc0, cc1)]
        for row in rows:
            color = colors.pop()
            _paint_horizontal(inp, out, row, spans, color)

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    builder = _build_row_mode if rng.random() < 0.5 else _build_col_mode
    built = builder(rng)
    if built is None:
        return None
    inp, expected = built
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")
