from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_H = 10
MAX_H = 30
MIN_W = 12
MAX_W = 30

MIN_SEGMENTS = 2
MAX_SEGMENTS = 5
MIN_INTERVAL = 2
MAX_INTERVAL = 7

RULER_COL = 1
SEGMENT_LEFT_MIN = 4
SEGMENT_RIGHT_MARGIN = 1
SEGMENT_ROW_GAP = 2

TOP_MARGIN_RANGE = (1, 4)
BOTTOM_TAIL_RANGE = (0, 3)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_ruler(grid: np.ndarray, bg: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != c1:
            continue
        if len(comp) != r1 - r0 + 1:
            continue
        colors = [int(grid[r, c0]) for r, _ in comp]
        counts = Counter(colors)
        if len(counts) < 2:
            continue
        dominant, dominant_count = counts.most_common(1)[0]
        tick_rows = [r for r in range(r0, r1 + 1) if int(grid[r, c0]) != int(dominant)]
        if not tick_rows or dominant_count <= len(comp) // 2:
            continue
        candidates.append(
            {
                "top": int(r0),
                "bottom": int(r1),
                "col": int(c0),
                "grey": int(dominant),
                "tick_rows": tick_rows,
                "height": int(r1 - r0 + 1),
            }
        )

    assert candidates, "expected one multicolor vertical ruler"
    candidates.sort(key=lambda cand: (-int(cand["height"]), int(cand["col"])))
    return candidates[0]


def _extract_segments(grid: np.ndarray, bg: int, ruler: dict[str, object]) -> dict[int, dict[str, int]]:
    mask = grid != bg
    for r in range(int(ruler["top"]), int(ruler["bottom"]) + 1):
        mask[r, int(ruler["col"])] = False

    segments: dict[int, dict[str, int]] = {}
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        colors = {int(grid[r, c]) for r, c in comp}
        if r0 != r1 or len(colors) != 1:
            continue
        color = next(iter(colors))
        segments[color] = {
            "row": int(r0),
            "c0": int(c0),
            "c1": int(c1),
            "length": int(c1 - c0 + 1),
        }

    assert segments, "expected horizontal colored segments"
    return segments


def _interval_mapping(
    grid: np.ndarray,
    ruler: dict[str, object],
    segments: dict[int, dict[str, int]],
) -> dict[int, tuple[int, int]]:
    col = int(ruler["col"])
    top = int(ruler["top"])
    tick_rows = [int(r) for r in ruler["tick_rows"]]
    tick_colors = [int(grid[r, col]) for r in tick_rows]
    segment_lengths = {int(color): int(data["length"]) for color, data in segments.items()}

    schemes: list[dict[int, tuple[int, int]]] = []

    prev = top
    from_top: dict[int, tuple[int, int]] = {}
    for row, color in zip(tick_rows, tick_colors):
        from_top[int(color)] = (int(prev), int(row))
        prev = row
    schemes.append(from_top)

    if len(tick_rows) >= 2:
        prev = tick_rows[0]
        between_ticks: dict[int, tuple[int, int]] = {}
        for row, color in zip(tick_rows[1:], tick_colors[1:]):
            between_ticks[int(color)] = (int(prev), int(row))
            prev = row
        schemes.append(between_ticks)

    for mapping in schemes:
        if set(mapping.keys()) != set(segment_lengths.keys()):
            continue
        if all(int(lower - upper) == int(segment_lengths[color]) for color, (upper, lower) in mapping.items()):
            return mapping

    fallback: dict[int, tuple[int, int]] = {}
    prev = top
    top_candidates = {int(color): (int(prev_row), int(row)) for prev_row, row, color in zip([top] + tick_rows[:-1], tick_rows, tick_colors)}
    tick_candidates = {
        int(color): (int(tick_rows[i - 1]), int(tick_rows[i]))
        for i, color in enumerate(tick_colors[1:], start=1)
    }

    for color, length in segment_lengths.items():
        choices: list[tuple[int, int]] = []
        if color in top_candidates and top_candidates[color][1] - top_candidates[color][0] == length:
            choices.append(top_candidates[color])
        if color in tick_candidates and tick_candidates[color][1] - tick_candidates[color][0] == length:
            choices.append(tick_candidates[color])
        unique_choices = list(dict.fromkeys(choices))
        assert len(unique_choices) == 1, "segment must match exactly one ruler interval"
        fallback[int(color)] = unique_choices[0]
    return fallback


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    ruler = _find_ruler(g, bg)
    segments = _extract_segments(g, bg, ruler)
    mapping = _interval_mapping(g, ruler, segments)

    out = g.copy()
    for data in segments.values():
        row = int(data["row"])
        out[row, int(data["c0"]) : int(data["c1"]) + 1] = bg

    col = int(ruler["col"])
    for color, (upper, lower) in mapping.items():
        assert int(g[lower, col]) == int(color), "interval must end at the matching colored tick"
        out[int(upper) + 1 : int(lower) + 1, col] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each horizontal colored segment into the matching span of the left ruler.

    Input structure:
    - The background is black.
    - Near the left edge there is a vertical grey ruler whose single-pixel colored tick marks interrupt the grey line at different heights.
    - In the interior there are several horizontal monochrome segments; each segment shares its color with one ruler tick and its length equals the span from that tick up to the previous ruler mark.

    Rule:
    - Find the ruler and read its colored ticks from top to bottom.
    - For each horizontal segment, match it to the ruler tick of the same color and verify that the segment length equals that tick's upward span.
    - Erase the segment from its original position.
    - Repaint the ruler cells in that span with the segment color, filling from just below the previous mark down to and including the matching tick while leaving other ruler cells unchanged.

    Key insight:
    - A segment's length is not arbitrary decoration: it is the exact size of the ruler interval that the segment must occupy, so color matching and geometric placement reinforce each other.
    """
    return _transform(grid)


def _pick_rows(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    order = list(candidates)
    random.shuffle(order)
    picked: list[int] = []
    for row in order:
        if all(abs(int(row) - int(prev)) >= int(min_gap) for prev in picked):
            picked.append(int(row))
            if len(picked) == count:
                return picked
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        width = random.randint(MIN_W, MAX_W)
        max_interval = min(MAX_INTERVAL, width - SEGMENT_LEFT_MIN - SEGMENT_RIGHT_MARGIN)
        if max_interval < MIN_INTERVAL:
            continue

        length_pool = list(range(MIN_INTERVAL, max_interval + 1))
        max_segments = min(MAX_SEGMENTS, len(length_pool))
        if max_segments < MIN_SEGMENTS:
            continue

        num_segments = random.randint(MIN_SEGMENTS, max_segments)
        lengths = random.sample(length_pool, k=num_segments)
        random.shuffle(lengths)

        top = random.randint(*TOP_MARGIN_RANGE)
        bottom_tail = random.randint(*BOTTOM_TAIL_RANGE)
        total_span = sum(lengths)
        min_height = max(MIN_H, top + total_span + bottom_tail + 1)
        if min_height > MAX_H:
            continue
        height = random.randint(min_height, MAX_H)

        colors = random.sample(COLOR_POOL, k=num_segments)
        tick_rows: list[int] = []
        cursor = top
        for length in lengths:
            cursor += int(length)
            tick_rows.append(int(cursor))

        ruler_bottom = tick_rows[-1] + bottom_tail
        if ruler_bottom >= height:
            continue

        grid = np.full((height, width), BG, dtype=int)
        grid[top : ruler_bottom + 1, RULER_COL] = GREY
        for row, color in zip(tick_rows, colors):
            grid[int(row), RULER_COL] = int(color)

        forbidden_rows = set(tick_rows)
        forbidden_rows.add(top)
        segment_rows = _pick_rows([r for r in range(height) if r not in forbidden_rows], num_segments, SEGMENT_ROW_GAP)
        if segment_rows is None:
            continue

        for row, color, length in zip(segment_rows, colors, lengths):
            start_min = SEGMENT_LEFT_MIN
            start_max = width - length - SEGMENT_RIGHT_MARGIN
            if start_max < start_min:
                break
            start = random.randint(start_min, start_max)
            grid[int(row), start : start + int(length)] = int(color)
        else:
            try:
                out = _transform(grid)
            except AssertionError:
                continue
            if np.array_equal(out, grid):
                continue
            return grid

    raise RuntimeError("failed to generate a valid ruler-segment puzzle")
