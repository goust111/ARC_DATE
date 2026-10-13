from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
ARC_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500


def _nonzero_runs(row: np.ndarray) -> list[tuple[int, int]]:
    cols = np.flatnonzero(row != BACKGROUND)
    if cols.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    prev = int(cols[0])
    for col0 in cols[1:]:
        col = int(col0)
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    return runs


def _parse_segments(grid: np.ndarray) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    content_len: int | None = None

    for r, row in enumerate(grid):
        runs = _nonzero_runs(row)
        if not runs:
            continue

        assert len(runs) == 1, "each active row must contain exactly one horizontal segment"
        c0, c1 = runs[0]
        assert c1 - c0 + 1 >= 3, "a segment needs two ports and at least one content cell"

        left = int(grid[r, c0])
        right = int(grid[r, c1])
        assert left != BACKGROUND and right != BACKGROUND and left != right

        content = np.asarray(grid[r, c0 + 1 : c1], dtype=int).copy()
        assert np.all(content != BACKGROUND), "segment content must be a solid colored row"
        assert left not in set(map(int, content)), "left port color cannot appear in its content"
        assert right not in set(map(int, content)), "right port color cannot appear in its content"

        if content_len is None:
            content_len = int(content.size)
        else:
            assert int(content.size) == content_len, "all contents must have equal length"

        segments.append(
            {
                "row": int(r),
                "c0": int(c0),
                "c1": int(c1),
                "left": left,
                "right": right,
                "content": content,
            }
        )

    assert segments, "expected at least one ported row segment"
    return segments


def _chain_segments(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    left_colors = [int(seg["left"]) for seg in segments]
    right_colors = [int(seg["right"]) for seg in segments]
    left_counts = Counter(left_colors)
    right_counts = Counter(right_colors)

    assert all(count == 1 for count in left_counts.values()), "left ports must be unique"
    assert all(count == 1 for count in right_counts.values()), "right ports must be unique"

    heads = [color for color in left_colors if color not in right_counts]
    tails = [color for color in right_colors if color not in left_counts]
    assert len(heads) == 1, "there must be exactly one left-only chain head color"
    assert len(tails) == 1, "there must be exactly one right-only chain tail color"
    assert len(set(left_colors) & set(right_colors)) == len(segments) - 1

    by_left = {int(seg["left"]): seg for seg in segments}
    ordered: list[dict[str, object]] = []
    seen_lefts: set[int] = set()
    current = heads[0]

    while current in by_left:
        assert current not in seen_lefts, "port chain cannot loop"
        seen_lefts.add(current)
        seg = by_left[current]
        ordered.append(seg)
        current = int(seg["right"])

    assert current == tails[0], "chain must end at the unique right-only color"
    assert len(ordered) == len(segments), "chain must cover every segment"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract same-length row contents and stack them in same-color port-chain order.

    Input structure:
    - The background is black.
    - Several distinct rows each contain exactly one solid horizontal non-black segment.
    - Each segment has one colored left port, equal-length colored content, and one colored right port.
    - Port colors do not appear inside their own content row; one port color is left-only, one is right-only,
      and every other port color appears once as a right port and once as another segment's left port.

    Rule:
    - Find the segment whose left port color never appears as a right port.
    - Repeatedly dock to the segment whose left port color equals the current segment's right port color.
    - Remove the two port pixels from every chained segment and stack the contents from top to bottom
      in chain order.

    Key insight:
    - The visible row positions are distractors: equal-color port docking gives the unique segment order,
      so chain order overrides row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    segments = _parse_segments(g)
    ordered = _chain_segments(segments)
    return np.vstack([np.asarray(seg["content"], dtype=int) for seg in ordered]).astype(int)


def _sample_spaced_rows(rng: np.random.Generator, count: int) -> tuple[int, list[int]]:
    min_height = max(2 * count + 1, count + 4)
    for _ in range(100):
        height = int(rng.integers(min_height, 31))
        candidates = np.arange(1, height - 1)
        rows = sorted(map(int, rng.choice(candidates, size=count, replace=False)))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return height, rows
    raise RuntimeError("failed to sample separated segment rows")


def _sample_contents(
    rng: np.random.Generator,
    palette: list[int],
    count: int,
    width: int,
) -> list[np.ndarray]:
    contents: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(300):
        if len(contents) == count:
            return contents
        arr = rng.choice(palette, size=width, replace=True).astype(int)
        if len(set(map(int, arr))) < min(2, len(palette)):
            continue
        key = tuple(map(int, arr))
        if key in seen:
            continue
        seen.add(key)
        contents.append(arr)

    raise RuntimeError("failed to sample distinct content rows")


def _row_order_output(grid: np.ndarray) -> np.ndarray:
    segments = _parse_segments(grid)
    segments.sort(key=lambda seg: int(seg["row"]))
    return np.vstack([np.asarray(seg["content"], dtype=int) for seg in segments]).astype(int)


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    segment_count = int(rng.integers(3, 7))
    content_width = int(rng.integers(3, 8))
    port_colors = list(map(int, rng.choice(ARC_COLORS, size=segment_count + 1, replace=False)))
    content_palette = [color for color in ARC_COLORS if color not in port_colors]
    if len(content_palette) < 2:
        return None

    contents = _sample_contents(rng, content_palette, segment_count, content_width)
    height, rows = _sample_spaced_rows(rng, segment_count)
    run_width = content_width + 2
    width = int(rng.integers(run_width + 2, 31))

    chain_to_row_slot = list(rng.permutation(segment_count))
    if chain_to_row_slot == list(range(segment_count)):
        chain_to_row_slot[0], chain_to_row_slot[1] = chain_to_row_slot[1], chain_to_row_slot[0]

    grid = np.zeros((height, width), dtype=int)
    expected_rows: list[np.ndarray] = []

    for chain_idx in range(segment_count):
        row = rows[chain_to_row_slot[chain_idx]]
        c0 = int(rng.integers(1, width - run_width))
        left = port_colors[chain_idx]
        right = port_colors[chain_idx + 1]
        content = contents[chain_idx]

        grid[row, c0] = left
        grid[row, c0 + 1 : c0 + 1 + content_width] = content
        grid[row, c0 + run_width - 1] = right
        expected_rows.append(content)

    expected = np.vstack(expected_rows).astype(int)

    try:
        actual = main(grid)
        row_order = _row_order_output(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(row_order, expected):
        return None

    parsed = _parse_segments(grid)
    ordered = _chain_segments(parsed)
    if len({tuple(map(int, seg["content"])) for seg in parsed}) != segment_count:
        return None
    if [int(seg["row"]) for seg in ordered] == sorted(int(seg["row"]) for seg in parsed):
        return None

    return grid


def generate_input() -> np.ndarray:
    """Generate a random, non-degenerate port-chain extraction input."""
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            assert main(grid).ndim == 2 and max(main(grid).shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid port-chain input")
