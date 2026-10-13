from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3)
PATTERN = 5
INNER_SIZES = (3, 5)
OUTER_FROM_INNER = {3: 5, 5: 7}
COUNT_OPTIONS = (
    (2, 2, 2),
    (3, 2, 2),
    (3, 3, 2),
    (3, 3, 3),
)
COUNT_WEIGHTS = (1, 3, 3, 2)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _rotations_distinct(arr: np.ndarray) -> bool:
    keys = {np.rot90(arr, k).tobytes() for k in range(4)}
    return len(keys) == 4


def _build_pattern_catalog(size: int) -> list[np.ndarray]:
    cells = [(r, c) for r in range(size) for c in range(size)]
    catalog: list[np.ndarray] = []
    for area in range(2, 5):
        for combo in itertools.combinations(cells, area):
            mask = np.zeros((size, size), dtype=bool)
            rows, cols = zip(*combo)
            mask[rows, cols] = True
            if len(_components(mask)) != 1:
                continue
            arr = np.zeros((size, size), dtype=int)
            arr[mask] = PATTERN
            if _rotations_distinct(arr):
                catalog.append(arr)
    return catalog


PATTERN_CATALOG = {size: _build_pattern_catalog(size) for size in INNER_SIZES}


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in FRAME_COLORS:
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h == w and h in (5, 7), "frame must be a 5x5 or 7x7 square ring"
            expected_area = h * w - (h - 2) * (w - 2)
            assert int(coords.shape[0]) == expected_area, "frame border must be one pixel thick"

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(patch[0, :] == color)
            assert np.all(patch[-1, :] == color)
            assert np.all(patch[:, 0] == color)
            assert np.all(patch[:, -1] == color)

            inner = patch[1:-1, 1:-1].copy()
            assert inner.shape[0] in INNER_SIZES and inner.shape[0] == inner.shape[1]
            assert set(map(int, np.unique(inner))) <= {BG, PATTERN}, "frame interior must contain only background and gray pattern"

            gray_mask = inner == PATTERN
            assert 2 <= int(gray_mask.sum()) <= 4, "pattern must have 2-4 pixels"
            gray_components = _components(gray_mask)
            assert len(gray_components) == 1, "pattern must be 4-connected"

            frames.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "left": int(c0),
                    "inner": inner,
                }
            )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each frame's internal gray pattern by an angle determined by its left-to-right index inside the same-color frame group.

    Input structure:
    - The grid is a black background containing several non-touching square colored frames.
    - Frame border colors are blue, red, or green, and each frame encloses a 3x3 or 5x5 interior.
    - Inside each frame is a small gray 4-connected asymmetric pattern on black.

    Rule:
    - Group frames by their border color.
    - Within each color group, sort frames by the frame's left border column.
    - Rotate only the interior pattern of the k-th frame in that color order by (k-1)*90 degrees clockwise around the interior square's center.
    - Keep every frame border unchanged, and restore vacated interior cells to black.

    Key insight:
    - Rotation angle is not a property of the pattern itself; it comes from the frame's ordinal position within its same-color group, so color grouping must be solved before rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _extract_frames(g)

    by_color: dict[int, list[dict]] = {color: [] for color in FRAME_COLORS}
    for frame in frames:
        by_color[frame["color"]].append(frame)

    for color, group in by_color.items():
        if not group:
            continue
        lefts = [frame["left"] for frame in group]
        assert len(lefts) == len(set(lefts)), f"same-color frames must have unique left columns for color {color}"
        for idx, frame in enumerate(sorted(group, key=lambda item: item["left"])):
            r0, c0, r1, c1 = frame["bbox"]
            inner = frame["inner"]
            rotated = np.rot90(inner, - (idx % 4))
            out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def _sample_counts() -> dict[int, int]:
    counts = list(random.choices(COUNT_OPTIONS, weights=COUNT_WEIGHTS, k=1)[0])
    random.shuffle(counts)
    return {color: counts[i] for i, color in enumerate(FRAME_COLORS)}


def _random_slots(limit: int) -> list[int]:
    assert limit >= 24
    extra = limit - 24
    gaps = np.random.multinomial(extra, [0.25, 0.25, 0.25, 0.25]).tolist()
    return [1 + gaps[0], 9 + gaps[0] + gaps[1], 17 + gaps[0] + gaps[1] + gaps[2]]


def _choose_positions(counts: dict[int, int]) -> dict[int, list[tuple[int, int]]]:
    latin = {
        1: [(0, 0), (1, 1), (2, 2)],
        2: [(0, 1), (1, 2), (2, 0)],
        3: [(0, 2), (1, 0), (2, 1)],
    }
    for _ in range(200):
        picks: dict[int, list[tuple[int, int]]] = {}
        for color in FRAME_COLORS:
            picks[color] = random.sample(latin[color], counts[color])
        x_positions = {color: sorted(col for _row, col in picks[color]) for color in FRAME_COLORS}
        if any(
            len(cols) >= 2 and any(other > cols[0] and other < cols[-1] for other_color, other_cols in x_positions.items() if other_color != color for other in other_cols)
            for color, cols in x_positions.items()
        ):
            return picks
    raise RuntimeError("failed to pick interleaved color positions")


def _permute_cells(cells: dict[int, list[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
    row_perm = random.sample([0, 1, 2], 3)
    col_perm = random.sample([0, 1, 2], 3)
    out: dict[int, list[tuple[int, int]]] = {}
    for color, positions in cells.items():
        out[color] = [(row_perm[row], col_perm[col]) for row, col in positions]
    return out


def _pattern_for_size(size: int) -> np.ndarray:
    return random.choice(PATTERN_CATALOG[size]).copy()


def _draw_frame(grid: np.ndarray, top: int, left: int, outer: int, color: int, pattern: np.ndarray) -> None:
    bottom = top + outer - 1
    right = left + outer - 1
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    inner = pattern.shape[0]
    grid[top + 1 : top + 1 + inner, left + 1 : left + 1 + inner] = pattern


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert inp.shape == out.shape

    frames = _extract_frames(inp)
    assert 6 <= len(frames) <= 9
    by_color: dict[int, list[dict]] = {color: [] for color in FRAME_COLORS}
    for frame in frames:
        by_color[frame["color"]].append(frame)

    counts = [len(by_color[color]) for color in FRAME_COLORS]
    assert all(2 <= count <= 3 for count in counts)

    global_changed = 0
    seen_180 = False
    for color in FRAME_COLORS:
        group = sorted(by_color[color], key=lambda item: item["left"])
        lefts = [frame["left"] for frame in group]
        assert len(lefts) == len(set(lefts))
        for idx, frame in enumerate(group):
            r0, c0, r1, c1 = frame["bbox"]
            expected_inner = np.rot90(frame["inner"], - (idx % 4))
            actual_inner = out[r0 + 1 : r1, c0 + 1 : c1]
            assert np.array_equal(actual_inner, expected_inner)
            if idx >= 1 and not np.array_equal(expected_inner, frame["inner"]):
                global_changed += 1
            if idx % 4 == 2:
                seen_180 = True
        if len(group) >= 2:
            assert any(
                other["left"] > group[0]["left"] and other["left"] < group[-1]["left"]
                for other in frames
                if other["color"] != color
            ) or len(group) == 2

    assert global_changed >= 2, "need multiple visibly rotated frames"
    if any(count == 3 for count in counts):
        assert seen_180, "a 3-frame group should exercise the 180-degree branch"

    assert not np.array_equal(inp, out), "input and output must differ"
    assert len({int(v) for v in np.unique(inp)} - {BG}) >= 4
    assert _get_bg(inp) == BG


def generate_input() -> np.ndarray:
    for _ in range(500):
        height = random.randint(24, 30)
        width = random.randint(24, 30)
        row_slots = _random_slots(height)
        col_slots = _random_slots(width)
        counts = _sample_counts()
        cells = _permute_cells(_choose_positions(counts))

        grid = np.zeros((height, width), dtype=int)
        for color in FRAME_COLORS:
            for row_idx, col_idx in cells[color]:
                inner = random.choice(INNER_SIZES)
                outer = OUTER_FROM_INNER[inner]
                row_jitter = random.randint(0, 7 - outer)
                col_jitter = random.randint(0, 7 - outer)
                top = row_slots[row_idx] + row_jitter
                left = col_slots[col_idx] + col_jitter
                _draw_frame(grid, top, left, outer, color, _pattern_for_size(inner))

        try:
            out = main(grid)
            _semantic_checks(grid, out)
            return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid puzzle instance")
