import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6

FRAME_COLORS = (BLUE, RED, PINK)
FRAME_SIZE = 5
INNER_SIZE = 3
STRIDE = 6
PADDING = 1


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask_1d.astype(bool).tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _parse_frames(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]], list[tuple[int, int]]]:
    row_runs = _runs(np.any(grid != BG, axis=1))
    col_runs = _runs(np.any(grid != BG, axis=0))
    assert len(row_runs) in (2, 3)
    assert len(col_runs) in (2, 3)
    assert len(row_runs) * len(col_runs) == 6
    assert all(r1 - r0 + 1 == FRAME_SIZE for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == FRAME_SIZE for c0, c1 in col_runs)

    frames: list[dict] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, block[block != BG]))
            assert len(colors) == 1
            color = colors.pop()
            assert color in FRAME_COLORS
            assert np.all(block[0, :] == color)
            assert np.all(block[-1, :] == color)
            assert np.all(block[:, 0] == color)
            assert np.all(block[:, -1] == color)
            interior = block[1:-1, 1:-1]
            assert np.all((interior == BG) | (interior == color))
            frames.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "color": color,
                    "interior_mask": interior == color,
                }
            )

    counts = defaultdict(int)
    for frame in frames:
        counts[frame["color"]] += 1
    assert set(counts) == set(FRAME_COLORS)
    assert all(counts[color] == 2 for color in FRAME_COLORS)
    return frames, row_runs, col_runs


def _hole_mask(interior_mask: np.ndarray) -> np.ndarray:
    black = ~interior_mask.astype(bool)
    reached = np.zeros_like(black, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(INNER_SIZE):
        for c in range(INNER_SIZE):
            if black[r, c] and (r in (0, INNER_SIZE - 1) or c in (0, INNER_SIZE - 1)):
                reached[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < INNER_SIZE and 0 <= nc < INNER_SIZE and black[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                q.append((nr, nc))

    return black & ~reached


def _outline_coords(r0: int, c0: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    coords: list[tuple[int, int]] = []
    for r in range(r0 - 1, r0 + FRAME_SIZE + 1):
        for c in range(c0 - 1, c0 + FRAME_SIZE + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if r0 <= r < r0 + FRAME_SIZE and c0 <= c < c0 + FRAME_SIZE:
                continue
            coords.append((r, c))
    return coords


def _group_union(frames: list[dict]) -> dict[int, np.ndarray]:
    unions: dict[int, np.ndarray] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[frame["color"]].append(frame)
    for color, group in grouped.items():
        union = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        for frame in group:
            union |= frame["interior_mask"]
        unions[color] = union
    return unions


def _expected_output(grid: np.ndarray) -> np.ndarray:
    frames, _, _ = _parse_frames(grid)
    unions = _group_union(frames)

    out = grid.copy()
    yellow_outline = np.zeros(grid.shape, dtype=bool)
    green_outline = np.zeros(grid.shape, dtype=bool)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[frame["color"]].append(frame)

    for color, group in grouped.items():
        union = unions[color]
        holes = _hole_mask(union)
        has_hole = bool(np.any(holes))
        for frame in group:
            r0 = frame["r0"]
            c0 = frame["c0"]
            interior_view = out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
            interior_view[union] = color
            if has_hole:
                interior_view[holes] = YELLOW

            for r, c in _outline_coords(r0, c0, grid.shape):
                if has_hole:
                    green_outline[r, c] = True
                else:
                    yellow_outline[r, c] = True

    out[yellow_outline] = YELLOW
    out[green_outline] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each same-colored frame pair by motif union, then choose outline and hole fill from the completed topology.

    Input structure:
    - A black background contains 6 equal 5x5 colored square frames arranged as a 2x3 or 3x2 lattice with 1-cell gaps.
    - Frame colors are exactly blue, red, and pink, each appearing on two frames; each frame has a solid 1-pixel border.
    - Inside every frame, the 3x3 interior contains only black or extra pixels of that frame's own color.

    Rule:
    - Group frames by border color and take the union of the two same-colored 3x3 interior motifs.
    - Paint that union back into both frames of the group.
    - In the completed 3x3 interior, treat black cells reachable from the interior edge by 4-connectivity as exterior; any unreachable black cell is a hole.
    - If a group's completed motif has a hole, fill each hole pixel yellow and draw a green 1-cell outside outline around both frames of that group.
    - If a group's completed motif has no hole, draw a yellow 1-cell outside outline around both frames of that group.
    - When outside outlines overlap in the 1-cell gaps, green overrides yellow; original frame and completed motif pixels stay unchanged.

    Key insight:
    - The same-color pairing determines a shared completed motif, and that completed motif's topology decides the later marking colors for the whole pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _expected_output(g)


HOLE_CORNERS = [(0, 0), (0, 2), (2, 0), (2, 2)]
ALL_POSITIONS = [(r, c) for r in range(INNER_SIZE) for c in range(INNER_SIZE)]


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_hole_union() -> np.ndarray:
    cells = {(0, 1), (1, 0), (1, 2), (2, 1)}
    extra_count = random.randint(0, len(HOLE_CORNERS))
    for corner in random.sample(HOLE_CORNERS, extra_count):
        cells.add(corner)
    return _mask_from_cells(cells)


def _random_non_hole_union() -> np.ndarray:
    for _ in range(200):
        size = random.randint(2, 6)
        cells = set(random.sample(ALL_POSITIONS, size))
        mask = _mask_from_cells(cells)
        if not np.any(_hole_mask(mask)):
            return mask
    raise RuntimeError("failed to sample a non-hole union")


def _split_union(union: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(union)]
    u = len(cells)
    assert 2 <= u <= 8

    feasible = []
    for s1 in range(1, 5):
        for s2 in range(1, 5):
            if s1 + s2 >= u:
                overlap = s1 + s2 - u
                if overlap <= min(s1, s2):
                    feasible.append((s1, s2, overlap))

    for _ in range(200):
        s1, s2, overlap = random.choice(feasible)
        shuffled = cells[:]
        random.shuffle(shuffled)
        shared = shuffled[:overlap]
        remaining = shuffled[overlap:]
        only1 = remaining[: s1 - overlap]
        only2 = remaining[s1 - overlap :]
        a = set(shared) | set(only1)
        b = set(shared) | set(only2)
        if a == b:
            continue
        if 1 <= len(a) <= 4 and 1 <= len(b) <= 4:
            return _mask_from_cells(a), _mask_from_cells(b)

    raise RuntimeError("failed to split union into two distinct input motifs")


def _frame_patch(color: int, motif: np.ndarray) -> np.ndarray:
    patch = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1][motif] = color
    return patch


def _validate_semantics(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert not np.any(np.isin(grid, [GREEN, YELLOW]))

    frames, row_runs, col_runs = _parse_frames(grid)
    assert len(row_runs) * len(col_runs) == 6

    grouped: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[frame["color"]].append(frame)

    statuses: list[bool] = []
    for color in FRAME_COLORS:
        group = grouped[color]
        assert len(group) == 2
        a = group[0]["interior_mask"]
        b = group[1]["interior_mask"]
        assert not np.array_equal(a, b)
        assert 1 <= int(np.sum(a)) <= 4
        assert 1 <= int(np.sum(b)) <= 4
        union = a | b
        has_hole = bool(np.any(_hole_mask(union)))
        statuses.append(has_hole)

    assert any(statuses)
    assert not all(statuses)

    out = main(grid)
    expected = _expected_output(grid)
    assert np.array_equal(out, expected)

    green_mask = out == GREEN
    yellow_mask = out == YELLOW
    assert not np.any(green_mask & (grid != BG))
    allowed_yellow = yellow_mask & (grid == BG)
    assert np.array_equal(yellow_mask, allowed_yellow)


def generate_input() -> np.ndarray:
    for _ in range(400):
        rows, cols = random.choice(((2, 3), (3, 2)))
        height = PADDING * 2 + rows * FRAME_SIZE + (rows - 1)
        width = PADDING * 2 + cols * FRAME_SIZE + (cols - 1)
        grid = np.zeros((height, width), dtype=int)

        hole_colors = set(random.sample(FRAME_COLORS, random.choice((1, 2))))
        motifs_by_color: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for color in FRAME_COLORS:
            union = _random_hole_union() if color in hole_colors else _random_non_hole_union()
            motifs_by_color[color] = _split_union(union)

        placements = [color for color in FRAME_COLORS for _ in range(2)]
        random.shuffle(placements)
        origins = [(PADDING + r * STRIDE, PADDING + c * STRIDE) for r in range(rows) for c in range(cols)]

        used_index: dict[int, int] = defaultdict(int)
        for (r0, c0), color in zip(origins, placements):
            motif_idx = used_index[color]
            used_index[color] += 1
            patch = _frame_patch(color, motifs_by_color[color][motif_idx])
            grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = patch

        try:
            _validate_semantics(grid)
            return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid input for puzzle 5")
