from __future__ import annotations

from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7
PALETTE = (2, 3, 4, 5, 6, 8, 9)
SIDE_TO_ROT = {"top": 0, "right": 3, "bottom": 2, "left": 1}
LAYOUTS = ((1, 2), (2, 1), (2, 2))
MAX_GENERATION_ATTEMPTS = 400


def _row_groups(flags: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(flags.tolist()):
        if value and start is None:
            start = int(idx)
        elif not value and start is not None:
            groups.append((start, int(idx) - 1))
            start = None
    if start is not None:
        groups.append((start, len(flags) - 1))
    return groups


def _orange_panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == ORANGE)
    assert coords.size > 0, "missing orange sample frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert np.all(grid[r0, c0 : c1 + 1] == ORANGE)
    assert np.all(grid[r1, c0 : c1 + 1] == ORANGE)
    assert np.all(grid[r0 : r1 + 1, c0] == ORANGE)
    assert np.all(grid[r0 : r1 + 1, c1] == ORANGE)
    return r0, c0, r1, c1


def _room_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    blue_mask = grid == BLUE
    wall_rows = [int(r) for r in np.flatnonzero(blue_mask.sum(axis=1) > 3)]
    wall_cols = [int(c) for c in np.flatnonzero(blue_mask.sum(axis=0) > 3)]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "missing room lattice"

    row_spans = [int(wall_rows[i + 1] - wall_rows[i] - 1) for i in range(len(wall_rows) - 1)]
    col_spans = [int(wall_cols[i + 1] - wall_cols[i] - 1) for i in range(len(wall_cols) - 1)]
    sizes = row_spans + col_spans
    assert sizes and all(size == sizes[0] and size > 0 for size in sizes), "rooms must be same square size"
    size = int(sizes[0])

    rooms = [
        (wall_rows[ri], wall_cols[ci], wall_rows[ri + 1], wall_cols[ci + 1])
        for ri in range(len(wall_rows) - 1)
        for ci in range(len(wall_cols) - 1)
    ]
    assert rooms, "no rooms found"
    return size, rooms


def _extract_samples(grid: np.ndarray, room_size: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = _orange_panel_bbox(grid)
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.shape[1] == room_size, "sample width must match room size"
    groups = _row_groups(np.any(interior != BLACK, axis=1))
    assert len(groups) == 2, "expected two sample patterns"
    samples: list[np.ndarray] = []
    for start, end in groups:
        assert end - start + 1 == room_size, "sample height must match room size"
        samples.append(interior[start : end + 1, :].copy())
    return samples


def _room_doors(grid: np.ndarray, room: tuple[int, int, int, int]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = room
    checks = (
        ("top", grid[r0, c0 + 1 : c1], lambda idx: (r0, c0 + 1 + idx)),
        ("bottom", grid[r1, c0 + 1 : c1], lambda idx: (r1, c0 + 1 + idx)),
        ("left", grid[r0 + 1 : r1, c0], lambda idx: (r0 + 1 + idx, c0)),
        ("right", grid[r0 + 1 : r1, c1], lambda idx: (r0 + 1 + idx, c1)),
    )

    doors: list[dict[str, object]] = []
    for side, segment, to_pos in checks:
        assert np.all(np.isin(segment, [BLACK, BLUE]))
        holes = np.flatnonzero(segment == BLACK)
        assert len(holes) <= 1, "each wall side may contain at most one door"
        if len(holes) == 1:
            doors.append({"side": side, "pos": tuple(map(int, to_pos(int(holes[0]))))})

    assert 1 <= len(doors) <= 2, "each room must have one or two doors"
    return doors


def _primary_side(doors: list[dict[str, object]]) -> str:
    chosen = min(doors, key=lambda item: (item["pos"][0], item["pos"][1]))  # type: ignore[index]
    return str(chosen["side"])


def _rotate_sample(sample: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(sample, SIDE_TO_ROT[side])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue-walled room with the sample pattern chosen by door count and rotated by door direction.

    Input structure:
    - The background is black.
    - Blue one-cell-thick horizontal and vertical walls form a lattice of same-sized square rooms on the left.
    - Each room boundary contains one or two one-cell black door gaps, while the room interiors start empty.
    - An orange rectangular frame on the right encloses two sample patterns stacked vertically; the top sample means "1 door" and the lower sample means "2 doors".

    Rule:
    - Recover the room size from the blue wall lattice and extract the two sample windows from the orange panel.
    - For every room, count its door gaps by checking the four wall sides.
    - Use the top sample for one-door rooms and the lower sample for two-door rooms.
    - Rotate the chosen sample from its default "door on top" orientation so that it points toward the room's door; with two doors, use the higher of the two doors as the primary one.
    - Copy the rotated sample into the room interior and leave the blue walls, black doors, and orange sample panel unchanged.

    Key insight:
    - A door plays two roles at once: its count selects which sample to use, and its side (or the higher side when there are two) fixes the rotation of that sample.
    """
    g = np.asarray(grid, dtype=int)
    room_size, rooms = _room_lattice(g)
    samples = _extract_samples(g, room_size)

    out = g.copy()
    for room in rooms:
        doors = _room_doors(g, room)
        sample = samples[len(doors) - 1]
        primary = _primary_side(doors)
        r0, c0, r1, c1 = room
        out[r0 + 1 : r1, c0 + 1 : c1] = _rotate_sample(sample, primary)
    return out


def _pattern_one(size: int, color: int, accent: int, rng: np.random.Generator) -> np.ndarray:
    pattern = np.full((size, size), BLACK, dtype=int)
    stem_col = int(rng.integers(1, size - 1))
    arm_row = int(rng.integers(1, size - 1))
    arm_end = int(rng.integers(0, stem_col + 1))

    pattern[:, stem_col] = color
    pattern[size - 1, stem_col:] = color
    pattern[arm_row, : arm_end + 1] = color
    pattern[0, stem_col] = accent
    pattern[arm_row, 0] = accent
    return pattern


def _pattern_two(size: int, color: int, accent: int, rng: np.random.Generator) -> np.ndarray:
    pattern = np.full((size, size), BLACK, dtype=int)
    stem_col = int(rng.integers(1, max(2, size - 1)))
    arm_row = int(rng.integers(1, size - 1))

    pattern[0, :] = color
    pattern[:, size - 1] = color
    pattern[1:, stem_col] = color
    pattern[arm_row, stem_col : size - 1] = color
    pattern[0, 0] = accent
    pattern[arm_row, stem_col] = accent
    return pattern


def _rotations_unique(pattern: np.ndarray) -> bool:
    signatures = {np.rot90(pattern, k).tobytes() for k in range(4)}
    return len(signatures) == 4


def _patterns_compatible(sample_a: np.ndarray, sample_b: np.ndarray) -> bool:
    if not (_rotations_unique(sample_a) and _rotations_unique(sample_b)):
        return False
    for ka in range(4):
        rot_a = np.rot90(sample_a, ka)
        for kb in range(4):
            if np.array_equal(rot_a, np.rot90(sample_b, kb)):
                return False
    return True


def _available_sides(room_row: int, room_col: int, n_rows: int, n_cols: int) -> list[str]:
    sides: list[str] = []
    if room_row == 0:
        sides.append("top")
    if room_row == n_rows - 1:
        sides.append("bottom")
    if room_col == 0:
        sides.append("left")
    if room_col == n_cols - 1:
        sides.append("right")
    return sides


def _choose_doors(
    room: tuple[int, int, int, int], sides: list[str], door_count: int, rng: np.random.Generator
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = room
    if door_count == 1:
        chosen_sides = [str(rng.choice(sides))]
    else:
        pairs = list(combinations(sides, 2))
        chosen_sides = pairs[int(rng.integers(0, len(pairs)))]

    doors: list[dict[str, object]] = []
    shared_rows: dict[str, int] = {}
    if set(chosen_sides) == {"left", "right"}:
        picks = [int(v) for v in rng.choice(np.arange(r0 + 1, r1), size=2, replace=False)]
        rng.shuffle(picks)
        shared_rows = {"left": picks[0], "right": picks[1]}

    for side in chosen_sides:
        if side == "top":
            pos = (r0, int(rng.integers(c0 + 1, c1)))
        elif side == "bottom":
            pos = (r1, int(rng.integers(c0 + 1, c1)))
        elif side == "left":
            pos = (shared_rows.get("left", int(rng.integers(r0 + 1, r1))), c0)
        else:
            pos = (shared_rows.get("right", int(rng.integers(r0 + 1, r1))), c1)
        doors.append({"side": side, "pos": tuple(map(int, pos))})

    return doors


def _draw_wall_lattice(
    grid: np.ndarray, top: int, left: int, room_size: int, n_rows: int, n_cols: int
) -> tuple[list[int], list[int]]:
    wall_rows = [int(top + idx * (room_size + 1)) for idx in range(n_rows + 1)]
    wall_cols = [int(left + idx * (room_size + 1)) for idx in range(n_cols + 1)]
    for row in wall_rows:
        grid[row, wall_cols[0] : wall_cols[-1] + 1] = BLUE
    for col in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, col] = BLUE
    return wall_rows, wall_cols


def _render_expected(
    grid: np.ndarray, room_specs: list[dict[str, object]], samples: list[np.ndarray]
) -> np.ndarray:
    out = grid.copy()
    for spec in room_specs:
        r0, c0, r1, c1 = spec["room"]  # type: ignore[assignment]
        doors = spec["doors"]  # type: ignore[assignment]
        sample = samples[int(spec["door_count"]) - 1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _rotate_sample(sample, _primary_side(doors))
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rows, n_cols = LAYOUTS[int(rng.integers(0, len(LAYOUTS)))]
    room_size = int(rng.integers(4, 7))

    room_h = n_rows * room_size + n_rows + 1
    room_w = n_cols * room_size + n_cols + 1
    panel_h = 2 * room_size + 3
    panel_w = room_size + 2

    left_margin = int(rng.integers(1, 3))
    gap = int(rng.integers(2, 4))
    right_margin = int(rng.integers(1, 3))
    total_w = left_margin + room_w + gap + panel_w + right_margin
    if total_w > 30:
        return None

    room_top = int(rng.integers(1, 3))
    panel_top = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    total_h = max(room_top + room_h, panel_top + panel_h) + bottom_margin
    if total_h > 30:
        return None

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    wall_rows, wall_cols = _draw_wall_lattice(grid, room_top, left_margin, room_size, n_rows, n_cols)

    colors = [int(v) for v in rng.choice(PALETTE, size=4, replace=False)]
    sample_one = _pattern_one(room_size, colors[0], colors[1], rng)
    sample_two = _pattern_two(room_size, colors[2], colors[3], rng)
    if not _patterns_compatible(sample_one, sample_two):
        return None

    panel_left = left_margin + room_w + gap
    grid[panel_top, panel_left : panel_left + panel_w] = ORANGE
    grid[panel_top + panel_h - 1, panel_left : panel_left + panel_w] = ORANGE
    grid[panel_top : panel_top + panel_h, panel_left] = ORANGE
    grid[panel_top : panel_top + panel_h, panel_left + panel_w - 1] = ORANGE
    grid[panel_top + 1 : panel_top + 1 + room_size, panel_left + 1 : panel_left + 1 + room_size] = sample_one
    grid[
        panel_top + room_size + 2 : panel_top + 2 * room_size + 2,
        panel_left + 1 : panel_left + 1 + room_size,
    ] = sample_two

    rooms = [
        (wall_rows[ri], wall_cols[ci], wall_rows[ri + 1], wall_cols[ci + 1])
        for ri in range(n_rows)
        for ci in range(n_cols)
    ]
    if len(rooms) == 2:
        door_counts = [1, 2]
        rng.shuffle(door_counts)
    else:
        door_counts = [1, 1, 2, 2]
        rng.shuffle(door_counts)

    room_specs: list[dict[str, object]] = []
    primary_sides: list[str] = []
    for idx, room in enumerate(rooms):
        ri = idx // n_cols
        ci = idx % n_cols
        sides = _available_sides(ri, ci, n_rows, n_cols)
        doors = _choose_doors(room, sides, int(door_counts[idx]), rng)
        for door in doors:
            rr, cc = door["pos"]  # type: ignore[assignment]
            grid[rr, cc] = BLACK
        primary = _primary_side(doors)
        primary_sides.append(primary)
        room_specs.append({"room": room, "doors": doors, "door_count": int(door_counts[idx])})

    if not any(side in {"left", "right"} for side in primary_sides):
        return None
    if not any(side in {"top", "bottom"} for side in primary_sides):
        return None

    samples = [sample_one, sample_two]
    expected = _render_expected(grid, room_specs, samples)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
