from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

MIN_HALF_SIZE = 9
MAX_HALF_SIZE = 13
MIN_RED_COUNT = 8
MAX_RED_COUNT = 15
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [c for c in range(w) if np.all(grid[:, c] == BLUE)]
    assert len(candidates) == 1, "expected exactly one full-height blue separator"
    return int(candidates[0])


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_bar = _separator_col(grid)
    left = np.asarray(grid[:, :c_bar], dtype=int)
    right = np.asarray(grid[:, c_bar + 1 :], dtype=int)
    assert left.shape == right.shape, "left and right halves must align"
    return left, right


def _seed_coord(right: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(right == GREEN)
    assert coords.shape[0] == 1, "expected exactly one green seed"
    return tuple(map(int, coords[0]))


def _spiral_draw(
    canvas: np.ndarray, seed: tuple[int, int]
) -> dict[str, object]:
    h, w = canvas.shape
    r, c = map(int, seed)
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    leg_len = 2
    dir_idx = 0
    max_attempts = 20 * h * w

    path_cells: list[tuple[int, int]] = []
    oob_count = 0

    while max_attempts > 0:
        for _ in range(2):
            dr, dc = dirs[dir_idx]
            for _step in range(leg_len):
                max_attempts -= 1
                nr, nc = r + dr, c + dc
                if not _in_bounds(canvas.shape, nr, nc):
                    oob_count += 1
                    continue

                value = int(canvas[nr, nc])
                if value == BLACK:
                    canvas[nr, nc] = GREEN
                    r, c = nr, nc
                    path_cells.append((r, c))
                    continue

                return {
                    "path_cells": path_cells,
                    "stopper": (int(nr), int(nc)),
                    "stop_value": value,
                    "oob_count": oob_count,
                }
            dir_idx = (dir_idx + 1) % 4
        leg_len += 2

    raise AssertionError("spiral did not reach a blocker within the safety bound")


def _blank_spiral_events(
    shape: tuple[int, int], seed: tuple[int, int], max_attempts: int = 240
) -> list[tuple[str, tuple[int, int]]]:
    r, c = map(int, seed)
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    leg_len = 2
    dir_idx = 0
    events: list[tuple[str, tuple[int, int]]] = []

    while len(events) < max_attempts:
        for _ in range(2):
            dr, dc = dirs[dir_idx]
            for _step in range(leg_len):
                nr, nc = r + dr, c + dc
                if not _in_bounds(shape, nr, nc):
                    events.append(("oob", (int(nr), int(nc))))
                    if len(events) >= max_attempts:
                        return events
                    continue

                r, c = nr, nc
                events.append(("draw", (int(r), int(c))))
                if len(events) >= max_attempts:
                    return events
            dir_idx = (dir_idx + 1) % 4
        leg_len += 2

    return events


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _expanded_zone(cells: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    zone.add((int(rr), int(cc)))
    return zone


def _expected_output(
    n: int,
    left_reds: set[tuple[int, int]],
    right_reds: set[tuple[int, int]],
    seed: tuple[int, int],
    path_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((n, n), BLACK, dtype=int)
    for r, c in right_reds:
        out[r, c] = RED
    out[seed] = GREEN

    for r in range(n):
        for c in range(n):
            if (r, c) not in left_reds and (r, c) not in right_reds:
                out[r, c] = ORANGE

    out[seed] = GREEN
    for rc in path_cells:
        out[rc] = GREEN
    return out


def _oob_before_draw_index(
    events: list[tuple[str, tuple[int, int]]], draw_index: int
) -> int:
    draw_count = 0
    oob_count = 0
    for kind, _pos in events:
        if kind == "oob":
            oob_count += 1
            continue
        if draw_count == draw_index:
            return oob_count
        draw_count += 1
    raise AssertionError("draw index exceeded the sampled spiral trace")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a NOR obstacle field from the two red masks, then stop a green spiral on the first blocker.

    Input structure:
    - A full-height blue column splits the grid into equal left and right halves.
    - Each half is a black background with sparse red pixels; the right half also has exactly one green seed.
    - The two halves are aligned cell-by-cell and act as binary masks where red means active.

    Rule:
    - Compute the pixelwise NOR of the two masks: a cell becomes an orange obstacle exactly when neither half has red there.
    - Discard the left half and the blue separator, keeping only the right-half canvas.
    - Preserve the right-half red pixels and the green seed, add the orange NOR obstacles, then grow a spiral from the seed with leg lengths 2,2,4,4,6,6,... and directions up, right, down, left.
    - On each spiral step, paint the next in-bounds black cell green; if the next cell is red or orange, stop without painting it; if the next step is out of bounds, skip it but still consume that step.

    Key insight:
    - The blockers are not drawn directly: they emerge from where both halves are empty, and that NOR pattern alone determines where the spiral is forced to stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    left, right = _split_input(g)
    out = right.copy()

    nor_mask = (left != RED) & (right != RED)
    out[(out == BLACK) & nor_mask] = ORANGE

    seed = _seed_coord(out)
    out[seed] = GREEN
    _spiral_draw(out, seed)
    return out


def _pick_mode() -> str:
    global _GEN_CALL_COUNT
    mode = "boundary" if _GEN_CALL_COUNT % 2 == 0 else "inner"
    _GEN_CALL_COUNT += 1
    return mode


def _pick_seed_and_path(
    n: int, mode: str
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int], int]:
    for _ in range(200):
        if mode == "boundary":
            if random.random() < 0.5:
                seed = (random.randint(2, n - 2), 1)
            else:
                seed = (n - 2, random.randint(1, n - 3))
        else:
            seed = (random.randint(3, n - 4), random.randint(3, n - 4))

        events = _blank_spiral_events((n, n), seed)
        path_prefix: list[tuple[int, int]] = []
        oob_before = 0
        candidates: list[tuple[list[tuple[int, int]], tuple[int, int], int]] = []

        for kind, pos in events:
            if kind == "oob":
                oob_before += 1
                continue

            if len(path_prefix) >= 4:
                candidates.append((path_prefix.copy(), pos, oob_before))
            path_prefix.append(pos)

        if mode == "boundary":
            filtered = [
                item
                for item in candidates
                if 1 <= item[2] <= 4 and 7 <= len(item[0]) <= 12
            ]
        else:
            filtered = [
                item
                for item in candidates
                if item[2] == 0 and 4 <= len(item[0]) <= 8
            ]

        if filtered:
            path_cells, stopper, oob_count = random.choice(filtered)
            return seed, path_cells, stopper, oob_count

    raise RuntimeError(f"failed to sample a valid {mode} spiral path")


def _sample_noncorridor_cells(
    n: int,
    forbidden: set[tuple[int, int]],
    preferred_block: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    all_cells = {(r, c) for r in range(n) for c in range(n)}
    preferred = sorted(all_cells - forbidden - preferred_block)
    if len(preferred) >= count:
        return random.sample(preferred, count)

    fallback = sorted(all_cells - forbidden)
    if len(fallback) >= count:
        return random.sample(fallback, count)
    return None


def _build_attempt(mode: str) -> tuple[np.ndarray, dict[str, object]] | None:
    n = random.randint(MIN_HALF_SIZE, MAX_HALF_SIZE)
    seed, path_cells, stopper, oob_before_stop = _pick_seed_and_path(n, mode)

    required_left = {seed, *path_cells}
    max_extra = min(5, MAX_RED_COUNT - len(required_left))
    min_extra = max(2, MIN_RED_COUNT - len(required_left))
    if min_extra > max_extra:
        return None
    left_extra_count = random.randint(min_extra, max_extra)

    corridor_zone = _expanded_zone(required_left | {stopper}, n)
    left_extra_cells = _sample_noncorridor_cells(
        n=n,
        forbidden=required_left | {stopper},
        preferred_block=corridor_zone,
        count=left_extra_count,
    )
    if left_extra_cells is None:
        return None

    max_overlap = min(3, left_extra_count - 1)
    if max_overlap < 1:
        return None
    overlap_count = random.randint(1, max_overlap)
    overlap_cells = set(random.sample(left_extra_cells, overlap_count))

    left_reds = required_left | set(left_extra_cells)

    right_total = random.randint(MIN_RED_COUNT, MAX_RED_COUNT)
    right_unique_count = right_total - overlap_count

    right_unique_cells = _sample_noncorridor_cells(
        n=n,
        forbidden=required_left | set(left_extra_cells) | {stopper},
        preferred_block=corridor_zone,
        count=right_unique_count,
    )
    if right_unique_cells is None:
        return None

    right_reds = overlap_cells | set(right_unique_cells)
    if seed in right_reds or stopper in right_reds:
        return None

    grid = np.full((n, 2 * n + 1), BLACK, dtype=int)
    grid[:, n] = BLUE
    for r, c in left_reds:
        grid[r, c] = RED
    for r, c in right_reds:
        grid[r, n + 1 + c] = RED
    grid[seed[0], n + 1 + seed[1]] = GREEN

    metadata: dict[str, object] = {
        "n": n,
        "mode": mode,
        "seed": seed,
        "path_cells": path_cells,
        "stopper": stopper,
        "oob_before_stop": oob_before_stop,
        "left_reds": left_reds,
        "right_reds": right_reds,
    }
    return grid, metadata


def _validate_attempt(grid: np.ndarray, metadata: dict[str, object]) -> None:
    n = int(metadata["n"])
    mode = str(metadata["mode"])
    seed = tuple(metadata["seed"])
    stopper = tuple(metadata["stopper"])
    path_cells = [tuple(rc) for rc in metadata["path_cells"]]
    left_reds = set(metadata["left_reds"])
    right_reds = set(metadata["right_reds"])
    oob_before_stop = int(metadata["oob_before_stop"])

    assert grid.shape == (n, 2 * n + 1)
    left, right = _split_input(grid)
    assert left.shape == right.shape == (n, n)

    parsed_left = _coord_set(np.argwhere(left == RED))
    parsed_right = _coord_set(np.argwhere(right == RED))
    assert parsed_left == left_reds
    assert parsed_right == right_reds
    assert _seed_coord(right) == seed

    assert MIN_RED_COUNT <= len(left_reds) <= MAX_RED_COUNT
    assert MIN_RED_COUNT <= len(right_reds) <= MAX_RED_COUNT
    overlap = left_reds & right_reds
    assert 1 <= len(overlap) <= 3
    assert seed in left_reds and seed not in right_reds
    assert stopper not in left_reds and stopper not in right_reds
    assert set(path_cells).issubset(left_reds)
    assert set(path_cells).isdisjoint(right_reds)

    left_only_extras = left_reds - right_reds - {seed} - set(path_cells)
    right_only = right_reds - left_reds
    assert left_only_extras, "need left-only cells so the left mask matters beyond the spiral"
    assert right_only, "need right-only cells so the right mask matters independently"

    nor_mask = (left != RED) & (right != RED)
    assert int(nor_mask.sum()) >= 5
    assert not bool(nor_mask[seed])
    assert bool(nor_mask[stopper])

    events = _blank_spiral_events((n, n), seed)
    draw_cells = [pos for kind, pos in events if kind == "draw"]
    assert [kind for kind, _pos in events[:4]] == ["draw", "draw", "draw", "draw"]
    assert draw_cells[: len(path_cells)] == path_cells
    assert draw_cells[len(path_cells)] == stopper
    assert len(path_cells) >= 4
    assert _oob_before_draw_index(events, len(path_cells)) == oob_before_stop
    if mode == "boundary":
        assert oob_before_stop > 0
    else:
        assert oob_before_stop == 0

    expected = _expected_output(n, left_reds, right_reds, seed, path_cells)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert int(actual[stopper]) == ORANGE
    assert all(int(actual[rc]) == GREEN for rc in path_cells)
    assert all(int(actual[rc]) == BLACK for rc in left_only_extras)
    assert all(int(actual[rc]) == RED for rc in right_only)


def generate_input() -> np.ndarray:
    mode = _pick_mode()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt(mode)
        if built is None:
            continue
        grid, metadata = built
        try:
            _validate_attempt(grid, metadata)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid puzzle input for mode={mode}")
