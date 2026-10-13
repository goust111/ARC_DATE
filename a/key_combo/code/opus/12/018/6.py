from __future__ import annotations

from functools import lru_cache

import numpy as np


BACKGROUND = 0
AXIS_COLOR = 5
DOT_COLORS = (1, 2, 3)
SPACING = 3
MAX_GENERATION_ATTEMPTS = 400


def _find_axis(grid: np.ndarray) -> int:
    axis_cols = np.where(np.all(grid == AXIS_COLOR, axis=0))[0]
    assert len(axis_cols) == 1, "expected exactly one grey mirror axis"
    return int(axis_cols[0])


def _detect_lattice_offsets(grid: np.ndarray, axis: int) -> tuple[int, int]:
    left = np.asarray(grid[:, :axis], dtype=int)
    rows, cols = np.where(left != BACKGROUND)
    assert len(rows) > 0, "expected colored dots on the left half"
    row_offset = int(np.bincount(rows % SPACING, minlength=SPACING).argmax())
    col_offset = int(np.bincount(cols % SPACING, minlength=SPACING).argmax())
    return row_offset, col_offset


def _extract_logical(grid: np.ndarray) -> tuple[int, list[int], list[int], np.ndarray]:
    axis = _find_axis(grid)
    row_offset, col_offset = _detect_lattice_offsets(grid, axis)
    row_positions = list(range(row_offset, grid.shape[0], SPACING))
    col_positions = list(range(col_offset, axis, SPACING))
    logical = np.zeros((len(row_positions), len(col_positions)), dtype=int)
    row_to_i = {r: i for i, r in enumerate(row_positions)}
    col_to_j = {c: j for j, c in enumerate(col_positions)}

    for r, c in np.argwhere(grid[:, :axis] != BACKGROUND):
        rr = int(r)
        cc = int(c)
        if rr in row_to_i and cc in col_to_j:
            logical[row_to_i[rr], col_to_j[cc]] = int(grid[rr, cc])

    return axis, row_positions, col_positions, logical


def _horizontal_symmetric(logical: np.ndarray, r: int, c: int) -> bool:
    rows, cols = logical.shape
    r0 = max(0, r - 1)
    r1 = min(rows - 1, r + 1)
    c0 = max(0, c - 1)
    c1 = min(cols - 1, c + 2)
    mirror_sum = 2 * c + 1

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            mc = mirror_sum - cc
            if not (c0 <= mc <= c1):
                return False
            if logical[rr, cc] != logical[rr, mc]:
                return False
    return True


def _vertical_symmetric(logical: np.ndarray, r: int, c: int) -> bool:
    rows, cols = logical.shape
    r0 = max(0, r - 1)
    r1 = min(rows - 1, r + 2)
    c0 = max(0, c - 1)
    c1 = min(cols - 1, c + 1)
    mirror_sum = 2 * r + 1

    for rr in range(r0, r1 + 1):
        mr = mirror_sum - rr
        if not (r0 <= mr <= r1):
            return False
        for cc in range(c0, c1 + 1):
            if logical[rr, cc] != logical[mr, cc]:
                return False
    return True


def _pair_stats(logical: np.ndarray) -> list[dict[str, int | str | bool]]:
    stats: list[dict[str, int | str | bool]] = []
    rows, cols = logical.shape

    for r in range(rows):
        for c in range(cols - 1):
            color = int(logical[r, c])
            if color != BACKGROUND and int(logical[r, c + 1]) == color:
                stats.append(
                    {
                        "orientation": "h",
                        "r": r,
                        "c": c,
                        "color": color,
                        "accepted": _horizontal_symmetric(logical, r, c),
                    }
                )

    for r in range(rows - 1):
        for c in range(cols):
            color = int(logical[r, c])
            if color != BACKGROUND and int(logical[r + 1, c]) == color:
                stats.append(
                    {
                        "orientation": "v",
                        "r": r,
                        "c": c,
                        "color": color,
                        "accepted": _vertical_symmetric(logical, r, c),
                    }
                )

    return stats


def _render_output_from_logical(
    logical: np.ndarray,
    row_positions: list[int],
    col_positions: list[int],
    axis: int,
    shape: tuple[int, int],
    *,
    gate: bool,
) -> np.ndarray:
    left = np.zeros((shape[0], axis), dtype=int)

    for r, phys_r in enumerate(row_positions):
        for c, phys_c in enumerate(col_positions):
            color = int(logical[r, c])
            if color != BACKGROUND:
                left[phys_r, phys_c] = color

    for pair in _pair_stats(logical):
        if gate and not bool(pair["accepted"]):
            continue
        color = int(pair["color"])
        r = int(pair["r"])
        c = int(pair["c"])
        if pair["orientation"] == "h":
            phys_r = row_positions[r]
            start = col_positions[c] + 1
            stop = col_positions[c + 1]
            left[phys_r, start:stop] = color
        else:
            phys_c = col_positions[c]
            start = row_positions[r] + 1
            stop = row_positions[r + 1]
            left[start:stop, phys_c] = color

    out = np.zeros(shape, dtype=int)
    out[:, :axis] = left
    out[:, axis] = AXIS_COLOR
    out[:, axis + 1 :] = np.fliplr(left)
    return out


def _render_point_mirror_only(
    logical: np.ndarray,
    row_positions: list[int],
    col_positions: list[int],
    axis: int,
    shape: tuple[int, int],
) -> np.ndarray:
    left = np.zeros((shape[0], axis), dtype=int)
    for r, phys_r in enumerate(row_positions):
        for c, phys_c in enumerate(col_positions):
            color = int(logical[r, c])
            if color != BACKGROUND:
                left[phys_r, phys_c] = color

    out = np.zeros(shape, dtype=int)
    out[:, :axis] = left
    out[:, axis] = AXIS_COLOR
    out[:, axis + 1 :] = np.fliplr(left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gate same-color lattice connections by local symmetry, then mirror the completed left half across a central axis.

    Input structure:
    - The canvas has a black background and an odd width with one full grey central column as the mirror axis.
    - Only the left half contains colored single-pixel dots, placed on a regular lattice with 3-cell spacing.
    - Right-half cells are blank, and dots may form horizontal or vertical same-color lattice-adjacent pairs.

    Rule:
    - For each horizontal same-color adjacent pair on the left lattice, inspect the pair's 1-step expanded local lattice window and keep the pair only if that window is symmetric about the pair's vertical bisector.
    - For each vertical same-color adjacent pair, perform the analogous symmetry test about the horizontal bisector.
    - Draw a 2-pixel connector between every pair that passes its local symmetry test.
    - Mirror the entire completed left half, including both dots and connectors, into the blank right half while preserving the grey axis.

    Key insight:
    - The task is not plain same-color connection: adjacency is filtered by a local reflection test, and only that gated structure is globally mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    axis, row_positions, col_positions, logical = _extract_logical(g)
    return _render_output_from_logical(
        logical,
        row_positions,
        col_positions,
        axis,
        g.shape,
        gate=True,
    )


def _seed_window(kind: str, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = anchor
    if kind == "h":
        return {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 3)}
    return {(rr, cc) for rr in range(r - 1, r + 3) for cc in range(c - 1, c + 2)}


def _rv_noise_candidates(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = anchor
    window = _seed_window("v", anchor)
    pair = {(r, c), (r + 1, c)}
    options = []
    for rr, cc in sorted(window):
        if (rr, cc) in pair:
            continue
        mirror = (2 * r + 1 - rr, cc)
        if mirror != (rr, cc):
            options.append((rr, cc))
    return options


@lru_cache(maxsize=None)
def _seed_combos(rows: int, cols: int) -> tuple[tuple[dict[str, object], ...], ...]:
    combos: list[tuple[dict[str, object], ...]] = []
    horizontal_seeds = []
    vertical_seeds = []
    rejected_vertical_seeds = []

    for r in range(1, rows - 1):
        for c in range(1, cols - 2):
            anchor = (r, c)
            horizontal_seeds.append(
                {
                    "kind": "ah",
                    "anchor": anchor,
                    "window": _seed_window("h", anchor),
                    "nonzero": {(r, c), (r, c + 1)},
                }
            )

    for r in range(1, rows - 2):
        for c in range(1, cols - 1):
            anchor = (r, c)
            vertical_seeds.append(
                {
                    "kind": "av",
                    "anchor": anchor,
                    "window": _seed_window("v", anchor),
                    "nonzero": {(r, c), (r + 1, c)},
                }
            )
            for noise in _rv_noise_candidates(anchor):
                rejected_vertical_seeds.append(
                    {
                        "kind": "rv",
                        "anchor": anchor,
                        "noise": noise,
                        "window": _seed_window("v", anchor),
                        "nonzero": {(r, c), (r + 1, c), noise},
                    }
                )

    def compatible(a: dict[str, object], b: dict[str, object]) -> bool:
        return set(a["nonzero"]).isdisjoint(set(b["window"])) and set(b["nonzero"]).isdisjoint(
            set(a["window"])
        )

    for h_seed in horizontal_seeds:
        for v_seed in vertical_seeds:
            if not compatible(h_seed, v_seed):
                continue
            for rv_seed in rejected_vertical_seeds:
                if compatible(h_seed, rv_seed) and compatible(v_seed, rv_seed):
                    combos.append((h_seed, v_seed, rv_seed))

    return tuple(combos)


def _apply_seed(logical: np.ndarray, seed: dict[str, object], palette: tuple[int, ...]) -> None:
    kind = str(seed["kind"])
    r, c = tuple(seed["anchor"])
    pair_color = int(np.random.choice(palette))

    if kind == "ah":
        logical[r, c] = pair_color
        logical[r, c + 1] = pair_color
        return

    if kind == "av":
        logical[r, c] = pair_color
        logical[r + 1, c] = pair_color
        return

    noise_choices = [color for color in palette if color != pair_color]
    noise_color = int(np.random.choice(noise_choices if noise_choices else palette))
    nr, nc = tuple(seed["noise"])
    logical[r, c] = pair_color
    logical[r + 1, c] = pair_color
    logical[nr, nc] = noise_color


def _render_input(
    logical: np.ndarray,
    row_offset: int,
    col_offset: int,
    bottom_margin: int,
) -> np.ndarray:
    rows, cols = logical.shape
    axis = col_offset + SPACING * (cols - 1) + 1
    height = row_offset + SPACING * (rows - 1) + 1 + bottom_margin
    width = 2 * axis + 1

    grid = np.zeros((height, width), dtype=int)
    grid[:, axis] = AXIS_COLOR
    for r in range(rows):
        for c in range(cols):
            color = int(logical[r, c])
            if color != BACKGROUND:
                grid[row_offset + SPACING * r, col_offset + SPACING * c] = color
    return grid


def _build_logical() -> tuple[np.ndarray, int, int, int]:
    rows = int(np.random.choice([6, 7]))
    cols = 5
    palette_size = int(np.random.choice([2, 3]))
    palette = tuple(int(v) for v in np.random.choice(DOT_COLORS, size=palette_size, replace=False))
    combo = _seed_combos(rows, cols)
    if not combo:
        raise RuntimeError("no compatible seed layouts for logical grid size")

    logical = np.zeros((rows, cols), dtype=int)
    seeds = combo[int(np.random.randint(len(combo)))]
    protected = set()
    for seed in seeds:
        _apply_seed(logical, seed, palette)
        protected.update(set(seed["window"]))

    target_points = int(np.random.randint(7, 11))
    empty = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if logical[r, c] == BACKGROUND and (r, c) not in protected
    ]
    np.random.shuffle(empty)

    for color in palette:
        if not np.any(logical == color):
            if not empty:
                break
            r, c = empty.pop()
            logical[r, c] = int(color)

    current_points = int(np.count_nonzero(logical))
    extras_needed = max(0, target_points - current_points)
    for _ in range(extras_needed):
        if not empty:
            break
        r, c = empty.pop()
        logical[r, c] = int(np.random.choice(palette))

    row_offset = int(np.random.choice([0, 1]))
    col_offset = int(np.random.choice([0, 1]))
    bottom_margin = int(np.random.choice([0, 1]))
    return logical, row_offset, col_offset, bottom_margin


def _validate_case(grid: np.ndarray, logical: np.ndarray) -> bool:
    point_count = int(np.count_nonzero(logical))
    if not (6 <= point_count <= 10):
        return False

    used_colors = sorted(int(v) for v in np.unique(logical) if int(v) != BACKGROUND)
    if not (2 <= len(used_colors) <= 3):
        return False

    stats = _pair_stats(logical)
    accepted_h = sum(pair["accepted"] and pair["orientation"] == "h" for pair in stats)
    accepted_v = sum(pair["accepted"] and pair["orientation"] == "v" for pair in stats)
    rejected_total = sum((not pair["accepted"]) for pair in stats)
    if accepted_h < 1 or accepted_v < 1 or accepted_h + accepted_v < 2 or rejected_total < 1:
        return False

    output = main(grid)
    axis, row_positions, col_positions, extracted = _extract_logical(grid)
    if not np.array_equal(extracted, logical):
        return False
    if not np.array_equal(output[:, :axis], np.fliplr(output[:, axis + 1 :])):
        return False
    if not np.any(output[:, axis + 1 :] != BACKGROUND):
        return False
    if np.count_nonzero(output[:, :axis]) <= np.count_nonzero(grid[:, :axis]):
        return False

    ungated = _render_output_from_logical(
        logical,
        row_positions,
        col_positions,
        axis,
        grid.shape,
        gate=False,
    )
    if np.array_equal(output, ungated):
        return False

    point_only = _render_point_mirror_only(
        logical,
        row_positions,
        col_positions,
        axis,
        grid.shape,
    )
    if np.array_equal(output, point_only):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        logical, row_offset, col_offset, bottom_margin = _build_logical()
        grid = _render_input(logical, row_offset, col_offset, bottom_margin)
        if max(grid.shape) > 30:
            continue
        if _validate_case(grid, logical):
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
