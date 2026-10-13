from __future__ import annotations

import numpy as np


FRAME_COLOR = 4
MASK_COLOR = 9
PALETTE = (0, 1, 2)
FOREGROUND = (1, 2)
GEN_SIZES = (5, 7)
GEN_MAX_ATTEMPTS = 500


def _symmetry_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, r),
                (r, last - c),
                (last - r, c),
                (last - r, last - c),
                (last - c, last - r),
                (c, last - r),
                (last - c, r),
            }
        )
    )


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            orbit = _symmetry_orbit(n, r, c)
            if orbit not in seen:
                seen.add(orbit)
                out.append(orbit)
    return out


def _special_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    return [orbit for orbit in _all_orbits(n) if len(orbit) in (2, 4)]


def _standard_complete(square: np.ndarray) -> np.ndarray:
    n, m = square.shape
    assert n == m, "symmetry completion expects a square"
    out = np.asarray(square, dtype=int).copy()
    for orbit in _all_orbits(n):
        visible = {int(out[r, c]) for r, c in orbit if int(out[r, c]) != MASK_COLOR}
        assert len(visible) == 1, "each orbit must expose exactly one visible color"
        fill = next(iter(visible))
        for r, c in orbit:
            if int(out[r, c]) == MASK_COLOR:
                out[r, c] = fill
    return out


def _apply_extra_rule(
    square: np.ndarray,
    orbit: tuple[tuple[int, int], ...],
    src_color: int,
    dst_color: int,
) -> np.ndarray:
    out = np.asarray(square, dtype=int).copy()
    values = {int(out[r, c]) for r, c in orbit}
    assert values == {src_color}, "target orbit must be uniformly recolorable"
    for r, c in orbit:
        out[r, c] = dst_color
    return out


def _infer_extra_rule(
    train_masked: np.ndarray, train_answer: np.ndarray
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    standard = _standard_complete(train_masked)
    diff_coords = np.argwhere(standard != train_answer)
    assert 2 <= len(diff_coords) <= 4, "extra rule should affect one small special orbit"
    orbit = _symmetry_orbit(standard.shape[0], int(diff_coords[0, 0]), int(diff_coords[0, 1]))
    assert set(map(tuple, diff_coords.tolist())) == set(orbit), "differences must be exactly one orbit"
    before = {int(standard[r, c]) for r, c in orbit}
    after = {int(train_answer[r, c]) for r, c in orbit}
    assert len(before) == 1 and len(after) == 1 and before != after, "expected a uniform recolor"
    return orbit, next(iter(before)), next(iter(after))


def _extract_regions(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == FRAME_COLOR)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == FRAME_COLOR)]

    if len(full_rows) == 2 and len(full_cols) == 4:
        top, bottom = full_rows
        assert bottom - top >= 3, "interior must be non-empty"
        n = bottom - top - 1
        regions = []
        for i in range(3):
            c0 = full_cols[i] + 1
            c1 = full_cols[i + 1]
            assert c1 - c0 == n, "all framed regions must be square and same size"
            regions.append(g[top + 1 : bottom, c0:c1].copy())
        return regions

    if len(full_rows) == 4 and len(full_cols) == 2:
        left, right = full_cols
        assert right - left >= 3, "interior must be non-empty"
        n = right - left - 1
        regions = []
        for i in range(3):
            r0 = full_rows[i] + 1
            r1 = full_rows[i + 1]
            assert r1 - r0 == n, "all framed regions must be square and same size"
            regions.append(g[r0:r1, left + 1 : right].copy())
        return regions

    raise AssertionError("expected one yellow strip of three framed squares")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a non-standard symmetry-completion rule from a training pair, then apply it to the test square.

    Input structure:
    - A yellow frame strip partitions the canvas into three adjacent equal odd-sized squares.
    - The first square is a masked blue/red/black pattern with maroon cells hiding part of an otherwise eight-way symmetric design.
    - The second square is the completed training answer.
    - The third square is another masked square built from the same symmetry primitive.

    Rule:
    - Extract the three framed squares from the yellow separators.
    - On the training input, perform standard D4 orbit completion: every maroon cell is replaced by the unique visible color in its eightfold symmetry orbit.
    - Compare that standard completion to the training answer and infer the extra rule: one whole special orbit of size 2 or 4 is recolored uniformly from one foreground color to the other.
    - Apply the same two-step rule to the test square and return only that completed test square.

    Key insight:
    - The answer is not plain symmetry restoration; the training pair teaches an additional orbit-level recolor that must be applied after completion.
    """
    train_masked, train_answer, test_masked = _extract_regions(np.asarray(grid, dtype=int))
    orbit, src_color, dst_color = _infer_extra_rule(train_masked, train_answer)
    return _apply_extra_rule(_standard_complete(test_masked), orbit, src_color, dst_color)


def _paint_orbits(
    n: int,
    orbit_colors: dict[tuple[tuple[int, int], ...], int],
) -> np.ndarray:
    square = np.zeros((n, n), dtype=int)
    for orbit, color in orbit_colors.items():
        for r, c in orbit:
            square[r, c] = int(color)
    return square


def _build_base_square(
    n: int,
    target_orbit: tuple[tuple[int, int], ...],
    src_color: int,
    rng: np.random.Generator,
) -> np.ndarray:
    orbits = _all_orbits(n)
    orbit_colors: dict[tuple[tuple[int, int], ...], int] = {}
    weighted_palette = np.array([0, 0, 0, 1, 1, 2, 2], dtype=int)
    for orbit in orbits:
        if orbit == target_orbit:
            orbit_colors[orbit] = int(src_color)
        else:
            orbit_colors[orbit] = int(weighted_palette[int(rng.integers(len(weighted_palette)))])

    non_target = [orbit for orbit in orbits if orbit != target_orbit]
    rng.shuffle(non_target)
    orbit_colors[non_target[0]] = 1
    orbit_colors[non_target[1]] = 2
    orbit_colors[non_target[2]] = src_color

    square = _paint_orbits(n, orbit_colors)
    nonzero = int(np.count_nonzero(square))
    assert nonzero >= len(target_orbit) + 4
    assert set(np.unique(square)).issuperset({0, 1, 2}) or set(np.unique(square)) == {1, 2}
    return square


def _make_masked_square(
    base: np.ndarray,
    target_orbit: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    *,
    min_total_masks: int,
    min_outside_target: int,
) -> np.ndarray:
    n = base.shape[0]
    masked = base.copy()
    orbits = _all_orbits(n)
    orbit_counts = {orbit: 0 for orbit in orbits}
    masked_cells: set[tuple[int, int]] = set()
    target_set = set(target_orbit)

    target_cells = list(target_orbit)
    rng.shuffle(target_cells)
    required_target = int(rng.integers(1, len(target_orbit)))
    for cell in target_cells[:required_target]:
        masked_cells.add(cell)
        orbit_counts[target_orbit] += 1

    target_total = int(rng.integers(min_total_masks, min_total_masks + 3))
    while True:
        outside_target = sum(1 for cell in masked_cells if cell not in target_set)
        if len(masked_cells) >= target_total and outside_target >= min_outside_target:
            break
        if outside_target < min_outside_target:
            candidate_orbits = [orbit for orbit in orbits if orbit != target_orbit]
        else:
            candidate_orbits = orbits
        orbit = candidate_orbits[int(rng.integers(len(candidate_orbits)))]
        cap = len(orbit) - 1
        if cap <= 0 or orbit_counts[orbit] >= cap:
            continue
        choices = [cell for cell in orbit if cell not in masked_cells]
        if not choices:
            continue
        cell = choices[int(rng.integers(len(choices)))]
        masked_cells.add(cell)
        orbit_counts[orbit] += 1

    outside_target = sum(1 for cell in masked_cells if cell not in target_set)
    assert outside_target >= min_outside_target, "mask should hide more than the recolored orbit"
    for orbit in orbits:
        assert orbit_counts[orbit] <= len(orbit) - 1, "every orbit needs a visible witness"
    for r, c in masked_cells:
        masked[r, c] = MASK_COLOR
    return masked


def _assemble_strip(
    train_masked: np.ndarray,
    train_answer: np.ndarray,
    test_masked: np.ndarray,
    orientation: str,
) -> np.ndarray:
    n = train_masked.shape[0]
    if orientation == "horizontal":
        grid = np.full((n + 2, 3 * n + 4), FRAME_COLOR, dtype=int)
        panels = (train_masked, train_answer, test_masked)
        for idx, panel in enumerate(panels):
            c0 = 1 + idx * (n + 1)
            grid[1 : n + 1, c0 : c0 + n] = panel
        return grid

    grid = np.full((3 * n + 4, n + 2), FRAME_COLOR, dtype=int)
    panels = (train_masked, train_answer, test_masked)
    for idx, panel in enumerate(panels):
        r0 = 1 + idx * (n + 1)
        grid[r0 : r0 + n, 1 : n + 1] = panel
    return grid


def _semantic_checks(
    train_masked: np.ndarray,
    train_answer: np.ndarray,
    test_masked: np.ndarray,
    target_orbit: tuple[tuple[int, int], ...],
    src_color: int,
    dst_color: int,
    expected: np.ndarray,
) -> None:
    train_standard = _standard_complete(train_masked)
    test_standard = _standard_complete(test_masked)
    inferred_orbit, inferred_src, inferred_dst = _infer_extra_rule(train_masked, train_answer)
    assert inferred_orbit == target_orbit
    assert inferred_src == src_color and inferred_dst == dst_color
    assert np.array_equal(_apply_extra_rule(train_standard, target_orbit, src_color, dst_color), train_answer)
    assert np.array_equal(_apply_extra_rule(test_standard, target_orbit, src_color, dst_color), expected)
    assert np.any(test_masked == MASK_COLOR), "test square must actually contain masked cells"
    assert any(int(test_masked[r, c]) == MASK_COLOR for r, c in target_orbit), "target orbit must be hidden in the test square"
    assert np.any(test_standard != expected), "extra recolor must affect the test output"
    assert np.any(train_standard != train_answer), "training answer must differ from plain symmetry completion"
    assert set(map(int, np.unique(train_masked))) & {1, 2}, "training input needs visible foreground clues"
    assert set(map(int, np.unique(test_masked))) & {1, 2}, "test input needs visible foreground clues"


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rng = np.random.default_rng()
        n = int(GEN_SIZES[int(rng.integers(len(GEN_SIZES)))])
        target_orbit = _special_orbits(n)[int(rng.integers(len(_special_orbits(n))))]
        src_color = int(FOREGROUND[int(rng.integers(len(FOREGROUND)))])
        dst_color = 1 if src_color == 2 else 2

        train_base = _build_base_square(n, target_orbit, src_color, rng)
        test_base = _build_base_square(n, target_orbit, src_color, rng)
        if np.array_equal(train_base, test_base):
            continue

        train_answer = _apply_extra_rule(train_base, target_orbit, src_color, dst_color)
        test_answer = _apply_extra_rule(test_base, target_orbit, src_color, dst_color)
        if np.array_equal(train_answer, test_answer):
            continue

        train_masked = _make_masked_square(
            train_base,
            target_orbit,
            rng,
            min_total_masks=4 if n == 5 else 6,
            min_outside_target=2,
        )
        test_masked = _make_masked_square(
            test_base,
            target_orbit,
            rng,
            min_total_masks=4 if n == 5 else 6,
            min_outside_target=2,
        )

        orientation = "horizontal" if int(rng.integers(2)) == 0 else "vertical"
        grid = _assemble_strip(train_masked, train_answer, test_masked, orientation)

        try:
            _semantic_checks(
                train_masked,
                train_answer,
                test_masked,
                target_orbit,
                src_color,
                dst_color,
                test_answer,
            )
            if not np.array_equal(main(grid), test_answer):
                continue
            return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid analogy symmetry puzzle")
