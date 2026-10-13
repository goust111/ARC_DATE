from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
SEED_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _max_radius(shape: tuple[int, int], r: int, c: int) -> int:
    return min(r - 1, c - 1, shape[0] - 2 - r, shape[1] - 2 - c)


def _signature_cells(r: int, c: int, radius: int) -> list[tuple[int, int]]:
    k = int(radius)
    return [
        (r - k - 1, c - k),
        (r - k, c - k - 1),
        (r - k - 1, c + k),
        (r - k, c + k + 1),
        (r + k + 1, c - k),
        (r + k, c - k - 1),
        (r + k + 1, c + k),
        (r + k, c + k + 1),
    ]


def _border_cells(r: int, c: int, radius: int) -> list[tuple[int, int]]:
    k = int(radius)
    top = [(r - k, cc) for cc in range(c - k, c + k + 1)]
    bottom = [(r + k, cc) for cc in range(c - k, c + k + 1)]
    left = [(rr, c - k) for rr in range(r - k + 1, r + k)]
    right = [(rr, c + k) for rr in range(r - k + 1, r + k)]
    return top + bottom + left + right


def _detect_complete_signatures(
    grid: np.ndarray, *, require_seed: bool = True
) -> list[tuple[int, int, int, int | None]]:
    g = np.asarray(grid, dtype=int)
    hits: list[tuple[int, int, int, int | None]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            center_color = int(g[r, c])
            has_seed = center_color not in (BLACK, RED)
            if require_seed and not has_seed:
                continue
            max_radius = _max_radius(g.shape, r, c)
            for radius in range(1, max_radius + 1):
                if all(int(g[rr, cc]) == RED for rr, cc in _signature_cells(r, c, radius)):
                    hits.append((r, c, radius, center_color if has_seed else None))
    hits.sort(key=lambda item: (item[0], item[1], item[2], -1 if item[3] is None else item[3]))
    return hits


def _paint_square_borders(
    grid: np.ndarray, signatures: list[tuple[int, int, int, int | None]]
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    paint_plan: dict[tuple[int, int], int] = {}
    for r, c, radius, color in signatures:
        assert color is not None, "complete signatures must have a colored center seed"
        for rr, cc in _border_cells(r, c, radius):
            if int(g[rr, cc]) != BLACK:
                continue
            prev = paint_plan.get((rr, cc))
            if prev is not None and prev != color:
                raise AssertionError("conflicting square borders would make the output ambiguous")
            paint_plan[(rr, cc)] = int(color)

    out = g.copy()
    for (rr, cc), color in paint_plan.items():
        out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct every never-drawn square whose eight red corner-outside marks are complete.

    Input structure:
    - A black background contains scattered red single-pixel corner marks and a few non-red seed pixels.
    - Each valid target square has odd side length, one colored seed at its center, and exactly eight red marks:
      for every corner, the two orthogonally adjacent outside cells are red.
    - Additional incomplete red mark groups may look similar and may also contain center seeds.

    Rule:
    - Search for every odd square whose eight outside-corner mark positions are all red.
    - If that square's center contains a non-black, non-red seed, draw the square border in the seed color.
    - Keep every original input pixel unchanged; incomplete red groups add nothing.

    Key insight:
    - The eight red marks alone determine the missing square geometry, while the center seed supplies only the border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    signatures = _detect_complete_signatures(g, require_seed=True)
    assert signatures, "expected at least one complete corner signature"
    return _paint_square_borders(g, signatures)


def _reserve_box(r: int, c: int, radius: int, *, moat: int = 1) -> tuple[int, int, int, int]:
    k = int(radius)
    return (r - k - 1 - moat, c - k - 1 - moat, r + k + 1 + moat, c + k + 1 + moat)


def _available_centers(reserved: np.ndarray, radius: int) -> list[tuple[int, int]]:
    h, w = reserved.shape
    k = int(radius)
    centers: list[tuple[int, int]] = []
    for r in range(k + 2, h - k - 2):
        for c in range(k + 2, w - k - 2):
            r0, c0, r1, c1 = _reserve_box(r, c, k, moat=1)
            if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                continue
            if not reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                centers.append((r, c))
    return centers


def _place_group(reserved: np.ndarray) -> tuple[int, int, int] | None:
    h, w = reserved.shape
    max_radius = min(5, (h - 5) // 2, (w - 5) // 2)
    if max_radius < 1:
        return None
    weighted_radii = [1, 2, 2, 3, 3, 4, 4, 5]
    weighted_radii = [radius for radius in weighted_radii if radius <= max_radius]
    random.shuffle(weighted_radii)
    for radius in weighted_radii:
        centers = _available_centers(reserved, radius)
        if not centers:
            continue
        r, c = random.choice(centers)
        r0, c0, r1, c1 = _reserve_box(r, c, radius, moat=1)
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        return r, c, radius
    return None


def _draw_complete_group(
    grid: np.ndarray, center_r: int, center_c: int, radius: int, color: int
) -> tuple[int, int, int, int]:
    for rr, cc in _signature_cells(center_r, center_c, radius):
        grid[rr, cc] = RED
    grid[center_r, center_c] = int(color)
    return center_r, center_c, radius, int(color)


def _draw_distractor(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    radius: int,
    *,
    seed_color: int | None,
    missing_count: int,
) -> dict:
    all_marks = _signature_cells(center_r, center_c, radius)
    missing = set(random.sample(all_marks, missing_count))
    kept_marks = [coord for coord in all_marks if coord not in missing]
    for rr, cc in kept_marks:
        grid[rr, cc] = RED
    if seed_color is not None:
        grid[center_r, center_c] = int(seed_color)
    return {
        "center": (center_r, center_c),
        "radius": int(radius),
        "seed_color": seed_color,
        "kept_marks": tuple(sorted(kept_marks)),
        "missing_count": int(missing_count),
    }


def _expected_output(
    grid: np.ndarray, complete: list[tuple[int, int, int, int]]
) -> np.ndarray:
    return _paint_square_borders(grid, complete)


def _validate_generated(
    grid: np.ndarray,
    complete: list[tuple[int, int, int, int]],
    distractors: list[dict],
) -> bool:
    if not complete or not distractors:
        return False
    if not any(d["seed_color"] is not None and d["missing_count"] <= 2 for d in distractors):
        return False

    for r, c, radius, color in complete:
        if int(grid[r, c]) != color:
            return False
        if any(int(grid[rr, cc]) != RED for rr, cc in _signature_cells(r, c, radius)):
            return False
        if any(int(grid[rr, cc]) != BLACK for rr, cc in _border_cells(r, c, radius)):
            return False

    raw_hits = {(r, c, radius) for r, c, radius, _color in _detect_complete_signatures(grid, require_seed=False)}
    expected_hits = {(r, c, radius) for r, c, radius, _color in complete}
    if raw_hits != expected_hits:
        return False

    detected = _detect_complete_signatures(grid, require_seed=True)
    if set(detected) != set(complete):
        return False

    expected = _expected_output(grid, complete)
    actual = main(grid)
    if np.array_equal(actual, grid) or not np.array_equal(actual, expected):
        return False

    for cell in np.argwhere((actual != grid) & (grid != BLACK)):
        if int(grid[tuple(cell)]) != int(actual[tuple(cell)]):
            return False

    return True


def _gen_attempt() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[dict]] | None:
    height = random.randint(15, 30)
    width = random.randint(15, 30)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    complete_count = random.choices([1, 2, 3], weights=[3, 4, 2])[0]
    distractor_count = random.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]

    complete_colors = random.sample(SEED_COLORS, k=complete_count)
    complete: list[tuple[int, int, int, int]] = []
    for color in complete_colors:
        placement = _place_group(reserved)
        if placement is None:
            return None
        center_r, center_c, radius = placement
        complete.append(_draw_complete_group(grid, center_r, center_c, radius, color))

    distractors: list[dict] = []
    for index in range(distractor_count):
        placement = _place_group(reserved)
        if placement is None:
            return None
        center_r, center_c, radius = placement
        if index == 0:
            seed_color = random.choice(SEED_COLORS)
            missing_count = random.choice([1, 2])
        elif index == 1:
            seed_color = None
            missing_count = random.choice([1, 2, 3])
        else:
            seed_color = random.choice(SEED_COLORS) if random.random() < 0.6 else None
            missing_count = random.choice([1, 2, 3, 4])
        distractors.append(
            _draw_distractor(
                grid,
                center_r,
                center_c,
                radius,
                seed_color=seed_color,
                missing_count=missing_count,
            )
        )

    if not _validate_generated(grid, complete, distractors):
        return None
    return grid, complete, distractors


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            grid, _complete, _distractors = result
            return grid
    raise RuntimeError(f"failed to generate a valid ghost-square input after {GEN_ATTEMPTS} attempts")
