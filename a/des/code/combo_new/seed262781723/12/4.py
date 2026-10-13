from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 400
GRID_MIN = 12
GRID_MAX = 28
SEED_COUNT_CHOICES = (1, 2)
RECT_GAP = 1


def _separator_axes(g: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == color)]
        if full_rows and full_cols:
            if len(full_rows) != 1 or len(full_cols) != 1:
                raise AssertionError("expected exactly one full row and one full column for the separator color")
            candidates.append((color, int(full_rows[0]), int(full_cols[0])))
    assert len(candidates) == 1, "expected exactly one cross-shaped separator"
    return candidates[0]


def _background_color(g: np.ndarray, sep_r: int, sep_c: int) -> int:
    mask = np.ones(g.shape, dtype=bool)
    mask[sep_r, :] = False
    mask[:, sep_c] = False
    vals, counts = np.unique(g[mask], return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _quadrant_name(r: int, c: int, sep_r: int, sep_c: int) -> str:
    if r < sep_r and c < sep_c:
        return "tl"
    if r < sep_r and c > sep_c:
        return "tr"
    if r > sep_r and c < sep_c:
        return "bl"
    if r > sep_r and c > sep_c:
        return "br"
    raise AssertionError("seed cannot lie on the separator")


def _rectangle_bbox(
    shape: tuple[int, int],
    seed: tuple[int, int],
    sep_r: int,
    sep_c: int,
) -> tuple[tuple[int, int, int, int], bool]:
    rows, cols = shape
    r, c = seed
    assert r != sep_r and c != sep_c, "seed cannot be on the separator"

    dr = -1 if r < sep_r else 1
    dc = -1 if c < sep_c else 1
    height = abs(r - sep_r)
    width = abs(c - sep_c)
    assert height > 0 and width > 0

    far_r = r + dr * (height - 1)
    far_c = c + dc * (width - 1)
    raw_r0 = min(r, far_r)
    raw_r1 = max(r, far_r)
    raw_c0 = min(c, far_c)
    raw_c1 = max(c, far_c)

    r0 = max(0, raw_r0)
    r1 = min(rows - 1, raw_r1)
    c0 = max(0, raw_c0)
    c1 = min(cols - 1, raw_c1)
    clipped = (r0, r1, c0, c1) != (raw_r0, raw_r1, raw_c0, raw_c1)
    return (int(r0), int(c0), int(r1), int(c1)), clipped


def _paint_seed(
    out: np.ndarray,
    color: int,
    seed: tuple[int, int],
    sep_r: int,
    sep_c: int,
) -> None:
    r0, c0, r1, c1 = _rectangle_bbox(out.shape, seed, sep_r, sep_c)[0]
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow every colored seed into a quadrant-directed rectangle sized by its distances to the cross separators.

    Input structure:
    - The background is black.
    - One gray full-width horizontal line and one gray full-height vertical line form a one-pixel-wide cross that splits the grid into four quadrants.
    - Each quadrant contains one or more single-pixel colored seeds away from the separator lines, with no repeated seed color inside the same quadrant.

    Rule:
    - For each seed, measure its positive row distance to the horizontal separator and its positive column distance to the vertical separator.
    - Use those distances as the rectangle height and width, and grow a solid rectangle of the seed's color away from the cross center along the seed's quadrant diagonal.
    - If the intended rectangle would leave the grid, clip it at the boundary; keep the gray separator lines unchanged.

    Key insight:
    - A seed encodes two independent quantities by position: its distances to the two separator lines set rectangle size, and its quadrant sets the growth direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, sep_r, sep_c = _separator_axes(g)
    bg = _background_color(g, sep_r, sep_c)
    out = g.copy()

    quadrant_colors: dict[str, set[int]] = {name: set() for name in QUADRANTS}
    seed_mask = np.ones(g.shape, dtype=bool)
    seed_mask[sep_r, :] = False
    seed_mask[:, sep_c] = False
    seed_mask &= g != bg
    seed_mask &= g != sep_color

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
    assert seeds, "expected at least one seed"

    for r, c in seeds:
        color = int(g[r, c])
        quadrant = _quadrant_name(r, c, sep_r, sep_c)
        assert color not in quadrant_colors[quadrant], "seed colors should be unique inside each quadrant"
        quadrant_colors[quadrant].add(color)
        _paint_seed(out, color, (r, c), sep_r, sep_c)

    out[sep_r, :] = sep_color
    out[:, sep_c] = sep_color
    return out


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = RECT_GAP,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_sep = ar1 + gap < br0 or br1 + gap < ar0
    col_sep = ac1 + gap < bc0 or bc1 + gap < ac0
    return not (row_sep or col_sep)


def _quadrant_ranges(
    rows: int,
    cols: int,
    sep_r: int,
    sep_c: int,
    quadrant: str,
) -> tuple[range, range]:
    if quadrant == "tl":
        return range(0, sep_r - 1), range(0, sep_c - 1)
    if quadrant == "tr":
        return range(0, sep_r - 1), range(sep_c + 2, cols)
    if quadrant == "bl":
        return range(sep_r + 2, rows), range(0, sep_c - 1)
    if quadrant == "br":
        return range(sep_r + 2, rows), range(sep_c + 2, cols)
    raise AssertionError(f"unknown quadrant {quadrant!r}")


def _quadrant_candidates(
    shape: tuple[int, int],
    sep_r: int,
    sep_c: int,
    quadrant: str,
) -> list[dict[str, object]]:
    rows, cols = shape
    row_range, col_range = _quadrant_ranges(rows, cols, sep_r, sep_c, quadrant)
    candidates: list[dict[str, object]] = []
    for r in row_range:
        for c in col_range:
            bbox, clipped = _rectangle_bbox(shape, (int(r), int(c)), sep_r, sep_c)
            r0, c0, r1, c1 = bbox
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area < 2:
                continue
            candidates.append(
                {
                    "seed": (int(r), int(c)),
                    "bbox": bbox,
                    "clipped": bool(clipped),
                }
            )
    random.shuffle(candidates)
    return candidates


def _ordered_second_pool(
    candidates: list[dict[str, object]],
    preferred_clip: bool | None,
) -> list[dict[str, object]]:
    if preferred_clip is None:
        return candidates[:]
    preferred = [cand for cand in candidates if bool(cand["clipped"]) == preferred_clip]
    other = [cand for cand in candidates if bool(cand["clipped"]) != preferred_clip]
    random.shuffle(preferred)
    random.shuffle(other)
    return preferred + other


def _choose_quadrant_seeds(
    shape: tuple[int, int],
    sep_r: int,
    sep_c: int,
    quadrant: str,
    seed_count: int,
    first_clipped: bool,
) -> list[dict[str, object]] | None:
    candidates = _quadrant_candidates(shape, sep_r, sep_c, quadrant)
    first_pool = [cand for cand in candidates if bool(cand["clipped"]) == first_clipped]
    random.shuffle(first_pool)
    if not first_pool:
        return None

    second_pref = random.choice([None, first_clipped, not first_clipped])
    for first in first_pool:
        chosen = [first]
        if seed_count == 1:
            return chosen

        second_candidates = [
            cand
            for cand in candidates
            if cand["seed"] != first["seed"]
            and not _boxes_too_close(
                tuple(first["bbox"]),  # type: ignore[arg-type]
                tuple(cand["bbox"]),  # type: ignore[arg-type]
            )
        ]
        for second in _ordered_second_pool(second_candidates, second_pref):
            chosen.append(second)
            return chosen
    return None


def _expected_output(
    shape: tuple[int, int],
    sep_r: int,
    sep_c: int,
    seed_specs: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[sep_r, :] = SEP
    out[:, sep_c] = SEP
    for seed, color in seed_specs:
        _paint_seed(out, color, seed, sep_r, sep_c)
    out[sep_r, :] = SEP
    out[:, sep_c] = SEP
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    sep_r = random.randint(4, rows - 5)
    sep_c = random.randint(4, cols - 5)

    seed_counts = {quadrant: random.choice(SEED_COUNT_CHOICES) for quadrant in QUADRANTS}
    clipped_quadrants = set(random.sample(list(QUADRANTS), random.randint(1, len(QUADRANTS) - 1)))

    chosen: dict[str, list[dict[str, object]]] = {}
    for quadrant in QUADRANTS:
        picks = _choose_quadrant_seeds(
            (rows, cols),
            sep_r,
            sep_c,
            quadrant,
            seed_counts[quadrant],
            quadrant in clipped_quadrants,
        )
        if picks is None:
            return None
        chosen[quadrant] = picks

    total_seeds = sum(len(picks) for picks in chosen.values())
    if total_seeds > len(PALETTE):
        return None

    all_bboxes = [tuple(cand["bbox"]) for picks in chosen.values() for cand in picks]
    for i in range(len(all_bboxes)):
        for j in range(i + 1, len(all_bboxes)):
            if _boxes_too_close(all_bboxes[i], all_bboxes[j]):
                return None

    clipped_flags = [bool(cand["clipped"]) for picks in chosen.values() for cand in picks]
    if not any(clipped_flags) or all(clipped_flags):
        return None

    colors = random.sample(PALETTE, total_seeds)
    grid = np.full((rows, cols), BG, dtype=int)
    grid[sep_r, :] = SEP
    grid[:, sep_c] = SEP

    color_index = 0
    seed_specs: list[tuple[tuple[int, int], int]] = []
    for quadrant in QUADRANTS:
        used_colors: set[int] = set()
        for cand in chosen[quadrant]:
            color = int(colors[color_index])
            color_index += 1
            assert color not in used_colors
            used_colors.add(color)
            r, c = cand["seed"]  # type: ignore[misc]
            grid[r, c] = color
            seed_specs.append(((int(r), int(c)), color))

    expected = _expected_output((rows, cols), sep_r, sep_c, seed_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
