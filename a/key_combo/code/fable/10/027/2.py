import itertools
import random

import numpy as np


DISPLAY_ROTATIONS = (0, -1, 2, 1)
ALIGN_ROTATIONS = (0, 1, 2, -1)
REFLECTION_KINDS = ("lr", "ud", "diag", "anti")
ALL_COLORS = tuple(range(10))
GEN_ATTEMPTS = 500


def _colors_present(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid)]


def _rotate_for_display(tile: np.ndarray, quadrant: int) -> np.ndarray:
    return np.rot90(tile, DISPLAY_ROTATIONS[quadrant]).copy()


def _align_to_canonical(tile: np.ndarray, quadrant: int) -> np.ndarray:
    return np.rot90(tile, ALIGN_ROTATIONS[quadrant]).copy()


def _reflect(tile: np.ndarray, kind: str) -> np.ndarray:
    if kind == "lr":
        return np.fliplr(tile).copy()
    if kind == "ud":
        return np.flipud(tile).copy()
    if kind == "diag":
        return tile.T.copy()
    if kind == "anti":
        return np.flipud(np.fliplr(tile.T)).copy()
    raise ValueError(f"unknown reflection kind: {kind}")


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    candidates = []
    for color in _colors_present(g):
        rows = np.where(np.all(g == color, axis=1))[0]
        cols = np.where(np.all(g == color, axis=0))[0]
        if len(rows) != 1 or len(cols) != 1:
            continue
        r = int(rows[0])
        c = int(cols[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            continue
        if r != h - 1 - r or c != w - 1 - c or r != c:
            continue
        candidates.append((color, r, c))
    assert len(candidates) == 1, "expected one unique separator cross"
    return candidates[0]


def _extract_quadrants(grid: np.ndarray, sep_r: int, sep_c: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    n = int(sep_r)
    assert g.shape == (2 * n + 1, 2 * n + 1), "cross must split the grid into equal square quadrants"
    return [
        g[:n, :n].copy(),
        g[:n, n + 1 :].copy(),
        g[n + 1 :, n + 1 :].copy(),
        g[n + 1 :, :n].copy(),
    ]


def _assemble_canvas(pattern: np.ndarray, separator_color: int) -> np.ndarray:
    n = int(pattern.shape[0])
    out = np.full((2 * n + 1, 2 * n + 1), int(separator_color), dtype=int)
    for quadrant in range(4):
        tile = _rotate_for_display(pattern, quadrant)
        if quadrant == 0:
            out[:n, :n] = tile
        elif quadrant == 1:
            out[:n, n + 1 :] = tile
        elif quadrant == 2:
            out[n + 1 :, n + 1 :] = tile
        else:
            out[n + 1 :, :n] = tile
    return out


def _evaluate_occluder(aligned_tiles: list[np.ndarray], occluder_color: int) -> tuple[int, np.ndarray] | None:
    visible = [tile != int(occluder_color) for tile in aligned_tiles]
    conflicts = np.zeros((4, 4), dtype=bool)
    for i in range(4):
        for j in range(i + 1, 4):
            overlap = visible[i] & visible[j]
            if np.any(aligned_tiles[i][overlap] != aligned_tiles[j][overlap]):
                conflicts[i, j] = True
                conflicts[j, i] = True

    traitors = [idx for idx in range(4) if int(conflicts[idx].sum()) == 3]
    if len(traitors) != 1:
        return None
    traitor = traitors[0]
    honest = [idx for idx in range(4) if idx != traitor]
    if any(conflicts[i, j] for i, j in itertools.combinations(honest, 2)):
        return None

    n = int(aligned_tiles[0].shape[0])
    pattern = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            votes = [
                int(aligned_tiles[idx][r, c])
                for idx in honest
                if visible[idx][r, c]
            ]
            if len(votes) < 2 or len(set(votes)) != 1:
                return None
            pattern[r, c] = votes[0]
    return traitor, pattern


def _solve_instance(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    separator_color, sep_r, sep_c = _find_separator_cross(grid)
    quadrants = _extract_quadrants(grid, sep_r, sep_c)
    aligned = [_align_to_canonical(tile, quadrant) for quadrant, tile in enumerate(quadrants)]

    interpretations = []
    for color in _colors_present(grid):
        if color == separator_color:
            continue
        solved = _evaluate_occluder(aligned, color)
        if solved is None:
            continue
        traitor, pattern = solved
        interpretations.append((color, traitor, pattern))

    assert len(interpretations) == 1, "expected one unique occluder / traitor interpretation"
    occluder_color, traitor, pattern = interpretations[0]
    return separator_color, sep_r, sep_c, occluder_color, traitor, pattern


def _subset_reconstruction(aligned_tiles: list[np.ndarray], occluder_color: int, subset: tuple[int, ...]) -> np.ndarray | None:
    n = int(aligned_tiles[0].shape[0])
    out = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            values = [
                int(aligned_tiles[idx][r, c])
                for idx in subset
                if int(aligned_tiles[idx][r, c]) != int(occluder_color)
            ]
            if not values or len(set(values)) != 1:
                return None
            out[r, c] = values[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the four-quadrant canvas by discarding the mirrored impostor and recombining the honest rotated copies.

    Input structure:
    - One full row and one full column of the same separator color split the grid into four equal square quadrants.
    - The quadrants are meant to encode the same square pattern at fixed orientations:
      upper-left 0 degrees, upper-right clockwise 90 degrees, lower-right 180 degrees, lower-left counterclockwise 90 degrees.
    - Exactly one quadrant is a mirrored fake instead of the correct rotation, and each quadrant also contains cells painted in a single occluder color.

    Rule:
    - Rotate each quadrant back to the common upper-left orientation.
    - Identify the unique occluder color and the unique quadrant whose visible cells conflict with all three others; discard that mirrored impostor.
    - Reconstruct the true pattern from the remaining three aligned copies by taking the shared visible color at every position.
    - Output the full canvas with the separator cross preserved and all four quadrants redrawn as the correct rotation of the recovered pattern, with no occlusion left.

    Key insight:
    - The four panels only become comparable after undoing their fixed quadrant rotations; then the mirrored fake is the only whole copy that visibly disagrees, so the true image is recovered by combining the three honest partial views.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    separator_color, _sep_r, _sep_c, _occluder_color, _traitor, pattern = _solve_instance(g)
    return _assemble_canvas(pattern, separator_color)


def _sample_cut(length: int) -> int:
    if length <= 2:
        return 1
    if length <= 4:
        return random.randint(1, length - 1)
    return random.randint(2, length - 2)


def _partition_square(side: int, pieces: int) -> list[tuple[int, int, int, int]] | None:
    regions = [(0, side, 0, side)]
    while len(regions) < pieces:
        candidates = []
        weights = []
        for idx, (r0, r1, c0, c1) in enumerate(regions):
            h = r1 - r0
            w = c1 - c0
            if h >= 2 or w >= 2:
                candidates.append(idx)
                weights.append(h * w)
        if not candidates:
            return None
        region_index = random.choices(candidates, weights=weights, k=1)[0]
        r0, r1, c0, c1 = regions.pop(region_index)
        h = r1 - r0
        w = c1 - c0
        orientations = []
        if h >= 2:
            orientations.append("h")
        if w >= 2:
            orientations.append("v")
        orientation = random.choice(orientations)
        if orientation == "h":
            cut = _sample_cut(h)
            regions.extend([(r0, r0 + cut, c0, c1), (r0 + cut, r1, c0, c1)])
        else:
            cut = _sample_cut(w)
            regions.extend([(r0, r1, c0, c0 + cut), (r0, r1, c0 + cut, c1)])
    return regions


def _jitter_pattern(pattern: np.ndarray, colors: list[int]) -> None:
    counts = {int(color): int(np.sum(pattern == int(color))) for color in colors}
    side = int(pattern.shape[0])
    steps = random.randint(1, max(1, side * side // 5))
    for _ in range(steps):
        r = random.randrange(side)
        c = random.randrange(side)
        old = int(pattern[r, c])
        if counts[old] <= 1:
            continue
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < side and 0 <= cc < side:
                color = int(pattern[rr, cc])
                if color != old:
                    neighbors.append(color)
        if not neighbors:
            continue
        new = random.choice(neighbors)
        pattern[r, c] = new
        counts[old] -= 1
        counts[new] += 1


def _pattern_is_rigid(pattern: np.ndarray) -> bool:
    transforms = [
        np.rot90(pattern, 1),
        np.rot90(pattern, 2),
        np.rot90(pattern, 3),
        np.fliplr(pattern),
        np.flipud(pattern),
        pattern.T,
        np.flipud(np.fliplr(pattern.T)),
    ]
    return all(not np.array_equal(pattern, variant) for variant in transforms)


def _build_true_pattern(side: int, colors: list[int]) -> np.ndarray | None:
    regions = _partition_square(side, len(colors))
    if regions is None:
        return None
    random.shuffle(colors)
    pattern = np.empty((side, side), dtype=int)
    for color, (r0, r1, c0, c1) in zip(colors, regions):
        pattern[r0:r1, c0:c1] = int(color)
    _jitter_pattern(pattern, colors)
    if len(_colors_present(pattern)) < 3:
        return None
    if not _pattern_is_rigid(pattern):
        return None
    return pattern


def _rect_mask(side: int, forbidden: np.ndarray | None, min_cells: int, max_cells: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((side, side), dtype=bool)
        patch_count = random.randint(1, 3)
        for _ in range(patch_count):
            rect_h = random.randint(1, max(1, side // 3))
            rect_w = random.randint(1, max(1, side // 3))
            r0 = random.randint(0, side - rect_h)
            c0 = random.randint(0, side - rect_w)
            mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        cells = int(mask.sum())
        if cells < min_cells or cells > max_cells:
            continue
        if forbidden is not None and np.any(mask & forbidden):
            continue
        return mask
    return None


def _make_masks(side: int, honest: list[int]) -> dict[int, np.ndarray] | None:
    max_cells = max(2, side * side // 4)
    used = np.zeros((side, side), dtype=bool)
    masks: dict[int, np.ndarray] = {}
    for idx in honest:
        mask = _rect_mask(side, used, min_cells=1, max_cells=max_cells)
        if mask is None:
            return None
        masks[idx] = mask
        used |= mask
    traitor_mask = _rect_mask(side, None, min_cells=1, max_cells=max(2, side * side // 3))
    if traitor_mask is None:
        return None
    traitor = next(idx for idx in range(4) if idx not in honest)
    masks[traitor] = traitor_mask
    return masks


def _render_input(pattern: np.ndarray, separator_color: int, occluder_color: int, traitor: int, reflection_kind: str, masks: dict[int, np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    side = int(pattern.shape[0])
    aligned_tiles = [pattern.copy() for _ in range(4)]
    aligned_tiles[traitor] = _reflect(pattern, reflection_kind)

    displayed_tiles = []
    displayed_masks = []
    for quadrant in range(4):
        displayed_tiles.append(_rotate_for_display(aligned_tiles[quadrant], quadrant))
        displayed_masks.append(_rotate_for_display(masks[quadrant].astype(int), quadrant).astype(bool))

    inp = np.full((2 * side + 1, 2 * side + 1), int(separator_color), dtype=int)
    for quadrant in range(4):
        tile = displayed_tiles[quadrant].copy()
        tile[displayed_masks[quadrant]] = int(occluder_color)
        if quadrant == 0:
            inp[:side, :side] = tile
        elif quadrant == 1:
            inp[:side, side + 1 :] = tile
        elif quadrant == 2:
            inp[side + 1 :, side + 1 :] = tile
        else:
            inp[side + 1 :, :side] = tile
    return inp, aligned_tiles, displayed_tiles


def _validate_instance(inp: np.ndarray, expected: np.ndarray, separator_color: int, occluder_color: int, traitor: int, pattern: np.ndarray, reflection_kind: str, masks: dict[int, np.ndarray]) -> bool:
    try:
        solved_sep, sep_r, sep_c, solved_occ, solved_traitor, solved_pattern = _solve_instance(inp)
    except AssertionError:
        return False

    if solved_sep != separator_color or solved_occ != occluder_color or solved_traitor != traitor:
        return False
    if not np.array_equal(solved_pattern, pattern):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    quadrants = _extract_quadrants(inp, sep_r, sep_c)
    aligned = [_align_to_canonical(tile, quadrant) for quadrant, tile in enumerate(quadrants)]
    honest = [idx for idx in range(4) if idx != traitor]

    if np.array_equal(pattern, _reflect(pattern, reflection_kind)):
        return False
    if any(int(masks[idx].sum()) == 0 for idx in range(4)):
        return False

    for honest_idx in honest:
        visible_conflict = (
            (aligned[traitor] != occluder_color)
            & (aligned[honest_idx] != occluder_color)
            & (aligned[traitor] != aligned[honest_idx])
        )
        if not np.any(visible_conflict):
            return False

    honest_vote = _subset_reconstruction(aligned, occluder_color, tuple(honest))
    if honest_vote is None or not np.array_equal(honest_vote, pattern):
        return False

    for subset in itertools.combinations(range(4), 3):
        rebuilt = _subset_reconstruction(aligned, occluder_color, subset)
        if traitor not in subset:
            if rebuilt is None or not np.array_equal(rebuilt, pattern):
                return False
        else:
            if rebuilt is not None and np.array_equal(rebuilt, pattern):
                return False

    for subset in itertools.combinations(range(4), 2):
        rebuilt = _subset_reconstruction(aligned, occluder_color, subset)
        if traitor in subset and rebuilt is not None and np.array_equal(rebuilt, pattern):
            return False

    if len(_colors_present(pattern)) < 3:
        return False
    if not _pattern_is_rigid(pattern):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    side = random.randint(5, 9)
    region_count = random.randint(3, min(5, len(ALL_COLORS) - 2))
    palette = random.sample(ALL_COLORS, region_count + 2)
    separator_color = palette[0]
    occluder_color = palette[1]
    pattern_colors = palette[2:]

    pattern = _build_true_pattern(side, pattern_colors.copy())
    if pattern is None:
        return None

    reflection_kind = random.choice(REFLECTION_KINDS)
    if np.array_equal(pattern, _reflect(pattern, reflection_kind)):
        return None

    traitor = random.randrange(4)
    honest = [idx for idx in range(4) if idx != traitor]
    masks = _make_masks(side, honest)
    if masks is None:
        return None

    inp, _aligned_tiles, _displayed_tiles = _render_input(
        pattern,
        separator_color,
        occluder_color,
        traitor,
        reflection_kind,
        masks,
    )
    expected = _assemble_canvas(pattern, separator_color)

    if not _validate_instance(
        inp,
        expected,
        separator_color,
        occluder_color,
        traitor,
        pattern,
        reflection_kind,
        masks,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-traitor rotation-vote puzzle")
