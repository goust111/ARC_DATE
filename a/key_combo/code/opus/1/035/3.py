from __future__ import annotations

import numpy as np


BG = 0
SEPARATOR = 5
LEFT_COLORS = (1, 2, 3, 4, 6)
RIGHT_COLORS = (7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PANEL_H_RANGE = (9, 13)
PANEL_W_RANGE = (9, 13)
LEFT_COMPONENT_COUNT_RANGE = (4, 6)
COMPONENT_AREA_RANGE = (3, 9)
LEFT_PLACEMENT_TRIES = 180
RIGHT_DISTRACTOR_TRIES = 120
GENERATION_TRIES = 500

_GEN_RNG = np.random.default_rng()


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_cols = np.flatnonzero(np.all(g == SEPARATOR, axis=0))
    assert divider_cols.size == 1, "expected exactly one full grey separator column"
    sep = int(divider_cols[0])
    assert 0 < sep < g.shape[1] - 1, "separator cannot lie on an outer border"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "left and right panels must have the same shape"
    assert not np.any(left == SEPARATOR), "separator color must stay in the divider column"
    assert not np.any(right == SEPARATOR), "separator color must stay in the divider column"
    return left, right


def _components_same_color(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _panel_rule(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = left.copy()
    for comp in _components_same_color(left, bg=BG):
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        overlap_colors = {int(v) for v in right[rows, cols] if int(v) != BG}
        if not overlap_colors:
            out[rows, cols] = BG
        elif len(overlap_colors) == 1:
            out[rows, cols] = next(iter(overlap_colors))
        else:
            out[rows, cols] = int(comp["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Resolve each left-panel component by counting how many distinct right-panel colors overlap it.

    Input structure:
    - The input is split by a single full grey separator column into two equal-size panels.
    - The left panel has a black background and 4-5 small 4-connected monochromatic components in distinct colors.
    - The right panel is another black-background panel whose colored pixels act as an aligned overlay lookup.

    Rule:
    - Ignore the grey divider and align the two panels by coordinates.
    - For each 4-connected component in the left panel, collect the set of non-black right-panel colors appearing on that component's cells.
    - If the set is empty, erase the whole component to black; if it has exactly one color, recolor the whole component to that color; if it has two or more colors, keep the component's original color.

    Key insight:
    - The right panel is not a binary mask: each left component queries the uniqueness of the right-panel colors covering it, and that color-count branch decides erase, recolor, or preserve.
    """
    left, right = _extract_panels(np.asarray(grid, dtype=int))
    return _panel_rule(left, right)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    row_keep = np.any(mask, axis=1)
    col_keep = np.any(mask, axis=0)
    return mask[row_keep][:, col_keep]


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(list(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    mask = np.zeros((int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _random_connected_shape(rng: np.random.Generator, area: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in coords:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if cand not in coords:
                    frontier.append(cand)
        pick = frontier[int(rng.integers(0, len(frontier)))]
        coords.add(pick)

    mask = _mask_from_coords(coords)
    if bool(rng.integers(0, 2)):
        mask = np.flipud(mask)
    if bool(rng.integers(0, 2)):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, int(rng.integers(0, 4)))
    return _trim_mask(mask)


def _coords_clear(occupied: np.ndarray, coords: np.ndarray, margin: int = 1) -> bool:
    h, w = occupied.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if occupied[r, c]:
            return False
        r0 = max(0, r - margin)
        r1 = min(h, r + margin + 1)
        c0 = max(0, c - margin)
        c1 = min(w, c + margin + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _paint_coords(grid: np.ndarray, coords: np.ndarray | set[tuple[int, int]], color: int) -> None:
    arr = coords if isinstance(coords, np.ndarray) else np.asarray(sorted(coords), dtype=int)
    if arr.size == 0:
        return
    grid[arr[:, 0], arr[:, 1]] = int(color)


def _place_left_components(
    h: int, w: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    n_components = int(rng.integers(*LEFT_COMPONENT_COUNT_RANGE))
    colors = rng.choice(np.asarray(LEFT_COLORS, dtype=int), size=n_components, replace=False)
    components: list[dict] = []

    for color_raw in colors:
        color = int(color_raw)
        placed = False
        for _ in range(LEFT_PLACEMENT_TRIES):
            area = int(rng.integers(*COMPONENT_AREA_RANGE))
            mask = _random_connected_shape(rng, area)
            mh, mw = mask.shape
            if mh > h or mw > w:
                continue
            r0 = int(rng.integers(0, h - mh + 1))
            c0 = int(rng.integers(0, w - mw + 1))
            coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
            if not _coords_clear(occupied, coords, margin=1):
                continue

            _paint_coords(grid, coords, color)
            occupied[coords[:, 0], coords[:, 1]] = True
            components.append({"color": color, "coords": coords, "area": area})
            placed = True
            break
        if not placed:
            return None

    return grid, components


def _random_connected_subset(
    coords_pool: set[tuple[int, int]], size: int, rng: np.random.Generator
) -> set[tuple[int, int]]:
    assert 1 <= size <= len(coords_pool)
    if size == len(coords_pool):
        return set(coords_pool)

    ordered = sorted(coords_pool)
    chosen = {ordered[int(rng.integers(0, len(ordered)))]}
    while len(chosen) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in chosen:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if cand in coords_pool and cand not in chosen:
                    frontier.append(cand)
        assert frontier, "pool should remain connected while growing a proper subset"
        chosen.add(frontier[int(rng.integers(0, len(frontier)))])
    return chosen


def _sample_branch_labels(n_components: int, rng: np.random.Generator) -> list[int]:
    labels = [0, 1, 2]
    extras = np.asarray([0, 1, 2], dtype=int)
    probs = np.asarray([0.25, 0.35, 0.40], dtype=float)
    while len(labels) < n_components:
        labels.append(int(rng.choice(extras, p=probs)))
    rng.shuffle(labels)
    return labels


def _build_right_panel(
    left: np.ndarray, components: list[dict], rng: np.random.Generator
) -> tuple[np.ndarray, dict] | None:
    right = np.zeros_like(left)
    intended: dict[int, dict] = {}
    labels = _sample_branch_labels(len(components), rng)

    for comp, label in zip(components, labels):
        comp_color = int(comp["color"])
        coord_set = {tuple(map(int, rc)) for rc in comp["coords"]}
        area = int(comp["area"])

        if label == 0:
            intended[comp_color] = {
                "colors": set(),
                "covered": 0,
                "output_color": BG,
            }
            continue

        if label == 1:
            color = int(rng.choice(np.asarray(RIGHT_COLORS, dtype=int)))
            subset_size = int(rng.integers(1, area))
            subset = _random_connected_subset(coord_set, subset_size, rng)
            _paint_coords(right, subset, color)
            intended[comp_color] = {
                "colors": {color},
                "covered": subset_size,
                "output_color": color,
            }
            continue

        max_colors = 3 if area >= 5 and rng.random() < 0.35 else 2
        n_colors = min(max_colors, area - 1)
        colors = rng.choice(np.asarray(RIGHT_COLORS, dtype=int), size=n_colors, replace=False)
        total_covered = int(rng.integers(n_colors, area))
        first_size = total_covered - (n_colors - 1)

        used: set[tuple[int, int]] = set()
        groups: list[set[tuple[int, int]]] = []

        first_group = _random_connected_subset(coord_set, first_size, rng)
        groups.append(first_group)
        used |= first_group

        remaining = sorted(coord_set - used)
        if len(remaining) < n_colors - 1:
            return None
        rng.shuffle(remaining)
        for idx in range(n_colors - 1):
            groups.append({remaining[idx]})

        for color_raw, group in zip(colors, groups):
            _paint_coords(right, group, int(color_raw))

        intended[comp_color] = {
            "colors": {int(c) for c in colors},
            "covered": total_covered,
            "output_color": comp_color,
        }

    if not _add_right_distractors(right, left, rng):
        return None
    return right, intended


def _add_right_distractors(right: np.ndarray, left: np.ndarray, rng: np.random.Generator) -> bool:
    h, w = right.shape
    occupied = right != BG
    placed = 0
    target = int(rng.integers(1, 4))

    for _ in range(target):
        success = False
        for _ in range(RIGHT_DISTRACTOR_TRIES):
            area = int(rng.integers(2, 6))
            mask = _random_connected_shape(rng, area)
            mh, mw = mask.shape
            if mh > h or mw > w:
                continue
            r0 = int(rng.integers(0, h - mh + 1))
            c0 = int(rng.integers(0, w - mw + 1))
            coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
            if np.any(left[coords[:, 0], coords[:, 1]] != BG):
                continue
            if not _coords_clear(occupied, coords, margin=1):
                continue
            color = int(rng.choice(np.asarray(RIGHT_COLORS, dtype=int)))
            _paint_coords(right, coords, color)
            occupied[coords[:, 0], coords[:, 1]] = True
            placed += 1
            success = True
            break
        if not success:
            continue

    return placed >= 1 and int(np.count_nonzero((left == BG) & (right != BG))) >= 2


def _expected_from_intended(left: np.ndarray, components: list[dict], intended: dict[int, dict]) -> np.ndarray:
    out = left.copy()
    for comp in components:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        out[rows, cols] = int(intended[int(comp["color"])]["output_color"])
    return out


def _naive_erase_then_local_overlay(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = left.copy()
    for comp in _components_same_color(left, bg=BG):
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        if not np.any(right[rows, cols] != BG):
            out[rows, cols] = BG
    overlay_mask = right != BG
    out[overlay_mask] = right[overlay_mask]
    return out


def _naive_min_color_repaint(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = left.copy()
    for comp in _components_same_color(left, bg=BG):
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        colors = sorted({int(v) for v in right[rows, cols] if int(v) != BG})
        if not colors:
            out[rows, cols] = BG
        else:
            out[rows, cols] = colors[0]
    return out


def _validate_instance(grid: np.ndarray, expected: np.ndarray, intended: dict[int, dict]) -> bool:
    try:
        left, right = _extract_panels(grid)
    except AssertionError:
        return False

    if expected.shape != left.shape:
        return False

    components = _components_same_color(left, bg=BG)
    if not (4 <= len(components) <= 5):
        return False

    colors = [int(comp["color"]) for comp in components]
    if len(colors) != len(set(colors)):
        return False
    if set(colors) != set(intended):
        return False
    if any(not (3 <= int(comp["area"]) <= 8) for comp in components):
        return False

    saw_zero = False
    saw_single = False
    saw_multi = False
    single_partial = False
    multi_partial = False

    for comp in components:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        overlap_colors = {int(v) for v in right[rows, cols] if int(v) != BG}
        covered = int(np.count_nonzero(right[rows, cols] != BG))
        info = intended[int(comp["color"])]
        if overlap_colors != info["colors"]:
            return False
        if covered != int(info["covered"]):
            return False

        if len(overlap_colors) == 0:
            saw_zero = True
        elif len(overlap_colors) == 1:
            saw_single = True
            if covered < int(comp["area"]):
                single_partial = True
        else:
            saw_multi = True
            if covered < int(comp["area"]):
                multi_partial = True

    if not (saw_zero and saw_single and saw_multi):
        return False
    if not (single_partial and multi_partial):
        return False
    if int(np.count_nonzero((left == BG) & (right != BG))) < 2:
        return False

    if not np.array_equal(_panel_rule(left, right), expected):
        return False
    if np.array_equal(_naive_erase_then_local_overlay(left, right), expected):
        return False
    if np.array_equal(_naive_min_color_repaint(left, right), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_GEN_RNG.integers(*PANEL_H_RANGE))
    w = int(_GEN_RNG.integers(*PANEL_W_RANGE))

    placed = _place_left_components(h, w, _GEN_RNG)
    if placed is None:
        return None
    left, components = placed

    built = _build_right_panel(left, components, _GEN_RNG)
    if built is None:
        return None
    right, intended = built

    expected = _expected_from_intended(left, components, intended)
    grid = np.concatenate([left, np.full((h, 1), SEPARATOR, dtype=int), right], axis=1)

    if not _validate_instance(grid, expected, intended):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GENERATION_TRIES} attempts")
