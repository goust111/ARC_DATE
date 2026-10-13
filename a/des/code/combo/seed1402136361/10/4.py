from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def _component_marker_and_body(
    grid: np.ndarray, coords: np.ndarray
) -> tuple[int, int, int, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    values = [int(grid[r, c]) for r, c in coords]
    colors = sorted(set(values))
    assert len(colors) == 2, "each foreground object must contain one body color and one marker color"

    coord_set = {(int(r), int(c)) for r, c in coords}
    counts = Counter(values)
    boundary_touches = {color: 0 for color in colors}
    for r, c in coords:
        color = int(grid[r, c])
        if any((nr, nc) not in coord_set for nr, nc in _neighbors4(int(r), int(c))):
            boundary_touches[color] += 1

    marker_candidates = [color for color in colors if boundary_touches[color] == 0]
    if len(marker_candidates) != 1:
        marker_candidates = [color for color in colors if counts[color] <= 3]
    assert len(marker_candidates) == 1, "failed to identify the interior marker color"

    marker_color = int(marker_candidates[0])
    body_color = int(colors[0] if colors[1] == marker_color else colors[1])
    marker_count = int(counts[marker_color])
    assert 1 <= marker_count <= 3, "marker count must encode a scale factor from 1 to 3"
    assert counts[body_color] > marker_count, "body must stay the dominant color inside each object"
    return body_color, marker_color, marker_count, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each object's internal marker-count into its output scale.

    Input structure:
    - One dominant background color.
    - Several separated 4-connected foreground objects.
    - Each object has one body color plus 1-3 internal marker pixels of a shared marker color.

    Rule:
    - Count the marker pixels k inside each object.
    - Remove the marker pixels and scale the remaining body shape by factor k.
    - Pack the scaled objects left-to-right with one background column between them.
    - Bottom-align all packed objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    components = _find_components(g != bg)
    assert components, "expected at least one foreground object"

    rendered: list[tuple[int, int, np.ndarray]] = []
    for coords in components:
        body_color, marker_color, factor, bbox = _component_marker_and_body(g, coords)
        r0, c0, r1, c1 = bbox

        body_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            if int(g[r, c]) == body_color:
                body_mask[int(r) - r0, int(c) - c0] = True

        assert body_mask.any(), "each object must retain body pixels after removing markers"
        scaled_mask = _scale_mask(body_mask, factor)
        patch = np.full(scaled_mask.shape, bg, dtype=int)
        patch[scaled_mask] = body_color
        rendered.append((c0, r0, patch))

    rendered.sort(key=lambda item: (int(item[0]), int(item[1])))
    out_h = max(int(patch.shape[0]) for _, _, patch in rendered)
    out_w = sum(int(patch.shape[1]) for _, _, patch in rendered) + len(rendered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c_cur = 0
    for _c0, _r0, patch in rendered:
        ph, pw = patch.shape
        r_start = out_h - ph
        mask = patch != bg
        target = out[r_start : r_start + ph, c_cur : c_cur + pw]
        target[mask] = patch[mask]
        c_cur += pw + 1

    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(10))
OBJECT_COUNT_RANGE = (2, 5)
GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
OBJECT_HEIGHT_RANGE = (3, 6)
OBJECT_WIDTH_RANGE = (3, 7)
MAX_NOTCHES = 4
MAX_SHAPE_ATTEMPTS = 80
MAX_PLACEMENT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 300
PATCH_BG = -1


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c]:
                continue
            if all(
                0 <= nr < height and 0 <= nc < width and mask[nr, nc]
                for nr, nc in _neighbors4(r, c)
            ):
                cells.append((r, c))
    return cells


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    seen = set()
    stack = [tuple(map(int, cells[0]))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c]:
                continue
            if any(
                not (0 <= nr < height and 0 <= nc < width and mask[nr, nc])
                for nr, nc in _neighbors4(r, c)
            ):
                cells.append((r, c))
    return cells


def _sample_body_mask(marker_count: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        height = random.randint(*OBJECT_HEIGHT_RANGE)
        width = random.randint(*OBJECT_WIDTH_RANGE)
        if (height - 2) * (width - 2) < marker_count:
            continue

        mask = np.ones((height, width), dtype=bool)
        for _step in range(random.randint(0, MAX_NOTCHES)):
            candidates = _boundary_cells(mask)
            random.shuffle(candidates)
            removed = False
            for r, c in candidates:
                trial = mask.copy()
                trial[r, c] = False
                if not _mask_connected(trial):
                    continue
                if len(_interior_cells(trial)) < marker_count:
                    continue
                if trial.sum() <= marker_count + 2:
                    continue
                mask = trial
                removed = True
                break
            if not removed:
                break

        interior = _interior_cells(mask)
        if len(interior) < marker_count:
            continue

        choices = interior[:]
        random.shuffle(choices)
        markers: list[tuple[int, int]] = []
        for r, c in choices:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in markers):
                markers.append((r, c))
                if len(markers) == marker_count:
                    break
        if len(markers) != marker_count:
            continue

        body_only = mask.copy()
        for r, c in markers:
            body_only[r, c] = False
        if not _mask_connected(body_only):
            continue

        return mask, markers
    return None


def _render_object(body_color: int, marker_color: int, marker_count: int) -> np.ndarray | None:
    sampled = _sample_body_mask(marker_count)
    if sampled is None:
        return None

    mask, markers = sampled
    patch = np.full(mask.shape, PATCH_BG, dtype=int)
    patch[mask] = body_color
    for r, c in markers:
        patch[r, c] = marker_color
    return patch


def _sample_marker_counts(n_objects: int) -> list[int]:
    if n_objects == 2:
        counts = random.sample([1, 2, 3], 2)
    elif random.random() < 0.5:
        counts = [1, 2, 3] + [random.randint(1, 3) for _ in range(n_objects - 3)]
    else:
        while True:
            counts = [random.randint(1, 3) for _ in range(n_objects)]
            if len(set(counts)) >= 2:
                break
    random.shuffle(counts)
    return counts


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = counts.most_common()
    return ranked[0][0] == bg and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])


def _place_objects(grid: np.ndarray, objects: list[np.ndarray]) -> bool:
    height, width = grid.shape
    forbidden = np.zeros_like(grid, dtype=bool)
    used_lefts: set[int] = set()

    order = list(range(len(objects)))
    random.shuffle(order)
    for idx in order:
        patch = objects[idx]
        ph, pw = patch.shape
        if ph > height or pw > width:
            return False

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            if c0 in used_lefts:
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc1 = min(width, c0 + pw + 1)
            if forbidden[rr0:rr1, cc0:cc1].any():
                continue

            sub = patch != PATCH_BG
            target = grid[r0 : r0 + ph, c0 : c0 + pw]
            target[sub] = patch[sub]
            forbidden[rr0:rr1, cc0:cc1] = True
            used_lefts.add(c0)
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    bg, marker_color = random.sample(COLOR_POOL, 2)
    body_palette = [color for color in COLOR_POOL if color not in (bg, marker_color)]
    random.shuffle(body_palette)
    body_colors = body_palette[:n_objects]

    marker_counts = _sample_marker_counts(n_objects)
    objects: list[np.ndarray] = []
    for body_color, marker_count in zip(body_colors, marker_counts):
        patch = _render_object(body_color, marker_color, marker_count)
        if patch is None:
            return None
        objects.append(patch)

    min_h = max(max(obj.shape[0] for obj in objects) + 2, GRID_HEIGHT_RANGE[0])
    min_w = max(sum(obj.shape[1] for obj in objects) + len(objects) + 1, GRID_WIDTH_RANGE[0])
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_w, GRID_WIDTH_RANGE[1])
    grid = np.full((height, width), bg, dtype=int)
    if not _place_objects(grid, objects):
        return None

    if not _background_is_dominant(grid, bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray) or out.ndim != 2 or out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
