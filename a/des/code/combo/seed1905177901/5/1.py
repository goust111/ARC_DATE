from __future__ import annotations

import random

import numpy as np


BG = 0
OBJECT = 5
LEGEND_SIZE = 5
LEGEND_COLORS = tuple(color for color in range(1, 10) if color != OBJECT)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 4)
BODY_H_RANGE = (3, 5)
BODY_W_RANGE = (3, 6)
GENERATION_ATTEMPTS = 400
PATCH_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

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
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _count_antennae(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    pixels = {tuple(map(int, coord)) for coord in coords}
    count = 0
    for r, c in pixels:
        degree = sum((nr, nc) in pixels for nr, nc in _neighbors4(r, c))
        if degree == 1:
            count += 1
    return count


def _legend_mapping(g: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    legend_coords = np.argwhere((g != BG) & (g != OBJECT))
    assert legend_coords.shape[0] == LEGEND_SIZE, "expected exactly five legend pixels"
    assert np.unique(legend_coords[:, 1]).size == 1, "legend must be a single right-side column"
    assert np.unique(legend_coords[:, 0]).size == LEGEND_SIZE, "legend rows must be distinct"

    order = np.argsort(legend_coords[:, 0])
    ordered = legend_coords[order]
    mapping = {
        index: int(g[int(r), int(c)])
        for index, (r, c) in enumerate(ordered, start=1)
    }
    return mapping, ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each grey object's single-pixel antennae, map the count through the
    right-side legend, fill the object's bounding box background with that
    color, and remove the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mapping, legend_coords = _legend_mapping(g)
    objects = _components(g == OBJECT)
    assert objects, "expected at least one grey object"

    out = g.copy()
    for r, c in legend_coords:
        out[int(r), int(c)] = BG

    for coords in objects:
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[int(r) - r0, int(c) - c0] = True

        antennae = _count_antennae(mask)
        assert 1 <= antennae <= LEGEND_SIZE, "antenna count must index the legend"
        fill_color = mapping[antennae]

        window = out[r0 : r1 + 1, c0 : c1 + 1]
        window[window == BG] = fill_color

    return out


def _attachment_candidates(body_h: int, body_w: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for c in range(body_w):
        candidates.append((-1, c))
        candidates.append((body_h, c))
    for r in range(body_h):
        candidates.append((r, -1))
        candidates.append((r, body_w))
    return candidates


def _antennae_separated(antennae: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(antennae):
        for r1, c1 in antennae[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _make_patch(antenna_count: int) -> np.ndarray:
    for _ in range(PATCH_ATTEMPTS):
        body_h = random.randint(*BODY_H_RANGE)
        body_w = random.randint(*BODY_W_RANGE)
        candidates = _attachment_candidates(body_h, body_w)
        chosen = random.sample(candidates, antenna_count)
        if not _antennae_separated(chosen):
            continue

        pixels = {(r, c) for r in range(body_h) for c in range(body_w)}
        pixels.update(chosen)
        min_r = min(r for r, _ in pixels)
        max_r = max(r for r, _ in pixels)
        min_c = min(c for _, c in pixels)
        max_c = max(c for _, c in pixels)

        patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
        for r, c in pixels:
            patch[r - min_r, c - min_c] = OBJECT

        if _count_antennae(patch == OBJECT) == antenna_count:
            return patch

    raise RuntimeError("failed to build a valid antenna patch")


def _sample_counts() -> list[int]:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    if random.random() < 0.65:
        counts = random.sample(range(1, LEGEND_SIZE + 1), n_objects)
    else:
        counts = [random.randint(1, LEGEND_SIZE) for _ in range(n_objects)]
        if len(set(counts)) == 1 and n_objects > 1:
            pivot = random.randrange(n_objects)
            counts[pivot] = 1 + (counts[pivot] % LEGEND_SIZE)
    random.shuffle(counts)
    return counts


def _grid_size_candidates(items: list[dict[str, object]]) -> list[tuple[int, int]]:
    max_h = max(int(item["patch"].shape[0]) for item in items)
    max_w = max(int(item["patch"].shape[1]) for item in items)
    reserved_area = sum(
        (int(item["patch"].shape[0]) + 2) * (int(item["patch"].shape[1]) + 2)
        for item in items
    )

    candidates: list[tuple[int, int]] = []
    for height in range(max(GRID_H_RANGE[0], max_h + 2, LEGEND_SIZE), GRID_H_RANGE[1] + 1):
        for width in range(max(GRID_W_RANGE[0], max_w + 4), GRID_W_RANGE[1] + 1):
            play_w = width - 2
            if play_w < max_w + 2:
                continue
            usable_area = (height - 2) * (play_w - 2)
            if reserved_area <= usable_area:
                candidates.append((height, width))
    return candidates


def _place_items(
    height: int,
    width: int,
    items: list[dict[str, object]],
) -> dict[int, tuple[int, int]] | None:
    play_w = width - 2
    reserve = np.zeros((height, play_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}

    order = sorted(
        range(len(items)),
        key=lambda idx: int(items[idx]["patch"].shape[0]) * int(items[idx]["patch"].shape[1]),
        reverse=True,
    )

    for idx in order:
        patch = items[idx]["patch"]
        patch_h, patch_w = map(int, patch.shape)
        if height - patch_h - 1 < 1 or play_w - patch_w - 1 < 1:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, height - patch_h - 1)
            left = random.randint(1, play_w - patch_w - 1)
            area = reserve[top - 1 : top + patch_h + 1, left - 1 : left + patch_w + 1]
            if area.any():
                continue
            reserve[top - 1 : top + patch_h + 1, left - 1 : left + patch_w + 1] = True
            placements[idx] = (top, left)
            placed = True
            break

        if not placed:
            return None

    return placements


def _build_expected(
    grid: np.ndarray,
    items: list[dict[str, object]],
    placements: dict[int, tuple[int, int]],
    legend_top: int,
    legend_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    legend_col = grid.shape[1] - 1
    out[legend_top : legend_top + LEGEND_SIZE, legend_col] = BG

    for idx, item in enumerate(items):
        top, left = placements[idx]
        patch = item["patch"]
        color = legend_colors[int(item["count"]) - 1]
        window = out[top : top + patch.shape[0], left : left + patch.shape[1]]
        window[patch == BG] = color

    return out


def _generate_one() -> np.ndarray | None:
    counts = _sample_counts()
    legend_colors = random.sample(list(LEGEND_COLORS), LEGEND_SIZE)
    items = [{"count": count, "patch": _make_patch(count)} for count in counts]

    size_candidates = _grid_size_candidates(items)
    if not size_candidates:
        return None
    height, width = random.choice(size_candidates)

    placements = _place_items(height, width, items)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    legend_col = width - 1
    legend_top = random.randint(0, height - LEGEND_SIZE)
    for offset, color in enumerate(legend_colors):
        grid[legend_top + offset, legend_col] = color

    for idx, item in enumerate(items):
        top, left = placements[idx]
        patch = item["patch"]
        window = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        window[patch == OBJECT] = OBJECT

    if len(_components(grid == OBJECT)) != len(items):
        return None

    expected = _build_expected(grid, items, placements, legend_top, legend_colors)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
