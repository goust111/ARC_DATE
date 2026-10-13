from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _majority_color_and_count(source: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in source.flat if int(v) not in (BLACK, GREY))
    assert counts, "expected at least one non-black, non-grey source color"
    max_count = max(counts.values())
    majority = [color for color, count in counts.items() if count == max_count]
    assert len(majority) == 1, "expected a unique majority color"
    return int(majority[0]), int(max_count)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each sufficiently large lower component using the majority color from the upper pattern.

    Input structure:
    - One full-width grey separator row splits the grid into an upper pattern region and a lower result region.
    - The upper region has a black background with scattered non-black, non-grey colored pixels.
    - The lower region has a black background with several disconnected non-black components.

    Rule:
    - Find the unique most frequent non-black, non-grey color C in the upper region and its pixel count K.
    - In the lower region, examine every 4-connected non-black component.
    - Recolor every such component whose area is at least K to color C; keep smaller components unchanged.
    - Preserve the upper region and the grey separator row.

    Key insight:
    - The upper pattern encodes two linked outputs at once: its majority color supplies the repaint color, and that same color's frequency supplies the area threshold.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _separator_row(g)
    source = g[:sep, :]
    lower = g[sep + 1 :, :]
    color, threshold = _majority_color_and_count(source)

    out = g.copy()
    for comp in _components4(lower != BLACK):
        if int(comp.shape[0]) < threshold:
            continue
        rr = comp[:, 0] + sep + 1
        cc = comp[:, 1]
        out[rr, cc] = color
    return out


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int).copy()
    pts[:, 0] -= int(pts[:, 0].min())
    pts[:, 1] -= int(pts[:, 1].min())
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]


def _transform_coords(coords: np.ndarray, rng: random.Random) -> np.ndarray:
    pts = np.asarray(coords, dtype=int).copy()
    if rng.random() < 0.5:
        pts[:, 1] = -pts[:, 1]
    for _ in range(rng.randrange(4)):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    return _normalize_coords(pts)


def _random_polyomino(area: int, rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    rows = [p[0] for p in trial]
                    cols = [p[1] for p in trial]
                    if max(rows) - min(rows) + 1 <= 5 and max(cols) - min(cols) + 1 <= 5:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        coords = _transform_coords(np.asarray(list(cells), dtype=int), rng)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if area >= 5 and min(h, w) == 1 and rng.random() < 0.75:
            continue
        if area / float(h * w) < 0.4:
            continue
        return coords
    raise RuntimeError(f"failed to build polyomino of area {area}")


def _can_place(occupied: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, occupied.shape)
    pts = {(int(r0 + r), int(c0 + c)) for r, c in shape}
    for rr, cc in pts:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if occupied[rr, cc]:
            return False
    for rr, cc in pts:
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in pts:
                return False
    return True


def _place_component(lower: np.ndarray, occupied: np.ndarray, color: int, area: int, rng: random.Random) -> bool:
    for _ in range(PLACEMENT_ATTEMPTS):
        shape = _random_polyomino(area, rng)
        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1
        if h > lower.shape[0] or w > lower.shape[1]:
            continue
        r0 = rng.randrange(lower.shape[0] - h + 1)
        c0 = rng.randrange(lower.shape[1] - w + 1)
        if not _can_place(occupied, shape, r0, c0):
            continue
        for r, c in shape:
            rr = r0 + int(r)
            cc = c0 + int(c)
            lower[rr, cc] = int(color)
            occupied[rr, cc] = True
        return True
    return False


def _choose_top_counts(k: int, n_colors: int, rng: random.Random) -> list[int]:
    max_total = min(25, k + (n_colors - 1) * (k - 1))
    min_total = k + (n_colors - 1)
    preferred_low = 15 if max_total >= 15 else max(8, min_total)
    total = rng.randint(preferred_low, max_total)
    remainder = total - k
    for _ in range(200):
        parts = [1] * (n_colors - 1)
        extra = remainder - (n_colors - 1)
        while extra > 0:
            idx = rng.randrange(n_colors - 1)
            if parts[idx] < k - 1:
                parts[idx] += 1
                extra -= 1
        if max(parts) < k:
            rng.shuffle(parts)
            return [k, *parts]
    raise RuntimeError(f"failed to choose top counts for k={k}, n_colors={n_colors}")


def _choose_component_areas(k: int, n_components: int, rng: random.Random) -> list[int]:
    areas = [k, min(8, k + 1), (k - 1) if k > 4 else rng.choice([2, 3])]
    while len(areas) < n_components:
        areas.append(rng.randint(2, 8))
    rng.shuffle(areas)
    return areas


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    k = rng.randint(4, 7)
    n_top_colors = 3 if k == 4 else rng.choice([3, 4])
    n_bottom_components = rng.randint(4, min(5, 8 - n_top_colors))

    top_h = rng.randint(4, 6)
    bottom_h = rng.randint(6, 9)
    width = rng.randint(16, 24)
    height = top_h + 1 + bottom_h
    if height > 30 or width > 30:
        return None

    top_colors = rng.sample(PALETTE, n_top_colors)
    majority_index = rng.randrange(n_top_colors)
    majority_color = top_colors[majority_index]
    other_top = [c for i, c in enumerate(top_colors) if i != majority_index]
    bottom_palette = [c for c in PALETTE if c not in top_colors]
    bottom_colors = rng.sample(bottom_palette, n_bottom_components)

    counts = _choose_top_counts(k, n_top_colors, rng)
    colors_in_count_order = [majority_color, *other_top]

    grid = np.zeros((height, width), dtype=int)
    grid[top_h, :] = GREY

    total_top = sum(counts)
    top_indices = rng.sample(range(top_h * width), total_top)
    cursor = 0
    for color, count in zip(colors_in_count_order, counts):
        for idx in top_indices[cursor : cursor + count]:
            r, c = divmod(idx, width)
            grid[r, c] = int(color)
        cursor += count

    lower = grid[top_h + 1 :, :]
    occupied = np.zeros_like(lower, dtype=bool)
    areas = _choose_component_areas(k, n_bottom_components, rng)
    pairs = sorted(zip(areas, bottom_colors), reverse=True)
    for area, color in pairs:
        if not _place_component(lower, occupied, color, area, rng):
            return None

    if rng.random() < 0.5:
        grid = np.fliplr(grid)
    if rng.random() < 0.35:
        top = np.flipud(grid[:top_h, :])
        bottom = np.flipud(grid[top_h + 1 :, :])
        grid = np.vstack(
            [
                top,
                np.full((1, width), GREY, dtype=int),
                bottom,
            ]
        )
    return np.asarray(grid, dtype=int)


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    sep = _separator_row(g)
    assert 0 < sep < g.shape[0] - 1

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    top_colors = sorted({int(v) for v in top.flat if int(v) not in (BLACK, GREY)})
    assert 3 <= len(top_colors) <= 4
    assert not np.any(bottom == GREY)

    majority_color, k = _majority_color_and_count(top)
    assert 4 <= k <= 7
    counts = Counter(int(v) for v in top.flat if int(v) not in (BLACK, GREY))
    assert counts[majority_color] == k
    assert sum(count == k for count in counts.values()) == 1

    components = _components4(bottom != BLACK)
    assert 4 <= len(components) <= 5
    areas = []
    lower_colors = set()
    for comp in components:
        area = int(comp.shape[0])
        areas.append(area)
        assert 2 <= area <= 8
        values = {int(bottom[r, c]) for r, c in comp}
        assert len(values) == 1
        color = next(iter(values))
        assert color not in top_colors
        lower_colors.add(color)
    assert len(lower_colors) == len(components)
    assert any(area < k for area in areas)
    assert any(area >= k for area in areas)
    assert k in areas
    assert any(area > k for area in areas)
    if k > 4:
        assert (k - 1) in areas

    sorted_areas = tuple(sorted(areas))
    target_sig = tuple(area >= k for area in sorted_areas)
    for fixed_k in range(4, 8):
        if fixed_k != k:
            assert target_sig != tuple(area >= fixed_k for area in sorted_areas)
    assert target_sig != tuple(area == k for area in sorted_areas)

    out = main(g)
    assert np.array_equal(out[: sep + 1, :], g[: sep + 1, :])
    changed = 0
    unchanged = 0
    lower_out = out[sep + 1 :, :]
    for comp in components:
        area = int(comp.shape[0])
        orig_color = int(bottom[tuple(comp[0])])
        new_colors = {int(lower_out[r, c]) for r, c in comp}
        assert len(new_colors) == 1
        new_color = next(iter(new_colors))
        if area >= k:
            assert new_color == majority_color
            changed += 1
        else:
            assert new_color == orig_color
            unchanged += 1
    assert changed >= 1 and unchanged >= 1
    assert not np.array_equal(out, g)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is None:
            continue
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
