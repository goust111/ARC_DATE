from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ORIENTATIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            result.append(comp)
    return result


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _read_left_legend(grid: np.ndarray) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, _ = grid.shape
    mapping: dict[int, int] = {}
    cells: set[tuple[int, int]] = set()
    r = 0
    while r < h:
        if int(grid[r, 0]) == BG:
            r += 1
            continue
        assert r + 1 < h and int(grid[r + 1, 0]) != BG, "legend entries must be vertical pairs"
        source = int(grid[r, 0])
        target = int(grid[r + 1, 0])
        assert source != BG and target != BG
        assert source not in mapping, "legend source colors must be unique"
        mapping[source] = target
        cells.add((r, 0))
        cells.add((r + 1, 0))
        r += 2
    assert mapping, "expected at least one left-edge legend pair"
    return mapping, cells


def _shape_template(height: int, width: int, orientation: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if orientation == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif orientation == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif orientation == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    elif orientation == "right":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(orientation)
    return mask


def _detect_orientation(mask: np.ndarray) -> str:
    h, w = mask.shape
    matches = [orientation for orientation in ORIENTATIONS if np.array_equal(mask, _shape_template(h, w, orientation))]
    assert len(matches) == 1, "component must be exactly one open U/C outline"
    return matches[0]


def _fill_stripes(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    orientation: str,
    fill_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if orientation == "up":
        assert r1 > r0 and c1 - c0 >= 2
        for step, r in enumerate(range(r1 - 1, r0 - 1, -1)):
            out[r, c0 + 1 : c1] = fill_color if step % 2 == 0 else BG
    elif orientation == "down":
        assert r1 > r0 and c1 - c0 >= 2
        for step, r in enumerate(range(r0 + 1, r1 + 1)):
            out[r, c0 + 1 : c1] = fill_color if step % 2 == 0 else BG
    elif orientation == "left":
        assert c1 > c0 and r1 - r0 >= 2
        for step, c in enumerate(range(c1 - 1, c0 - 1, -1)):
            out[r0 + 1 : r1, c] = fill_color if step % 2 == 0 else BG
    elif orientation == "right":
        assert c1 > c0 and r1 - r0 >= 2
        for step, c in enumerate(range(c0 + 1, c1 + 1)):
            out[r0 + 1 : r1, c] = fill_color if step % 2 == 0 else BG
    else:
        raise ValueError(orientation)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the left-edge color legend, detect each monochrome open U/C outline,
    then fill its interior with alternating stripes whose orientation follows
    the opening direction and whose color comes from the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mapping, legend_cells = _read_left_legend(g)
    object_mask = g != BG
    for r, c in legend_cells:
        object_mask[r, c] = False

    out = g.copy()
    objects = _components(object_mask)
    assert objects, "expected at least one outline object"
    for coords in objects:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        outline_color = next(iter(colors))
        assert outline_color in mapping, "every outline color must appear in the legend"

        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            patch[r - r0, c - c0] = True

        orientation = _detect_orientation(patch)
        _fill_stripes(out, (r0, c0, r1, c1), orientation, mapping[outline_color])

    return out


def _make_outline(height: int, width: int, orientation: str, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[_shape_template(height, width, orientation)] = color
    return patch


def _legend_layout(entry_count: int, rng: random.Random) -> tuple[list[int], int]:
    gaps = [rng.randint(0, 1) for _ in range(max(0, entry_count - 1))]
    total = 2 * entry_count + sum(gaps)
    return gaps, total


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
    min_col: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for r in range(1, h - ph):
        for c in range(min_col, w - pw):
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(h, r + ph + 1)
            cc1 = min(w, c + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r, c))
    if not positions:
        return False

    r, c = rng.choice(positions)
    mask = patch != BG
    grid[r : r + ph, c : c + pw][mask] = patch[mask]
    occupied[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)] = True
    return True


def _build_sample(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.randint(1, 4)
    num_entries = min(4, num_objects + rng.randint(0, 1))
    directions = rng.sample(list(ORIENTATIONS), k=num_objects)
    gaps, legend_height = _legend_layout(num_entries, rng)

    min_h = max(12, legend_height + 2, 8 + 2 * num_objects)
    min_w = max(14, 10 + 3 * num_objects)
    if min_h > 24 or min_w > 28:
        return None
    height = rng.randint(min_h, 24)
    width = rng.randint(min_w, 28)

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    sources = color_pool[:num_entries]
    targets = color_pool[num_entries : 2 * num_entries]
    object_colors = rng.sample(sources, k=num_objects)

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    legend_top = rng.randint(0, height - legend_height)
    row = legend_top
    for idx, (source, target) in enumerate(zip(sources, targets)):
        grid[row, 0] = source
        grid[row + 1, 0] = target
        occupied[row : row + 2, 0:2] = True
        row += 2
        if idx < len(gaps):
            row += gaps[idx]
    occupied[:, 0:2] = True

    max_dim = 8 if num_objects <= 2 else 6
    specs = []
    for direction, color in zip(directions, object_colors):
        ph = rng.randint(4, min(max_dim, height - 2))
        pw = rng.randint(4, min(max_dim + 1, width - 3))
        specs.append((ph * pw, ph, pw, direction, color))
    specs.sort(reverse=True)

    for _, ph, pw, direction, color in specs:
        patch = _make_outline(ph, pw, direction, color)
        if not _place_patch(grid, occupied, patch, rng, min_col=2):
            return None

    transformed = main(grid)
    assert transformed.shape == grid.shape
    assert np.any(transformed != grid), "sample must produce a nontrivial output"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_sample(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
