import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
GEN_MAX_ATTEMPTS = 400
GEN_MAX_RINGS = 5
GEN_MARGIN_RANGE = (1, 4)
GEN_OUTER_SIZE_RANGE = (6, 22)


def _majority_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _ring_path(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    assert r1 > r0 and c1 > c0
    path = [(r0, c) for c in range(c0, c1 + 1)]
    path.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    path.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    path.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return path


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each ring's marker pixels by depth parity along the ring boundary."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _majority_color(g)
    components = _find_components(g != bg)
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], -comp["bbox"][2], -comp["bbox"][3]))

    out = g.copy()
    prev_bbox = None
    for depth, comp in enumerate(components, start=1):
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 > r0 and c1 > c0
        if prev_bbox is not None:
            pr0, pc0, pr1, pc1 = prev_bbox
            assert pr0 < r0 and pc0 < c0 and pr1 > r1 and pc1 > c1
        prev_bbox = comp["bbox"]

        path = _ring_path(r0, c0, r1, c1)
        path_set = set(path)
        comp_set = set(comp["coords"])
        assert len(path) == len(path_set)
        assert comp_set == path_set

        ring_colors = Counter(int(g[r, c]) for r, c in path)
        ring_color = int(ring_colors.most_common(1)[0][0])
        shift = 2 if depth % 2 == 1 else -2

        for r, c in path:
            out[r, c] = ring_color

        for i, (r, c) in enumerate(path):
            marker_color = int(g[r, c])
            if marker_color == ring_color:
                continue
            nr, nc = path[(i + shift) % len(path)]
            out[nr, nc] = marker_color

    return out


def _choose_ring_count() -> int:
    candidates = list(range(1, GEN_MAX_RINGS + 1))
    weights = [1] + [3] * (len(candidates) - 1)
    return random.choices(candidates, weights=weights, k=1)[0]


def _build_candidate() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    ring_count = _choose_ring_count()
    min_outer = 2 + 4 * (ring_count - 1)
    outer_low = max(min_outer, GEN_OUTER_SIZE_RANGE[0])
    outer_high = GEN_OUTER_SIZE_RANGE[1]
    if outer_low > outer_high:
        return None

    outer_h = random.randint(outer_low, outer_high)
    outer_w = random.randint(outer_low, outer_high)
    margin_h = random.randint(*GEN_MARGIN_RANGE)
    margin_w = random.randint(*GEN_MARGIN_RANGE)

    h = outer_h + 2 * margin_h
    w = outer_w + 2 * margin_w
    grid = np.full((h, w), bg, dtype=int)

    ring_colors = random.sample([c for c in PALETTE if c != bg], ring_count)
    top = margin_h
    left = margin_w
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    for depth in range(1, ring_count + 1):
        r0 = top + 2 * (depth - 1)
        c0 = left + 2 * (depth - 1)
        r1 = bottom - 2 * (depth - 1)
        c1 = right - 2 * (depth - 1)
        path = _ring_path(r0, c0, r1, c1)
        ring_color = ring_colors[depth - 1]
        for r, c in path:
            grid[r, c] = ring_color

        max_markers = max(1, min(5, (len(path) - 1) // 3))
        marker_count = random.randint(1, max_markers)
        marker_positions = random.sample(range(len(path)), marker_count)
        marker_palette = [c for c in PALETTE if c != bg and c != ring_color]
        for idx in marker_positions:
            r, c = path[idx]
            grid[r, c] = random.choice(marker_palette)

    if _majority_color(grid) != bg:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
