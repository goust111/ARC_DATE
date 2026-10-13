from __future__ import annotations

import random

import numpy as np


BLACK = 0
STRIP_RADIUS = 2
MIN_SPAN = 6
MAX_SPAN = 16
MAX_GENERATION_ATTEMPTS = 500


def _marker_pairs(g: np.ndarray) -> list[dict]:
    pairs = []
    for color in np.unique(g):
        color = int(color)
        if color == BLACK:
            continue
        coords = np.argwhere(g == color)
        if coords.shape[0] != 2:
            continue
        (r0, c0), (r1, c1) = [(int(r), int(c)) for r, c in coords]
        if r0 == r1 or c0 == c1:
            pairs.append({"color": color, "a": (r0, c0), "b": (r1, c1)})
    assert pairs, "expected at least one pair of aligned two-pixel markers"
    return pairs


def _strip_bounds(pair: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    H, W = shape
    (r0, c0), (r1, c1) = pair["a"], pair["b"]
    if r0 == r1:
        rr0, rr1 = r0 - STRIP_RADIUS, r0 + STRIP_RADIUS
        cc0, cc1 = sorted((c0, c1))
    else:
        rr0, rr1 = sorted((r0, r1))
        cc0, cc1 = c0 - STRIP_RADIUS, c0 + STRIP_RADIUS
    assert 0 <= rr0 <= rr1 < H and 0 <= cc0 <= cc1 < W, "marker strip leaves the grid"
    return rr0, rr1, cc0, cc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the blank half of each marker-defined strip with the point-rotated copy of the colored pattern.

    Input structure:
    - Black background with several marker pairs; each pair is two single pixels
      of one non-black color, and different pairs use different marker colors.
    - The two markers of a pair lie on the same row or the same column.
    - Their connecting line is the center of a width-five strip: horizontal
      pairs extend two rows above and below, vertical pairs extend two columns
      left and right.
    - In exactly one half of each strip there are non-black, non-marker pattern
      pixels; the opposite half is black.

    Rule:
    - For every marker pair, use the midpoint of the two marker pixels as the
      center of symmetry.
    - Copy every non-black, non-marker pixel inside that strip to its position
      after a 180 degree rotation around the midpoint.
    - Keep all original pixels, and restore marker pixels if a copied pattern
      would land on a marker.

    Key insight:
    - The matching region is not reflected across the marker line; it is copied
      by point symmetry through the midpoint between the two same-colored markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs = _marker_pairs(g)
    marker_colors = {int(pair["color"]) for pair in pairs}
    marker_cells = {pair["a"] for pair in pairs} | {pair["b"] for pair in pairs}

    out = g.copy()
    for pair in pairs:
        (r0, c0), (r1, c1) = pair["a"], pair["b"]
        rr0, rr1, cc0, cc1 = _strip_bounds(pair, g.shape)

        for r in range(rr0, rr1 + 1):
            for c in range(cc0, cc1 + 1):
                value = int(g[r, c])
                if value == BLACK or value in marker_colors:
                    continue
                mr = r0 + r1 - r
                mc = c0 + c1 - c
                if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and (mr, mc) not in marker_cells:
                    out[mr, mc] = value

    for r, c in marker_cells:
        out[r, c] = g[r, c]
    return out


# --------------- generator ---------------


def _rect_free(blocked: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int = 1) -> bool:
    H, W = blocked.shape
    rr0 = max(0, int(r0) - pad)
    rr1 = min(H - 1, int(r1) + pad)
    cc0 = max(0, int(c0) - pad)
    cc1 = min(W - 1, int(c1) + pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _half_cells(pair: dict, side: int) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = pair["a"], pair["b"]
    rr0, rr1, cc0, cc1 = pair["strip"]
    cells = []
    if r0 == r1:
        middle = c0 + c1
        for r in range(rr0, rr1 + 1):
            for c in range(cc0 + 1, cc1):
                if side < 0 and 2 * c < middle:
                    cells.append((r, c))
                elif side > 0 and 2 * c > middle:
                    cells.append((r, c))
    else:
        middle = r0 + r1
        for r in range(rr0 + 1, rr1):
            for c in range(cc0, cc1 + 1):
                if side < 0 and 2 * r < middle:
                    cells.append((r, c))
                elif side > 0 and 2 * r > middle:
                    cells.append((r, c))
    return cells


def _grow_pattern(cells: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    allowed = set(cells)
    start = random.choice(cells)
    chosen = {start}
    frontier = {start}

    while len(chosen) < target:
        candidates = set()
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in allowed and nb not in chosen:
                    candidates.add(nb)
        if not candidates:
            candidates = allowed - chosen
        pick = random.choice(list(candidates))
        chosen.add(pick)
        frontier.add(pick)
    return list(chosen)


def _sample_pair(H: int, W: int, orient: str, blocked: np.ndarray) -> dict | None:
    candidates = []
    if orient == "horizontal" and H >= 5 and W >= MIN_SPAN + 1:
        for r in range(STRIP_RADIUS, H - STRIP_RADIUS):
            for span in range(MIN_SPAN, min(MAX_SPAN, W - 1) + 1):
                for c0 in range(0, W - span):
                    c1 = c0 + span
                    rr0, rr1 = r - STRIP_RADIUS, r + STRIP_RADIUS
                    cc0, cc1 = c0, c1
                    if _rect_free(blocked, rr0, rr1, cc0, cc1):
                        candidates.append(((r, c0), (r, c1), (rr0, rr1, cc0, cc1)))
    elif orient == "vertical" and W >= 5 and H >= MIN_SPAN + 1:
        for c in range(STRIP_RADIUS, W - STRIP_RADIUS):
            for span in range(MIN_SPAN, min(MAX_SPAN, H - 1) + 1):
                for r0 in range(0, H - span):
                    r1 = r0 + span
                    rr0, rr1 = r0, r1
                    cc0, cc1 = c - STRIP_RADIUS, c + STRIP_RADIUS
                    if _rect_free(blocked, rr0, rr1, cc0, cc1):
                        candidates.append(((r0, c), (r1, c), (rr0, rr1, cc0, cc1)))
    if not candidates:
        return None

    a, b, strip = random.choice(candidates)
    side = random.choice((-1, 1))
    pair = {"a": a, "b": b, "strip": strip, "side": side, "orient": orient}
    if len(_half_cells(pair, side)) < 3:
        return None
    rr0, rr1, cc0, cc1 = strip
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return pair


def _sample_scene() -> tuple[np.ndarray, list[tuple[int, int]], list[int]] | None:
    H = random.randint(12, 30)
    W = random.randint(12, 30)
    n_pairs = random.randint(1, 4)

    grid = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)
    pairs = []

    orientations = [random.choice(("horizontal", "vertical")) for _ in range(n_pairs)]
    if n_pairs >= 2 and len(set(orientations)) == 1 and random.random() < 0.6:
        orientations[random.randrange(n_pairs)] = "vertical" if orientations[0] == "horizontal" else "horizontal"
    random.shuffle(orientations)

    for orient in orientations:
        pair = _sample_pair(H, W, orient, blocked)
        if pair is None:
            return None
        pairs.append(pair)

    colors = random.sample(range(1, 10), n_pairs + 1)
    marker_colors = colors[:n_pairs]
    pattern_pool = colors[n_pairs:]

    all_pattern_cells: list[tuple[int, int]] = []
    for pair in pairs:
        cells = _half_cells(pair, int(pair["side"]))
        target = random.randint(3, min(12, len(cells)))
        pattern = _grow_pattern(cells, target)
        pair["pattern"] = pattern
        all_pattern_cells.extend(pattern)

    extra_pattern_colors = [c for c in range(1, 10) if c not in marker_colors and c not in pattern_pool]
    random.shuffle(extra_pattern_colors)
    usable_colors = pattern_pool + extra_pattern_colors[:2]
    max_used = max(1, min(3, len(usable_colors), len(all_pattern_cells) // 3))
    used_count = random.randint(1, max_used)
    used_pattern_colors = usable_colors[:used_count]

    counts = [3] * used_count
    remaining = len(all_pattern_cells) - sum(counts)
    for _ in range(remaining):
        counts[random.randrange(used_count)] += 1

    pattern_values = []
    for color, count in zip(used_pattern_colors, counts):
        pattern_values.extend([color] * count)
    random.shuffle(pattern_values)

    for pair, color in zip(pairs, marker_colors):
        pair["color"] = color
        (r0, c0), (r1, c1) = pair["a"], pair["b"]
        grid[r0, c0] = color
        grid[r1, c1] = color

    for (r, c), color in zip(all_pattern_cells, pattern_values):
        grid[r, c] = int(color)

    return grid, all_pattern_cells, marker_colors


def _valid_generated(grid: np.ndarray, marker_colors: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    for color in marker_colors:
        coords = np.argwhere(grid == int(color))
        if coords.shape[0] != 2:
            return False
        if coords[0, 0] != coords[1, 0] and coords[0, 1] != coords[1, 1]:
            return False
    for color in set(map(int, np.unique(grid))) - {BLACK, *map(int, marker_colors)}:
        coords = np.argwhere(grid == color)
        if coords.shape[0] == 2 and (coords[0, 0] == coords[1, 0] or coords[0, 1] == coords[1, 1]):
            return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _sample_scene()
        if scene is None:
            continue
        grid, _pattern_cells, marker_colors = scene
        if _valid_generated(grid, marker_colors):
            return grid
    raise RuntimeError("failed to generate valid marker-strip symmetry input")
