import numpy as np


GRAY = 5
GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (10, 22)
ANCHOR_COUNT_RANGE = (2, 5)
ANCHOR_MIN_SEPARATION = 4
MIN_TIE_CELLS = 4
MIN_REGION_CELLS = 6
TRUE_SEGMENT_COUNT_RANGE = (1, 3)
FALSE_SEGMENT_COUNT_RANGE = (1, 3)
SEGMENT_LENGTH_RANGE = (1, 5)
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _voronoi_partition(shape: tuple[int, int], anchors: list[tuple[int, int, int]]):
    h, w = shape
    coords = np.array([(r, c) for r, c, _ in anchors], dtype=int)
    colors = np.array([color for _, _, color in anchors], dtype=int)

    rr = np.arange(h, dtype=int)[:, None, None]
    cc = np.arange(w, dtype=int)[None, :, None]
    dist = np.abs(rr - coords[:, 0][None, None, :]) + np.abs(cc - coords[:, 1][None, None, :])

    winners = dist.argmin(axis=2)
    min_dist = dist.min(axis=2)
    tie = (dist == min_dist[:, :, None]).sum(axis=2) > 1
    fill = colors[winners]
    return fill, tie, winners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each cell by the nearest scattered anchor, keeping only the true equidistant separators.

    Input structure:
    - bg = most common color.
    - Several non-gray single-pixel anchors of distinct colors are scattered across the grid.
    - Some gray (5) pixels mark partial predicted boundaries between anchor regions.

    Rule:
    - Ignore the gray markers and compute Manhattan distance from every non-anchor cell to each anchor.
    - A uniquely closest anchor paints the cell with that anchor's color; cells tied between two or more anchors stay bg.
    - Gray marker cells stay gray only when they lie on the true tied boundary; otherwise they are overwritten by the winning anchor color.
    - Anchor pixels themselves remain unchanged.

    Key insight:
    - The gray pixels are only hypotheses about the separators, so the real answer comes from recomputing the full Manhattan Voronoi partition from the anchors alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    anchors = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != bg) & (g != GRAY))]
    assert anchors, "expected at least one anchor"

    fill, tie, _ = _voronoi_partition(g.shape, anchors)
    out = fill.copy()
    out[tie] = bg
    out[tie & (g == GRAY)] = GRAY

    for r, c, color in anchors:
        out[r, c] = color

    return out


def _sample_anchor_positions(h: int, w: int, count: int, rng: np.random.Generator):
    positions: list[tuple[int, int]] = []
    for _ in range(80):
        if rng.random() < 0.5:
            spans = [span for span in range(4, w) if span % 2 == 0]
            if not spans:
                continue
            span = int(rng.choice(spans))
            row = int(rng.integers(0, h))
            col0 = int(rng.integers(0, w - span))
            positions = [(row, col0), (row, col0 + span)]
        else:
            spans = [span for span in range(4, h) if span % 2 == 0]
            if not spans:
                continue
            span = int(rng.choice(spans))
            col = int(rng.integers(0, w))
            row0 = int(rng.integers(0, h - span))
            positions = [(row0, col), (row0 + span, col)]
        break

    if len(positions) != 2:
        return None

    attempts = 0
    while len(positions) < count and attempts < 300:
        attempts += 1
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if any(abs(r - pr) + abs(c - pc) < ANCHOR_MIN_SEPARATION for pr, pc in positions):
            continue
        positions.append((r, c))

    if len(positions) != count:
        return None
    return positions


def _random_walk(mask: np.ndarray, occupied: np.ndarray, target_len: int, rng: np.random.Generator):
    h, w = mask.shape
    available = np.argwhere(mask & ~occupied)
    if len(available) == 0:
        return []

    start = tuple(map(int, available[int(rng.integers(len(available)))]))
    path = [start]
    used = {start}
    prev_dir = None

    while len(path) < target_len:
        r, c = path[-1]
        neighbors = []
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not mask[nr, nc] or occupied[nr, nc] or (nr, nc) in used:
                continue
            score = 1 if prev_dir == (dr, dc) else 0
            neighbors.append((score, nr, nc, dr, dc))

        if not neighbors:
            break

        best_score = max(score for score, *_ in neighbors)
        choices = [item for item in neighbors if item[0] == best_score]
        _, nr, nc, dr, dc = choices[int(rng.integers(len(choices)))]
        path.append((nr, nc))
        used.add((nr, nc))
        prev_dir = (dr, dc)

    return path


def _dilate_tie_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in range(-radius, radius + 1):
            rem = radius - abs(dr)
            for dc in range(-rem, rem + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _place_boundary_markers(
    tie: np.ndarray,
    winners: np.ndarray,
    anchor_mask: np.ndarray,
    rng: np.random.Generator,
):
    h, w = tie.shape
    occupied = anchor_mask.copy()

    true_gray = np.zeros((h, w), dtype=bool)
    true_available = tie & ~anchor_mask
    num_true_segments = int(rng.integers(TRUE_SEGMENT_COUNT_RANGE[0], TRUE_SEGMENT_COUNT_RANGE[1] + 1))
    for _ in range(num_true_segments):
        remaining = int(np.count_nonzero(true_available & ~occupied))
        if remaining <= 1:
            break
        max_len = min(SEGMENT_LENGTH_RANGE[1], remaining - 1)
        if max_len < SEGMENT_LENGTH_RANGE[0]:
            break
        target_len = int(rng.integers(SEGMENT_LENGTH_RANGE[0], max_len + 1))
        path = _random_walk(true_available, occupied, target_len, rng)
        for r, c in path:
            occupied[r, c] = True
            true_gray[r, c] = True

    leftover_true_boundary = tie & ~anchor_mask & ~true_gray
    if not np.any(true_gray) or not np.any(leftover_true_boundary):
        return None

    candidate_false = _dilate_tie_mask(tie, radius=int(rng.integers(1, 3))) & ~tie & ~anchor_mask
    if not np.any(candidate_false):
        return None

    false_gray = np.zeros((h, w), dtype=bool)
    num_false_segments = int(rng.integers(FALSE_SEGMENT_COUNT_RANGE[0], FALSE_SEGMENT_COUNT_RANGE[1] + 1))
    for _ in range(num_false_segments):
        starts = np.argwhere(candidate_false & ~occupied)
        if len(starts) == 0:
            break
        start = tuple(map(int, starts[int(rng.integers(len(starts)))]))
        label = int(winners[start])
        same_region = candidate_false & (winners == label)
        target_len = int(rng.integers(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1))
        path = _random_walk(same_region, occupied, target_len, rng)
        for r, c in path:
            occupied[r, c] = True
            false_gray[r, c] = True

    if not np.any(false_gray):
        return None

    return true_gray | false_gray


def _gen_attempt(rng: np.random.Generator):
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    anchor_count = int(rng.integers(ANCHOR_COUNT_RANGE[0], ANCHOR_COUNT_RANGE[1] + 1))

    palette = [color for color in range(10) if color != GRAY]
    bg = int(rng.choice(palette))
    anchor_colors = list(map(int, rng.choice([c for c in palette if c != bg], size=anchor_count, replace=False)))

    positions = _sample_anchor_positions(h, w, anchor_count, rng)
    if positions is None:
        return None

    anchors = [(r, c, color) for (r, c), color in zip(positions, anchor_colors)]
    fill, tie, winners = _voronoi_partition((h, w), anchors)

    anchor_mask = np.zeros((h, w), dtype=bool)
    for r, c, _ in anchors:
        anchor_mask[r, c] = True

    unique_cells = ~tie
    region_sizes = np.bincount(winners[unique_cells].ravel(), minlength=anchor_count)
    if np.any(region_sizes < MIN_REGION_CELLS):
        return None

    if int(np.count_nonzero(tie & ~anchor_mask)) < MIN_TIE_CELLS:
        return None

    gray_mask = _place_boundary_markers(tie, winners, anchor_mask, rng)
    if gray_mask is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[gray_mask] = GRAY
    for r, c, color in anchors:
        grid[r, c] = color

    if _most_common_color(grid) != bg:
        return None
    if not np.any((grid == bg) & tie):
        return None
    if not np.any((grid == GRAY) & tie):
        return None
    if not np.any((grid == GRAY) & ~tie):
        return None
    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(np.random.default_rng())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
