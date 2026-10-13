from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 5
FRAGMENT_COLORS = tuple(color for color in range(10) if color not in {BG, FRAME})
GENERATION_ATTEMPTS = 300
REGION_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _transform_shape(
    shape: tuple[tuple[int, int], ...], rotation: int, mirror: bool
) -> tuple[tuple[int, int], ...]:
    points = list(shape)
    if mirror:
        points = [(r, -c) for r, c in points]
    for _ in range(rotation):
        points = [(c, -r) for r, c in points]
    return _normalize_shape(tuple(points))


def _build_shapes() -> list[tuple[tuple[int, int], ...]]:
    base_shapes = [
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
    ]
    unique: set[tuple[tuple[int, int], ...]] = set()
    for base in base_shapes:
        for rotation in range(4):
            for mirror in (False, True):
                unique.add(_transform_shape(base, rotation, mirror))

    weighted: list[tuple[tuple[int, int], ...]] = []
    for shape in sorted(unique):
        size = len(shape)
        weight = {1: 5, 2: 4, 3: 3, 4: 2}.get(size, 1)
        weighted.extend([shape] * weight)
    return weighted


SHAPES = _build_shapes()


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _count_color_fragments(region: np.ndarray) -> Counter[int]:
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    counts: Counter[int] = Counter()
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if color in {BG, FRAME} or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(region[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            counts[color] += 1
    return counts


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frame_components = _mask_components(grid == FRAME)
    assert frame_components, "expected at least one grey frame"
    bboxes: list[tuple[int, int, int, int]] = []
    for coords in frame_components:
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame must enclose a non-empty interior"

        actual = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            actual[r - r0, c - c0] = True

        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(actual, expected), "grey component must be a 1-pixel rectangular border"
        bboxes.append((r0, c0, r1, c1))
    return bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count fragment-color votes inside each grey rectangle, then repaint that rectangle as a solid block of the winning color.

    Input structure:
    - The background is Black.
    - One or more non-touching Grey 1-pixel-thick rectangular frames appear on the grid.
    - Inside each frame are several separated monochrome fragments in colors other
      than Black and Grey.
    - Each frame interior contains at least two fragment colors, and exactly one
      color has a strictly larger number of 4-connected fragments than every other color.

    Rule:
    - Detect every Grey rectangular frame and inspect only its enclosed interior.
    - Count interior fragments by 4-connected component, grouped by color rather
      than by pixel area.
    - For each frame, choose the unique color with the highest fragment count.
    - Recolor the frame border to that winning color and fill the entire interior
      of the frame with the same color, leaving the outside Black background unchanged.

    Key insight:
    - The deciding signal is the number of disconnected fragments of each color
      inside a closed frame; after counting those votes, the whole rectangle becomes
      one solid block in the winning color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    bboxes = _frame_bboxes(g)
    inside_any = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        inside_any[r0 + 1 : r1, c0 + 1 : c1] = True
    assert not np.any((g != BG) & (g != FRAME) & ~inside_any), "colored fragments must lie inside frames"

    out = np.full_like(g, BG)
    for r0, c0, r1, c1 in bboxes:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        counts = _count_color_fragments(interior)
        assert len(counts) >= 2, "each region needs at least two fragment colors"
        ranking = counts.most_common()
        assert len(ranking) == 1 or ranking[0][1] > ranking[1][1], "winner must be unique"
        winner = int(ranking[0][0])
        out[r0 : r1 + 1, c0 : c1 + 1] = winner
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _can_place_shape(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> bool:
    h, w = occupied.shape
    cells = {(top + dr, left + dc) for dr, dc in shape}
    for r, c in cells:
        if occupied[r, c]:
            return False
    for r, c in cells:
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in cells:
                return False
    return True


def _place_fragment(region: np.ndarray, occupied: np.ndarray, color: int) -> bool:
    h, w = region.shape
    candidates: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    for shape in SHAPES:
        sh, sw = _shape_dims(shape)
        if sh > h or sw > w:
            continue
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                if _can_place_shape(occupied, shape, top, left):
                    candidates.append((shape, top, left))
    if not candidates:
        return False
    shape, top, left = random.choice(candidates)
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        region[rr, cc] = color
        occupied[rr, cc] = True
    return True


def _distribute_extra(slots: int, total: int, cap: int) -> list[int]:
    extras = [0] * slots
    while total > 0:
        choices = [idx for idx, value in enumerate(extras) if value < cap]
        if not choices:
            break
        idx = random.choice(choices)
        extras[idx] += 1
        total -= 1
    return extras


def _sample_fragment_counts(max_fragments: int) -> list[int]:
    max_colors = min(4, max_fragments - 1)
    n_colors = random.randint(2, max_colors)
    winner_upper = max_fragments - (n_colors - 1)
    winner_count = random.randint(2, winner_upper)
    extras_budget = max_fragments - (winner_count + (n_colors - 1))
    extra_total = random.randint(0, extras_budget)
    extras = _distribute_extra(n_colors - 1, extra_total, winner_count - 2)
    return [winner_count] + [1 + extra for extra in extras]


def _build_region(inner_h: int, inner_w: int) -> np.ndarray | None:
    capacity = ((inner_h + 1) // 2) * ((inner_w + 1) // 2)
    max_fragments = min(8, capacity)
    if max_fragments < 3:
        return None

    for _ in range(REGION_ATTEMPTS):
        counts = _sample_fragment_counts(max_fragments)
        colors = random.sample(FRAGMENT_COLORS, len(counts))
        winner = colors[0]

        region = np.zeros((inner_h, inner_w), dtype=int)
        occupied = np.zeros((inner_h, inner_w), dtype=bool)
        fragment_colors: list[int] = []
        for color, count in zip(colors, counts):
            fragment_colors.extend([int(color)] * int(count))
        random.shuffle(fragment_colors)

        ok = True
        for color in fragment_colors:
            if not _place_fragment(region, occupied, color):
                ok = False
                break
        if not ok:
            continue

        counts_found = _count_color_fragments(region)
        ranking = counts_found.most_common()
        if (
            len(counts_found) == len(counts)
            and ranking[0][0] == winner
            and (len(ranking) == 1 or ranking[0][1] > ranking[1][1])
        ):
            return region
    return None


def _place_frames(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    max_area = (height * width) // max(2, count * 2)
    size_choices = [
        (outer_h, outer_w)
        for outer_h in range(6, min(12, height - 2) + 1)
        for outer_w in range(6, min(12, width - 2) + 1)
        if outer_h * outer_w <= max_area
    ]
    if not size_choices:
        return None

    reserved = np.zeros((height, width), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            outer_h, outer_w = random.choice(size_choices)
            r0 = random.randint(1, height - outer_h - 1)
            c0 = random.randint(1, width - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            pad_r0 = r0 - 1
            pad_c0 = c0 - 1
            pad_r1 = r1 + 1
            pad_c1 = c1 + 1
            if reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                continue
            reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
            boxes.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return boxes


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(14, 30)
    width = random.randint(14, 30)
    max_frames = min(4, max(1, (height * width) // 140 + 1))
    frame_count = random.randint(1, max_frames)

    boxes = _place_frames(height, width, frame_count)
    if boxes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for r0, c0, r1, c1 in boxes:
        _draw_frame(grid, r0, c0, r1, c1)
        region = _build_region(r1 - r0 - 1, c1 - c0 - 1)
        if region is None:
            return None
        grid[r0 + 1 : r1, c0 + 1 : c1] = region

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
