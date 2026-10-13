import numpy as np


BACKGROUND = 0
TARGET = 2
DIVIDER = 8
SEED_COLORS = (1, 3, 4, 5, 6, 7, 9)
CARDINALS = ("up", "right", "down", "left")
MAX_GENERATION_ATTEMPTS = 300

SEED_PATTERNS = (
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
)

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate-copy each region's asymmetric seed onto its red targets according to target direction.

    Input structure:
    - Black background fills the canvas.
    - Complete azure rows and complete azure columns partition the grid into rectangular regions.
    - Every region contains exactly one small connected monochrome seed pattern and one to three isolated red target pixels.
    - The seed pattern is asymmetric under 90-degree rotation, and the red targets are separated from it.

    Rule:
    - For each region, find the seed pattern and its bounding-box center.
    - For each red target, compare its row and column offsets from that center; the larger-magnitude offset gives the target's main direction.
    - Use the direction-to-rotation map: up = 0 degrees, right = 90 degrees clockwise, down = 180 degrees, left = 270 degrees clockwise.
    - Copy the seed pattern with that rotation and place the rotated copy so the target pixel is at the copied pattern's bounding-box center; the original seed stays in place and the red target is overwritten.

    Key insight:
    - Each target encodes two things at once: its location tells where to stamp the seed, and its direction from the seed center tells how much to rotate it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    divider = _find_divider_color(g)
    row_spans = _spans(g.shape[0], [r for r in range(g.shape[0]) if np.all(g[r, :] == divider)])
    col_spans = _spans(g.shape[1], [c for c in range(g.shape[1]) if np.all(g[:, c] == divider)])
    assert row_spans and col_spans

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            seed_color, seed_cells, targets = _parse_region(region)
            seed_offsets2, center2 = _seed_offsets2(seed_cells)

            for tr, tc in targets:
                out[r0 + tr, c0 + tc] = BACKGROUND
                turns = _direction_to_turns(_main_direction((tr, tc), center2))
                for dr2, dc2 in _rotate_offsets2(seed_offsets2, turns):
                    rr2 = 2 * tr + dr2
                    cc2 = 2 * tc + dc2
                    assert rr2 % 2 == 0 and cc2 % 2 == 0
                    rr = rr2 // 2
                    cc = cc2 // 2
                    assert 0 <= rr < region.shape[0] and 0 <= cc < region.shape[1]
                    out[r0 + rr, c0 + cc] = seed_color

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")


def _find_divider_color(grid: np.ndarray) -> int:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == BACKGROUND:
            continue
        has_row = any(np.all(grid[r, :] == color) for r in range(grid.shape[0]))
        has_col = any(np.all(grid[:, c] == color) for c in range(grid.shape[1]))
        if has_row and has_col:
            candidates.append(color)
    assert len(candidates) == 1
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    points = [-1] + sorted(separators) + [length]
    spans = []
    for left, right in zip(points, points[1:]):
        if right - left > 1:
            spans.append((left + 1, right - 1))
    return spans


def _parse_region(region: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    colors = {int(color) for color in np.unique(region)}
    colors.discard(BACKGROUND)
    colors.discard(TARGET)
    assert len(colors) == 1
    seed_color = colors.pop()

    seed_mask = region == seed_color
    components = _components(seed_mask)
    assert len(components) == 1
    seed_cells = components[0]
    assert 3 <= len(seed_cells) <= 5

    targets = np.argwhere(region == TARGET)
    assert 1 <= len(targets) <= 3
    for tr, tc in targets:
        for sr, sc in seed_cells:
            assert max(abs(int(tr) - int(sr)), abs(int(tc) - int(sc))) > 1
    return seed_color, seed_cells, targets


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    h, w = mask.shape
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(sorted(cells), dtype=int))
    return parts


def _seed_offsets2(seed_cells: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rows = seed_cells[:, 0]
    cols = seed_cells[:, 1]
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    center2 = (rmin + rmax, cmin + cmax)
    offsets2 = []
    for r, c in seed_cells:
        offsets2.append((2 * int(r) - center2[0], 2 * int(c) - center2[1]))
    return offsets2, center2


def _main_direction(target: tuple[int, int], center2: tuple[int, int]) -> str:
    dr2 = 2 * int(target[0]) - center2[0]
    dc2 = 2 * int(target[1]) - center2[1]
    if abs(dr2) > abs(dc2):
        return "up" if dr2 < 0 else "down"
    return "left" if dc2 < 0 else "right"


def _direction_to_turns(direction: str) -> int:
    return {"up": 0, "right": 1, "down": 2, "left": 3}[direction]


def _rotate_offsets2(offsets2: list[tuple[int, int]], turns_cw: int) -> list[tuple[int, int]]:
    rotated = []
    for dr2, dc2 in offsets2:
        rr2, cc2 = dr2, dc2
        for _ in range(turns_cw % 4):
            rr2, cc2 = cc2, -rr2
        rotated.append((rr2, cc2))
    return rotated


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_region_count = int(rng.integers(2, 4))
    col_region_count = int(rng.integers(2, 4))
    row_sizes = _region_sizes(rng, row_region_count)
    col_sizes = _region_sizes(rng, col_region_count)

    height = sum(row_sizes) + row_region_count - 1
    width = sum(col_sizes) + col_region_count - 1
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    row_spans = _layout_spans(row_sizes)
    col_spans = _layout_spans(col_sizes)

    for i in range(1, row_region_count):
        sep_row = row_spans[i - 1][1] + 1
        grid[sep_row, :] = DIVIDER
    for j in range(1, col_region_count):
        sep_col = col_spans[j - 1][1] + 1
        grid[:, sep_col] = DIVIDER

    region_count = row_region_count * col_region_count
    target_counts = rng.integers(1, 4, size=region_count)
    region_dirs = _assign_directions(rng, target_counts)

    region_index = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region_h = r1 - r0 + 1
            region_w = c1 - c0 + 1
            region = _build_region(rng, region_h, region_w, region_dirs[region_index])
            if region is None:
                return None
            grid[r0 : r1 + 1, c0 : c1 + 1] = region
            region_index += 1

    return grid


def _region_sizes(rng: np.random.Generator, count: int) -> list[int]:
    sizes = [9] * count
    max_total = 30 - (count - 1)
    remaining = max_total - sum(sizes)
    indices = list(rng.permutation(count))
    for idx in indices:
        extra_cap = min(4, remaining)
        if extra_cap <= 0:
            break
        extra = int(rng.integers(0, extra_cap + 1))
        sizes[idx] += extra
        remaining -= extra
    return sizes


def _layout_spans(sizes: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for i, size in enumerate(sizes):
        spans.append((start, start + size - 1))
        start += size
        if i != len(sizes) - 1:
            start += 1
    return spans


def _assign_directions(rng: np.random.Generator, target_counts: np.ndarray) -> list[list[str]]:
    region_count = len(target_counts)
    assignments = [[] for _ in range(region_count)]
    cover_regions = list(rng.permutation(region_count)[:4])
    for direction, region_idx in zip(CARDINALS, cover_regions):
        assignments[region_idx].append(direction)

    for region_idx, count in enumerate(target_counts):
        while len(assignments[region_idx]) < int(count):
            choices = [direction for direction in CARDINALS if direction not in assignments[region_idx]]
            assignments[region_idx].append(str(rng.choice(choices)))
        rng.shuffle(assignments[region_idx])
    return assignments


def _build_region(rng: np.random.Generator, height: int, width: int, directions: list[str]) -> np.ndarray | None:
    region = np.full((height, width), BACKGROUND, dtype=int)
    seed_color = int(rng.choice(SEED_COLORS))

    seed_top = int(rng.integers(3, height - 5))
    seed_left = int(rng.integers(3, width - 5))
    pattern = list(SEED_PATTERNS[int(rng.integers(len(SEED_PATTERNS)))])
    pattern = _rotate_pattern(pattern, int(rng.integers(4)))
    seed_cells = [(seed_top + r, seed_left + c) for r, c in pattern]

    for r, c in seed_cells:
        region[r, c] = seed_color

    candidates = _target_candidates(height, width, seed_cells)
    chosen = []
    for direction in directions:
        options = list(candidates[direction])
        if not options:
            return None
        perm = rng.permutation(len(options))
        selected = None
        for idx in perm:
            pos = options[int(idx)]
            if all(not _boxes_overlap(pos, other) for other in chosen):
                selected = pos
                break
        if selected is None:
            return None
        chosen.append(selected)

    for r, c in chosen:
        region[r, c] = TARGET
    return region


def _rotate_pattern(pattern: list[tuple[int, int]], turns_cw: int) -> list[tuple[int, int]]:
    rotated = []
    for r, c in pattern:
        dr, dc = r - 1, c - 1
        for _ in range(turns_cw % 4):
            dr, dc = dc, -dr
        rotated.append((dr + 1, dc + 1))
    return rotated


def _target_candidates(height: int, width: int, seed_cells: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    rows = [r for r, _ in seed_cells]
    cols = [c for _, c in seed_cells]
    center_r = (min(rows) + max(rows)) // 2
    center_c = (min(cols) + max(cols)) // 2
    seed_set = set(seed_cells)
    candidates = {direction: [] for direction in CARDINALS}

    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if (r, c) in seed_set:
                continue
            if any(max(abs(r - sr), abs(c - sc)) <= 1 for sr, sc in seed_cells):
                continue
            dr = r - center_r
            dc = c - center_c
            if dr == 0 and dc == 0:
                continue
            if abs(dr) == abs(dc):
                continue
            if abs(dr) > abs(dc):
                direction = "up" if dr < 0 else "down"
            else:
                direction = "left" if dc < 0 else "right"
            candidates[direction].append((r, c))

    return candidates


def _boxes_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2
