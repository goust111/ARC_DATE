import random

import numpy as np


BG_COLOR = 0
DIVIDER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
LIBRARY_PATTERN_SIZE = 3
MAX_GENERATION_ATTEMPTS = 300
QUADRANTS = ("tl", "tr", "bl", "br")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _flip_signature(mask: np.ndarray) -> tuple[str, ...]:
    variants = (
        mask,
        np.fliplr(mask),
        np.flipud(mask),
        np.flipud(np.fliplr(mask)),
    )
    return min(_mask_key(variant) for variant in variants)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False

    h, w = mask.shape
    start_r, start_c = map(int, cells[0])
    stack = [(start_r, start_c)]
    seen = {(start_r, start_c)}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_pattern_groups() -> dict[tuple[str, ...], list[np.ndarray]]:
    groups: dict[tuple[str, ...], list[np.ndarray]] = {}
    for bits in range(1 << (LIBRARY_PATTERN_SIZE * LIBRARY_PATTERN_SIZE)):
        mask = np.array(
            [
                [(bits >> (r * LIBRARY_PATTERN_SIZE + c)) & 1 for c in range(LIBRARY_PATTERN_SIZE)]
                for r in range(LIBRARY_PATTERN_SIZE)
            ],
            dtype=bool,
        )
        if not mask[1, 1]:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not 4 <= int(mask.sum()) <= 6:
            continue
        if not _is_connected(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue

        signature = _flip_signature(mask)
        groups.setdefault(signature, []).append(mask)
    return groups


PATTERN_GROUPS = _build_pattern_groups()


def _find_split_lines(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and bool(np.all(grid[:, c] == grid[0, c]))
    ]

    matches = [
        (r, c, color_r)
        for r, color_r in row_candidates
        for c, color_c in col_candidates
        if color_r == color_c
    ]
    assert len(matches) == 1, "expected exactly one full grey row/column pair"
    return matches[0]


def _find_mono_components(grid: np.ndarray, bg: int, ignore_colors: set[int] | None = None) -> list[dict]:
    ignore = set() if ignore_colors is None else set(ignore_colors)
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == bg or color in ignore:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [coord[0] for coord in coords]
            cols = [coord[1] for coord in coords]
            components.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _extract_library(
    grid: np.ndarray,
    split_r: int,
    split_c: int,
    bg: int,
    divider: int,
) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int]]:
    top_left = grid[:split_r, :split_c]
    components = _find_mono_components(top_left, bg=bg, ignore_colors={divider})
    top_row_components = [comp for comp in components if comp["bbox"][0] == 0]
    top_row_components.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))

    library = []
    next_col = 0
    for comp in top_row_components:
        _r0, c0, _r1, c1 = comp["bbox"]
        if not library:
            if c0 != 0:
                continue
            library.append(comp)
            next_col = int(c1) + 2
            continue
        if c0 != next_col:
            if c0 > next_col:
                break
            continue
        library.append(comp)
        next_col = int(c1) + 2

    assert 2 <= len(library) <= 4, "expected a 2-4 pattern library in the top-left corner"

    patterns: dict[int, np.ndarray] = {}
    lib_r1 = 0
    lib_c1 = 0
    for comp in library:
        color = int(comp["color"])
        assert color not in patterns, "library colors must be unique"
        r0, c0, r1, c1 = comp["bbox"]
        patterns[color] = (top_left[r0 : r1 + 1, c0 : c1 + 1] == color).copy()
        lib_r1 = max(lib_r1, int(r1))
        lib_c1 = max(lib_c1, int(c1))

    return patterns, (0, 0, lib_r1, lib_c1)


def _orient_mask(mask: np.ndarray, seed_r: int, seed_c: int, split_r: int, split_c: int) -> np.ndarray:
    if seed_r < split_r and seed_c < split_c:
        return mask
    if seed_r < split_r and seed_c > split_c:
        return np.fliplr(mask)
    if seed_r > split_r and seed_c < split_c:
        return np.flipud(mask)
    return np.flipud(np.fliplr(mask))


def _stamp_mask(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    center_r: int,
    center_c: int,
    protected: np.ndarray,
) -> None:
    h, w = out.shape
    mh, mw = mask.shape
    anchor_r = mh // 2
    anchor_c = mw // 2
    for dr in range(mh):
        for dc in range(mw):
            if not mask[dr, dc]:
                continue
            rr = center_r + dr - anchor_r
            cc = center_c + dc - anchor_c
            if 0 <= rr < h and 0 <= cc < w and not protected[rr, cc]:
                out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each seed into its color-matched library sprite, with quadrant-specific flips.

    Input structure:
    - The background is the most common color, and one full grey row plus one full grey column split the grid into four quadrants.
    - The top-left corner contains a small left-to-right library of 2-4 monochrome sprites on the background, separated by one background column; each sprite's color is its lookup key.
    - Elsewhere in the quadrants, singleton seed pixels use those same key colors.

    Rule:
    - For each seed, select the library sprite with the same color.
    - Use the seed quadrant to choose the orientation: top-left = original, top-right = horizontal flip, bottom-left = vertical flip, bottom-right = both flips.
    - Erase the seeds, then stamp each oriented sprite so the center of its bounding box lands on the seed position.
    - Keep the library rectangle and the grey divider lines unchanged; if stamps overlap, process seeds by increasing column and then increasing row so later stamps overwrite earlier ones.

    Key insight:
    - The seed color chooses which sprite to copy, while the quadrant independently chooses how that sprite is flipped; the seed is only the placement anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    split_r, split_c, divider = _find_split_lines(g, bg)
    patterns, library_bbox = _extract_library(g, split_r, split_c, bg, divider)

    out = g.copy()
    protected = np.zeros(g.shape, dtype=bool)
    protected[split_r, :] = True
    protected[:, split_c] = True
    lr0, lc0, lr1, lc1 = library_bbox
    protected[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    seeds = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if protected[r, c] or color in (bg, divider):
                continue
            if color not in patterns:
                continue
            seeds.append((r, c, color))
            out[r, c] = bg

    for r, c, color in sorted(seeds, key=lambda item: (item[1], item[0])):
        oriented = _orient_mask(patterns[color], r, c, split_r, split_c)
        _stamp_mask(out, oriented, color, r, c, protected)

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_patterns(count: int) -> list[np.ndarray]:
    signatures = random.sample(list(PATTERN_GROUPS.keys()), count)
    return [random.choice(PATTERN_GROUPS[signature]).copy() for signature in signatures]


def _sample_seed_counts(pattern_count: int) -> list[int] | None:
    for _ in range(80):
        counts = [random.randint(0, 3) for _ in range(4)]
        total = sum(counts)
        if total < max(4, pattern_count):
            continue
        if total > 8:
            continue
        if sum(count > 0 for count in counts) < 3:
            continue
        return counts
    return None


def _candidate_centers(
    h: int,
    w: int,
    split_r: int,
    split_c: int,
    quadrant: str,
    library_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    if quadrant == "tl":
        row_range = range(1, split_r - 1)
        col_range = range(1, split_c - 1)
    elif quadrant == "tr":
        row_range = range(1, split_r - 1)
        col_range = range(split_c + 2, w - 1)
    elif quadrant == "bl":
        row_range = range(split_r + 2, h - 1)
        col_range = range(1, split_c - 1)
    else:
        row_range = range(split_r + 2, h - 1)
        col_range = range(split_c + 2, w - 1)

    candidates = []
    for r in row_range:
        for c in col_range:
            box = (r - 1, c - 1, r + 1, c + 1)
            if quadrant == "tl" and _boxes_overlap(box, library_bbox):
                continue
            candidates.append((r, c))
    return candidates


def _choose_seed_centers(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    for _ in range(40):
        trial = candidates[:]
        random.shuffle(trial)
        chosen = []
        boxes = []
        for r, c in trial:
            box = (r - 1, c - 1, r + 1, c + 1)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            chosen.append((r, c))
            boxes.append(box)
            if len(chosen) == count:
                return chosen
    return None


def _render_expected(
    grid: np.ndarray,
    split_r: int,
    split_c: int,
    patterns: dict[int, np.ndarray],
    library_bbox: tuple[int, int, int, int],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    protected = np.zeros(grid.shape, dtype=bool)
    protected[split_r, :] = True
    protected[:, split_c] = True
    lr0, lc0, lr1, lc1 = library_bbox
    protected[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    for r, c, _color in seeds:
        out[r, c] = BG_COLOR

    for r, c, color in sorted(seeds, key=lambda item: (item[1], item[0])):
        oriented = _orient_mask(patterns[color], r, c, split_r, split_c)
        _stamp_mask(out, oriented, color, r, c, protected)
    return out


def _gen_attempt() -> np.ndarray | None:
    pattern_count = random.randint(2, 4)
    pattern_masks = _sample_patterns(pattern_count)
    pattern_colors = random.sample(COLOR_POOL, pattern_count)
    library_patterns = {color: mask for color, mask in zip(pattern_colors, pattern_masks)}

    library_width = pattern_count * LIBRARY_PATTERN_SIZE + (pattern_count - 1)
    min_width = max(21, library_width + 14)
    if min_width > 30:
        return None

    h = random.randint(19, 30)
    w = random.randint(min_width, 30)
    split_r = random.randint(8, h - 8)
    split_c = random.randint(library_width + 6, w - 8)
    library_bbox = (0, 0, LIBRARY_PATTERN_SIZE - 1, library_width - 1)

    counts = _sample_seed_counts(pattern_count)
    if counts is None:
        return None

    centers_by_quadrant: dict[str, list[tuple[int, int]]] = {}
    for quadrant, count in zip(QUADRANTS, counts):
        candidates = _candidate_centers(h, w, split_r, split_c, quadrant, library_bbox)
        centers = _choose_seed_centers(candidates, count)
        if centers is None:
            return None
        centers_by_quadrant[quadrant] = centers

    seed_slots = []
    for quadrant, count in zip(QUADRANTS, counts):
        seed_slots.extend([quadrant] * count)
    random.shuffle(seed_slots)

    seed_colors = pattern_colors[:] + random.choices(pattern_colors, k=len(seed_slots) - pattern_count)
    random.shuffle(seed_colors)

    seeds = []
    for quadrant, color in zip(seed_slots, seed_colors):
        center = centers_by_quadrant[quadrant].pop()
        seeds.append((center[0], center[1], int(color)))

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[split_r, :] = DIVIDER_COLOR
    grid[:, split_c] = DIVIDER_COLOR

    for index, color in enumerate(pattern_colors):
        c0 = index * (LIBRARY_PATTERN_SIZE + 1)
        mask = library_patterns[color]
        grid[0:LIBRARY_PATTERN_SIZE, c0 : c0 + LIBRARY_PATTERN_SIZE][mask] = int(color)

    for r, c, color in seeds:
        if grid[r, c] != BG_COLOR:
            return None
        grid[r, c] = int(color)

    expected = _render_expected(grid, split_r, split_c, library_patterns, library_bbox, seeds)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
