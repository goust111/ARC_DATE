import random

import numpy as np


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator_lines(grid: np.ndarray) -> tuple[int, int, int]:
    row_candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    col_candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]

    matches = []
    for r in row_candidates:
        for c in col_candidates:
            if int(grid[r, 0]) == int(grid[0, c]):
                matches.append((r, c, int(grid[r, 0])))

    assert len(matches) == 1, "expected exactly one full-width and one full-height separator line"
    return matches[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_component(coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    rel = [(r - r0, c - c0) for r, c in coords]
    anchor = min(rel)
    return {
        "shape": (r1 - r0 + 1, c1 - c0 + 1),
        "coords": rel,
        "anchor": anchor,
    }


def _transform_rel_coords(
    coords: list[tuple[int, int]],
    anchor: tuple[int, int],
    shape: tuple[int, int],
    mode: str,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = shape

    def transform(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        if mode == "h":
            return (r, w - 1 - c)
        if mode == "v":
            return (h - 1 - r, c)
        if mode == "hv":
            return (h - 1 - r, w - 1 - c)
        if mode == "id":
            return (r, c)
        raise AssertionError(f"unknown mode {mode}")

    return [transform(cell) for cell in coords], transform(anchor)


def _extract_palette_patterns(
    grid: np.ndarray,
    split_r: int,
    split_c: int,
    bg: int,
    line_color: int,
) -> dict[int, dict]:
    palette = grid[:split_r, :split_c]
    mask = (palette != bg) & (palette != line_color)
    comps = _components(mask)
    assert comps, "expected at least one palette object"

    patterns: dict[int, dict] = {}
    for comp in comps:
        colors = {int(palette[r, c]) for r, c in comp}
        assert len(colors) == 1, "palette objects must be monochrome"
        color = colors.pop()
        assert color not in patterns, "palette colors must be unique"
        patterns[color] = _normalize_component(comp)
    return patterns


def _seed_mode(r: int, c: int, split_r: int, split_c: int) -> str:
    if r < split_r and c > split_c:
        return "h"
    if r > split_r and c < split_c:
        return "v"
    if r > split_r and c > split_c:
        return "hv"
    raise AssertionError("seed must be in one of the three canvas quadrants")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each palette sprite onto a seed pixel using the flip implied by the seed's quadrant.

    Input structure:
    - The background is a mostly black canvas split by one full horizontal separator line and one full vertical separator line of the same color.
    - The top-left quadrant is a palette area containing several small 4-connected monochrome sprites, one sprite per color.
    - The other three quadrants contain isolated single-pixel seeds whose colors match palette sprites.

    Rule:
    - Identify the palette sprite for each seed by matching color.
    - If the seed is in the top-right quadrant, mirror the sprite left-right; if it is in the bottom-left quadrant, mirror it up-down; if it is in the bottom-right quadrant, apply both mirrors.
    - Stamp the transformed sprite so that the seed lands on the transformed position corresponding to the sprite's original top-left pixel; keep the palette area and separator lines unchanged.

    Key insight:
    - The color chooses the template, while the seed's quadrant independently chooses which reflection of that template to draw.
    """
    g = np.asarray(grid, dtype=int)
    split_r, split_c, line_color = _find_separator_lines(g)
    bg = _most_common_color(g[g != line_color])
    patterns = _extract_palette_patterns(g, split_r, split_c, bg, line_color)

    out = g.copy()
    seed_mask = (g != bg) & (g != line_color)
    seed_mask[:split_r, :split_c] = False
    seed_mask[split_r, :] = False
    seed_mask[:, split_c] = False

    for r, c in np.argwhere(seed_mask):
        color = int(g[r, c])
        assert color in patterns, "every seed color must exist in the palette"
        pattern = patterns[color]
        coords, anchor = _transform_rel_coords(
            pattern["coords"],
            pattern["anchor"],
            pattern["shape"],
            _seed_mode(int(r), int(c), split_r, split_c),
        )
        for pr, pc in coords:
            rr = int(r) + pr - anchor[0]
            cc = int(c) + pc - anchor[1]
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            out[rr, cc] = color

    return out


# --------------- generator ---------------

BACKGROUND = 0
LINE_COLOR = 2
COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]

TOP_LEFT_RANGE = (7, 12)
BOTTOM_RANGE = (7, 12)
RIGHT_RANGE = (7, 12)
PATTERN_COUNT_RANGE = (2, 4)
SEEDS_PER_QUADRANT = (1, 2)
MAX_GEN_ATTEMPTS = 300


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _build_mask_library() -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_masks: list[np.ndarray] = []
    asym_masks: list[np.ndarray] = []
    seen = set()

    for bits in range(1, 1 << 9):
        arr = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                arr[idx // 3, idx % 3] = True
        if not _is_connected(arr):
            continue
        if not (2 <= int(arr.sum()) <= 6):
            continue

        rows, cols = np.where(arr)
        cropped = arr[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        key = (cropped.shape, tuple(int(v) for v in cropped.ravel()))
        if key in seen:
            continue
        seen.add(key)
        all_masks.append(cropped.copy())
        if not np.array_equal(cropped, np.fliplr(cropped)) and not np.array_equal(cropped, np.flipud(cropped)):
            asym_masks.append(cropped.copy())

    return all_masks, asym_masks


MASK_LIBRARY, ASYMMETRIC_MASKS = _build_mask_library()


def _render_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = color


def _place_palette_masks(region_h: int, region_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((region_h, region_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda idx: (-(masks[idx].shape[0] * masks[idx].shape[1]), -int(masks[idx].sum())))

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        candidates = []
        for top in range(region_h - mh + 1):
            for left in range(region_w - mw + 1):
                r0 = max(0, top - 1)
                r1 = min(region_h, top + mh + 1)
                c0 = max(0, left - 1)
                c1 = min(region_w, left + mw + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        placements[idx] = random.choice(candidates)
        top, left = placements[idx]
        occupied[top : top + mh, left : left + mw] = True

    return [(int(top), int(left)) for top, left in placements]


def _quadrant_bounds(mode: str, split_r: int, split_c: int, h: int, w: int) -> tuple[int, int, int, int]:
    if mode == "h":
        return (0, split_r - 1, split_c + 1, w - 1)
    if mode == "v":
        return (split_r + 1, h - 1, 0, split_c - 1)
    if mode == "hv":
        return (split_r + 1, h - 1, split_c + 1, w - 1)
    raise AssertionError(f"unknown quadrant mode {mode}")


def _stamp_expected(
    expected: np.ndarray,
    seed_r: int,
    seed_c: int,
    color: int,
    pattern: dict,
    mode: str,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    coords, anchor = _transform_rel_coords(pattern["coords"], pattern["anchor"], pattern["shape"], mode)
    abs_coords = [(seed_r + r - anchor[0], seed_c + c - anchor[1]) for r, c in coords]
    rows = [r for r, _ in abs_coords]
    cols = [c for _, c in abs_coords]
    for r, c in abs_coords:
        expected[r, c] = color
    return abs_coords, (min(rows), min(cols), max(rows), max(cols))


def _place_seeds_for_mode(
    grid: np.ndarray,
    expected: np.ndarray,
    reserved_boxes: np.ndarray,
    blocked_seeds: np.ndarray,
    split_r: int,
    split_c: int,
    mode: str,
    colors: list[int],
    patterns_by_color: dict[int, dict],
) -> bool:
    h, w = grid.shape
    rq0, rq1, cq0, cq1 = _quadrant_bounds(mode, split_r, split_c, h, w)
    requests = sorted(
        colors,
        key=lambda color: (
            -(patterns_by_color[color]["shape"][0] * patterns_by_color[color]["shape"][1]),
            -len(patterns_by_color[color]["coords"]),
        ),
    )

    for color in requests:
        pattern = patterns_by_color[color]
        coords, anchor = _transform_rel_coords(pattern["coords"], pattern["anchor"], pattern["shape"], mode)
        deltas = [(r - anchor[0], c - anchor[1]) for r, c in coords]
        min_dr = min(dr for dr, _ in deltas)
        max_dr = max(dr for dr, _ in deltas)
        min_dc = min(dc for _, dc in deltas)
        max_dc = max(dc for _, dc in deltas)

        row_lo = rq0 - min_dr
        row_hi = rq1 - max_dr
        col_lo = cq0 - min_dc
        col_hi = cq1 - max_dc
        if row_lo > row_hi or col_lo > col_hi:
            return False

        candidates = []
        for seed_r in range(row_lo, row_hi + 1):
            for seed_c in range(col_lo, col_hi + 1):
                if blocked_seeds[seed_r, seed_c]:
                    continue
                abs_coords = [(seed_r + dr, seed_c + dc) for dr, dc in deltas]
                rows = [r for r, _ in abs_coords]
                cols = [c for _, c in abs_coords]
                br0, bc0, br1, bc1 = min(rows), min(cols), max(rows), max(cols)
                rr0 = max(0, br0 - 1)
                rr1 = min(h, br1 + 2)
                cc0 = max(0, bc0 - 1)
                cc1 = min(w, bc1 + 2)
                if reserved_boxes[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((seed_r, seed_c))

        if not candidates:
            return False

        seed_r, seed_c = random.choice(candidates)
        grid[seed_r, seed_c] = color
        _, (br0, bc0, br1, bc1) = _stamp_expected(expected, seed_r, seed_c, color, pattern, mode)
        reserved_boxes[br0 : br1 + 1, bc0 : bc1 + 1] = True
        blocked_seeds[max(0, seed_r - 1) : min(h, seed_r + 2), max(0, seed_c - 1) : min(w, seed_c + 2)] = True

    return True


def _gen_attempt() -> np.ndarray | None:
    split_r = random.randint(*TOP_LEFT_RANGE)
    bottom_h = random.randint(*BOTTOM_RANGE)
    split_c = random.randint(*TOP_LEFT_RANGE)
    right_w = random.randint(*RIGHT_RANGE)
    h = split_r + 1 + bottom_h
    w = split_c + 1 + right_w
    if h > 30 or w > 30:
        return None

    num_patterns = random.randint(*PATTERN_COUNT_RANGE)
    if len(COLOR_POOL) < num_patterns or not ASYMMETRIC_MASKS:
        return None

    probe_mask = random.choice(ASYMMETRIC_MASKS)
    mask_choices = [mask for mask in MASK_LIBRARY if not np.array_equal(mask, probe_mask)]
    if len(mask_choices) < num_patterns - 1:
        return None

    masks = [probe_mask] + random.sample(mask_choices, num_patterns - 1)
    placements = _place_palette_masks(split_r, split_c, masks)
    if placements is None:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[split_r, :] = LINE_COLOR
    grid[:, split_c] = LINE_COLOR
    expected = grid.copy()

    palette_colors = random.sample(COLOR_POOL, num_patterns)
    patterns_by_color: dict[int, dict] = {}

    for color, mask, (top, left) in zip(palette_colors, masks, placements):
        _render_mask(grid, mask, top, left, color)
        _render_mask(expected, mask, top, left, color)
        rows, cols = np.where(mask)
        coords = list(zip(rows.tolist(), cols.tolist()))
        patterns_by_color[color] = _normalize_component(coords)

    probe_color = palette_colors[0]
    seeds_per_quadrant = random.randint(*SEEDS_PER_QUADRANT)
    extras = palette_colors[1:].copy()
    random.shuffle(extras)

    seeds_by_mode = {
        "h": [probe_color],
        "v": [probe_color],
        "hv": [probe_color],
    }
    if seeds_per_quadrant == 2:
        while len(extras) < 3:
            extras.append(random.choice(palette_colors))
        random.shuffle(extras)
        seeds_by_mode["h"].append(extras[0])
        seeds_by_mode["v"].append(extras[1])
        seeds_by_mode["hv"].append(extras[2])

    reserved_boxes = np.zeros((h, w), dtype=bool)
    blocked_seeds = np.zeros((h, w), dtype=bool)
    for mode in ("h", "v", "hv"):
        if not _place_seeds_for_mode(
            grid,
            expected,
            reserved_boxes,
            blocked_seeds,
            split_r,
            split_c,
            mode,
            seeds_by_mode[mode],
            patterns_by_color,
        ):
            return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
