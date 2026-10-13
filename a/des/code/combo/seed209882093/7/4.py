from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
ARROW = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_PATCHES = {
    "up": np.array(
        [
            [ARROW, ARROW, ARROW],
            [BG, ARROW, BG],
            [BG, ARROW, BG],
        ],
        dtype=int,
    ),
    "down": np.array(
        [
            [BG, ARROW, BG],
            [BG, ARROW, BG],
            [ARROW, ARROW, ARROW],
        ],
        dtype=int,
    ),
    "left": np.array(
        [
            [ARROW, BG, BG],
            [ARROW, ARROW, ARROW],
            [ARROW, BG, BG],
        ],
        dtype=int,
    ),
    "right": np.array(
        [
            [BG, BG, ARROW],
            [ARROW, ARROW, ARROW],
            [BG, BG, ARROW],
        ],
        dtype=int,
    ),
}


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave non-empty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final non-empty region"
    spans.append((start, length - 1))
    return spans


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_coords(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty foreground object"
    target = {tuple(map(int, coord)) for coord in coords}
    stack = [next(iter(target))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in target and (nr, nc) not in seen:
                stack.append((nr, nc))
    assert seen == target, "expected one 4-connected foreground object per region"
    return coords.astype(int, copy=False)


def _arrow_direction(local_coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(local_coords)
    scores = {
        (-1, 0): int(np.sum(local_coords[:, 0] == r0)),
        (1, 0): int(np.sum(local_coords[:, 0] == r1)),
        (0, -1): int(np.sum(local_coords[:, 1] == c0)),
        (0, 1): int(np.sum(local_coords[:, 1] == c1)),
    }
    best = max(scores.values())
    winners = [direction for direction, score in scores.items() if score == best]
    assert len(winners) == 1 and best >= 3, "maroon component must form a cardinal arrow"
    return winners[0]


def _parse_region(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> dict:
    r0, c0, r1, c1 = bounds
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    fg_mask = (patch != BG) & (patch != SEP)
    assert bool(np.any(fg_mask)), "each region must contain one object"

    colors = {int(v) for v in np.unique(patch[fg_mask])}
    local_coords = _connected_coords(fg_mask)
    lr0, lc0, lr1, lc1 = _bbox(local_coords)
    assert lr0 > 0 and lc0 > 0 and lr1 < patch.shape[0] - 1 and lc1 < patch.shape[1] - 1

    abs_coords = local_coords + np.array([r0, c0], dtype=int)
    info = {
        "bounds": bounds,
        "coords": abs_coords.astype(int, copy=False),
        "bbox": (lr0 + r0, lc0 + c0, lr1 + r0, lc1 + c0),
    }
    if colors == {ARROW}:
        info["kind"] = "arrow"
        info["direction"] = _arrow_direction(local_coords)
        return info

    assert ARROW not in colors, "pattern colors must differ from the arrow color"
    assert len(colors) == 1, "each pattern region must contain one monochrome object"
    info["kind"] = "pattern"
    info["color"] = next(iter(colors))
    return info


def _shift_for_destination(
    pattern_bbox: tuple[int, int, int, int],
    arrow_bounds: tuple[int, int, int, int],
    move_direction: tuple[int, int],
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = pattern_bbox
    ar0, ac0, ar1, ac1 = arrow_bounds
    dr, dc = move_direction
    if dr == -1:
        return ar0 + 1 - pr0, 0
    if dr == 1:
        return ar1 - 1 - pr1, 0
    if dc == -1:
        return 0, ac0 + 1 - pc0
    assert dc == 1
    return 0, ac1 - 1 - pc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each region pattern into its adjacent arrow region in the direction opposite the arrow.

    Input structure
    - black background
    - full grey separator rows/columns partition the canvas into rectangular regions
    - each region contains either one interior monochrome pattern or one interior maroon arrow
    - every arrow points from its own region toward the adjacent pattern region it controls

    Rule
    - pair each arrow with the adjacent pattern region that lies in the arrow's pointing direction
    - reverse the arrow direction to get the actual movement direction
    - translate the whole pattern straight across the shared separator into the arrow region
    - place the moved pattern one cell away from the far boundary in that reversed direction
    - erase the original pattern and arrow; keep grey separators unchanged
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert len(row_spans) * len(col_spans) >= 2, "expected a partitioned grid"

    regions: dict[tuple[int, int], dict] = {}
    arrows: list[tuple[int, int]] = []
    patterns: list[tuple[int, int]] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            info = _parse_region(g, (r0, c0, r1, c1))
            regions[(ri, ci)] = info
            if info["kind"] == "arrow":
                arrows.append((ri, ci))
            else:
                patterns.append((ri, ci))

    assert arrows and patterns and len(arrows) == len(patterns)

    incoming: dict[tuple[int, int], list[tuple[int, int]]] = {}
    assignments: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for arrow_idx in arrows:
        ri, ci = arrow_idx
        dr, dc = regions[arrow_idx]["direction"]
        pattern_idx = (ri + dr, ci + dc)
        assert pattern_idx in regions, "arrow must point to an adjacent region"
        assert regions[pattern_idx]["kind"] == "pattern", "arrow must point to a pattern region"
        incoming.setdefault(pattern_idx, []).append(arrow_idx)
        assignments.append((pattern_idx, arrow_idx, (-dr, -dc)))

    assert all(len(srcs) == 1 for srcs in incoming.values()), "each pattern must be targeted by one arrow"
    assert set(incoming) == set(patterns), "every pattern region must have a unique arrow neighbor"

    out = g.copy()
    for info in regions.values():
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for pattern_idx, arrow_idx, move_direction in assignments:
        pattern = regions[pattern_idx]
        arrow = regions[arrow_idx]
        shift_r, shift_c = _shift_for_destination(pattern["bbox"], arrow["bounds"], move_direction)
        shifted = pattern["coords"] + np.array([shift_r, shift_c], dtype=int)
        ar0, ac0, ar1, ac1 = arrow["bounds"]
        assert bool(
            np.all(
                (shifted[:, 0] > ar0)
                & (shifted[:, 0] < ar1)
                & (shifted[:, 1] > ac0)
                & (shifted[:, 1] < ac1)
            )
        ), "moved pattern must stay strictly inside the arrow region"
        assert bool(np.all(out[shifted[:, 0], shifted[:, 1]] == BG)), "moved patterns must not collide"
        out[shifted[:, 0], shifted[:, 1]] = int(pattern["color"])

    return out


# --------------- generator ---------------

REGION_SHAPES = (
    (1, 2),
    (1, 4),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 4),
)
ROW_HEIGHT_RANGE = (5, 8)
COL_WIDTH_RANGE = (5, 8)
MAX_SHAPE_SPAN = 4
GEN_RETRIES = 400


def _region_bounds_from_sizes(
    row_heights: list[int],
    col_widths: list[int],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(col_widths) + len(col_widths) - 1
    grid = np.full((height, width), BG, dtype=int)

    row_spans: list[tuple[int, int]] = []
    col_spans: list[tuple[int, int]] = []

    cursor = 0
    for i, size in enumerate(row_heights):
        row_spans.append((cursor, cursor + size - 1))
        cursor += size
        if i != len(row_heights) - 1:
            grid[cursor, :] = SEP
            cursor += 1

    cursor = 0
    for i, size in enumerate(col_widths):
        col_spans.append((cursor, cursor + size - 1))
        cursor += size
        if i != len(col_widths) - 1:
            grid[:, cursor] = SEP
            cursor += 1

    return grid, row_spans, col_spans


def _tile_regions(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    total = rows * cols
    used = np.zeros((rows, cols), dtype=bool)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def backtrack(filled: int) -> bool:
        if filled == total:
            return True
        for r in range(rows):
            for c in range(cols):
                if not used[r, c]:
                    start = (r, c)
                    break
            else:
                continue
            break

        neighbors = []
        for dr, dc in random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 4):
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and not used[nr, nc]:
                neighbors.append((nr, nc))
        random.shuffle(neighbors)
        for other in neighbors:
            used[start] = True
            used[other] = True
            pairs.append((start, other))
            if backtrack(filled + 2):
                return True
            pairs.pop()
            used[start] = False
            used[other] = False
        return False

    if backtrack(0):
        return pairs
    return None


def _grow_polyomino(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier
        cells.add(random.choice(frontier))
    patch = np.full((height, width), BG, dtype=int)
    for r, c in cells:
        patch[r, c] = 1
    return patch


def _random_pattern_patch(max_height: int, max_width: int) -> np.ndarray:
    height = random.randint(1, min(MAX_SHAPE_SPAN, max_height))
    width = random.randint(1, min(MAX_SHAPE_SPAN, max_width))
    area = random.randint(1, height * width)
    patch = _grow_polyomino(height, width, area)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch.copy()


def _place_patch(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    patch: np.ndarray,
    color: int,
) -> np.ndarray:
    r0, c0, r1, c1 = region
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    ph, pw = patch.shape
    top = r0 + random.randint(1, height - ph - 1)
    left = c0 + random.randint(1, width - pw - 1)
    coords = np.argwhere(patch > 0)
    abs_coords = coords + np.array([top, left], dtype=int)
    grid[abs_coords[:, 0], abs_coords[:, 1]] = color
    return abs_coords.astype(int, copy=False)


def _render_expected(
    inp: np.ndarray,
    records: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for record in records:
        out[record["pattern_coords"][:, 0], record["pattern_coords"][:, 1]] = BG
        out[record["arrow_coords"][:, 0], record["arrow_coords"][:, 1]] = BG

    for record in records:
        shift = np.array(
            _shift_for_destination(record["pattern_bbox"], record["arrow_region"], record["move_direction"]),
            dtype=int,
        )
        shifted = record["pattern_coords"] + shift
        out[shifted[:, 0], shifted[:, 1]] = int(record["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(REGION_SHAPES)
    row_heights = [random.randint(*ROW_HEIGHT_RANGE) for _ in range(rows)]
    col_widths = [random.randint(*COL_WIDTH_RANGE) for _ in range(cols)]
    base, row_spans, col_spans = _region_bounds_from_sizes(row_heights, col_widths)

    pairs = _tile_regions(rows, cols)
    if pairs is None:
        return None

    inp = base.copy()
    records: list[dict] = []
    for cell_a, cell_b in pairs:
        if random.random() < 0.5:
            arrow_cell, pattern_cell = cell_a, cell_b
        else:
            arrow_cell, pattern_cell = cell_b, cell_a

        ari, aci = arrow_cell
        pri, pci = pattern_cell
        dr = pri - ari
        dc = pci - aci
        if (dr, dc) not in DIRS.values():
            return None

        arrow_region = (
            row_spans[ari][0],
            col_spans[aci][0],
            row_spans[ari][1],
            col_spans[aci][1],
        )
        pattern_region = (
            row_spans[pri][0],
            col_spans[pci][0],
            row_spans[pri][1],
            col_spans[pci][1],
        )

        arrow_h = arrow_region[2] - arrow_region[0] + 1
        arrow_w = arrow_region[3] - arrow_region[1] + 1
        pattern_h = pattern_region[2] - pattern_region[0] + 1
        pattern_w = pattern_region[3] - pattern_region[1] + 1
        max_h = min(arrow_h - 2, pattern_h - 2)
        max_w = min(arrow_w - 2, pattern_w - 2)
        if max_h < 1 or max_w < 1:
            return None

        patch = _random_pattern_patch(max_h, max_w)
        color = random.choice(PATTERN_COLORS)
        pattern_coords = _place_patch(inp, pattern_region, patch, color)
        pattern_bbox = _bbox(pattern_coords)

        arrow_name = next(name for name, direction in DIRS.items() if direction == (dr, dc))
        arrow_patch = ARROW_PATCHES[arrow_name]
        arrow_coords = _place_patch(inp, arrow_region, arrow_patch, ARROW)

        records.append(
            {
                "pattern_coords": pattern_coords,
                "pattern_bbox": pattern_bbox,
                "arrow_coords": arrow_coords,
                "arrow_region": arrow_region,
                "move_direction": (-dr, -dc),
                "color": color,
            }
        )

    expected = _render_expected(inp, records)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
