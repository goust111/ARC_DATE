from __future__ import annotations

from collections import Counter
import random

import numpy as np


NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


GRID_HEIGHT_RANGE_BY_COUNT = {2: (18, 24), 3: (24, 28)}
GRID_WIDTH_RANGE_BY_COUNT = {2: (16, 24), 3: (22, 28)}
OUTLINE_COUNT_CHOICES = [2, 3]
OUTLINE_COUNT_WEIGHTS = [1, 3]
OUTLINE_HEIGHT_RANGE_BY_COUNT = {2: (4, 8), 3: (4, 6)}
MIN_INTERIOR_CELLS = 6
MIN_INTERIOR_BLACK_CELLS = 1
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _outline_objects(grid: np.ndarray) -> list[dict]:
    objs: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != 0):
        for coords in _components(grid == color):
            rows = sorted(int(r) for r in np.unique(coords[:, 0]))
            if len(rows) < 2:
                continue
            row_pairs: list[tuple[int, int, int]] = []
            valid = True
            interior_cells: list[tuple[int, int]] = []
            for r in rows:
                cols = sorted(int(c) for c in coords[coords[:, 0] == r, 1])
                if len(cols) != 2:
                    valid = False
                    break
                c0, c1 = cols
                if c1 <= c0:
                    valid = False
                    break
                row_pairs.append((r, c0, c1))
                for c in range(c0 + 1, c1):
                    interior_cells.append((r, c))
            if not valid or not interior_cells:
                continue
            objs.append(
                {
                    "color": color,
                    "coords": coords,
                    "rows": row_pairs,
                    "interior": interior_cells,
                }
            )
    return objs


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    outlines = _outline_objects(g)
    assert outlines, "expected at least one outline"

    out = np.zeros_like(g)
    details: list[dict] = []
    for obj in outlines:
        color = int(obj["color"])
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = color

        counts: Counter[int] = Counter()
        spans: list[tuple[int, int, int]] = []
        for r, c0, c1 in obj["rows"]:
            if c1 - c0 > 1:
                inner = g[r, c0 + 1 : c1]
                for value in inner:
                    value = int(value)
                    if value != 0 and value != color:
                        counts[value] += 1
                spans.append((r, c0 + 1, c1 - 1))
        assert counts, "each outline must contain at least one marker"

        max_count = max(counts.values())
        winners = [marker for marker, cnt in counts.items() if cnt == max_count]
        assert len(winners) == 1, "marker frequency inside each outline must have a unique maximum"
        fill_color = int(winners[0])

        for r, left, right in spans:
            out[r, left : right + 1] = fill_color

        details.append(
            {
                "outline_color": color,
                "fill_color": fill_color,
                "counts": dict(counts),
                "rows": list(obj["rows"]),
            }
        )

    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each outline interior with the marker color that is most frequent inside that same outline.

    Input structure:
    - The grid has black background plus 2-3 monochromatic 8-connected outlines in distinct colors.
    - Each outline occupies several rows, and on every occupied row it has exactly two pixels: the left and right boundary.
    - Cells strictly between the two boundary pixels are either black or isolated marker pixels in colors different from every outline color; some extra marker pixels may also sit outside all outlines.

    Rule:
    - Detect every monochromatic outline component and, row by row, treat the cells strictly between its two boundary pixels as that outline's interior.
    - Count the non-black marker colors that appear inside that interior, using only that outline's own interior as the counting domain.
    - Rebuild the output from black: keep each outline boundary in its original color, fill its whole interior with the uniquely most frequent interior marker color, and discard every marker that lies outside all outline interiors.

    Key insight:
    - The outline geometry defines the local sampling region, so the winning fill color is not a global color statistic but a per-region frequency computed only inside that outline.
    """
    out, _details = _solve_with_details(grid)
    return out


def _make_outline_shape(height_range: tuple[int, int]) -> dict | None:
    height = random.randint(*height_range)
    pinch_choices = list(range(height))
    pinch_weights = [2 if idx in (0, height - 1) else 3 for idx in pinch_choices]
    pinch_row = random.choices(pinch_choices, weights=pinch_weights, k=1)[0]

    left = [0] * height
    right = [0] * height
    left[pinch_row] = 0
    right[pinch_row] = 1

    for step in (-1, 1):
        prev = pinch_row
        row = prev + step
        while 0 <= row < height:
            left[row] = left[prev] - random.choice([0, 1, 1])
            right[row] = right[prev] + random.choice([0, 1, 1])
            prev = row
            row += step

    shift = -min(left)
    left = [x + shift for x in left]
    right = [x + shift for x in right]
    width = max(right) + 1

    outline_cells: list[tuple[int, int]] = []
    interior_cells: list[tuple[int, int]] = []
    filled_cells: list[tuple[int, int]] = []
    for r in range(height):
        c0 = left[r]
        c1 = right[r]
        outline_cells.append((r, c0))
        outline_cells.append((r, c1))
        for c in range(c0, c1 + 1):
            filled_cells.append((r, c))
        for c in range(c0 + 1, c1):
            interior_cells.append((r, c))

    if len(interior_cells) < MIN_INTERIOR_CELLS:
        return None
    if max(right[r] - left[r] - 1 for r in range(height)) < 2:
        return None
    if len(interior_cells) - 4 < MIN_INTERIOR_BLACK_CELLS:
        return None

    filled_set = set(filled_cells)
    padded_set: set[tuple[int, int]] = set()
    for r, c in filled_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                padded_set.add((r + dr, c + dc))

    return {
        "height": height,
        "width": width,
        "outline": outline_cells,
        "interior": interior_cells,
        "filled": filled_cells,
        "padded": padded_set,
    }


def _partition_total(total: int, parts: int, min_part: int) -> list[int] | None:
    extra = total - parts * min_part
    if extra < 0:
        return None
    values = [min_part] * parts
    for _ in range(extra):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _layout_zones(outline_count: int, grid_h: int, grid_w: int) -> list[tuple[int, int, int, int]] | None:
    inner_h = grid_h - 2
    inner_w = grid_w - 2
    gap = 2
    min_side = 6

    if outline_count == 2:
        layouts = ["cols", "rows"]
        random.shuffle(layouts)
        for layout in layouts:
            if layout == "cols":
                parts = _partition_total(inner_w - gap, 2, min_side)
                if parts is None or inner_h < min_side:
                    continue
                zones = []
                c = 1
                for width in parts:
                    zones.append((1, c, inner_h, width))
                    c += width + gap
                return zones
            parts = _partition_total(inner_h - gap, 2, min_side)
            if parts is None or inner_w < min_side:
                continue
            zones = []
            r = 1
            for height in parts:
                zones.append((r, 1, height, inner_w))
                r += height + gap
            return zones
        return None

    layouts = ["cols3", "rows3", "top2_bottom1", "top1_bottom2"]
    random.shuffle(layouts)
    for layout in layouts:
        if layout == "cols3":
            parts = _partition_total(inner_w - 2 * gap, 3, min_side)
            if parts is None or inner_h < min_side:
                continue
            zones = []
            c = 1
            for width in parts:
                zones.append((1, c, inner_h, width))
                c += width + gap
            return zones
        if layout == "rows3":
            parts = _partition_total(inner_h - 2 * gap, 3, min_side)
            if parts is None or inner_w < min_side:
                continue
            zones = []
            r = 1
            for height in parts:
                zones.append((r, 1, height, inner_w))
                r += height + gap
            return zones
        if layout == "top2_bottom1":
            row_parts = _partition_total(inner_h - gap, 2, min_side)
            col_parts = _partition_total(inner_w - gap, 2, min_side)
            if row_parts is None or col_parts is None:
                continue
            top_h, bot_h = row_parts
            left_w, right_w = col_parts
            return [
                (1, 1, top_h, left_w),
                (1, 1 + left_w + gap, top_h, right_w),
                (1 + top_h + gap, 1, bot_h, inner_w),
            ]
        row_parts = _partition_total(inner_h - gap, 2, min_side)
        col_parts = _partition_total(inner_w - gap, 2, min_side)
        if row_parts is None or col_parts is None:
            continue
        top_h, bot_h = row_parts
        left_w, right_w = col_parts
        return [
            (1, 1, top_h, inner_w),
            (1 + top_h + gap, 1, bot_h, left_w),
            (1 + top_h + gap, 1 + left_w + gap, bot_h, right_w),
        ]
    return None


def _place_in_zones(outline_count: int, zones: list[tuple[int, int, int, int]]) -> list[dict] | None:
    placements: list[dict] = []
    height_range = OUTLINE_HEIGHT_RANGE_BY_COUNT[outline_count]
    for zone_r, zone_c, zone_h, zone_w in zones:
        placed = None
        max_height = min(height_range[1], zone_h - 2)
        if max_height < height_range[0] or zone_w < 6:
            return None
        for _ in range(80):
            shape = _make_outline_shape((height_range[0], max_height))
            if shape is None:
                continue
            if shape["height"] > zone_h - 2 or shape["width"] > zone_w - 2:
                continue
            top = zone_r + random.randint(1, zone_h - shape["height"] - 1)
            left = zone_c + random.randint(1, zone_w - shape["width"] - 1)
            placed = dict(shape)
            placed["top"] = top
            placed["left"] = left
            break
        if placed is None:
            return None
        placements.append(placed)
    return placements


def _blocked_from_placements(grid_shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    h, w = grid_shape
    blocked = np.zeros((h, w), dtype=bool)
    for placed in placements:
        top = placed["top"]
        left = placed["left"]
        for rr, cc in placed["filled"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    tr, tc = top + rr + dr, left + cc + dc
                    if 0 <= tr < h and 0 <= tc < w:
                        blocked[tr, tc] = True
    return blocked


def _choose_isolated_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    shuffled = cells[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    for cell in shuffled:
        if cell in blocked:
            continue
        chosen.append(cell)
        r, c = cell
        blocked.add((r, c))
        blocked.add((r - 1, c))
        blocked.add((r + 1, c))
        blocked.add((r, c - 1))
        blocked.add((r, c + 1))
        if len(chosen) == count:
            return chosen
    return None


def _build_grid(
    shape: tuple[int, int],
    placements: list[dict],
    outline_colors: list[int],
    fill_specs: list[dict],
    noise: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)

    for placed, outline_color, spec in zip(placements, outline_colors, fill_specs):
        top = placed["top"]
        left = placed["left"]
        for rr, cc in placed["outline"]:
            inp[top + rr, left + cc] = outline_color
            expected[top + rr, left + cc] = outline_color
        for rr, cc in placed["interior"]:
            expected[top + rr, left + cc] = spec["winner"]
        for (rr, cc), color in spec["markers"]:
            inp[top + rr, left + cc] = color

    for r, c, color in noise:
        inp[r, c] = color

    return inp, expected


def _pick_fill_specs(placements: list[dict], marker_pool: list[int]) -> list[dict] | None:
    if len(marker_pool) < len(placements) + 2:
        return None

    winners = random.sample(marker_pool, len(placements))
    specs: list[dict] = []
    for idx, placed in enumerate(placements):
        interior = placed["interior"]
        color_count = random.choice([2, 3])
        available = [c for c in marker_pool if c != winners[idx]]
        others = random.sample(available, color_count - 1)
        colors = [winners[idx], *others]

        patterns = [(3, 1), (3, 2), (4, 1), (4, 2)] if color_count == 2 else [(3, 1, 1), (4, 1, 1), (4, 2, 1)]
        counts_options = [
            pattern
            for pattern in patterns
            if sum(pattern) <= len(interior) - MIN_INTERIOR_BLACK_CELLS
        ]
        if not counts_options:
            return None
        counts = list(random.choice(counts_options))
        total_markers = sum(counts)
        chosen_cells = _choose_isolated_cells(interior, total_markers)
        if chosen_cells is None:
            return None

        random.shuffle(chosen_cells)
        markers: list[tuple[tuple[int, int], int]] = []
        cursor = 0
        for color, count in zip(colors, counts):
            for cell in chosen_cells[cursor : cursor + count]:
                markers.append((cell, color))
            cursor += count

        specs.append(
            {
                "winner": winners[idx],
                "markers": markers,
                "counts": dict(zip(colors, counts)),
            }
        )

    if len(set(spec["winner"] for spec in specs)) != len(specs):
        return None
    return specs


def _pick_noise(
    grid_shape: tuple[int, int],
    blocked: np.ndarray,
    fill_specs: list[dict],
    marker_pool: list[int],
) -> list[tuple[int, int, int]] | None:
    h, w = grid_shape
    if random.random() < 0.35:
        return []

    candidates = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    losers = [
        color
        for spec in fill_specs
        for color in spec["counts"]
        if color != int(spec["winner"])
    ]
    if losers:
        noise_color = random.choice(losers)
    else:
        noise_color = random.choice([c for c in marker_pool if c not in [spec["winner"] for spec in fill_specs]])

    min_winner = min(int(spec["counts"][spec["winner"]]) for spec in fill_specs)
    noise_count = random.randint(min_winner + 1, min(min_winner + 3, 6))
    cells = _choose_isolated_cells(candidates, noise_count)
    if cells is None:
        return None

    noise = [(r, c, noise_color) for r, c in cells]
    extra_budget = min(2, len(candidates) - len(cells))
    if extra_budget > 0 and random.random() < 0.5:
        remaining = [
            cell
            for cell in candidates
            if cell not in cells
            and all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in cells)
        ]
        extra_cells = _choose_isolated_cells(remaining, extra_budget)
        if extra_cells is not None:
            extra_colors = [c for c in marker_pool if c != noise_color]
            for r, c in extra_cells[: random.randint(1, len(extra_cells))]:
                noise.append((r, c, random.choice(extra_colors)))
    return noise


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    placements: list[dict],
    outline_colors: list[int],
    fill_specs: list[dict],
    noise: list[tuple[int, int, int]],
) -> bool:
    try:
        actual, details = _solve_with_details(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(details) != len(placements):
        return False
    if len(set(spec["winner"] for spec in fill_specs)) != len(fill_specs):
        return False

    fill_changed_black = False
    loser_present = False
    for placed, outline_color, spec in zip(placements, outline_colors, fill_specs):
        top = placed["top"]
        left = placed["left"]
        row_pairs = {}
        for rr, cc in placed["outline"]:
            row_pairs.setdefault(top + rr, []).append(left + cc)
        for cols in row_pairs.values():
            if len(cols) != 2:
                return False
        counts = Counter()
        for rr, cc in placed["interior"]:
            value = int(grid[top + rr, left + cc])
            if value == 0:
                fill_changed_black = True
            if value != 0:
                counts[value] += 1
        if dict(counts) != spec["counts"]:
            return False
        if max(counts.values()) != counts[spec["winner"]]:
            return False
        if list(counts.values()).count(counts[spec["winner"]]) != 1:
            return False
        if any(color != spec["winner"] for color in counts):
            loser_present = True
        if outline_color in counts:
            return False

    if not fill_changed_black or not loser_present:
        return False

    if noise:
        noise_counts = Counter(color for _r, _c, color in noise)
        total_counts = Counter()
        for spec in fill_specs:
            total_counts.update(spec["counts"])
        total_counts.update(noise_counts)
        global_color, _global_count = total_counts.most_common(1)[0]
        if global_color in [spec["winner"] for spec in fill_specs]:
            return False

    return True


def _gen_attempt(outline_count: int) -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE_BY_COUNT[outline_count])
    grid_w = random.randint(*GRID_WIDTH_RANGE_BY_COUNT[outline_count])

    palette = list(range(1, 10))
    outline_colors = random.sample(palette, outline_count)
    marker_pool = [color for color in palette if color not in outline_colors]

    zones = _layout_zones(outline_count, grid_h, grid_w)
    if zones is None:
        return None
    placements = _place_in_zones(outline_count, zones)
    if placements is None:
        return None
    blocked = _blocked_from_placements((grid_h, grid_w), placements)

    fill_specs = _pick_fill_specs(placements, marker_pool)
    if fill_specs is None:
        return None

    noise = _pick_noise((grid_h, grid_w), blocked, fill_specs, marker_pool)
    if noise is None:
        return None

    grid, expected = _build_grid((grid_h, grid_w), placements, outline_colors, fill_specs, noise)
    if not _semantic_ok(grid, expected, placements, outline_colors, fill_specs, noise):
        return None
    return grid


def generate_input() -> np.ndarray:
    outline_count = random.choices(OUTLINE_COUNT_CHOICES, weights=OUTLINE_COUNT_WEIGHTS, k=1)[0]
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(outline_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid local-frequency-outline input")
