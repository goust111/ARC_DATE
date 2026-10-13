from __future__ import annotations

from collections import deque
from itertools import combinations, product
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def find_separator_row(grid: np.ndarray, bg: int = BLACK) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    assert len(rows) == 1, "expected exactly one non-background full-row separator"
    sep = rows[0]
    assert 0 < sep < grid.shape[0] - 1, "separator must split the grid into two parts"
    return sep


def parse_legend(legend: np.ndarray, bg: int = BLACK) -> dict[int, int]:
    mapping: dict[int, int] = {}
    comps = components_from_mask(legend != bg)
    assert comps, "expected at least one legend domino"
    for comp in comps:
        assert len(comp) == 2, "each legend entry must be a 1x2 domino"
        (r0, c0), (r1, c1) = comp.tolist()
        assert r0 == r1 and abs(c0 - c1) == 1, "legend dominoes must be horizontal"
        left_c = min(c0, c1)
        right_c = max(c0, c1)
        key = int(legend[r0, left_c])
        value = int(legend[r0, right_c])
        assert key != bg and value != bg
        assert key not in mapping, "legend keys must be unique"
        mapping[key] = value
    return mapping


def region_marker_counts(
    work: np.ndarray, coords: np.ndarray, markers: dict[tuple[int, int], int]
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            color = markers.get((nr, nc))
            if color is not None:
                counts[color] = counts.get(color, 0) + 1
    return counts


def analyze_regions(grid: np.ndarray):
    sep = find_separator_row(grid)
    work = grid[:sep, :]
    legend = grid[sep + 1 :, :]
    mapping = parse_legend(legend)

    markers = {
        (r, c): int(work[r, c])
        for r in range(work.shape[0])
        for c in range(work.shape[1])
        if int(work[r, c]) not in (BLACK, GREY)
    }
    assert markers, "expected marker pixels in the work area"
    assert set(markers.values()).issubset(set(mapping)), "every marker color must appear as a legend key"

    regions = components_from_mask(work == GREY)
    assert len(regions) >= 2, "expected multiple grey regions"
    info = []
    for coords in regions:
        counts = region_marker_counts(work, coords, markers)
        assert counts, "each region must touch at least one marker"
        best = max(counts.values())
        winners = [color for color, count in counts.items() if count == best]
        assert len(winners) == 1, "adjacency majority must be unique"
        winner = winners[0]
        info.append(
            {
                "coords": coords,
                "counts": counts,
                "winner": winner,
                "fill": mapping[winner],
            }
        )
    return sep, mapping, markers, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey region with the legend value mapped from its majority-adjacent marker color.

    Input structure:
    - A full separator row splits the grid into a top work area and a bottom legend area.
    - The work area has a black background, several grey 4-connected regions, and colored marker pixels touching those regions.
    - The legend area contains horizontal 1x2 dominoes whose left cell is a marker color key and right cell is the fill color value.

    Rule:
    - For each grey region, count side-adjacencies from the region to marker pixels by marker color.
    - Take the unique color with the strictly largest adjacency count as the winning key.
    - Look up that key in the legend and recolor the whole grey region with the legend's value color.
    - Keep the marker pixels, legend dominoes, separator row, and black background unchanged.

    Key insight:
    - The neighboring markers only decide which legend key wins; the actual fill color is the separate value hidden in the lower lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep, _mapping, _markers, info = analyze_regions(g)
    out = g.copy()
    for region in info:
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(region["fill"])
    return out


def derangement(values: list[int]) -> list[int]:
    vals = values[:]
    for _ in range(100):
        random.shuffle(vals)
        if all(a != b for a, b in zip(values, vals)):
            return vals[:]
    raise RuntimeError("failed to sample a derangement")


def build_shape(max_h: int, max_w: int) -> np.ndarray | None:
    modes = ["rectangle", "lshape", "tshape"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "rectangle" and max_h >= 3 and max_w >= 3:
            h = random.randint(3, min(5, max_h))
            w = random.randint(3, min(6, max_w))
            return np.ones((h, w), dtype=bool)

        if mode == "lshape" and max_h >= 4 and max_w >= 4:
            h = random.randint(4, min(6, max_h))
            w = random.randint(4, min(6, max_w))
            vh = random.randint(max(2, h // 2), h)
            hw = random.randint(max(2, w // 2), w)
            vt = random.randint(1, min(2, w - 1))
            ht = random.randint(1, min(2, h - 1))
            patch = np.zeros((h, w), dtype=bool)
            corner = random.choice(["tl", "tr", "bl", "br"])
            if corner == "tl":
                patch[:vh, :vt] = True
                patch[:ht, :hw] = True
            elif corner == "tr":
                patch[:vh, w - vt :] = True
                patch[:ht, w - hw :] = True
            elif corner == "bl":
                patch[h - vh :, :vt] = True
                patch[h - ht :, :hw] = True
            else:
                patch[h - vh :, w - vt :] = True
                patch[h - ht :, w - hw :] = True
            if patch.any() and not patch.all():
                return patch

        if mode == "tshape" and max_h >= 4 and max_w >= 5:
            h = random.randint(4, min(6, max_h))
            w = random.randint(5, min(7, max_w))
            bar_t = random.randint(1, min(2, h - 2))
            stem_t = random.randint(1, min(2, w - 2))
            stem_c0 = random.randint(1, w - stem_t - 1)
            patch = np.zeros((h, w), dtype=bool)
            patch[:bar_t, :] = True
            patch[:, stem_c0 : stem_c0 + stem_t] = True
            patch = np.rot90(patch, random.randrange(4))
            if patch.shape[0] <= max_h and patch.shape[1] <= max_w:
                return patch
    return None


def split_range(length: int, gutter: int = 2) -> tuple[tuple[int, int], tuple[int, int]]:
    total = length - gutter
    first = random.randint(total // 2 - 1, total // 2 + 1)
    first = max(5, min(total - 5, first))
    second = total - first
    return (0, first - 1), (first + gutter, first + gutter + second - 1)


def region_slots(grid: np.ndarray, region_mask: np.ndarray) -> list[tuple[int, int]]:
    slots = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK:
                continue
            contact = 0
            for nr, nc in neighbors4(r, c):
                if in_bounds(grid.shape, nr, nc) and region_mask[nr, nc]:
                    contact += 1
            if contact == 1:
                slots.append((r, c))
    return slots


def choose_marker_plan(keys: list[int], winner: int, n_slots: int) -> dict[int, int] | None:
    others_pool = [k for k in keys if k != winner]
    options: list[tuple[int, dict[int, int]]] = []
    for n_other in range(1, min(2, len(others_pool)) + 1):
        for others in combinations(others_pool, n_other):
            participants = [winner, *others]
            max_winner = min(5, n_slots - n_other)
            for winner_count in range(2, max_winner + 1):
                for loser_counts in product(range(1, winner_count), repeat=n_other):
                    total = winner_count + sum(loser_counts)
                    if total > n_slots or total < len(participants) + 1:
                        continue
                    gap = winner_count - max(loser_counts)
                    plan = {winner: winner_count}
                    for color, count in zip(others, loser_counts):
                        plan[color] = count
                    score = 10 * len(participants) - gap
                    if gap == 1:
                        score += 4
                    options.append((score, plan))
    if not options:
        return None
    options.sort(key=lambda item: item[0], reverse=True)
    top = [plan for _score, plan in options[: min(10, len(options))]]
    return random.choice(top)


def place_markers(grid: np.ndarray, region_mask: np.ndarray, plan: dict[int, int]) -> bool:
    slots = region_slots(grid, region_mask)
    needed = sum(plan.values())
    if len(slots) < needed:
        return False
    random.shuffle(slots)
    index = 0
    items = list(plan.items())
    random.shuffle(items)
    for color, count in items:
        for _ in range(count):
            if index >= len(slots):
                return False
            r, c = slots[index]
            grid[r, c] = color
            index += 1
    return True


def place_legend(grid: np.ndarray, start_row: int, mapping: dict[int, int]) -> None:
    h, w = grid.shape
    legend_h = h - start_row
    rows = [start_row, start_row + 2] if legend_h >= 3 else [start_row]
    slots = []
    for r in rows:
        if r >= h:
            continue
        for c in range(1, w - 2, 3):
            slots.append((r, c))
    assert len(slots) >= len(mapping)
    chosen = random.sample(slots, len(mapping))
    items = list(mapping.items())
    random.shuffle(items)
    for (key, value), (r, c) in zip(items, chosen):
        grid[r, c] = key
        grid[r, c + 1] = value


def counts_for_region(grid: np.ndarray, coords: np.ndarray) -> dict[int, int]:
    markers = {
        (r, c): int(grid[r, c])
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) not in (BLACK, GREY)
    }
    return region_marker_counts(grid, coords, markers)


def validate_instance(inp: np.ndarray, expected_winners: list[int], key_count: int) -> bool:
    try:
        sep, mapping, _markers, info = analyze_regions(inp)
        out = main(inp)
    except Exception:
        return False

    if len(mapping) != key_count:
        return False
    if any(key == value for key, value in mapping.items()):
        return False
    if len(info) != len(expected_winners):
        return False

    winners = [region["winner"] for region in info]
    if sorted(winners) != sorted(expected_winners):
        return False
    if set(winners) != set(mapping):
        return False

    fill_colors = set()
    for region in info:
        counts = region["counts"]
        if len(counts) < 2:
            return False
        best = counts[region["winner"]]
        runner_up = max(count for color, count in counts.items() if color != region["winner"])
        if best <= runner_up:
            return False
        if mapping[region["winner"]] == region["winner"]:
            return False
        coords = region["coords"]
        fill = int(out[coords[0, 0], coords[0, 1]])
        if fill != mapping[region["winner"]]:
            return False
        if not np.all(out[coords[:, 0], coords[:, 1]] == fill):
            return False
        fill_colors.add(fill)

    if len(fill_colors) < 2:
        return False

    changed = inp != out
    work_mask = np.zeros_like(inp, dtype=bool)
    work_mask[:sep, :] = inp[:sep, :] == GREY
    if not np.array_equal(changed, work_mask):
        return False

    return True


def generate_candidate() -> tuple[np.ndarray, list[int], int] | None:
    work_h = random.randint(12, 18)
    legend_h = random.randint(3, 4)
    h = work_h + 1 + legend_h
    w = random.randint(20, 28)

    sep_color = random.choice(PALETTE)
    free_colors = [c for c in PALETTE if c != sep_color]

    key_count = random.choice([3, 4])
    if key_count == 4:
        keys = random.sample(free_colors, 4)
        values = derangement(keys)
    else:
        if random.random() < 0.55:
            chosen = random.sample(free_colors, 6)
            keys = chosen[:3]
            values = chosen[3:]
        else:
            keys = random.sample(free_colors, 3)
            values = derangement(keys)
    mapping = {key: value for key, value in zip(keys, values)}

    region_count = key_count if key_count == 4 else random.choice([3, 4])
    winners = keys[:]
    while len(winners) < region_count:
        winners.append(random.choice(keys))
    random.shuffle(winners)

    grid = np.zeros((h, w), dtype=int)
    grid[work_h, :] = sep_color

    row_a, row_b = split_range(work_h, gutter=2)
    col_a, col_b = split_range(w, gutter=2)
    cells = [
        (row_a[0], row_a[1], col_a[0], col_a[1]),
        (row_a[0], row_a[1], col_b[0], col_b[1]),
        (row_b[0], row_b[1], col_a[0], col_a[1]),
        (row_b[0], row_b[1], col_b[0], col_b[1]),
    ]
    chosen_cells = random.sample(cells, region_count)

    region_masks: list[np.ndarray] = []
    for cell in chosen_cells:
        r0, r1, c0, c1 = cell
        cell_h = r1 - r0 + 1
        cell_w = c1 - c0 + 1
        shape = build_shape(cell_h - 2, cell_w - 2)
        if shape is None:
            return None
        sh, sw = shape.shape
        ar = random.randint(r0 + 1, r1 - sh)
        ac = random.randint(c0 + 1, c1 - sw)
        mask = np.zeros((work_h, w), dtype=bool)
        mask[ar : ar + sh, ac : ac + sw] = shape
        if np.any(grid[:work_h, :][mask] != BLACK):
            return None
        grid[:work_h, :][mask] = GREY
        region_masks.append(mask)

    for winner, mask in zip(winners, region_masks):
        plan = choose_marker_plan(keys, winner, len(region_slots(grid[:work_h, :], mask)))
        if plan is None:
            return None
        if not place_markers(grid[:work_h, :], mask, plan):
            return None

    place_legend(grid, work_h + 1, mapping)
    return grid, winners, key_count


def generate_input() -> np.ndarray:
    for _ in range(300):
        candidate = generate_candidate()
        if candidate is None:
            continue
        inp, winners, key_count = candidate
        if validate_instance(inp, winners, key_count):
            return inp
    raise RuntimeError("failed to generate a valid input")
