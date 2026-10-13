from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SCATTER_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 500

_GEN_COUNTER = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _constant_divider_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GREY))]


def _constant_divider_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]


def _spans(dividers: list[int], limit: int) -> list[tuple[int, int]]:
    cuts = [-1] + sorted(int(v) for v in dividers) + [int(limit)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a > 1:
            spans.append((int(a + 1), int(b - 1)))
    return spans


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _direction_from_domino(region: np.ndarray, cells: list[tuple[int, int]]) -> tuple[int, int]:
    reds = [cell for cell in cells if int(region[cell]) == RED]
    greens = [cell for cell in cells if int(region[cell]) == GREEN]
    assert len(reds) == 1 and len(greens) == 1, "domino must contain one red and one green"
    rr, rc = reds[0]
    gr, gc = greens[0]
    dr = int(gr - rr)
    dc = int(gc - rc)
    assert abs(dr) + abs(dc) == 1, "domino must be 4-connected"
    return dr, dc


def _analyze_region(region: np.ndarray) -> dict[str, object]:
    rg_mask = np.isin(region, [RED, GREEN])
    mixed_dominoes: list[list[tuple[int, int]]] = []
    for comp in _components(rg_mask):
        colors = {int(region[r, c]) for r, c in comp}
        if len(comp) == 2 and colors == {RED, GREEN}:
            mixed_dominoes.append(comp)
        else:
            assert colors <= {RED, GREEN}, "unexpected component analysis state"
            assert not ({RED, GREEN} <= colors), "only size-2 red-green components may exist"
    assert len(mixed_dominoes) <= 1, "each region may contain at most one domino"

    domino_cells: set[tuple[int, int]] = set()
    direction: tuple[int, int] | None = None
    if mixed_dominoes:
        domino_cells = set(mixed_dominoes[0])
        direction = _direction_from_domino(region, mixed_dominoes[0])

    scatter_mask = (region != BLACK) & (region != GREY)
    for r, c in domino_cells:
        scatter_mask[r, c] = False

    scatter_colors = {int(v) for v in region[scatter_mask]}
    assert len(scatter_colors) == 1, "each region must have exactly one scatter color"
    scatter_color = next(iter(scatter_colors))
    scatter_count = int(np.count_nonzero(scatter_mask))
    assert 1 <= scatter_count <= 5, "scatter count must stay in the specified range"

    return {
        "scatter_color": int(scatter_color),
        "scatter_count": int(scatter_count),
        "has_domino": bool(direction is not None),
        "direction": direction,
    }


def _extract_regions(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, object]], list[int], list[int]]:
    row_dividers = _constant_divider_rows(grid)
    col_dividers = _constant_divider_cols(grid)
    row_spans = _spans(row_dividers, grid.shape[0])
    col_spans = _spans(col_dividers, grid.shape[1])
    assert 2 <= len(row_spans) <= 4
    assert 2 <= len(col_spans) <= 4

    regions: list[dict[str, object]] = []
    lookup: dict[tuple[int, int], dict[str, object]] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            region = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
            info = _analyze_region(region)
            info.update(
                {
                    "index": (int(ri), int(ci)),
                    "bounds": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": region.shape,
                }
            )
            regions.append(info)
            lookup[(int(ri), int(ci))] = info
    return regions, lookup, row_dividers, col_dividers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the highest-count regions, but let winning red-green dominoes redirect the fill into neighboring regions.

    Input structure:
    - Black background with full grey separator rows and columns partitioning the grid into 6-9 rectangular regions.
    - Each region contains 1-5 scattered pixels of exactly one color from blue/red/green/yellow/orange.
    - Some regions also contain one 4-connected red-green domino; the domino direction is read from red toward green.

    Rule:
    - Split the grid by the grey separator lines and count each region's scattered non-black pixels, excluding domino cells.
    - Only regions whose scatter count equals the global maximum win; every other region becomes black.
    - A winning region without a domino fills itself with its scatter color, while a winning region with a domino stays black and instead fills the separator-adjacent region in the domino direction, using the target region's own scatter color.

    Key insight:
    - Counting selects the winners, but a winner with a red-green domino gives its fill privilege to a neighboring region indicated by the domino orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions, lookup, row_dividers, col_dividers = _extract_regions(g)
    max_count = max(int(info["scatter_count"]) for info in regions)
    winners = [info for info in regions if int(info["scatter_count"]) == max_count]

    out = np.zeros_like(g)
    if row_dividers:
        out[row_dividers, :] = GREY
    if col_dividers:
        out[:, col_dividers] = GREY

    fill_targets: set[tuple[int, int]] = set()
    for info in winners:
        ri, ci = info["index"]
        if bool(info["has_domino"]):
            dr, dc = info["direction"]
            target_index = (int(ri + dr), int(ci + dc))
            assert target_index in lookup, "domino winner must point to an adjacent region"
            fill_targets.add(target_index)
        else:
            fill_targets.add((int(ri), int(ci)))

    for target_index in fill_targets:
        target = lookup[target_index]
        r0, c0, r1, c1 = target["bounds"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(target["scatter_color"])

    return out


def _layout_variant(counter: int) -> tuple[int, int]:
    cycle = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
    base = cycle[counter % len(cycle)]
    if random.random() < 0.35:
        return random.choice(cycle)
    return base


def _region_neighbors(ri: int, ci: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    neighbors: list[tuple[int, int, int, int]] = []
    for dr, dc in DIRECTIONS:
        nr = int(ri + dr)
        nc = int(ci + dc)
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc, int(dr), int(dc)))
    return neighbors


def _pick_cells(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if len(candidates) < count:
        return None
    random.shuffle(candidates)
    return [(int(r), int(c)) for r, c in candidates[:count]]


def _build_region_patch(
    height: int,
    width: int,
    scatter_color: int,
    scatter_count: int,
    domino_direction: tuple[int, int] | None,
) -> np.ndarray | None:
    patch = np.zeros((height, width), dtype=int)
    domino_cells: set[tuple[int, int]] = set()
    if domino_direction is not None:
        dr, dc = domino_direction
        starts = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if 0 <= r + dr < height and 0 <= c + dc < width
        ]
        random.shuffle(starts)
        if not starts:
            return None
        red_r, red_c = starts[0]
        green_r, green_c = int(red_r + dr), int(red_c + dc)
        patch[red_r, red_c] = RED
        patch[green_r, green_c] = GREEN
        domino_cells = {(int(red_r), int(red_c)), (int(green_r), int(green_c))}

    candidates: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            cell = (int(r), int(c))
            if cell in domino_cells:
                continue
            if scatter_color in (RED, GREEN):
                if any(nei in domino_cells for nei in _neighbors4(r, c)):
                    continue
            candidates.append(cell)

    chosen = _pick_cells(candidates, scatter_count)
    if chosen is None:
        return None
    for r, c in chosen:
        patch[r, c] = int(scatter_color)
    return patch


def _place_regions(
    row_sizes: list[int],
    col_sizes: list[int],
    specs: dict[tuple[int, int], dict[str, object]],
) -> np.ndarray | None:
    h = sum(row_sizes) + len(row_sizes) - 1
    w = sum(col_sizes) + len(col_sizes) - 1
    grid = np.zeros((h, w), dtype=int)

    r = 0
    for ri, region_h in enumerate(row_sizes):
        c = 0
        for ci, region_w in enumerate(col_sizes):
            spec = specs[(ri, ci)]
            patch = _build_region_patch(
                region_h,
                region_w,
                int(spec["scatter_color"]),
                int(spec["scatter_count"]),
                spec["domino_direction"],
            )
            if patch is None:
                return None
            grid[r : r + region_h, c : c + region_w] = patch
            c += region_w
            if ci + 1 < len(col_sizes):
                grid[:, c] = GREY
                c += 1
        r += region_h
        if ri + 1 < len(row_sizes):
            grid[r, :] = GREY
            r += 1
    return grid


def _weighted_scatter_color() -> int:
    return random.choice([BLUE, RED, GREEN, YELLOW, ORANGE, RED, GREEN])


def _choose_structure(counter: int) -> tuple[dict[tuple[int, int], dict[str, object]], list[int], list[int]] | None:
    rows, cols = _layout_variant(counter)
    region_ids = [(ri, ci) for ri in range(rows) for ci in range(cols)]
    row_sizes = [random.randint(3, 5) for _ in range(rows)]
    col_sizes = [random.randint(3, 5) for _ in range(cols)]

    donor_goal = 1 if counter % 4 in (0, 2) else 2
    donor_goal = min(donor_goal, max(1, len(region_ids) - 2))

    candidate_donors = [rid for rid in region_ids if _region_neighbors(rid[0], rid[1], rows, cols)]
    random.shuffle(candidate_donors)
    donors: list[tuple[int, int]] = candidate_donors[:donor_goal]
    if len(donors) != donor_goal:
        return None

    donor_targets: dict[tuple[int, int], tuple[int, int]] = {}
    used_targets: set[tuple[int, int]] = set()
    for donor in donors:
        choices = [
            (nr, nc, dr, dc)
            for nr, nc, dr, dc in _region_neighbors(donor[0], donor[1], rows, cols)
            if (nr, nc) not in donors and (nr, nc) not in used_targets
        ]
        if not choices:
            choices = [
                (nr, nc, dr, dc)
                for nr, nc, dr, dc in _region_neighbors(donor[0], donor[1], rows, cols)
                if (nr, nc) not in donors
            ]
        if not choices:
            return None
        nr, nc, _dr, _dc = random.choice(choices)
        donor_targets[donor] = (int(nr), int(nc))
        used_targets.add((int(nr), int(nc)))

    filler_goal = 2 if counter % 4 == 2 and len(region_ids) >= 7 else 1
    fill_candidates = [rid for rid in region_ids if rid not in donors and rid not in used_targets]
    if len(fill_candidates) < filler_goal:
        return None
    random.shuffle(fill_candidates)
    self_fillers = fill_candidates[:filler_goal]
    max_regions = set(donors) | set(self_fillers)

    remaining = [rid for rid in region_ids if rid not in max_regions]
    extra_domino_goal = 1 if len(donors) == 1 else random.choice([0, 1])
    extra_dominoes: set[tuple[int, int]] = set()
    if extra_domino_goal:
        options = [rid for rid in remaining if _region_neighbors(rid[0], rid[1], rows, cols)]
        if not options:
            return None
        random.shuffle(options)
        extra_dominoes = set(options[:extra_domino_goal])
        if len(extra_dominoes) != extra_domino_goal:
            return None

    max_count = random.randint(2, 5)
    specs: dict[tuple[int, int], dict[str, object]] = {}
    for rid in region_ids:
        specs[rid] = {
            "scatter_color": _weighted_scatter_color(),
            "scatter_count": int(max_count if rid in max_regions else random.randint(1, max_count - 1)),
            "domino_direction": None,
        }

    for donor, target in donor_targets.items():
        dr = int(target[0] - donor[0])
        dc = int(target[1] - donor[1])
        specs[donor]["domino_direction"] = (dr, dc)
        if specs[target]["scatter_color"] == specs[donor]["scatter_color"]:
            choices = [color for color in SCATTER_COLORS if color != specs[donor]["scatter_color"]]
            specs[target]["scatter_color"] = random.choice(choices)

    for rid in extra_dominoes:
        choices = _region_neighbors(rid[0], rid[1], rows, cols)
        nr, nc, dr, dc = random.choice(choices)
        specs[rid]["domino_direction"] = (int(dr), int(dc))
        target = (int(nr), int(nc))
        if specs[target]["scatter_color"] == specs[rid]["scatter_color"] and random.random() < 0.7:
            choices = [color for color in SCATTER_COLORS if color != specs[rid]["scatter_color"]]
            specs[target]["scatter_color"] = random.choice(choices)

    filled_colors = {int(specs[rid]["scatter_color"]) for rid in self_fillers}
    filled_colors |= {int(specs[target]["scatter_color"]) for target in donor_targets.values()}
    if len(filled_colors) < 2:
        adjust_target = random.choice(list(donor_targets.values()) + self_fillers)
        current = int(specs[adjust_target]["scatter_color"])
        specs[adjust_target]["scatter_color"] = random.choice([color for color in SCATTER_COLORS if color != current])

    return specs, row_sizes, col_sizes


def _region_contents(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bounds
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _is_uniform_region(grid: np.ndarray, bounds: tuple[int, int, int, int], color: int) -> bool:
    return bool(np.all(_region_contents(grid, bounds) == int(color)))


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        regions, lookup, row_dividers, col_dividers = _extract_regions(grid)
        row_spans = _spans(row_dividers, grid.shape[0])
        col_spans = _spans(col_dividers, grid.shape[1])
        if not (6 <= len(regions) <= 9):
            return False
        if any((r1 - r0 + 1) not in {3, 4, 5} for r0, r1 in row_spans):
            return False
        if any((c1 - c0 + 1) not in {3, 4, 5} for c0, c1 in col_spans):
            return False

        max_count = max(int(info["scatter_count"]) for info in regions)
        winners = [info for info in regions if int(info["scatter_count"]) == max_count]
        if len(winners) < 2:
            return False
        if not any(bool(info["has_domino"]) for info in winners):
            return False
        if not any(not bool(info["has_domino"]) for info in winners):
            return False
        if sum(bool(info["has_domino"]) for info in regions) < 2:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if row_dividers and not np.all(out[row_dividers, :] == GREY):
            return False
        if col_dividers and not np.all(out[:, col_dividers] == GREY):
            return False

        targeted: set[tuple[int, int]] = set()
        self_filled: set[tuple[int, int]] = set()
        for info in winners:
            idx = info["index"]
            if bool(info["has_domino"]):
                dr, dc = info["direction"]
                target = (int(idx[0] + dr), int(idx[1] + dc))
                if target not in lookup:
                    return False
                targeted.add(target)
                if target == idx:
                    return False
                donor_bounds = info["bounds"]
                if not _is_uniform_region(out, donor_bounds, BLACK):
                    return False
            else:
                self_filled.add(idx)
                if not _is_uniform_region(out, info["bounds"], int(info["scatter_color"])):
                    return False

        for idx in targeted:
            target_info = lookup[idx]
            if not _is_uniform_region(out, target_info["bounds"], int(target_info["scatter_color"])):
                return False

        untouched = [
            info
            for info in regions
            if info["index"] not in targeted and info["index"] not in self_filled
        ]
        if not untouched:
            return False
        if not any(_is_uniform_region(out, info["bounds"], BLACK) for info in untouched):
            return False

        filled_colors = {
            int(lookup[idx]["scatter_color"]) for idx in targeted | self_filled
        }
        if len(filled_colors) < 2:
            return False

        return True
    except Exception:
        return False


def _generate_once(counter: int) -> np.ndarray | None:
    built = _choose_structure(counter)
    if built is None:
        return None
    specs, row_sizes, col_sizes = built
    grid = _place_regions(row_sizes, col_sizes, specs)
    if grid is None:
        return None
    if max(grid.shape) > 30:
        return None
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for attempt in range(MAX_GEN_ATTEMPTS):
        counter = _GEN_COUNTER + attempt
        grid = _generate_once(counter)
        if grid is not None:
            _GEN_COUNTER = counter + 1
            return grid
    raise RuntimeError("failed to generate a valid redirected-domino counting grid")
