from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 400
SPRITE_RETRIES = 200


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    return comps


def divider_lines(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    rows = [r for r in range(g.shape[0]) if int(g[r, 0]) != bg and np.all(g[r] == g[r, 0])]
    cols = [c for c in range(g.shape[1]) if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])]
    return rows, cols


def parse_case(grid: np.ndarray) -> tuple[int, list[int], list[int], tuple[int, int], list[tuple[int, int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    row_lines, col_lines = divider_lines(g, bg)

    divider_mask = np.zeros_like(g, dtype=bool)
    if row_lines:
        divider_mask[row_lines, :] = True
    if col_lines:
        divider_mask[:, col_lines] = True

    comps = component_coords((g != bg) & ~divider_mask)
    source_anchor: tuple[int, int] | None = None
    source_items: list[tuple[int, int, int]] | None = None
    targets: list[tuple[int, int]] = []

    for coords in comps:
        vals = [int(g[r, c]) for r, c in coords]
        grey_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == GREY]
        if len(coords) == 1 and vals[0] == GREY:
            targets.append((int(coords[0, 0]), int(coords[0, 1])))
            continue
        if len(grey_cells) == 1:
            if source_anchor is not None:
                raise AssertionError("expected exactly one source sprite")
            source_anchor = grey_cells[0]
            sr, sc = source_anchor
            source_items = [
                (int(r - sr), int(c - sc), int(g[r, c]))
                for r, c in coords
            ]
            continue
        raise AssertionError("unexpected non-divider foreground object")

    if source_anchor is None or source_items is None:
        raise AssertionError("missing source sprite")
    if not targets:
        raise AssertionError("missing target markers")
    return bg, row_lines, col_lines, source_anchor, source_items, sorted(targets)


def transformed_cells(
    anchor: tuple[int, int],
    items: list[tuple[int, int, int]],
    flip_r: bool,
    flip_c: bool,
) -> list[tuple[int, int, int]]:
    ar, ac = anchor
    sr = -1 if flip_r else 1
    sc = -1 if flip_c else 1
    return [(ar + sr * dr, ac + sc * dc, color) for dr, dc, color in items]


def line_cross_count(lines: list[int], a: int, b: int) -> int:
    lo = min(int(a), int(b))
    hi = max(int(a), int(b))
    return sum(lo < x < hi for x in lines)


def paint_cells(grid: np.ndarray, cells: list[tuple[int, int, int]]) -> None:
    for r, c, color in cells:
        grid[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the anchored multicolor sprite onto every gray target, flipping once per crossed divider line.

    Input structure:
    - One 4-connected multicolor source sprite contains exactly one gray anchor pixel.
    - Full-row and full-column divider lines cut the background into rectangular compartments.
    - Additional isolated gray single-pixel markers sit on background cells away from the source.

    Rule:
    - Keep the source sprite and all divider lines unchanged.
    - For each isolated gray marker, place a copy of the entire source sprite so the source anchor lands on that marker.
    - Count horizontal divider rows and vertical divider columns strictly between the source anchor and the target marker.
    - An odd number of crossed horizontal dividers flips the copy top-bottom; an odd number of crossed vertical dividers flips it left-right, always around the anchor pixel.

    Key insight:
    - The gray markers specify destination only; the copy orientation is determined by divider-parity between the source anchor and each target.
    """
    g = np.asarray(grid, dtype=int)
    _bg, row_lines, col_lines, source_anchor, source_items, targets = parse_case(g)
    sr, sc = source_anchor

    out = g.copy()
    for tr, tc in targets:
        flip_r = bool(line_cross_count(row_lines, sr, tr) % 2)
        flip_c = bool(line_cross_count(col_lines, sc, tc) % 2)
        paint_cells(out, transformed_cells((tr, tc), source_items, flip_r, flip_c))
    return out


def band_intervals(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    lines: list[int] = []
    cur = 0
    for i, size in enumerate(sizes):
        start = cur
        end = cur + size - 1
        bands.append((start, end))
        cur = end + 1
        if i < len(sizes) - 1:
            lines.append(cur)
            cur += 1
    return bands, lines


def safe_bounds(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    rb: int,
    cb: int,
) -> tuple[int, int, int, int]:
    r0, r1 = row_bands[rb]
    c0, c1 = col_bands[cb]
    if rb > 0:
        r0 += 1
    if rb < len(row_bands) - 1:
        r1 -= 1
    if cb > 0:
        c0 += 1
    if cb < len(col_bands) - 1:
        c1 -= 1
    return r0, r1, c0, c1


def signature(items: list[tuple[int, int, int]], flip_r: bool, flip_c: bool) -> dict[tuple[int, int], int]:
    sr = -1 if flip_r else 1
    sc = -1 if flip_c else 1
    return {(sr * dr, sc * dc): color for dr, dc, color in items}


def generate_sprite(colors: list[int]) -> list[tuple[int, int, int]]:
    for _ in range(SPRITE_RETRIES):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        anchor = (random.randrange(h), random.randrange(w))
        cells = {anchor}
        target_area = random.randint(5, min(7, h * w))

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue

        non_anchor = [cell for cell in sorted(cells) if cell != anchor]
        color_count = random.randint(2, min(4, len(colors), len(non_anchor)))
        base_colors = random.sample(colors, color_count)
        assigned = base_colors[:]
        while len(assigned) < len(non_anchor):
            assigned.append(random.choice(base_colors))
        random.shuffle(assigned)

        ar, ac = anchor
        items = [(0, 0, GREY)]
        for (r, c), color in zip(non_anchor, assigned):
            items.append((r - ar, c - ac, int(color)))

        sig = signature(items, False, False)
        if sig == signature(items, True, False):
            continue
        if sig == signature(items, False, True):
            continue
        if sig == signature(items, True, True):
            continue
        return sorted(items)

    raise RuntimeError("failed to build asymmetric sprite")


def anchor_candidates(
    bounds: tuple[int, int, int, int],
    items: list[tuple[int, int, int]],
    flip_r: bool,
    flip_c: bool,
) -> list[tuple[int, int]]:
    trs = transformed_cells((0, 0), items, flip_r, flip_c)
    drs = [r for r, _c, _color in trs]
    dcs = [c for _r, c, _color in trs]
    r0, r1, c0, c1 = bounds
    ar0 = r0 - min(drs)
    ar1 = r1 - max(drs)
    ac0 = c0 - min(dcs)
    ac1 = c1 - max(dcs)
    if ar0 > ar1 or ac0 > ac1:
        return []
    return [(r, c) for r in range(ar0, ar1 + 1) for c in range(ac0, ac1 + 1)]


def cell_set(cells: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c, _color in cells}


def touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r, c) in b:
            return True
        for nbr in neighbors4(r, c):
            if nbr in b:
                return True
    return False


def isolated_from(anchor: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if anchor in occupied:
        return False
    return all(nbr not in occupied for nbr in neighbors4(*anchor))


def choose_same_compartment_target(
    bounds: tuple[int, int, int, int],
    items: list[tuple[int, int, int]],
    source_anchor: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[int, int, int]]] | None:
    source_cells = transformed_cells(source_anchor, items, False, False)
    source_set = cell_set(source_cells)
    candidates = anchor_candidates(bounds, items, False, False)
    random.shuffle(candidates)
    for anchor in candidates:
        if not isolated_from(anchor, source_set):
            continue
        copy_cells = transformed_cells(anchor, items, False, False)
        copy_set = cell_set(copy_cells)
        if touches4(source_set, copy_set):
            continue
        return anchor, copy_cells
    return None


def build_band_sizes(count: int, source_index: int) -> list[int]:
    if count == 2:
        sizes = [random.randint(8, 11) for _ in range(2)]
        sizes[source_index] = random.randint(10, 12)
        return sizes
    sizes = [random.randint(6, 8) for _ in range(3)]
    sizes[source_index] = random.randint(9, 11)
    return sizes


def required_targets(nr: int, nc: int, sr: int, sc: int) -> list[tuple[str, int, int]]:
    odd_r = sr - 1 if sr > 0 else sr + 1
    odd_c = sc - 1 if sc > 0 else sc + 1
    specs = [
        ("same", sr, sc),
        ("odd_c", sr, odd_c),
        ("odd_r", odd_r, sc),
        ("odd_both", odd_r, odd_c),
    ]
    if nc == 3:
        even_c = 0 if sc == 2 else 2
        specs.append(("even_c", sr, even_c))
    if nr == 3:
        even_r = 0 if sr == 2 else 2
        specs.append(("even_r", even_r, sc))
    if nr == 3 and nc == 3 and random.random() < 0.5:
        even_r = 0 if sr == 2 else 2
        even_c = 0 if sc == 2 else 2
        specs.append(("even_both", even_r, even_c))
    return specs


def generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GENERATION_RETRIES):
        n_row_bands = 3 if random.random() < 0.6 else 2
        n_col_bands = 3 if random.random() < 0.6 else 2
        source_rb = 0 if random.random() < 0.5 else n_row_bands - 1
        source_cb = 0 if random.random() < 0.5 else n_col_bands - 1

        row_sizes = build_band_sizes(n_row_bands, source_rb)
        col_sizes = build_band_sizes(n_col_bands, source_cb)
        row_bands, row_lines = band_intervals(row_sizes)
        col_bands, col_lines = band_intervals(col_sizes)
        h = row_bands[-1][1] + 1
        w = col_bands[-1][1] + 1

        divider_color = random.choice(PALETTE)
        sprite_palette = [c for c in PALETTE if c != divider_color]
        sprite_items = generate_sprite(sprite_palette)

        inp = np.zeros((h, w), dtype=int)
        for r in row_lines:
            inp[r, :] = divider_color
        for c in col_lines:
            inp[:, c] = divider_color

        source_bounds = safe_bounds(row_bands, col_bands, source_rb, source_cb)
        source_candidates = anchor_candidates(source_bounds, sprite_items, False, False)
        random.shuffle(source_candidates)
        if not source_candidates:
            continue

        source_anchor: tuple[int, int] | None = None
        same_target_anchor: tuple[int, int] | None = None
        same_copy_cells: list[tuple[int, int, int]] | None = None
        for cand in source_candidates:
            picked = choose_same_compartment_target(source_bounds, sprite_items, cand)
            if picked is None:
                continue
            source_anchor = cand
            same_target_anchor, same_copy_cells = picked
            break
        if source_anchor is None or same_target_anchor is None or same_copy_cells is None:
            continue

        source_cells = transformed_cells(source_anchor, sprite_items, False, False)
        paint_cells(inp, source_cells)

        target_specs = required_targets(n_row_bands, n_col_bands, source_rb, source_cb)
        target_anchors: list[tuple[int, int, str, bool, bool]] = []
        placed_output_sets = {frozenset(cell_set(source_cells)), frozenset(cell_set(same_copy_cells))}
        target_anchor_cells = {same_target_anchor}
        target_anchors.append((same_target_anchor[0], same_target_anchor[1], "same", False, False))

        ok = True
        for label, rb, cb in target_specs:
            if label == "same":
                continue
            flip_r = (abs(rb - source_rb) % 2) == 1
            flip_c = (abs(cb - source_cb) % 2) == 1
            bounds = safe_bounds(row_bands, col_bands, rb, cb)
            candidates = anchor_candidates(bounds, sprite_items, flip_r, flip_c)
            random.shuffle(candidates)
            picked_anchor = None
            picked_set: set[tuple[int, int]] | None = None
            for anchor in candidates:
                if anchor in target_anchor_cells or not isolated_from(anchor, target_anchor_cells):
                    continue
                copy_cells = transformed_cells(anchor, sprite_items, flip_r, flip_c)
                copy_set = cell_set(copy_cells)
                if frozenset(copy_set) in placed_output_sets:
                    continue
                picked_anchor = anchor
                picked_set = copy_set
                break
            if picked_anchor is None or picked_set is None:
                ok = False
                break
            target_anchor_cells.add(picked_anchor)
            placed_output_sets.add(frozenset(picked_set))
            target_anchors.append((picked_anchor[0], picked_anchor[1], label, flip_r, flip_c))

        if not ok:
            continue

        for tr, tc, _label, _fr, _fc in target_anchors:
            inp[tr, tc] = GREY

        out = inp.copy()
        for tr, tc, _label, flip_r, flip_c in target_anchors:
            paint_cells(out, transformed_cells((tr, tc), sprite_items, flip_r, flip_c))

        try:
            solved = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(solved, out):
            continue

        bg, parsed_rows, parsed_cols, parsed_anchor, parsed_items, parsed_targets = parse_case(inp)
        if bg != 0:
            continue
        if parsed_rows != row_lines or parsed_cols != col_lines:
            continue
        if parsed_anchor != source_anchor or sorted(parsed_items) != sorted(sprite_items):
            continue
        if len(parsed_targets) != len(target_anchors):
            continue

        labels_seen = {label for _tr, _tc, label, _fr, _fc in target_anchors}
        if not {"same", "odd_c", "odd_r", "odd_both"} <= labels_seen:
            continue
        if n_col_bands == 3 and "even_c" not in labels_seen:
            continue
        if n_row_bands == 3 and "even_r" not in labels_seen:
            continue

        return inp, out

    raise RuntimeError("failed to generate a valid case")


def generate_input() -> np.ndarray:
    return generate_case()[0]
