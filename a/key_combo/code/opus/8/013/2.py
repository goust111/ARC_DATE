import itertools
import random

import numpy as np


BG = 0
MARKER = 7
FG_POOL = [1, 2, 3, 4, 5, 6, 8, 9]
GEN_RETRIES = 400


def _all_orbits(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    orbits = []
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            orbit = {
                (int(r), int(c)),
                (int(h - 1 - r), int(c)),
                (int(r), int(w - 1 - c)),
                (int(h - 1 - r), int(w - 1 - c)),
            }
            orbits.append(tuple(sorted(orbit)))
    return orbits


def _rows_compatible(a: np.ndarray, b: np.ndarray, marker: int) -> bool:
    return bool(np.all((a == marker) | (b == marker) | (a == b)))


def _compress_rows(grid: np.ndarray, marker: int) -> np.ndarray:
    kept = [grid[0].copy()]
    for row in grid[1:]:
        if not _rows_compatible(kept[-1], row, marker):
            kept.append(row.copy())
    return np.stack(kept, axis=0)


def _compress_grid(grid: np.ndarray, marker: int) -> np.ndarray:
    rows = _compress_rows(np.asarray(grid, dtype=int), marker)
    cols = _compress_rows(rows.T, marker).T
    return cols


def _fill_by_symmetry(grid: np.ndarray, marker: int) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    saw_marker = False
    for orbit in _all_orbits(h, w):
        values = [int(g[r, c]) for r, c in orbit]
        known = {value for value in values if value != marker}
        if not known:
            return None
        if len(known) != 1:
            return None
        fill = known.pop()
        if any(value == marker for value in values):
            saw_marker = True
        for r, c in orbit:
            out[r, c] = fill
    if not saw_marker:
        return None
    return out


def _candidate_solutions(grid: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    out = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        compressed = _compress_grid(g, color)
        if not np.any(compressed == color):
            continue
        filled = _fill_by_symmetry(compressed, color)
        if filled is None:
            continue
        out.append((int(color), compressed, filled))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the inflated motif by treating the unique marker color as transparent, then repair its symmetry holes.

    Input structure:
    - The background is black.
    - A small bilaterally symmetric motif has been inflated by repeating each source row and source column 1 to 3 times.
    - A few single orange marker pixels replace original motif colors inside the inflated grid, breaking some otherwise identical adjacent rows and columns.

    Rule:
    - Try each non-background color as the possible marker color.
    - For each hypothesis, treat that color as a wildcard and compress top-to-bottom adjacent duplicate rows, then left-to-right adjacent duplicate columns, where wildcard cells match any color.
    - Keep the unique hypothesis whose compressed grid can be completed into both horizontal and vertical mirror symmetry.
    - In the compressed grid, fill every marker cell from the unique color already fixed elsewhere in the same symmetry orbit.

    Key insight:
    - The correct marker color is the only one that simultaneously restores the hidden row/column repetition and yields a symmetry-completable compressed motif.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _candidate_solutions(g)
    assert len(candidates) == 1, "expected exactly one valid marker-color hypothesis"
    _marker, _compressed, filled = candidates[0]
    return filled


def _build_symmetric_pattern(
    rng: random.Random,
) -> tuple[np.ndarray, list[int], list[tuple[tuple[int, int], ...]]]:
    for _ in range(300):
        # Even-sized biaxial motifs would still contain a duplicate center row/col
        # and be compressed further, so 5x5 is the only stable size in the stated range.
        h = 5
        w = 5
        fg_count = 2 if rng.random() < 0.55 else 3
        fg_colors = rng.sample(FG_POOL, fg_count)
        palette = [BG] + fg_colors
        weights = [1] + [3] * fg_count
        orbits = _all_orbits(h, w)

        grid = np.zeros((h, w), dtype=int)
        for orbit in orbits:
            color = rng.choices(palette, weights=weights, k=1)[0]
            for r, c in orbit:
                grid[r, c] = color

        if not np.any(grid == BG):
            continue
        if any(not np.any(grid == color) for color in fg_colors):
            continue
        if any(np.array_equal(grid[r], grid[r + 1]) for r in range(h - 1)):
            continue
        if any(np.array_equal(grid[:, c], grid[:, c + 1]) for c in range(w - 1)):
            continue
        if len({tuple(row.tolist()) for row in grid}) < 3:
            continue
        if len({tuple(col.tolist()) for col in grid.T}) < 3:
            continue
        return grid, fg_colors, orbits
    raise RuntimeError("failed to build a symmetric base pattern")


def _choose_repeats(rng: random.Random, h: int, w: int) -> tuple[list[int], list[int]]:
    for _ in range(300):
        row_repeats = [rng.randint(1, 3) for _ in range(h)]
        col_repeats = [rng.randint(1, 3) for _ in range(w)]
        if sum(row_repeats) > 30 or sum(col_repeats) > 30:
            continue
        if max(row_repeats) == 1 or max(col_repeats) == 1:
            continue
        if sum(rep > 1 for rep in row_repeats) < 3:
            continue
        if sum(rep > 1 for rep in col_repeats) < 3:
            continue
        if row_repeats == row_repeats[::-1] and col_repeats == col_repeats[::-1]:
            continue
        return row_repeats, col_repeats
    raise RuntimeError("failed to choose repeat factors")


def _hole_options(
    orbit: tuple[tuple[int, int], ...], row_repeats: list[int], col_repeats: list[int]
) -> list[tuple[tuple[int, int], ...]]:
    eligible = [
        cell
        for cell in orbit
        if row_repeats[cell[0]] > 1 and col_repeats[cell[1]] > 1
    ]
    options = set()
    for size in (1, 2):
        for subset in itertools.combinations(eligible, size):
            if size >= len(orbit):
                continue
            rows = {r for r, _ in subset}
            cols = {c for _, c in subset}
            if len(rows) != size or len(cols) != size:
                continue
            options.add(tuple(sorted(subset)))
    return sorted(options, key=lambda subset: (len(subset), subset))


def _pick_holes(
    rng: random.Random,
    orbits: list[tuple[tuple[int, int], ...]],
    row_repeats: list[int],
    col_repeats: list[int],
) -> list[tuple[int, int]] | None:
    eligible_rows = {idx for idx, rep in enumerate(row_repeats) if rep > 1}
    eligible_cols = {idx for idx, rep in enumerate(col_repeats) if rep > 1}
    max_holes = min(4, len(eligible_rows), len(eligible_cols))
    if max_holes < 2:
        return None

    per_orbit = []
    for orbit in orbits:
        options = _hole_options(orbit, row_repeats, col_repeats)
        if options:
            rng.shuffle(options)
            per_orbit.append((orbit, options))
    rng.shuffle(per_orbit)
    if not per_orbit:
        return None

    max_gain = [0] * (len(per_orbit) + 1)
    for idx in range(len(per_orbit) - 1, -1, -1):
        max_here = max(len(option) for option in per_orbit[idx][1])
        max_gain[idx] = max_gain[idx + 1] + max_here

    targets = list(range(2, max_holes + 1))
    rng.shuffle(targets)

    def search(
        idx: int,
        target: int,
        used_rows: set[int],
        used_cols: set[int],
        chosen: list[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        if len(chosen) == target:
            return chosen.copy()
        if idx == len(per_orbit):
            return None
        if len(chosen) + max_gain[idx] < target:
            return None

        orbit, options = per_orbit[idx]
        for option in options:
            rows = {r for r, _ in option}
            cols = {c for _, c in option}
            if rows & used_rows or cols & used_cols:
                continue
            result = search(
                idx + 1,
                target,
                used_rows | rows,
                used_cols | cols,
                chosen + list(option),
            )
            if result is not None:
                return result
        return search(idx + 1, target, used_rows, used_cols, chosen)

    for target in targets:
        picked = search(0, target, set(), set(), [])
        if picked is not None:
            return picked
    return None


def _expand_grid(base: np.ndarray, row_repeats: list[int], col_repeats: list[int]) -> np.ndarray:
    rows = np.repeat(base, row_repeats, axis=0)
    return np.repeat(rows, col_repeats, axis=1)


def _group_starts(repeats: list[int]) -> list[int]:
    starts = []
    pos = 0
    for rep in repeats:
        starts.append(pos)
        pos += rep
    return starts


def _add_decoy_markers(
    expanded: np.ndarray,
    rng: random.Random,
    row_repeats: list[int],
    col_repeats: list[int],
    existing: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    row_starts = _group_starts(row_repeats)
    col_starts = _group_starts(col_repeats)
    used_rows = {r for r, _ in existing}
    used_cols = {c for _, c in existing}

    candidates = []
    for br, r0 in enumerate(row_starts):
        for bc, c0 in enumerate(col_starts):
            for dr in range(row_repeats[br]):
                for dc in range(col_repeats[bc]):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r0 + dr
                    cc = c0 + dc
                    candidates.append((rr, cc))
    rng.shuffle(candidates)

    max_total = min(6, expanded.shape[0], expanded.shape[1])
    totals = list(range(max(3, len(existing)), max_total + 1))
    rng.shuffle(totals)
    for total in totals:
        need = total - len(existing)
        picked = []
        cur_rows = set(used_rows)
        cur_cols = set(used_cols)
        for rr, cc in candidates:
            if rr in cur_rows or cc in cur_cols:
                continue
            picked.append((rr, cc))
            cur_rows.add(rr)
            cur_cols.add(cc)
            if len(picked) == need:
                return existing + picked
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base_out, fg_colors, orbits = _build_symmetric_pattern(rng)
    h, w = base_out.shape
    row_repeats, col_repeats = _choose_repeats(rng, h, w)
    holes = _pick_holes(rng, orbits, row_repeats, col_repeats)
    if holes is None:
        return None

    hole_set = set(holes)
    touched_orbits = {
        orbit
        for orbit in orbits
        if any(cell in hole_set for cell in orbit)
    }
    for color in fg_colors:
        has_intact_orbit = any(
            int(base_out[orbit[0]]) == int(color) and orbit not in touched_orbits
            for orbit in orbits
        )
        if not has_intact_orbit:
            return None

    expanded = _expand_grid(base_out, row_repeats, col_repeats)
    row_starts = _group_starts(row_repeats)
    col_starts = _group_starts(col_repeats)
    marker_coords = []
    for r, c in holes:
        rr = row_starts[r]
        cc = col_starts[c]
        expanded[rr, cc] = MARKER
        marker_coords.append((rr, cc))

    marker_coords = _add_decoy_markers(
        expanded, rng, row_repeats, col_repeats, marker_coords
    )
    if marker_coords is None:
        return None
    for rr, cc in marker_coords:
        expanded[rr, cc] = MARKER

    if len(marker_coords) != len(set(marker_coords)):
        return None
    if len({r for r, _ in marker_coords}) != len(marker_coords):
        return None
    if len({c for _, c in marker_coords}) != len(marker_coords):
        return None

    compressed = _compress_grid(expanded, MARKER)
    expected_compressed = base_out.copy()
    for r, c in holes:
        expected_compressed[r, c] = MARKER
    if not np.array_equal(compressed, expected_compressed):
        return None

    expected_out = _fill_by_symmetry(expected_compressed, MARKER)
    if expected_out is None or not np.array_equal(expected_out, base_out):
        return None

    candidates = _candidate_solutions(expanded)
    if len(candidates) != 1 or candidates[0][0] != MARKER:
        return None
    if not np.array_equal(candidates[0][1], expected_compressed):
        return None
    if not np.array_equal(candidates[0][2], base_out):
        return None
    if not np.array_equal(main(expanded), base_out):
        return None
    if expanded.shape[0] <= h or expanded.shape[1] <= w:
        return None
    return expanded


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid puzzle after {GEN_RETRIES} attempts")
