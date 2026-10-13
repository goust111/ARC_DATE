import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

GEN_RETRIES = 400


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    row_hits = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    col_hits = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full horizontal and one full vertical divider"
    return int(row_hits[0]), int(col_hits[0])


def _quadrant_slices(divider_r: int, divider_c: int) -> dict[tuple[int, int], tuple[slice, slice]]:
    return {
        (0, 0): (slice(0, divider_r), slice(0, divider_c)),
        (0, 1): (slice(0, divider_r), slice(divider_c + 1, None)),
        (1, 0): (slice(divider_r + 1, None), slice(0, divider_c)),
        (1, 1): (slice(divider_r + 1, None), slice(divider_c + 1, None)),
    }


def _extract_quadrants(grid: np.ndarray, divider_r: int, divider_c: int) -> dict[tuple[int, int], np.ndarray]:
    parts = {}
    for key, (rs, cs) in _quadrant_slices(divider_r, divider_c).items():
        parts[key] = grid[rs, cs].copy()
    return parts


def _infer_missing_obstacles(quadrants: dict[tuple[int, int], np.ndarray], missing: tuple[int, int]) -> np.ndarray:
    qh, qw = quadrants[(0, 0)].shape
    pred = np.full((qh, qw), BLACK, dtype=int)
    for r in range(qh):
        for c in range(qw):
            votes = []
            for pos, quad in quadrants.items():
                if pos == missing:
                    continue
                rr = r if pos[0] == missing[0] else qh - 1 - r
                cc = c if pos[1] == missing[1] else qw - 1 - c
                votes.append(int(quad[rr, cc] == GREEN))
            assert votes and len(set(votes)) == 1, "three visible quadrants must imply a unique mirrored obstacle mask"
            if votes[0]:
                pred[r, c] = GREEN
    return pred


def _obstacle_quadrant(r: int, c: int, divider_r: int, divider_c: int) -> tuple[int, int] | None:
    if r == divider_r or c == divider_c:
        return None
    return (0 if r < divider_r else 1, 0 if c < divider_c else 1)


def _grow_paths(
    grid: np.ndarray,
    divider_r: int,
    divider_c: int,
    *,
    missing: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    base = np.asarray(grid, dtype=int)
    out = base.copy()
    height, width = out.shape
    starts = sorted(int(c) for c in np.where(base[-1] == RED)[0])
    assert 2 <= len(starts) <= 3, "expected 2-3 bottom-row control pixels"

    details: list[dict[str, object]] = []
    for start_c in starts:
        r = height - 1
        c = start_c
        path = [(r, c)]
        turns = 0
        hidden_turns = 0
        hidden_events = 0
        blocked_by_hidden = False
        stopped = False

        while r > 0:
            nr = r - 1
            if nr == divider_r:
                nr -= 1
            if nr < 0:
                break

            above = int(base[nr, c])
            if above == BLACK:
                r = nr
                out[r, c] = RED
                path.append((r, c))
                continue

            if above == GREEN:
                nc = c + 1
                is_hidden = missing is not None and _obstacle_quadrant(nr, c, divider_r, divider_c) == missing
                if nc >= width or nc == divider_c or out[r, nc] != BLACK:
                    if is_hidden:
                        hidden_events += 1
                        blocked_by_hidden = True
                    stopped = True
                    break
                if is_hidden:
                    hidden_events += 1
                    hidden_turns += 1
                c = nc
                out[r, c] = RED
                path.append((r, c))
                turns += 1
                continue

            raise AssertionError("path saw a non-background, non-obstacle cell above")

        details.append(
            {
                "start_col": start_c,
                "path": path,
                "turns": turns,
                "hidden_turns": hidden_turns,
                "hidden_events": hidden_events,
                "blocked_by_hidden": blocked_by_hidden,
                "stopped": stopped,
                "reached_top": path[-1][0] == 0,
            }
        )

    return out, details


def _complete_obstacles(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    divider_r, divider_c = _find_dividers(g)
    assert divider_r > 0 and divider_c > 0, "quadrants must have positive size"
    assert divider_r == g.shape[0] - divider_r - 1
    assert divider_c == g.shape[1] - divider_c - 1

    quadrants = _extract_quadrants(g, divider_r, divider_c)
    green_counts = {pos: int(np.sum(quad == GREEN)) for pos, quad in quadrants.items()}
    missing = [pos for pos, count in green_counts.items() if count == 0]
    assert len(missing) == 1, "expected exactly one obstacle-free quadrant"
    missing_pos = missing[0]
    for pos, count in green_counts.items():
        if pos != missing_pos:
            assert 3 <= count <= 6, "each visible obstacle quadrant must contain 3-6 green pixels"

    pred = _infer_missing_obstacles(quadrants, missing_pos)
    out = g.copy()
    rs, cs = _quadrant_slices(divider_r, divider_c)[missing_pos]
    target = out[rs, cs]
    assert not np.any(target == GREEN), "missing quadrant must have no visible obstacles"
    target[pred == GREEN] = GREEN
    return out, missing_pos, divider_r, divider_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing mirrored obstacle quadrant, then grow red paths upward through that full obstacle field.

    Input structure:
    - The background is black, with one full grey horizontal divider and one full grey vertical divider splitting the grid into four equal quadrants.
    - Three quadrants contain 3-6 green obstacle pixels whose local layouts are horizontal and/or vertical mirrors of one hidden fourth layout.
    - The remaining quadrant has no green obstacles in the input.
    - Two or three red control pixels sit on the bottom row in the same half-columns as the missing quadrant.

    Rule:
    - Identify the obstacle-free quadrant and infer its green pixels by mirroring the other three quadrants within the 2x2 divider frame.
    - Draw those inferred green pixels into the empty quadrant.
    - From each bottom red control, grow a red path upward: move into the nearest non-divider cell above if it is black; if that cell is green, shift one cell right on the current row instead, but stop if that right target is out of bounds, on the vertical divider, or already non-black.
    - The horizontal divider is transparent to upward motion: skip that divider row without painting it and continue checking the next row above.

    Key insight:
    - The hidden quadrant is not decorative: its mirrored obstacle pixels are the only way to determine later right-deflections of the red paths, so symmetry completion and path growth must be solved together.
    """
    completed, missing_pos, divider_r, divider_c = _complete_obstacles(grid)
    out, _ = _grow_paths(completed, divider_r, divider_c, missing=missing_pos)
    return out


def _mirror_from_missing(mask: set[tuple[int, int]], qh: int, qw: int, missing: tuple[int, int]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    out: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for qr in (0, 1):
        for qc in (0, 1):
            coords = set()
            for r, c in mask:
                rr = r if qr == missing[0] else qh - 1 - r
                cc = c if qc == missing[1] else qw - 1 - c
                coords.add((rr, cc))
            out[(qr, qc)] = coords
    return out


def _is_asymmetric(mask: set[tuple[int, int]], qh: int, qw: int) -> bool:
    lr = {(r, qw - 1 - c) for r, c in mask}
    ud = {(qh - 1 - r, c) for r, c in mask}
    return mask != lr and mask != ud


def _choose_start_columns(qw: int, count: int, blocked_hidden: bool) -> list[int] | None:
    combos = []
    if blocked_hidden:
        special = qw - 2
        pool = list(range(0, qw - 3))
        for picks in _combinations(pool, count - 1):
            cols = list(picks) + [special]
            if _min_gap(cols) >= 2:
                combos.append(cols)
    else:
        pool = list(range(0, qw - 2))
        for cols in _combinations(pool, count):
            cols = list(cols)
            if cols[-1] <= qw - 3 and _min_gap(cols) >= 2:
                combos.append(cols)
    if not combos:
        return None
    return list(random.choice(combos))


def _combinations(values: list[int], k: int):
    if k == 0:
        yield ()
        return
    if len(values) < k:
        return
    if k == 1:
        for v in values:
            yield (v,)
        return
    for i, v in enumerate(values):
        for rest in _combinations(values[i + 1 :], k - 1):
            yield (v,) + rest


def _min_gap(values: list[int]) -> int:
    if len(values) < 2:
        return 999
    return min(b - a for a, b in zip(values, values[1:]))


def _build_hidden_mask(qh: int, qw: int, starts: list[int], blocked_hidden: bool) -> set[tuple[int, int]] | None:
    mask: set[tuple[int, int]] = set()
    first_rows: list[int] = []
    for start in starts:
        choices = list(range(2, qh - 1))
        if not choices:
            return None
        h1 = random.choice(choices)
        first_rows.append(h1)
        mask.add((h1, start))

    special_start = starts[-1]
    special_h1 = first_rows[-1]
    h2_choices = list(range(0, special_h1 - 1))
    if not h2_choices:
        return None
    h2 = random.choice(h2_choices)
    mask.add((h2, special_start + 1))

    if blocked_hidden:
        assert special_start == qw - 2
    if len(mask) < 3 or len(mask) > 6:
        return None
    if not _is_asymmetric(mask, qh, qw):
        return None
    return mask


def _paint_coords(grid: np.ndarray, coords: set[tuple[int, int]], rs: slice, cs: slice, color: int) -> None:
    patch = grid[rs, cs]
    for r, c in coords:
        patch[r, c] = color


def _global_column(local_col: int, divider_c: int, bottom_half: tuple[int, int]) -> int:
    return local_col if bottom_half[1] == 0 else divider_c + 1 + local_col


def _validate_generated_input(
    grid: np.ndarray,
    missing: tuple[int, int],
    starts: list[int],
    blocked_hidden: bool,
) -> bool:
    completed, inferred_missing, divider_r, divider_c = _complete_obstacles(grid)
    if inferred_missing != missing:
        return False

    solved, details = _grow_paths(completed, divider_r, divider_c, missing=missing)
    if len(details) != len(starts):
        return False
    if any(int(d["turns"]) < 1 for d in details):
        return False
    if sum(int(d["hidden_events"]) > 0 for d in details) < 1:
        return False
    if blocked_hidden and not any(bool(d["blocked_by_hidden"]) for d in details):
        return False
    if not blocked_hidden and any(bool(d["stopped"]) for d in details[:-1]):
        return False

    path_cells = set()
    for d in details:
        for cell in d["path"]:
            if cell in path_cells:
                return False
            path_cells.add(cell)

    naive, _ = _grow_paths(grid, divider_r, divider_c, missing=missing)
    if np.array_equal(naive == RED, solved == RED):
        return False

    out = main(grid)
    if not np.array_equal(out, solved):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(6, 9)
    qw = random.randint(6, 9)
    height = 2 * qh + 1
    width = 2 * qw + 1
    divider_r = qh
    divider_c = qw

    missing = random.choice([(0, 0), (0, 1)])
    bottom_known = (1, missing[1])
    blocked_hidden = random.choice([False, False, True])

    possible_counts = [n for n in (2, 3) if _choose_start_columns(qw, n, blocked_hidden) is not None]
    if not possible_counts:
        return None
    control_count = random.choice(possible_counts)
    starts = _choose_start_columns(qw, control_count, blocked_hidden)
    if starts is None:
        return None

    hidden_mask = _build_hidden_mask(qh, qw, starts, blocked_hidden)
    if hidden_mask is None:
        return None

    mirrored = _mirror_from_missing(hidden_mask, qh, qw, missing)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[divider_r, :] = GREY
    grid[:, divider_c] = GREY

    slices = _quadrant_slices(divider_r, divider_c)
    for pos, coords in mirrored.items():
        if pos == missing:
            continue
        rs, cs = slices[pos]
        _paint_coords(grid, coords, rs, cs, GREEN)

    for start in starts:
        c = _global_column(start, divider_c, bottom_known)
        grid[-1, c] = RED

    if not _validate_generated_input(grid, missing, starts, blocked_hidden):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-completion path-growth input")
