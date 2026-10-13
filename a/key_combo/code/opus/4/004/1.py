from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8

RNG = np.random.default_rng()
MAX_ATTEMPTS = 600


def _extract_stacked(grid: np.ndarray) -> np.ndarray:
    rows = []
    widths = set()
    for r in range(grid.shape[0]):
        blue_cols = np.flatnonzero(grid[r] == BLUE)
        teal_cols = np.flatnonzero(grid[r] == TEAL)
        if blue_cols.size == 0 and teal_cols.size == 0:
            continue
        if blue_cols.size != 1 or teal_cols.size != 1:
            raise ValueError("each marked row must contain exactly one blue and one teal marker")
        left = int(blue_cols[0])
        right = int(teal_cols[0])
        if left >= right - 1:
            raise ValueError("markers must bound a non-empty segment")
        widths.add(right - left - 1)
        rows.append(grid[r, left + 1 : right].copy())
    if not rows or len(widths) != 1:
        raise ValueError("expected at least one equal-width marked segment")
    return np.vstack(rows)


def _render_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    pr, pc = tile.shape
    row_ids = np.arange(height) % pr
    col_ids = np.arange(width) % pc
    return tile[row_ids][:, col_ids]


def _minimal_periods(full: np.ndarray) -> tuple[int, int]:
    height, width = full.shape
    row_ids = np.arange(height)
    col_ids = np.arange(width)

    for pr in range(1, height + 1):
        if np.array_equal(full, full[row_ids % pr]):
            break
    for pc in range(1, width + 1):
        if np.array_equal(full, full[:, col_ids % pc]):
            break
    return int(pr), int(pc)


def _candidate_from_period(observed: np.ndarray, pr: int, pc: int) -> dict | None:
    height, width = observed.shape
    tile = np.zeros((pr, pc), dtype=int)
    seen = np.zeros((pr, pc), dtype=bool)

    for r, c in np.argwhere(observed != BLACK):
        rr = int(r) % pr
        cc = int(c) % pc
        color = int(observed[r, c])
        if seen[rr, cc] and int(tile[rr, cc]) != color:
            return None
        tile[rr, cc] = color
        seen[rr, cc] = True

    if not seen.all():
        return None

    full = _render_from_tile(tile, (height, width))
    if np.any((observed != BLACK) & (observed != full)):
        return None

    min_pr, min_pc = _minimal_periods(full)
    if (min_pr, min_pc) != (pr, pc):
        return None

    return {"pr": int(pr), "pc": int(pc), "tile": tile, "full": full}


def _enumerate_minimal_candidates(observed: np.ndarray) -> list[dict]:
    height, width = observed.shape
    dedup: dict[bytes, dict] = {}

    for pr in range(1, height + 1):
        for pc in range(1, width + 1):
            candidate = _candidate_from_period(observed, pr, pc)
            if candidate is None:
                continue
            key = candidate["full"].tobytes()
            if key not in dedup:
                dedup[key] = candidate

    return sorted(
        dedup.values(),
        key=lambda item: (item["pr"] * item["pc"], item["pr"], item["pc"]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the marked paragraph rows, stack them, and repair every hole from the smallest shared periodic tile.

    Input structure:
    - The grid is mostly black background with a handful of non-adjacent marked rows.
    - Each marked row has one blue control pixel on the left and one teal control pixel on the right.
    - Between the markers lies an equal-width segment made of red, green, and black hole cells.

    Rule:
    - Keep only the rows that contain both control markers and extract the cells strictly between them.
    - Stack those extracted segments from top to bottom into one rectangular matrix.
    - Treat black cells inside that matrix as missing, then find the smallest row/column periods whose repeated tile matches every visible non-black cell.
    - Fill every missing cell from that inferred periodic tile and return the completed stacked matrix.

    Key insight:
    - No marked row should be solved in isolation: the control pixels define which slices to stack, and only their joint non-black constraints determine the hidden 2D periodic tile.
    """
    observed = _extract_stacked(np.asarray(grid, dtype=int))
    candidates = _enumerate_minimal_candidates(observed)
    if not candidates:
        raise ValueError("no periodic completion matches the visible cells")
    return candidates[0]["full"]


def _sample_tile(pr: int, pc: int) -> np.ndarray:
    for _ in range(300):
        tile = RNG.choice(np.array([RED, GREEN], dtype=int), size=(pr, pc))
        if len(np.unique(tile)) < 2:
            continue
        if _minimal_periods(tile) != (pr, pc):
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a non-degenerate tile")


def _coverage_counts(mask: np.ndarray, pr: int, pc: int) -> np.ndarray:
    counts = np.zeros((pr, pc), dtype=int)
    for r, c in np.argwhere(mask):
        counts[int(r) % pr, int(c) % pc] += 1
    return counts


def _sample_observation(full: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    height, width = full.shape
    by_class: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for rr in range(pr):
        for cc in range(pc):
            by_class[(rr, cc)] = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if r % pr == rr and c % pc == cc
            ]

    for _ in range(250):
        mask = np.zeros((height, width), dtype=bool)
        row_load = np.zeros(height, dtype=int)

        class_keys = list(by_class.keys())
        RNG.shuffle(class_keys)
        for key in class_keys:
            positions = by_class[key]
            min_load = min(row_load[r] for r, _ in positions)
            options = [(r, c) for r, c in positions if row_load[r] == min_load]
            r, c = options[int(RNG.integers(len(options)))]
            mask[r, c] = True
            row_load[r] += 1

        extra_prob = float(RNG.uniform(0.03, 0.14))
        mask |= RNG.random((height, width)) < extra_prob

        valid = True
        for r in range(height):
            visible = int(mask[r].sum())
            if visible < 2:
                hidden = np.flatnonzero(~mask[r])
                if hidden.size < 2 - visible:
                    valid = False
                    break
                chosen = RNG.choice(hidden, size=2 - visible, replace=False)
                mask[r, chosen] = True
                visible = int(mask[r].sum())

            limit = width - 2
            if visible > limit:
                counts = _coverage_counts(mask, pr, pc)
                removable = [
                    int(c)
                    for c in np.flatnonzero(mask[r])
                    if counts[r % pr, int(c) % pc] > 1
                ]
                need = visible - limit
                if len(removable) < need:
                    valid = False
                    break
                chosen = RNG.choice(np.array(removable, dtype=int), size=need, replace=False)
                mask[r, chosen] = False

            residue_seen = np.zeros(pc, dtype=bool)
            for c in np.flatnonzero(mask[r]):
                residue_seen[int(c) % pc] = True
            if residue_seen.all():
                counts = _coverage_counts(mask, pr, pc)
                removable_classes = []
                rr = r % pr
                for cc in range(pc):
                    cols = [int(c) for c in np.flatnonzero(mask[r]) if int(c) % pc == cc]
                    if cols and counts[rr, cc] > len(cols):
                        removable_classes.append(cols)
                if not removable_classes:
                    valid = False
                    break
                chosen_cols = removable_classes[int(RNG.integers(len(removable_classes)))]
                mask[r, chosen_cols] = False
                if int(mask[r].sum()) < 2:
                    valid = False
                    break

        if not valid:
            continue
        if not np.all(_coverage_counts(mask, pr, pc) > 0):
            continue

        observed = np.where(mask, full, BLACK)
        if _case_is_valid(observed, full, pr, pc):
            return observed
    return None


def _case_is_valid(observed: np.ndarray, full: np.ndarray, pr: int, pc: int) -> bool:
    if observed.shape != full.shape:
        return False
    if np.array_equal(observed, full):
        return False
    if np.any((observed != BLACK) & (observed != full)):
        return False
    if np.any(observed.sum(axis=1) == 0):
        return False
    if any(np.count_nonzero(row == BLACK) < 2 for row in observed):
        return False
    if not np.all(_coverage_counts(observed != BLACK, pr, pc) > 0):
        return False

    candidates = _enumerate_minimal_candidates(observed)
    if len(candidates) != 1:
        return False
    candidate = candidates[0]
    if (candidate["pr"], candidate["pc"]) != (pr, pc):
        return False
    if not np.array_equal(candidate["full"], full):
        return False

    for row in observed:
        residue_seen = np.zeros(pc, dtype=bool)
        for c in np.flatnonzero(row != BLACK):
            residue_seen[int(c) % pc] = True
        if residue_seen.all():
            return False

    return True


def _build_input(observed: np.ndarray) -> np.ndarray:
    n_rows, width = observed.shape
    top_margin = int(RNG.integers(1, 4))
    bottom_margin = int(RNG.integers(1, 4))
    gaps = RNG.integers(1, 4, size=n_rows - 1) if n_rows > 1 else np.array([], dtype=int)

    marked_rows = [top_margin]
    for gap in gaps:
        marked_rows.append(marked_rows[-1] + 1 + int(gap))

    height = marked_rows[-1] + 1 + bottom_margin
    left = int(RNG.integers(1, 5))
    right = left + width + 1
    right_margin = int(RNG.integers(1, 5))
    width_total = right + 1 + right_margin

    grid = np.zeros((height, width_total), dtype=int)
    for idx, r in enumerate(marked_rows):
        grid[r, left] = BLUE
        grid[r, right] = TEAL
        grid[r, left + 1 : right] = observed[idx]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pr = int(RNG.integers(2, 5))
        pc = int(RNG.integers(2, 5))
        n_rows = int(RNG.integers(max(4, pr + 1), 7))
        width = int(RNG.integers(2 * pc, min(12, 3 * pc) + 1))

        tile = _sample_tile(pr, pc)
        full = _render_from_tile(tile, (n_rows, width))
        observed = _sample_observation(full, pr, pc)
        if observed is None:
            continue

        grid = _build_input(observed)
        if max(grid.shape) > 30:
            continue
        if not np.array_equal(main(grid), full):
            continue
        return grid

    raise RuntimeError("failed to generate a valid stacked-period task")
