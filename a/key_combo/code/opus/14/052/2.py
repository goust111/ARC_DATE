from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 600
ERASE_RADIUS = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _separator_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(candidates) == 1, "expected exactly one solid grey separator column"
    sep = int(candidates[0])
    assert sep * 2 + 1 == grid.shape[1], "expected equal-size side panels"
    return sep


def _extract_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    left_mask = left == RED
    right_mask = right == BLUE

    blockers: set[tuple[int, int]] = set()
    for r, c in np.argwhere(left == YELLOW):
        blockers.add((int(r), int(c)))
    for r, c in np.argwhere(right == YELLOW):
        blockers.add((int(r), int(c)))
    return left_mask, right_mask, sorted(blockers)


def _tile_seed(seed_mask: np.ndarray) -> tuple[np.ndarray, int]:
    box = _bbox(seed_mask)
    assert box is not None, "symmetric difference must be non-empty"
    r0, _c0, r1, _c1 = box
    period = int(r1 - r0 + 1)
    assert period >= 2, "seed bbox height must be at least 2"

    h, w = seed_mask.shape
    tiled = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(seed_mask):
        r = int(r)
        c = int(c)
        rr = r
        while rr >= 0:
            tiled[rr, c] = True
            rr -= period
        rr = r + period
        while rr < h:
            tiled[rr, c] = True
            rr += period
    return tiled, period


def _erase_by_blockers(mask: np.ndarray, blockers: list[tuple[int, int]]) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    for r, c in blockers:
        r0 = max(0, int(r) - ERASE_RADIUS)
        r1 = min(h - 1, int(r) + ERASE_RADIUS)
        c0 = max(0, int(c) - ERASE_RADIUS)
        c1 = min(w - 1, int(c) + ERASE_RADIUS)
        out[r0 : r1 + 1, c0 : c1 + 1] = False
    return out


def _render_output(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the XOR seed between matched side panels, tile it vertically by its bbox height, then cut holes around blockers.

    Input structure:
    - A black grid is split into two equal-width panels by one solid grey separator column.
    - The left panel contains one 4-connected red shape, the right panel contains one 4-connected blue shape, and the two shapes are compared in shared panel coordinates.
    - Two or three yellow blocker pixels may appear inside either panel; their panel-local coordinates matter, but the separator and original panel contents do not appear in the output.

    Rule:
    - Compare the red and blue shapes cell-by-cell in panel coordinates and keep exactly the symmetric-difference cells as a green seed mask.
    - Let H be the seed mask's bbox height; copy every seed cell upward and downward in steps of H so the pattern fills the full output height while keeping the same columns and vertical phase.
    - Project every yellow blocker into the single-panel coordinate frame and erase all tiled green cells within Chebyshev distance at most 2 of each blocker.

    Key insight:
    - The two panel shapes only define the seed through XOR, and the blocker pixels act afterward on the already tiled pattern rather than limiting the seed itself.
    """
    g = np.asarray(grid, dtype=int)
    left_mask, right_mask, blockers = _extract_masks(g)
    seed_mask = left_mask ^ right_mask
    tiled, _period = _tile_seed(seed_mask)
    clipped = _erase_by_blockers(tiled, blockers)
    return _render_output(clipped)


def _chebyshev_hits(mask: np.ndarray, cell: tuple[int, int]) -> bool:
    r, c = cell
    h, w = mask.shape
    r0 = max(0, r - ERASE_RADIUS)
    r1 = min(h - 1, r + ERASE_RADIUS)
    c0 = max(0, c - ERASE_RADIUS)
    c1 = min(w - 1, c + ERASE_RADIUS)
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _chebyshev_hits_only_repeated(seed_mask: np.ndarray, tiled_mask: np.ndarray, cell: tuple[int, int]) -> bool:
    repeated_only = tiled_mask & ~seed_mask
    return _chebyshev_hits(repeated_only, cell) and not _chebyshev_hits(seed_mask, cell)


def _shared_staircase(path_cols: list[int]) -> tuple[set[tuple[int, int]], list[list[int]]]:
    shared: set[tuple[int, int]] = {(0, int(path_cols[0]))}
    row_cols: dict[int, set[int]] = {0: {int(path_cols[0])}}
    for r in range(len(path_cols) - 1):
        cur = int(path_cols[r])
        nxt = int(path_cols[r + 1])
        if nxt != cur:
            shared.add((r, nxt))
            row_cols.setdefault(r, set()).add(nxt)
        shared.add((r + 1, nxt))
        row_cols.setdefault(r + 1, set()).add(nxt)
    ordered = [sorted(row_cols.get(r, set())) for r in range(len(path_cols))]
    return shared, ordered


def _sample_path_cols(rng: np.random.Generator, motif_h: int, motif_w: int) -> list[int]:
    if motif_w == 5:
        return [2] * motif_h
    cols = [int(rng.integers(2, motif_w - 2))]
    for _ in range(1, motif_h):
        step = int(rng.choice([-1, 0, 1]))
        cols.append(int(np.clip(cols[-1] + step, 2, motif_w - 3)))
    return cols


def _sample_attachment_rows(
    rng: np.random.Generator,
    motif_h: int,
    shared_size: int,
) -> tuple[set[int], set[int]]:
    min_count = max(1, 5 - shared_size)
    max_count = min(motif_h, 12 - shared_size)
    if max_count < min_count:
        raise ValueError("attachment counts cannot satisfy panel size bounds")
    left_count = int(rng.integers(min_count, max_count + 1))
    right_count = int(rng.integers(min_count, max_count + 1))

    top_side = "L" if rng.random() < 0.5 else "R"
    bottom_side = "R" if top_side == "L" else "L"
    left_rows: set[int] = {0} if top_side == "L" else {motif_h - 1}
    right_rows: set[int] = {0} if top_side == "R" else {motif_h - 1}

    while len(left_rows) < left_count:
        left_rows.add(int(rng.integers(0, motif_h)))
    while len(right_rows) < right_count:
        right_rows.add(int(rng.integers(0, motif_h)))
    return left_rows, right_rows


def _build_example(rng: np.random.Generator) -> np.ndarray | None:
    panel_h = int(rng.integers(10, 15))
    panel_w = int(rng.integers(10, 14))
    motif_h = int(rng.integers(3, 6))
    motif_w = int(rng.integers(5, 7))
    if motif_h >= panel_h or motif_w >= panel_w:
        return None

    top = int(rng.integers(1, panel_h - motif_h))
    left = int(rng.integers(1, panel_w - motif_w))
    path_cols = _sample_path_cols(rng, motif_h, motif_w)
    shared_local, row_cols = _shared_staircase(path_cols)
    shared_size = len(shared_local)

    try:
        left_rows, right_rows = _sample_attachment_rows(rng, motif_h, shared_size)
    except ValueError:
        return None

    left_only_local: set[tuple[int, int]] = set()
    right_only_local: set[tuple[int, int]] = set()
    for r in left_rows:
        row = row_cols[r]
        left_only_local.add((r, min(row) - 1))
    for r in right_rows:
        row = row_cols[r]
        right_only_local.add((r, max(row) + 1))

    local_w = {
        c for _r, c in shared_local | left_only_local | right_only_local
    }
    if min(local_w) < 0 or max(local_w) >= motif_w:
        return None

    shared = {(top + r, left + c) for r, c in shared_local}
    left_only = {(top + r, left + c) for r, c in left_only_local}
    right_only = {(top + r, left + c) for r, c in right_only_local}
    left_shape = shared | left_only
    right_shape = shared | right_only

    left_mask = np.zeros((panel_h, panel_w), dtype=bool)
    right_mask = np.zeros((panel_h, panel_w), dtype=bool)
    for r, c in left_shape:
        left_mask[r, c] = True
    for r, c in right_shape:
        right_mask[r, c] = True

    if not (_is_connected(left_mask) and _is_connected(right_mask)):
        return None
    if not (5 <= int(left_mask.sum()) <= 12 and 5 <= int(right_mask.sum()) <= 12):
        return None

    seed_mask = left_mask ^ right_mask
    box = _bbox(seed_mask)
    if box is None or box[0] != top or box[2] != top + motif_h - 1:
        return None
    tiled_mask, _period = _tile_seed(seed_mask)
    if not np.any(tiled_mask & ~seed_mask):
        return None

    right_special = [
        (r, c)
        for r in range(panel_h)
        for c in range(panel_w)
        if not right_mask[r, c] and _chebyshev_hits_only_repeated(seed_mask, tiled_mask, (r, c))
    ]
    left_generic = [
        (r, c)
        for r in range(panel_h)
        for c in range(panel_w)
        if not left_mask[r, c] and _chebyshev_hits(tiled_mask, (r, c))
    ]
    if not right_special or not left_generic:
        return None

    right_block = right_special[int(rng.integers(0, len(right_special)))]
    left_choices = [cell for cell in left_generic if cell != right_block]
    if not left_choices:
        return None
    left_block = left_choices[int(rng.integers(0, len(left_choices)))]

    blockers: list[tuple[str, tuple[int, int]]] = [("L", left_block), ("R", right_block)]
    want_third = bool(rng.random() < 0.55)
    if want_third:
        third_pool: list[tuple[str, tuple[int, int]]] = []
        used_local = {left_block, right_block}
        for side, mask in (("L", left_mask), ("R", right_mask)):
            for r in range(panel_h):
                for c in range(panel_w):
                    if (r, c) in used_local or mask[r, c]:
                        continue
                    if _chebyshev_hits(tiled_mask, (r, c)):
                        third_pool.append((side, (r, c)))
        if third_pool:
            blockers.append(third_pool[int(rng.integers(0, len(third_pool)))])

    local_blockers = sorted({cell for _side, cell in blockers})
    clipped = _erase_by_blockers(tiled_mask, local_blockers)
    if not clipped.any() or np.array_equal(clipped, tiled_mask):
        return None

    grid = np.zeros((panel_h, panel_w * 2 + 1), dtype=int)
    grid[:, panel_w] = GREY
    grid[:, :panel_w][left_mask] = RED
    grid[:, panel_w + 1 :][right_mask] = BLUE
    for side, (r, c) in blockers:
        gc = c if side == "L" else panel_w + 1 + c
        if grid[r, gc] != BLACK:
            return None
        grid[r, gc] = YELLOW

    if not _validate_example(grid):
        return None
    return grid


def _validate_example(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(np.unique(g)).issubset({BLACK, BLUE, RED, YELLOW, GREY}):
        return False

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    left_mask = left == RED
    right_mask = right == BLUE
    if not (_is_connected(left_mask) and _is_connected(right_mask)):
        return False
    if not (5 <= int(left_mask.sum()) <= 12 and 5 <= int(right_mask.sum()) <= 12):
        return False

    blockers = sorted({tuple(map(int, rc)) for rc in np.argwhere(left == YELLOW)} | {tuple((int(r), int(c))) for r, c in np.argwhere(right == YELLOW)})
    if len(blockers) not in {2, 3}:
        return False
    if len(np.argwhere(left == YELLOW)) == 0 or len(np.argwhere(right == YELLOW)) == 0:
        return False

    overlap = left_mask & right_mask
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    if not overlap.any() or not left_only.any() or not right_only.any():
        return False

    seed_mask = left_mask ^ right_mask
    box = _bbox(seed_mask)
    if box is None or box[2] - box[0] + 1 < 2:
        return False
    tiled_mask, _period = _tile_seed(seed_mask)
    if not np.any(tiled_mask & ~seed_mask):
        return False

    if not any(_chebyshev_hits_only_repeated(seed_mask, tiled_mask, cell) for cell in blockers):
        return False
    clipped = _erase_by_blockers(tiled_mask, blockers)
    if not clipped.any() or np.array_equal(clipped, tiled_mask):
        return False

    out = main(g)
    if out.shape != left.shape:
        return False
    if not set(np.unique(out)).issubset({BLACK, GREEN}):
        return False
    return np.array_equal(out, _render_output(clipped))


def generate_input() -> np.ndarray:
    seen = getattr(generate_input, "_seen_outputs", set())
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_example(rng)
        if grid is None:
            continue
        out = main(grid)
        sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)
        generate_input._seen_outputs = seen
        return grid
    raise RuntimeError("failed to generate a valid non-duplicate example")
