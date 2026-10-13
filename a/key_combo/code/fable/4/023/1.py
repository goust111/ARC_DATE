import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5

GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _find_separators(grid: np.ndarray) -> tuple[int, int]:
    grey_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    grey_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert grey_rows.size == 1 and grey_cols.size == 1, "expected one full grey row and one full grey column"
    sep_r = int(grey_rows[0])
    sep_c = int(grey_cols[0])
    assert 0 < sep_r < grid.shape[0] - 1 and 0 < sep_c < grid.shape[1] - 1
    return sep_r, sep_c


def _read_signatures(grid: np.ndarray, sep_r: int, sep_c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main = grid[sep_r + 1 :, sep_c + 1 :]
    row_sig = np.count_nonzero(grid[sep_r + 1 :, :sep_c] == YELLOW, axis=1).astype(int)
    col_sig = np.count_nonzero(grid[:sep_r, sep_c + 1 :] == YELLOW, axis=0).astype(int)
    assert main.shape == (row_sig.size, col_sig.size), "bands must align to the main area"
    return main, row_sig, col_sig


def _project(coords: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    row_sig = np.bincount(coords[:, 0], minlength=shape[0]).astype(int)
    col_sig = np.bincount(coords[:, 1], minlength=shape[1]).astype(int)
    return row_sig, col_sig


def _match_components(
    main: np.ndarray, row_sig: np.ndarray, col_sig: np.ndarray
) -> tuple[list[np.ndarray], int, int]:
    comps = _components(main != BLACK)
    assert comps, "expected at least one interior sprite"

    joint_matches: list[np.ndarray] = []
    row_matches = 0
    col_matches = 0
    for coords in comps:
        comp_row_sig, comp_col_sig = _project(coords, main.shape)
        row_ok = np.array_equal(comp_row_sig, row_sig)
        col_ok = np.array_equal(comp_col_sig, col_sig)
        row_matches += int(row_ok)
        col_matches += int(col_ok)
        if row_ok and col_ok:
            joint_matches.append(coords)
    return joint_matches, row_matches, col_matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the unique interior sprite whose row and column projections match the yellow histogram bands.

    Input structure:
    - Black background with one full grey separator row and one full grey separator column, leaving a black top-left corner panel.
    - Yellow bars above and left of the main area encode per-column and per-row counts aligned to the main-area coordinates.
    - The main area contains several disconnected same-color sprites on black.

    Rule:
    - Read the top band as the target sprite's absolute column-count signature and the left band as its absolute row-count signature.
    - For each interior sprite, count its occupied cells in every main-area row and every main-area column.
    - Output an all-black grid of the same size, keeping only the unique sprite whose two signatures both match the bands.

    Key insight:
    - Each yellow band is only one projection of the target; distractor sprites can match a single axis, so only the joint row-and-column signature identifies the answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_r, sep_c = _find_separators(g)
    main_area, row_sig, col_sig = _read_signatures(g, sep_r, sep_c)
    joint_matches, _, _ = _match_components(main_area, row_sig, col_sig)
    assert len(joint_matches) == 1, "expected exactly one sprite to match both projections"

    coords = joint_matches[0]
    out = np.zeros_like(g)
    global_rows = coords[:, 0] + sep_r + 1
    global_cols = coords[:, 1] + sep_c + 1
    out[global_rows, global_cols] = g[global_rows, global_cols]
    return out


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        height = int(rng.integers(2, 6))
        width = int(rng.integers(2, 6))
        max_area = min(height * width, 10)
        target_area = int(rng.integers(4, max_area + 1))

        mask = np.zeros((height, width), dtype=bool)
        start = (int(rng.integers(height)), int(rng.integers(width)))
        mask[start] = True

        while int(mask.sum()) < target_area:
            frontier = set()
            for rr, cc in np.argwhere(mask):
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            nr, nc = list(frontier)[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True

        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.sum() < 4:
            continue
        return mask
    raise RuntimeError("failed to generate a connected mask")


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, *, padding: int = 1) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False

    rr, cc = np.nonzero(mask)
    rr = rr + int(r0)
    cc = cc + int(c0)
    if occupied[rr, cc].any():
        return False

    if padding <= 0:
        return True

    for dr in range(-padding, padding + 1):
        for dc in range(-padding, padding + 1):
            nr = rr + dr
            nc = cc + dc
            in_bounds = (0 <= nr) & (nr < h) & (0 <= nc) & (nc < w)
            if occupied[nr[in_bounds], nc[in_bounds]].any():
                return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[rr + int(r0), cc + int(c0)] = int(color)


def _make_main_area(rng: np.random.Generator) -> np.ndarray:
    target_mask = _random_connected_mask(rng)
    row_decoy_mask = np.fliplr(target_mask) if bool(rng.integers(2)) else target_mask.copy()
    col_decoy_mask = np.flipud(target_mask) if bool(rng.integers(2)) else target_mask.copy()

    th, tw = target_mask.shape
    row_gap = int(rng.integers(2, 5))
    col_gap = int(rng.integers(2, 5))
    margins = rng.integers(1, 3, size=4)
    top_margin, bottom_margin, left_margin, right_margin = map(int, margins)

    min_h = max(8, top_margin + bottom_margin + 2 * th + col_gap)
    min_w = max(9, left_margin + right_margin + 2 * tw + row_gap)
    main_h = int(rng.integers(min_h, min(min_h + 5, 19) + 1))
    main_w = int(rng.integers(min_w, min(min_w + 6, 20) + 1))

    main = np.zeros((main_h, main_w), dtype=int)
    occupied = np.zeros((main_h, main_w), dtype=bool)

    row_dir = 1 if bool(rng.integers(2)) else -1
    col_dir = 1 if bool(rng.integers(2)) else -1

    if row_dir == 1:
        tc_min = left_margin
        tc_max = main_w - right_margin - (2 * tw + row_gap)
    else:
        tc_min = left_margin + tw + row_gap
        tc_max = main_w - right_margin - tw
    if col_dir == 1:
        tr_min = top_margin
        tr_max = main_h - bottom_margin - (2 * th + col_gap)
    else:
        tr_min = top_margin + th + col_gap
        tr_max = main_h - bottom_margin - th

    target_r = int(rng.integers(tr_min, tr_max + 1))
    target_c = int(rng.integers(tc_min, tc_max + 1))

    if row_dir == 1:
        extra = int(rng.integers(0, main_w - right_margin - (target_c + 2 * tw + row_gap) + 1))
        row_decoy_c = target_c + tw + row_gap + extra
    else:
        extra = int(rng.integers(0, target_c - (left_margin + tw + row_gap) + 1))
        row_decoy_c = target_c - (tw + row_gap + extra)
    row_decoy_r = target_r

    if col_dir == 1:
        extra = int(rng.integers(0, main_h - bottom_margin - (target_r + 2 * th + col_gap) + 1))
        col_decoy_r = target_r + th + col_gap + extra
    else:
        extra = int(rng.integers(0, target_r - (top_margin + th + col_gap) + 1))
        col_decoy_r = target_r - (th + col_gap + extra)
    col_decoy_c = target_c

    placements = [
        (target_mask, target_r, target_c),
        (row_decoy_mask, row_decoy_r, row_decoy_c),
        (col_decoy_mask, col_decoy_r, col_decoy_c),
    ]

    for mask, r0, c0 in placements:
        assert _can_place(occupied, mask, r0, c0, padding=1)
        _paint_mask(main, mask, r0, c0, GREEN)
        _paint_mask(occupied, mask, r0, c0, 1)

    extra_count = int(rng.integers(0, 4))
    for _ in range(extra_count):
        extra_mask = target_mask.copy() if rng.random() < 0.35 else _random_connected_mask(rng)
        placed = False
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, main_h - extra_mask.shape[0]):
            for c0 in range(1, main_w - extra_mask.shape[1]):
                candidates.append((r0, c0))
        for idx in rng.permutation(len(candidates)):
            r0, c0 = candidates[int(idx)]
            if _can_place(occupied, extra_mask, r0, c0, padding=1):
                _paint_mask(main, extra_mask, r0, c0, GREEN)
                _paint_mask(occupied, extra_mask, r0, c0, 1)
                placed = True
                break
        if not placed:
            continue

    return main


def _target_signatures(main: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comps = _components(main == GREEN)
    if len(comps) < 3:
        raise AssertionError("expected target plus single-axis distractors")

    best = None
    best_score = -1
    for coords in comps:
        row_sig, col_sig = _project(coords, main.shape)
        row_matches = 0
        col_matches = 0
        joint_matches = 0
        for other in comps:
            other_row, other_col = _project(other, main.shape)
            row_matches += int(np.array_equal(other_row, row_sig))
            col_matches += int(np.array_equal(other_col, col_sig))
            joint_matches += int(np.array_equal(other_row, row_sig) and np.array_equal(other_col, col_sig))
        score = int(row_matches >= 2) + int(col_matches >= 2) + int(joint_matches == 1)
        if score > best_score:
            best = (row_sig, col_sig, row_matches, col_matches, joint_matches)
            best_score = score
    assert best is not None
    row_sig, col_sig, row_matches, col_matches, joint_matches = best
    assert row_matches >= 2 and col_matches >= 2 and joint_matches == 1
    return row_sig, col_sig


def _build_grid(main: np.ndarray, row_sig: np.ndarray, col_sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    top_band = int(col_sig.max()) + int(rng.integers(0, 3))
    left_band = int(row_sig.max()) + int(rng.integers(0, 3))

    grid = np.zeros((top_band + 1 + main.shape[0], left_band + 1 + main.shape[1]), dtype=int)
    grid[top_band, :] = GREY
    grid[:, left_band] = GREY
    grid[top_band + 1 :, left_band + 1 :] = main

    for c, count in enumerate(col_sig):
        if count:
            grid[top_band - count : top_band, left_band + 1 + c] = YELLOW
    for r, count in enumerate(row_sig):
        if count:
            grid[top_band + 1 + r, left_band - count : left_band] = YELLOW
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(PLACEMENT_RETRIES):
        try:
            main_area = _make_main_area(rng)
            row_sig, col_sig = _target_signatures(main_area)
        except AssertionError:
            continue

        grid = _build_grid(main_area, row_sig, col_sig, rng)
        if max(grid.shape) > 30:
            continue

        sep_r, sep_c = _find_separators(grid)
        parsed_main, parsed_row_sig, parsed_col_sig = _read_signatures(grid, sep_r, sep_c)
        joint_matches, row_matches, col_matches = _match_components(parsed_main, parsed_row_sig, parsed_col_sig)
        if len(joint_matches) != 1 or row_matches < 2 or col_matches < 2:
            continue

        try:
            out = main(grid)
        except Exception:
            continue

        if np.count_nonzero(out) == 0:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-projection puzzle")
