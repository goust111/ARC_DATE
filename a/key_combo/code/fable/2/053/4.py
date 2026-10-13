from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
INPUT_COLORS = [3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 600


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    counts = Counter(map(int, flat))
    return int(counts.most_common(1)[0][0])


def _apply_d4(grid: np.ndarray, transform_id: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    t = int(transform_id)
    if t < 4:
        return np.rot90(g, t).copy()
    return np.fliplr(np.rot90(g, t - 4)).copy()


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert w == 2 * h + 1, "expected two square panels split by one separator column"
    sep_col = int(h)
    sep_color = int(g[0, sep_col])
    assert np.all(g[:, sep_col] == sep_color), "separator column must be solid"
    return sep_col, sep_color


def _panel_background(left: np.ndarray, right: np.ndarray, sep_color: int) -> int:
    visible = np.concatenate(
        [
            np.asarray(left[left != int(sep_color)], dtype=int),
            np.asarray(right[right != int(sep_color)], dtype=int),
        ]
    )
    assert visible.size > 0, "panels must contain some non-separator pixels"
    return _mode_color(visible)


def _clean_panel(panel: np.ndarray, sep_color: int, bg: int) -> np.ndarray:
    out = np.asarray(panel, dtype=int).copy()
    out[out == int(sep_color)] = int(bg)
    return out


def _best_alignment(left: np.ndarray, right: np.ndarray) -> tuple[int, np.ndarray, list[int]]:
    candidates: list[np.ndarray] = []
    scores: list[int] = []
    for transform_id in range(8):
        aligned = _apply_d4(right, transform_id)
        candidates.append(aligned)
        scores.append(int(np.count_nonzero(left != aligned)))
    best_score = min(scores)
    assert scores.count(best_score) == 1, "alignment transform must be unique"
    best_id = int(scores.index(best_score))
    return best_id, candidates[best_id], scores


def _compare_panels(left: np.ndarray, aligned_right: np.ndarray, bg: int) -> np.ndarray:
    left = np.asarray(left, dtype=int)
    right = np.asarray(aligned_right, dtype=int)
    out = np.full(left.shape, int(bg), dtype=int)

    same = left == right
    out[same] = left[same]

    left_only = (left != int(bg)) & (right == int(bg))
    right_only = (left == int(bg)) & (right != int(bg))
    out[left_only] = RED
    out[right_only] = BLUE

    recolor = (~same) & (left != int(bg)) & (right != int(bg))
    assert not bool(recolor.any()), "valid instances use add/remove edits, not foreground recolors"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove separator-colored stains, recover the hidden dihedral panel alignment, and output a cellwise diff map.

    Input structure:
    - Two equal square panels sit side by side with a single solid separator column between them.
    - Inside the panels, the separator color also appears as scattered stain pixels on top of the panel background.
    - Apart from those stains, the right panel is a rotated or mirrored copy of the left panel with a few true add/remove edits.

    Rule:
    - Treat every separator-colored pixel inside the panels as ignorable background noise.
    - Find the unique rotation/reflection that aligns the right panel back to the left panel's orientation.
    - Compare the aligned panels cell by cell: equal cells keep their color, left-only foreground becomes red, and right-only foreground becomes blue.
    - Return the comparison image at the single-panel size.

    Key insight:
    - The divider color is not meaningful content inside either panel; only after erasing those stain pixels does one dihedral transform become uniquely correct.
    """
    g = np.asarray(grid, dtype=int)
    sep_col, sep_color = _find_separator(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    bg = _panel_background(left, right, sep_color)
    clean_left = _clean_panel(left, sep_color, bg)
    clean_right = _clean_panel(right, sep_color, bg)

    _best_id, aligned_right, _scores = _best_alignment(clean_left, clean_right)
    return _compare_panels(clean_left, aligned_right, bg)


def _grow_shape(h: int, w: int, area: int) -> list[tuple[int, int]]:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return []
        cells.add(random.choice(frontier))
    coords = sorted(cells)
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if area >= 4 and (len(rows) == 1 or len(cols) == 1):
        return []
    return coords


def _can_place(mask: np.ndarray, coords: list[tuple[int, int]], r0: int, c0: int) -> bool:
    n = int(mask.shape[0])
    for dr, dc in coords:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if not (0 <= rr < n and 0 <= cc < n):
            return False
        r1 = max(0, rr - 1)
        r2 = min(n, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(n, cc + 2)
        if bool(mask[r1:r2, c1:c2].any()):
            return False
    return True


def _paint_shape(panel: np.ndarray, mask: np.ndarray, coords: list[tuple[int, int]], r0: int, c0: int, color: int) -> None:
    for dr, dc in coords:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        panel[rr, cc] = int(color)
        mask[rr, cc] = True


def _object_count_for_size(size: int) -> int:
    if size <= 8:
        return random.randint(3, 4)
    if size <= 11:
        return random.randint(4, 5)
    return random.randint(5, 6)


def _build_left_panel(size: int) -> np.ndarray | None:
    panel = np.full((size, size), BLACK, dtype=int)
    occ = np.zeros((size, size), dtype=bool)
    n_objects = _object_count_for_size(size)
    colors = random.sample(INPUT_COLORS, k=n_objects)

    for color in colors:
        placed = False
        for _ in range(120):
            h = random.randint(2, min(5, size - 1))
            w = random.randint(2, min(5, size - 1))
            area = random.randint(3, min(h * w, 7))
            coords = _grow_shape(h, w, area)
            if not coords:
                continue
            if size - h - 1 < 1 or size - w - 1 < 1:
                continue
            r0 = random.randint(1, size - h - 1)
            c0 = random.randint(1, size - w - 1)
            if not _can_place(occ, coords, r0, c0):
                continue
            _paint_shape(panel, occ, coords, r0, c0, color)
            placed = True
            break
        if not placed:
            return None

    area = int(np.count_nonzero(panel != BLACK))
    if not (size * size // 7 <= area <= size * size // 2):
        return None

    distances = [
        int(np.count_nonzero(panel != _apply_d4(panel, transform_id)))
        for transform_id in range(1, 8)
    ]
    if min(distances) < max(6, area // 4):
        return None
    return panel


def _pick_inverse_alignment(left: np.ndarray, right: np.ndarray) -> tuple[int, list[int]] | None:
    scores = [int(np.count_nonzero(left != _apply_d4(right, tid))) for tid in range(8)]
    if min(scores) != 0 or scores.count(0) != 1:
        return None
    return int(scores.index(0)), scores


def _choose_add_positions(panel: np.ndarray, count: int) -> list[tuple[int, int]]:
    bg_cells = [tuple(map(int, rc)) for rc in np.argwhere(panel == BLACK)]
    random.shuffle(bg_cells)
    chosen: list[tuple[int, int]] = []
    for r, c in bg_cells:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) == int(count):
            break
    return chosen


def _neighbor_colors(panel: np.ndarray, r: int, c: int) -> list[int]:
    colors: list[int] = []
    h, w = map(int, panel.shape)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(panel[rr, cc]) != BLACK:
                colors.append(int(panel[rr, cc]))
    return colors


def _edit_right_panel(right: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    out = np.asarray(right, dtype=int).copy()
    fg_cells = [tuple(map(int, rc)) for rc in np.argwhere(out != BLACK)]
    if len(fg_cells) < 8:
        return None

    remove_count = random.randint(1, min(3, max(1, len(fg_cells) // 6)))
    add_count = random.randint(1, 3)

    removed = random.sample(fg_cells, k=remove_count)
    for r, c in removed:
        out[r, c] = BLACK

    add_positions = _choose_add_positions(out, add_count)
    if len(add_positions) != add_count:
        return None

    used_colors = sorted({int(x) for x in np.unique(right) if int(x) != BLACK})
    for r, c in add_positions:
        near = _neighbor_colors(out, r, c)
        palette = near if near and random.random() < 0.75 else used_colors
        out[r, c] = int(random.choice(palette))

    return out, remove_count, add_count


def _add_stains(panel: np.ndarray, sep_color: int, count: int) -> np.ndarray:
    out = np.asarray(panel, dtype=int).copy()
    bg_cells = [tuple(map(int, rc)) for rc in np.argwhere(out == BLACK)]
    if len(bg_cells) < count:
        raise ValueError("not enough background for stains")
    for r, c in random.sample(bg_cells, k=count):
        out[r, c] = int(sep_color)
    return out


def _build_grid(left: np.ndarray, right: np.ndarray, sep_color: int) -> np.ndarray:
    n = int(left.shape[0])
    grid = np.full((n, 2 * n + 1), BLACK, dtype=int)
    grid[:, :n] = left
    grid[:, n] = int(sep_color)
    grid[:, n + 1 :] = right
    return grid


def _semantic_checks(
    left_clean: np.ndarray,
    right_clean: np.ndarray,
    left_noisy: np.ndarray,
    right_noisy: np.ndarray,
    sep_color: int,
    align_id: int,
    expected: np.ndarray,
) -> bool:
    clean_best = [int(np.count_nonzero(left_clean != _apply_d4(right_clean, tid))) for tid in range(8)]
    if clean_best.count(min(clean_best)) != 1:
        return False
    if int(clean_best[align_id]) != min(clean_best):
        return False

    raw_best = [int(np.count_nonzero(left_noisy != _apply_d4(right_noisy, tid))) for tid in range(8)]
    if min(raw_best) <= min(clean_best):
        return False
    if min(raw_best) == int(clean_best[align_id]):
        return False

    if not np.any(left_noisy == int(sep_color)):
        return False
    if not np.any(right_noisy == int(sep_color)):
        return False

    if not np.any(expected == RED):
        return False
    if not np.any(expected == BLUE):
        return False
    preserved = (expected != BLACK) & (expected != RED) & (expected != BLUE)
    if int(np.count_nonzero(preserved)) < 6:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(7, 14)
    left_clean = _build_left_panel(size)
    if left_clean is None:
        return None

    sep_color = int(random.choice(INPUT_COLORS))
    if sep_color in set(map(int, np.unique(left_clean))):
        remaining = [c for c in INPUT_COLORS if c != sep_color and c not in {RED, BLUE}]
        repl = int(random.choice(remaining))
        left_clean = left_clean.copy()
        left_clean[left_clean == int(sep_color)] = repl
        sep_color = int(random.choice([c for c in INPUT_COLORS if c not in set(map(int, np.unique(left_clean)))]))

    build_id = random.randrange(8)
    right_base = _apply_d4(left_clean, build_id)
    inverse_info = _pick_inverse_alignment(left_clean, right_base)
    if inverse_info is None:
        return None
    align_id, _zero_scores = inverse_info

    edited = _edit_right_panel(right_base)
    if edited is None:
        return None
    right_clean, _remove_count, _add_count = edited

    aligned_right = _apply_d4(right_clean, align_id)
    try:
        expected = _compare_panels(left_clean, aligned_right, BLACK)
    except AssertionError:
        return None

    left_noise_count = random.randint(2, min(7, size))
    right_noise_count = random.randint(2, min(7, size))
    try:
        left_noisy = _add_stains(left_clean, sep_color, left_noise_count)
        right_noisy = _add_stains(right_clean, sep_color, right_noise_count)
    except ValueError:
        return None

    if not _semantic_checks(
        left_clean,
        right_clean,
        left_noisy,
        right_noisy,
        sep_color,
        align_id,
        expected,
    ):
        return None

    grid = _build_grid(left_noisy, right_noisy, sep_color)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stained-panel comparison instance")
