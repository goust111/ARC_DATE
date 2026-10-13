from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ANCHOR_COLOR = 4  # ARC yellow
PALETTE = tuple(range(10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _wave_shift(distance: int, anchor_on_left: bool) -> int:
    if distance % 2 == 0:
        return 0
    toward_anchor = -1 if anchor_on_left else 1
    if ((distance - 1) // 2) % 2 == 0:
        return toward_anchor
    return -toward_anchor


def _shift_mask(mask: np.ndarray, anchor_row: int, anchor_on_left: bool) -> np.ndarray | None:
    out = np.zeros_like(mask, dtype=bool)
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            continue
        shift = _wave_shift(abs(r - anchor_row), anchor_on_left)
        new_cols = cols + shift
        if int(new_cols.min()) < 0 or int(new_cols.max()) >= mask.shape[1]:
            return None
        out[r, new_cols] = True
    return out


def _render_output(
    grid: np.ndarray,
    shifted_mask: np.ndarray,
    *,
    bg: int,
    fg: int,
    anchor_rc: tuple[int, int],
) -> np.ndarray:
    out = np.full_like(grid, bg)
    out[shifted_mask] = fg
    out[anchor_rc] = ANCHOR_COLOR
    return out


def _detect_scene(grid: np.ndarray) -> tuple[int, int, np.ndarray, tuple[int, int, int, int], tuple[int, int], bool]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    anchor_coords = np.argwhere(g == ANCHOR_COLOR)
    assert len(anchor_coords) == 1
    ar, ac = map(int, anchor_coords[0])

    fg_colors = sorted(set(map(int, np.unique(g))) - {bg, ANCHOR_COLOR})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    fg_mask = g == fg
    comps = _components_mask(fg_mask)
    assert len(comps) == 1
    box = _bbox(fg_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    assert r0 <= ar <= r1
    assert ac < c0 or ac > c1
    anchor_on_left = ac < c0

    for r in range(r0, r1 + 1):
        assert bool(fg_mask[r].any())

    return bg, fg, fg_mask, box, (ar, ac), anchor_on_left


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    bg, fg, fg_mask, _box, anchor_rc, anchor_on_left = _detect_scene(grid)
    shifted = _shift_mask(fg_mask, anchor_rc[0], anchor_on_left)
    assert shifted is not None
    return _render_output(grid, shifted, bg=bg, fg=fg, anchor_rc=anchor_rc)


def main(grid: np.ndarray) -> np.ndarray:
    """Bend the single foreground lattice into a horizontal wave phased by an external yellow anchor.

    Input structure
    - The background is dominant, with exactly one non-yellow foreground color forming a single 4-connected lattice-like object.
    - Every row inside that object's bounding box contains at least one foreground pixel.
    - A unique yellow pixel lies outside the object, on the same row as one bbox row, strictly to the object's left or right.

    Rule
    - Keep the row aligned with the yellow anchor fixed.
    - Propagate upward and downward from that row using the same distance rule: even-distance rows stay put.
    - Odd-distance rows shift horizontally by one cell; distance 1 moves toward the anchor side, distance 3 moves away, distance 5 moves toward again, and so on.
    - Preserve the yellow anchor pixel in place.

    Key insight
    - The wave phase is anchored by the external yellow pixel's row and side, not by the top or bottom of the lattice itself.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _sample_bar_indices(length: int, count: int) -> list[int] | None:
    assert count >= 2
    required = [0, length - 1]
    need = count - 2
    if need == 0:
        return required
    interior = list(range(1, length - 1))
    for _ in range(60):
        picks = sorted(random.sample(interior, need))
        bars = sorted(required + picks)
        if min(bars[i + 1] - bars[i] for i in range(len(bars) - 1)) >= 2:
            return bars
    return None


def _build_lattice_mask(height: int, width: int) -> np.ndarray | None:
    max_h = min(4, (height + 1) // 2)
    max_w = min(4, (width + 1) // 2)
    n_hbars = random.randint(2, max_h)
    n_vbars = random.randint(2, max_w)
    h_rows = _sample_bar_indices(height, n_hbars)
    v_cols = _sample_bar_indices(width, n_vbars)
    if h_rows is None or v_cols is None:
        return None

    mask = np.zeros((height, width), dtype=bool)
    mask[h_rows, :] = True
    mask[:, v_cols] = True

    if len(_components_mask(mask)) != 1:
        return None
    if not np.all(mask.any(axis=1)):
        return None
    if np.all(mask):
        return None
    return mask


def _false_output(
    grid: np.ndarray,
    fg_mask: np.ndarray,
    *,
    bg: int,
    fg: int,
    anchor_rc: tuple[int, int],
    anchor_row: int,
    anchor_on_left: bool,
) -> np.ndarray | None:
    shifted = _shift_mask(fg_mask, anchor_row, anchor_on_left)
    if shifted is None:
        return None
    return _render_output(grid, shifted, bg=bg, fg=fg, anchor_rc=anchor_rc)


def _semantic_checks(
    grid: np.ndarray,
    *,
    expected: np.ndarray,
    bg: int,
    fg: int,
    fg_mask: np.ndarray,
    box: tuple[int, int, int, int],
    anchor_rc: tuple[int, int],
    anchor_on_left: bool,
) -> bool:
    r0, _c0, r1, _c1 = box
    ar, _ac = anchor_rc
    local_anchor = ar - r0
    height = r1 - r0 + 1

    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, grid):
        return False
    if expected[anchor_rc] != ANCHOR_COLOR:
        return False
    if int(np.count_nonzero(expected == fg)) != int(np.count_nonzero(fg_mask)):
        return False
    if local_anchor <= 0 or local_anchor >= height - 1:
        return False
    if max(local_anchor, height - 1 - local_anchor) < 3:
        return False

    moved_above = [r for r in range(r0, ar) if _wave_shift(ar - r, anchor_on_left) != 0]
    moved_below = [r for r in range(ar + 1, r1 + 1) if _wave_shift(r - ar, anchor_on_left) != 0]
    if not moved_above or not moved_below:
        return False

    visible_shifts = {
        _wave_shift(d, anchor_on_left)
        for d in range(1, max(local_anchor, height - 1 - local_anchor) + 1, 2)
    }
    if visible_shifts != {-1, 1}:
        return False

    wrong_variants = []
    for wrong_row in (r0, r1):
        wrong = _false_output(
            grid,
            fg_mask,
            bg=bg,
            fg=fg,
            anchor_rc=anchor_rc,
            anchor_row=wrong_row,
            anchor_on_left=anchor_on_left,
        )
        if wrong is not None:
            wrong_variants.append(wrong)
    wrong_variants.append(
        _false_output(
            grid,
            fg_mask,
            bg=bg,
            fg=fg,
            anchor_rc=anchor_rc,
            anchor_row=ar,
            anchor_on_left=not anchor_on_left,
        )
    )
    wrong_variants.append(
        _false_output(
            grid,
            fg_mask,
            bg=bg,
            fg=fg,
            anchor_rc=anchor_rc,
            anchor_row=r1,
            anchor_on_left=not anchor_on_left,
        )
    )
    for wrong in wrong_variants:
        if wrong is not None and np.array_equal(wrong, expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in PALETTE if c != ANCHOR_COLOR])
    fg = random.choice([c for c in PALETTE if c not in (bg, ANCHOR_COLOR)])

    height = random.randint(7, 15)
    width = random.randint(5, 12)
    lattice_mask = _build_lattice_mask(height, width)
    if lattice_mask is None:
        return None

    valid_anchor_rows = [
        r for r in range(1, height - 1) if max(r, height - 1 - r) >= 3
    ]
    if not valid_anchor_rows:
        return None
    anchor_local_r = random.choice(valid_anchor_rows)
    anchor_on_left = random.random() < 0.5
    anchor_gap = random.randint(1, 3)

    top_margin = random.randint(1, 5)
    bottom_margin = random.randint(1, 5)
    if anchor_on_left:
        left_margin = anchor_gap + random.randint(1, 4)
        right_margin = random.randint(1, 5)
    else:
        left_margin = random.randint(1, 5)
        right_margin = anchor_gap + random.randint(1, 4)

    h = top_margin + height + bottom_margin
    w = left_margin + width + right_margin
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), bg, dtype=int)
    r0 = top_margin
    c0 = left_margin
    grid[r0 : r0 + height, c0 : c0 + width][lattice_mask] = fg

    anchor_r = r0 + anchor_local_r
    anchor_c = c0 - anchor_gap if anchor_on_left else c0 + width - 1 + anchor_gap
    if not (0 <= anchor_c < w):
        return None
    grid[anchor_r, anchor_c] = ANCHOR_COLOR

    scene = _detect_scene(grid)
    bg2, fg2, fg_mask, box, anchor_rc, anchor_side = scene
    shifted = _shift_mask(fg_mask, anchor_rc[0], anchor_side)
    if shifted is None:
        return None
    expected = _render_output(grid, shifted, bg=bg2, fg=fg2, anchor_rc=anchor_rc)

    if not _semantic_checks(
        grid,
        expected=expected,
        bg=bg2,
        fg=fg2,
        fg_mask=fg_mask,
        box=box,
        anchor_rc=anchor_rc,
        anchor_on_left=anchor_side,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-wave input")
