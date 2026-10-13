from __future__ import annotations

import numpy as np


BG = 0
DIVIDER = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
MAX_RETRIES = 400

_RNG = np.random.default_rng()
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _randint(lo: int, hi: int) -> int:
    """Inclusive integer draw."""
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(items):
    """Choose one item from a non-empty sequence."""
    return items[int(_RNG.integers(0, len(items)))]


def _pick_color(exclude: set[int] | None = None) -> int:
    """Pick one panel color that is not excluded."""
    banned = exclude or set()
    choices = [color for color in PALETTE if int(color) not in banned]
    assert choices, "expected at least one available color"
    return int(_choice(choices))


def _sample_colors(count: int, exclude: set[int] | None = None) -> list[int]:
    """Sample distinct panel colors."""
    banned = exclude or set()
    choices = [color for color in PALETTE if int(color) not in banned]
    assert len(choices) >= count, "not enough colors to sample"
    picked = _RNG.choice(np.array(choices, dtype=int), size=count, replace=False)
    return [int(v) for v in picked.tolist()]


def _divider_col(grid: np.ndarray) -> int:
    """The puzzle uses one central vertical divider and equal-width side panels."""
    h, w = grid.shape
    assert h > 0 and w % 2 == 1, "expected an odd-width grid"
    mid = w // 2
    divider = grid[:, mid]
    assert np.all(divider == divider[0]), "divider column must be uniform"
    return int(mid)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the mismatching mirror pixels between the left and right panels.

    Input structure:
    - A single vertical maroon divider column splits the grid into two equal-width panels.
    - The background is black.
    - The right panel is intended to be the horizontal mirror of the left panel, except
      for a set of mismatching pixel pairs across the mirror line.

    Rule:
    - Compare each left-panel pixel with its horizontally mirrored partner in the right panel.
    - Whenever the two colors differ, exchange those two colors across the pair.
    - Leave the divider and every already-matching mirror pair unchanged.

    Key insight:
    - The anomalies are paired by mirror symmetry, and the fix is a swap across each
      mirrored pair rather than deleting or copying only one side.
    """
    g = np.asarray(grid, dtype=int)
    mid = _divider_col(g)
    out = g.copy()
    width = g.shape[1]

    for c in range(mid):
        mc = width - 1 - c
        left_col = g[:, c]
        right_col = g[:, mc]
        mismatched = left_col != right_col
        out[mismatched, c] = right_col[mismatched]
        out[mismatched, mc] = left_col[mismatched]

    return out


def _grow_blob(h: int, w: int, area: int) -> np.ndarray:
    """Grow a 4-connected blob inside an h x w box."""
    mask = np.zeros((h, w), dtype=bool)
    seed = (_randint(0, h - 1), _randint(0, w - 1))
    mask[seed] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in _DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier, "blob growth frontier should not be empty"
        nr, nc = _choice(list(frontier))
        mask[nr, nc] = True

    return mask


def _make_shape(max_h: int, max_w: int) -> np.ndarray:
    """Build a small readable monochrome patch."""
    for _ in range(120):
        kinds = ["rect", "line", "blob"]
        if max_h >= 3 and max_w >= 3:
            kinds.append("frame")
            kinds.append("plus")
        kind = str(_choice(kinds))

        if kind == "rect":
            h = _randint(1, min(4, max_h))
            w = _randint(1, min(4, max_w))
            if h * w < 2:
                continue
            return np.ones((h, w), dtype=bool)

        if kind == "line":
            if _RNG.random() < 0.5:
                return np.ones((1, _randint(2, min(5, max_w))), dtype=bool)
            return np.ones((_randint(2, min(5, max_h)), 1), dtype=bool)

        if kind == "frame":
            h = _randint(3, min(5, max_h))
            w = _randint(3, min(5, max_w))
            patch = np.ones((h, w), dtype=bool)
            patch[1:-1, 1:-1] = False
            return patch

        if kind == "plus":
            h_choices = [side for side in (3, 5) if side <= max_h]
            w_choices = [side for side in (3, 5) if side <= max_w]
            if not h_choices or not w_choices:
                continue
            h = int(_choice(h_choices))
            w = int(_choice(w_choices))
            patch = np.zeros((h, w), dtype=bool)
            patch[h // 2, :] = True
            patch[:, w // 2] = True
            return patch

        h = _randint(2, min(5, max_h))
        w = _randint(2, min(5, max_w))
        capacity = h * w
        if capacity <= 3:
            return np.ones((h, w), dtype=bool)
        low = max(3, (capacity + 2) // 3)
        high = capacity - 1
        if low > high:
            low = high = capacity
        area = _randint(low, high)
        return _grow_blob(h, w, area)

    raise RuntimeError("failed to construct a shape")


def _stamp_blocked(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Reserve one-cell padding around a placed patch."""
    h, w = blocked.shape
    for r, c in np.argwhere(patch):
        rr = int(r0 + int(r))
        cc = int(c0 + int(c))
        blocked[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = True


def _place_patch(panel: np.ndarray, blocked: np.ndarray, patch: np.ndarray, color: int) -> bool:
    """Place a patch without touching existing objects."""
    h, w = panel.shape
    ph, pw = patch.shape
    choices: list[tuple[int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & patch):
                continue
            choices.append((int(r0), int(c0)))

    if not choices:
        return False

    r0, c0 = _choice(choices)
    window = panel[r0 : r0 + ph, c0 : c0 + pw]
    window[patch] = int(color)
    _stamp_blocked(blocked, patch, int(r0), int(c0))
    return True


def _make_base_panel(height: int, width: int) -> np.ndarray:
    """Generate a clean left panel before injecting mirror mismatches."""
    area = int(height * width)
    max_objects = min(6, max(3, area // 18))

    for _ in range(200):
        panel = np.full((height, width), BG, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)
        object_count = _randint(2, max_objects)
        distinct_color_count = _randint(2, min(5, object_count))
        color_pool = _sample_colors(distinct_color_count)
        object_colors = [color_pool[i % distinct_color_count] for i in range(object_count)]
        _RNG.shuffle(object_colors)

        ok = True
        for color in object_colors:
            placed = False
            for _shape_try in range(80):
                patch = _make_shape(height, width)
                if _place_patch(panel, blocked, patch, int(color)):
                    placed = True
                    break
            if not placed:
                ok = False
                break

        non_bg = int(np.count_nonzero(panel != BG))
        used_colors = {int(v) for v in np.unique(panel) if int(v) != BG}
        if ok and max(6, area // 10) <= non_bg <= area // 2 and len(used_colors) >= 2:
            return panel

    raise RuntimeError("failed to generate a base panel")


def _difference_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    """Choose well-spaced left-panel positions for mismatches."""
    cells = [(r, c) for r in range(height) for c in range(width)]
    order = _RNG.permutation(len(cells))
    chosen: list[tuple[int, int]] = []

    for idx in order.tolist():
        r, c = cells[int(idx)]
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == count:
                return chosen

    return None


def _inject_differences(left: np.ndarray, right: np.ndarray) -> None:
    """Modify mirrored pairs so the panels disagree at a small set of positions."""
    h, w = left.shape
    count = _randint(3, min(6, max(3, (h * w) // 20)))
    positions = _difference_positions(h, w, count)
    assert positions is not None, "failed to choose mismatch positions"

    cycle = ["two_colors", "left_only", "right_only"]
    diff_types: list[str] = []
    while len(diff_types) < count:
        perm = cycle.copy()
        _RNG.shuffle(perm)
        diff_types.extend(perm)

    for (r, c), diff_type in zip(positions, diff_types[:count], strict=False):
        mc = w - 1 - c
        base = int(left[r, c])

        if diff_type == "two_colors":
            left_color, right_color = _sample_colors(2, exclude={base} if base != BG else None)
            left[r, c] = int(left_color)
            right[r, mc] = int(right_color)
            continue

        if diff_type == "left_only":
            if base != BG and _RNG.random() < 0.5:
                left_color = base
            else:
                left_color = _pick_color(exclude={base} if base != BG else None)
            left[r, c] = int(left_color)
            right[r, mc] = BG
            continue

        if base != BG and _RNG.random() < 0.5:
            right_color = base
        else:
            right_color = _pick_color(exclude={base} if base != BG else None)
        left[r, c] = BG
        right[r, mc] = int(right_color)


def _compose_grid(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Join both panels with the divider column."""
    h, w = left.shape
    grid = np.full((h, 2 * w + 1), BG, dtype=int)
    grid[:, :w] = left
    grid[:, w] = DIVIDER
    grid[:, w + 1 :] = right
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    """Reference implementation used by the generator's internal validation."""
    g = np.asarray(grid, dtype=int)
    mid = _divider_col(g)
    out = g.copy()
    width = g.shape[1]
    for c in range(mid):
        mc = width - 1 - c
        left_col = g[:, c].copy()
        right_col = g[:, mc].copy()
        mismatched = left_col != right_col
        out[mismatched, c] = right_col[mismatched]
        out[mismatched, mc] = left_col[mismatched]
    return out


def _valid_instance(grid: np.ndarray) -> bool:
    """Check that the generated task matches the intended mechanism."""
    mid = _divider_col(grid)
    left = grid[:, :mid]
    right = grid[:, mid + 1 :]
    mirrored_right = np.fliplr(right)
    mismatch_count = int(np.count_nonzero(left != mirrored_right))
    values, counts = np.unique(grid, return_counts=True)
    dominant = int(values[int(np.argmax(counts))])

    if mismatch_count < 3:
        return False
    if dominant != BG:
        return False

    panel_colors = {int(v) for v in np.unique(np.concatenate((left.ravel(), right.ravel())))}
    if len(panel_colors - {BG}) < 2:
        return False

    out = main(grid)
    if not np.array_equal(out, _expected_output(grid)):
        return False

    return True


def generate_input() -> np.ndarray:
    """Generate mirrored double-panel scenes with paired swap anomalies."""
    for _ in range(MAX_RETRIES):
        height = _randint(8, 18)
        panel_width = _randint(6, 14)
        if 2 * panel_width + 1 > 30:
            continue

        left = _make_base_panel(height, panel_width)
        right = np.fliplr(left).copy()
        _inject_differences(left, right)
        grid = _compose_grid(left, right)

        if _valid_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
