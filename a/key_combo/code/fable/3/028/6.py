from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


COLORS = tuple(range(10))
GEN_RETRIES = 500
MIN_AXIS_LEN = 8


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _line_candidates(grid: np.ndarray, bg: int) -> list[tuple[str, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int, int]] = []

    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        row_lines = [r for r in range(h) if bool(np.all(grid[r, :] == color))]
        if len(row_lines) == 2 and int(np.count_nonzero(grid == color)) == 2 * w:
            candidates.append(("horizontal", int(color), int(row_lines[0]), int(row_lines[1])))

        col_lines = [c for c in range(w) if bool(np.all(grid[:, c] == color))]
        if len(col_lines) == 2 and int(np.count_nonzero(grid == color)) == 2 * h:
            candidates.append(("vertical", int(color), int(col_lines[0]), int(col_lines[1])))

    return candidates


def _find_mirrors(grid: np.ndarray) -> tuple[int, str, int, int, int, int]:
    bg = _bg_color(grid)
    candidates = _line_candidates(grid, bg)
    assert len(candidates) == 1, "expected one pair of same-colored full mirror lines"

    orientation, mirror_color, a, b = candidates[0]
    assert b > a + 1, "mirror lines must enclose a nonempty strip"

    line_mask = np.zeros(grid.shape, dtype=bool)
    if orientation == "horizontal":
        line_mask[[a, b], :] = True
    else:
        line_mask[:, [a, b]] = True

    non_bg_off_lines = (grid != bg) & ~line_mask
    assert bool(non_bg_off_lines.any()), "expected a foreground motif between the mirrors"
    motif_colors = set(map(int, np.unique(grid[non_bg_off_lines])))
    assert len(motif_colors) == 1, "expected one motif color outside the mirror lines"
    motif_color = motif_colors.pop()
    assert motif_color != mirror_color, "motif and mirror colors must differ"

    if orientation == "horizontal":
        outside = non_bg_off_lines.copy()
        outside[a + 1 : b, :] = False
    else:
        outside = non_bg_off_lines.copy()
        outside[:, a + 1 : b] = False
    assert not bool(outside.any()), "input motif pixels must lie only in the mirror corridor"

    return bg, orientation, mirror_color, motif_color, a, b


def _project_offset(value: int, line0: int, line1: int) -> tuple[int, int] | None:
    period = 2 * (line1 - line0)
    rel = int(value) - int(line0)
    mod = rel % period

    if mod == 0 or mod == period // 2:
        return None
    if 0 < mod < period // 2:
        return int(line0 + mod), 0
    return int(line0 + period - mod), 1


def _render_by_mirrors(
    grid: np.ndarray,
    orientation: str,
    mirror_color: int,
    motif_color: int,
    line0: int,
    line1: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape

    if orientation == "horizontal":
        motif = grid[line0 + 1 : line1, :] == int(motif_color)
        for r in range(h):
            mapped = _project_offset(r, line0, line1)
            if mapped is None:
                continue
            base_r, parity = mapped
            for c in range(w):
                if bool(motif[base_r - line0 - 1, c]):
                    out[r, c] = int(mirror_color if parity else motif_color)
    else:
        motif = grid[:, line0 + 1 : line1] == int(motif_color)
        for c in range(w):
            mapped = _project_offset(c, line0, line1)
            if mapped is None:
                continue
            base_c, parity = mapped
            for r in range(h):
                if bool(motif[r, base_c - line0 - 1]):
                    out[r, c] = int(mirror_color if parity else motif_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly reflect the motif between two parallel mirror lines and recolor copies by reflection parity.

    Input structure:
    - A 2D grid contains exactly two full parallel mirror rows or mirror columns of one color M.
    - The only other foreground color F appears as a motif strictly inside the strip between those mirrors.
    - The regions outside the two mirror lines are background before the transformation.

    Rule:
    - Keep the original mirror lines and the original between-mirrors motif unchanged.
    - Treat the enclosed strip as generation 0 and extend it through the two fixed mirror axes in both directions.
    - Every cell projected to a motif pixel by an even number of reflections is colored F, and every odd-generation reflected motif pixel is colored M.
    - Reflected copies are clipped to the input grid; virtual mirror-axis rows or columns remain background unless they are the two original mirrors.

    Key insight:
    - Two parallel mirrors generate alternating flipped and translated images; the visible color of each copy is determined by the parity of the number of reflections, not just by the nearest mirror.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _bg, orientation, mirror_color, motif_color, line0, line1 = _find_mirrors(g)
    return _render_by_mirrors(g, orientation, mirror_color, motif_color, line0, line1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                out.add((nr, nc))
    return list(out)


def _gen_motif(depth: int, length: int) -> np.ndarray:
    min_area = depth + 2
    max_area = max(min_area, min(depth * length // 3, depth + 12, 24))

    for _ in range(200):
        target_area = random.randint(min_area, max_area)
        mask = np.zeros((depth, length), dtype=bool)

        c = random.randint(1, length - 2)
        mask[0, c] = True
        for r in range(1, depth):
            prev_c = c
            c = max(0, min(length - 1, c + random.choice((-1, 0, 1))))
            mask[r, prev_c] = True
            mask[r, c] = True

        while int(mask.sum()) < target_area:
            cells = _frontier(mask)
            if not cells:
                break
            r, c = random.choice(cells)
            mask[r, c] = True

        coords = np.argwhere(mask)
        used_cols = set(map(int, coords[:, 1]))
        row_patterns = {tuple(map(bool, mask[r, :])) for r in range(depth)}
        if int(mask.sum()) != target_area:
            continue
        if not _connected(mask):
            continue
        if not np.all(np.any(mask, axis=1)):
            continue
        if len(used_cols) < 3 or len(row_patterns) < 2:
            continue
        if bool(np.array_equal(mask, mask[::-1, :])):
            continue
        if bool(np.any(np.all(mask, axis=1))):
            continue
        return mask

    raise RuntimeError("failed to generate a nondegenerate motif")


def _choose_dimensions(orientation: str, distance: int, length: int) -> tuple[int, int, int, int]:
    min_margin = distance + 1
    central_span = distance + 1
    min_perp = 2 * min_margin + central_span
    perp = random.randint(min_perp, 30)
    free = perp - min_perp
    near_margin = min_margin + random.randint(0, free)
    far_margin = perp - central_span - near_margin
    if orientation == "horizontal":
        return perp, length, near_margin, far_margin
    return length, perp, near_margin, far_margin


def _assemble_grid(
    orientation: str,
    bg: int,
    mirror_color: int,
    motif_color: int,
    distance: int,
    motif: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    length = int(motif.shape[1])
    h, w, before, _after = _choose_dimensions(orientation, distance, length)
    line0 = before
    line1 = before + distance
    grid = np.full((h, w), int(bg), dtype=int)

    if orientation == "horizontal":
        assert motif.shape == (distance - 1, w)
        grid[[line0, line1], :] = int(mirror_color)
        grid[line0 + 1 : line1, :][motif] = int(motif_color)
    else:
        assert motif.shape == (distance - 1, h)
        grid[:, [line0, line1]] = int(mirror_color)
        for d in range(distance - 1):
            grid[motif[d, :], line0 + 1 + d] = int(motif_color)

    return grid, line0, line1


def _copy_stats(grid: np.ndarray, out: np.ndarray, orientation: str, line0: int, line1: int) -> dict[str, int]:
    stats = {"odd_before": 0, "odd_after": 0, "even_before": 0, "even_after": 0}

    if orientation == "horizontal":
        for r, c in np.argwhere((grid == _bg_color(grid)) & (out != _bg_color(grid))):
            mapped = _project_offset(int(r), line0, line1)
            if mapped is None:
                continue
            _base, parity = mapped
            side = "before" if int(r) < line0 else "after"
            stats[("odd" if parity else "even") + "_" + side] += 1
    else:
        for r, c in np.argwhere((grid == _bg_color(grid)) & (out != _bg_color(grid))):
            mapped = _project_offset(int(c), line0, line1)
            if mapped is None:
                continue
            _base, parity = mapped
            side = "before" if int(c) < line0 else "after"
            stats[("odd" if parity else "even") + "_" + side] += 1

    return stats


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
    distance = random.randint(4, 9)
    depth = distance - 1

    bg, mirror_color, motif_color = random.sample(COLORS, 3)
    length = random.randint(MIN_AXIS_LEN, 22)
    motif = _gen_motif(depth, length)

    try:
        grid, line0, line1 = _assemble_grid(
            orientation,
            bg,
            mirror_color,
            motif_color,
            distance,
            motif,
        )
    except AssertionError:
        return None

    try:
        detected_bg, detected_orientation, detected_mirror, detected_motif, a, b = _find_mirrors(grid)
    except AssertionError:
        return None
    if (detected_bg, detected_orientation, detected_mirror, detected_motif, a, b) != (
        bg,
        orientation,
        mirror_color,
        motif_color,
        line0,
        line1,
    ):
        return None

    expected = _render_by_mirrors(grid, orientation, mirror_color, motif_color, line0, line1)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    stats = _copy_stats(grid, expected, orientation, line0, line1)
    if not all(value > 0 for value in stats.values()):
        return None
    if int(np.count_nonzero(expected != grid)) < 3 * int(motif.sum()):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-mirror corridor input")
