from __future__ import annotations

from collections import deque
import random

import numpy as np


GEN_ATTEMPTS = 400
PALETTE = list(range(10))
MIN_MARKERS = 4
MAX_MARKERS = 7


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != int(bg)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _marker_candidate(grid: np.ndarray, bg: int, color: int) -> tuple[int, tuple[int, int]] | None:
    coords = np.argwhere(grid == color)
    if coords.shape[0] < 2 * MIN_MARKERS:
        return None

    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    row_counts = [int(np.count_nonzero(coords[:, 0] == row)) for row in rows]
    col_counts = [int(np.count_nonzero(coords[:, 1] == col)) for col in cols]

    options: list[tuple[int, tuple[int, int]]] = []
    if len(rows) == 2 and len(cols) >= MIN_MARKERS and min(row_counts) >= MIN_MARKERS:
        options.append((0, (rows[0], rows[1])))
    if len(cols) == 2 and len(rows) >= MIN_MARKERS and min(col_counts) >= MIN_MARKERS:
        options.append((1, (cols[0], cols[1])))
    if len(options) != 1:
        return None

    axis, lines = options[0]
    other = np.argwhere((grid != bg) & (grid != color))
    if other.size == 0:
        return None
    lo, hi = sorted(lines)
    if not np.all((other[:, axis] > lo) & (other[:, axis] < hi)):
        return None
    return axis, (lo, hi)


def _find_mirror_data(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates: list[tuple[int, int, int, int]] = []
    for color in _colors_present(g, bg):
        marker = _marker_candidate(g, bg, color)
        if marker is None:
            continue
        axis, (lo, hi) = marker
        candidates.append((color, axis, lo, hi))
    assert len(candidates) == 1, "expected exactly one sparse marker-line color"
    return candidates[0]


def _paint_family(
    out: np.ndarray, color: int, axis: int, fixed: int, base: int, step: int
) -> None:
    limit = out.shape[axis]

    pos = int(base)
    while pos >= 0:
        if axis == 0:
            cell = (pos, fixed)
        else:
            cell = (fixed, pos)
        if out[cell] == _get_bg(out):
            out[cell] = int(color)
        else:
            assert int(out[cell]) == int(color)
        pos -= 2 * step

    pos = int(base) + 2 * step
    while pos < limit:
        if axis == 0:
            cell = (pos, fixed)
        else:
            cell = (fixed, pos)
        if out[cell] == _get_bg(out):
            out[cell] = int(color)
        else:
            assert int(out[cell]) == int(color)
        pos += 2 * step


def _render_gallery(grid: np.ndarray, marker_color: int, axis: int, lo: int, hi: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    step = hi - lo
    assert step >= 2

    motif_coords = np.argwhere((g != bg) & (g != marker_color))
    assert motif_coords.size > 0
    for rr, cc in motif_coords:
        r = int(rr)
        c = int(cc)
        normal = r if axis == 0 else c
        fixed = c if axis == 0 else r
        assert lo < normal < hi
        color = int(g[r, c])
        _paint_family(out, color, axis, fixed, normal, step)
        _paint_family(out, color, axis, fixed, 2 * lo - normal, step)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect the corridor motif back and forth between two sparse mirror lines until every in-bounds image is drawn.

    Input structure:
    - One color appears only as sparse pixels on exactly two parallel horizontal or vertical marker lines.
    - All other foreground pixels form a multicolor motif strictly between those two lines; outside the lines is background.
    - The sparse marker pixels indicate the full mirror lines even though most cells on those lines are blank.

    Rule:
    - Keep the sparse marker pixels unchanged.
    - Treat the motif as the source image inside the marked corridor.
    - Draw every image produced by alternating reflections across the two mirror lines, so copies repeat perpendicular to the lines with period equal to the corridor width and odd-numbered corridors are flipped.

    Key insight:
    - Two parallel mirrors do not stop after one reflection: they generate an infinite alternating image gallery, and the true axes must first be inferred from the sparse marker points.
    """
    g = np.asarray(grid, dtype=int)
    marker_color, axis, lo, hi = _find_mirror_data(g)
    return _render_gallery(g, marker_color, axis, lo, hi)


def _sample_sparse_positions(length: int) -> list[int] | None:
    max_count = min(MAX_MARKERS, (length + 1) // 2)
    if max_count < MIN_MARKERS:
        return None
    for _ in range(120):
        count = random.randint(MIN_MARKERS, max_count)
        chosen = sorted(random.sample(range(length), count))
        if any(b - a < 2 for a, b in zip(chosen, chosen[1:])):
            continue
        if chosen[0] > 2 or chosen[-1] < length - 3:
            continue
        return chosen
    return None


def _normal_candidates(length: int, step: int) -> list[int]:
    options = []
    for lo in range(step + 1, length - step - 1):
        hi = lo + step
        if hi >= length - 1:
            continue
        left_margin = lo
        right_margin = length - 1 - hi
        if left_margin < step + 1 or right_margin < step + 1:
            continue
        if left_margin >= 2 * step + 1 or right_margin >= 2 * step + 1:
            options.append(lo)
    return options


def _grow_mask(height: int, width: int, target: int) -> np.ndarray | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not _is_connected(mask):
        return None
    if not mask.any(axis=0).all() or not mask.any(axis=1).all():
        return None
    return mask


def _make_patch(patch_h: int, patch_w: int, flip_axis: int, bg: int, marker_color: int) -> np.ndarray | None:
    area = patch_h * patch_w
    min_cells = max(patch_h + patch_w - 1, int(np.ceil(area * 0.5)))
    max_cells = max(min_cells, int(np.floor(area * 0.8)))
    colors = random.sample([c for c in PALETTE if c not in {bg, marker_color}], random.randint(2, 4))
    phase = random.randrange(9)

    for _ in range(120):
        target = random.randint(min_cells, max_cells)
        mask = _grow_mask(patch_h, patch_w, target)
        if mask is None or np.array_equal(mask, np.flip(mask, axis=flip_axis)):
            continue

        patch = np.full((patch_h, patch_w), bg, dtype=int)
        for r, c in zip(*np.nonzero(mask)):
            color = colors[(3 * r + 5 * c + phase) % len(colors)]
            patch[r, c] = color
        if len(set(map(int, patch[patch != bg]))) < 2:
            continue
        if np.array_equal(patch, np.flip(patch, axis=flip_axis)):
            continue
        return patch
    return None


def _strip_index(x: int, lo: int, step: int) -> int:
    for n in range(-20, 21):
        if lo + n * step < x < lo + (n + 1) * step:
            return n
    raise AssertionError("point is not inside any corridor strip")


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        marker_color, axis, lo, hi = _find_mirror_data(inp)
    except Exception:
        return False

    bg = _get_bg(inp)
    motif_mask = (inp != bg) & (inp != marker_color)
    if not motif_mask.any():
        return False

    step = hi - lo
    coords = np.argwhere(motif_mask)
    if not np.all((coords[:, axis] > lo) & (coords[:, axis] < hi)):
        return False

    motif_box = _bbox(motif_mask)
    if motif_box is None:
        return False
    patch = inp[motif_box[0] : motif_box[2] + 1, motif_box[1] : motif_box[3] + 1]
    flip_axis = 0 if axis == 0 else 1
    if np.array_equal(patch, np.flip(patch, axis=flip_axis)):
        return False

    added = (out != inp) & (out != bg)
    if not added.any():
        return False

    strip_indices = set()
    odd_seen = False
    even_far_seen = False
    for rr, cc in np.argwhere(out != bg):
        r = int(rr)
        c = int(cc)
        if int(out[r, c]) == marker_color:
            continue
        normal = r if axis == 0 else c
        if normal == lo or normal == hi:
            return False
        n = _strip_index(normal, lo, step)
        strip_indices.add(n)
        if n % 2 != 0:
            odd_seen = True
        if n % 2 == 0 and n != 0:
            even_far_seen = True
    if not odd_seen or not even_far_seen:
        return False
    if all(abs(n) <= 1 for n in strip_indices):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _build_attempt() -> np.ndarray | None:
    axis = random.randint(0, 1)
    normal_len = random.randint(22, 30)
    parallel_len = random.randint(16, 24)
    step_choices = [step for step in range(4, 7) if _normal_candidates(normal_len, step)]
    if not step_choices:
        return None
    step = random.choice(step_choices)
    lo = random.choice(_normal_candidates(normal_len, step))
    hi = lo + step

    positions = _sample_sparse_positions(parallel_len)
    if positions is None:
        return None

    bg, marker_color = random.sample(PALETTE, 2)

    if axis == 1:
        height, width = parallel_len, normal_len
        patch_h = random.randint(5, min(10, height - 4))
        patch_w = random.randint(2, step - 2)
        flip_axis = 1
    else:
        height, width = normal_len, parallel_len
        patch_h = random.randint(2, step - 2)
        patch_w = random.randint(5, min(10, width - 4))
        flip_axis = 0

    patch = _make_patch(patch_h, patch_w, flip_axis, bg, marker_color)
    if patch is None:
        return None

    if axis == 1:
        r0 = random.randint(2, height - patch_h - 2)
        c0 = lo + random.randint(1, step - patch_w - 1)
    else:
        r0 = lo + random.randint(1, step - patch_h - 1)
        c0 = random.randint(2, width - patch_w - 2)

    inp = np.full((height, width), bg, dtype=int)
    if axis == 1:
        inp[positions, lo] = marker_color
        inp[positions, hi] = marker_color
    else:
        inp[lo, positions] = marker_color
        inp[hi, positions] = marker_color

    patch_mask = patch != bg
    view = inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    if np.any(view[patch_mask] != bg):
        return None
    view[patch_mask] = patch[patch_mask]

    try:
        marker_data = _find_mirror_data(inp)
    except Exception:
        return None
    if marker_data != (marker_color, axis, lo, hi):
        return None

    out = _render_gallery(inp, marker_color, axis, lo, hi)
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-mirror corridor input")
