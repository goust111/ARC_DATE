from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 2
SEPARATOR_COLOR = 4
SHADOW_COLOR = 5
DISTRACTOR_COLORS = (1, 3, 6, 7, 8, 9)
GEN_RETRIES = 400
RNG = np.random.default_rng()


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the invisible translated template from its top and left count shadows, then draw it.

    Input structure:
    - A full-width separator row and a full-height separator column split the grid into a top band, a left band, and the main canvas.
    - The main canvas contains one red multi-cell template placed flush to a corner, plus one non-red distractor shape.
    - The top band holds grey vertical shadow bars, and the left band holds grey horizontal shadow bars, encoding the invisible copy's per-column and per-row occupancies.

    Rule:
    - Crop the red corner template to its bounding box and read its column-count and row-count sequences.
    - In the top band, find the unique horizontal alignment whose shadow counts exactly equal the template's column counts; do the same in the left band for the row counts.
    - Crop away both bands and separators, then paste one red translated copy of the template at the intersection of those two alignments, keeping the visible template and distractor unchanged.

    Key insight:
    - The target object never appears in the input canvas at all; only its one-dimensional shadows exist, so the template must be projected onto both axes and aligned against those shadows to recover the hidden location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_r, sep_c = _find_separators(g)
    canvas = g[sep_r + 1 :, sep_c + 1 :].copy()
    top_band = g[:sep_r, sep_c + 1 :]
    left_band = g[sep_r + 1 :, :sep_c]

    template_mask = canvas == TEMPLATE_COLOR
    bbox = _bbox(template_mask)
    assert bbox is not None, "expected one visible red template"
    r0, c0, r1, c1 = bbox
    sprite_mask = template_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert sprite_mask.any()

    col_counts = sprite_mask.sum(axis=0).astype(int)
    row_counts = sprite_mask.sum(axis=1).astype(int)
    assert np.all(col_counts > 0) and np.all(row_counts > 0)

    top_counts = np.count_nonzero(top_band == SHADOW_COLOR, axis=0).astype(int)
    left_counts = np.count_nonzero(left_band == SHADOW_COLOR, axis=1).astype(int)

    hidden_c = _unique_alignment(top_counts, col_counts)
    hidden_r = _unique_alignment(left_counts, row_counts)

    out = canvas.copy()
    out[hidden_r : hidden_r + sprite_mask.shape[0], hidden_c : hidden_c + sprite_mask.shape[1]][sprite_mask] = TEMPLATE_COLOR
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid shadow-localization input")


def _gen_attempt() -> np.ndarray | None:
    template_h = int(RNG.integers(3, 6))
    template_w = int(RNG.integers(3, 6))
    template = _random_connected_mask(template_h, template_w, require_alternative=True)
    if template is None:
        return None

    row_counts = template.sum(axis=1).astype(int)
    col_counts = template.sum(axis=0).astype(int)

    top_band_h = int(col_counts.max() + RNG.integers(0, 3))
    left_band_w = int(row_counts.max() + RNG.integers(0, 3))

    min_canvas_h = max(10, template_h + 5)
    min_canvas_w = max(10, template_w + 5)
    max_canvas_h = min(22, 30 - (top_band_h + 1))
    max_canvas_w = min(22, 30 - (left_band_w + 1))
    if min_canvas_h > max_canvas_h or min_canvas_w > max_canvas_w:
        return None

    canvas_h = int(RNG.integers(min_canvas_h, max_canvas_h + 1))
    canvas_w = int(RNG.integers(min_canvas_w, max_canvas_w + 1))

    corner = int(RNG.integers(0, 4))
    template_pos = _corner_anchor(corner, canvas_h, canvas_w, template_h, template_w)

    hidden_pos = _pick_free_anchor(
        canvas_h,
        canvas_w,
        template_h,
        template_w,
        forbidden=[(*template_pos, template_h, template_w)],
        pad=1,
    )
    if hidden_pos is None:
        return None

    same_size_distractor = bool(RNG.integers(0, 2))
    distractor_h = template_h if same_size_distractor else int(RNG.integers(2, 6))
    distractor_w = template_w if same_size_distractor else int(RNG.integers(2, 6))
    distractor = _random_distractor_mask(distractor_h, distractor_w, template, row_counts, col_counts)
    if distractor is None:
        return None

    distractor_pos = _pick_free_anchor(
        canvas_h,
        canvas_w,
        distractor.shape[0],
        distractor.shape[1],
        forbidden=[
            (*template_pos, template_h, template_w),
            (*hidden_pos, template_h, template_w),
        ],
        pad=1,
        avoid_corners=True,
    )
    if distractor_pos is None:
        return None

    canvas = np.full((canvas_h, canvas_w), BACKGROUND, dtype=int)
    _paint_mask(canvas, template, template_pos[0], template_pos[1], TEMPLATE_COLOR)
    distractor_color = int(RNG.choice(DISTRACTOR_COLORS))
    _paint_mask(canvas, distractor, distractor_pos[0], distractor_pos[1], distractor_color)

    output = canvas.copy()
    _paint_mask(output, template, hidden_pos[0], hidden_pos[1], TEMPLATE_COLOR)

    inp = np.full((top_band_h + 1 + canvas_h, left_band_w + 1 + canvas_w), BACKGROUND, dtype=int)
    inp[top_band_h, :] = SEPARATOR_COLOR
    inp[:, left_band_w] = SEPARATOR_COLOR
    inp[top_band_h, left_band_w] = SEPARATOR_COLOR
    inp[top_band_h + 1 :, left_band_w + 1 :] = canvas

    hidden_cols = template.sum(axis=0).astype(int)
    hidden_rows = template.sum(axis=1).astype(int)
    for offset, count in enumerate(hidden_cols):
        col = left_band_w + 1 + hidden_pos[1] + offset
        inp[top_band_h - count : top_band_h, col] = SHADOW_COLOR
    for offset, count in enumerate(hidden_rows):
        row = top_band_h + 1 + hidden_pos[0] + offset
        inp[row, left_band_w - count : left_band_w] = SHADOW_COLOR

    if not _is_valid_instance(inp, output, template, hidden_pos, distractor, distractor_pos):
        return None
    return inp


def _find_separators(grid: np.ndarray) -> tuple[int, int]:
    height, width = grid.shape
    sep_rows = [r for r in range(height) if np.all(grid[r, :] == SEPARATOR_COLOR)]
    sep_cols = [c for c in range(width) if np.all(grid[:, c] == SEPARATOR_COLOR)]
    assert len(sep_rows) == 1 and len(sep_cols) == 1
    return int(sep_rows[0]), int(sep_cols[0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _unique_alignment(signal: np.ndarray, pattern: np.ndarray) -> int:
    starts = []
    width = len(pattern)
    for start in range(len(signal) - width + 1):
        if np.array_equal(signal[start : start + width], pattern):
            starts.append(start)
    assert len(starts) == 1, f"expected unique alignment, found {starts}"
    return int(starts[0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(height: int, width: int, *, require_alternative: bool) -> np.ndarray | None:
    area = height * width
    min_fill = max(height + width - 2, (area + 1) // 2)
    max_fill = area - 1
    if min_fill > max_fill:
        return None

    for _ in range(120):
        target = int(RNG.integers(min_fill, max_fill + 1))
        mask = np.zeros((height, width), dtype=bool)
        start = (int(RNG.integers(0, height)), int(RNG.integers(0, width)))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            nr, nc = frontier_list[int(RNG.integers(0, len(frontier_list)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        if not np.all(mask.sum(axis=0) > 0) or not np.all(mask.sum(axis=1) > 0):
            continue
        row_counts = mask.sum(axis=1).astype(int)
        col_counts = mask.sum(axis=0).astype(int)
        if len(set(map(int, row_counts))) < 2 or len(set(map(int, col_counts))) < 2:
            continue
        if require_alternative and not _has_alternative_mask(row_counts, col_counts, mask):
            continue
        return mask
    return None


def _has_alternative_mask(row_counts: np.ndarray, col_counts: np.ndarray, original: np.ndarray) -> bool:
    row_counts_t = tuple(map(int, row_counts))
    col_counts_t = tuple(map(int, col_counts))
    height, width = original.shape

    row_patterns: dict[int, list[tuple[int, ...]]] = {}
    for count in set(row_counts_t):
        patterns = []
        for cols in itertools.combinations(range(width), count):
            row = [0] * width
            for c in cols:
                row[c] = 1
            patterns.append(tuple(row))
        row_patterns[count] = patterns

    memo: set[tuple[int, tuple[int, ...], bool]] = set()
    original_rows = [tuple(map(int, row)) for row in original.astype(int)]

    def search(r: int, remaining: tuple[int, ...], same_prefix: bool) -> bool:
        state = (r, remaining, same_prefix)
        if state in memo:
            return False
        if r == height:
            return all(v == 0 for v in remaining) and not same_prefix

        rows_left = height - r - 1
        for pattern in row_patterns[row_counts_t[r]]:
            new_remaining = tuple(remaining[c] - pattern[c] for c in range(width))
            if any(v < 0 or v > rows_left for v in new_remaining):
                continue
            next_same = same_prefix and pattern == original_rows[r]
            if search(r + 1, new_remaining, next_same):
                return True

        memo.add(state)
        return False

    return search(0, col_counts_t, True)


def _corner_anchor(corner: int, canvas_h: int, canvas_w: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == 0:
        return 0, 0
    if corner == 1:
        return 0, canvas_w - obj_w
    if corner == 2:
        return canvas_h - obj_h, 0
    return canvas_h - obj_h, canvas_w - obj_w


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah - 1 + pad < br
        or br + bh - 1 + pad < ar
        or ac + aw - 1 + pad < bc
        or bc + bw - 1 + pad < ac
    )


def _pick_free_anchor(
    canvas_h: int,
    canvas_w: int,
    obj_h: int,
    obj_w: int,
    *,
    forbidden: list[tuple[int, int, int, int]],
    pad: int,
    avoid_corners: bool = False,
) -> tuple[int, int] | None:
    choices = []
    for r in range(canvas_h - obj_h + 1):
        for c in range(canvas_w - obj_w + 1):
            rect = (r, c, obj_h, obj_w)
            if any(_rects_overlap(rect, other, pad) for other in forbidden):
                continue
            if avoid_corners and (
                (r == 0 and c == 0)
                or (r == 0 and c + obj_w == canvas_w)
                or (r + obj_h == canvas_h and c == 0)
                or (r + obj_h == canvas_h and c + obj_w == canvas_w)
            ):
                continue
            choices.append((r, c))
    if not choices:
        return None
    return choices[int(RNG.integers(0, len(choices)))]


def _random_distractor_mask(
    height: int,
    width: int,
    template: np.ndarray,
    template_rows: np.ndarray,
    template_cols: np.ndarray,
) -> np.ndarray | None:
    for _ in range(160):
        distractor = _random_connected_mask(height, width, require_alternative=False)
        if distractor is None:
            continue
        if distractor.shape == template.shape and np.array_equal(distractor, template):
            continue
        same_rows = distractor.shape[0] == len(template_rows) and np.array_equal(distractor.sum(axis=1), template_rows)
        same_cols = distractor.shape[1] == len(template_cols) and np.array_equal(distractor.sum(axis=0), template_cols)
        if same_rows and same_cols:
            continue
        return distractor
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    view = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    view[mask] = color


def _is_valid_instance(
    inp: np.ndarray,
    output: np.ndarray,
    template: np.ndarray,
    hidden_pos: tuple[int, int],
    distractor: np.ndarray,
    distractor_pos: tuple[int, int],
) -> bool:
    try:
        produced = main(inp)
    except Exception:
        return False
    if not np.array_equal(produced, output):
        return False

    sep_r, sep_c = _find_separators(inp)
    canvas = inp[sep_r + 1 :, sep_c + 1 :]
    top_counts = np.count_nonzero(inp[:sep_r, sep_c + 1 :] == SHADOW_COLOR, axis=0).astype(int)
    left_counts = np.count_nonzero(inp[sep_r + 1 :, :sep_c] == SHADOW_COLOR, axis=1).astype(int)
    template_rows = template.sum(axis=1).astype(int)
    template_cols = template.sum(axis=0).astype(int)

    horiz = [c for c in range(canvas.shape[1] - template.shape[1] + 1) if np.array_equal(top_counts[c : c + template.shape[1]], template_cols)]
    vert = [r for r in range(canvas.shape[0] - template.shape[0] + 1) if np.array_equal(left_counts[r : r + template.shape[0]], template_rows)]
    if horiz != [hidden_pos[1]] or vert != [hidden_pos[0]]:
        return False
    if np.any(top_counts[: hidden_pos[1]] != 0) or np.any(top_counts[hidden_pos[1] + template.shape[1] :] != 0):
        return False
    if np.any(left_counts[: hidden_pos[0]] != 0) or np.any(left_counts[hidden_pos[0] + template.shape[0] :] != 0):
        return False

    bbox = _bbox(canvas == TEMPLATE_COLOR)
    if bbox is None:
        return False
    br0, bc0, br1, bc1 = bbox
    if (br1 - br0 + 1, bc1 - bc0 + 1) != template.shape:
        return False

    dr, dc = distractor_pos
    distractor_view = canvas[dr : dr + distractor.shape[0], dc : dc + distractor.shape[1]]
    if int(np.count_nonzero(distractor_view)) != int(distractor.sum()):
        return False
    if distractor.shape == template.shape:
        same_rows = np.array_equal(distractor.sum(axis=1), template_rows)
        same_cols = np.array_equal(distractor.sum(axis=0), template_cols)
        if same_rows and same_cols:
            return False
    return True
