import hashlib
import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _fill_component(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    fill = np.zeros(shape, dtype=bool)
    rows = sorted(set(map(int, coords[:, 0])))
    for r in rows:
        cols = coords[coords[:, 0] == r, 1]
        assert cols.size == 2, "each occupied row must have exactly two boundary pixels"
        c0 = int(cols.min())
        c1 = int(cols.max())
        fill[r, c0 : c1 + 1] = True
    return fill


def _spans_to_coords(spans: list[tuple[int, int]]) -> np.ndarray:
    pts = []
    for r, (left, right) in enumerate(spans):
        pts.append((r, left))
        pts.append((r, right))
    return np.asarray(pts, dtype=int)


def _boundary_touch(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    h = max(int(coords_a[:, 0].max()), int(coords_b[:, 0].max())) + 3
    w = max(int(coords_a[:, 1].max()), int(coords_b[:, 1].max())) + 3
    dilated = np.zeros((h, w), dtype=bool)
    aa = coords_a + 1
    bb = coords_b + 1
    for r, c in aa:
        dilated[r - 1 : r + 2, c - 1 : c + 2] = True
    return bool(np.any(dilated[bb[:, 0], bb[:, 1]]))


def _wrong_outermost_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    color = next(int(v) for v in np.unique(g) if int(v) != bg)
    out = np.full_like(g, bg)
    for r in range(g.shape[0]):
        cols = np.where(g[r] == color)[0]
        if cols.size:
            out[r, int(cols.min()) : int(cols.max()) + 1] = color
    return out


def _wrong_adjacent_pair_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    color = next(int(v) for v in np.unique(g) if int(v) != bg)
    out = np.full_like(g, bg)
    for r in range(g.shape[0]):
        cols = sorted(np.where(g[r] == color)[0].tolist())
        if len(cols) == 2:
            out[r, cols[0] : cols[1] + 1] = color
        elif len(cols) == 4:
            out[r, cols[0] : cols[1] + 1] = color
            out[r, cols[2] : cols[3] + 1] = color
        else:
            assert len(cols) == 0, "generator expects only 0, 2, or 4 foreground pixels per row"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill two same-colored sparse outlines separately, then erase their overlap by symmetric difference.

    Input structure:
    - The grid has a dominant background and exactly two disconnected sparse outlines of one foreground color.
    - Each outline is 8-connected, and every row touched by one outline contains exactly two pixels: its left and right row boundaries.
    - Some rows contain boundary pixels from both outlines, producing four same-colored pixels on that row.

    Rule:
    - Split the foreground by connectivity into the two outlines before doing any row pairing.
    - For each outline independently, fill every occupied row from its left boundary pixel to its right boundary pixel.
    - Paint cells covered by exactly one filled shape with the outline color; paint doubly covered cells back to the background.

    Key insight:
    - The visible outlines must be separated first; pairing the four pixels on a shared row directly fills the wrong span, while the true answer is the symmetric difference of the two independently filled solids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    color = colors[0]

    comps = _components(g == color, connectivity=8)
    assert len(comps) == 2, "expected exactly two disconnected outlines"

    fills = [_fill_component(g.shape, comp) for comp in comps]
    out = np.full_like(g, bg)
    out[fills[0] ^ fills[1]] = color
    return out


_INC_CHOICES = {0: [(0, 0), (-1, -1), (1, 1)], 1: [(-1, 0), (0, 1)], 2: [(-1, 1)]}
_DEC_CHOICES = {0: [(0, 0), (-1, -1), (1, 1)], 1: [(1, 0), (0, -1)], 2: [(1, -1)]}
_SHIFT_CHOICES = [(-1, -1), (0, 0), (1, 1)]
_MAX_GEN_ATTEMPTS = 800
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _alloc_steps(total: int, n_steps: int) -> list[int]:
    vals = [0] * n_steps
    for _ in range(total):
        open_slots = [idx for idx, v in enumerate(vals) if v < 2]
        vals[random.choice(open_slots)] += 1
    random.shuffle(vals)
    return vals


def _generate_outer_spans() -> list[tuple[int, int]]:
    while True:
        height = random.randint(8, 14)
        plateau = random.randint(0, min(2, height - 5))
        remaining = height - 1 - plateau
        up = random.randint(2, remaining - 2)
        down = remaining - up
        peak_cap = min(11, 1 + 2 * min(up, down))
        peak = random.randint(min(6, peak_cap), peak_cap)

        incs = _alloc_steps(peak - 1, up)
        decs = _alloc_steps(peak - 1, down)

        left = 0
        right = 1
        lefts = [left]
        rights = [right]

        for inc in incs:
            dl, dr = random.choice(_INC_CHOICES[inc])
            left += dl
            right += dr
            lefts.append(left)
            rights.append(right)

        for _ in range(plateau):
            dl, dr = random.choice(_SHIFT_CHOICES)
            left += dl
            right += dr
            lefts.append(left)
            rights.append(right)

        for dec in decs:
            dl, dr = random.choice(_DEC_CHOICES[dec])
            left += dl
            right += dr
            lefts.append(left)
            rights.append(right)

        gaps = [r - l for l, r in zip(lefts, rights)]
        if max(gaps) < 6 or sum(g >= 5 for g in gaps) < 5:
            continue

        min_c = min(lefts)
        lefts = [x - min_c for x in lefts]
        rights = [x - min_c for x in rights]

        if random.random() < 0.5:
            max_c = max(rights)
            lefts, rights = [max_c - x for x in rights], [max_c - x for x in lefts]
        if random.random() < 0.5:
            lefts.reverse()
            rights.reverse()

        return list(zip(lefts, rights))


def _find_inner_spans(outer_spans: list[tuple[int, int]]) -> list[tuple[int, int] | None] | None:
    h = len(outer_spans)
    outer_coords = _spans_to_coords(outer_spans)

    row_windows = []
    for start in range(1, h - 3):
        for end in range(start + 3, h - 1):
            window = outer_spans[start : end + 1]
            if all((right - left) >= 5 for left, right in window):
                row_windows.append((start, end))
    random.shuffle(row_windows)

    for start, end in row_windows:
        window = outer_spans[start : end + 1]
        options_by_row: list[list[tuple[int, int]]] = []
        for idx, (outer_left, outer_right) in enumerate(window):
            row_options = []
            for left in range(outer_left + 2, outer_right - 2):
                for right in range(left + 1, outer_right - 1):
                    gap = right - left
                    if idx in (0, len(window) - 1) and gap != 1:
                        continue
                    row_options.append((left, right))
            if not row_options:
                break
            options_by_row.append(row_options)
        else:
            layers: list[dict[tuple[int, int], tuple[int, int] | None]] = [{pair: None for pair in options_by_row[0]}]
            feasible = True
            for row_idx in range(1, len(options_by_row)):
                cur: dict[tuple[int, int], tuple[int, int]] = {}
                for pair in options_by_row[row_idx]:
                    preds = [
                        prev
                        for prev in layers[-1]
                        if abs(pair[0] - prev[0]) <= 1 and abs(pair[1] - prev[1]) <= 1
                    ]
                    if preds:
                        cur[pair] = random.choice(preds)
                if not cur:
                    feasible = False
                    break
                layers.append(cur)
            if not feasible:
                continue

            end_pair = random.choice(list(layers[-1]))
            path = [end_pair]
            for idx in range(len(layers) - 1, 0, -1):
                path.append(layers[idx][path[-1]])
            path.reverse()

            inner_spans: list[tuple[int, int] | None] = [None] * h
            for offset, pair in enumerate(path):
                inner_spans[start + offset] = pair

            inner_coords = np.asarray(
                [(r, pair[0]) for r, pair in enumerate(inner_spans) if pair is not None]
                + [(r, pair[1]) for r, pair in enumerate(inner_spans) if pair is not None],
                dtype=int,
            )
            if _boundary_touch(outer_coords, inner_coords):
                continue

            gaps = [pair[1] - pair[0] for pair in path]
            inner_area = sum(gap + 1 for gap in gaps)
            outer_area = sum(right - left + 1 for left, right in outer_spans)
            if max(gaps) < 3:
                continue
            if not (0.12 <= inner_area / outer_area <= 0.45):
                continue
            return inner_spans

    return None


def _render_grid(
    outer_spans: list[tuple[int, int]],
    inner_spans: list[tuple[int, int] | None],
    *,
    color: int,
    bg: int,
) -> np.ndarray:
    outer_coords = _spans_to_coords(outer_spans)
    inner_coords = np.asarray(
        [(r, pair[0]) for r, pair in enumerate(inner_spans) if pair is not None]
        + [(r, pair[1]) for r, pair in enumerate(inner_spans) if pair is not None],
        dtype=int,
    )
    all_coords = np.vstack([outer_coords, inner_coords])
    base_h = int(all_coords[:, 0].max()) + 1
    base_w = int(all_coords[:, 1].max()) + 1

    pad_top = random.randint(1, 4)
    pad_left = random.randint(1, 4)
    pad_bottom = random.randint(1, min(4, 30 - base_h - pad_top))
    pad_right = random.randint(1, min(4, 30 - base_w - pad_left))

    h = base_h + pad_top + pad_bottom
    w = base_w + pad_left + pad_right
    grid = np.full((h, w), bg, dtype=int)

    outer_shifted = outer_coords + np.array([pad_top, pad_left])
    inner_shifted = inner_coords + np.array([pad_top, pad_left])
    grid[outer_shifted[:, 0], outer_shifted[:, 1]] = color
    grid[inner_shifted[:, 0], inner_shifted[:, 1]] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        outer_spans = _generate_outer_spans()
        inner_spans = _find_inner_spans(outer_spans)
        if inner_spans is None:
            continue

        color = random.randint(1, 9)
        grid = _render_grid(outer_spans, inner_spans, color=color, bg=0)

        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, _wrong_outermost_fill(grid)):
            continue
        if np.array_equal(out, _wrong_adjacent_pair_fill(grid)):
            continue

        comps = _components(grid == color, connectivity=8)
        if len(comps) != 2:
            continue
        fills = [_fill_component(grid.shape, comp) for comp in comps]
        overlap_area = int((fills[0] & fills[1]).sum())
        if overlap_area < 5:
            continue
        if overlap_area * 4 >= int((fills[0] | fills[1]).sum()):
            continue

        four_point_rows = sum(int(np.count_nonzero(grid[r] == color) == 4) for r in range(grid.shape[0]))
        if four_point_rows < 3:
            continue

        input_sig = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
        output_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_INPUT_SIGS.add(input_sig)
        _SEEN_OUTPUT_SIGS.add(output_sig)
        return grid

    raise RuntimeError("failed to generate a valid nested-overlap outline puzzle")
