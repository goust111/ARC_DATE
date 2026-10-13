import random
from collections import Counter

import numpy as np


PIECE_SIZE = 4
INNER_SIZE = 2
MAX_ATTEMPTS = 300

BG = 0
GREY = 5
EDGE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
INNER_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
SLOT_PITCH = 7
SLOT_JITTER = 2
EXTRA_SLACK_RANGE = (0, 4)


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    color, freq = counts.most_common(1)[0]
    assert sum(v == freq for v in counts.values()) == 1, "expected a unique dominant color"
    return int(color)


def _pick_single(candidates: list[dict], key: str) -> dict:
    assert candidates, f"missing {key} legend"
    lengths = [cand[key] for cand in candidates]
    best = max(lengths)
    best_items = [cand for cand in candidates if cand[key] == best]
    assert len(best_items) == 1, f"ambiguous {key} legend"
    return best_items[0]


def _find_legends(grid: np.ndarray, bg: int) -> tuple[list[int], list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    vertical: list[dict] = []
    horizontal: list[dict] = []

    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        area = int(coords.shape[0])
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        if not touches_edge or area != hh * ww:
            continue
        item = {"coords": coords, "bbox": (r0, c0, r1, c1), "h": hh, "w": ww}
        if ww == 1 and hh >= 2:
            vertical.append(item)
        if hh == 1 and ww >= 2:
            horizontal.append(item)

    v = _pick_single(vertical, "h")
    u = _pick_single(horizontal, "w")

    vr0, vc0, vr1, _ = v["bbox"]
    hr0, hc0, _, hc1 = u["bbox"]
    row_colors = [int(grid[r, vc0]) for r in range(vr0, vr1 + 1)]
    col_colors = [int(grid[hr0, c]) for c in range(hc0, hc1 + 1)]
    assert len(set(row_colors)) == len(row_colors), "row legend colors must be unique"
    assert len(set(col_colors)) == len(col_colors), "column legend colors must be unique"

    legend_cells = set(map(tuple, v["coords"])) | set(map(tuple, u["coords"]))
    return row_colors, col_colors, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the 2x2 inner patterns by decoding each fragment's row and column
    from the top-edge and left-edge legend colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    row_colors, col_colors, legend_cells = _find_legends(g, bg)
    row_lookup = {color: idx for idx, color in enumerate(row_colors)}
    col_lookup = {color: idx for idx, color in enumerate(col_colors)}

    out = np.full((len(row_colors) * INNER_SIZE, len(col_colors) * INNER_SIZE), bg, dtype=int)
    occupied = np.zeros((len(row_colors), len(col_colors)), dtype=bool)

    for coords in _components(g != bg):
        coord_set = set(map(tuple, coords))
        if coord_set <= legend_cells:
            continue

        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == PIECE_SIZE and w == PIECE_SIZE, "unexpected non-legend component"
        assert int(coords.shape[0]) == PIECE_SIZE * PIECE_SIZE, "piece must be a filled square"

        piece = g[r0 : r1 + 1, c0 : c1 + 1]
        row_color = _dominant_color(piece[0, :])
        col_color = _dominant_color(piece[1:, 0])
        assert row_color in row_lookup, "piece row color missing from legend"
        assert col_color in col_lookup, "piece column color missing from legend"

        rr = row_lookup[row_color]
        cc = col_lookup[col_color]
        assert not occupied[rr, cc], "duplicate piece position"
        out[rr * INNER_SIZE : (rr + 1) * INNER_SIZE, cc * INNER_SIZE : (cc + 1) * INNER_SIZE] = piece[1:3, 1:3]
        occupied[rr, cc] = True

    assert occupied.all(), "expected one piece for every legend row/column pair"
    return out


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _sample_unique_patterns(count: int) -> list[np.ndarray]:
    seen: set[tuple[int, ...]] = set()
    patterns: list[np.ndarray] = []
    while len(patterns) < count:
        arr = np.array([random.choice(INNER_COLOR_POOL) for _ in range(INNER_SIZE * INNER_SIZE)], dtype=int).reshape(INNER_SIZE, INNER_SIZE)
        key = tuple(map(int, arr.reshape(-1)))
        if len(set(key)) == 1 or key in seen:
            continue
        seen.add(key)
        patterns.append(arr)
    return patterns


def _make_piece(row_color: int, col_color: int, inner: np.ndarray) -> np.ndarray:
    piece = np.full((PIECE_SIZE, PIECE_SIZE), GREY, dtype=int)
    piece[0, :] = row_color
    piece[1:3, 0] = col_color
    piece[1:3, 1:3] = inner
    return piece


def _assemble_expected(patterns: list[np.ndarray], n_rows: int, n_cols: int) -> np.ndarray:
    out = np.full((n_rows * INNER_SIZE, n_cols * INNER_SIZE), BG, dtype=int)
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            out[r * INNER_SIZE : (r + 1) * INNER_SIZE, c * INNER_SIZE : (c + 1) * INNER_SIZE] = patterns[idx]
            idx += 1
    return out


def _slot_grid(piece_count: int) -> tuple[int, int]:
    candidates = [
        (sr, sc)
        for sr in range(2, 7)
        for sc in range(2, 7)
        if sr * sc >= piece_count and sr * sc <= piece_count + 8
    ]
    assert candidates
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    n_rows = _randint(ROW_COUNT_RANGE)
    n_cols = _randint(COL_COUNT_RANGE)
    piece_count = n_rows * n_cols

    palette = list(EDGE_COLOR_POOL)
    random.shuffle(palette)
    row_colors = palette[:n_rows]
    col_colors = palette[n_rows : n_rows + n_cols]

    vertical_side = random.choice(("left", "right"))
    horizontal_side = random.choice(("top", "bottom"))
    slot_rows, slot_cols = _slot_grid(piece_count)

    top_clear = 2 if horizontal_side == "top" else 1
    bottom_clear = 2 if horizontal_side == "bottom" else 1
    left_clear = 2 if vertical_side == "left" else 1
    right_clear = 2 if vertical_side == "right" else 1

    h = top_clear + (slot_rows - 1) * SLOT_PITCH + SLOT_JITTER + PIECE_SIZE + bottom_clear + _randint(EXTRA_SLACK_RANGE)
    w = left_clear + (slot_cols - 1) * SLOT_PITCH + SLOT_JITTER + PIECE_SIZE + right_clear + _randint(EXTRA_SLACK_RANGE)
    if h - n_rows - 4 < 0 or w - n_cols - 4 < 0:
        return None

    grid = np.full((h, w), BG, dtype=int)

    vcol = 0 if vertical_side == "left" else w - 1
    hrow = 0 if horizontal_side == "top" else h - 1
    vstart = random.randint(2, h - n_rows - 2)
    hstart = random.randint(2, w - n_cols - 2)

    for idx, color in enumerate(row_colors):
        grid[vstart + idx, vcol] = color
    for idx, color in enumerate(col_colors):
        grid[hrow, hstart + idx] = color

    patterns = _sample_unique_patterns(piece_count)
    expected = _assemble_expected(patterns, n_rows, n_cols)

    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    chosen = random.sample(slots, piece_count)
    random.shuffle(chosen)

    idx = 0
    for rr in range(n_rows):
        for cc in range(n_cols):
            slot_r, slot_c = chosen[idx]
            idx += 1
            r = top_clear + slot_r * SLOT_PITCH + random.randint(0, SLOT_JITTER)
            c = left_clear + slot_c * SLOT_PITCH + random.randint(0, SLOT_JITTER)
            grid[r : r + PIECE_SIZE, c : c + PIECE_SIZE] = _make_piece(row_colors[rr], col_colors[cc], patterns[(rr * n_cols) + cc])

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
