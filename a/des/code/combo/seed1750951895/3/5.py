import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 300
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
STAMP_TEMPLATES = [
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=int,
    ),
]
QUADRANTS = ("ul", "ur", "ll", "lr")
ROTATION_BY_QUADRANT = {"ul": 0, "ur": 3, "lr": 2, "ll": 1}


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _crop_patch(panel: np.ndarray, color: int):
    bb = _bbox(panel == color)
    assert bb is not None
    r0, c0, r1, c1 = bb
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BACKGROUND, dtype=int)
    sub = panel[r0 : r1 + 1, c0 : c1 + 1]
    patch[sub == color] = color
    return patch


def _is_connected(mask: np.ndarray) -> bool:
    parts = _components(mask)
    return len(parts) == 1 and int(np.count_nonzero(mask)) == len(parts[0])


def _separator_info(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_ids = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1 and int(g[r, 0]) != BACKGROUND]
    col_ids = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1 and int(g[0, c]) != BACKGROUND]
    assert row_ids and col_ids, "expected separator rows and columns"
    sep_colors = {int(g[r, 0]) for r in row_ids} | {int(g[0, c]) for c in col_ids}
    assert len(sep_colors) == 1, "separator lines must share one color"
    sep = next(iter(sep_colors))

    def spans(length: int, ids: list[int]):
        out = []
        start = 0
        ids_set = set(ids)
        for i in range(length + 1):
            boundary = i == length or i in ids_set
            if boundary:
                if start < i:
                    out.append((start, i))
                start = i + 1
        return out

    row_spans = spans(h, row_ids)
    col_spans = spans(w, col_ids)
    assert len({r1 - r0 for r0, r1 in row_spans}) == 1
    assert len({c1 - c0 for c0, c1 in col_spans}) == 1
    return sep, row_spans, col_spans


def _panel_quadrant(r: int, c: int, h: int, w: int) -> str:
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    assert r != cr and c != cc, "marker cannot lie on the center row or column"
    top = r < cr
    left = c < cc
    if top and left:
        return "ul"
    if top and not left:
        return "ur"
    if not top and left:
        return "ll"
    return "lr"


def _stamp_centered(dst: np.ndarray, patch: np.ndarray, anchor_r: int, anchor_c: int, color: int):
    ph, pw = patch.shape
    r0 = anchor_r - ph // 2
    c0 = anchor_c - pw // 2
    r1 = r0 + ph
    c1 = c0 + pw
    assert 0 <= r0 and 0 <= c0 and r1 <= dst.shape[0] and c1 <= dst.shape[1], "stamp must fit panel"
    mask = patch == color
    view = dst[r0:r1, c0:c1]
    assert np.all((view == BACKGROUND) | (~mask)), "stamped copies must not overlap"
    view[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each separator-bounded panel, treat the large monochrome component as a stamp and
    isolated same-color pixels as targets. Stamp copies onto the targets, rotating by the
    target pixel's quadrant relative to the panel center, then erase the original source.
    """
    g = np.asarray(grid, dtype=int)
    sep, row_spans, col_spans = _separator_info(g)
    out = np.full_like(g, BACKGROUND)
    out[g == sep] = sep

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0:r1, c0:c1]
            colors = [int(x) for x in np.unique(panel) if int(x) not in (BACKGROUND, sep)]
            assert len(colors) == 1, "each panel must use one foreground color"
            color = colors[0]
            comps = _components(panel == color)
            big = [comp for comp in comps if len(comp) > 3]
            small = [comp for comp in comps if len(comp) == 1]
            assert len(big) == 1, "expected one stamp object per panel"
            assert len(small) == len(comps) - 1 and 1 <= len(small) <= 3, "expected 1-3 isolated markers"

            stamp_mask = np.zeros_like(panel, dtype=bool)
            for rr, cc in big[0]:
                stamp_mask[int(rr), int(cc)] = True
            patch = _crop_patch(np.where(stamp_mask, color, BACKGROUND), color)

            panel_out = np.full_like(panel, BACKGROUND)
            for comp in small:
                rr, cc = map(int, comp[0])
                quadrant = _panel_quadrant(rr, cc, panel.shape[0], panel.shape[1])
                rotated = np.rot90(patch, k=ROTATION_BY_QUADRANT[quadrant])
                _stamp_centered(panel_out, rotated, rr, cc, color)

            out[r0:r1, c0:c1] = panel_out

    return out


def _rotations_unique(mask: np.ndarray) -> bool:
    rots = {tuple(np.rot90(mask, k).ravel().tolist()) for k in range(4)}
    return len(rots) == 4


def _expand_template(template: np.ndarray, color: int) -> np.ndarray:
    arr = np.where(template > 0, color, BACKGROUND)
    bb = _bbox(arr == color)
    assert bb is not None
    r0, c0, r1, c1 = bb
    arr = arr[r0 : r1 + 1, c0 : c1 + 1]
    assert arr.shape[0] == arr.shape[1] and arr.shape[0] % 2 == 1
    assert _is_connected(arr == color)
    assert _rotations_unique(arr == color)
    assert np.count_nonzero(arr == color) > 3
    return arr


def _corner_positions(panel_h: int, panel_w: int, stamp_h: int, stamp_w: int):
    margin = 1
    max_r = panel_h - stamp_h - margin
    max_c = panel_w - stamp_w - margin
    min_r = 0
    min_c = 0
    if max_r < min_r:
        max_r = 0
    if max_c < min_c:
        max_c = 0
    choices = [
        (min_r, min_c),
        (min_r, max_c),
        (max_r, min_c),
        (max_r, max_c),
    ]
    return list(dict.fromkeys(choices))


def _quadrant_candidates(panel_h: int, panel_w: int, radius_r: int, radius_c: int, quadrant: str):
    cr = (panel_h - 1) / 2.0
    cc = (panel_w - 1) / 2.0
    rows = []
    cols = []
    for r in range(radius_r, panel_h - radius_r):
        if quadrant[0] == "u" and r < cr:
            rows.append(r)
        if quadrant[0] == "l" and r > cr:
            rows.append(r)
    for c in range(radius_c, panel_w - radius_c):
        if quadrant[1] == "l" and c < cc:
            cols.append(c)
        if quadrant[1] == "r" and c > cc:
            cols.append(c)
    return [(r, c) for r in rows for c in cols]


def _far_from_stamp(cells: list[tuple[int, int]], stamp: np.ndarray, top: int, left: int) -> bool:
    stamp_cells = {(top + r, left + c) for r, c in np.argwhere(stamp != BACKGROUND)}
    for r, c in cells:
        for sr, sc in stamp_cells:
            if max(abs(r - sr), abs(c - sc)) <= 1:
                return False
    return True


def _panels_non_overlapping(markers: list[tuple[int, int]], stamp_h: int, stamp_w: int) -> bool:
    boxes = []
    for r, c in markers:
        r0 = r - stamp_h // 2
        c0 = c - stamp_w // 2
        boxes.append((r0, c0, r0 + stamp_h - 1, c0 + stamp_w - 1))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a = boxes[i]
            b = boxes[j]
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                return False
    return True


def _place_panel(panel_h: int, panel_w: int, sep: int, color: int, quadrants: list[str]):
    stamp = _expand_template(np.rot90(random.choice(STAMP_TEMPLATES), random.randrange(4)), color)
    sh, sw = stamp.shape
    radius_r = sh // 2
    radius_c = sw // 2
    grid = np.full((panel_h, panel_w), BACKGROUND, dtype=int)

    stamp_positions = _corner_positions(panel_h, panel_w, sh, sw)
    random.shuffle(stamp_positions)
    placed = False
    stamp_top = stamp_left = 0
    for top, left in stamp_positions:
        if top + sh <= panel_h and left + sw <= panel_w:
            grid[top : top + sh, left : left + sw][stamp != BACKGROUND] = color
            stamp_top, stamp_left = top, left
            placed = True
            break
    assert placed

    marker_positions = []
    for quadrant in quadrants:
        candidates = _quadrant_candidates(panel_h, panel_w, radius_r, radius_c, quadrant)
        random.shuffle(candidates)
        chosen = None
        for r, c in candidates:
            if any(max(abs(r - rr), abs(c - cc)) <= max(sh, sw) for rr, cc in marker_positions):
                continue
            trial = marker_positions + [(r, c)]
            if not _panels_non_overlapping(trial, sh, sw):
                continue
            if not _far_from_stamp([(r, c)], stamp, stamp_top, stamp_left):
                continue
            chosen = (r, c)
            break
        if chosen is None:
            return None
        marker_positions.append(chosen)

    for r, c in marker_positions:
        grid[r, c] = color
    return grid


def _gen_attempt():
    panel_rows = random.choice([2, 3])
    panel_cols = random.choice([2, 3])
    panel_h = random.choice([7, 9, 11])
    panel_w = random.choice([7, 9, 11])
    sep = 5
    grid_h = panel_rows * panel_h + (panel_rows - 1)
    grid_w = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    for r in range(panel_h, grid_h, panel_h + 1):
        grid[r, :] = sep
    for c in range(panel_w, grid_w, panel_w + 1):
        grid[:, c] = sep

    panel_count = panel_rows * panel_cols
    color_choices = [random.choice(PALETTE) for _ in range(panel_count)]
    quadrant_pool = list(QUADRANTS)
    random.shuffle(quadrant_pool)

    used_quadrants = []
    for idx in range(panel_count):
        n_markers = random.choice([1, 2, 3])
        if idx < 4:
            quadrants = [quadrant_pool[idx]]
        else:
            quadrants = random.sample(QUADRANTS, n_markers)
        while len(quadrants) < n_markers:
            pick = random.choice(QUADRANTS)
            if pick not in quadrants:
                quadrants.append(pick)
        random.shuffle(quadrants)
        used_quadrants.extend(quadrants)

        panel = _place_panel(panel_h, panel_w, sep, color_choices[idx], quadrants)
        if panel is None:
            return None
        pr = idx // panel_cols
        pc = idx % panel_cols
        r0 = pr * (panel_h + 1)
        c0 = pc * (panel_w + 1)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    counts = Counter(used_quadrants)
    if any(counts[q] == 0 for q in QUADRANTS):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
