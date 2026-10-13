import random
from collections import deque

import numpy as np


BG = 0
SEP = 3
PALETTE = [1, 2, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 1), (2, 1), (2, 0)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 0), (1, 1)],
]

SELECTED_RECTS = [
    np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], dtype=int),
]


def _normalize(coords):
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _transform_shape(coords):
    arr = np.array(coords, dtype=int)
    if random.random() < 0.5:
        arr[:, 1] *= -1
    if random.random() < 0.5:
        arr[:, 0] *= -1
    if random.random() < 0.5:
        arr = arr[:, [1, 0]]
    return _normalize(arr)


def _shape_dims(coords):
    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def _components(grid, bg=BG):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objs = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            objs.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox_left": int(coords[:, 1].min()),
                    "bbox_top": int(coords[:, 0].min()),
                }
            )
    return objs


def main(grid: np.ndarray) -> np.ndarray:
    """
    In three separator-split panels, select the left/middle/right ordered object
    from panel 1/2/3 and overlay them on one panel-sized canvas. Later panels
    overwrite earlier pixels where selected objects overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert (w - 2) % 3 == 0, "expected three equal-width panels and two separators"
    panel_w = (w - 2) // 3
    sep_cols = [panel_w, 2 * panel_w + 1]
    for col in sep_cols:
        assert np.all(g[:, col] == SEP), "separator columns must be solid green"

    panels = [
        g[:, :panel_w],
        g[:, panel_w + 1 : 2 * panel_w + 1],
        g[:, 2 * panel_w + 2 :],
    ]
    out = np.full((h, panel_w), BG, dtype=int)

    for panel_index, panel in enumerate(panels):
        objs = _components(panel, bg=BG)
        assert len(objs) == 3, "each panel must contain exactly three objects"
        assert len({obj["color"] for obj in objs}) == 3, "panel object colors must be distinct"
        lefts = [obj["bbox_left"] for obj in objs]
        assert len(set(lefts)) == 3, "objects must be rankable by bbox left column"
        objs.sort(key=lambda obj: (obj["bbox_left"], obj["bbox_top"]))
        chosen = objs[panel_index]
        coords = chosen["coords"]
        out[coords[:, 0], coords[:, 1]] = int(chosen["color"])

    return out


def _touches_existing(occupied, rr, cc):
    h, w = occupied.shape
    for r, c in zip(rr, cc):
        if occupied[r, c]:
            return True
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return True
    return False


def _place_shape(panel, occupied, coords, color, top, left):
    rr = coords[:, 0] + top
    cc = coords[:, 1] + left
    h, w = panel.shape
    if rr.min() < 0 or cc.min() < 0 or rr.max() >= h or cc.max() >= w:
        return False
    if _touches_existing(occupied, rr, cc):
        return False
    panel[rr, cc] = color
    occupied[rr, cc] = True
    return True


def _random_distractor_shape(max_h, max_w):
    for _ in range(40):
        coords = _transform_shape(random.choice(SHAPES))
        sh, sw = _shape_dims(coords)
        if sh <= max_h and sw <= max_w:
            return coords
    return np.array([(0, 0), (0, 1)], dtype=int)


def _allowed_cols(panel_index, sel_left, shape_w, panel_w):
    max_left = panel_w - shape_w
    if panel_index == 0:
        base = [c for c in range(sel_left + 2, max_left + 1)]
    elif panel_index == 1:
        base = [c for c in range(0, sel_left - shape_w)] + [c for c in range(sel_left + 2, max_left + 1)]
    else:
        base = [c for c in range(0, sel_left - shape_w)]
    random.shuffle(base)
    return base


def _choose_selected_layout(h, panel_w, overlap_mode):
    for _ in range(120):
        shape = random.choice(SELECTED_RECTS).copy()
        sh, sw = _shape_dims(shape)
        if overlap_mode:
            if sw < 2:
                continue
            d12 = random.randint(1, sw)
            d23 = random.randint(1, sw)
            if d12 >= sw and d23 >= sw:
                continue
            max_c1 = panel_w - sw - d12 - d23
            if max_c1 < 0:
                continue
            c1 = random.randint(0, max_c1)
            c2 = c1 + d12
            c3 = c2 + d23
            if c3 + sw > panel_w:
                continue
            top = random.randint(0, h - sh)
            return shape, top, (c1, c2, c3)
        for _inner in range(80):
            lefts = sorted(random.sample(range(0, panel_w - sw + 1), 3))
            tops = tuple(random.randint(0, h - sh) for _ in range(3))
            masks = _render_selected_sets(shape, tops, lefts)
            ok = True
            for i in range(3):
                for j in range(i + 1, 3):
                    if masks[i] & masks[j]:
                        ok = False
            if ok:
                return shape, tops, tuple(lefts)
    return None


def _render_selected_sets(shape, tops, lefts):
    rendered = []
    for top, left in zip(tops, lefts):
        rendered.append({(top + int(r), left + int(c)) for r, c in shape})
    return rendered


def _gen_attempt():
    h = random.randint(7, 10)
    panel_w = random.randint(7, 10)
    overlap_mode = random.random() < 0.5
    layout = _choose_selected_layout(h, panel_w, overlap_mode)
    if layout is None:
        return None

    if overlap_mode:
        shape, top, lefts = layout
        tops = (top, top, top)
    else:
        shape, tops, lefts = layout

    selected_sets = _render_selected_sets(shape, tops, lefts)
    if overlap_mode:
        overlap_count = 0
        for i in range(3):
            for j in range(i + 1, 3):
                overlap_count += len(selected_sets[i] & selected_sets[j])
        if overlap_count == 0:
            return None
    else:
        for i in range(3):
            for j in range(i + 1, 3):
                if selected_sets[i] & selected_sets[j]:
                    return None

    selected_colors = random.sample(PALETTE, 3)
    panels = []
    for panel_index in range(3):
        panel = np.full((h, panel_w), BG, dtype=int)
        occupied = np.zeros((h, panel_w), dtype=bool)
        if not _place_shape(panel, occupied, shape, selected_colors[panel_index], tops[panel_index], lefts[panel_index]):
            return None

        used_colors = {selected_colors[panel_index]}
        for _ in range(2):
            placed = False
            for _attempt in range(120):
                color = random.choice([c for c in PALETTE if c not in used_colors])
                coords = _random_distractor_shape(h, panel_w)
                sh, sw = _shape_dims(coords)
                cols = _allowed_cols(panel_index, lefts[panel_index], sw, panel_w)
                if not cols:
                    continue
                for left in cols:
                    rows = list(range(0, h - sh + 1))
                    random.shuffle(rows)
                    for top in rows:
                        if _place_shape(panel, occupied, coords, color, top, left):
                            used_colors.add(color)
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                return None
        panels.append(panel)

    grid = np.full((h, 3 * panel_w + 2), BG, dtype=int)
    grid[:, panel_w] = SEP
    grid[:, 2 * panel_w + 1] = SEP
    grid[:, :panel_w] = panels[0]
    grid[:, panel_w + 1 : 2 * panel_w + 1] = panels[1]
    grid[:, 2 * panel_w + 2 :] = panels[2]

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (h, panel_w):
        return None
    if not np.array_equal(out, main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
