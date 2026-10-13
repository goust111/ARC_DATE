import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


ORTH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ORTH4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _normalize(coords: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_has_hole(mask: np.ndarray) -> bool:
    inv = ~mask
    for comp in _components4(inv):
        if any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in comp):
            continue
        return True
    return False


def _panel_spans(length: int, separators: Sequence[int]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for idx in sorted(set(int(x) for x in separators)):
        if start <= idx - 1:
            spans.append((int(start), int(idx - 1)))
        start = idx + 1
    if start <= length - 1:
        spans.append((int(start), int(length - 1)))
    return spans


def _extract_panel(panel: np.ndarray, bg: int, sep: int) -> Dict[str, object]:
    assert not np.any(panel == sep), "separator color must not appear inside panels"

    colors = [int(c) for c in np.unique(panel) if int(c) != bg]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    color = int(colors[0])

    mask = panel == color
    objects = _components4(mask)
    assert len(objects) == 1, "each panel must contain exactly one 4-connected foreground object"

    holes = []
    for comp in _components4(panel == bg):
        if any(r == 0 or c == 0 or r == panel.shape[0] - 1 or c == panel.shape[1] - 1 for r, c in comp):
            continue
        holes.append({"coords": tuple(comp), "shape": _normalize(comp)})

    return {
        "color": color,
        "shape": _normalize(objects[0]),
        "holes": holes,
    }


def _collect_actions(
    panels: Dict[Tuple[int, int], Dict[str, object]],
    row_count: int,
    col_count: int,
) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[Tuple[int, int], int, Tuple[int, int]]]]:
    actions: Dict[Tuple[int, int], int] = {}
    matched_holes: List[Tuple[Tuple[int, int], int, Tuple[int, int]]] = []

    def match(donor_idx: Tuple[int, int], recv_idx: Tuple[int, int]) -> None:
        donor = panels[donor_idx]
        recv = panels[recv_idx]
        donor_shape = donor["shape"]
        for hole_idx, hole in enumerate(recv["holes"]):
            if hole["shape"] != donor_shape:
                continue
            matched_holes.append((recv_idx, int(hole_idx), donor_idx))
            r0, c0 = recv["offset"]
            for hr, hc in hole["coords"]:
                key = (int(r0 + hr), int(c0 + hc))
                color = int(donor["color"])
                existing = actions.get(key)
                assert existing in (None, color), "conflicting donors matched the same hole pixel"
                actions[key] = color

    for pr in range(row_count):
        for pc in range(col_count):
            if pc + 1 < col_count:
                match((pr, pc), (pr, pc + 1))
                match((pr, pc + 1), (pr, pc))
            if pr + 1 < row_count:
                match((pr, pc), (pr + 1, pc))
                match((pr + 1, pc), (pr, pc))

    return actions, matched_holes


def _analyze_grid(grid: np.ndarray) -> Tuple[int, int, List[Tuple[int, int]], List[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, object]], Dict[Tuple[int, int], int], List[Tuple[Tuple[int, int], int, Tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)

    candidates = []
    for sep in [int(c) for c in np.unique(g) if int(c) != bg]:
        sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
        sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
        if not sep_rows and not sep_cols:
            continue

        row_spans = _panel_spans(h, sep_rows)
        col_spans = _panel_spans(w, sep_cols)
        if not row_spans or not col_spans:
            continue
        if len(row_spans) * len(col_spans) < 2:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        panels: Dict[Tuple[int, int], Dict[str, object]] = {}
        ok = True
        for pr, (r0, r1) in enumerate(row_spans):
            for pc, (c0, c1) in enumerate(col_spans):
                panel = g[r0 : r1 + 1, c0 : c1 + 1]
                try:
                    info = _extract_panel(panel, bg=bg, sep=sep)
                except AssertionError:
                    ok = False
                    break
                info["offset"] = (int(r0), int(c0))
                panels[(pr, pc)] = info
            if not ok:
                break
        if not ok:
            continue

        actions, matched_holes = _collect_actions(panels, len(row_spans), len(col_spans))
        score = (
            1 if matched_holes else 0,
            len(row_spans) * len(col_spans),
            len(sep_rows) + len(sep_cols),
        )
        candidates.append((score, sep, row_spans, col_spans, panels, actions, matched_holes))

    assert candidates, "failed to identify a valid separator layout"
    _, sep, row_spans, col_spans, panels, actions, matched_holes = max(candidates, key=lambda item: item[0])
    return bg, sep, row_spans, col_spans, panels, actions, matched_holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each panel object as a stamp and fill matching enclosed holes in adjacent panels.
    """
    g = np.asarray(grid, dtype=int)
    bg, _sep, _rows, _cols, _panels, actions, _matched_holes = _analyze_grid(g)

    out = g.copy()
    for (r, c), color in actions.items():
        assert int(out[r, c]) == bg, "only background hole pixels may be filled"
        out[r, c] = int(color)
    return out


def _rand_polyomino(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        box_h = rng.randint(1, max_h)
        box_w = rng.randint(1, max_w)
        area = box_h * box_w
        if area < 3:
            continue
        target = rng.randint(3, min(area, max(3, area - 1)))
        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        frontier = set()

        def add_frontier(cell: Tuple[int, int]) -> None:
            cr, cc = cell
            for dr, dc in ORTH4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(start)
        while len(cells) < target and frontier:
            pick = rng.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            add_frontier(pick)
        if len(cells) != target:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if _mask_has_hole(mask):
            continue

        k = rng.randrange(4)
        mask = np.rot90(mask, k)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = _crop_mask(mask)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w and not _mask_has_hole(mask):
            return mask
    raise AssertionError("failed to sample a valid donor shape")


def _place_mask(panel: np.ndarray, mask: np.ndarray, color: int, rng: random.Random) -> None:
    ph, pw = panel.shape
    mh, mw = mask.shape
    assert mh <= ph and mw <= pw

    row_choices = list(range(ph - mh + 1))
    col_choices = list(range(pw - mw + 1))
    if ph - mh >= 2:
        row_choices = list(range(1, ph - mh))
    if pw - mw >= 2:
        col_choices = list(range(1, pw - mw))
    top = rng.choice(row_choices)
    left = rng.choice(col_choices)
    panel[top : top + mh, left : left + mw][mask] = int(color)


def _build_shell_with_hole(hole_mask: np.ndarray, panel_h: int, panel_w: int, rng: random.Random) -> np.ndarray:
    hh, hw = hole_mask.shape
    outer_h_min = hh + 2
    outer_w_min = hw + 2
    assert outer_h_min <= panel_h and outer_w_min <= panel_w

    outer_h = rng.randint(outer_h_min, min(panel_h, hh + 4))
    outer_w = rng.randint(outer_w_min, min(panel_w, hw + 4))
    shell = np.ones((outer_h, outer_w), dtype=bool)

    hole_top = rng.randint(1, outer_h - hh - 1)
    hole_left = rng.randint(1, outer_w - hw - 1)
    view = shell[hole_top : hole_top + hh, hole_left : hole_left + hw]
    view[hole_mask] = False
    return shell


def _shape_signature(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return _normalize(coords)


def _make_distinct_shape(
    rng: random.Random,
    panel_h: int,
    panel_w: int,
    forbidden: Sequence[Tuple[Tuple[int, int], ...]],
) -> np.ndarray:
    max_h = min(5, panel_h - 2)
    max_w = min(5, panel_w - 2)
    assert max_h >= 1 and max_w >= 1
    for _ in range(200):
        shape = _rand_polyomino(rng, max_h=max_h, max_w=max_w)
        sig = _shape_signature(shape)
        if sig not in forbidden:
            return shape
    raise AssertionError("failed to find a distinct filler shape")


def _render_grid(
    layout: Tuple[int, int],
    panel_h: int,
    panel_w: int,
    bg: int,
    sep: int,
    panel_masks: Dict[Tuple[int, int], np.ndarray],
    panel_colors: Dict[Tuple[int, int], int],
    rng: random.Random,
) -> np.ndarray:
    rows, cols = layout
    grid_h = rows * panel_h + (rows - 1)
    grid_w = cols * panel_w + (cols - 1)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for r in range(1, rows):
        grid[r * panel_h + (r - 1), :] = sep
    for c in range(1, cols):
        grid[:, c * panel_w + (c - 1)] = sep

    for pr in range(rows):
        for pc in range(cols):
            r0 = pr * (panel_h + 1)
            c0 = pc * (panel_w + 1)
            panel = np.full((panel_h, panel_w), bg, dtype=int)
            _place_mask(panel, panel_masks[(pr, pc)], panel_colors[(pr, pc)], rng)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    return grid


def _adjacent_edges(rows: int, cols: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
    return edges


def _gen_attempt(rng: random.Random) -> np.ndarray:
    layout = rng.choice([(1, 2), (2, 1), (2, 2)])
    rows, cols = layout
    panel_h = rng.randint(7, 10)
    panel_w = rng.randint(7, 10)
    panel_positions = [(r, c) for r in range(rows) for c in range(cols)]

    num_panels = rows * cols
    palette = list(range(10))
    bg, sep = rng.sample(palette, 2)
    remaining = [c for c in palette if c not in {bg, sep}]
    panel_color_values = rng.sample(remaining, num_panels)
    panel_colors = {pos: panel_color_values[i] for i, pos in enumerate(panel_positions)}

    edge = rng.choice(_adjacent_edges(rows, cols))
    if rng.random() < 0.5:
        donor_pos, receiver_pos = edge
    else:
        receiver_pos, donor_pos = edge

    donor_shape = _make_distinct_shape(rng, panel_h, panel_w, forbidden=[])
    donor_sig = _shape_signature(donor_shape)

    receiver_shape = _build_shell_with_hole(donor_shape, panel_h, panel_w, rng)

    panel_masks: Dict[Tuple[int, int], np.ndarray] = {
        donor_pos: donor_shape,
        receiver_pos: receiver_shape,
    }
    forbidden = [donor_sig]
    for pos in panel_positions:
        if pos in panel_masks:
            continue
        filler = _make_distinct_shape(rng, panel_h, panel_w, forbidden=forbidden)
        forbidden.append(_shape_signature(filler))
        panel_masks[pos] = filler

    grid = _render_grid(
        layout=layout,
        panel_h=panel_h,
        panel_w=panel_w,
        bg=bg,
        sep=sep,
        panel_masks=panel_masks,
        panel_colors=panel_colors,
        rng=rng,
    )

    out = main(grid)
    if np.array_equal(grid, out):
        raise AssertionError("generator must create at least one fill action")

    _bg, _sep, _row_spans, _col_spans, _panels, actions, matched_holes = _analyze_grid(grid)
    assert len(matched_holes) == 1, "generator must keep the matching relation unambiguous"
    assert len(actions) == int(np.sum(donor_shape)), "filled area must equal the donor shape size"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
