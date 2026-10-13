import numpy as np


BLACK = 0
MAGENTA = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _shape_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _is_hint_component(comp: dict, height: int, width: int) -> bool:
    if int(comp["area"]) != 2:
        return False
    coords = comp["coords"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    r_span = int(rows.max() - rows.min() + 1)
    c_span = int(cols.max() - cols.min() + 1)
    if (r_span, c_span) not in ((1, 2), (2, 1)):
        return False
    touches_bottom = int(rows.max()) == height - 1
    touches_left = int(cols.min()) == 0
    touches_right = int(cols.max()) == width - 1
    return touches_bottom and (touches_left or touches_right)


def _separator_columns(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    cols = [c for c in range(w) if bool(np.all(grid[:, c] == MAGENTA))]
    assert cols, "expected at least one full-height magenta separator"
    assert 0 not in cols and (w - 1) not in cols, "separator cannot be on the border"
    assert len(cols) == len(set(cols))
    for c in cols:
        assert bool(np.all(grid[:, c] == MAGENTA))
    return cols


def _panel_bounds(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_cols = _separator_columns(grid)
    bounds: list[tuple[int, int]] = []
    start = 0
    for sep in sep_cols:
        assert start < sep, "panels must have positive width"
        bounds.append((start, sep))
        start = sep + 1
    assert start < grid.shape[1], "final panel must have positive width"
    bounds.append((start, grid.shape[1]))
    assert len(bounds) >= 2 and len(bounds) % 2 == 0, "panels must form adjacent pairs"
    return bounds


def _parse_panel(panel: np.ndarray) -> dict:
    height, width = panel.shape
    comps = _connected_components(panel, bg=BLACK)
    assert len(comps) == 2, "expected one shape and one cue block per panel"
    hint_candidates = [comp for comp in comps if _is_hint_component(comp, height, width)]
    shape_candidates = [comp for comp in comps if comp not in hint_candidates]
    assert len(hint_candidates) == 1, "expected exactly one bottom-corner cue block"
    assert len(shape_candidates) == 1, "expected exactly one shape"

    hint = hint_candidates[0]
    shape = shape_candidates[0]
    assert int(shape["area"]) >= 4, "shape must have area at least four"
    assert int(shape["color"]) != int(hint["color"]), "shape and cue colors must differ"
    bbox, mask = _shape_mask(shape["coords"])

    return {
        "hint_color": int(hint["color"]),
        "shape_color": int(shape["color"]),
        "shape_coords": shape["coords"],
        "shape_bbox": bbox,
        "shape_mask": mask,
    }


def _draw_mask(
    out: np.ndarray,
    mask: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    panel_start: int,
    panel_stop: int,
    color: int,
) -> None:
    mh, mw = mask.shape
    assert anchor_r >= 0 and anchor_c >= 0
    assert anchor_r + mh <= out.shape[0]
    assert anchor_c + mw <= panel_stop - panel_start
    rr, cc = np.nonzero(mask)
    target_rows = anchor_r + rr
    target_cols = panel_start + anchor_c + cc
    assert bool(np.all(out[target_rows, target_cols] == BLACK)), "copied shape must land on empty cells"
    out[target_rows, target_cols] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each panel pair's shape into its neighbor while recoloring it with the destination panel's cue color.

    Input structure:
    - The background is black and one or more full-height magenta columns split the grid into an even number of vertical panels.
    - Each panel contains exactly two non-black objects: one monochrome connected shape with area at least four, and one two-pixel monochrome cue block anchored in a bottom corner.
    - The cue block and the shape use different colors, and panels are processed as adjacent left-to-right pairs.

    Rule:
    - Keep every magenta separator column unchanged.
    - In each panel pair, erase both original shapes but leave both cue blocks untouched.
    - Copy the left shape's bitmap into the right panel at the right shape's original bbox top-left, and copy the right shape into the left panel at the left shape's original bbox top-left.
    - Recolor each copied shape with the destination panel's own cue-block color rather than the source shape's color.

    Key insight:
    - Shape geometry and shape color come from different places: the neighbor supplies the bitmap, while the destination panel's corner cue supplies the paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bounds = _panel_bounds(g)

    panels: list[dict] = []
    for start, stop in bounds:
        info = _parse_panel(g[:, start:stop])
        info["panel_start"] = int(start)
        info["panel_stop"] = int(stop)
        panels.append(info)

    out = g.copy()
    for info in panels:
        coords = info["shape_coords"]
        out[coords[:, 0], info["panel_start"] + coords[:, 1]] = BLACK

    for i in range(0, len(panels), 2):
        left = panels[i]
        right = panels[i + 1]

        left_r0, left_c0, _left_r1, _left_c1 = left["shape_bbox"]
        right_r0, right_c0, _right_r1, _right_c1 = right["shape_bbox"]

        _draw_mask(
            out,
            right["shape_mask"],
            left_r0,
            left_c0,
            left["panel_start"],
            left["panel_stop"],
            left["hint_color"],
        )
        _draw_mask(
            out,
            left["shape_mask"],
            right_r0,
            right_c0,
            right["panel_start"],
            right["panel_stop"],
            right["hint_color"],
        )

    return out


def _cue_cells(height: int, width: int, side: str, orientation: str) -> list[tuple[int, int]]:
    if side == "left" and orientation == "horizontal":
        return [(height - 1, 0), (height - 1, 1)]
    if side == "right" and orientation == "horizontal":
        return [(height - 1, width - 2), (height - 1, width - 1)]
    if side == "left" and orientation == "vertical":
        return [(height - 2, 0), (height - 1, 0)]
    assert side == "right" and orientation == "vertical"
    return [(height - 2, width - 1), (height - 1, width - 1)]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_grown_mask(max_h: int, max_w: int) -> np.ndarray | None:
    h = int(RNG.integers(2, max_h + 1))
    w = int(RNG.integers(2, max_w + 1))
    target_area = int(RNG.integers(4, h * w + 1))
    mask = np.zeros((h, w), dtype=bool)
    seed = (int(RNG.integers(h)), int(RNG.integers(w)))
    mask[seed] = True
    cells = {seed}
    frontier = set()
    for nr, nc in _neighbors4(*seed):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < target_area and frontier:
        choice = tuple(frontier)[int(RNG.integers(len(frontier)))]
        frontier.remove(choice)
        cells.add(choice)
        mask[choice] = True
        for nr, nc in _neighbors4(*choice):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if int(mask.sum()) < 4:
        return None
    return _trim_mask(mask)


def _shape_rectangle(max_h: int, max_w: int) -> np.ndarray:
    h = int(RNG.integers(2, max_h + 1))
    w = int(RNG.integers(2, max_w + 1))
    return np.ones((h, w), dtype=bool)


def _shape_l(max_h: int, max_w: int) -> np.ndarray | None:
    h = int(RNG.integers(2, max_h + 1))
    w = int(RNG.integers(2, max_w + 1))
    if h + w - 1 < 4:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    rotation = int(RNG.integers(4))
    return np.rot90(mask, rotation)


def _shape_t(max_h: int, max_w: int) -> np.ndarray | None:
    vertical = bool(RNG.integers(2))
    if vertical:
        if max_h < 3 or max_w < 2:
            return None
        h = int(RNG.integers(3, max_h + 1))
        w = int(RNG.integers(2, max_w + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, w // 2] = True
        mask[0, :] = True
    else:
        if max_h < 2 or max_w < 3:
            return None
        h = int(RNG.integers(2, max_h + 1))
        w = int(RNG.integers(3, max_w + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2, :] = True
        mask[:, w // 2] = True
    return mask


def _shape_plus(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 3 or max_w < 3:
        return None
    h_choices = [v for v in range(3, max_h + 1) if v % 2 == 1]
    w_choices = [v for v in range(3, max_w + 1) if v % 2 == 1]
    if not h_choices or not w_choices:
        return None
    h = int(RNG.choice(h_choices))
    w = int(RNG.choice(w_choices))
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _shape_stair(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    h = int(RNG.integers(2, max_h + 1))
    w = int(RNG.integers(2, max_w + 1))
    steps = min(h, w)
    mask = np.zeros((h, w), dtype=bool)
    for i in range(steps):
        mask[i, : i + 1] = True
    if int(mask.sum()) < 4:
        return None
    return _trim_mask(mask)


def _sample_shape(max_h: int, max_w: int, forbidden_areas: set[int]) -> np.ndarray | None:
    generators = (
        _shape_rectangle,
        _shape_l,
        _shape_t,
        _shape_plus,
        _shape_stair,
        _random_grown_mask,
    )
    for _ in range(100):
        maker = generators[int(RNG.integers(len(generators)))]
        mask = maker(max_h, max_w)
        if mask is None:
            continue
        mask = _trim_mask(mask)
        mh, mw = mask.shape
        area = int(mask.sum())
        if area < 4 or area in forbidden_areas:
            continue
        if mh <= max_h and mw <= max_w:
            return mask
    return None


def _pair_shapes(
    left_room: tuple[int, int],
    right_room: tuple[int, int],
    forbidden_areas: set[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = min(left_room[0], right_room[0])
    max_w = min(left_room[1], right_room[1])
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(100):
        left = _sample_shape(max_h, max_w, forbidden_areas)
        if left is None:
            continue
        left_area = int(left.sum())
        right = _sample_shape(max_h, max_w, forbidden_areas | {left_area})
        if right is None:
            continue
        if left.shape != right.shape or left_area != int(right.sum()):
            return left, right
    return None


def _panel_anchor(height: int, width: int) -> tuple[int, int]:
    shape_bottom = height - 4
    anchor_r = int(RNG.integers(1, shape_bottom))
    anchor_c = int(RNG.integers(1, width - 2))
    return anchor_r, anchor_c


def _room_from_anchor(height: int, width: int, anchor: tuple[int, int]) -> tuple[int, int]:
    anchor_r, anchor_c = anchor
    shape_bottom = height - 4
    max_h = shape_bottom - anchor_r + 1
    max_w = (width - 2) - anchor_c + 1
    return max_h, max_w


def _paint_panel_shape(grid: np.ndarray, panel_start: int, anchor: tuple[int, int], mask: np.ndarray, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[anchor[0] + rr, panel_start + anchor[1] + cc] = int(color)


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        rr, cc = np.nonzero(spec["shape_mask"])
        out[spec["shape_anchor"][0] + rr, spec["panel_start"] + spec["shape_anchor"][1] + cc] = BLACK

    for i in range(0, len(specs), 2):
        left = specs[i]
        right = specs[i + 1]
        _draw_mask(
            out,
            right["shape_mask"],
            left["shape_anchor"][0],
            left["shape_anchor"][1],
            left["panel_start"],
            left["panel_stop"],
            left["hint_color"],
        )
        _draw_mask(
            out,
            left["shape_mask"],
            right["shape_anchor"][0],
            right["shape_anchor"][1],
            right["panel_start"],
            right["panel_stop"],
            right["hint_color"],
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_count = 2 if bool(RNG.integers(2)) else 4
    height = int(RNG.integers(9, 17))
    if panel_count == 2:
        widths = [int(RNG.integers(6, 13)) for _ in range(panel_count)]
    else:
        widths = [int(RNG.integers(5, 7)) for _ in range(panel_count)]

    total_width = int(sum(widths) + panel_count - 1)
    if total_width > 30:
        return None

    colors = list(RNG.permutation(PALETTE))
    needed = 2 * panel_count
    shape_colors = colors[:panel_count]
    hint_colors = colors[panel_count:needed]

    grid = np.full((height, total_width), BLACK, dtype=int)
    specs: list[dict] = []
    used_areas: set[int] = set()

    cursor = 0
    panel_starts: list[int] = []
    panel_stops: list[int] = []
    for idx, width in enumerate(widths):
        panel_starts.append(cursor)
        cursor += width
        panel_stops.append(cursor)
        if idx != panel_count - 1:
            grid[:, cursor] = MAGENTA
            cursor += 1

    for pair_idx in range(0, panel_count, 2):
        left_width = widths[pair_idx]
        right_width = widths[pair_idx + 1]
        left_anchor = _panel_anchor(height, left_width)
        right_anchor = _panel_anchor(height, right_width)
        pair = _pair_shapes(
            _room_from_anchor(height, left_width, left_anchor),
            _room_from_anchor(height, right_width, right_anchor),
            used_areas,
        )
        if pair is None:
            return None
        left_mask, right_mask = pair
        used_areas.add(int(left_mask.sum()))
        used_areas.add(int(right_mask.sum()))

        for panel_offset, width, anchor, mask in (
            (pair_idx, left_width, left_anchor, left_mask),
            (pair_idx + 1, right_width, right_anchor, right_mask),
        ):
            hint_side = "left" if bool(RNG.integers(2)) else "right"
            hint_orientation = "horizontal" if bool(RNG.integers(2)) else "vertical"
            hint_cells = _cue_cells(height, width, hint_side, hint_orientation)
            for rr, cc in hint_cells:
                grid[rr, panel_starts[panel_offset] + cc] = int(hint_colors[panel_offset])
            _paint_panel_shape(grid, panel_starts[panel_offset], anchor, mask, shape_colors[panel_offset])
            specs.append(
                {
                    "panel_start": panel_starts[panel_offset],
                    "panel_stop": panel_stops[panel_offset],
                    "shape_anchor": anchor,
                    "shape_mask": mask,
                    "shape_color": int(shape_colors[panel_offset]),
                    "hint_color": int(hint_colors[panel_offset]),
                }
            )

    expected = _expected_from_specs(grid, specs)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-swap puzzle")
