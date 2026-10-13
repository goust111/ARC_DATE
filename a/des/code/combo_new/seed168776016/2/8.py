from __future__ import annotations

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(
    grid: np.ndarray, allowed: np.ndarray, forbidden: set[int]
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not allowed[r, c]:
                continue
            color = int(grid[r, c])
            if color in forbidden:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not allowed[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return comps


def _find_cross_and_legend(
    grid: np.ndarray, bg: int
) -> tuple[int, int, int, dict[str, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, dict[str, int]]] = []

    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            legend_vals = [int(x) for x in block.ravel()]
            if bg in legend_vals or len(set(legend_vals)) != 4:
                continue

            row_rest = np.concatenate((grid[r, :c], grid[r, c + 2 :]))
            col_rest = np.concatenate((grid[:r, c], grid[r + 2 :, c]))
            if row_rest.size == 0 or col_rest.size == 0:
                continue
            if not np.all(row_rest == row_rest[0]):
                continue

            line = int(row_rest[0])
            if line == bg or line in legend_vals:
                continue
            if not np.all(col_rest == line):
                continue
            if int(np.count_nonzero(grid[r, :] != line)) != 2:
                continue
            if int(np.count_nonzero(grid[:, c] != line)) != 2:
                continue

            legend = {
                "tl": int(grid[r, c]),
                "tr": int(grid[r, c + 1]),
                "bl": int(grid[r + 1, c]),
                "br": int(grid[r + 1, c + 1]),
            }
            candidates.append((r, c, line, legend))

    assert len(candidates) == 1, f"expected exactly one cross legend, found {len(candidates)}"
    return candidates[0]


def _draw_border(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    immutable: np.ndarray,
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox

    for cc in range(c0 - 1, c1 + 2):
        if 0 <= cc < w and 0 <= r0 - 1 < h and not immutable[r0 - 1, cc]:
            out[r0 - 1, cc] = color
        if 0 <= cc < w and 0 <= r1 + 1 < h and not immutable[r1 + 1, cc]:
            out[r1 + 1, cc] = color

    for rr in range(r0 - 1, r1 + 2):
        if 0 <= rr < h and 0 <= c0 - 1 < w and not immutable[rr, c0 - 1]:
            out[rr, c0 - 1] = color
        if 0 <= rr < h and 0 <= c1 + 1 < w and not immutable[rr, c1 + 1]:
            out[rr, c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Frame each quadrant object using the legend color embedded at the central cross.

    Input structure:
    - The background is a dominant color.
    - One full horizontal line and one full vertical line split the grid into four quadrants.
    - A 2x2 color block is embedded at the cross, with its upper-left, upper-right, lower-left, and lower-right pixels serving as quadrant legend colors.
    - Each quadrant contains separated 4-connected monochrome objects that do not cross the split lines.

    Rule:
    - Detect the central 2x2 legend and map its four pixels to the top-left, top-right, bottom-left, and bottom-right quadrants.
    - Find every colored object inside each quadrant and compute its tight bounding box.
    - Draw a one-pixel-thick rectangular border just outside that bounding box using the legend color for the object's quadrant.
    - Keep the original objects unchanged, and never overwrite the split lines or the legend block.

    Key insight:
    - The 2x2 block at the cross is a positional palette: quadrant location chooses the border color, while the object's own geometry only determines the rectangle to outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6

    bg = _most_common_color(g)
    cross_r, cross_c, line, legend = _find_cross_and_legend(g, bg)
    forbidden = {bg, line, *legend.values()}

    immutable = np.zeros_like(g, dtype=bool)
    immutable[cross_r, :] = True
    immutable[:, cross_c] = True
    immutable[cross_r : cross_r + 2, cross_c : cross_c + 2] = True

    quadrants = [
        ("tl", slice(0, cross_r), slice(0, cross_c)),
        ("tr", slice(0, cross_r), slice(cross_c + 1, w)),
        ("bl", slice(cross_r + 1, h), slice(0, cross_c)),
        ("br", slice(cross_r + 1, h), slice(cross_c + 1, w)),
    ]

    out = g.copy()
    object_count = 0
    for name, rs, cs in quadrants:
        allowed = np.zeros_like(g, dtype=bool)
        allowed[rs, cs] = True
        allowed[cross_r : cross_r + 2, cross_c : cross_c + 2] = False
        comps = _find_components(g, allowed, forbidden)
        for comp in comps:
            object_count += 1
            _draw_border(out, comp["bbox"], legend[name], immutable)

    assert object_count > 0, "expected at least one object"
    return out


# --------------- generator ---------------

_BG = 8
_LINE = 5
_COLOR_POOL = tuple(c for c in range(10) if c not in (_BG, _LINE))

_GRID_MIN = 16
_GRID_MAX = 30
_QUADRANT_MIN = 5
_GEN_RETRIES = 300
_PLACE_RETRIES = 80


def _random_blob(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in _N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _shape_mask(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray:
    modes = ["rect", "line", "blob"]
    if max_h >= 2 and max_w >= 2:
        modes.append("ell")
    mode = modes[int(rng.integers(0, len(modes)))]

    if mode == "rect":
        h = int(rng.integers(1, min(max_h, 4) + 1))
        w = int(rng.integers(1, min(max_w, 4) + 1))
        return np.ones((h, w), dtype=bool)

    if mode == "line":
        horizontal_ok = max_w >= 2
        vertical_ok = max_h >= 2
        if not horizontal_ok and not vertical_ok:
            return np.ones((1, 1), dtype=bool)
        if horizontal_ok and vertical_ok:
            horizontal = bool(rng.integers(0, 2))
        else:
            horizontal = horizontal_ok
        if horizontal:
            length = int(rng.integers(2, min(max_w, 5) + 1))
            return np.ones((1, length), dtype=bool)
        length = int(rng.integers(2, min(max_h, 5) + 1))
        return np.ones((length, 1), dtype=bool)

    if mode == "ell":
        h = int(rng.integers(2, min(max_h, 4) + 1))
        w = int(rng.integers(2, min(max_w, 4) + 1))
        mask = np.zeros((h, w), dtype=bool)
        orientation = int(rng.integers(0, 4))
        if orientation == 0:
            mask[0, :] = True
            mask[:, 0] = True
        elif orientation == 1:
            mask[0, :] = True
            mask[:, -1] = True
        elif orientation == 2:
            mask[-1, :] = True
            mask[:, 0] = True
        else:
            mask[-1, :] = True
            mask[:, -1] = True
        return mask

    h = int(rng.integers(2, min(max_h, 4) + 1))
    w = int(rng.integers(2, min(max_w, 4) + 1))
    max_area = min(h * w, 7)
    min_area = min(2, max_area)
    area = int(rng.integers(min_area, max_area + 1))
    return _random_blob(rng, h, w, area)


def _reserve_bbox(
    reserved: np.ndarray,
    immutable: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> None:
    rr0 = r0 - 1
    cc0 = c0 - 1
    rr1 = r0 + h
    cc1 = c0 + w
    for rr in range(rr0, rr1 + 1):
        for cc in range(cc0, cc1 + 1):
            if not immutable[rr, cc]:
                reserved[rr, cc] = True


def _can_place(
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
    immutable: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
) -> bool:
    h, w = mask.shape
    coords = np.argwhere(mask)
    for dr, dc in coords:
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        if blocked[rr, cc] or int(grid[rr, cc]) != _BG:
            return False

    rr0 = r0 - 1
    cc0 = c0 - 1
    rr1 = r0 + h
    cc1 = c0 + w
    if rr0 < 0 or cc0 < 0 or rr1 >= grid.shape[0] or cc1 >= grid.shape[1]:
        return False
    region_reserved = reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]
    region_immutable = immutable[rr0 : rr1 + 1, cc0 : cc1 + 1]
    return not bool(np.any(region_reserved & ~region_immutable))


def _paint_object(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    coords = np.argwhere(mask)
    for dr, dc in coords:
        grid[r0 + int(dr), c0 + int(dc)] = color


def _quadrant_specs(h: int, w: int, r: int, c: int) -> dict[str, dict[str, int | bool]]:
    return {
        "tl": {"r0": 1, "r1": r - 1, "c0": 1, "c1": c - 1, "top": True, "left": True},
        "tr": {"r0": 1, "r1": r - 1, "c0": c + 1, "c1": w - 2, "top": True, "left": False},
        "bl": {"r0": r + 1, "r1": h - 2, "c0": 1, "c1": c - 1, "top": False, "left": True},
        "br": {"r0": r + 1, "r1": h - 2, "c0": c + 1, "c1": w - 2, "top": False, "left": False},
    }


def _touch_choice(rng: np.random.Generator) -> tuple[bool, bool]:
    choices = ((False, False), (True, False), (False, True), (True, True))
    return choices[int(rng.integers(0, len(choices)))]


def _position_ranges(
    spec: dict[str, int | bool],
    h: int,
    w: int,
    cross_r: int,
    cross_c: int,
    touch_h: bool,
    touch_v: bool,
) -> tuple[list[int], list[int]]:
    row_lo = int(spec["r0"])
    row_hi = int(spec["r1"]) - h + 1
    col_lo = int(spec["c0"])
    col_hi = int(spec["c1"]) - w + 1
    if row_lo > row_hi or col_lo > col_hi:
        return [], []

    if touch_h:
        if bool(spec["top"]):
            row_lo = row_hi = cross_r - h
        else:
            row_lo = row_hi = cross_r + 1
    if touch_v:
        if bool(spec["left"]):
            col_lo = col_hi = cross_c - w
        else:
            col_lo = col_hi = cross_c + 1
    if row_lo > row_hi or col_lo > col_hi:
        return [], []

    rows = list(range(row_lo, row_hi + 1))
    cols = list(range(col_lo, col_hi + 1))
    return rows, cols


def _object_count(rng: np.random.Generator, qh: int, qw: int) -> int:
    area = qh * qw
    if area >= 90 and float(rng.random()) < 0.25:
        return 3
    return 1 if float(rng.random()) < 0.5 else 2


def _place_one_object(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved: np.ndarray,
    immutable: np.ndarray,
    spec: dict[str, int | bool],
    cross_r: int,
    cross_c: int,
    color: int,
) -> bool:
    qh = int(spec["r1"]) - int(spec["r0"]) + 1
    qw = int(spec["c1"]) - int(spec["c0"]) + 1
    if qh < 1 or qw < 1:
        return False

    for _ in range(_PLACE_RETRIES):
        mask = _shape_mask(rng, min(5, qh), min(5, qw))
        mh, mw = mask.shape
        touch_h, touch_v = _touch_choice(rng)
        rows, cols = _position_ranges(spec, mh, mw, cross_r, cross_c, touch_h, touch_v)
        if not rows or not cols:
            continue
        rng.shuffle(rows)
        rng.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                if not _can_place(grid, blocked, reserved, immutable, r0, c0, mask):
                    continue
                _paint_object(grid, r0, c0, mask, color)
                _reserve_bbox(reserved, immutable, r0, c0, mh, mw)
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    w = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    cross_r = int(rng.integers(_QUADRANT_MIN, h - _QUADRANT_MIN))
    cross_c = int(rng.integers(_QUADRANT_MIN, w - _QUADRANT_MIN))

    grid = np.full((h, w), _BG, dtype=int)
    grid[cross_r, :] = _LINE
    grid[:, cross_c] = _LINE

    legend_colors = rng.choice(np.asarray(_COLOR_POOL), size=4, replace=False)
    grid[cross_r : cross_r + 2, cross_c : cross_c + 2] = legend_colors.reshape(2, 2)

    immutable = np.zeros((h, w), dtype=bool)
    immutable[cross_r, :] = True
    immutable[:, cross_c] = True
    immutable[cross_r : cross_r + 2, cross_c : cross_c + 2] = True

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    blocked[cross_r, :] = True
    blocked[:, cross_c] = True
    blocked[cross_r : cross_r + 2, cross_c : cross_c + 2] = True

    reserved = np.zeros((h, w), dtype=bool)
    specs = _quadrant_specs(h, w, cross_r, cross_c)
    object_palette = [c for c in _COLOR_POOL if c not in set(map(int, legend_colors))]
    total_objects = 0

    for name, spec in specs.items():
        qh = int(spec["r1"]) - int(spec["r0"]) + 1
        qw = int(spec["c1"]) - int(spec["c0"]) + 1
        if qh < 2 or qw < 2:
            return None
        n_obj = _object_count(rng, qh, qw)
        for _ in range(n_obj):
            color = int(object_palette[int(rng.integers(0, len(object_palette)))])
            if not _place_one_object(rng, grid, blocked, reserved, immutable, spec, cross_r, cross_c, color):
                return None
            total_objects += 1

    if total_objects < 4:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
