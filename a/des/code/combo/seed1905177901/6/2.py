import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
RNG = np.random.default_rng()

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
PAIR_COUNT_RANGE = (2, 5)
PAIR_ATTEMPTS = 120
GEN_ATTEMPTS = 400
FG_COLORS = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            colors = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                colors.append(int(grid[rr, cc]))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            r0 = int(coords_arr[:, 0].min())
            c0 = int(coords_arr[:, 1].min())
            r1 = int(coords_arr[:, 0].max())
            c1 = int(coords_arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords_arr[:, 0] - r0, coords_arr[:, 1] - c0] = True
            uniq_colors = sorted(set(colors))
            comps.append(
                {
                    "coords": coords_arr,
                    "coord_set": set(coords),
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(len(coords)),
                    "color": uniq_colors[0],
                    "colors": uniq_colors,
                }
            )
    return comps


def _reflect_vertical(coords: set[tuple[int, int]], axis: int) -> set[tuple[int, int]]:
    return {(r, 2 * axis - c) for r, c in coords}


def _reflect_horizontal(coords: set[tuple[int, int]], axis: int) -> set[tuple[int, int]]:
    return {(2 * axis - r, c) for r, c in coords}


def _pair_relation(a: dict, b: dict) -> dict | None:
    a_r0, a_c0, a_r1, a_c1 = a["bbox"]
    b_r0, b_c0, b_r1, b_c1 = b["bbox"]

    left, right = (a, b) if a_c0 <= b_c0 else (b, a)
    l_r0, l_c0, l_r1, l_c1 = left["bbox"]
    r_r0, r_c0, r_r1, r_c1 = right["bbox"]
    if l_c1 < r_c0:
        axis2 = l_c1 + r_c0
        if axis2 % 2 == 0:
            axis = axis2 // 2
            if l_c1 < axis < r_c0 and _reflect_vertical(left["coord_set"], axis) == right["coord_set"]:
                return {
                    "orientation": "vertical",
                    "axis": axis,
                    "pair": (left, right),
                }

    top, bottom = (a, b) if a_r0 <= b_r0 else (b, a)
    t_r0, t_c0, t_r1, t_c1 = top["bbox"]
    b_r0, b_c0, b_r1, b_c1 = bottom["bbox"]
    if t_r1 < b_r0:
        axis2 = t_r1 + b_r0
        if axis2 % 2 == 0:
            axis = axis2 // 2
            if t_r1 < axis < b_r0 and _reflect_horizontal(top["coord_set"], axis) == bottom["coord_set"]:
                return {
                    "orientation": "horizontal",
                    "axis": axis,
                    "pair": (top, bottom),
                }

    return None


def _relation_by_indices(comps: list[dict]) -> dict[tuple[int, int], dict]:
    rels = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            rel = _pair_relation(comps[i], comps[j])
            if rel is not None:
                rels[(i, j)] = rel
    return rels


def _enumerate_matchings(rels: dict[tuple[int, int], dict], n: int, limit: int = 2) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []

    def rec(unmatched: tuple[int, ...], cur: list[tuple[int, int]]) -> None:
        if len(out) >= limit:
            return
        if not unmatched:
            out.append(cur.copy())
            return
        i = unmatched[0]
        for j in unmatched[1:]:
            key = (i, j)
            if key not in rels:
                continue
            nxt = tuple(u for u in unmatched[1:] if u != j)
            cur.append(key)
            rec(nxt, cur)
            cur.pop()

    rec(tuple(range(n)), [])
    return out


def _solve_pairs(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected at least one foreground component"
    assert len(comps) % 2 == 0, "foreground object count must be even"
    assert all(len(comp["colors"]) == 1 for comp in comps), "objects must be monochrome"

    rels = _relation_by_indices(comps)
    matchings = _enumerate_matchings(rels, len(comps), limit=2)
    assert len(matchings) == 1, "expected a unique symmetry pairing"
    chosen = [rels[key] for key in matchings[0]]
    return bg, comps, chosen


def _pair_color(first: dict, second: dict) -> int:
    if first["area"] > second["area"]:
        return int(first["color"])
    if second["area"] > first["area"]:
        return int(second["color"])
    first_key = (first["bbox"][0], first["bbox"][1])
    second_key = (second["bbox"][0], second["bbox"][1])
    return int(first["color"] if first_key <= second_key else second["color"])


def _projected_cells(rel: dict) -> list[tuple[int, int]]:
    first, second = rel["pair"]
    union = first["coord_set"] | second["coord_set"]
    if rel["orientation"] == "vertical":
        rows = sorted({r for r, _ in union})
        return [(r, int(rel["axis"])) for r in rows]
    cols = sorted({c for _, c in union})
    return [(int(rel["axis"]), c) for c in cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect symmetric mirror pairs of objects and collapse each pair onto its symmetry axis.

    For every pair of monochrome 4-connected objects that are exact reflections across
    a horizontal or vertical integer grid axis, remove the originals and paint the
    projection of their union onto that axis. The projected object keeps the color of
    the larger source object; ties use the object that is earlier by top-left position.
    """
    g = np.asarray(grid, dtype=int)
    bg, _comps, pairs = _solve_pairs(g)

    out = np.full_like(g, int(bg))
    for rel in pairs:
        color = _pair_color(*rel["pair"])
        for r, c in _projected_cells(rel):
            assert int(out[r, c]) in {int(bg), color}, "projected pairs must not conflict"
            out[r, c] = color
    return out


def _segments_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    left = int(RNG.integers(0, width))
    right = int(RNG.integers(left, min(width - 1, left + 1) + 1))
    mask[0, left : right + 1] = True
    prev_l, prev_r = left, right
    for r in range(1, height):
        anchor = int(RNG.integers(prev_l, prev_r + 1))
        l_lo = max(0, anchor - 2)
        l_hi = min(anchor, width - 1)
        left = int(RNG.integers(l_lo, l_hi + 1))
        r_lo = max(anchor, left)
        r_hi = min(width - 1, anchor + 2)
        right = int(RNG.integers(r_lo, r_hi + 1))
        mask[r, left : right + 1] = True
        prev_l, prev_r = left, right
    return mask


def _generate_mask(orientation: str) -> np.ndarray:
    for _ in range(80):
        if orientation == "vertical":
            height = int(RNG.integers(3, 7))
            width = int(RNG.integers(2, 6))
            mask = _segments_mask(height, width)
            if np.all(mask.any(axis=1)) and not np.array_equal(mask, np.fliplr(mask)):
                return mask
        else:
            width = int(RNG.integers(3, 7))
            height = int(RNG.integers(2, 6))
            mask = _segments_mask(width, height).T
            if np.all(mask.any(axis=0)) and not np.array_equal(mask, np.flipud(mask)):
                return mask
    raise RuntimeError("failed to sample a chiral connected mask")


def _mask_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    rows, cols = np.where(mask)
    return [(int(r0 + r), int(c0 + c)) for r, c in zip(rows, cols)]


def _too_close(candidate: list[tuple[int, int]], existing: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for dr, dc in DIR8:
            if (r + dr, c + dc) in existing:
                return True
        if (r, c) in existing:
            return True
    return False


def _projection_too_close(candidate: list[tuple[int, int]], existing: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in existing:
            return True
        for dr, dc in DIR8:
            if (r + dr, c + dc) in existing:
                return True
    return False


def _pair_projection(mask: np.ndarray, orientation: str, axis: int, r0: int, c0: int) -> list[tuple[int, int]]:
    if orientation == "vertical":
        rows = np.flatnonzero(mask.any(axis=1))
        return [(int(r0 + r), int(axis)) for r in rows]
    cols = np.flatnonzero(mask.any(axis=0))
    return [(int(axis), int(c0 + c)) for c in cols]


def _choose_orientations(pair_count: int) -> list[str]:
    if pair_count == 2:
        if RNG.random() < 0.7:
            return ["vertical", "horizontal"] if RNG.random() < 0.5 else ["horizontal", "vertical"]
        return [str(RNG.choice(["vertical", "horizontal"])) for _ in range(pair_count)]

    orient = [str(RNG.choice(["vertical", "horizontal"])) for _ in range(pair_count)]
    if "vertical" not in orient:
        orient[int(RNG.integers(0, pair_count))] = "vertical"
    if "horizontal" not in orient:
        orient[int(RNG.integers(0, pair_count))] = "horizontal"
    return orient


def _sample_pair_spec(h: int, w: int, orientation: str, shape_bank: dict[str, list[np.ndarray]]) -> dict | None:
    reuse_ok = shape_bank[orientation] and RNG.random() < 0.35
    mask = shape_bank[orientation][int(RNG.integers(0, len(shape_bank[orientation])))] if reuse_ok else _generate_mask(orientation)
    if not reuse_ok:
        shape_bank[orientation].append(mask.copy())

    mh, mw = mask.shape
    gap = int(RNG.integers(1, 4))
    color_a, color_b = map(int, RNG.choice(FG_COLORS, size=2, replace=False))

    if orientation == "vertical":
        axis_lo = mw + gap
        axis_hi = w - mw - gap - 1
        if axis_lo > axis_hi:
            return None
        axis = int(RNG.integers(axis_lo, axis_hi + 1))
        left_c0 = axis - gap - (mw - 1)
        right_c0 = axis + gap
        r0 = int(RNG.integers(0, h - mh + 1))
        left_mask = mask
        right_mask = np.fliplr(mask)
        left_cells = _mask_cells(left_mask, r0, left_c0)
        right_cells = _mask_cells(right_mask, r0, right_c0)
        proj = _pair_projection(mask, orientation, axis, r0, left_c0)
        return {
            "orientation": orientation,
            "axis": axis,
            "first_cells": left_cells,
            "second_cells": right_cells,
            "first_color": color_a,
            "second_color": color_b,
            "projection": proj,
        }

    axis_lo = mh + gap
    axis_hi = h - mh - gap - 1
    if axis_lo > axis_hi:
        return None
    axis = int(RNG.integers(axis_lo, axis_hi + 1))
    top_r0 = axis - gap - (mh - 1)
    bot_r0 = axis + gap
    c0 = int(RNG.integers(0, w - mw + 1))
    top_mask = mask
    bot_mask = np.flipud(mask)
    top_cells = _mask_cells(top_mask, top_r0, c0)
    bot_cells = _mask_cells(bot_mask, bot_r0, c0)
    proj = _pair_projection(mask, orientation, axis, top_r0, c0)
    return {
        "orientation": orientation,
        "axis": axis,
        "first_cells": top_cells,
        "second_cells": bot_cells,
        "first_color": color_a,
        "second_color": color_b,
        "projection": proj,
    }


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(*GRID_H_RANGE))
    w = int(RNG.integers(*GRID_W_RANGE))
    pair_count = int(RNG.integers(*PAIR_COUNT_RANGE))
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    projected: set[tuple[int, int]] = set()
    shape_bank = {"vertical": [], "horizontal": []}

    for orientation in _choose_orientations(pair_count):
        placed = False
        for _ in range(PAIR_ATTEMPTS):
            spec = _sample_pair_spec(h, w, orientation, shape_bank)
            if spec is None:
                continue
            all_cells = spec["first_cells"] + spec["second_cells"]
            if _too_close(all_cells, occupied):
                continue
            if _projection_too_close(spec["projection"], projected):
                continue

            _paint_cells(grid, spec["first_cells"], spec["first_color"])
            _paint_cells(grid, spec["second_cells"], spec["second_color"])
            occupied.update(all_cells)
            projected.update(spec["projection"])
            placed = True
            break
        if not placed:
            return None

    try:
        bg, comps, pairs = _solve_pairs(grid)
        if bg != 0:
            return None
        if len(comps) != 2 * pair_count or len(pairs) != pair_count:
            return None
        out = main(grid)
        if np.count_nonzero(out) == 0:
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-pair sample")
