import numpy as np


_RNG = np.random.default_rng()
_GEN_COUNTER = 0

_BG = 0
_GREY = 5
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_GENERATION_ATTEMPTS = 400

_SOURCE_LINE_SIDES = ("top", "bottom", "left", "right")
_ANCHOR_LINE_SIDES = ("top", "left")
_LENGTH_PAIRS_BY_TURNS = {
    0: ((1, 1), (2, 2), (3, 3)),
    1: ((1, 2), (2, 3)),
    2: ((1, 3), (3, 1)),
    3: ((2, 1), (3, 2)),
}


def _parse_shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_BASE_SHAPES = [
    _parse_shape(["111", "100", "111"]),
    _parse_shape(["111", "101", "110"]),
    _parse_shape(["110", "101", "111"]),
    _parse_shape(["1111", "1001"]),
    _parse_shape(["1001", "1111"]),
    _parse_shape(["111", "101", "101", "111"]),
    _parse_shape(["1111", "1001", "1101", "1111"]),
    _parse_shape(["1111", "1011", "1001", "1111"]),
]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch, (r0, c0, r1, c1)


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    need = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(coords) == need


def _find_workspace_frame(grid: np.ndarray, bg: int) -> dict:
    candidates: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            if _is_rectangular_frame(coords):
                r0, c0, r1, c1 = _bbox(coords)
                candidates.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": (r0, c0, r1, c1),
                        "area": len(coords),
                    }
                )
    assert candidates, "expected a rectangular workspace frame"
    candidates.sort(key=lambda item: (item["area"], item["bbox"][0], item["bbox"][1]))
    return candidates[-1]


def _touches_component(
    obj_coords: list[tuple[int, int]], grey_comp: list[tuple[int, int]], shape: tuple[int, int]
) -> bool:
    h, w = shape
    obj_set = set(obj_coords)
    for r, c in grey_comp:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in obj_set:
                return True
    return False


def _line_length_for_object(
    obj_coords: list[tuple[int, int]],
    grey_components: list[list[tuple[int, int]]],
    shape: tuple[int, int],
) -> int:
    matches = [comp for comp in grey_components if _touches_component(obj_coords, comp, shape)]
    assert len(matches) == 1, "each object must have exactly one adjacent grey segment"
    comp = matches[0]
    rows = {r for r, _c in comp}
    cols = {c for _r, c in comp}
    assert len(rows) == 1 or len(cols) == 1, "grey segment must be straight"
    assert 1 <= len(comp) <= 3, "grey segment length must be 1..3"
    return len(comp)


def _paint_mask(out: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    assert 0 <= r0 <= out.shape[0] - h
    assert 0 <= c0 <= out.shape[1] - w
    window = out[r0 : r0 + h, c0 : c0 + w]
    window[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each upper sprite into the lower workspace anchor of the same color.

    The clockwise rotation count is encoded by the difference between the grey
    source segment length and the grey target segment length: (B - A) mod 4.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _BG if np.any(g == _BG) else _most_common_color(g)

    frame = _find_workspace_frame(g, bg)
    grey = int(frame["color"])
    fr0, fc0, fr1, fc1 = frame["bbox"]
    inner_r0, inner_c0 = fr0 + 1, fc0 + 1
    inner_r1, inner_c1 = fr1 - 1, fc1 - 1

    grey_components = [
        comp
        for comp in _components(g == grey)
        if set(comp) != set(frame["coords"])
    ]

    entries: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, grey):
            continue
        comps = _components(g == color)
        anchor_candidates: list[list[tuple[int, int]]] = []
        stamp_candidates: list[list[tuple[int, int]]] = []
        for comp in comps:
            inside = all(inner_r0 <= r <= inner_r1 and inner_c0 <= c <= inner_c1 for r, c in comp)
            if inside and len(comp) == 1:
                anchor_candidates.append(comp)
            else:
                stamp_candidates.append(comp)
        assert len(anchor_candidates) == 1, "each color must have one anchor"
        assert len(stamp_candidates) == 1, "each color must have one source stamp"

        anchor_coords = anchor_candidates[0]
        stamp_coords = stamp_candidates[0]
        stamp_mask, stamp_bbox = _mask_from_coords(stamp_coords)
        a_len = _line_length_for_object(stamp_coords, grey_components, (h, w))
        b_len = _line_length_for_object(anchor_coords, grey_components, (h, w))
        entries.append(
            {
                "color": color,
                "anchor": anchor_coords[0],
                "stamp_mask": stamp_mask,
                "stamp_bbox": stamp_bbox,
                "turns": (b_len - a_len) % 4,
            }
        )

    entries.sort(key=lambda item: (item["stamp_bbox"][0], item["stamp_bbox"][1], item["color"]))

    out = g.copy()
    for item in entries:
        mask = _rotate_cw(item["stamp_mask"], item["turns"])
        ar, ac = item["anchor"]
        _paint_mask(out, mask, item["color"], ar, ac)

    return out


def _build_source_line(mask: np.ndarray, length: int, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side in ("top", "bottom"):
        row = -1 if side == "top" else h
        edge_row = 0 if side == "top" else h - 1
        contacts = [c for c in range(w) if mask[edge_row, c]]
        contact = int(contacts[int(_RNG.integers(len(contacts)))])
        lo = contact - length + 1
        hi = contact
        start = int(_RNG.integers(lo, hi + 1))
        return [(row, start + i) for i in range(length)]
    col = -1 if side == "left" else w
    edge_col = 0 if side == "left" else w - 1
    contacts = [r for r in range(h) if mask[r, edge_col]]
    contact = int(contacts[int(_RNG.integers(len(contacts)))])
    lo = contact - length + 1
    hi = contact
    start = int(_RNG.integers(lo, hi + 1))
    return [(start + i, col) for i in range(length)]


def _build_source_pack(mask: np.ndarray, color: int, length: int, side: str) -> np.ndarray:
    stamp_cells = [(r, c) for r, c in np.argwhere(mask)]
    line_cells = _build_source_line(mask, length, side)
    all_cells = stamp_cells + line_cells
    min_r = min(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    max_r = max(r for r, _c in all_cells)
    max_c = max(c for _r, c in all_cells)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), _BG, dtype=int)
    for r, c in stamp_cells:
        patch[r - min_r, c - min_c] = int(color)
    for r, c in line_cells:
        patch[r - min_r, c - min_c] = _GREY
    return patch


def _anchor_line_cells(length: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        start = int(_RNG.integers(-length + 1, 1))
        return [(-1, start + i) for i in range(length)]
    start = int(_RNG.integers(-length + 1, 1))
    return [(start + i, -1) for i in range(length)]


def _shape_rotations_are_distinct(mask: np.ndarray) -> bool:
    seen = {np.rot90(mask, k).tobytes() + bytes(np.rot90(mask, k).shape) for k in range(4)}
    return len(seen) == 4


def _valid_base_shapes() -> list[np.ndarray]:
    valid = []
    for shape in _BASE_SHAPES:
        if not shape[0, 0] or not shape[0, -1] or not shape[-1, 0] or not shape[-1, -1]:
            continue
        if _shape_rotations_are_distinct(shape):
            valid.append(shape)
    assert valid
    return valid


_STAMP_LIBRARY = _valid_base_shapes()


def _choose_stamp_specs(count: int) -> list[dict]:
    global _GEN_COUNTER
    colors = list(_RNG.choice(_COLOR_POOL, size=count, replace=False))
    _RNG.shuffle(colors)

    specs: list[dict] = []
    offset = _GEN_COUNTER % 4
    for idx in range(count):
        base = _STAMP_LIBRARY[int(_RNG.integers(len(_STAMP_LIBRARY)))]
        source_rot = int(_RNG.integers(4))
        mask = _rotate_cw(base, source_rot)
        turns = (offset + idx) % 4
        pair = _LENGTH_PAIRS_BY_TURNS[turns][int(_RNG.integers(len(_LENGTH_PAIRS_BY_TURNS[turns])))]
        source_side = _SOURCE_LINE_SIDES[int(_RNG.integers(len(_SOURCE_LINE_SIDES)))]
        target_side = _ANCHOR_LINE_SIDES[int(_RNG.integers(len(_ANCHOR_LINE_SIDES)))]
        specs.append(
            {
                "color": int(colors[idx]),
                "mask": mask,
                "turns": turns,
                "source_len": int(pair[0]),
                "target_len": int(pair[1]),
                "source_side": source_side,
                "target_side": target_side,
            }
        )
    _GEN_COUNTER += 1
    return specs


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = _GREY
    grid[r0 + h - 1, c0 : c0 + w] = _GREY
    grid[r0 : r0 + h, c0] = _GREY
    grid[r0 : r0 + h, c0 + w - 1] = _GREY


def _cells_free_with_padding(
    occupied: set[tuple[int, int]], cells: list[tuple[int, int]], shape: tuple[int, int], padding: int
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - padding, r + padding + 1):
            for cc in range(c - padding, c + padding + 1):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in occupied:
                    return False
    return True


def _try_place_anchor(
    inner_bbox: tuple[int, int, int, int],
    patch_shape: tuple[int, int],
    line_cells_rel: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    ir0, ic0, ir1, ic1 = inner_bbox
    ph, pw = patch_shape

    min_line_r = min(dr for dr, _dc in line_cells_rel)
    min_line_c = min(dc for _dr, dc in line_cells_rel)
    max_line_r = max(dr for dr, _dc in line_cells_rel)
    max_line_c = max(dc for _dr, dc in line_cells_rel)

    r_lo = max(ir0 - min_line_r, ir0)
    c_lo = max(ic0 - min_line_c, ic0)
    r_hi = min(ir1 - max_line_r, ir1 - ph + 1)
    c_hi = min(ic1 - max_line_c, ic1 - pw + 1)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    candidates = [
        (r, c)
        for r in range(r_lo, r_hi + 1)
        for c in range(c_lo, c_hi + 1)
    ]
    if not candidates:
        return None
    order = _RNG.permutation(len(candidates))
    for idx in order:
        ar, ac = candidates[int(idx)]
        cells = [(ar, ac)] + [(ar + dr, ac + dc) for dr, dc in line_cells_rel]
        safe_r0, safe_c0 = ir0 + 1, ic0 + 1
        safe_r1, safe_c1 = ir1 - 1, ic1 - 1
        if any(
            not (safe_r0 <= r <= safe_r1 and safe_c0 <= c <= safe_c1)
            for r, c in cells
        ):
            continue
        if _cells_free_with_padding(occupied, cells, (ir1 + 1, ic1 + 1), padding=1):
            return int(ar), int(ac)
    return None


def _candidate_workspace_size(specs: list[dict]) -> tuple[int, int]:
    max_h = 0
    max_w = 0
    for spec in specs:
        rotated = _rotate_cw(spec["mask"], spec["turns"])
        max_h = max(max_h, int(rotated.shape[0]))
        max_w = max(max_w, int(rotated.shape[1]))
    inner_h = int(_RNG.integers(max(8, max_h + 3), max(11, max_h + 7)))
    inner_w = int(_RNG.integers(max(8, max_w + 4), max(12, max_w + 9)))
    return inner_h, inner_w


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        rotated = _rotate_cw(spec["mask"], spec["turns"])
        ar, ac = spec["anchor"]
        _paint_mask(out, rotated, spec["color"], ar, ac)
    return out


def _gen_attempt() -> np.ndarray | None:
    count = int(_RNG.integers(2, 4))
    specs = _choose_stamp_specs(count)

    source_packs = [
        _build_source_pack(spec["mask"], spec["color"], spec["source_len"], spec["source_side"])
        for spec in specs
    ]
    pack_gaps = [int(_RNG.integers(1, 3)) for _ in range(max(0, count - 1))]
    top_margin = int(_RNG.integers(1, 3))
    after_sources_gap = int(_RNG.integers(1, 3))
    top_rows: list[int] = []
    cursor = top_margin
    for i, pack in enumerate(source_packs):
        top_rows.append(cursor)
        cursor += int(pack.shape[0])
        if i < len(pack_gaps):
            cursor += pack_gaps[i]
    workspace_top = cursor + after_sources_gap

    inner_h, inner_w = _candidate_workspace_size(specs)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    left_margin = int(_RNG.integers(1, 3))
    right_margin = int(_RNG.integers(1, 4))
    workspace_left = left_margin + int(_RNG.integers(0, 3))
    width_need = max(workspace_left + outer_w + right_margin, max(pack.shape[1] + 2 for pack in source_packs))
    grid_w = width_need + int(_RNG.integers(0, 4))
    bottom_margin = int(_RNG.integers(1, 3))
    grid_h = workspace_top + outer_h + bottom_margin

    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    _paint_frame(grid, workspace_top, workspace_left, outer_h, outer_w)

    for row, pack in zip(top_rows, source_packs):
        col_lo = 1
        col_hi = grid_w - int(pack.shape[1]) - 1
        if col_lo > col_hi:
            return None
        col = int(_RNG.integers(col_lo, col_hi + 1))
        patch_h, patch_w = pack.shape
        if np.any(grid[row : row + patch_h, col : col + patch_w] != _BG):
            return None
        non_bg = pack != _BG
        grid[row : row + patch_h, col : col + patch_w][non_bg] = pack[non_bg]

    inner_bbox = (workspace_top + 1, workspace_left + 1, workspace_top + inner_h, workspace_left + inner_w)
    occupied_pairs: set[tuple[int, int]] = set()
    for spec in specs:
        line_rel = _anchor_line_cells(spec["target_len"], spec["target_side"])
        rotated = _rotate_cw(spec["mask"], spec["turns"])
        anchor = _try_place_anchor(inner_bbox, rotated.shape, line_rel, occupied_pairs)
        if anchor is None:
            return None
        ar, ac = anchor
        spec["anchor"] = (ar, ac)
        grid[ar, ac] = int(spec["color"])
        occupied_pairs.add((ar, ac))
        for dr, dc in line_rel:
            rr = ar + dr
            cc = ac + dc
            if grid[rr, cc] != _BG:
                return None
            grid[rr, cc] = _GREY
            occupied_pairs.add((rr, cc))

    expected = _expected_from_specs(grid, specs)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
