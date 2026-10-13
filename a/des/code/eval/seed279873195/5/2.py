import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int):
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": int(color), "coords": coords, "bbox": _bbox(coords)})
    return comps


def _axis_center_distance(component, grid_shape, orientation: str) -> float:
    r0, c0, r1, c1 = component["bbox"]
    if orientation == "H":
        grid_center = (grid_shape[1] - 1) / 2.0
        comp_center = (c0 + c1) / 2.0
    else:
        grid_center = (grid_shape[0] - 1) / 2.0
        comp_center = (r0 + r1) / 2.0
    return abs(comp_center - grid_center)


def _closer_component_index(comp_a, comp_b, grid_shape, orientation: str):
    da = _axis_center_distance(comp_a, grid_shape, orientation)
    db = _axis_center_distance(comp_b, grid_shape, orientation)
    if da < db:
        return 0
    if db < da:
        return 1
    if orientation == "H":
        ca = (comp_a["bbox"][1] + comp_a["bbox"][3]) / 2.0
        cb = (comp_b["bbox"][1] + comp_b["bbox"][3]) / 2.0
    else:
        ca = (comp_a["bbox"][0] + comp_a["bbox"][2]) / 2.0
        cb = (comp_b["bbox"][0] + comp_b["bbox"][2]) / 2.0
    return 0 if ca <= cb else 1


def _movement_amounts(comp_a, comp_b, grid_shape, orientation: str, gap: int):
    base = gap // 2
    if gap % 2 == 0:
        return base, base
    closer = _closer_component_index(comp_a, comp_b, grid_shape, orientation)
    if closer == 0:
        return base + 1, base
    return base, base + 1


def _horizontal_gap(left, right):
    row_to_rightmost = {}
    row_to_leftmost = {}
    for r, c in left["coords"]:
        row_to_rightmost[r] = max(row_to_rightmost.get(r, c), c)
    for r, c in right["coords"]:
        row_to_leftmost[r] = min(row_to_leftmost.get(r, c), c)
    r0, _, r1, _ = left["bbox"]
    gaps = set()
    for r in range(r0, r1 + 1):
        assert r in row_to_rightmost and r in row_to_leftmost
        gaps.add(int(row_to_leftmost[r] - row_to_rightmost[r] - 1))
    assert len(gaps) == 1
    gap = gaps.pop()
    assert gap >= 1
    return gap


def _vertical_gap(top, bottom):
    col_to_bottommost = {}
    col_to_topmost = {}
    for r, c in top["coords"]:
        col_to_bottommost[c] = max(col_to_bottommost.get(c, r), r)
    for r, c in bottom["coords"]:
        col_to_topmost[c] = min(col_to_topmost.get(c, r), r)
    _, c0, _, c1 = top["bbox"]
    gaps = set()
    for c in range(c0, c1 + 1):
        assert c in col_to_bottommost and c in col_to_topmost
        gaps.add(int(col_to_topmost[c] - col_to_bottommost[c] - 1))
    assert len(gaps) == 1
    gap = gaps.pop()
    assert gap >= 1
    return gap


def _paint_coords(grid: np.ndarray, coords, color: int):
    for r, c in coords:
        assert grid[r, c] == BG
        grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert 3 <= len(colors) <= 5

    out = np.full_like(g, BG)
    for color in colors:
        comps = _find_color_components(g, color)
        assert len(comps) == 2
        a, b = comps
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        same_rows = (ar0, ar1) == (br0, br1)
        same_cols = (ac0, ac1) == (bc0, bc1)
        assert same_rows != same_cols

        if same_rows:
            left, right = sorted(comps, key=lambda comp: comp["bbox"][1])
            gap = _horizontal_gap(left, right)
            move_left, move_right = _movement_amounts(left, right, g.shape, "H", gap)
            moved_left = [(r, c + move_left) for r, c in left["coords"]]
            moved_right = [(r, c - move_right) for r, c in right["coords"]]
            _paint_coords(out, moved_left, color)
            _paint_coords(out, moved_right, color)
        else:
            top, bottom = sorted(comps, key=lambda comp: comp["bbox"][0])
            gap = _vertical_gap(top, bottom)
            move_top, move_bottom = _movement_amounts(top, bottom, g.shape, "V", gap)
            moved_top = [(r + move_top, c) for r, c in top["coords"]]
            moved_bottom = [(r - move_bottom, c) for r, c in bottom["coords"]]
            _paint_coords(out, moved_top, color)
            _paint_coords(out, moved_bottom, color)

    return out


def _shift_coords(coords, dr: int, dc: int):
    return [(r + dr, c + dc) for r, c in coords]


def _cut_sequence(length: int, max_cut: int):
    if max_cut == 0:
        return [0] * length
    low = 0 if max_cut == 1 else 1
    high = max_cut if max_cut == 1 else max_cut - 1
    start = int(RNG.integers(low, high + 1))
    seq = [start]
    for _ in range(1, length):
        step = int(RNG.choice([-1, 0, 1]))
        nxt = int(np.clip(seq[-1] + step, 0, max_cut))
        seq.append(nxt)
    if length >= 3 and max_cut >= 2 and len(set(seq)) == 1 and RNG.random() < 0.75:
        idx = int(RNG.integers(0, length))
        delta = -1 if seq[idx] == max_cut else 1 if seq[idx] == 0 else int(RNG.choice([-1, 1]))
        seq[idx] = int(np.clip(seq[idx] + delta, 0, max_cut))
    return seq


def _build_pair_spec():
    orientation = "H" if RNG.random() < 0.5 else "V"
    odd_gap = RNG.random() < 0.5
    if orientation == "H":
        rect_h = int(RNG.integers(2, 6))
        rect_w = int(RNG.integers(4, 9))
        cut = _cut_sequence(rect_h, rect_w - 2)
        left = [(r, c) for r in range(rect_h) for c in range(cut[r] + 1)]
        right = [(r, c) for r in range(rect_h) for c in range(cut[r] + 1, rect_w)]
        spread = max(cut) - min(cut)
    else:
        rect_h = int(RNG.integers(4, 9))
        rect_w = int(RNG.integers(2, 6))
        cut = _cut_sequence(rect_w, rect_h - 2)
        left = [(r, c) for c in range(rect_w) for r in range(cut[c] + 1)]
        right = [(r, c) for c in range(rect_w) for r in range(cut[c] + 1, rect_h)]
        spread = max(cut) - min(cut)
    min_gap = max(1, spread + 1)
    candidates = [g for g in range(min_gap, 9) if (g % 2 == 1) == odd_gap]
    if not candidates:
        candidates = [g for g in range(min_gap, 11) if (g % 2 == 1) == odd_gap]
    gap = int(RNG.choice(candidates))
    return {
        "orientation": orientation,
        "rect_h": rect_h,
        "rect_w": rect_w,
        "gap": gap,
        "part_a": left,
        "part_b": right,
    }


def _bbox_free(occ: np.ndarray, bbox, pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0] - 1, r1 + pad)
    cc1 = min(occ.shape[1] - 1, c1 + pad)
    return not bool(occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(occ: np.ndarray, bbox, pad: int = 1):
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0] - 1, r1 + pad)
    cc1 = min(occ.shape[1] - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _coords_in_bounds(coords, shape) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _candidate_input_parts(final_a, final_b, orientation: str, gap: int, shape):
    base = gap // 2
    candidates = []
    if orientation == "H":
        if gap % 2 == 0:
            candidates.append((_shift_coords(final_a, 0, -base), _shift_coords(final_b, 0, base), 0))
        else:
            candidates.append((_shift_coords(final_a, 0, -(base + 1)), _shift_coords(final_b, 0, base), 0))
            candidates.append((_shift_coords(final_a, 0, -base), _shift_coords(final_b, 0, base + 1), 1))
    else:
        if gap % 2 == 0:
            candidates.append((_shift_coords(final_a, -base, 0), _shift_coords(final_b, base, 0), 0))
        else:
            candidates.append((_shift_coords(final_a, -(base + 1), 0), _shift_coords(final_b, base, 0), 0))
            candidates.append((_shift_coords(final_a, -base, 0), _shift_coords(final_b, base + 1, 0), 1))

    valid = []
    for cand_a, cand_b, extra_idx in candidates:
        if not (_coords_in_bounds(cand_a, shape) and _coords_in_bounds(cand_b, shape)):
            continue
        comp_a = {"bbox": _bbox(cand_a)}
        comp_b = {"bbox": _bbox(cand_b)}
        da = _axis_center_distance(comp_a, shape, orientation)
        db = _axis_center_distance(comp_b, shape, orientation)
        if gap % 2 == 0:
            valid.append((cand_a, cand_b))
        elif abs(da - db) > 1e-9 and ((da < db and extra_idx == 0) or (db < da and extra_idx == 1)):
            valid.append((cand_a, cand_b))
    return valid


def _place_pairs(shape, specs, colors):
    h, w = shape
    input_occ = np.zeros((h, w), dtype=bool)
    output_occ = np.zeros((h, w), dtype=bool)
    placements = []

    for spec, color in zip(specs, colors):
        rect_h = spec["rect_h"]
        rect_w = spec["rect_w"]
        for _ in range(300):
            r0 = int(RNG.integers(0, h - rect_h + 1))
            c0 = int(RNG.integers(0, w - rect_w + 1))
            out_bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _bbox_free(output_occ, out_bbox, pad=1):
                continue

            final_a = _shift_coords(spec["part_a"], r0, c0)
            final_b = _shift_coords(spec["part_b"], r0, c0)
            candidates = _candidate_input_parts(final_a, final_b, spec["orientation"], spec["gap"], shape)
            if not candidates:
                continue
            RNG.shuffle(candidates)
            placed = False
            for input_a, input_b in candidates:
                in_bbox = _bbox(input_a + input_b)
                if not _bbox_free(input_occ, in_bbox, pad=1):
                    continue
                placements.append(
                    {
                        "color": int(color),
                        "input_a": input_a,
                        "input_b": input_b,
                        "out_bbox": out_bbox,
                    }
                )
                _mark_bbox(output_occ, out_bbox, pad=1)
                _mark_bbox(input_occ, in_bbox, pad=1)
                placed = True
                break
            if placed:
                break
        else:
            return None
    return placements


def _build_grids(shape, placements):
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    for item in placements:
        color = item["color"]
        _paint_coords(inp, item["input_a"], color)
        _paint_coords(inp, item["input_b"], color)
        r0, c0, r1, c1 = item["out_bbox"]
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == BG)
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return inp, out


def _valid_scene(inp: np.ndarray, expected: np.ndarray) -> bool:
    if np.count_nonzero(inp == BG) <= inp.size // 2:
        return False
    colors = [int(c) for c in np.unique(inp) if int(c) != BG]
    if not (3 <= len(colors) <= 5):
        return False
    for color in colors:
        comps = _find_color_components(inp, color)
        if len(comps) != 2:
            return False
        a, b = comps
        same_rows = (a["bbox"][0], a["bbox"][2]) == (b["bbox"][0], b["bbox"][2])
        same_cols = (a["bbox"][1], a["bbox"][3]) == (b["bbox"][1], b["bbox"][3])
        if same_rows == same_cols:
            return False
    return np.array_equal(main(inp), expected)


def _gen_attempt():
    pair_count = int(RNG.integers(3, 6))
    min_side = 16 + 2 * pair_count
    h = int(RNG.integers(min_side, 31))
    w = int(RNG.integers(min_side, 31))
    colors = list(RNG.choice(np.array(PALETTE, dtype=int), size=pair_count, replace=False))
    specs = [_build_pair_spec() for _ in range(pair_count)]
    placements = _place_pairs((h, w), specs, colors)
    if placements is None:
        return None
    inp, expected = _build_grids((h, w), placements)
    if not _valid_scene(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
