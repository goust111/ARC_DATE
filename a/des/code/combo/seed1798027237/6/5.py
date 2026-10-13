import numpy as np
from collections import deque


BG = 0
AXIS_COLOR = 9
OBJECT_COLORS = (1, 2, 3, 4, 5, 6)
GRID_MIN = 18
GRID_MAX = 28
PAIR_COUNT_RANGE = (2, 4)
ROW_GAP_RANGE = (1, 4)
NEAR_GAP_RANGE = (0, 3)
SHAPE_BOX_H_RANGE = (2, 6)
SHAPE_BOX_W_RANGE = (2, 6)
SHAPE_AREA_RANGE = (3, 10)
GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _crop_coords(coords):
    arr = np.asarray(list(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords):
    r0, c0, r1, c1 = _crop_coords(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _reflect_shape(norm, h: int, w: int, axis_kind: str):
    if axis_kind == "vertical":
        return tuple(sorted((r, w - 1 - c) for r, c in norm))
    return tuple(sorted((h - 1 - r, c) for r, c in norm))


def _reflect_absolute(coords, axis_kind: str, axis_index: int):
    if axis_kind == "vertical":
        return [(int(r), int(2 * axis_index - c)) for r, c in coords]
    return [(int(2 * axis_index - r), int(c)) for r, c in coords]


def _crop_non_bg(grid: np.ndarray) -> np.ndarray:
    nz = np.argwhere(grid != BG)
    assert nz.size > 0
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _find_components(grid: np.ndarray):
    h, w = grid.shape
    mask = (grid != BG) & (grid != AXIS_COLOR)
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                assert int(grid[rr, cc]) == color
                for nr, nc in _neighbors(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append((color, coords))
    return comps


def _extract_objects(grid: np.ndarray, axis_kind: str, axis_index: int):
    objects = []
    for color, coords in _find_components(grid):
        arr = np.asarray(coords, dtype=int)
        if axis_kind == "vertical":
            if np.all(arr[:, 1] < axis_index):
                side = -1
                dist = int(axis_index - arr[:, 1].max() - 1)
                parallel_center2 = int(arr[:, 0].min() + arr[:, 0].max())
            else:
                assert np.all(arr[:, 1] > axis_index)
                side = 1
                dist = int(arr[:, 1].min() - axis_index - 1)
                parallel_center2 = int(arr[:, 0].min() + arr[:, 0].max())
        else:
            if np.all(arr[:, 0] < axis_index):
                side = -1
                dist = int(axis_index - arr[:, 0].max() - 1)
                parallel_center2 = int(arr[:, 1].min() + arr[:, 1].max())
            else:
                assert np.all(arr[:, 0] > axis_index)
                side = 1
                dist = int(arr[:, 0].min() - axis_index - 1)
                parallel_center2 = int(arr[:, 1].min() + arr[:, 1].max())

        norm, box_h, box_w = _normalize_coords(coords)
        shape_key = (box_h, box_w, norm)
        mirror_key = (box_h, box_w, _reflect_shape(norm, box_h, box_w, axis_kind))
        objects.append(
            {
                "color": int(color),
                "coords": [(int(r), int(c)) for r, c in coords],
                "coord_set": set((int(r), int(c)) for r, c in coords),
                "side": side,
                "dist": dist,
                "shape_key": shape_key,
                "mirror_key": mirror_key,
                "parallel_center2": parallel_center2,
            }
        )
    assert objects and len(objects) % 2 == 0
    return objects


def _pair_score(a, b, axis_kind: str, axis_index: int):
    if a["side"] == b["side"]:
        return None
    neg = a if a["side"] < 0 else b
    pos = b if a["side"] < 0 else a
    if neg["mirror_key"] != pos["shape_key"]:
        return None
    if a["dist"] == b["dist"]:
        return None
    far = a if a["dist"] > b["dist"] else b
    near = b if far is a else a
    reflected = set(_reflect_absolute(far["coords"], axis_kind, axis_index))
    overlap = len(reflected & near["coord_set"])
    if overlap == 0:
        return None
    return overlap * 1000 - abs(a["parallel_center2"] - b["parallel_center2"]) - abs(a["dist"] - b["dist"])


def _match_pairs(objects, axis_kind: str, axis_index: int):
    neg = [obj for obj in objects if obj["side"] < 0]
    pos = [obj for obj in objects if obj["side"] > 0]
    assert len(neg) == len(pos)

    candidate_map = {}
    for i, obj_a in enumerate(neg):
        opts = []
        for j, obj_b in enumerate(pos):
            score = _pair_score(obj_a, obj_b, axis_kind, axis_index)
            if score is not None:
                opts.append((j, score))
        opts.sort(key=lambda item: (-item[1], item[0]))
        assert opts
        candidate_map[i] = opts

    best_score = None
    best_pairs = None

    def search(chosen, used_pos, total_score):
        nonlocal best_score, best_pairs
        if len(chosen) == len(neg):
            if best_score is None or total_score > best_score:
                best_score = total_score
                best_pairs = sorted(chosen.items())
            return

        remaining = [i for i in range(len(neg)) if i not in chosen]
        next_i = min(
            remaining,
            key=lambda idx: sum(1 for j, _ in candidate_map[idx] if j not in used_pos),
        )
        options = [(j, s) for j, s in candidate_map[next_i] if j not in used_pos]
        for j, score in options:
            chosen[next_i] = j
            used_pos.add(j)
            search(chosen, used_pos, total_score + score)
            used_pos.remove(j)
            del chosen[next_i]

    search({}, set(), 0)
    assert best_pairs is not None
    return [(neg[i], pos[j]) for i, j in best_pairs]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each mirrored pair of objects, fold the farther object across the maroon axis,
    and merge it onto the nearer side while keeping the nearer object's color on overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = np.argwhere(g == AXIS_COLOR)
    assert markers.shape[0] >= 2
    unique_rows = np.unique(markers[:, 0])
    unique_cols = np.unique(markers[:, 1])
    if unique_rows.size == 1:
        axis_kind = "horizontal"
        axis_index = int(unique_rows[0])
    else:
        assert unique_cols.size == 1
        axis_kind = "vertical"
        axis_index = int(unique_cols[0])

    objects = _extract_objects(g, axis_kind, axis_index)
    pairs = _match_pairs(objects, axis_kind, axis_index)

    out = g.copy()
    out[out == AXIS_COLOR] = BG

    far_objects = []
    for left_obj, right_obj in pairs:
        far = left_obj if left_obj["dist"] > right_obj["dist"] else right_obj
        far_objects.append(far)
    for far in far_objects:
        for r, c in far["coords"]:
            out[r, c] = BG

    for left_obj, right_obj in pairs:
        far = left_obj if left_obj["dist"] > right_obj["dist"] else right_obj
        for r, c in _reflect_absolute(far["coords"], axis_kind, axis_index):
            if out[r, c] == BG:
                out[r, c] = far["color"]

    return _crop_non_bg(out)


def _shape_mask_to_coords(mask: np.ndarray):
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _random_shape():
    for _ in range(200):
        box_h = int(_RNG.integers(SHAPE_BOX_H_RANGE[0], SHAPE_BOX_H_RANGE[1]))
        box_w = int(_RNG.integers(SHAPE_BOX_W_RANGE[0], SHAPE_BOX_W_RANGE[1]))
        area_hi = min(SHAPE_AREA_RANGE[1], box_h * box_w)
        area_lo = min(area_hi, SHAPE_AREA_RANGE[0])
        area = int(_RNG.integers(area_lo, area_hi + 1))

        start = (int(_RNG.integers(box_h)), int(_RNG.integers(box_w)))
        cells = {start}
        frontier = set(_neighbors(start[0], start[1], box_h, box_w))
        while len(cells) < area and frontier:
            frontier_list = list(frontier)
            pick = frontier_list[int(_RNG.integers(len(frontier_list)))]
            frontier.remove(pick)
            cells.add(pick)
            for nxt in _neighbors(pick[0], pick[1], box_h, box_w):
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != area:
            continue

        norm, h, w = _normalize_coords(cells)
        if h < 2 or w < 2:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        return mask
    return None


def _canonical_pair_signature(mask: np.ndarray):
    coords = _shape_mask_to_coords(mask)
    norm, h, w = _normalize_coords(coords)
    mirror = _reflect_shape(norm, h, w, "vertical")
    return min((h, w, norm), (h, w, mirror))


def _split_padding(total_extra: int):
    left = int(_RNG.integers(1, total_extra))
    return left, total_extra - left


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int):
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    assert np.all(sub[mask] == BG)
    sub[mask] = color


def _paint_mask_with_priority(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int):
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[np.asarray(mask) & (sub == BG)] = color


def _reflect_mask_h(mask: np.ndarray):
    return mask[:, ::-1]


def _sample_axis_markers(length: int):
    parity = int(_RNG.integers(2))
    slots = list(range(parity, length, 2))
    count = min(len(slots), int(_RNG.integers(3, min(7, len(slots)) + 1)))
    picks = _RNG.choice(slots, size=count, replace=False)
    return sorted(int(x) for x in picks)


def _build_vertical_instance(pair_specs, total_h: int, total_w: int, pad_top: int, pad_left: int, axis_col: int):
    grid = np.zeros((total_h, total_w), dtype=int)
    expected = np.zeros((total_h, total_w), dtype=int)
    axis_global = pad_left + axis_col

    for spec in pair_specs:
        top = pad_top + spec["top"]
        mask = spec["mask"]
        mirror_mask = _reflect_mask_h(mask)
        h, w = mask.shape
        if spec["far_left"]:
            far_left = axis_global - spec["far_gap"] - w
            near_left = axis_global + 1 + spec["near_gap"]
            _paint_mask(grid, top, far_left, mask, spec["far_color"])
            _paint_mask(grid, top, near_left, mirror_mask, spec["near_color"])
            _paint_mask(expected, top, near_left, mirror_mask, spec["near_color"])
            reflected_far_left = axis_global + 1 + spec["far_gap"]
            _paint_mask_with_priority(expected, top, reflected_far_left, mirror_mask, spec["far_color"])
        else:
            near_left = axis_global - spec["near_gap"] - w
            far_left = axis_global + 1 + spec["far_gap"]
            _paint_mask(grid, top, near_left, mask, spec["near_color"])
            _paint_mask(grid, top, far_left, mirror_mask, spec["far_color"])
            _paint_mask(expected, top, near_left, mask, spec["near_color"])
            reflected_far_left = axis_global - spec["far_gap"] - w
            _paint_mask_with_priority(expected, top, reflected_far_left, mask, spec["far_color"])

    for row in _sample_axis_markers(total_h):
        assert grid[row, axis_global] == BG
        grid[row, axis_global] = AXIS_COLOR

    return grid, _crop_non_bg(expected)


def _gen_attempt():
    pair_count = int(_RNG.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1]))
    specs = []
    signatures = set()
    for _ in range(pair_count):
        for _inner in range(200):
            mask = _random_shape()
            if mask is None:
                continue
            signature = _canonical_pair_signature(mask)
            if signature in signatures:
                continue
            h, w = mask.shape
            near_gap = int(_RNG.integers(NEAR_GAP_RANGE[0], NEAR_GAP_RANGE[1]))
            max_delta = min(w - 1, 3)
            if max_delta < 1:
                continue
            delta = int(_RNG.integers(1, max_delta + 1))
            far_gap = near_gap + delta
            near_color, far_color = _RNG.choice(OBJECT_COLORS, size=2, replace=False)
            specs.append(
                {
                    "mask": mask,
                    "near_gap": near_gap,
                    "far_gap": far_gap,
                    "far_left": bool(_RNG.integers(2)),
                    "near_color": int(near_color),
                    "far_color": int(far_color),
                }
            )
            signatures.add(signature)
            break
        else:
            return None

    top = 0
    for index, spec in enumerate(specs):
        spec["top"] = top
        top += spec["mask"].shape[0]
        if index + 1 < len(specs):
            top += int(_RNG.integers(ROW_GAP_RANGE[0], ROW_GAP_RANGE[1]))
    local_h = top

    left_reach = []
    right_reach = []
    for spec in specs:
        w = spec["mask"].shape[1]
        reach = spec["far_gap"] + w
        left_reach.append(reach)
        right_reach.append(reach)
    axis_col = max(left_reach)
    local_w = axis_col + 1 + max(right_reach)

    min_total_h = max(GRID_MIN, local_h + 2)
    max_total_h = GRID_MAX
    min_total_w = max(GRID_MIN, local_w + 2)
    max_total_w = GRID_MAX
    if min_total_h > max_total_h or min_total_w > max_total_w:
        return None

    total_h = int(_RNG.integers(min_total_h, max_total_h + 1))
    total_w = int(_RNG.integers(min_total_w, max_total_w + 1))
    pad_top, pad_bottom = _split_padding(total_h - local_h)
    pad_left, pad_right = _split_padding(total_w - local_w)
    assert pad_bottom >= 1 and pad_right >= 1

    grid, expected = _build_vertical_instance(specs, total_h, total_w, pad_top, pad_left, axis_col)
    if bool(_RNG.integers(2)):
        grid = grid.T
        expected = expected.T

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
