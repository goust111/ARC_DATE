from __future__ import annotations

import random

import numpy as np


BG = 0
ARROW = 2
SEPARATOR = 6
PATTERN_COLORS = (1, 3, 4, 5, 7, 8, 9)
TRANSFORMS = (
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ARROW_OFFSETS = {
    "up": ((1, 0), (1, 1), (0, 1)),
    "right": ((0, 0), (1, 0), (0, 1)),
    "down": ((0, 0), (0, 1), (1, 0)),
    "left": ((0, 1), (1, 1), (1, 0)),
}

# --------------- solver ---------------


def _apply_transform(arr: np.ndarray, flip_lr: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.fliplr(arr) if flip_lr else arr
    return np.rot90(out, int(rot_k)).copy()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(arr: np.ndarray) -> np.ndarray:
    mask = arr if arr.dtype == bool else arr != BG
    r0, c0, r1, c1 = _bbox(mask)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    tight = _tight_crop(mask.astype(bool))
    return int(tight.shape[0]), int(tight.shape[1]), tuple(int(v) for v in tight.ravel())


def _canonical_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return min(_mask_key(_apply_transform(mask, flip, rot)) for flip, rot in TRANSFORMS)


def _find_transform(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[bool, int]:
    target = _tight_crop(mask_b.astype(bool))
    matches = []
    for flip, rot in TRANSFORMS:
        cand = _tight_crop(_apply_transform(mask_a.astype(bool), flip, rot))
        if cand.shape == target.shape and np.array_equal(cand, target):
            matches.append((flip, rot))
    assert len(matches) == 1, "paired shapes must determine a unique D4 transform"
    return matches[0]


def _line_runs(indices: list[int]) -> list[tuple[int, int]]:
    assert indices
    indices = sorted(int(v) for v in indices)
    runs = []
    start = end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            runs.append((start, end))
            start = end = idx
    runs.append((start, end))
    return runs


def _separator_structure(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, g.shape)
    candidates = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, _line_runs(rows), _line_runs(cols)))
    assert len(candidates) == 1, "expected one separator color forming full rows and columns"
    return candidates[0]


def _spans(length: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for run_start, run_end in runs:
        assert start < run_start, "separator lines must leave nonempty regions"
        spans.append((start, run_start - 1))
        start = run_end + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _is_l_arrow(mask: np.ndarray) -> bool:
    if int(mask.sum()) != 3:
        return False
    r0, c0, r1, c1 = _bbox(mask)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return sub.shape == (2, 2) and int(sub.sum()) == 3


def _parse_region(region: np.ndarray) -> dict:
    red = region == ARROW
    if not red.any():
        assert np.all(region == BG), "empty regions must be pure background"
        return {"empty": True}

    assert _is_l_arrow(red), "arrow must be a 3-pixel L-shape"
    pattern_mask = (region != BG) & (region != ARROW)
    assert pattern_mask.any(), "non-empty region must contain a seed pattern"

    pr0, pc0, pr1, pc1 = _bbox(pattern_mask)
    red_pts = np.argwhere(red)
    if int(red_pts[:, 0].max()) < pr0:
        direction = "up"
    elif int(red_pts[:, 0].min()) > pr1:
        direction = "down"
    elif int(red_pts[:, 1].max()) < pc0:
        direction = "left"
    elif int(red_pts[:, 1].min()) > pc1:
        direction = "right"
    else:
        raise AssertionError("arrow must lie entirely on one side of the pattern")

    pattern = region[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    return {
        "empty": False,
        "direction": direction,
        "pattern": pattern,
        "mask": pattern != BG,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape seed regions under D4 symmetry, use each region's red arrow
    to choose an adjacent empty target region, and copy the seed there using the
    orientation of its paired partner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _sep_color, row_runs, col_runs = _separator_structure(g)
    row_spans = _spans(g.shape[0], row_runs)
    col_spans = _spans(g.shape[1], col_runs)

    regions: dict[tuple[int, int], dict] = {}
    nonempty: list[dict] = []
    out = g.copy()
    out[out == ARROW] = BG

    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            info = _parse_region(g[r0 : r1 + 1, c0 : c1 + 1])
            info.update(
                {
                    "ri": int(ri),
                    "ci": int(ci),
                    "r0": int(r0),
                    "c0": int(c0),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                }
            )
            regions[(ri, ci)] = info
            if not info["empty"]:
                nonempty.append(info)

    assert nonempty and len(nonempty) % 2 == 0

    grouped: dict[tuple[int, int, tuple[int, ...]], list[dict]] = {}
    for info in nonempty:
        grouped.setdefault(_canonical_signature(info["mask"]), []).append(info)
    assert all(len(items) == 2 for items in grouped.values()), "shape pairing must be unambiguous"

    for items in grouped.values():
        a, b = items
        a["partner"] = b
        b["partner"] = a

    for info in nonempty:
        dr, dc = DIRS[info["direction"]]
        target_key = (info["ri"] + dr, info["ci"] + dc)
        assert target_key in regions, "arrow must point to an adjacent region"
        target = regions[target_key]
        assert target["empty"], "teleport target region must be empty"

        flip, rot = _find_transform(info["mask"], info["partner"]["mask"])
        teleported = _apply_transform(info["pattern"], flip, rot)
        th, tw = map(int, teleported.shape)
        assert th <= target["height"] and tw <= target["width"]

        rr = target["r0"] + (target["height"] - th) // 2
        cc = target["c0"] + (target["width"] - tw) // 2
        patch = out[rr : rr + th, cc : cc + tw]
        mask = teleported != BG
        assert np.all(patch[mask] == BG), "teleported patterns must not overlap"
        patch[mask] = teleported[mask]

    return out


# --------------- generator ---------------


GRID_DIMS = ((3, 4), (4, 3), (4, 4), (3, 5), (5, 3))
REGION_HEIGHT_CHOICES = (8, 9, 10)
REGION_WIDTH_CHOICES = (8, 9, 10)
PAIR_COUNT_CAP = 3
SHAPE_BOX_MIN = 3
SHAPE_BOX_MAX = 5
SHAPE_CELLS_MIN = 4
SHAPE_CELLS_MAX = 8
GEN_ATTEMPTS = 400
SELECTION_ATTEMPTS = 60
SHAPE_ATTEMPTS = 120


def _grid_origin(rr: int, cc: int, region_h: int, region_w: int) -> tuple[int, int]:
    return rr * (region_h + 1), cc * (region_w + 1)


def _center_paste(canvas: np.ndarray, pattern: np.ndarray) -> None:
    h, w = map(int, canvas.shape)
    ph, pw = map(int, pattern.shape)
    r0 = (h - ph) // 2
    c0 = (w - pw) // 2
    mask = pattern != BG
    canvas[r0 : r0 + ph, c0 : c0 + pw][mask] = pattern[mask]


def _render_source_region(region_h: int, region_w: int, pattern: np.ndarray, direction: str) -> np.ndarray | None:
    ph, pw = map(int, pattern.shape)
    top_pad = 2 if direction == "up" else 0
    bottom_pad = 2 if direction == "down" else 0
    left_pad = 2 if direction == "left" else 0
    right_pad = 2 if direction == "right" else 0
    avail_h = region_h - top_pad - bottom_pad
    avail_w = region_w - left_pad - right_pad
    if ph > avail_h or pw > avail_w:
        return None

    pr0 = top_pad + (avail_h - ph) // 2
    pc0 = left_pad + (avail_w - pw) // 2
    out = np.full((region_h, region_w), BG, dtype=int)
    mask = pattern != BG
    out[pr0 : pr0 + ph, pc0 : pc0 + pw][mask] = pattern[mask]

    if direction == "up":
        ar0 = pr0 - 2
        ac0 = pc0 + (pw - 2) // 2
    elif direction == "down":
        ar0 = pr0 + ph
        ac0 = pc0 + (pw - 2) // 2
    elif direction == "left":
        ar0 = pr0 + (ph - 2) // 2
        ac0 = pc0 - 2
    else:
        ar0 = pr0 + (ph - 2) // 2
        ac0 = pc0 + pw

    if not (0 <= ar0 <= region_h - 2 and 0 <= ac0 <= region_w - 2):
        return None
    for dr, dc in ARROW_OFFSETS[direction]:
        out[ar0 + dr, ac0 + dc] = ARROW
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _random_connected_mask(h: int, w: int, n_cells: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    cells = [seed]
    while len(cells) < n_cells:
        frontier = set()
        for r, c in cells:
            frontier.update((rr, cc) for rr, cc in _neighbors4(r, c, h, w) if not mask[rr, cc])
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        mask[cell] = True
        cells.append(cell)
    return _tight_crop(mask)


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    seen = set()
    for flip, rot in TRANSFORMS:
        oriented = _tight_crop(_apply_transform(mask.astype(bool), flip, rot))
        key = _mask_key(oriented)
        if key not in seen:
            seen.add(key)
            out.append(oriented.astype(bool))
    return out


def _random_base_shape(existing: set[tuple[int, int, tuple[int, ...]]]) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int, tuple[int, ...]]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(SHAPE_BOX_MIN, SHAPE_BOX_MAX)
        box_w = random.randint(SHAPE_BOX_MIN, SHAPE_BOX_MAX)
        max_cells = min(SHAPE_CELLS_MAX, box_h * box_w - 1)
        if max_cells < SHAPE_CELLS_MIN:
            continue
        mask = _random_connected_mask(box_h, box_w, random.randint(SHAPE_CELLS_MIN, max_cells))
        if mask is None:
            continue
        tight = _tight_crop(mask.astype(bool))
        if not (SHAPE_BOX_MIN <= tight.shape[0] <= SHAPE_BOX_MAX and SHAPE_BOX_MIN <= tight.shape[1] <= SHAPE_BOX_MAX):
            continue
        orientations = _unique_orientations(tight)
        if len(orientations) != 8:
            continue
        signature = _canonical_signature(tight)
        if signature in existing:
            continue
        return tight.astype(bool), orientations, signature
    return None


def _region_edges(rows: int, cols: int) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    out = {name: [] for name in DIRS}
    for r in range(rows):
        for c in range(cols):
            for name, (dr, dc) in DIRS.items():
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    out[name].append(((r, c), (rr, cc)))
    return out


def _select_directed_edges(rows: int, cols: int, n_edges: int) -> list[tuple[tuple[int, int], tuple[int, int], str]] | None:
    edges_by_dir = _region_edges(rows, cols)

    def search(occupied: set[tuple[int, int]], picked: list[tuple[tuple[int, int], tuple[int, int], str]]):
        if len(picked) == n_edges:
            return list(picked)

        available = {
            name: [(src, dst) for src, dst in edges if src not in occupied and dst not in occupied]
            for name, edges in edges_by_dir.items()
        }
        valid_dirs = [name for name, edges in available.items() if edges]
        if not valid_dirs:
            return None

        random.shuffle(valid_dirs)
        valid_dirs.sort(key=lambda name: len(available[name]))
        for name in valid_dirs:
            candidates = available[name][:]
            random.shuffle(candidates)
            for src, dst in candidates:
                occupied.add(src)
                occupied.add(dst)
                picked.append((src, dst, name))
                result = search(occupied, picked)
                if result is not None:
                    return result
                picked.pop()
                occupied.remove(src)
                occupied.remove(dst)
        return None

    for _ in range(SELECTION_ATTEMPTS):
        result = search(set(), [])
        if result is not None:
            random.shuffle(result)
            return result
    return None


def _pattern_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = int(color)
    return out


def _build_expected(inp: np.ndarray, sources: list[dict], rows: int, cols: int, region_h: int, region_w: int) -> np.ndarray:
    del rows, cols
    out = inp.copy()
    out[out == ARROW] = BG
    by_id = {info["id"]: info for info in sources}
    for info in sources:
        partner = by_id[info["partner_id"]]
        flip, rot = _find_transform(info["mask"], partner["mask"])
        teleported = _apply_transform(info["pattern"], flip, rot)
        tr, tc = info["target"]
        r0, c0 = _grid_origin(tr, tc, region_h, region_w)
        patch = out[r0 : r0 + region_h, c0 : c0 + region_w]
        _center_paste(patch, teleported)
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(GRID_DIMS)
    region_h = random.choice(REGION_HEIGHT_CHOICES)
    region_w = random.choice(REGION_WIDTH_CHOICES)
    total_regions = rows * cols
    max_pairs = min(PAIR_COUNT_CAP, total_regions // 4)
    pair_count = random.randint(1, max_pairs)
    source_count = 2 * pair_count

    selected = _select_directed_edges(rows, cols, source_count)
    if selected is None:
        return None

    height = rows * region_h + (rows - 1)
    width = cols * region_w + (cols - 1)
    inp = np.full((height, width), BG, dtype=int)
    for rr in range(1, rows):
        inp[rr * (region_h + 1) - 1, :] = SEPARATOR
    for cc in range(1, cols):
        inp[:, cc * (region_w + 1) - 1] = SEPARATOR

    sources: list[dict] = []
    existing_shapes: set[tuple[int, int, tuple[int, ...]]] = set()
    random.shuffle(selected)
    source_id = 0
    for pair_index in range(pair_count):
        shape_info = _random_base_shape(existing_shapes)
        if shape_info is None:
            return None
        _base_mask, orientations, signature = shape_info
        existing_shapes.add(signature)
        specs = selected[2 * pair_index : 2 * pair_index + 2]
        for src, dst, direction in specs:
            oriented_mask = random.choice(orientations)
            color = random.choice(PATTERN_COLORS)
            pattern = _pattern_from_mask(oriented_mask, color)
            region = _render_source_region(region_h, region_w, pattern, direction)
            if region is None:
                return None
            r0, c0 = _grid_origin(src[0], src[1], region_h, region_w)
            inp[r0 : r0 + region_h, c0 : c0 + region_w] = region
            sources.append(
                {
                    "id": source_id,
                    "pair_index": pair_index,
                    "source": src,
                    "target": dst,
                    "direction": direction,
                    "pattern": pattern,
                    "mask": pattern != BG,
                }
            )
            source_id += 1

    for pair_index in range(pair_count):
        pair_sources = [info for info in sources if info["pair_index"] == pair_index]
        assert len(pair_sources) == 2
        a, b = pair_sources
        a["partner_id"] = b["id"]
        b["partner_id"] = a["id"]

    expected = _build_expected(inp, sources, rows, cols, region_h, region_w)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid teleport-pair ARC input")
