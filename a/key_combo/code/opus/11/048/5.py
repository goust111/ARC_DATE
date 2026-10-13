import random
from collections import deque

import numpy as np


BG = 0
SPRITE = 1
DIFF = 6
COPY = 8

GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    out[shifted[:, 0], shifted[:, 1]] = True
    return out


def _place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _mask_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components_of_mask(mask)) == 1


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_sprite_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    if area < max(4, (h * w + 2) // 3):
        return False
    if area > h * w - 1:
        return False
    return _mask_connected(mask) and _full_bbox(mask)


def _double_center_from_bbox(box: tuple[int, int, int, int]) -> int:
    return int(box[1] + box[3])


def _left_from_double_center(double_center: int, width: int) -> int:
    return (int(double_center) - (int(width) - 1)) // 2


def _align_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = max(mask_a.shape[1], mask_b.shape[1])
    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    a[: mask_a.shape[0], : mask_a.shape[1]] = mask_a
    b[: mask_b.shape[0], : mask_b.shape[1]] = mask_b
    return a, b


def _xor_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a, b = _align_masks(mask_a, mask_b)
    return a ^ b


def _union_masks(masks: list[np.ndarray]) -> np.ndarray:
    assert masks
    h = max(mask.shape[0] for mask in masks)
    w = max(mask.shape[1] for mask in masks)
    out = np.zeros((h, w), dtype=bool)
    for mask in masks:
        out[: mask.shape[0], : mask.shape[1]] |= mask
    return out


def _blue_components(grid: np.ndarray) -> list[dict]:
    mask = grid == SPRITE
    comps = []
    for coords in _components_of_mask(mask):
        box = _bbox(coords)
        comps.append({"coords": coords, "bbox": box, "mask": _crop_component_mask(coords)})
    return comps


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert g.shape[0] % 2 == 0
    assert set(map(int, np.unique(g))).issubset({BG, SPRITE})

    h, _w = g.shape
    mid = h // 2
    top: dict[int, dict] = {}
    bottom: dict[int, dict] = {}

    for comp in _blue_components(g):
        r0, _c0, r1, _c1 = comp["bbox"]
        key = _double_center_from_bbox(comp["bbox"])
        if r1 < mid:
            assert key not in top
            top[key] = comp
        else:
            assert r0 >= mid
            assert key not in bottom
            bottom[key] = comp

    assert 3 <= len(top) <= 4
    assert 3 <= len(bottom) <= 4

    keys = sorted(set(top) | set(bottom))
    assert len(keys) in (4, 5)
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    assert len(set(gaps)) == 1

    top_rows = sorted(comp["bbox"][0] for comp in top.values())
    bottom_rows = sorted(comp["bbox"][0] for comp in bottom.values())
    top_row = int(round(float(np.median(top_rows))))
    bottom_row = int(round(float(np.median(bottom_rows))))

    complete_pairs = []
    missing_columns = []
    for key in keys:
        top_comp = top.get(key)
        bottom_comp = bottom.get(key)
        if top_comp is not None and bottom_comp is not None:
            complete_pairs.append(
                {
                    "key": key,
                    "top_mask": top_comp["mask"],
                    "bottom_mask": bottom_comp["mask"],
                    "xor": _xor_mask(top_comp["mask"], bottom_comp["mask"]),
                }
            )
        else:
            assert top_comp is not None or bottom_comp is not None
            missing_columns.append(
                {
                    "key": key,
                    "missing_half": "top" if top_comp is None else "bottom",
                    "present_mask": bottom_comp["mask"] if top_comp is None else top_comp["mask"],
                }
            )

    assert len(missing_columns) in (1, 2)
    assert any(not pair["xor"].any() for pair in complete_pairs)
    assert any(pair["xor"].any() for pair in complete_pairs)
    return {
        "grid": g,
        "top": top,
        "bottom": bottom,
        "keys": keys,
        "top_row": top_row,
        "bottom_row": bottom_row,
        "complete_pairs": complete_pairs,
        "missing_columns": missing_columns,
    }


def _fill_decision(analysis: dict, present_mask: np.ndarray) -> tuple[int, np.ndarray]:
    matched = [
        pair["xor"]
        for pair in analysis["complete_pairs"]
        if np.array_equal(pair["top_mask"], present_mask) or np.array_equal(pair["bottom_mask"], present_mask)
    ]
    if matched and all(not xor.any() for xor in matched):
        return COPY, present_mask

    non_empty = [xor for xor in matched if xor.any()]
    if not non_empty:
        non_empty = [pair["xor"] for pair in analysis["complete_pairs"] if pair["xor"].any()]
    return DIFF, _union_masks(non_empty)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare aligned top/bottom lattice sprites and fill each missing slot with either a teal copy or a pink XOR-difference mask.

    Input structure:
    - The grid has a black background and even height, split into equal top and bottom halves.
    - Blue 4-connected sprites occupy a shared equally spaced column lattice across the two halves.
    - Some columns contain a blue sprite in both halves, while one or two columns are missing in exactly one half.

    Rule:
    - Pair top and bottom sprites by lattice column and align paired masks by their bounding-box top-left corners.
    - Compute each paired column's XOR mask: cells occupied in exactly one of the two aligned sprites are differences.
    - For every missing column, use the sprite in the other half as the template. If every complete pair containing that same template shape has empty XOR, fill the missing slot with a full teal copy of the template. Otherwise, union that template's non-empty XOR masks, and fill the missing slot with only those difference pixels in pink.
    - Keep every original blue sprite unchanged.

    Key insight:
    - The lattice tells which sprites should be compared, and the XOR result controls not only whether to fill a gap but also whether the fill is a full copy or only the difference pattern.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    out = analysis["grid"].copy()

    for missing in analysis["missing_columns"]:
        color, fill_mask = _fill_decision(analysis, missing["present_mask"])
        r0 = analysis["top_row"] if missing["missing_half"] == "top" else analysis["bottom_row"]
        c0 = _left_from_double_center(missing["key"], fill_mask.shape[1])
        h, w = fill_mask.shape
        target = out[r0 : r0 + h, c0 : c0 + w]
        assert target.shape == (h, w)
        assert np.all(target == BG)
        _place_mask(out, fill_mask, r0, c0, color)

    return out


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    mask[r, c] = True
    while (r, c) != end:
        dr = 0 if r == end[0] else (1 if end[0] > r else -1)
        dc = 0 if c == end[1] else (1 if end[1] > c else -1)
        if dr != 0 and dc != 0:
            if random.random() < 0.5:
                r += dr
            else:
                c += dc
        elif dr != 0:
            r += dr
        else:
            c += dc
        mask[r, c] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((nr, nc))
    return list(out)


def _random_sprite_mask(height: int, width: int) -> np.ndarray:
    min_area = max(4, (height * width + 2) // 3)
    max_area = height * width - 1
    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        random.shuffle(anchors)
        mask[anchors[0]] = True
        for start, end in zip(anchors, anchors[1:]):
            _carve_path(mask, start, end)
        current_area = int(mask.sum())
        if current_area > max_area:
            continue
        target_area = random.randint(max(min_area, current_area), max_area)
        while int(mask.sum()) < target_area:
            frontier = _frontier(mask)
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True
        if _valid_sprite_mask(mask):
            return mask
    raise RuntimeError("failed to build sprite mask")


def _mutate_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    options = []
    cells = [(r, c) for r in range(h) for c in range(w)]
    for r, c in cells:
        candidate = mask.copy()
        candidate[r, c] = ~candidate[r, c]
        if not np.array_equal(candidate, mask) and _valid_sprite_mask(candidate):
            if 1 <= int((candidate ^ mask).sum()) <= 2:
                options.append(candidate)
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            candidate = mask.copy()
            for r, c in (cells[i], cells[j]):
                candidate[r, c] = ~candidate[r, c]
            if not np.array_equal(candidate, mask) and _valid_sprite_mask(candidate):
                if 1 <= int((candidate ^ mask).sum()) <= 2:
                    options.append(candidate)
    if not options:
        raise RuntimeError("failed to mutate sprite mask")
    return random.choice(options)


def _random_distinct_mask(height: int, width: int, used: list[np.ndarray]) -> np.ndarray:
    for _ in range(200):
        mask = _random_sprite_mask(height, width)
        if all(not np.array_equal(mask, prev) for prev in used):
            return mask
    raise RuntimeError("failed to produce distinct mask")


def _expected_output_from_meta(inp: np.ndarray, meta: dict) -> np.ndarray:
    out = inp.copy()
    copy_slot = meta["copy_slot"]
    diff_slot = meta["diff_slot"]
    _place_mask(out, copy_slot["mask"], copy_slot["row"], copy_slot["col"], COPY)
    _place_mask(out, diff_slot["mask"], diff_slot["row"], diff_slot["col"], DIFF)
    return out


def _validate_generated_case(inp: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or inp.shape[0] % 2 != 0:
        return False

    analysis = _analyze_grid(inp)
    if len(analysis["missing_columns"]) != 2:
        return False
    halves = {missing["missing_half"] for missing in analysis["missing_columns"]}
    if halves != {"top", "bottom"}:
        return False

    copy_matches = [
        pair["xor"]
        for pair in analysis["complete_pairs"]
        if np.array_equal(pair["top_mask"], meta["copy_template"])
        or np.array_equal(pair["bottom_mask"], meta["copy_template"])
    ]
    diff_matches = [
        pair["xor"]
        for pair in analysis["complete_pairs"]
        if np.array_equal(pair["top_mask"], meta["diff_template"])
        or np.array_equal(pair["bottom_mask"], meta["diff_template"])
    ]
    if not copy_matches or not diff_matches:
        return False
    if not all(not xor.any() for xor in copy_matches):
        return False
    if not any(xor.any() for xor in diff_matches):
        return False

    expected = _expected_output_from_meta(inp, meta)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    diff = actual != inp
    if np.any(actual[inp == SPRITE] != SPRITE):
        return False
    if not np.all(np.isin(actual, [BG, SPRITE, DIFF, COPY])):
        return False
    if int(diff.sum()) != int(meta["copy_slot"]["mask"].sum() + meta["diff_slot"]["mask"].sum()):
        return False
    return True


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    sprite_h = random.choice([3, 4])
    sprite_w = random.choice([3, 4])
    col_count = random.choice([4, 5])

    possible_steps = [5]
    if col_count == 4:
        possible_steps.append(6)
    step = random.choice(possible_steps)

    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    width = left_margin + sprite_w + (col_count - 1) * step + right_margin
    if width > 30:
        return None

    half = random.randint(max(sprite_h + 3, 8), 12)
    top_row = random.randint(1, half - sprite_h - 1)
    bottom_row = half + random.randint(1, half - sprite_h - 1)
    height = half * 2
    if height > 30:
        return None

    eq_mask = _random_distinct_mask(sprite_h, sprite_w, [])
    diff_a = _random_distinct_mask(sprite_h, sprite_w, [eq_mask])
    diff_b = _mutate_mask(diff_a)
    while np.array_equal(diff_b, eq_mask):
        diff_b = _mutate_mask(diff_a)

    used = [eq_mask, diff_a, diff_b]
    extra_eq = _random_distinct_mask(sprite_h, sprite_w, used) if col_count == 5 else None

    columns = [
        {"role": "eq_pair", "top": eq_mask, "bottom": eq_mask},
        {"role": "diff_pair", "top": diff_a, "bottom": diff_b},
        {"role": "copy_missing", "top": None, "bottom": eq_mask},
        {"role": "diff_missing", "top": diff_a if random.random() < 0.5 else diff_b, "bottom": None},
    ]
    if col_count == 5:
        columns.append({"role": "extra_eq_pair", "top": extra_eq, "bottom": extra_eq})

    if random.random() < 0.5:
        for col in columns:
            if col["role"] == "copy_missing":
                col["top"], col["bottom"] = col["bottom"], None
            elif col["role"] == "diff_missing":
                col["top"], col["bottom"] = None, col["top"]

    if random.random() < 0.5:
        diff_col = next(col for col in columns if col["role"] == "diff_pair")
        diff_col["top"], diff_col["bottom"] = diff_col["bottom"], diff_col["top"]

    random.shuffle(columns)
    c_positions = [left_margin + idx * step for idx in range(col_count)]

    inp = np.zeros((height, width), dtype=int)
    meta = {"copy_template": eq_mask, "diff_template": diff_a, "copy_slot": None, "diff_slot": None}

    diff_union = _xor_mask(diff_a, diff_b)
    for idx, col in enumerate(columns):
        c0 = c_positions[idx]
        if col["top"] is not None:
            _place_mask(inp, col["top"], top_row, c0, SPRITE)
        if col["bottom"] is not None:
            _place_mask(inp, col["bottom"], bottom_row, c0, SPRITE)

        if col["role"] == "copy_missing":
            missing_top = col["top"] is None
            row = top_row if missing_top else bottom_row
            meta["copy_slot"] = {"row": row, "col": c0, "mask": eq_mask}
        elif col["role"] == "diff_missing":
            missing_top = col["top"] is None
            row = top_row if missing_top else bottom_row
            meta["diff_slot"] = {"row": row, "col": c0, "mask": diff_union}

    if meta["copy_slot"] is None or meta["diff_slot"] is None:
        return None
    if not _validate_generated_case(inp, meta):
        return None
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
