from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
BASE_MASK_SHAPES = ((3, 3), (3, 4), (4, 4), (3, 5))
PAIR_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
PLACEMENT_GAP = 1
GEN_RETRIES = 400

_MASK_LIBRARY: list[dict[str, object]] | None = None


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        return (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    raise ValueError(f"unsupported connectivity: {connectivity}")


def _connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    nbrs = _neighbors(connectivity)
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False
    h, w = mask.shape
    return (
        0 in rows
        and h - 1 in rows
        and 0 in cols
        and w - 1 in cols
    )


def _mask_key(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _canonical_mask_key(mask: np.ndarray) -> bytes:
    comp = ~mask
    a = _mask_key(mask)
    b = _mask_key(comp)
    return a if a < b else b


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 8) -> list[dict[str, object]]:
    h, w = grid.shape
    nbrs = _neighbors(connectivity)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comp = {
                "color": color,
                "coords": arr,
                "bbox": _bbox(arr),
            }
            comp["mask"] = _component_mask(arr)
            comps.append(comp)
    return comps


def _extract_bar(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    col = w - 1
    rows = []
    r = h - 1
    while r >= 0 and int(grid[r, col]) != bg:
        rows.append(r)
        r -= 1
    rows.reverse()
    assert len(rows) >= 2, "expected a bottom-aligned vertical color bar"
    colors = [int(grid[row, col]) for row in rows]
    assert len(set(colors)) == len(colors), "bar colors must be unique"
    mask = np.zeros_like(grid, dtype=bool)
    mask[rows, col] = True
    return colors, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary colored fragments, complete each pair into a full rectangle,
    and recolor that rectangle using the lower color from the edge bar.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    bar_colors, bar_mask = _extract_bar(g, bg)
    rank = {color: idx for idx, color in enumerate(bar_colors)}

    work = g.copy()
    work[bar_mask] = bg
    comps = _find_components(work, bg=bg, connectivity=8)
    assert comps and len(comps) % 2 == 0
    assert {int(comp["color"]) for comp in comps} == set(bar_colors)

    by_key: dict[tuple[int, int, bytes], list[int]] = {}
    for idx, comp in enumerate(comps):
        mask = np.asarray(comp["mask"], dtype=bool)
        h, w = mask.shape
        by_key.setdefault((h, w, _mask_key(mask)), []).append(idx)

    used: set[int] = set()
    out = np.full_like(g, bg)
    for idx, comp in enumerate(comps):
        if idx in used:
            continue
        mask = np.asarray(comp["mask"], dtype=bool)
        h, w = mask.shape
        need = (h, w, _mask_key(~mask))
        candidates = [j for j in by_key.get(need, []) if j not in used and j != idx]
        assert candidates, "every fragment must have a unique complementary partner"
        partner_idx = candidates[0]
        partner = comps[partner_idx]
        used.add(idx)
        used.add(partner_idx)

        a_bbox = tuple(map(int, comp["bbox"]))
        b_bbox = tuple(map(int, partner["bbox"]))
        if (a_bbox[0], a_bbox[1]) <= (b_bbox[0], b_bbox[1]):
            anchor = comp
        else:
            anchor = partner

        color_a = int(comp["color"])
        color_b = int(partner["color"])
        paint = color_a if rank[color_a] > rank[color_b] else color_b

        r0, c0, _r1, _c1 = map(int, anchor["bbox"])
        out[r0 : r0 + h, c0 : c0 + w] = paint

    return out


def _enumerate_valid_masks(shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = shape
    n = h * w
    full = (1 << n) - 1
    seen: set[int] = set()
    masks: list[np.ndarray] = []
    for bits in range(1, full):
        if bits in seen:
            continue
        comp_bits = full ^ bits
        seen.add(bits)
        seen.add(comp_bits)

        mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool).reshape(h, w)
        comp = ~mask
        if mask.sum() < 3 or comp.sum() < 3:
            continue
        if not _touches_all_sides(mask) or not _touches_all_sides(comp):
            continue
        if _connected(mask, connectivity=8) and _connected(comp, connectivity=8):
            masks.append(mask)
    return masks


def _mask_library() -> list[dict[str, object]]:
    global _MASK_LIBRARY
    if _MASK_LIBRARY is not None:
        return _MASK_LIBRARY

    entries: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}
    for base_shape in BASE_MASK_SHAPES:
        for base_mask in _enumerate_valid_masks(base_shape):
            variants = [base_mask]
            if base_shape[0] != base_shape[1]:
                variants.append(base_mask.T.copy())
            for mask in variants:
                shape = tuple(map(int, mask.shape))
                key = (shape, _canonical_mask_key(mask))
                if key not in entries:
                    entries[key] = {"shape": shape, "mask": mask.copy(), "key": key}

    _MASK_LIBRARY = list(entries.values())
    assert len(_MASK_LIBRARY) >= 20
    return _MASK_LIBRARY


def _candidate_positions(
    occ: np.ndarray,
    box_h: int,
    box_w: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    max_r = row_hi - box_h + 1
    max_c = col_hi - box_w + 1
    if max_r < row_lo or max_c < col_lo:
        return positions
    for r0 in range(row_lo, max_r + 1):
        for c0 in range(col_lo, max_c + 1):
            rr0 = max(0, r0 - PLACEMENT_GAP)
            rr1 = min(h, r0 + box_h + PLACEMENT_GAP)
            cc0 = max(0, c0 - PLACEMENT_GAP)
            cc1 = min(w, c0 + box_w + PLACEMENT_GAP)
            if not occ[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _mark_occ(occ: np.ndarray, r0: int, c0: int, box_h: int, box_w: int) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - PLACEMENT_GAP)
    rr1 = min(h, r0 + box_h + PLACEMENT_GAP)
    cc0 = max(0, c0 - PLACEMENT_GAP)
    cc1 = min(w, c0 + box_w + PLACEMENT_GAP)
    occ[rr0:rr1, cc0:cc1] = True


def _paint_fragment(grid: np.ndarray, color: int, mask: np.ndarray, origin: tuple[int, int]) -> None:
    r0, c0 = origin
    h, w = mask.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[mask] = color


def _pair_edges(colors: list[int]) -> list[tuple[int, int]]:
    cycle = colors[:]
    random.shuffle(cycle)
    return list(zip(cycle, cycle[1:] + cycle[:1]))


def _choose_pair_shapes(pair_count: int) -> list[dict[str, object]]:
    library = _mask_library()
    return random.sample(library, pair_count)


def _place_pairs(
    pair_specs: list[dict[str, object]],
    height: int,
    width: int,
) -> list[dict[str, object]] | None:
    usable_w = width - 2
    top_hi = height // 2 - 1
    bottom_lo = height // 2 + 1
    occ = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object] | None] = [None] * len(pair_specs)

    order = sorted(
        range(len(pair_specs)),
        key=lambda idx: -int(np.prod(pair_specs[idx]["shape"])),
    )
    for idx in order:
        spec = pair_specs[idx]
        box_h, box_w = map(int, spec["shape"])
        top_positions = _candidate_positions(occ, box_h, box_w, 0, top_hi, 0, usable_w - 1)
        if not top_positions:
            return None
        top_origin = random.choice(top_positions)
        _mark_occ(occ, top_origin[0], top_origin[1], box_h, box_w)

        bottom_positions = _candidate_positions(
            occ,
            box_h,
            box_w,
            max(bottom_lo, top_origin[0] + 1),
            height - 1,
            0,
            usable_w - 1,
        )
        if not bottom_positions:
            return None
        bottom_origin = random.choice(bottom_positions)
        _mark_occ(occ, bottom_origin[0], bottom_origin[1], box_h, box_w)

        placed[idx] = {
            **spec,
            "top_origin": top_origin,
            "bottom_origin": bottom_origin,
        }

    assert all(item is not None for item in placed)
    return [item for item in placed if item is not None]


def _build_scene(
    pair_specs: list[dict[str, object]],
    bar_colors: list[int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    rank = {color: idx for idx, color in enumerate(bar_colors)}

    for offset, color in enumerate(reversed(bar_colors), start=1):
        grid[height - offset, width - 1] = color

    for spec in pair_specs:
        top_mask = np.asarray(spec["top_mask"], dtype=bool)
        bottom_mask = np.asarray(spec["bottom_mask"], dtype=bool)
        top_color = int(spec["top_color"])
        bottom_color = int(spec["bottom_color"])
        top_origin = tuple(map(int, spec["top_origin"]))
        bottom_origin = tuple(map(int, spec["bottom_origin"]))
        _paint_fragment(grid, top_color, top_mask, top_origin)
        _paint_fragment(grid, bottom_color, bottom_mask, bottom_origin)

        paint = top_color if rank[top_color] > rank[bottom_color] else bottom_color
        box_h, box_w = top_mask.shape
        r0, c0 = top_origin
        out[r0 : r0 + box_h, c0 : c0 + box_w] = paint

    return grid, out


def _validate_fragment_color_counts(grid: np.ndarray, bar_colors: list[int]) -> bool:
    bg = BG
    bar_colors_seen, bar_mask = _extract_bar(grid, bg)
    if bar_colors_seen != bar_colors:
        return False
    work = grid.copy()
    work[bar_mask] = bg
    comps = _find_components(work, bg=bg, connectivity=8)
    counts: dict[int, int] = {}
    for comp in comps:
        counts[int(comp["color"])] = counts.get(int(comp["color"]), 0) + 1
    return counts == {color: 2 for color in bar_colors}


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, pair_count)
    bar_colors = colors[:]
    random.shuffle(bar_colors)
    pair_edges = _pair_edges(colors)
    chosen_shapes = _choose_pair_shapes(pair_count)

    pair_specs: list[dict[str, object]] = []
    for (color_a, color_b), shape_info in zip(pair_edges, chosen_shapes):
        mask = np.asarray(shape_info["mask"], dtype=bool).copy()
        if random.random() < 0.5:
            mask = ~mask
        if random.random() < 0.5:
            top_color, bottom_color = color_a, color_b
        else:
            top_color, bottom_color = color_b, color_a
        pair_specs.append(
            {
                "shape": tuple(map(int, mask.shape)),
                "top_mask": mask,
                "bottom_mask": ~mask,
                "top_color": top_color,
                "bottom_color": bottom_color,
            }
        )

    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    placed = _place_pairs(pair_specs, height, width)
    if placed is None:
        return None

    grid, expected = _build_scene(placed, bar_colors, height, width)
    if not _validate_fragment_color_counts(grid, bar_colors):
        return None
    if _get_bg(grid) != BG:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
