from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
PAIR_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (16, 26)
PAIR_BBOX_H_RANGE = (3, 6)
PAIR_BBOX_W_RANGE = (3, 6)
GEN_RETRIES = 400
PAIR_SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None, None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _is_connected_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _bbox_is_full(mask: np.ndarray) -> bool:
    cropped, _ = _crop_mask(mask)
    return cropped is not None and cropped.shape == mask.shape


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            colors = set()
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
                colors.add(int(grid[rr, cc]))

            comp = {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "area": len(coords),
                "colors": colors,
                "color": next(iter(colors)) if len(colors) == 1 else None,
            }
            comps.append(comp)
    return comps


def _find_lookup_component(components: list[dict]) -> dict:
    candidates = [
        comp
        for comp in components
        if len(comp["colors"]) > 1
        and bool(comp["mask"].all())
        and int(comp["mask"].shape[1]) == 3
    ]
    assert len(candidates) == 1, "expected a unique 3-column multicolor lookup table"
    return candidates[0]


def _read_lookup(grid: np.ndarray, comp: dict, bg: int) -> dict[frozenset[int], int]:
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[1] == 3
    assert np.all(patch != bg)

    lookup: dict[frozenset[int], int] = {}
    for row in patch:
        a, b, out = map(int, row)
        assert a != b and out not in (a, b)
        key = frozenset((a, b))
        assert key not in lookup
        lookup[key] = out
    return lookup


def _find_pairs(fragments: list[dict], lookup: dict[frozenset[int], int]) -> list[tuple[dict, dict]]:
    by_shape: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, frag in enumerate(fragments):
        by_shape.setdefault(_mask_key(frag["mask"]), []).append(idx)

    candidates: list[tuple[int, int]] = []
    involvement: list[list[int]] = [[] for _ in fragments]

    for i, anchor in enumerate(fragments):
        hole_mask = ~anchor["mask"]
        if not bool(hole_mask.any()) or not _is_connected_mask(hole_mask):
            continue

        hole_crop, _ = _crop_mask(hole_mask)
        assert hole_crop is not None
        key = _mask_key(hole_crop)

        for j in by_shape.get(key, []):
            if i == j:
                continue
            filler = fragments[j]
            if anchor["color"] == filler["color"]:
                continue
            if frozenset((int(anchor["color"]), int(filler["color"]))) not in lookup:
                continue
            if (anchor["bbox"][0], anchor["bbox"][1]) >= (filler["bbox"][0], filler["bbox"][1]):
                continue
            candidates.append((i, j))

    for idx, (i, j) in enumerate(candidates):
        involvement[i].append(idx)
        involvement[j].append(idx)

    assert candidates, "expected at least one fragment pair"
    assert len(candidates) * 2 == len(fragments), "pairing must cover all fragments exactly once"
    assert all(len(v) == 1 for v in involvement), "pairing must be unambiguous"

    return [(fragments[i], fragments[j]) for i, j in candidates]


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble complementary fragments into rectangles and recolor via a lookup table."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    comps = _find_components(g, bg)
    lookup_comp = _find_lookup_component(comps)
    lookup = _read_lookup(g, lookup_comp, bg)

    fragments = [comp for comp in comps if comp is not lookup_comp]
    assert fragments and all(len(comp["colors"]) == 1 for comp in fragments)

    pairs = _find_pairs(fragments, lookup)
    out = np.full_like(g, bg)
    for anchor, filler in pairs:
        key = frozenset((int(anchor["color"]), int(filler["color"])))
        fill_color = int(lookup[key])
        r0, c0, r1, c1 = map(int, anchor["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    min_area = max(2, area // 4)
    max_area = min(area - 2, (3 * area) // 5)
    target = random.randint(min_area, max_area)

    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    frontier = set()
    for nr, nc in _neighbors4(*seed):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_pair_shape(used_keys: set[tuple[tuple[int, int], bytes]]) -> dict | None:
    for _ in range(PAIR_SHAPE_RETRIES):
        h = random.randint(*PAIR_BBOX_H_RANGE)
        w = random.randint(*PAIR_BBOX_W_RANGE)

        filler_full = _random_connected_mask(h, w)
        anchor_mask = ~filler_full
        if not _is_connected_mask(filler_full):
            continue
        if not _is_connected_mask(anchor_mask):
            continue
        if not _bbox_is_full(anchor_mask):
            continue

        filler_mask, offset = _crop_mask(filler_full)
        assert filler_mask is not None and offset is not None
        if filler_mask.shape == (h, w):
            continue
        if int(filler_mask.sum()) < 2:
            continue

        key = _mask_key(filler_mask)
        if key in used_keys:
            continue

        return {
            "anchor_mask": anchor_mask,
            "filler_mask": filler_mask,
            "hole_offset": offset,
            "anchor_shape": (h, w),
            "shape_key": key,
        }
    return None


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _draw_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    if H < pair_count + 6 or W < 10:
        return None

    input_colors = random.sample(PALETTE, 2 * pair_count)
    random.shuffle(input_colors)

    pair_infos = []
    used_shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    for idx in range(pair_count):
        shape = _sample_pair_shape(used_shape_keys)
        if shape is None:
            return None
        used_shape_keys.add(shape["shape_key"])

        a, b = input_colors[2 * idx : 2 * idx + 2]
        result = random.choice([c for c in PALETTE if c not in (a, b)])
        if random.random() < 0.5:
            anchor_color, filler_color = a, b
        else:
            anchor_color, filler_color = b, a

        pair_infos.append(
            {
                **shape,
                "anchor_color": anchor_color,
                "filler_color": filler_color,
                "result_color": result,
            }
        )

    pair_infos.sort(
        key=lambda info: info["anchor_mask"].shape[0] * info["anchor_mask"].shape[1],
        reverse=True,
    )

    reserved = np.zeros((H, W), dtype=bool)
    table_h = pair_count
    table_w = 3
    table_r0 = H - table_h
    table_c0 = W - table_w
    _reserve(reserved, table_r0, table_c0, table_h, table_w, pad=1)

    placed = []
    for info in pair_infos:
        ah, aw = info["anchor_mask"].shape
        fh, fw = info["filler_mask"].shape

        anchor_pos = None
        for _ in range(PLACEMENT_RETRIES):
            ar = random.randint(0, H - ah)
            ac = random.randint(0, W - aw)
            if _can_place(reserved, ar, ac, ah, aw, pad=1):
                anchor_pos = (ar, ac)
                _reserve(reserved, ar, ac, ah, aw, pad=1)
                break
        if anchor_pos is None:
            return None

        filler_pos = None
        for _ in range(PLACEMENT_RETRIES):
            fr = random.randint(0, H - fh)
            fc = random.randint(0, W - fw)
            if (fr, fc) <= anchor_pos:
                continue
            if _can_place(reserved, fr, fc, fh, fw, pad=1):
                filler_pos = (fr, fc)
                _reserve(reserved, fr, fc, fh, fw, pad=1)
                break
        if filler_pos is None:
            return None

        placed.append({**info, "anchor_pos": anchor_pos, "filler_pos": filler_pos})

    grid = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)

    for info in placed:
        ar, ac = info["anchor_pos"]
        fr, fc = info["filler_pos"]
        _draw_mask(grid, ar, ac, info["anchor_mask"], int(info["anchor_color"]))
        _draw_mask(grid, fr, fc, info["filler_mask"], int(info["filler_color"]))
        ah, aw = info["anchor_mask"].shape
        expected[ar : ar + ah, ac : ac + aw] = int(info["result_color"])

    row_order = list(range(pair_count))
    random.shuffle(row_order)
    for row_idx, pair_idx in enumerate(row_order):
        info = placed[pair_idx]
        left = int(info["anchor_color"])
        mid = int(info["filler_color"])
        if random.random() < 0.5:
            left, mid = mid, left
        grid[table_r0 + row_idx, table_c0 : table_c0 + 3] = [left, mid, int(info["result_color"])]

    if _get_bg(grid) != BG:
        return None
    if int(np.count_nonzero(grid)) * 2 >= int(grid.size):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment sample")
