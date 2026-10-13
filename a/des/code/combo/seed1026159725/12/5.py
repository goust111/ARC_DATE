from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


GRAY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _aligned_top_left(src_bbox: tuple[int, int, int, int], dst_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = src_bbox
    src_h = r1 - r0 + 1
    src_w = c1 - c0 + 1
    dst_h, dst_w = map(int, dst_shape)
    return r0 + (src_h - dst_h) // 2, c0 + (src_w - dst_w) // 2


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare each colored object with the unique gray mold, preserving matches and reshaping mismatches.

    Input structure
    - The background is the dominant color.
    - There is exactly one gray 4-connected object acting as the mold.
    - All other non-background objects are monochromatic 4-connected components.

    Rule
    - Extract the mold's translation-normalized binary shape.
    - For each non-gray object, compare its normalized shape with the mold.
    - Matching objects stay unchanged.
    - Non-matching objects are replaced by the mold shape in the same color, centered on the original bbox center.
    - The mold itself remains unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _dominant_color(g)
    comps = _find_components(g, bg)

    molds = [comp for comp in comps if int(comp["color"]) == GRAY]
    assert len(molds) == 1
    mold = molds[0]
    mold_mask = np.asarray(mold["mask"], dtype=bool)
    mold_key = _shape_key(mold_mask)

    out = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    mr0, mc0, mr1, mc1 = mold["bbox"]
    mold_patch = occupied[mr0 : mr1 + 1, mc0 : mc1 + 1]
    assert not np.any(mold_patch[mold_mask])
    mold_patch[mold_mask] = True
    _paint_mask(out, mold_mask, mr0, mc0, GRAY)

    for comp in comps:
        if comp is mold:
            continue

        src_mask = np.asarray(comp["mask"], dtype=bool)
        if _shape_key(src_mask) == mold_key:
            dst_mask = src_mask
            dst_r0, dst_c0 = comp["bbox"][0], comp["bbox"][1]
        else:
            dst_mask = mold_mask
            dst_r0, dst_c0 = _aligned_top_left(comp["bbox"], mold_mask.shape)

        dst_h, dst_w = dst_mask.shape
        assert 0 <= dst_r0 and 0 <= dst_c0
        assert dst_r0 + dst_h <= h and dst_c0 + dst_w <= w

        occ_patch = occupied[dst_r0 : dst_r0 + dst_h, dst_c0 : dst_c0 + dst_w]
        assert not np.any(occ_patch[dst_mask])
        occ_patch[dst_mask] = True
        _paint_mask(out, dst_mask, dst_r0, dst_c0, int(comp["color"]))

    return out


# --------------- generator ---------------

GRID_RANGE = (12, 18)
NUM_OBJECTS_RANGE = (2, 5)
MOLD_SIDE_RANGE = (2, 5)
MAX_OBJECT_SIDE = 6
MAX_RETRIES = 400
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
OBJ_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(height: int, width: int, min_area: int | None = None) -> np.ndarray | None:
    area_min = min_area if min_area is not None else max(3, (height * width + 1) // 2)
    area_max = height * width
    if area_min > area_max:
        return None

    for _ in range(120):
        target = random.randint(area_min, area_max)
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        cells = {start}
        frontier = set()
        for dr, dc in DIR4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                frontier.add((nr, nc))

        while len(cells) < target and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            if (rr, cc) in cells:
                continue
            cells.add((rr, cc))
            mask[rr, cc] = True
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target:
            continue
        if not _spans_bbox(mask):
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _all_same_parity_dims(height: int, width: int) -> list[tuple[int, int]]:
    out = []
    for h in range(2, MAX_OBJECT_SIDE + 1):
        for w in range(2, MAX_OBJECT_SIDE + 1):
            if h % 2 == height % 2 and w % 2 == width % 2:
                out.append((h, w))
    return out


def _random_mismatch_mask(mold_mask: np.ndarray) -> np.ndarray | None:
    mold_key = _shape_key(mold_mask)
    mh, mw = mold_mask.shape
    dims = _all_same_parity_dims(mh, mw)
    random.shuffle(dims)
    for h, w in dims:
        for _ in range(40):
            min_area = max(3, (h + w) // 2)
            mask = _random_connected_mask(h, w, min_area=min_area)
            if mask is None:
                continue
            if _shape_key(mask) != mold_key:
                return mask
    return None


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in DIR8:
        rs_src = slice(max(0, -dr), min(mask.shape[0], mask.shape[0] - dr))
        cs_src = slice(max(0, -dc), min(mask.shape[1], mask.shape[1] - dc))
        rs_dst = slice(max(0, dr), min(mask.shape[0], mask.shape[0] + dr))
        cs_dst = slice(max(0, dc), min(mask.shape[1], mask.shape[1] + dc))
        out[rs_dst, cs_dst] |= mask[rs_src, cs_src]
    return out


def _can_place(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > occ.shape[0] or c0 + w > occ.shape[1]:
        return False
    return not np.any(occ[r0 : r0 + h, c0 : c0 + w] & mask)


def _mark_with_margin(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = mask.shape
    occ[r0 : r0 + h, c0 : c0 + w] |= _dilate(mask)


def _paint_object(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    _paint_mask(grid, mask, r0, c0, color)


def _status_pattern(count: int) -> list[bool]:
    matches = count // 2
    if count % 2 == 1 and random.random() < 0.5:
        matches += 1
    matches = min(count - 1, max(1, matches))
    states = [True] * matches + [False] * (count - matches)
    random.shuffle(states)
    return states


def _build_expected(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    bg = random.choice(BG_CHOICES)

    mold_h = random.randint(*MOLD_SIDE_RANGE)
    mold_w = random.randint(*MOLD_SIDE_RANGE)
    mold_mask = _random_connected_mask(mold_h, mold_w)
    if mold_mask is None:
        return None

    num_objects = random.randint(*NUM_OBJECTS_RANGE)
    states = _status_pattern(num_objects)
    colors = random.sample([c for c in OBJ_CHOICES if c != bg and c != GRAY], num_objects)

    input_occ = np.zeros((h, w), dtype=bool)
    output_occ = np.zeros((h, w), dtype=bool)
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    mold_positions = [
        (r0, c0)
        for r0 in range(h - mold_h + 1)
        for c0 in range(w - mold_w + 1)
    ]
    random.shuffle(mold_positions)
    mold_pos = None
    for r0, c0 in mold_positions:
        if _can_place(input_occ, mold_mask, r0, c0) and _can_place(output_occ, mold_mask, r0, c0):
            mold_pos = (r0, c0)
            break
    if mold_pos is None:
        return None

    mr0, mc0 = mold_pos
    _paint_object(inp, mold_mask, mr0, mc0, GRAY)
    _paint_object(out, mold_mask, mr0, mc0, GRAY)
    _mark_with_margin(input_occ, mold_mask, mr0, mc0)
    _mark_with_margin(output_occ, mold_mask, mr0, mc0)

    for color, is_match in zip(colors, states):
        if is_match:
            src_mask = mold_mask.copy()
            dst_mask = src_mask
        else:
            src_mask = _random_mismatch_mask(mold_mask)
            if src_mask is None:
                return None
            dst_mask = mold_mask

        sh, sw = src_mask.shape
        candidates = [
            (r0, c0)
            for r0 in range(h - sh + 1)
            for c0 in range(w - sw + 1)
        ]
        random.shuffle(candidates)

        placed = False
        for sr0, sc0 in candidates:
            dr0, dc0 = (sr0, sc0) if is_match else _aligned_top_left((sr0, sc0, sr0 + sh - 1, sc0 + sw - 1), mold_mask.shape)
            if not _can_place(input_occ, src_mask, sr0, sc0):
                continue
            if not _can_place(output_occ, dst_mask, dr0, dc0):
                continue
            _paint_object(inp, src_mask, sr0, sc0, color)
            _paint_object(out, dst_mask, dr0, dc0, color)
            _mark_with_margin(input_occ, src_mask, sr0, sc0)
            _mark_with_margin(output_occ, dst_mask, dr0, dc0)
            placed = True
            break

        if not placed:
            return None

    if Counter(inp.ravel()).most_common(1)[0][0] != bg:
        return None
    if Counter(out.ravel()).most_common(1)[0][0] != bg:
        return None
    if np.array_equal(inp, out):
        return None

    try:
        solved = _build_expected(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
