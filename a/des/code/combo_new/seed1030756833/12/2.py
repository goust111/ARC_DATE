import random
from collections import deque
from functools import lru_cache

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
RULE_COUNT_CHOICES = (2, 3)
LEGEND_WIDTH = 3
OBJECT_BORDER = 2
OBJECT_PADDING = 1
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 250
SHAPE_RETRIES = 600
PAIR_DIMS = ((3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors(r: int, c: int, connectivity: int):
    yield from _neighbors4(r, c)
    if connectivity == 8:
        yield r - 1, c - 1
        yield r - 1, c + 1
        yield r + 1, c - 1
        yield r + 1, c + 1


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors(r, c, connectivity):
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    return bool(mask.any()) and len(_components(mask, connectivity=connectivity)) == 1


def _spans_all_edges(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _read_legend(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    H, W = grid.shape
    candidates = []
    corners = (
        ("TL", range(H), 0),
        ("TR", range(H), W - LEGEND_WIDTH),
        ("BL", range(H - 1, -1, -1), 0),
        ("BR", range(H - 1, -1, -1), W - LEGEND_WIDTH),
    )

    for name, row_iter, c0 in corners:
        if c0 < 0:
            continue
        rows = []
        for r in row_iter:
            cells = [int(x) for x in grid[int(r), c0 : c0 + LEGEND_WIDTH]]
            if all(x != BLACK for x in cells) and len(set(cells)) == LEGEND_WIDTH:
                rows.append(int(r))
            else:
                break
        if rows:
            candidates.append((len(rows), name, rows, c0))

    assert candidates, "expected a corner color legend"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _count, _name, rows, c0 = candidates[0]

    rules: dict[tuple[int, int], int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r in sorted(rows):
        a, b, result = [int(x) for x in grid[r, c0 : c0 + LEGEND_WIDTH]]
        rules[(a, b)] = result
        rules[(b, a)] = result
        legend_mask[r, c0 : c0 + LEGEND_WIDTH] = True
    return rules, legend_mask


def _extract_fragments(grid: np.ndarray, legend_mask: np.ndarray) -> list[dict]:
    obj_mask = (grid != BLACK) & (~legend_mask)
    fragments = []
    for coords in _components(obj_mask, connectivity=8):
        vals = [int(grid[r, c]) for r, c in coords]
        if any(v != vals[0] for v in vals):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[int(r) - r0, int(c) - c0] = True
        fragments.append(
            {
                "color": vals[0],
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
            }
        )
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary scattered color fragments into solid mixed-color rectangles.

    Input structure:
    - The background is black.
    - A corner legend contains horizontal three-pixel rows: left color, right color, result color.
    - Away from the legend are monochrome connected fragments on the black background.

    Rule:
    - Read every legend row as a color rule A+B -> C.
    - Find fragment pairs with different colors, equal bbox size, and complementary masks inside that bbox.
    - Replace each matched pair by a filled rectangle of the legend result color at the midpoint of the two bbox top-left positions.
    - Clear the legend and any fragments that do not belong to a complementary color-rule pair.

    Key insight:
    - The corner triplets are a color-mixing table, while the scattered pieces pair by shape complement before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    rules, legend_mask = _read_legend(g)
    fragments = _extract_fragments(g, legend_mask)

    candidates = []
    for i in range(len(fragments)):
        a = fragments[i]
        ah, aw = a["mask"].shape
        ar0, ac0, _ar1, _ac1 = a["bbox"]
        for j in range(i + 1, len(fragments)):
            b = fragments[j]
            if int(a["color"]) == int(b["color"]):
                continue
            result = rules.get((int(a["color"]), int(b["color"])))
            if result is None:
                continue
            if b["mask"].shape != (ah, aw):
                continue
            if not np.array_equal(a["mask"], ~b["mask"]):
                continue
            br0, bc0, _br1, _bc1 = b["bbox"]
            mid = ((int(ar0) + int(br0)) // 2, (int(ac0) + int(bc0)) // 2)
            candidates.append((mid, i, j, int(result)))

    out = np.zeros_like(g)
    used: set[int] = set()
    for _mid, i, j, result in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        a = fragments[i]
        b = fragments[j]
        h, w = a["mask"].shape
        ar0, ac0, _ar1, _ac1 = a["bbox"]
        br0, bc0, _br1, _bc1 = b["bbox"]
        r0 = (int(ar0) + int(br0)) // 2
        c0 = (int(ac0) + int(bc0)) // 2
        out[r0 : r0 + h, c0 : c0 + w] = int(result)
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    r = random.randrange(h)
    c = random.randrange(w)
    mask[r, c] = True

    while int(mask.sum()) < area:
        frontier = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _valid_pair_mask(mask: np.ndarray) -> bool:
    other = ~mask
    return (
        _is_connected(mask, connectivity=8)
        and _is_connected(other, connectivity=8)
        and _spans_all_edges(mask)
        and _spans_all_edges(other)
    )


@lru_cache(maxsize=None)
def _valid_pair_masks(h: int, w: int) -> tuple[np.ndarray, ...]:
    total = h * w
    min_area = max(4, total // 3)
    max_area = min(total - 4, (2 * total) // 3)
    masks = []
    for bits in range(1, (1 << total) - 1):
        area = int(bits.bit_count())
        if area < min_area or area > max_area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for idx in range(total):
            if bits & (1 << idx):
                mask[idx // w, idx % w] = True
        if _valid_pair_mask(mask):
            masks.append(mask)
    return tuple(masks)


def _make_pair_mask() -> np.ndarray:
    dims = [dims for dims in PAIR_DIMS if _valid_pair_masks(*dims)]
    random.shuffle(dims)
    for h, w in dims:
        masks = _valid_pair_masks(h, w)
        if masks:
            return random.choice(masks).copy()
    raise RuntimeError("failed to make complementary connected masks")


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_distractor_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        area = random.randint(2, max(2, min(h * w - 1, 8)))
        mask = _random_connected_mask(h, w, area)
        if mask is not None:
            return _crop_mask(mask)
    raise RuntimeError("failed to make distractor mask")


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    H, W = occupied.shape
    r_start = max(0, r0 - pad)
    c_start = max(0, c0 - pad)
    r_stop = min(H, r0 + h + pad)
    c_stop = min(W, c0 + w + pad)
    occupied[r_start:r_stop, c_start:c_stop] = True


def _can_place(occupied: np.ndarray, h: int, w: int, r0: int, c0: int, pad: int) -> bool:
    H, W = occupied.shape
    r_start = max(0, r0 - pad)
    c_start = max(0, c0 - pad)
    r_stop = min(H, r0 + h + pad)
    c_stop = min(W, c0 + w + pad)
    return not bool(occupied[r_start:r_stop, c_start:c_stop].any())


def _random_location(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    r_min = OBJECT_BORDER
    c_min = OBJECT_BORDER
    r_max = H - h - OBJECT_BORDER
    c_max = W - w - OBJECT_BORDER
    if r_min > r_max or c_min > c_max:
        return None
    choices = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    random.shuffle(choices)
    for r, c in choices[:PLACEMENT_RETRIES]:
        if _can_place(occupied, h, w, r, c, OBJECT_PADDING):
            return int(r), int(c)
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = int(color)


def _legend_origin(H: int, W: int, n_rules: int, corner: str) -> tuple[int, int]:
    r0 = 0 if corner[0] == "T" else H - n_rules
    c0 = 0 if corner[1] == "L" else W - LEGEND_WIDTH
    return r0, c0


def _paint_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    rules: list[tuple[int, int, int]],
    corner: str,
) -> None:
    H, W = grid.shape
    r0, c0 = _legend_origin(H, W, len(rules), corner)
    for i, (a, b, result) in enumerate(rules):
        grid[r0 + i, c0 : c0 + LEGEND_WIDTH] = [int(a), int(b), int(result)]
    _reserve(occupied, r0, c0, len(rules), LEGEND_WIDTH, 1)


def _build_expected(grid_shape: tuple[int, int], pair_infos: list[dict]) -> np.ndarray | None:
    H, W = grid_shape
    out = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    for info in pair_infos:
        (r1, c1), (r2, c2) = info["positions"]
        h, w = info["shape"]
        r0 = (int(r1) + int(r2)) // 2
        c0 = (int(c1) + int(c2)) // 2
        if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
            return None
        if occupied[r0 : r0 + h, c0 : c0 + w].any():
            return None
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        out[r0 : r0 + h, c0 : c0 + w] = int(info["result"])
    return out


def _sample_rules(n_rules: int) -> list[tuple[int, int, int]]:
    colors = random.sample(COLORS, 3 * n_rules)
    rules = []
    for i in range(n_rules):
        a, b, result = colors[3 * i : 3 * i + 3]
        if random.random() < 0.5:
            a, b = b, a
        rules.append((int(a), int(b), int(result)))
    random.shuffle(rules)
    return rules


def _make_pieces(rules: list[tuple[int, int, int]]) -> tuple[list[dict], list[dict]]:
    pieces = []
    pair_infos = []
    for pair_id, (a, b, result) in enumerate(rules):
        mask = _make_pair_mask()
        other = ~mask
        if random.random() < 0.5:
            mask, other = other, mask
        h, w = mask.shape
        pair_infos.append(
            {
                "pair_id": pair_id,
                "result": int(result),
                "shape": (h, w),
                "positions": [],
            }
        )
        pieces.append({"pair_id": pair_id, "color": int(a), "mask": mask})
        pieces.append({"pair_id": pair_id, "color": int(b), "mask": other})

    if random.random() < 0.5:
        result_colors = [int(rule[2]) for rule in rules]
        pieces.append(
            {
                "pair_id": None,
                "color": random.choice(result_colors),
                "mask": _make_distractor_mask(),
            }
        )
    random.shuffle(pieces)
    return pieces, pair_infos


def _place_pieces(
    grid: np.ndarray,
    occupied: np.ndarray,
    pieces: list[dict],
    pair_infos: list[dict],
) -> bool:
    pair_by_id = {int(info["pair_id"]): info for info in pair_infos}
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["mask"].size)
    for idx in order:
        piece = pieces[idx]
        mask = piece["mask"]
        h, w = mask.shape
        loc = _random_location(occupied, h, w)
        if loc is None:
            return False
        r0, c0 = loc
        _paint_mask(grid, mask, int(piece["color"]), r0, c0)
        _reserve(occupied, r0, c0, h, w, OBJECT_PADDING)
        if piece["pair_id"] is not None:
            pair_by_id[int(piece["pair_id"])]["positions"].append((r0, c0))
    return all(len(info["positions"]) == 2 for info in pair_infos)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n_rules = random.choice(RULE_COUNT_CHOICES)
    corner = random.choice(("TL", "TR", "BL", "BR"))
    rules = _sample_rules(n_rules)

    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    _paint_legend(grid, occupied, rules, corner)

    pieces, pair_infos = _make_pieces(rules)
    if not _place_pieces(grid, occupied, pieces, pair_infos):
        return None

    expected = _build_expected(grid.shape, pair_infos)
    if expected is None:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment input")
