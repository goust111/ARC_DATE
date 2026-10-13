import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

PAIR_COUNTS = [2, 3, 4]
PAIR_WEIGHTS = [3, 4, 3]
GRID_MIN = 20
GRID_MAX = 30
GEN_RETRIES = 400
REGION_MASK_RETRIES = 80

BASE_DIMS = [
    (h, w)
    for h in range(6, 11)
    for w in range(6, 11)
    if 24 <= h * w <= 60
]

PATTERNS_WITH_CENTER = [
    [(-1, 0), (0, 0), (1, 0)],
    [(0, -1), (0, 0), (0, 1)],
    [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
    [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)],
    [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)],
    [(-1, 0), (0, 0), (1, 0), (1, -1), (1, 1)],
    [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, -1), (1, 0), (1, 1)],
]

PATTERNS_WITHOUT_CENTER = [
    [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)],
    [(-1, 1), (0, 1), (1, -1), (1, 0), (1, 1)],
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)],
    [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)],
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)],
    [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1)],
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crosshair_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
    if len(coords) != 5:
        return None
    colors = Counter(int(grid[r, c]) for r, c in coords)
    if colors.get(GREY, 0) != 4 or len(colors) != 2:
        return None
    centers = [(r, c) for r, c in coords if int(grid[r, c]) != GREY]
    if len(centers) != 1:
        return None
    center = centers[0]
    r, c = center
    if set(coords) != {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}:
        return None
    return int(grid[r, c]), center


def _extract_stamp(
    grid: np.ndarray, coords: list[tuple[int, int]]
) -> tuple[int, int, list[tuple[int, int]]]:
    color_counts = Counter(int(grid[r, c]) for r, c in coords)
    main_color, main_count = max(color_counts.items(), key=lambda item: item[1])
    assert sum(count == main_count for count in color_counts.values()) == 1

    stamp = [(r, c) for r, c in coords if int(grid[r, c]) != main_color]
    assert 3 <= len(stamp) <= 7
    stamp_colors = {int(grid[r, c]) for r, c in stamp}
    assert len(stamp_colors) == 1
    stamp_color = next(iter(stamp_colors))

    assert len(_components(_coords_to_mask(stamp))) == 1
    r0, c0, r1, c1 = _bbox(stamp)
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0
    rc = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    offsets = [(r - rc, c - cc) for r, c in stamp]
    return main_color, stamp_color, offsets


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each hidden interior stamp and reprint it at the matching crosshair.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    components = _components(g != BG)
    crosshairs: dict[int, tuple[int, int]] = {}
    regions: list[list[tuple[int, int]]] = []

    for coords in components:
        info = _crosshair_info(g, coords)
        if info is None:
            regions.append(coords)
            continue
        color, center = info
        assert color not in crosshairs
        crosshairs[color] = center

    assert regions and len(regions) == len(crosshairs)
    for coords in regions:
        main_color, stamp_color, offsets = _extract_stamp(g, coords)
        assert main_color in crosshairs
        cr, cc = crosshairs[main_color]
        for dr, dc in offsets:
            tr, tc = cr + dr, cc + dc
            assert 0 <= tr < h and 0 <= tc < w
            out[tr, tc] = stamp_color

    return out


def _rotate_pattern(pattern: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(-c, r) for r, c in pattern]


def _flip_pattern(pattern: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, -c) for r, c in pattern]


def _sample_pattern(rng: random.Random, include_center: bool) -> list[tuple[int, int]]:
    base = list(rng.choice(PATTERNS_WITH_CENTER if include_center else PATTERNS_WITHOUT_CENTER))
    if rng.random() < 0.5:
        base = _flip_pattern(base)
    for _ in range(rng.randrange(4)):
        base = _rotate_pattern(base)
    return base


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inner = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return inner
    inner[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return inner


def _valid_stamp_centers(mask: np.ndarray, pattern: list[tuple[int, int]]) -> list[tuple[int, int]]:
    inner = _erode4(mask)
    h, w = mask.shape
    centers: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(inner)):
        ok = True
        for dr, dc in pattern:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w and inner[nr, nc]):
                ok = False
                break
        if ok:
            centers.append((int(r), int(c)))
    return centers


def _make_region_mask(
    rng: random.Random, pattern: list[tuple[int, int]]
) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(REGION_MASK_RETRIES):
        h, w = rng.choice(BASE_DIMS)
        mask = np.ones((h, w), dtype=bool)

        cut_h_max = min(2, h - 4)
        cut_w_max = min(2, w - 4)
        corners = ["tl", "tr", "bl", "br"]
        rng.shuffle(corners)
        for corner in corners[: rng.choices([0, 1, 2], weights=[2, 3, 2], k=1)[0]]:
            cut_h = rng.randint(1, cut_h_max)
            cut_w = rng.randint(1, cut_w_max)
            if corner == "tl":
                mask[:cut_h, :cut_w] = False
            elif corner == "tr":
                mask[:cut_h, w - cut_w :] = False
            elif corner == "bl":
                mask[h - cut_h :, :cut_w] = False
            else:
                mask[h - cut_h :, w - cut_w :] = False

        sides = []
        if w >= 7:
            sides.extend(["top", "bottom"])
        if h >= 7:
            sides.extend(["left", "right"])
        if sides and rng.random() < 0.5:
            side = rng.choice(sides)
            if side in {"top", "bottom"}:
                length = rng.randint(2, min(3, w - 4))
                start = rng.randint(2, w - length - 2)
                row = 0 if side == "top" else h - 1
                mask[row, start : start + length] = False
            else:
                length = rng.randint(2, min(3, h - 4))
                start = rng.randint(2, h - length - 2)
                col = 0 if side == "left" else w - 1
                mask[start : start + length, col] = False

        if not (20 <= int(mask.sum()) <= 60):
            continue
        if len(_components(mask)) != 1:
            continue

        for _ in range(rng.randrange(4)):
            mask = np.rot90(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)

        centers = _valid_stamp_centers(mask, pattern)
        if centers:
            return mask, rng.choice(centers)
    return None


def _find_slots(blocked: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    H, W = blocked.shape
    slots: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not blocked[r0 : r0 + h, c0 : c0 + w].any():
                slots.append((r0, c0))
    return slots


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _crosshair_box(pattern: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    row_offsets = [-1, 0, 1] + [r for r, _ in pattern]
    col_offsets = [-1, 0, 1] + [c for _, c in pattern]
    return min(row_offsets), max(row_offsets), min(col_offsets), max(col_offsets)


def _draw_region(
    grid: np.ndarray,
    origin: tuple[int, int],
    mask: np.ndarray,
    anchor: tuple[int, int],
    main_color: int,
    stamp_color: int,
    pattern: list[tuple[int, int]],
) -> None:
    r0, c0 = origin
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = main_color
    ar, ac = anchor
    for dr, dc in pattern:
        patch[ar + dr, ac + dc] = stamp_color


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        cr, cc = spec["crosshair_center"]
        for dr, dc in spec["pattern"]:
            out[cr + dr, cc + dc] = spec["stamp_color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.choices(PAIR_COUNTS, weights=PAIR_WEIGHTS, k=1)[0]
    side_min = GRID_MIN + 2 * (n_pairs - 2)
    h = rng.randint(side_min, GRID_MAX)
    w = rng.randint(side_min, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = COLOR_POOL[:]
    rng.shuffle(colors)
    main_colors = colors[:n_pairs]
    stamp_colors = colors[n_pairs : 2 * n_pairs]

    include_center_flags = [True, False]
    while len(include_center_flags) < n_pairs:
        include_center_flags.append(rng.random() < 0.5)
    rng.shuffle(include_center_flags)

    specs: list[dict] = []
    for i in range(n_pairs):
        pattern = _sample_pattern(rng, include_center_flags[i])
        region = _make_region_mask(rng, pattern)
        if region is None:
            return None
        mask, anchor = region
        specs.append(
            {
                "main_color": main_colors[i],
                "stamp_color": stamp_colors[i],
                "pattern": pattern,
                "mask": mask,
                "anchor": anchor,
            }
        )

    for spec in sorted(specs, key=lambda item: int(item["mask"].size), reverse=True):
        mask = spec["mask"]
        mh, mw = mask.shape
        slots = _find_slots(blocked, mh, mw)
        if not slots:
            return None
        origin = rng.choice(slots)
        spec["region_origin"] = origin
        _draw_region(
            grid,
            origin,
            mask,
            spec["anchor"],
            spec["main_color"],
            spec["stamp_color"],
            spec["pattern"],
        )
        _reserve(blocked, origin[0], origin[1], mh, mw, pad=1)

    for spec in sorted(
        specs,
        key=lambda item: (_crosshair_box(item["pattern"])[1] - _crosshair_box(item["pattern"])[0])
        * (_crosshair_box(item["pattern"])[3] - _crosshair_box(item["pattern"])[2]),
        reverse=True,
    ):
        rlo, rhi, clo, chi = _crosshair_box(spec["pattern"])
        bh = rhi - rlo + 1
        bw = chi - clo + 1
        slots = _find_slots(blocked, bh, bw)
        if not slots:
            return None
        box_r0, box_c0 = rng.choice(slots)
        cr = box_r0 - rlo
        cc = box_c0 - clo
        grid[cr, cc] = spec["main_color"]
        grid[cr - 1, cc] = GREY
        grid[cr + 1, cc] = GREY
        grid[cr, cc - 1] = GREY
        grid[cr, cc + 1] = GREY
        spec["crosshair_center"] = (cr, cc)
        _reserve(blocked, box_r0, box_c0, bh, bw, pad=1)

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
