import random
from collections import Counter, defaultdict, deque

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    color, freq = max(counts.items(), key=lambda item: (item[1], -item[0]))
    assert sum(1 for value in counts.values() if value == freq) == 1
    return int(color)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotation_for_bbox(bbox: tuple[int, int, int, int], center_r: int, center_c: int) -> int:
    r0, c0, r1, c1 = bbox
    row_sum = r0 + r1
    col_sum = c0 + c1
    center_row_sum = 2 * center_r
    center_col_sum = 2 * center_c
    assert row_sum != center_row_sum
    assert col_sum != center_col_sum

    top = row_sum < center_row_sum
    left = col_sum < center_col_sum
    if top and not left:
        return 0
    if top and left:
        return 1
    if not top and left:
        return 2
    return 3


def _rotate_patch_centered(
    patch: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    k: int,
) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = bbox
    rotated = np.rot90(patch, k)
    rh, rw = rotated.shape
    nr0_num = r0 + r1 - (rh - 1)
    nc0_num = c0 + c1 - (rw - 1)
    assert nr0_num % 2 == 0
    assert nc0_num % 2 == 0
    nr0 = nr0_num // 2
    nc0 = nc0_num // 2
    if k == 0:
        assert nr0 == r0 and nc0 == c0
    assert np.any(rotated != bg)
    return rotated, int(nr0), int(nc0)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each foreground object by a quadrant-dependent multiple of 90 degrees.

    Input structure
    - The background is the dominant color.
    - The grid has odd height and odd width, so it has a unique center cell.
    - Foreground objects are 4-connected and each object lies entirely in one quadrant,
      never touching the center row or center column.

    Rule
    - Objects in the upper-right quadrant are unchanged.
    - Objects in the upper-left quadrant rotate 90 degrees counter-clockwise.
    - Objects in the lower-left quadrant rotate 180 degrees.
    - Objects in the lower-right quadrant rotate 90 degrees clockwise.
    - Each rotated object keeps the center of its original bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1

    center_r, center_c = h // 2, w // 2
    bg = _dominant_color(g)
    comps = _components(g != bg)
    assert comps

    out = np.full_like(g, bg)
    for coords in comps:
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        assert r1 < center_r or r0 > center_r
        assert c1 < center_c or c0 > center_c

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        k = _rotation_for_bbox(bbox, center_r, center_c)
        rotated, nr0, nc0 = _rotate_patch_centered(patch, bbox, bg, k)
        rh, rw = rotated.shape
        assert 0 <= nr0 <= h - rh
        assert 0 <= nc0 <= w - rw

        mask = rotated != bg
        target = out[nr0 : nr0 + rh, nc0 : nc0 + rw]
        assert not np.any((target != bg) & mask)
        target[mask] = rotated[mask]

    return out


QUADRANTS = ("UL", "UR", "LL", "LR")
ODD_SIDES = (11, 13, 15, 17)
EXTRA_OBJECTS = (0, 1, 2)
PALETTE = tuple(range(10))
MAX_ATTEMPTS = 400
OBJECT_RETRIES = 120
MASK_RETRIES = 160


def _quadrant_bounds(rows: int, cols: int, quadrant: str) -> tuple[int, int, int, int]:
    cr, cc = rows // 2, cols // 2
    if quadrant == "UL":
        return 0, cr - 1, 0, cc - 1
    if quadrant == "UR":
        return 0, cr - 1, cc + 1, cols - 1
    if quadrant == "LL":
        return cr + 1, rows - 1, 0, cc - 1
    return cr + 1, rows - 1, cc + 1, cols - 1


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _add_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        moves = []
        if r < er:
            moves.append((r + 1, c))
        elif r > er:
            moves.append((r - 1, c))
        if c < ec:
            moves.append((r, c + 1))
        elif c > ec:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        mask[r, c] = True


def _mask_touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_mask(height: int, width: int, rotation_k: int) -> np.ndarray | None:
    min_area = max(height + width - 1, (height * width + 2) // 3)
    max_area = max(min_area, height * width - 1)

    for _ in range(MASK_RETRIES):
        mask = np.zeros((height, width), dtype=bool)

        c = random.randrange(width)
        mask[0, c] = True
        for r in range(height - 1):
            for _ in range(random.randint(0, 2)):
                step = random.choice((-1, 1))
                if 0 <= c + step < width:
                    c += step
                    mask[r, c] = True
            mask[r + 1, c] = True

        existing = list(map(tuple, np.argwhere(mask)))
        _add_path(mask, (random.randrange(height), 0), random.choice(existing))
        existing = list(map(tuple, np.argwhere(mask)))
        _add_path(mask, (random.randrange(height), width - 1), random.choice(existing))

        target_area = random.randint(min_area, max_area)
        while int(mask.sum()) < target_area:
            frontier = []
            for cell in map(tuple, np.argwhere(mask)):
                for nbr in _neighbors(cell, height, width):
                    if not mask[nbr]:
                        frontier.append(nbr)
            if not frontier:
                break
            mask[random.choice(frontier)] = True

        if not _mask_touches_all_sides(mask):
            continue
        if len(_components(mask)) != 1:
            continue
        if rotation_k != 0 and np.array_equal(mask, np.rot90(mask, rotation_k)):
            continue
        return mask
    return None


def _object_plan(quadrant: str, color: int, max_side: int) -> dict | None:
    size_choices = [side for side in (5, 4, 3, 2) if side <= max_side]
    if not size_choices:
        return None

    for _ in range(OBJECT_RETRIES):
        side = random.choice(size_choices)
        same_parity = [value for value in range(side, 1, -2)]
        height = random.choice(same_parity)
        width = random.choice(same_parity)
        k = {"UL": 1, "UR": 0, "LL": 2, "LR": 3}[quadrant]
        mask = _make_mask(height, width, k)
        if mask is None:
            continue
        return {
            "quadrant": quadrant,
            "color": int(color),
            "side": int(side),
            "height": int(height),
            "width": int(width),
            "mask": mask,
            "k": int(k),
        }
    return None


def _place_box(
    quadrant: str,
    side: int,
    rows: int,
    cols: int,
    occupied: dict[str, list[tuple[int, int, int, int]]],
) -> tuple[int, int] | None:
    r_lo, r_hi, c_lo, c_hi = _quadrant_bounds(rows, cols, quadrant)
    if r_hi - r_lo + 1 < side or c_hi - c_lo + 1 < side:
        return None

    candidates = [
        (r, c)
        for r in range(r_lo, r_hi - side + 2)
        for c in range(c_lo, c_hi - side + 2)
    ]
    random.shuffle(candidates)
    for sr, sc in candidates:
        box = (sr, sc, sr + side - 1, sc + side - 1)
        if all(_separated(box, other) for other in occupied[quadrant]):
            occupied[quadrant].append(box)
            return sr, sc
    return None


def _render_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    assert not np.any((target != bg) & mask)
    target[mask] = patch[mask]


def _build_expected(rows: int, cols: int, bg: int, plans: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), bg, dtype=int)
    out = np.full((rows, cols), bg, dtype=int)
    for plan in plans:
        side = plan["side"]
        height = plan["height"]
        width = plan["width"]
        sr = plan["square_r"]
        sc = plan["square_c"]

        in_r = sr + (side - height) // 2
        in_c = sc + (side - width) // 2
        patch = _render_patch(plan["mask"], plan["color"], bg)
        _paste(inp, patch, in_r, in_c, bg)

        rotated = np.rot90(patch, plan["k"])
        out_r = sr + (side - rotated.shape[0]) // 2
        out_c = sc + (side - rotated.shape[1]) // 2
        _paste(out, rotated, out_r, out_c, bg)
    return inp, out


def _valid_structure(grid: np.ndarray, expected_count: int) -> bool:
    rows, cols = grid.shape
    cr, cc = rows // 2, cols // 2
    bg = _dominant_color(grid)
    if int(np.count_nonzero(grid == bg)) <= rows * cols // 2:
        return False
    comps = _components(grid != bg)
    if len(comps) != expected_count:
        return False
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        if not (r1 < cr or r0 > cr):
            return False
        if not (c1 < cc or c0 > cc):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rows = random.choice(ODD_SIDES)
    cols = random.choice(ODD_SIDES)
    bg = random.choice(PALETTE)
    total = 4 + random.choice(EXTRA_OBJECTS)
    quadrants = list(QUADRANTS) + random.sample(list(QUADRANTS), total - 4)
    random.shuffle(quadrants)
    colors = random.sample([c for c in PALETTE if c != bg], total)

    max_side = min(rows // 2, cols // 2)
    plans = []
    for quadrant, color in zip(quadrants, colors):
        plan = _object_plan(quadrant, color, max_side)
        if plan is None:
            return None
        plans.append(plan)

    occupied: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for plan in sorted(plans, key=lambda item: item["side"], reverse=True):
        placed = _place_box(plan["quadrant"], plan["side"], rows, cols, occupied)
        if placed is None:
            return None
        plan["square_r"], plan["square_c"] = placed

    inp, expected = _build_expected(rows, cols, bg, plans)
    if not _valid_structure(inp, total):
        return None
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
