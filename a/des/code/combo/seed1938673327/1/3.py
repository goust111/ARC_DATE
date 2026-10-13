import random

import numpy as np


BG = 0
WALL = 1
RED = 2
GREEN = 3
PAYLOAD_COLORS = (4, 5, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300

SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _extract_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions = []

    for coords in _components(g == WALL):
        r0, c0, r1, c1 = _bbox(coords)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        assert box_h >= 5 and box_w >= 5, "wall rectangle is too small"
        assert (box_h - box_w) % 2 == 0, "region dimensions must have matching parity"

        wall_set = {tuple(map(int, cell)) for cell in coords}
        border = _border_cells(r0, c0, r1, c1)
        assert all(cell in wall_set or g[cell] != WALL for cell in border)
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in wall_set), "wall pixels must lie on the border"

        gaps = [(r, c) for r, c in border if int(g[r, c]) != WALL]
        assert len(gaps) == 1, "expected exactly one wall gap"
        gap_r, gap_c = gaps[0]

        side_count = sum(
            [
                gap_r == r0,
                gap_r == r1,
                gap_c == c0,
                gap_c == c1,
            ]
        )
        assert side_count == 1, "gap must lie on exactly one side"

        if gap_r == r0:
            assert c0 < gap_c < c1, "gap cannot be on a corner"
            marker_r, marker_c = gap_r - 1, gap_c
        elif gap_r == r1:
            assert c0 < gap_c < c1, "gap cannot be on a corner"
            marker_r, marker_c = gap_r + 1, gap_c
        elif gap_c == c0:
            assert r0 < gap_r < r1, "gap cannot be on a corner"
            marker_r, marker_c = gap_r, gap_c - 1
        else:
            assert r0 < gap_r < r1, "gap cannot be on a corner"
            marker_r, marker_c = gap_r, gap_c + 1

        assert 0 <= marker_r < h and 0 <= marker_c < w, "marker must stay in bounds"
        marker_color = int(g[marker_r, marker_c])
        assert marker_color in (RED, GREEN), "gap marker must be red or green"

        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "marker": (marker_r, marker_c),
                "clockwise": marker_color == RED,
            }
        )

    assert regions, "expected at least one blue-walled region"
    regions.sort(key=lambda region: region["bbox"])
    return regions


def _rotate_point(r: int, c: int, bbox: tuple[int, int, int, int], clockwise: bool) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    sr = r0 + r1
    sc = c0 + c1
    if clockwise:
        nr2 = sr + 2 * c - sc
        nc2 = sc - 2 * r + sr
    else:
        nr2 = sr - 2 * c + sc
        nc2 = sc + 2 * r - sr
    assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation must land on cell centers"
    return nr2 // 2, nc2 // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the red/green pixel outside each wall gap to rotate that room's interior 90 degrees.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for region in _extract_regions(g):
        r0, c0, r1, c1 = region["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(g[r, c])
                if color in (BG, WALL):
                    continue
                nr, nc = _rotate_point(r, c, region["bbox"], region["clockwise"])
                if r0 < nr < r1 and c0 < nc < c1:
                    out[nr, nc] = color

    return out


def _choose_region_dims(rng: random.Random) -> tuple[int, int]:
    if rng.random() < 0.5:
        size = rng.choice([5, 6, 7, 8, 9, 10])
        return size, size

    parity = rng.choice([0, 1])
    dims = [d for d in range(5, 11) if d % 2 == parity and d >= 5 + (parity == 0)]
    h = rng.choice(dims)
    w_choices = [d for d in dims if d != h]
    w = rng.choice(w_choices)
    return h, w


def _random_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    order = list(range(len(SHAPES)))
    rng.shuffle(order)
    for idx in order:
        base = SHAPES[idx]
        rotations = list(range(4))
        rng.shuffle(rotations)
        for k in rotations:
            shape = np.rot90(base, k)
            if shape.shape[0] <= max_h and shape.shape[1] <= max_w:
                return shape.astype(bool)
    return None


def _can_place(mask: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    sh, sw = shape.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    coords = np.argwhere(shape)
    for rr, cc in coords:
        gr = top + int(rr)
        gc = left + int(cc)
        if mask[gr, gc]:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = gr + dr
                nc = gc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    return False
    return True


def _stamp(mask: np.ndarray, patch: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    for rr, cc in np.argwhere(shape):
        gr = top + int(rr)
        gc = left + int(cc)
        mask[gr, gc] = True
        patch[gr, gc] = color


def _place_objects(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    force_edge_object: bool,
) -> np.ndarray | None:
    region = np.zeros((inner_h, inner_w), dtype=int)
    occ = np.zeros((inner_h, inner_w), dtype=bool)
    max_objects = min(4, max(1, (inner_h * inner_w) // 7))
    target_count = rng.randint(1, max_objects)
    colors = rng.sample(PAYLOAD_COLORS, k=target_count)

    for idx, color in enumerate(colors):
        shape = _random_shape(rng, inner_h, inner_w)
        if shape is None:
            return None

        sh, sw = shape.shape
        placed = False
        attempts = []

        if idx == 0 and force_edge_object:
            if inner_w > inner_h:
                edge_col = 0 if rng.random() < 0.5 else inner_w - sw
                for row in range(inner_h - sh + 1):
                    attempts.append((row, edge_col))
            else:
                edge_row = 0 if rng.random() < 0.5 else inner_h - sh
                for col in range(inner_w - sw + 1):
                    attempts.append((edge_row, col))

        all_positions = [(r, c) for r in range(inner_h - sh + 1) for c in range(inner_w - sw + 1)]
        rng.shuffle(all_positions)
        attempts.extend(all_positions)

        seen = set()
        for top, left in attempts:
            if (top, left) in seen:
                continue
            seen.add((top, left))
            if _can_place(occ, shape, top, left):
                _stamp(occ, region, shape, top, left, color)
                placed = True
                break
        if not placed:
            return None

    return region


def _reserve_ok(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return not np.any(used[r0 : r1 + 1, c0 : c1 + 1])


def _draw_region(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap_side: str,
    gap_offset: int,
    clockwise: bool,
    payload: np.ndarray,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL

    if gap_side == "top":
        gap = (r0, c0 + 1 + gap_offset)
        marker = (r0 - 1, gap[1])
    elif gap_side == "bottom":
        gap = (r1, c0 + 1 + gap_offset)
        marker = (r1 + 1, c0 + 1 + gap_offset)
    elif gap_side == "left":
        gap = (r0 + 1 + gap_offset, c0)
        marker = (gap[0], c0 - 1)
    else:
        gap = (r0 + 1 + gap_offset, c1)
        marker = (gap[0], c1 + 1)

    grid[gap] = BG
    grid[marker] = RED if clockwise else GREEN
    grid[r0 + 1 : r1, c0 + 1 : c1] = payload


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    region_count = rng.randint(1, 4)
    dims = [_choose_region_dims(rng) for _ in range(region_count)]

    min_h = max(18, sum(h for h, _ in dims[:2]) + 8)
    min_w = max(18, sum(w for _, w in dims[:2]) + 8)
    canvas_h = rng.randint(min_h, min(30, min_h + 8))
    canvas_w = rng.randint(min_w, min(30, min_w + 8))

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    used = np.zeros_like(grid, dtype=bool)
    specs = []

    for h, w in dims:
        placed = False
        for _ in range(200):
            r0 = rng.randint(1, canvas_h - h - 2)
            c0 = rng.randint(1, canvas_w - w - 2)
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(canvas_h - 1, r1 + 1)
            cc1 = min(canvas_w - 1, c1 + 1)
            if _reserve_ok(used, rr0, cc0, rr1, cc1):
                used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
                specs.append((r0, c0, r1, c1))
                placed = True
                break
        if not placed:
            return None

    rng.shuffle(specs)
    directions = [rng.random() < 0.5 for _ in range(region_count)]

    for idx, bbox in enumerate(specs):
        r0, c0, r1, c1 = bbox
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        gap_side = rng.choice(["top", "bottom", "left", "right"])
        gap_offset = rng.randrange(inner_w if gap_side in ("top", "bottom") else inner_h)
        force_edge = inner_h != inner_w and rng.random() < 0.7
        payload = _place_objects(rng, inner_h, inner_w, force_edge)
        if payload is None:
            return None
        _draw_region(grid, bbox, gap_side, gap_offset, directions[idx], payload)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
