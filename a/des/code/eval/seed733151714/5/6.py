import random
from collections import deque

import numpy as np


BG = 0
CORE_COLOR = 4
BASELINE_COLOR = 9
RING_COLOR_POOL = (1, 2, 3, 5, 6, 7, 8)

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

ARROW_SHAPES = {
    "up": {(0, 1), (1, 0), (1, 2)},
    "right": {(0, 0), (1, 1), (2, 0)},
    "down": {(0, 0), (0, 2), (1, 1)},
    "left": {(0, 1), (1, 0), (2, 1)},
}
ARROW_TO_TURNS_CW = {"up": 0, "right": 1, "down": 2, "left": 3}
TURNS_CW_TO_ARROW = {v: k for k, v in ARROW_TO_TURNS_CW.items()}

GEN_RETRIES = 400
CORE_RETRIES = 200


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalized_coords(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _is_ring_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    norm = _normalized_coords(coords)
    border = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    return norm == border


def _arrow_turns_cw(coords: list[tuple[int, int]]) -> int | None:
    if len(coords) != 3:
        return None
    norm = _normalized_coords(coords)
    for name, shape in ARROW_SHAPES.items():
        if norm == shape:
            return ARROW_TO_TURNS_CW[name]
    return None


def _is_horizontal_line(coords: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in coords}
    if len(rows) != 1:
        return False
    cols = sorted(c for _, c in coords)
    return len(cols) >= 4 and cols == list(range(cols[0], cols[-1] + 1))


def _rotate_point_cw(
    r: int, c: int, center2: tuple[int, int], turns_cw: int
) -> tuple[int, int]:
    turns = turns_cw % 4
    y2 = 2 * r - center2[0]
    x2 = 2 * c - center2[1]
    if turns == 1:
        y2, x2 = x2, -y2
    elif turns == 2:
        y2, x2 = -y2, -x2
    elif turns == 3:
        y2, x2 = -x2, y2
    rr2 = center2[0] + y2
    cc2 = center2[1] + x2
    assert rr2 % 2 == 0 and cc2 % 2 == 0
    return rr2 // 2, cc2 // 2


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate nested square-ring layers by color-matched arrow instructions and erase the legend."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    rings_by_color: dict[int, dict] = {}
    turns_by_color: dict[int, int] = {}
    payload: list[tuple[int, int, int]] = []
    baseline_count = 0

    for color in (int(c) for c in np.unique(g) if int(c) != BG):
        for comp in _components_mask(g == color, connectivity=8):
            if _is_ring_component(comp):
                assert color not in rings_by_color
                rings_by_color[color] = {
                    "color": color,
                    "coords": comp,
                    "bbox": _bbox(comp),
                }
                continue

            turns = _arrow_turns_cw(comp)
            if turns is not None:
                assert color not in turns_by_color
                turns_by_color[color] = turns
                continue

            if _is_horizontal_line(comp):
                baseline_count += 1
                continue

            for r, c in comp:
                payload.append((r, c, color))

    assert baseline_count == 1
    assert rings_by_color
    assert set(rings_by_color) == set(turns_by_color)
    assert payload

    rings = sorted(
        rings_by_color.values(),
        key=lambda item: (
            (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1),
            item["bbox"][0],
        ),
        reverse=True,
    )

    center2 = (
        rings[0]["bbox"][0] + rings[0]["bbox"][2],
        rings[0]["bbox"][1] + rings[0]["bbox"][3],
    )
    for idx, ring in enumerate(rings):
        r0, c0, r1, c1 = ring["bbox"]
        assert r0 + r1 == center2[0] and c0 + c1 == center2[1]
        if idx:
            pr0, pc0, pr1, pc1 = rings[idx - 1]["bbox"]
            assert pr0 < r0 < r1 < pr1
            assert pc0 < c0 < c1 < pc1

    out = np.full_like(g, BG)
    cumulative = 0
    for ring in rings:
        cumulative = (cumulative + turns_by_color[ring["color"]]) % 4
        for r, c in ring["coords"]:
            rr, cc = _rotate_point_cw(r, c, center2, cumulative)
            assert 0 <= rr < h and 0 <= cc < w
            assert out[rr, cc] in (BG, ring["color"])
            out[rr, cc] = ring["color"]

    for r, c, color in payload:
        rr, cc = _rotate_point_cw(r, c, center2, cumulative)
        assert 0 <= rr < h and 0 <= cc < w
        assert out[rr, cc] in (BG, color)
        out[rr, cc] = color

    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = []
        for r, c in zip(*np.where(mask)):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        if rot.shape == mask.shape:
            variants.append(rot)
        flip = np.fliplr(rot)
        if flip.shape == mask.shape:
            variants.append(flip)
    for variant in variants[1:]:
        if np.array_equal(mask, variant):
            return False
    return True


def _make_core_mask(h: int, w: int) -> np.ndarray:
    min_area = max(4, (h * w) // 3)
    max_area = max(min_area, h * w - 2)
    for _ in range(CORE_RETRIES):
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(h, w, area)
        if not _touches_all_bbox_sides(mask):
            continue
        if not _is_asymmetric(mask):
            continue
        return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    if w >= 3:
        mask[0, 1] = True
    return mask


def _draw_ring(grid: np.ndarray, top: int, left: int, side: int, color: int) -> None:
    bottom = top + side - 1
    right = left + side - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _arrow_patch(direction: str, color: int) -> np.ndarray:
    coords = ARROW_SHAPES[direction]
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in coords:
        patch[r, c] = color
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    target = grid[top : top + ph, left : left + pw]
    overlap = (target != BG) & (patch != BG)
    assert not overlap.any()
    target[patch != BG] = patch[patch != BG]


def _make_instruction_block(colors: list[int], turns: dict[int, int]) -> np.ndarray:
    n = len(colors)
    block_h = 4
    block_w = 4 * n - 1
    block = np.full((block_h, block_w), BG, dtype=int)
    block[-1, :] = BASELINE_COLOR
    order = colors[:]
    random.shuffle(order)
    for idx, color in enumerate(order):
        direction = TURNS_CW_TO_ARROW[turns[color]]
        patch = _arrow_patch(direction, color)
        top = block_h - 2 - (patch.shape[0] - 1)
        left = 4 * idx
        _place_patch(block, patch, top, left)
    return block


def _gen_attempt() -> np.ndarray | None:
    layers = random.randint(2, 5)
    inner_gap_side = random.choice((5, 7, 9))
    outer_side = inner_gap_side + 2 + 4 * (layers - 1)

    extra_h = random.randint(0, 8)
    extra_w = random.randint(0, 8)
    gap = random.randint(2, 4)

    colors = random.sample(RING_COLOR_POOL, layers)
    turns = {color: random.randint(0, 3) for color in colors}
    instruction = _make_instruction_block(colors, turns)
    ih, iw = instruction.shape

    h = outer_side + ih + gap + 2 + extra_h
    w = max(outer_side, iw) + 2 + extra_w
    grid = np.full((h, w), BG, dtype=int)

    instruction_on_top = bool(random.getrandbits(1))
    if instruction_on_top:
        inst_top = random.randint(1, h - (ih + gap + outer_side) - 1)
        struct_top = inst_top + ih + gap
    else:
        struct_top = random.randint(1, h - (outer_side + gap + ih) - 1)
        inst_top = struct_top + outer_side + gap

    struct_left = random.randint(1, w - outer_side - 1)
    inst_left = random.randint(1, w - iw - 1)

    for depth, color in enumerate(colors):
        side = outer_side - 4 * depth
        top = struct_top + 2 * depth
        left = struct_left + 2 * depth
        _draw_ring(grid, top, left, side, color)

    inner_top = struct_top + 2 * layers
    inner_left = struct_left + 2 * layers
    core_h = random.randint(3, min(5, inner_gap_side))
    core_w = random.randint(3, min(5, inner_gap_side))
    core_mask = _make_core_mask(core_h, core_w)
    core_top = inner_top + random.randint(0, inner_gap_side - core_h)
    core_left = inner_left + random.randint(0, inner_gap_side - core_w)
    grid[core_top : core_top + core_h, core_left : core_left + core_w][core_mask] = CORE_COLOR

    _place_patch(grid, instruction, inst_top, inst_left)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.count_nonzero(out) <= np.count_nonzero(grid == CORE_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rotation sample")
