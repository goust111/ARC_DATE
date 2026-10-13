from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
ARROW_COLOR = 5
LEGEND_COLOR = 8
RING_COLORS = (1, 2, 3, 4, 6, 7, 9)
SIDES = ("right", "down", "left", "up")
SIDE_TO_VEC = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
SIDE_TO_TURNS = {
    "right": 0,
    "down": 1,
    "left": 2,
    "up": 3,
}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

HOLE_SHAPES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
]

ARROW_PATTERNS = {
    "right": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    "up": np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
}

GRID_H_RANGE = (30, 35)
GRID_W_RANGE = (30, 35)
LEGEND_BAND_H = 12
OBJECT_COUNT_CHOICES = (4, 5)
TAIL_LENGTH_RANGE = (2, 4)
PLACEMENT_RETRIES = 300
GEN_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.flat).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return int(dr + dc)


def shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(int(x) for x in mask.shape), mask.astype(np.uint8).tobytes())


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def extract_hole(component_mask: np.ndarray) -> dict[str, object] | None:
    holes = []
    h, w = component_mask.shape
    for coords in components_mask(~component_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        hole_mask[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append({"mask": hole_mask, "bbox": (hr0, hc0, hr1, hc1)})
    if len(holes) != 1:
        return None
    return holes[0]


def tail_from_component(component_mask: np.ndarray, hole_bbox: tuple[int, int, int, int]) -> dict[str, object] | None:
    hr0, hc0, hr1, hc1 = hole_bbox
    body_bbox = (hr0 - 1, hc0 - 1, hr1 + 1, hc1 + 1)
    br0, bc0, br1, bc1 = body_bbox
    tail = []
    for r, c in np.argwhere(component_mask):
        rr = int(r)
        cc = int(c)
        if br0 <= rr <= br1 and bc0 <= cc <= bc1:
            continue
        tail.append((rr, cc))
    if not tail:
        return None

    tail_coords = np.asarray(tail, dtype=int)
    rows = sorted({int(r) for r in tail_coords[:, 0]})
    cols = sorted({int(c) for c in tail_coords[:, 1]})
    if len(rows) == 1:
        row = rows[0]
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        if cols[-1] < bc0:
            side = "left"
            end = (row, cols[0])
        elif cols[0] > bc1:
            side = "right"
            end = (row, cols[-1])
        else:
            return None
    elif len(cols) == 1:
        col = cols[0]
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        if rows[-1] < br0:
            side = "up"
            end = (rows[0], col)
        elif rows[0] > br1:
            side = "down"
            end = (rows[-1], col)
        else:
            return None
    else:
        return None

    return {
        "side": side,
        "length": int(len(tail)),
        "end": (int(end[0]), int(end[1])),
        "body_bbox": body_bbox,
    }


def classify_arrow(coords: np.ndarray) -> str | None:
    pts = [(int(r), int(c)) for r, c in coords]
    if len(pts) != 4:
        return None

    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in pts:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    for row, cols in by_row.items():
        if len(cols) != 3:
            continue
        cols = sorted(cols)
        if cols != list(range(cols[0], cols[0] + 3)):
            continue
        tip = [(r, c) for r, c in pts if r != row]
        if len(tip) != 1:
            continue
        tr, tc = tip[0]
        if abs(tr - row) != 1:
            continue
        if tc == cols[0]:
            return "left"
        if tc == cols[-1]:
            return "right"

    for col, rows in by_col.items():
        if len(rows) != 3:
            continue
        rows = sorted(rows)
        if rows != list(range(rows[0], rows[0] + 3)):
            continue
        tip = [(r, c) for r, c in pts if c != col]
        if len(tip) != 1:
            continue
        tr, tc = tip[0]
        if abs(tc - col) != 1:
            continue
        if tr == rows[0]:
            return "up"
        if tr == rows[-1]:
            return "down"
    return None


def collect_nonbg_component(
    grid: np.ndarray,
    start: tuple[int, int],
    forbidden: set[int],
) -> np.ndarray:
    h, w = grid.shape
    sr, sc = start
    seen = np.zeros((h, w), dtype=bool)
    queue = deque([(sr, sc)])
    seen[sr, sc] = True
    cells = []
    while queue:
        r, c = queue.popleft()
        cells.append((r, c))
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if seen[nr, nc]:
                continue
            color = int(grid[nr, nc])
            if color in forbidden:
                continue
            seen[nr, nc] = True
            queue.append((nr, nc))
    return np.asarray(cells, dtype=int)


def find_sprite_component(
    grid: np.ndarray,
    tail_end: tuple[int, int],
    ring_color: int,
    bg: int,
) -> np.ndarray:
    tr, tc = tail_end
    h, w = grid.shape
    forbidden = {bg, ARROW_COLOR, ring_color}
    seeds = []
    for dr, dc in DIRS8:
        nr, nc = tr + dr, tc + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in forbidden:
            seeds.append((nr, nc))
    assert seeds, "sprite must touch the tail end"
    return collect_nonbg_component(grid, seeds[0], forbidden)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    arrows = []
    for coords in components_mask(g == ARROW_COLOR, connectivity=8):
        direction = classify_arrow(coords)
        if direction is None:
            continue
        arrows.append({"bbox": bbox_coords(coords), "direction": direction})
    assert arrows, "expected grey arrow legends"

    legends = []
    rings = []
    for color in sorted(int(c) for c in np.unique(g)):
        if color in (bg, ARROW_COLOR):
            continue
        for coords in components_mask(g == color, connectivity=8):
            r0, c0, r1, c1 = bbox_coords(coords)
            comp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            comp_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            hole = extract_hole(comp_mask)
            if hole is None:
                continue
            hole_key = shape_key(hole["mask"])
            tail = tail_from_component(comp_mask, hole["bbox"])
            entry = {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "hole_key": hole_key,
                "tail": tail,
            }
            if tail is None:
                legends.append(entry)
            else:
                end_r, end_c = tail["end"]
                entry["tail_end_abs"] = (r0 + end_r, c0 + end_c)
                rings.append(entry)

    assert legends and rings, "expected both legends and tail rings"

    legends.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    unused_arrows = set(range(len(arrows)))
    legend_lookup: dict[tuple[tuple[int, int], bytes], int] = {}
    for legend in legends:
        best_idx = None
        best_score = None
        lr0, lc0, lr1, lc1 = legend["bbox"]
        lcr = lr0 + lr1
        lcc = lc0 + lc1
        for idx in unused_arrows:
            arrow = arrows[idx]
            ar0, ac0, ar1, ac1 = arrow["bbox"]
            acr = ar0 + ar1
            acc = ac0 + ac1
            score = (
                bbox_distance(legend["bbox"], arrow["bbox"]),
                abs(lcr - acr) + abs(lcc - acc),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None and best_score[0] <= 4
        arrow = arrows[best_idx]
        legend_lookup[legend["hole_key"]] = SIDE_TO_TURNS[arrow["direction"]]
        unused_arrows.remove(best_idx)

    out = g.copy()
    for ring in sorted(rings, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        turns = legend_lookup[ring["hole_key"]]
        sprite_coords = find_sprite_component(g, ring["tail_end_abs"], int(ring["color"]), bg)
        sr0, sc0, sr1, sc1 = bbox_coords(sprite_coords)
        sprite_patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        rotated = np.rot90(sprite_patch, -turns)
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
        mask = rotated != bg
        out[sr0 : sr1 + 1, sc0 : sc1 + 1][mask] = rotated[mask]

    return out


def shell_from_hole(hole_mask: np.ndarray) -> np.ndarray:
    h, w = hole_mask.shape
    shell = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in np.argwhere(hole_mask):
        rr = int(r) + 1
        cc = int(c) + 1
        shell[rr - 1 : rr + 2, cc - 1 : cc + 2] = True
    shell[1 : h + 1, 1 : w + 1][hole_mask] = False
    return shell


def find_attachment(shell: np.ndarray, side: str) -> tuple[int, int]:
    h, w = shell.shape
    if side == "left":
        candidates = [(r, 0) for r in range(h) if shell[r, 0]]
        key = lambda rc: abs(rc[0] - (h - 1) / 2)
    elif side == "right":
        candidates = [(r, w - 1) for r in range(h) if shell[r, w - 1]]
        key = lambda rc: abs(rc[0] - (h - 1) / 2)
    elif side == "up":
        candidates = [(0, c) for c in range(w) if shell[0, c]]
        key = lambda rc: abs(rc[1] - (w - 1) / 2)
    else:
        candidates = [(h - 1, c) for c in range(w) if shell[h - 1, c]]
        key = lambda rc: abs(rc[1] - (w - 1) / 2)
    assert candidates, "tail side must touch the shell outline"
    return min(candidates, key=key)


def contact_ok(sprite_patch: np.ndarray, side: str) -> bool:
    n = sprite_patch.shape[0]
    mid = n // 2
    if side == "right":
        return int(sprite_patch[mid, 0]) != BG
    if side == "left":
        return int(sprite_patch[mid, n - 1]) != BG
    if side == "down":
        return int(sprite_patch[0, mid]) != BG
    return int(sprite_patch[n - 1, mid]) != BG


def generate_sprite(side: str, ring_color: int, rng: random.Random) -> np.ndarray:
    palette = [color for color in RING_COLORS if color != ring_color]
    for _ in range(200):
        n = rng.choice([2, 3])
        occ_count = 3 if n == 2 else rng.randint(4, 6)
        coords = rng.sample([(r, c) for r in range(n) for c in range(n)], occ_count)
        occ = np.zeros((n, n), dtype=bool)
        for r, c in coords:
            occ[r, c] = True
        if len(components_mask(occ, connectivity=8)) != 1:
            continue
        if any(np.array_equal(occ, np.rot90(occ, k)) for k in (1, 2, 3)):
            continue

        color_count = rng.choice([2, 3])
        colors = rng.sample(palette, color_count)
        patch = np.zeros((n, n), dtype=int)
        used = set()
        occ_list = list(coords)
        rng.shuffle(occ_list)
        for idx, (r, c) in enumerate(occ_list):
            if idx < color_count:
                color = colors[idx]
            else:
                color = rng.choice(colors)
            patch[r, c] = color
            used.add(color)
        if len(used) < 2:
            continue
        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue

        variants = [np.rot90(patch, k) for k in range(4)]
        viable = [variant for variant in variants if contact_ok(variant, side)]
        if viable:
            return rng.choice(viable)
    raise RuntimeError("failed to build a non-symmetric sprite")


def build_object_patch(
    hole_mask: np.ndarray,
    ring_color: int,
    side: str,
    tail_len: int,
    sprite_patch: np.ndarray,
) -> np.ndarray:
    shell = shell_from_hole(hole_mask)
    attach_r, attach_c = find_attachment(shell, side)

    coords: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(shell):
        coords.append((int(r), int(c), ring_color))

    dr, dc = SIDE_TO_VEC[side]
    tail_end = (attach_r + dr * tail_len, attach_c + dc * tail_len)
    for step in range(1, tail_len + 1):
        coords.append((attach_r + dr * step, attach_c + dc * step, ring_color))

    n = int(sprite_patch.shape[0])
    if side == "right":
        sprite_top = tail_end[0] - n // 2
        sprite_left = tail_end[1] + 1
    elif side == "left":
        sprite_top = tail_end[0] - n // 2
        sprite_left = tail_end[1] - n
    elif side == "down":
        sprite_top = tail_end[0] + 1
        sprite_left = tail_end[1] - n // 2
    else:
        sprite_top = tail_end[0] - n
        sprite_left = tail_end[1] - n // 2

    for r in range(n):
        for c in range(n):
            color = int(sprite_patch[r, c])
            if color == BG:
                continue
            coords.append((sprite_top + r, sprite_left + c, color))

    min_r = min(r for r, _c, _color in coords)
    min_c = min(c for _r, c, _color in coords)
    max_r = max(r for r, _c, _color in coords)
    max_c = max(c for _r, c, _color in coords)
    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c, color in coords:
        patch[r - min_r, c - min_c] = color
    return patch


def build_legend_patch(hole_mask: np.ndarray, direction: str) -> np.ndarray:
    shell = shell_from_hole(hole_mask)
    arrow = ARROW_PATTERNS[direction]
    total_h = max(shell.shape[0], arrow.shape[0])
    total_w = shell.shape[1] + 2 + arrow.shape[1]
    patch = np.zeros((total_h, total_w), dtype=int)

    shell_top = (total_h - shell.shape[0]) // 2
    arrow_top = (total_h - arrow.shape[0]) // 2
    patch[shell_top : shell_top + shell.shape[0], : shell.shape[1]][shell] = LEGEND_COLOR
    arrow_cols = shell.shape[1] + 2
    patch[arrow_top : arrow_top + arrow.shape[0], arrow_cols : arrow_cols + arrow.shape[1]][arrow] = ARROW_COLOR
    return patch


def can_place(
    reserved: np.ndarray,
    patch_shape: tuple[int, int],
    top: int,
    left: int,
    pad: int = 1,
) -> bool:
    h, w = patch_shape
    rh, rw = reserved.shape
    if top < 0 or left < 0 or top + h > rh or left + w > rw:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rh, top + h + pad)
    c1 = min(rw, left + w + pad)
    return not reserved[r0:r1, c0:c1].any()


def mark_reserved(reserved: np.ndarray, patch_shape: tuple[int, int], top: int, left: int, pad: int = 1) -> None:
    h, w = patch_shape
    rh, rw = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rh, top + h + pad)
    c1 = min(rw, left + w + pad)
    reserved[r0:r1, c0:c1] = True


def paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    sub = grid[top : top + h, left : left + w]
    assert not np.any(mask & (sub != BG))
    sub[mask] = patch[mask]


def transform_grid(grid: np.ndarray, op: int) -> np.ndarray:
    if op < 4:
        return np.rot90(grid, op)
    return np.rot90(np.fliplr(grid), op - 4)


def gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(*GRID_H_RANGE)
    w = rng.randrange(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    directions = list(SIDES)
    rng.shuffle(directions)
    legend_patches = [build_legend_patch(HOLE_SHAPES[i], directions[i]) for i in range(len(HOLE_SHAPES))]
    legend_order = list(range(len(legend_patches)))
    rng.shuffle(legend_order)
    for idx in legend_order:
        patch = legend_patches[idx]
        placed = False
        max_top = LEGEND_BAND_H - patch.shape[0]
        if max_top < 0:
            return None
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, max_top)
            left = rng.randint(0, w - patch.shape[1])
            if not can_place(reserved, patch.shape, top, left, pad=1):
                continue
            paste_patch(grid, patch, top, left)
            mark_reserved(reserved, patch.shape, top, left, pad=1)
            placed = True
            break
        if not placed:
            return None

    object_count = rng.choice(OBJECT_COUNT_CHOICES)
    shape_ids = list(range(len(HOLE_SHAPES)))
    while len(shape_ids) < object_count:
        shape_ids.append(rng.randrange(len(HOLE_SHAPES)))
    rng.shuffle(shape_ids)
    ring_colors = rng.sample(list(RING_COLORS), object_count)
    object_patches = []
    for shape_id, ring_color in zip(shape_ids, ring_colors):
        side = rng.choice(SIDES)
        tail_len = rng.randint(*TAIL_LENGTH_RANGE)
        sprite = generate_sprite(side, ring_color, rng)
        object_patches.append(
            build_object_patch(HOLE_SHAPES[shape_id], ring_color, side, tail_len, sprite)
        )

    order = sorted(range(len(object_patches)), key=lambda idx: object_patches[idx].size, reverse=True)
    for idx in order:
        patch = object_patches[idx]
        placed = False
        min_top = LEGEND_BAND_H + 2
        max_top = h - patch.shape[0]
        if max_top < min_top:
            return None
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(min_top, max_top)
            left = rng.randint(0, w - patch.shape[1])
            if not can_place(reserved, patch.shape, top, left, pad=1):
                continue
            paste_patch(grid, patch, top, left)
            mark_reserved(reserved, patch.shape, top, left, pad=1)
            placed = True
            break
        if not placed:
            return None

    grid = transform_grid(grid, rng.randrange(8))
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
