import random
from collections import Counter, deque

import numpy as np


BG = 0
FENCE = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
OBJECT_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
SIDES = ("top", "bottom", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

GEN_ATTEMPTS = 400
DESC_ATTEMPTS = 250
LAYOUT_ATTEMPTS = 120


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(np.min(coords[:, 0]))
    c0 = int(np.min(coords[:, 1]))
    r1 = int(np.max(coords[:, 0]))
    c1 = int(np.max(coords[:, 1]))
    return r0, c0, r1, c1


def _point_rect_chebyshev(
    point: tuple[int, int], rect: tuple[int, int, int, int]
) -> int:
    pr, pc = point
    r0, c0, r1, c1 = rect
    dr = 0 if r0 <= pr <= r1 else min(abs(pr - r0), abs(pr - r1))
    dc = 0 if c0 <= pc <= c1 else min(abs(pc - c0), abs(pc - c1))
    return max(dr, dc)


def _gate_cell(
    fence_h: int, fence_w: int, side: str, pos: int
) -> tuple[int, int]:
    if side == "top":
        return 0, pos
    if side == "bottom":
        return fence_h - 1, pos
    if side == "left":
        return pos, 0
    return pos, fence_w - 1


def _outside_cell(
    fence_h: int, fence_w: int, side: str, pos: int
) -> tuple[int, int]:
    if side == "top":
        return -1, pos
    if side == "bottom":
        return fence_h, pos
    if side == "left":
        return pos, -1
    return pos, fence_w


def _expand_mask(
    shape: tuple[int, int], start: np.ndarray, blocked: np.ndarray, steps: int
) -> np.ndarray:
    h, w = shape
    dist = -np.ones((h, w), dtype=int)
    q = deque()

    for r, c in start:
        rr, cc = int(r), int(c)
        if dist[rr, cc] == 0:
            continue
        dist[rr, cc] = 0
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        if dist[r, c] >= steps:
            continue
        nd = int(dist[r, c]) + 1
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = nd
            q.append((nr, nc))

    return dist >= 0


def _fence_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.flat))
    non_bg = [(count, color) for color, count in counts.items() if color != bg]
    assert non_bg, "expected non-background colors"
    non_bg.sort()
    return int(non_bg[-1][1])


def _find_objects(
    grid: np.ndarray, bg: int, fence_color: int
) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.flat)) - {bg, fence_color}):
        for coords in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(coords)
            area = int(coords.shape[0])
            if area != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            infos.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "object_coords": coords.astype(int),
                }
            )

    assert infos, "expected enclosed colored rectangles"
    return infos


def _find_fence_for_object(
    grid: np.ndarray,
    bg: int,
    fence_color: int,
    obj_info: dict[str, object],
) -> dict[str, object]:
    h, w = grid.shape
    obj_r0, obj_c0, obj_r1, obj_c1 = map(int, obj_info["bbox"])
    obj_color = int(obj_info["color"])
    obj_coords = np.asarray(obj_info["object_coords"], dtype=int)

    best = None
    best_area = None
    for r0 in range(obj_r0):
        for c0 in range(obj_c0):
            if grid[r0, c0] != fence_color:
                continue
            for r1 in range(obj_r1 + 1, h):
                if grid[r1, c0] != fence_color:
                    continue
                if r1 - r0 + 1 < 5:
                    continue
                for c1 in range(obj_c1 + 1, w):
                    if c1 - c0 + 1 < 5:
                        continue
                    if grid[r0, c1] != fence_color or grid[r1, c1] != fence_color:
                        continue

                    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                    if np.any(interior == fence_color):
                        continue

                    inner_mask = (interior != bg) & (interior != fence_color)
                    if not np.any(inner_mask):
                        continue
                    colors = np.unique(interior[inner_mask])
                    if colors.size != 1 or int(colors[0]) != obj_color:
                        continue

                    candidate_coords = np.argwhere(interior == obj_color) + np.array(
                        [r0 + 1, c0 + 1],
                        dtype=int,
                    )
                    if candidate_coords.shape != obj_coords.shape:
                        continue
                    if _bbox(candidate_coords) != (obj_r0, obj_c0, obj_r1, obj_c1):
                        continue

                    gates = 0
                    ok = True
                    for cc in range(c0 + 1, c1):
                        top_val = int(grid[r0, cc])
                        bot_val = int(grid[r1, cc])
                        if top_val == bg:
                            gates += 1
                        elif top_val != fence_color:
                            ok = False
                        if bot_val == bg:
                            gates += 1
                        elif bot_val != fence_color:
                            ok = False
                    for rr in range(r0 + 1, r1):
                        left_val = int(grid[rr, c0])
                        right_val = int(grid[rr, c1])
                        if left_val == bg:
                            gates += 1
                        elif left_val != fence_color:
                            ok = False
                        if right_val == bg:
                            gates += 1
                        elif right_val != fence_color:
                            ok = False
                    if not ok or not (1 <= gates <= 4):
                        continue

                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if best is None or area < best_area:
                        best = {
                            "bbox": (r0, c0, r1, c1),
                            "gates": int(gates),
                            "color": obj_color,
                            "object_coords": obj_coords,
                        }
                        best_area = area

    assert best is not None, "could not match an object to a surrounding fence"
    return best


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    fence_color = _fence_color(g, bg)
    objects = _find_objects(g, bg, fence_color)
    infos = [_find_fence_for_object(g, bg, fence_color, obj_info) for obj_info in objects]

    out = g.copy()
    blocked = g == fence_color
    infos.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))

    for info in infos:
        mask = _expand_mask(
            g.shape,
            np.asarray(info["object_coords"], dtype=int),
            blocked,
            int(info["gates"]),
        )
        out[mask & ~blocked] = int(info["color"])

    out[blocked] = fence_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored object outward by as many blocked ring layers as its surrounding grey fence has gates.

    Input structure:
    - The background is black.
    - Several single-pixel-thick grey rectangular fences appear on the grid, each with one to four missing side pixels acting as gates.
    - Inside every fence there is exactly one small solid monochrome rectangle, and different fences use different object colors.

    Rule:
    - For each grey fence, count how many gate pixels are missing from its rectangular border.
    - Starting from the enclosed colored rectangle, expand that color for the same number of 8-neighbor layers while treating grey fence pixels as blocking walls.
    - The expansion is clipped by the fence border, but it can pass through gate cells and continue outside; fences themselves remain grey.
    - If multiple expansions overlap, process fences from left to right and then top to bottom, so later fences overwrite earlier colors on non-grey cells.

    Key insight:
    - The gate count gives the expansion radius, while the gate positions determine where the otherwise rectangular growth can leak beyond the enclosing fence.
    """
    return _solve(grid)


def _sample_desc(
    rng: random.Random,
    color: int,
    desired_outside: bool | None = None,
    required_sides: tuple[str, ...] = (),
) -> dict[str, object] | None:
    for _ in range(DESC_ATTEMPTS):
        min_steps = max(1, len(required_sides))
        if desired_outside is True:
            min_steps = max(3, min_steps)
        steps = rng.randint(min_steps, 4)

        obj_h, obj_w = rng.choice(OBJECT_SHAPES)
        fence_h = rng.randint(max(6, obj_h + 4), 10)
        fence_w = rng.randint(max(6, obj_w + 4), 10)

        top_lo = 2
        top_hi = fence_h - obj_h - 2
        left_lo = 2
        left_hi = fence_w - obj_w - 2
        if top_lo > top_hi or left_lo > left_hi:
            continue

        bias_side = None
        if desired_outside is True:
            bias_side = required_sides[0] if required_sides else rng.choice(SIDES)

        if bias_side == "top":
            top_hi = min(top_hi, steps - 1)
        elif bias_side == "bottom":
            top_lo = max(top_lo, fence_h - obj_h - steps + 1)
        elif bias_side == "left":
            left_hi = min(left_hi, steps - 1)
        elif bias_side == "right":
            left_lo = max(left_lo, fence_w - obj_w - steps + 1)

        if top_lo > top_hi or left_lo > left_hi:
            continue

        obj_r0 = rng.randint(top_lo, top_hi)
        obj_c0 = rng.randint(left_lo, left_hi)
        obj_bbox = (obj_r0, obj_c0, obj_r0 + obj_h - 1, obj_c0 + obj_w - 1)

        free_sides = [side for side in SIDES if side not in required_sides]
        if steps - len(required_sides) > len(free_sides):
            continue
        gate_sides = list(required_sides) + rng.sample(free_sides, steps - len(required_sides))

        gates: dict[str, int] = {}
        for side in gate_sides:
            if desired_outside is True and side == bias_side:
                if side in ("top", "bottom"):
                    gates[side] = rng.randint(obj_bbox[1], obj_bbox[3])
                else:
                    gates[side] = rng.randint(obj_bbox[0], obj_bbox[2])
            else:
                limit = fence_w - 2 if side in ("top", "bottom") else fence_h - 2
                gates[side] = rng.randint(1, limit)

        spills_outside = any(
            _point_rect_chebyshev(_outside_cell(fence_h, fence_w, side, pos), obj_bbox) <= steps
            for side, pos in gates.items()
        )
        if desired_outside is not None and spills_outside != desired_outside:
            continue

        return {
            "color": color,
            "steps": steps,
            "fence_h": fence_h,
            "fence_w": fence_w,
            "obj_h": obj_h,
            "obj_w": obj_w,
            "obj_r0": obj_r0,
            "obj_c0": obj_c0,
            "gates": gates,
        }
    return None


def _draw_desc(grid: np.ndarray, top: int, left: int, desc: dict[str, object]) -> None:
    fence_h = int(desc["fence_h"])
    fence_w = int(desc["fence_w"])
    color = int(desc["color"])
    obj_h = int(desc["obj_h"])
    obj_w = int(desc["obj_w"])
    obj_r0 = int(desc["obj_r0"])
    obj_c0 = int(desc["obj_c0"])
    gates = dict(desc["gates"])

    grid[top, left : left + fence_w] = FENCE
    grid[top + fence_h - 1, left : left + fence_w] = FENCE
    grid[top : top + fence_h, left] = FENCE
    grid[top : top + fence_h, left + fence_w - 1] = FENCE

    for side, pos in gates.items():
        gr, gc = _gate_cell(fence_h, fence_w, side, int(pos))
        grid[top + gr, left + gc] = BG

    grid[
        top + obj_r0 : top + obj_r0 + obj_h,
        left + obj_c0 : left + obj_c0 + obj_w,
    ] = color


def _crop_canvas(rng: random.Random, canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_pad = rng.randint(0, min(4, int(r0)))
    left_pad = rng.randint(0, min(4, int(c0)))
    bottom_pad = rng.randint(0, min(4, int(canvas.shape[0] - 1 - r1)))
    right_pad = rng.randint(0, min(4, int(canvas.shape[1] - 1 - c1)))

    if rng.random() < 0.5:
        edge = rng.choice(("top", "bottom", "left", "right"))
        if edge == "top":
            top_pad = 0
        elif edge == "bottom":
            bottom_pad = 0
        elif edge == "left":
            left_pad = 0
        else:
            right_pad = 0

    return canvas[
        int(r0) - top_pad : int(r1) + bottom_pad + 1,
        int(c0) - left_pad : int(c1) + right_pad + 1,
    ]


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _validate_generated(grid: np.ndarray, want_overlap: bool, want_mixed: bool) -> bool:
    try:
        out = _solve(grid)
        bg = _most_common_color(grid)
        fence_color = _fence_color(grid, bg)
        objects = _find_objects(grid, bg, fence_color)
        infos = [_find_fence_for_object(grid, bg, fence_color, obj_info) for obj_info in objects]
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    blocked = grid == fence_color
    bbox_masks = [_bbox_mask(grid.shape, info["bbox"]) for info in infos]
    expand_masks = [
        _expand_mask(
            grid.shape,
            np.asarray(info["object_coords"], dtype=int),
            blocked,
            int(info["gates"]),
        )
        for info in infos
    ]

    leak_flags = [
        bool(np.any(mask & ~bbox_mask & ~blocked))
        for mask, bbox_mask in zip(expand_masks, bbox_masks)
    ]
    if not any(leak_flags):
        return False
    if want_mixed and not (any(leak_flags) and any(not flag for flag in leak_flags)):
        return False

    for i in range(len(infos)):
        for j in range(len(infos)):
            if i == j:
                continue
            if np.any(expand_masks[i] & bbox_masks[j] & ~blocked):
                return False

    overlap = False
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            shared = (
                expand_masks[i]
                & expand_masks[j]
                & ~blocked
                & ~bbox_masks[i]
                & ~bbox_masks[j]
            )
            if np.any(shared):
                overlap = True
    if overlap != want_overlap:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_fences = rng.randint(2, 3)
    want_overlap = rng.random() < 0.5
    want_mixed = rng.random() < 0.6
    layout = rng.choice(("horizontal", "vertical"))
    colors = rng.sample(list(OBJECT_COLORS), n_fences)

    if layout == "horizontal":
        overlap_reqs = (("right",), ("left",))
        leak_reqs = (("right",), ("left",))
    else:
        overlap_reqs = (("bottom",), ("top",))
        leak_reqs = (("bottom",), ("top",))

    descs = []
    for i, color in enumerate(colors):
        desired = None
        required_sides: tuple[str, ...] = ()
        if want_overlap and i < 2:
            desired = True
            required_sides = overlap_reqs[i]
        elif want_mixed and i == 0:
            desired = True
            required_sides = leak_reqs[0]
        elif want_mixed and i == 1:
            desired = False
        elif i == 0:
            desired = True
            required_sides = leak_reqs[0]

        desc = _sample_desc(rng, color, desired_outside=desired, required_sides=required_sides)
        if desc is None:
            return None
        descs.append(desc)

    for _ in range(LAYOUT_ATTEMPTS):
        canvas = np.full((30, 30), BG, dtype=int)

        if layout == "horizontal":
            max_h = max(int(desc["fence_h"]) for desc in descs)
            total_w = sum(int(desc["fence_w"]) for desc in descs)
            gap_lo, gap_hi = (1, 2) if want_overlap else (4, 7)
            gaps = [rng.randint(gap_lo, gap_hi) for _ in range(n_fences - 1)]
            total_w += sum(gaps)
            if total_w > 30 or max_h > 30:
                continue
            top = rng.randint(0, 30 - max_h)
            left = rng.randint(0, 30 - total_w)
            cursor = left
            positions = []
            for idx, desc in enumerate(descs):
                positions.append((top, cursor))
                cursor += int(desc["fence_w"])
                if idx < len(gaps):
                    cursor += gaps[idx]
        else:
            max_w = max(int(desc["fence_w"]) for desc in descs)
            total_h = sum(int(desc["fence_h"]) for desc in descs)
            gap_lo, gap_hi = (1, 2) if want_overlap else (4, 7)
            gaps = [rng.randint(gap_lo, gap_hi) for _ in range(n_fences - 1)]
            total_h += sum(gaps)
            if total_h > 30 or max_w > 30:
                continue
            top = rng.randint(0, 30 - total_h)
            left = rng.randint(0, 30 - max_w)
            cursor = top
            positions = []
            for idx, desc in enumerate(descs):
                positions.append((cursor, left))
                cursor += int(desc["fence_h"])
                if idx < len(gaps):
                    cursor += gaps[idx]

        for (top, left), desc in zip(positions, descs):
            _draw_desc(canvas, top, left, desc)

        grid = _crop_canvas(rng, canvas)
        if max(grid.shape) > 30:
            continue
        if _validate_generated(grid, want_overlap=want_overlap, want_mixed=want_mixed):
            return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
