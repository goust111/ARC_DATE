import numpy as np
from collections import deque


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "size": int(arr.shape[0]),
                }
            )
    return components


def _fence_info(mask: np.ndarray) -> dict | None:
    h, w = map(int, mask.shape)
    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None
    missing = np.argwhere(border & ~mask)
    if missing.shape[0] != 1:
        return None

    mr, mc = map(int, missing[0])
    side = None
    if mr == 0 and 0 < mc < w - 1:
        side = "top"
        rotation_k = 0
    elif mc == w - 1 and 0 < mr < h - 1:
        side = "right"
        rotation_k = 3
    elif mr == h - 1 and 0 < mc < w - 1:
        side = "bottom"
        rotation_k = 2
    elif mc == 0 and 0 < mr < h - 1:
        side = "left"
        rotation_k = 1
    if side is None:
        return None

    return {
        "side": side,
        "rotation_k": rotation_k,
        "cavity_h": int(h - 2),
        "cavity_w": int(w - 2),
        "cavity_area": int((h - 2) * (w - 2)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate outside seeds from fence-gap directions and plant them into the matching fence interiors.

    Input structure:
    - The background is black.
    - Several monochrome rectangular ring fences appear, each with a one-pixel gap on exactly one side of its border.
    - Several monochrome connected seed shapes sit outside the fences; fence colors are pairwise distinct and seed colors are pairwise distinct.

    Rule:
    - Detect each fence from its bounding-box border and read the gap side: top means 0 degrees, right means 90 degrees clockwise, bottom means 180 degrees, and left means 90 degrees counterclockwise.
    - Sort fences by interior cavity area from small to large, sort seeds by pixel count from small to large, and pair them in that order.
    - Rotate each paired seed by the fence's angle, recolor it to the fence color, and stamp it centered inside the fence interior, using upper-left bias when exact centering is impossible.
    - Keep the original fences, including their gaps, and leave the outside seeds in place.

    Key insight:
    - The gap encodes a rotation command, while the seed-to-fence matching comes from an independent size ordering rather than color or proximity.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    fences = []
    seeds = []
    for comp in _find_components(g, bg):
        info = _fence_info(comp["mask"])
        if info is None:
            seeds.append(comp)
            continue
        fences.append({**comp, **info})

    fences.sort(key=lambda item: (item["cavity_area"], item["bbox"][0], item["bbox"][1]))
    seeds.sort(key=lambda item: (item["size"], item["bbox"][0], item["bbox"][1]))
    assert len(fences) == len(seeds) and fences, "expected equally many fences and seeds"

    out = g.copy()
    for fence, seed in zip(fences, seeds):
        rotated = np.rot90(seed["mask"], k=int(fence["rotation_k"]))
        sh, sw = map(int, rotated.shape)
        assert sh <= fence["cavity_h"] and sw <= fence["cavity_w"], "rotated seed must fit inside the fence cavity"

        r0, c0, _, _ = fence["bbox"]
        top = int(r0 + 1 + (fence["cavity_h"] - sh) // 2)
        left = int(c0 + 1 + (fence["cavity_w"] - sw) // 2)
        out[top : top + sh, left : left + sw][rotated] = int(fence["color"])

    return out


_RNG = np.random.default_rng()
_GRID_MIN = 20
_GRID_MAX = 30
_PAIR_OPTIONS = (2, 3, 4)
_COLOR_POOL = tuple(range(1, 10))
_CAVITY_OPTIONS = {
    16: ((4, 4),),
    20: ((4, 5), (5, 4)),
    24: ((4, 6), (6, 4)),
    25: ((5, 5),),
    28: ((4, 7), (7, 4)),
    30: ((5, 6), (6, 5)),
    35: ((5, 7), (7, 5)),
    36: ((6, 6),),
}
_SEED_COUNTS = (3, 4, 5, 6, 7, 8)
_SIDES = ("top", "right", "bottom", "left")
_MAX_GEN_ATTEMPTS = 300
_PLACE_ATTEMPTS = 400
_SHAPE_ATTEMPTS = 300


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_bbox(
    canvas_h: int,
    canvas_w: int,
    obj_h: int,
    obj_w: int,
    reserved: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    if obj_h + 2 > canvas_h or obj_w + 2 > canvas_w:
        return None

    for _ in range(_PLACE_ATTEMPTS):
        r0 = int(_RNG.integers(1, canvas_h - obj_h))
        c0 = int(_RNG.integers(1, canvas_w - obj_w))
        padded = (r0 - 1, c0 - 1, r0 + obj_h, c0 + obj_w)
        if any(_rectangles_overlap(padded, other) for other in reserved):
            continue
        reserved.append(padded)
        return r0, c0
    return None


def _seed_is_asymmetric(mask: np.ndarray) -> bool:
    base = _mask_key(mask)
    return all(_mask_key(_trim_mask(np.rot90(mask, k))) != base for k in (1, 2, 3))


def _grow_connected_mask(box_h: int, box_w: int, area: int) -> np.ndarray | None:
    start = (int(_RNG.integers(box_h)), int(_RNG.integers(box_w)))
    cells = {start}

    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        chosen = tuple(frontier)[int(_RNG.integers(len(frontier)))]
        cells.add(chosen)

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim_mask(mask)


def _generate_seed_mask(area: int, max_side: int) -> np.ndarray | None:
    for _ in range(_SHAPE_ATTEMPTS):
        box_h = int(_RNG.integers(2, max_side + 1))
        box_w = int(_RNG.integers(2, max_side + 1))
        if area > box_h * box_w:
            continue

        mask = _grow_connected_mask(box_h, box_w, area)
        if mask is None:
            continue
        h, w = map(int, mask.shape)
        if h < 2 or w < 2:
            continue
        if h >= max_side + 1 or w >= max_side + 1:
            continue
        if _fence_info(mask) is not None:
            continue
        if not _seed_is_asymmetric(mask):
            continue
        return mask
    return None


def _make_fence_mask(cavity_h: int, cavity_w: int, side: str) -> np.ndarray:
    h = cavity_h + 2
    w = cavity_w + 2
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    if side == "top":
        mask[0, int(_RNG.integers(1, w - 1))] = False
    elif side == "right":
        mask[int(_RNG.integers(1, h - 1)), -1] = False
    elif side == "bottom":
        mask[-1, int(_RNG.integers(1, w - 1))] = False
    else:
        mask[int(_RNG.integers(1, h - 1)), 0] = False
    return mask


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    grid[top : top + h, left : left + w][mask] = int(color)


def _build_expected_output(
    grid: np.ndarray,
    fences: list[dict],
    seeds: list[dict],
) -> np.ndarray:
    out = grid.copy()
    ordered_fences = sorted(fences, key=lambda item: (item["cavity_area"], item["top"], item["left"]))
    ordered_seeds = sorted(seeds, key=lambda item: (item["area"], item["top"], item["left"]))
    for fence, seed in zip(ordered_fences, ordered_seeds):
        rotated = np.rot90(seed["mask"], k=int(fence["rotation_k"]))
        sh, sw = map(int, rotated.shape)
        top = int(fence["top"] + 1 + (fence["cavity_h"] - sh) // 2)
        left = int(fence["left"] + 1 + (fence["cavity_w"] - sw) // 2)
        out[top : top + sh, left : left + sw][rotated] = int(fence["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(_PAIR_OPTIONS))
    colors = list(_RNG.choice(_COLOR_POOL, size=pair_count * 2, replace=False))
    fence_colors = colors[:pair_count]
    seed_colors = colors[pair_count:]
    sides = list(_RNG.permutation(_SIDES)[:pair_count])

    chosen_areas = list(_RNG.choice(tuple(_CAVITY_OPTIONS.keys()), size=pair_count, replace=False))
    fence_specs = []
    for color, area, side in zip(fence_colors, chosen_areas, sides):
        cavity_h, cavity_w = _CAVITY_OPTIONS[int(area)][int(_RNG.integers(len(_CAVITY_OPTIONS[int(area)])))]
        fence_specs.append(
            {
                "color": int(color),
                "cavity_h": int(cavity_h),
                "cavity_w": int(cavity_w),
                "cavity_area": int(area),
                "side": side,
                "rotation_k": {"top": 0, "right": 3, "bottom": 2, "left": 1}[side],
            }
        )
    fence_specs.sort(key=lambda item: item["cavity_area"])

    seed_areas = sorted(int(x) for x in _RNG.choice(_SEED_COUNTS, size=pair_count, replace=False))
    seed_specs = []
    for color, area, fence in zip(seed_colors, seed_areas, fence_specs):
        max_side = min(int(fence["cavity_h"]), int(fence["cavity_w"])) - 1
        mask = _generate_seed_mask(area, max_side)
        if mask is None:
            return None
        seed_specs.append({"color": int(color), "mask": mask, "area": int(area)})

    canvas_h = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    canvas_w = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    fences_with_pos = []
    for fence in sorted(fence_specs, key=lambda item: -(item["cavity_h"] + 2) * (item["cavity_w"] + 2)):
        mask = _make_fence_mask(int(fence["cavity_h"]), int(fence["cavity_w"]), str(fence["side"]))
        top_left = _place_bbox(canvas_h, canvas_w, int(mask.shape[0]), int(mask.shape[1]), reserved)
        if top_left is None:
            return None
        top, left = top_left
        _paint_mask(grid, mask, top, left, int(fence["color"]))
        fences_with_pos.append({**fence, "mask": mask, "top": int(top), "left": int(left)})

    seeds_with_pos = []
    for seed in sorted(seed_specs, key=lambda item: -item["mask"].shape[0] * item["mask"].shape[1]):
        h, w = map(int, seed["mask"].shape)
        top_left = _place_bbox(canvas_h, canvas_w, h, w, reserved)
        if top_left is None:
            return None
        top, left = top_left
        _paint_mask(grid, seed["mask"], top, left, int(seed["color"]))
        seeds_with_pos.append({**seed, "top": int(top), "left": int(left)})

    expected = _build_expected_output(grid, fences_with_pos, seeds_with_pos)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
