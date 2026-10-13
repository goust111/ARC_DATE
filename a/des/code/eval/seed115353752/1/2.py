from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))

OUTPUT_GAP = 1
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400

PAIR_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (4, 7)
INPUT_SIDE_MIN = 18
PLACEMENT_MARGIN = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == int(bg):
                continue

            color = int(g[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            mask = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
            for r, c in coords:
                mask[r - top, c - left] = True

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (top, left, bottom, right),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    return components


def _rectangle_candidates(mask_a: np.ndarray, mask_b: np.ndarray) -> list[dict]:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    coords_a = [tuple(map(int, rc)) for rc in np.argwhere(mask_a)]
    set_a = set(coords_a)
    area_total = int(mask_a.sum() + mask_b.sum())
    candidates: dict[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]], dict] = {}

    for rot in range(4):
        rot_b = np.rot90(mask_b, k=rot)
        coords_b = [tuple(map(int, rc)) for rc in np.argwhere(rot_b)]
        bh, bw = map(int, rot_b.shape)

        for dr in range(-bh, int(mask_a.shape[0]) + 1):
            for dc in range(-bw, int(mask_a.shape[1]) + 1):
                shifted_b = [(int(r + dr), int(c + dc)) for r, c in coords_b]
                if any(cell in set_a for cell in shifted_b):
                    continue

                min_r = min(0, *(r for r, _c in shifted_b))
                max_r = max(int(mask_a.shape[0]) - 1, *(r for r, _c in shifted_b))
                min_c = min(0, *(c for _r, c in shifted_b))
                max_c = max(int(mask_a.shape[1]) - 1, *(c for _r, c in shifted_b))
                rect_h = int(max_r - min_r + 1)
                rect_w = int(max_c - min_c + 1)
                if rect_h * rect_w != area_total:
                    continue

                norm_a = tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords_a))
                norm_b = tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in shifted_b))
                key = (norm_a, norm_b)
                candidates[key] = {
                    "height": rect_h,
                    "width": rect_w,
                    "rotation": int(rot),
                    "offset": (int(dr), int(dc)),
                }

    return list(candidates.values())


def _assemble_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    candidates = _rectangle_candidates(mask_a, mask_b)
    assert len(candidates) == 1, f"expected a unique assembly, found {len(candidates)}"
    candidate = candidates[0]
    return int(candidate["height"]), int(candidate["width"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, rotate one fragment in each pair to interlock into a rectangle,
    then pack the rectangles left-to-right from largest area to smallest.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    rectangles: list[dict] = []
    for color, pair in by_color.items():
        assert len(pair) == 2, f"color {color} must have exactly two fragments"
        h, w = _assemble_pair(pair[0]["mask"], pair[1]["mask"])
        anchor = min((int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in pair)
        rectangles.append(
            {
                "color": int(color),
                "height": int(h),
                "width": int(w),
                "area": int(h * w),
                "anchor": anchor,
            }
        )

    rectangles.sort(key=lambda item: (-int(item["area"]), item["anchor"][0], item["anchor"][1]))
    out_h = max(int(item["height"]) for item in rectangles)
    out_w = sum(int(item["width"]) for item in rectangles) + OUTPUT_GAP * (len(rectangles) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    col = 0
    for rect in rectangles:
        h = int(rect["height"])
        w = int(rect["width"])
        out[:h, col : col + w] = int(rect["color"])
        col += w + OUTPUT_GAP

    return out


def _random_rectangle_dims(used_areas: set[int]) -> tuple[int, int] | None:
    candidates = []
    for h in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
        for w in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
            area = h * w
            if area in used_areas:
                continue
            candidates.append((h, w))
    return random.choice(candidates) if candidates else None


def _make_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    orientation = random.choice(("vertical", "horizontal"))

    if orientation == "vertical":
        cuts = [random.randint(1, w - 1) for _ in range(h)]
        if len(set(cuts)) < 2:
            return None
        mask_a = np.zeros((h, w), dtype=bool)
        for r, cut in enumerate(cuts):
            mask_a[int(r), : int(cut)] = True
    else:
        cuts = [random.randint(1, h - 1) for _ in range(w)]
        if len(set(cuts)) < 2:
            return None
        mask_a = np.zeros((h, w), dtype=bool)
        for c, cut in enumerate(cuts):
            mask_a[: int(cut), int(c)] = True

    area_a = int(mask_a.sum())
    area_b = int(h * w - area_a)
    if min(area_a, area_b) < max(5, (h * w) // 4):
        return None

    mask_b = ~mask_a
    part_a = _crop_mask(mask_a)
    part_b = _crop_mask(mask_b)
    if len(_rectangle_candidates(part_a, part_b)) != 1:
        return None
    return part_a, part_b


def _make_pair(color: int, bg: int, used_areas: set[int]) -> dict | None:
    dims = _random_rectangle_dims(used_areas)
    if dims is None:
        return None
    h, w = dims

    for _ in range(80):
        parts = _make_partition(h, w)
        if parts is None:
            continue
        mask_a, mask_b = parts
        rot_a = random.randint(0, 3)
        rot_b = random.randint(0, 3)
        piece_a = np.rot90(mask_a, k=rot_a)
        piece_b = np.rot90(mask_b, k=rot_b)
        rel_rot = (rot_b - rot_a) % 4

        if rel_rot not in {0, 1, 2, 3}:
            continue

        used_areas.add(h * w)
        return {
            "color": int(color),
            "height": int(h),
            "width": int(w),
            "area": int(h * w),
            "patches": [
                _mask_to_patch(piece_a, color, bg),
                _mask_to_patch(piece_b, color, bg),
            ],
        }

    return None


def _build_expected(rectangles: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: -int(item["area"]))
    out_h = max(int(item["height"]) for item in ordered)
    out_w = sum(int(item["width"]) for item in ordered) + OUTPUT_GAP * (len(ordered) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    col = 0
    for rect in ordered:
        h = int(rect["height"])
        w = int(rect["width"])
        out[:h, col : col + w] = int(rect["color"])
        col += w + OUTPUT_GAP

    return out


def _sample_canvas(patches: list[np.ndarray], total_fg: int, max_area: int) -> tuple[int, int] | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    lower = max(INPUT_SIDE_MIN, max_h + 2 * PLACEMENT_MARGIN + 1, max_w + 2 * PLACEMENT_MARGIN + 1)
    if lower > MAX_GRID_SIDE:
        return None

    for _ in range(40):
        h = random.randint(lower, MAX_GRID_SIDE)
        w = random.randint(lower, MAX_GRID_SIDE)
        if h * w - total_fg <= max_area:
            continue
        return int(h), int(w)
    return None


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    rows, cols = blocked.shape
    if top < 0 or left < 0 or top + h > rows or left + w > cols:
        return False
    r0 = max(0, top - PLACEMENT_MARGIN)
    c0 = max(0, left - PLACEMENT_MARGIN)
    r1 = min(rows, top + h + PLACEMENT_MARGIN)
    c1 = min(cols, left + w + PLACEMENT_MARGIN)
    return not bool(blocked[r0:r1, c0:c1].any())


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != int(bg)
    region[mask] = patch[mask]


def _place_patches(patches: list[np.ndarray], bg: int, shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    ordered = list(patches)
    random.shuffle(ordered)
    ordered.sort(key=lambda patch: (int(patch.shape[0] * patch.shape[1]), int(patch.shape[0])), reverse=True)

    for patch in ordered:
        ph, pw = map(int, patch.shape)
        positions = [
            (r, c)
            for r in range(h - ph + 1)
            for c in range(w - pw + 1)
            if _can_place(blocked, r, c, ph, pw)
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        _paste_patch(grid, patch, top, left, bg)
        blocked[top : top + ph, left : left + pw] = True

    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    pair_count = random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    colors = random.sample([c for c in PALETTE if c != bg], pair_count)

    rectangles: list[dict] = []
    used_areas: set[int] = set()
    for color in colors:
        pair = _make_pair(int(color), int(bg), used_areas)
        if pair is None:
            return None
        rectangles.append(pair)

    expected = _build_expected(rectangles, int(bg))
    if expected.shape[0] > MAX_GRID_SIDE or expected.shape[1] > MAX_GRID_SIDE:
        return None

    patches = [patch for rect in rectangles for patch in rect["patches"]]
    total_fg = sum(int(rect["area"]) for rect in rectangles)
    max_area = max(int(rect["area"]) for rect in rectangles)
    canvas = _sample_canvas(patches, total_fg, max_area)
    if canvas is None:
        return None

    grid = _place_patches(patches, int(bg), canvas)
    if grid is None:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
