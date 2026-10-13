import itertools
import random
from collections import Counter, deque

import numpy as np


WALL_COLOR = 9
TOOTH_COLOR = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
WALL_THICKNESS = 3
HOLE_COUNT_RANGE = (2, 4)
MAX_KEYS_PER_SAMPLE = 4
PATTERN_OUTWARD_DEPTH = 2
PATTERN_PARALLEL_MARGIN = 1
GEN_RETRIES = 400
PLACE_RETRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return (tuple(int(x) for x in mask.shape), tuple(int(x) for x in mask.astype(np.uint8).ravel()))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_mask(mask: np.ndarray, orientation: str) -> np.ndarray:
    return np.fliplr(mask) if orientation == "vertical" else np.flipud(mask)


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, str, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best = None
    for color in [int(c) for c in np.unique(g) if int(c) != int(bg)]:
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)

        if r0 == 0 and r1 == h - 1 and c1 - c0 + 1 <= min(5, w // 2):
            strip = g[:, c0 : c1 + 1]
            if bool(np.isin(strip, [bg, color]).all()):
                score = (int(coords.shape[0]), -(c1 - c0 + 1), -abs((c0 + c1) - (w - 1)))
                cand = (score, int(color), "vertical", (c0, c1))
                if best is None or cand[0] > best[0]:
                    best = cand

        if c0 == 0 and c1 == w - 1 and r1 - r0 + 1 <= min(5, h // 2):
            strip = g[r0 : r1 + 1, :]
            if bool(np.isin(strip, [bg, color]).all()):
                score = (int(coords.shape[0]), -(r1 - r0 + 1), -abs((r0 + r1) - (h - 1)))
                cand = (score, int(color), "horizontal", (r0, r1))
                if best is None or cand[0] > best[0]:
                    best = cand

    assert best is not None, "failed to detect central wall"
    return int(best[1]), str(best[2]), (int(best[3][0]), int(best[3][1]))


def _find_holes(grid: np.ndarray, bg: int, orientation: str, wall_bounds: tuple[int, int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if orientation == "vertical":
        c0, c1 = wall_bounds
        local_mask = g[:, c0 : c1 + 1] == bg
        row_offset = 0
        col_offset = c0
    else:
        r0, r1 = wall_bounds
        local_mask = g[r0 : r1 + 1, :] == bg
        row_offset = r0
        col_offset = 0

    holes: list[dict] = []
    for comp in _components(local_mask):
        mask = _coords_to_mask(comp)
        rr0, cc0, rr1, cc1 = _bbox(comp)
        holes.append(
            {
                "bbox": (
                    int(rr0 + row_offset),
                    int(cc0 + col_offset),
                    int(rr1 + row_offset),
                    int(cc1 + col_offset),
                ),
                "mask": mask,
                "key": _shape_key(mask),
            }
        )
    return holes


def _infer_tooth_color(grid: np.ndarray, comps: list[list[tuple[int, int]]], exclude: set[int]) -> int:
    if not comps:
        return TOOTH_COLOR

    totals: Counter[int] = Counter()
    presence: Counter[int] = Counter()
    for comp in comps:
        colors_here: set[int] = set()
        for r, c in comp:
            color = int(grid[r, c])
            if color in exclude:
                continue
            totals[color] += 1
            colors_here.add(color)
        for color in colors_here:
            presence[color] += 1

    candidates = [color for color, count in presence.items() if int(count) == len(comps)]
    if candidates:
        return int(max(candidates, key=lambda color: (int(totals[color]), -int(color))))
    return int(max(totals, key=totals.get))


def _component_side(bbox: tuple[int, int, int, int], orientation: str, wall_bounds: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if orientation == "vertical":
        wall_c0, wall_c1 = wall_bounds
        if c1 < wall_c0:
            return "left"
        if c0 > wall_c1:
            return "right"
    else:
        wall_r0, wall_r1 = wall_bounds
        if r1 < wall_r0:
            return "top"
        if r0 > wall_r1:
            return "bottom"
    raise AssertionError("component overlaps the wall unexpectedly")


def _project_pixels(
    pattern_pixels: list[tuple[int, int, int]],
    tooth_bbox: tuple[int, int, int, int],
    hole_bbox: tuple[int, int, int, int],
    orientation: str,
    side: str,
    wall_bounds: tuple[int, int],
) -> list[tuple[int, int, int]]:
    tr0, tc0, tr1, tc1 = tooth_bbox
    tooth_h = int(tr1 - tr0 + 1)
    tooth_w = int(tc1 - tc0 + 1)

    if orientation == "vertical":
        wall_c0, wall_c1 = wall_bounds
        target_r0 = int(hole_bbox[0])
        target_c0 = int(wall_c1 + 1 if side == "left" else wall_c0 - tooth_w)
        return [
            (
                int(target_r0 + (r - tr0)),
                int(target_c0 + (tooth_w - 1 - (c - tc0))),
                int(color),
            )
            for r, c, color in pattern_pixels
        ]

    wall_r0, wall_r1 = wall_bounds
    target_r0 = int(wall_r1 + 1 if side == "top" else wall_r0 - tooth_h)
    target_c0 = int(hole_bbox[1])
    return [
        (
            int(target_r0 + (tooth_h - 1 - (r - tr0))),
            int(target_c0 + (c - tc0)),
            int(color),
        )
        for r, c, color in pattern_pixels
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey key-tooth to the mirrored lockhole inside the central wall,
    erase the key, and project its colored backside pattern to the opposite side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = _get_bg(g)
    wall_color, orientation, wall_bounds = _find_wall(g, bg)

    key_mask = (g != bg) & (g != wall_color)
    comps = _components(key_mask)
    tooth_color = _infer_tooth_color(g, comps, {bg, wall_color})
    holes = _find_holes(g, bg, orientation, wall_bounds)
    hole_lookup = {hole["key"]: hole for hole in holes}

    clear_mask = np.zeros((h, w), dtype=bool)
    projected: list[tuple[int, int, int]] = []

    for comp in comps:
        comp_bbox = _bbox(comp)
        side = _component_side(comp_bbox, orientation, wall_bounds)
        tooth_coords = [(r, c) for r, c in comp if int(g[r, c]) == int(tooth_color)]
        if not tooth_coords:
            continue
        tooth_bbox = _bbox(tooth_coords)
        tooth_mask = _coords_to_mask(tooth_coords)
        match_key = _shape_key(_mirror_mask(tooth_mask, orientation))
        hole = hole_lookup.get(match_key)
        if hole is None:
            continue

        for r, c in comp:
            clear_mask[r, c] = True

        pattern_pixels = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) != int(tooth_color)]
        projected.extend(_project_pixels(pattern_pixels, tooth_bbox, hole["bbox"], orientation, side, wall_bounds))

    out = g.copy()
    out[clear_mask] = bg
    for r, c, color in projected:
        if 0 <= r < h and 0 <= c < w and int(out[r, c]) == int(bg):
            out[r, c] = int(color)
    return out


def _is_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    pts = set(coords)
    start = next(iter(pts))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _build_shape_catalog() -> list[np.ndarray]:
    shapes: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for h in range(2, 6):
        for w in range(2, WALL_THICKNESS + 1):
            cells = [(r, c) for r in range(h) for c in range(w)]
            for area in range(3, min(6, h * w) + 1):
                for combo in itertools.combinations(cells, area):
                    if not _is_connected(combo):
                        continue
                    mask = _coords_to_mask(list(combo))
                    mh, mw = mask.shape
                    if mh < 2 or mw < 2:
                        continue
                    if int(mask.sum()) == mh * mw:
                        continue
                    if bool(np.array_equal(mask, np.fliplr(mask))):
                        continue
                    key = _shape_key(mask)
                    shapes[key] = mask
    catalog = list(shapes.values())
    assert len(catalog) >= 12
    return catalog


VERTICAL_SHAPES = _build_shape_catalog()
HORIZONTAL_SHAPES = [mask.T.copy() for mask in VERTICAL_SHAPES if not bool(np.array_equal(mask.T, np.flipud(mask.T)))]


def _orientation_shapes(orientation: str) -> list[np.ndarray]:
    return VERTICAL_SHAPES if orientation == "vertical" else HORIZONTAL_SHAPES


def _wall_bounds_for_shape(height: int, width: int, orientation: str) -> tuple[int, int]:
    if orientation == "vertical":
        c0 = width // 2 - WALL_THICKNESS // 2
        return c0, c0 + WALL_THICKNESS - 1
    r0 = height // 2 - WALL_THICKNESS // 2
    return r0, r0 + WALL_THICKNESS - 1


def _paint_wall(grid: np.ndarray, orientation: str, wall_bounds: tuple[int, int], wall_color: int) -> None:
    if orientation == "vertical":
        c0, c1 = wall_bounds
        grid[:, c0 : c1 + 1] = int(wall_color)
    else:
        r0, r1 = wall_bounds
        grid[r0 : r1 + 1, :] = int(wall_color)


def _place_holes(
    grid: np.ndarray,
    hole_masks: list[np.ndarray],
    orientation: str,
    wall_bounds: tuple[int, int],
    bg: int,
) -> list[dict] | None:
    h, w = grid.shape
    strip_h = h if orientation == "vertical" else WALL_THICKNESS
    strip_w = WALL_THICKNESS if orientation == "vertical" else w
    blocked = np.zeros((strip_h, strip_w), dtype=bool)
    holes: list[dict] = []
    order = list(range(len(hole_masks)))
    random.shuffle(order)

    for idx in order:
        mask = hole_masks[idx]
        mh, mw = mask.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            if orientation == "vertical":
                r0 = random.randint(2, h - mh - 3)
                c0 = random.randint(0, WALL_THICKNESS - mw)
            else:
                r0 = random.randint(0, WALL_THICKNESS - mh)
                c0 = random.randint(2, w - mw - 3)

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(strip_h, r0 + mh + 1)
            cc1 = min(strip_w, c0 + mw + 1)
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue

            blocked[rr0:rr1, cc0:cc1] = True
            coords = [(r0 + r, c0 + c) for r, c in np.argwhere(mask)]
            if orientation == "vertical":
                gc0 = wall_bounds[0] + c0
                for rr, cc in coords:
                    grid[rr, wall_bounds[0] + cc] = int(bg)
                holes.append(
                    {
                        "bbox": (int(r0), int(gc0), int(r0 + mh - 1), int(gc0 + mw - 1)),
                        "mask": mask.copy(),
                        "key": _shape_key(mask),
                    }
                )
            else:
                gr0 = wall_bounds[0] + r0
                for rr, cc in coords:
                    grid[wall_bounds[0] + rr, cc] = int(bg)
                holes.append(
                    {
                        "bbox": (int(gr0), int(c0), int(gr0 + mh - 1), int(c0 + mw - 1)),
                        "mask": mask.copy(),
                        "key": _shape_key(mask),
                    }
                )
            placed = True
            break
        if not placed:
            return None

    holes.sort(key=lambda hole: hole["bbox"])
    return holes


def _make_pattern_local(mask: np.ndarray, side: str, bg: int) -> list[tuple[int, int, int]] | None:
    tooth_area = int(mask.sum())
    target = random.randint(1, min(4, max(1, tooth_area - 1)))
    h, w = mask.shape

    if side == "left":
        anchors = [(r, -1) for r in range(h) if bool(mask[r, 0])]
        region = {(r, c) for r in range(-PATTERN_PARALLEL_MARGIN, h + PATTERN_PARALLEL_MARGIN) for c in range(-PATTERN_OUTWARD_DEPTH, 0)}
    elif side == "right":
        anchors = [(r, w) for r in range(h) if bool(mask[r, w - 1])]
        region = {(r, c) for r in range(-PATTERN_PARALLEL_MARGIN, h + PATTERN_PARALLEL_MARGIN) for c in range(w, w + PATTERN_OUTWARD_DEPTH)}
    elif side == "top":
        anchors = [(-1, c) for c in range(w) if bool(mask[0, c])]
        region = {(r, c) for r in range(-PATTERN_OUTWARD_DEPTH, 0) for c in range(-PATTERN_PARALLEL_MARGIN, w + PATTERN_PARALLEL_MARGIN)}
    else:
        anchors = [(h, c) for c in range(w) if bool(mask[h - 1, c])]
        region = {(r, c) for r in range(h, h + PATTERN_OUTWARD_DEPTH) for c in range(-PATTERN_PARALLEL_MARGIN, w + PATTERN_PARALLEL_MARGIN)}

    if not anchors:
        return None

    pattern = {random.choice(anchors)}
    while len(pattern) < target:
        frontier = set()
        for r, c in pattern:
            for dr, dc in DIR4:
                nxt = (int(r + dr), int(c + dc))
                if nxt in region and nxt not in pattern:
                    frontier.add(nxt)
        if not frontier:
            break
        pattern.add(random.choice(sorted(frontier)))

    palette = [color for color in range(10) if color not in {bg, WALL_COLOR, TOOTH_COLOR}]
    random.shuffle(palette)
    color_count = random.randint(1, min(3, len(pattern)))
    colors = palette[:color_count]
    pattern_list = []
    cells = list(pattern)
    random.shuffle(cells)
    for idx, (r, c) in enumerate(cells):
        color = colors[0] if idx == 0 else random.choice(colors)
        pattern_list.append((int(r), int(c), int(color)))
    return pattern_list


def _assemble_key(mask: np.ndarray, side: str, bg: int) -> dict | None:
    tooth_cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    pattern_local = _make_pattern_local(mask, side, bg)
    if pattern_local is None:
        return None

    all_cells = tooth_cells + [(r, c) for r, c, _ in pattern_local]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    shift_r = -min_r
    shift_c = -min_c

    norm_tooth = [(r + shift_r, c + shift_c) for r, c in tooth_cells]
    norm_pattern = [(r + shift_r, c + shift_c, color) for r, c, color in pattern_local]
    tooth_bbox = (shift_r, shift_c, shift_r + mask.shape[0] - 1, shift_c + mask.shape[1] - 1)

    return {
        "height": int(max_r - min_r + 1),
        "width": int(max_c - min_c + 1),
        "tooth_cells": norm_tooth,
        "pattern_cells": norm_pattern,
        "tooth_bbox_local": tuple(int(x) for x in tooth_bbox),
        "tooth_mask": mask.copy(),
    }


def _coords_clear(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    unique = set(coords)
    if len(unique) != len(coords):
        return False
    for r, c in unique:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if bool(occupied[rr, cc]):
                    return False
    return True


def _place_key(
    grid: np.ndarray,
    occupied: np.ndarray,
    key: dict,
    orientation: str,
    side: str,
    wall_bounds: tuple[int, int],
) -> dict | None:
    h, w = grid.shape
    key_h = int(key["height"])
    key_w = int(key["width"])

    if orientation == "vertical":
        wall_c0, wall_c1 = wall_bounds
        row_min, row_max = 1, h - key_h - 2
        if side == "left":
            col_min, col_max = 1, wall_c0 - key_w - 2
        else:
            col_min, col_max = wall_c1 + 2, w - key_w - 2
    else:
        wall_r0, wall_r1 = wall_bounds
        col_min, col_max = 1, w - key_w - 2
        if side == "top":
            row_min, row_max = 1, wall_r0 - key_h - 2
        else:
            row_min, row_max = wall_r1 + 2, h - key_h - 2

    if row_min > row_max or col_min > col_max:
        return None

    for _ in range(PLACE_RETRIES):
        r0 = random.randint(row_min, row_max)
        c0 = random.randint(col_min, col_max)
        abs_tooth = [(r0 + r, c0 + c) for r, c in key["tooth_cells"]]
        abs_pattern = [(r0 + r, c0 + c, color) for r, c, color in key["pattern_cells"]]
        if not _coords_clear(abs_tooth + [(r, c) for r, c, _ in abs_pattern], occupied):
            continue

        for r, c in abs_tooth:
            grid[r, c] = TOOTH_COLOR
            occupied[r, c] = True
        for r, c, color in abs_pattern:
            grid[r, c] = int(color)
            occupied[r, c] = True

        tr0, tc0, tr1, tc1 = key["tooth_bbox_local"]
        return {
            "all_cells": abs_tooth + [(r, c) for r, c, _ in abs_pattern],
            "pattern_cells": abs_pattern,
            "tooth_bbox": (int(r0 + tr0), int(c0 + tc0), int(r0 + tr1), int(c0 + tc1)),
        }
    return None


def _expected_output(
    grid: np.ndarray,
    key_infos: list[dict],
    orientation: str,
    side: str,
    wall_bounds: tuple[int, int],
) -> np.ndarray | None:
    h, w = grid.shape
    out = grid.copy()
    projected: dict[tuple[int, int], int] = {}
    for info in key_infos:
        for r, c in info["all_cells"]:
            out[r, c] = _get_bg(grid)

        pixels = _project_pixels(
            info["pattern_cells"],
            info["tooth_bbox"],
            info["hole_bbox"],
            orientation,
            side,
            wall_bounds,
        )
        for r, c, color in pixels:
            if not (0 <= r < h and 0 <= c < w):
                return None
            if int(grid[r, c]) != int(_get_bg(grid)):
                return None
            if (r, c) in projected:
                return None
            projected[(r, c)] = int(color)

    for (r, c), color in projected.items():
        out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    orientation = random.choice(["vertical", "horizontal"])
    side = random.choice(["left", "right"] if orientation == "vertical" else ["top", "bottom"])
    bg = random.choice([color for color in range(10) if color not in {WALL_COLOR, TOOTH_COLOR}])

    grid = np.full((height, width), bg, dtype=int)
    wall_bounds = _wall_bounds_for_shape(height, width, orientation)
    _paint_wall(grid, orientation, wall_bounds, WALL_COLOR)

    shapes = _orientation_shapes(orientation)
    hole_count = random.randint(*HOLE_COUNT_RANGE)
    hole_masks = random.sample(shapes, hole_count)
    holes = _place_holes(grid, hole_masks, orientation, wall_bounds, bg)
    if holes is None:
        return None

    key_count = random.randint(1, min(hole_count, MAX_KEYS_PER_SAMPLE))
    matched_holes = random.sample(holes, key_count)
    random.shuffle(matched_holes)

    occupied = grid != bg
    key_infos: list[dict] = []
    for hole in matched_holes:
        tooth_mask = _mirror_mask(hole["mask"], orientation)
        key = _assemble_key(tooth_mask, side, bg)
        if key is None:
            return None
        placed = _place_key(grid, occupied, key, orientation, side, wall_bounds)
        if placed is None:
            return None
        placed["hole_bbox"] = hole["bbox"]
        key_infos.append(placed)

    expected = _expected_output(grid, key_infos, orientation, side, wall_bounds)
    if expected is None:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid puzzle-7 sample")
