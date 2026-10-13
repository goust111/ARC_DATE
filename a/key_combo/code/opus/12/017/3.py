from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


Coord = tuple[int, int]

_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_GEN_RETRY_LIMIT = 500
_CANVAS_MIN = 15
_CANVAS_MAX = 28
_LAYOUT_OPTIONS = {
    2: [(1, 2, (0, 1)), (2, 1, (0, 1)), (2, 2, (0, 1)), (2, 2, (0, 2)), (2, 2, (0, 3)), (2, 2, (1, 3))],
    3: [(1, 3, (0, 1, 2)), (3, 1, (0, 1, 2)), (2, 2, (0, 1, 2)), (2, 2, (0, 1, 3)), (2, 2, (0, 2, 3)), (2, 2, (1, 2, 3))],
    4: [(1, 4, (0, 1, 2, 3)), (4, 1, (0, 1, 2, 3)), (2, 2, (0, 1, 2, 3))],
}
_PATTERN_SIZE_WEIGHTS = [((2, 2), 4), ((2, 3), 3), ((3, 2), 3), ((3, 3), 2), ((1, 1), 1), ((1, 2), 1), ((2, 1), 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[Coord] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[Coord]]:
    h, w = mask.shape
    dirs = _DIR4 if connectivity == 4 else _DIR8
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _holes_in_object(local_grid: np.ndarray, obj_color: int) -> list[list[Coord]]:
    obj_mask = local_grid == int(obj_color)
    padded = np.pad(obj_mask, 1, constant_values=False)
    free = ~padded
    holes: list[list[Coord]] = []
    for region in _components(free, connectivity=4):
        touches_edge = any(
            r == 0 or c == 0 or r == padded.shape[0] - 1 or c == padded.shape[1] - 1
            for r, c in region
        )
        if touches_edge:
            continue
        holes.append([(r - 1, c - 1) for r, c in region])
    return holes


def _seed_crop_from_hole(local_grid: np.ndarray, hole: list[Coord], obj_color: int) -> tuple[np.ndarray, list[Coord]] | None:
    seed_coords = [(r, c) for r, c in hole if int(local_grid[r, c]) not in (0, int(obj_color))]
    if not seed_coords:
        return None
    bbox = _bbox_from_coords(seed_coords)
    crop = _crop_bbox(local_grid, bbox)
    crop = np.where(crop == int(obj_color), 0, crop)
    return crop, seed_coords


def _analyze_object(grid: np.ndarray, coords: list[Coord], color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = _holes_in_object(local, int(color))
    if not (2 <= len(holes) <= 4):
        return None

    seeded_holes: list[tuple[np.ndarray, list[Coord]]] = []
    for hole in holes:
        payload = _seed_crop_from_hole(local, hole, int(color))
        if payload is not None:
            seeded_holes.append(payload)
    if len(seeded_holes) != 1:
        return None

    pattern, local_seed_coords = seeded_holes[0]
    global_seed_coords = [(r0 + r, c0 + c) for r, c in local_seed_coords]
    return {
        "color": int(color),
        "coords": np.asarray(coords, dtype=int),
        "hole_count": int(len(holes)),
        "pattern": np.asarray(pattern, dtype=int),
        "top_row": int(min(r for r, _c in coords)),
        "seed_coords": global_seed_coords,
    }


def _scene_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        for comp in _components(g == int(color), connectivity=4):
            info = _analyze_object(g, comp, int(color))
            if info is not None:
                objects.append(info)
    return objects


def _tile_pattern(pattern: np.ndarray, copies: int) -> np.ndarray:
    h, w = pattern.shape
    out = np.zeros((h, copies * w + (copies - 1)), dtype=int)
    for idx in range(copies):
        c0 = idx * (w + 1)
        out[:, c0 : c0 + w] = pattern
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each object's enclosed holes, then tile the seeded hole's cropped pattern that many times.

    Input structure:
    - The grid has black background and 1-2 disconnected single-color foreground objects that do not touch the border.
    - Each object is 4-connected and encloses 2-4 4-connected internal cavities separated by 1-pixel walls.
    - Exactly one cavity per object contains a few non-object seed pixels; the other cavities stay pure black.

    Rule:
    - For each foreground object, count all of its enclosed cavities, including the one that contains seed pixels.
    - Inside that unique seeded cavity, crop the seed pixels to their tight bounding box while preserving their colors and internal black gaps.
    - Repeat that cropped seed pattern horizontally as many times as the cavity count, with one black column between copies.
    - If there are multiple objects, order them by the row of their topmost object pixel and stack their tiled rows vertically with one black row between them.

    Key insight:
    - The holes play two different roles at once: every hole contributes to the repeat count, but only the one containing seed pixels supplies the pattern content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _scene_objects(g)
    assert objects, "expected at least one qualifying holed object"
    objects.sort(key=lambda item: item["top_row"])

    rows = [_tile_pattern(obj["pattern"], obj["hole_count"]) for obj in objects]
    total_h = sum(row.shape[0] for row in rows) + len(rows) - 1
    total_w = max(row.shape[1] for row in rows)
    out = np.zeros((total_h, total_w), dtype=int)
    r = 0
    for row in rows:
        out[r : r + row.shape[0], : row.shape[1]] = row
        r += row.shape[0] + 1
    return out


def _build_pattern_bank() -> dict[tuple[int, int], list[np.ndarray]]:
    groups: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for h in range(1, 4):
        for w in range(1, 4):
            total = h * w
            for bits in range(1, 1 << total):
                if bits.bit_count() > 4:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(total):
                    if (bits >> idx) & 1:
                        mask[idx // w, idx % w] = True
                coords = np.argwhere(mask)
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0)
                if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (h, w):
                    continue
                groups[(h, w)].append(mask)
    return groups


_PATTERN_BANK = _build_pattern_bank()


def _choose_weighted(items: list[tuple[tuple[int, int], int]]) -> tuple[int, int]:
    total = sum(weight for _item, weight in items)
    pick = random.randint(1, total)
    acc = 0
    for item, weight in items:
        acc += weight
        if pick <= acc:
            return item
    return items[-1][0]


def _sample_pattern(prefer_large: bool, forbidden_counts: set[int]) -> np.ndarray:
    size_choices = _PATTERN_SIZE_WEIGHTS if prefer_large else [((1, 1), 1), ((1, 2), 1), ((2, 1), 1), ((2, 2), 2), ((2, 3), 2), ((3, 2), 2), ((3, 3), 1)]
    for _ in range(200):
        h, w = _choose_weighted(size_choices)
        mask = random.choice(_PATTERN_BANK[(h, w)])
        if int(mask.sum()) in forbidden_counts:
            continue
        return mask
    h, w = (2, 2)
    return random.choice(_PATTERN_BANK[(h, w)])


def _chamber_starts(lengths: list[int]) -> list[int]:
    starts = []
    pos = 1
    for length in lengths:
        starts.append(pos)
        pos += length + 1
    return starts


def _add_tabs(patch: np.ndarray, obj_color: int) -> np.ndarray:
    out = np.pad(patch, 1, constant_values=0)
    obj_coords = np.argwhere(out == int(obj_color))
    r0, c0, r1, c1 = _bbox_from_coords(obj_coords)
    side_pool = ["top", "bottom", "left", "right"]
    random.shuffle(side_pool)
    for side in side_pool[: random.randint(0, 2)]:
        if side in ("top", "bottom"):
            span = c1 - c0 + 1
            length = random.randint(1, max(1, span // 3))
            start = random.randint(c0, c1 - length + 1)
            rr = r0 - 1 if side == "top" else r1 + 1
            out[rr, start : start + length] = int(obj_color)
        else:
            span = r1 - r0 + 1
            length = random.randint(1, max(1, span // 3))
            start = random.randint(r0, r1 - length + 1)
            cc = c0 - 1 if side == "left" else c1 + 1
            out[start : start + length, cc] = int(obj_color)
    nz = np.argwhere(out != 0)
    return _crop_bbox(out, _bbox_from_coords(nz))


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    transformed = np.rot90(patch, random.randint(0, 3))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    if random.random() < 0.5:
        transformed = np.flipud(transformed)
    return np.asarray(transformed, dtype=int)


def _build_object_patch(hole_count: int, obj_color: int, seed_color: int, force_mismatch: bool) -> tuple[np.ndarray, dict] | None:
    for _ in range(300):
        rows, cols, selected = random.choice(_LAYOUT_OPTIONS[hole_count])
        selected = tuple(int(x) for x in selected)
        seed_cell = random.choice(selected)
        pattern_mask = _sample_pattern(prefer_large=random.random() < 0.75, forbidden_counts={hole_count} if force_mismatch else set())
        ph, pw = pattern_mask.shape

        row_heights = [random.randint(2, 4) for _ in range(rows)]
        col_widths = [random.randint(2, 4) for _ in range(cols)]
        sr = seed_cell // cols
        sc = seed_cell % cols
        row_heights[sr] = max(row_heights[sr], ph)
        col_widths[sc] = max(col_widths[sc], pw)
        if row_heights[sr] == ph and col_widths[sc] == pw:
            if random.random() < 0.5 and row_heights[sr] < 4:
                row_heights[sr] += 1
            elif col_widths[sc] < 4:
                col_widths[sc] += 1
            elif row_heights[sr] < 5:
                row_heights[sr] += 1
            elif col_widths[sc] < 5:
                col_widths[sc] += 1
            else:
                continue

        patch_h = sum(row_heights) + rows + 1
        patch_w = sum(col_widths) + cols + 1
        if patch_h > 20 or patch_w > 20:
            continue

        patch = np.full((patch_h, patch_w), int(obj_color), dtype=int)
        row_starts = _chamber_starts(row_heights)
        col_starts = _chamber_starts(col_widths)
        for cell in selected:
            rr = cell // cols
            cc = cell % cols
            r0 = row_starts[rr]
            c0 = col_starts[cc]
            patch[r0 : r0 + row_heights[rr], c0 : c0 + col_widths[cc]] = 0

        chamber_h = row_heights[sr]
        chamber_w = col_widths[sc]
        if chamber_h < ph or chamber_w < pw:
            continue
        place_options = []
        for off_r in range(chamber_h - ph + 1):
            for off_c in range(chamber_w - pw + 1):
                place_options.append((off_r, off_c))
        if not place_options:
            continue
        off_r, off_c = random.choice(place_options)
        chamber_r0 = row_starts[sr]
        chamber_c0 = col_starts[sc]
        patch[chamber_r0 + off_r : chamber_r0 + off_r + ph, chamber_c0 + off_c : chamber_c0 + off_c + pw][pattern_mask] = int(seed_color)

        if random.random() < 0.7:
            patch = _add_tabs(patch, int(obj_color))
        patch = _transform_patch(patch)

        comps = _components(patch == int(obj_color), connectivity=4)
        if len(comps) != 1:
            continue
        info = _analyze_object(patch, comps[0], int(obj_color))
        if info is None or info["hole_count"] != hole_count:
            continue
        if force_mismatch and int(np.count_nonzero(info["pattern"])) == hole_count:
            continue
        return patch, info
    return None


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 1 or c0 < 1 or r0 + h >= canvas.shape[0] or c0 + w >= canvas.shape[1]:
        return False
    for pr, pc in np.argwhere(patch != 0):
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        if int(canvas[rr, cc]) != 0:
            return False
        for dr, dc in _DIR8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and int(canvas[nr, nc]) != 0:
                return False
    return True


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != 0
    view = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _build_expected(specs: list[dict]) -> np.ndarray:
    rows = [_tile_pattern(spec["pattern"], spec["hole_count"]) for spec in sorted(specs, key=lambda item: item["top_row"])]
    total_h = sum(row.shape[0] for row in rows) + len(rows) - 1
    total_w = max(row.shape[1] for row in rows)
    out = np.zeros((total_h, total_w), dtype=int)
    r = 0
    for row in rows:
        out[r : r + row.shape[0], : row.shape[1]] = row
        r += row.shape[0] + 1
    return out


def _scene_is_semantic(grid: np.ndarray, expected_count: int) -> bool:
    objects = _scene_objects(grid)
    if len(objects) != expected_count:
        return False
    hole_counts = [obj["hole_count"] for obj in objects]
    if len(set(hole_counts)) != len(hole_counts):
        return False
    coverage = np.zeros_like(grid, dtype=bool)
    seed_count_mismatch = False
    for obj in objects:
        coverage[obj["coords"][:, 0], obj["coords"][:, 1]] = True
        for r, c in obj["seed_coords"]:
            coverage[r, c] = True
        if int(np.count_nonzero(obj["pattern"])) != obj["hole_count"]:
            seed_count_mismatch = True
    if np.any((grid != 0) & ~coverage):
        return False
    if not seed_count_mismatch:
        return False
    for obj in objects:
        if obj["top_row"] <= 0:
            return False
        pattern = obj["pattern"]
        if pattern.shape[0] > 3 or pattern.shape[1] > 3:
            return False
        if np.count_nonzero(pattern) > 4:
            return False
    if expected_count == 2:
        top_rows = [obj["top_row"] for obj in objects]
        if len(set(top_rows)) != 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = 1 if random.random() < 0.45 else 2
    hole_counts = random.sample([2, 3, 4], k=object_count)

    palette = list(range(1, 10))
    random.shuffle(palette)
    object_colors = palette[:object_count]
    seed_colors = palette[object_count : 2 * object_count]

    built: list[tuple[np.ndarray, dict]] = []
    mismatch_index = random.randrange(object_count)
    for idx in range(object_count):
        payload = _build_object_patch(
            hole_count=hole_counts[idx],
            obj_color=object_colors[idx],
            seed_color=seed_colors[idx],
            force_mismatch=(idx == mismatch_index),
        )
        if payload is None:
            return None
        built.append(payload)

    min_h = max(_CANVAS_MIN, max(patch.shape[0] for patch, _info in built) + 4)
    min_w = max(_CANVAS_MIN, max(patch.shape[1] for patch, _info in built) + 4)
    if object_count == 2:
        min_h = max(min_h, sum(patch.shape[0] for patch, _info in built) // 2 + 6)
        min_w = max(min_w, max(patch.shape[1] for patch, _info in built) + 6)
    if min_h > _CANVAS_MAX or min_w > _CANVAS_MAX:
        return None

    canvas_h = random.randint(min_h, _CANVAS_MAX)
    canvas_w = random.randint(min_w, _CANVAS_MAX)
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    order = list(range(object_count))
    random.shuffle(order)
    placed_specs: list[dict] = []
    for idx in order:
        patch, info = built[idx]
        choices = [
            (r, c)
            for r in range(1, canvas_h - patch.shape[0])
            for c in range(1, canvas_w - patch.shape[1])
            if _can_place(grid, patch, r, c)
        ]
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        _paste(grid, patch, r0, c0)
        placed_specs.append(
            {
                "hole_count": info["hole_count"],
                "pattern": info["pattern"],
                "top_row": r0 + info["top_row"],
            }
        )

    expected = _build_expected(placed_specs)
    if max(expected.shape) > 30:
        return None
    if not _scene_is_semantic(grid, object_count):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
