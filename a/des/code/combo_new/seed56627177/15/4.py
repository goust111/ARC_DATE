import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 300
PLACE_ATTEMPTS = 200


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "area": int(arr.shape[0]),
                    "coords": arr,
                    "bbox": _bbox(arr),
                }
            )
    return comps


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    height, width = passable.shape
    exterior = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(height):
        for c in (0, width - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, height, width):
            if passable[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _analyze_large_object(grid: np.ndarray, comp: dict) -> dict:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    obj_mask = np.zeros_like(patch, dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    obj_mask[rel[:, 0], rel[:, 1]] = True

    non_object = ~obj_mask
    exterior = _flood_from_border(non_object)
    enclosed = non_object & ~exterior
    hole_comps = _mask_components(enclosed)
    assert len(hole_comps) == 1, "each large object must have exactly one enclosed cavity"
    hole = hole_comps[0]

    marker_cells: list[tuple[int, int, int]] = []
    for rr, cc in hole:
        value = int(patch[rr, cc])
        if value != BG:
            marker_cells.append((int(rr), int(cc), value))
    assert len(marker_cells) == 1, "each hole must contain exactly one marker pixel"
    marker_r, marker_c, marker_color = marker_cells[0]
    assert marker_color != color, "marker color must differ from object color"

    hole_values = {int(patch[rr, cc]) for rr, cc in hole}
    assert hole_values <= {BG, marker_color}, "hole may only contain background plus its marker"

    hole_cells = [(int(rr), int(cc)) for rr, cc in hole]
    rel_cells = [(rr - marker_r, cc - marker_c) for rr, cc in hole_cells]
    return {
        "object_color": color,
        "bbox": (r0, c0, r1, c1),
        "hole_cells_local": hole_cells,
        "marker_local": (marker_r, marker_c),
        "marker_global": (r0 + marker_r, c0 + marker_c),
        "marker_color": marker_color,
        "stamp_offsets": rel_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each large object and restamp its enclosed hole shape onto the matching outside anchor.

    Input structure:
    - The background is black.
    - The grid contains several large single-color 4-connected objects, each with one enclosed cavity inside its bbox.
    - Every cavity contains exactly one non-black marker pixel, and the same marker color appears once more as an isolated outside anchor pixel.

    Rule:
    - Find every large monochrome object and isolate its unique enclosed cavity as the template footprint.
    - Treat the whole enclosed cavity footprint, including the marker cell's location, as the stamp shape.
    - Match the cavity's marker color to the outside anchor of the same color, then translate the template so the marker position lands on that anchor.
    - Clear the original objects, their internal markers, and the anchor pixels; draw only the translated cavity templates in the source object colors.

    Key insight:
    - The answer is defined by each object's negative space rather than by the object itself, and the marker color acts as a remote address for where that hole-shape template must be stamped.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    comps = _color_components(g, bg=BG)
    large = [comp for comp in comps if int(comp["area"]) >= 9]
    assert large, "expected at least one large object"

    infos = [_analyze_large_object(g, comp) for comp in large]
    marker_colors = [int(info["marker_color"]) for info in infos]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be unique across objects"

    out = np.full((height, width), BG, dtype=int)
    for info in infos:
        marker_color = int(info["marker_color"])
        positions = np.argwhere(g == marker_color)
        assert positions.shape[0] == 2, "marker color must appear exactly twice"
        marker_pos = np.array(info["marker_global"], dtype=int)
        if np.array_equal(positions[0], marker_pos):
            anchor = positions[1]
        else:
            anchor = positions[0]

        for dr, dc in info["stamp_offsets"]:
            rr = int(anchor[0]) + int(dr)
            cc = int(anchor[1]) + int(dc)
            assert 0 <= rr < height and 0 <= cc < width, "stamp must stay inside the grid"
            prev = int(out[rr, cc])
            assert prev in (BG, int(info["object_color"])), "translated templates may not conflict"
            out[rr, cc] = int(info["object_color"])
    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        box_h = rng.randint(2, 5)
        box_w = rng.randint(2, 5)
        target = rng.randint(4, min(8, box_h * box_w))
        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, box_h, box_w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        arr = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(arr)
        arr[:, 0] -= r0
        arr[:, 1] -= c0
        hole_h = int(r1 - r0 + 1)
        hole_w = int(c1 - c0 + 1)
        if hole_h < 2 or hole_w < 2 or arr.shape[0] < 4:
            continue
        mask = np.zeros((hole_h, hole_w), dtype=bool)
        mask[arr[:, 0], arr[:, 1]] = True
        exterior = _flood_from_border(~mask)
        if np.any((~mask) & ~exterior):
            continue
        return mask
    raise RuntimeError("failed to sample a connected hole mask")


def _crop_nonzero(patch: np.ndarray, hole_cells: list[tuple[int, int]], marker: tuple[int, int]):
    coords = np.argwhere(patch != BG)
    r0, c0, r1, c1 = _bbox(coords)
    cropped = patch[r0 : r1 + 1, c0 : c1 + 1].copy()
    new_hole = [(rr - r0, cc - c0) for rr, cc in hole_cells]
    new_marker = (marker[0] - r0, marker[1] - c0)
    return cropped, new_hole, new_marker


def _build_object_patch(rng: random.Random, object_color: int, marker_color: int) -> dict:
    for _ in range(80):
        hole_mask = _random_connected_mask(rng)
        hole_h, hole_w = hole_mask.shape

        pad_t = rng.randint(1, 3)
        pad_b = rng.randint(1, 3)
        pad_l = rng.randint(1, 3)
        pad_r = rng.randint(1, 3)
        ext_t = rng.randint(0, 2)
        ext_b = rng.randint(0, 2)
        ext_l = rng.randint(0, 2)
        ext_r = rng.randint(0, 2)

        base_h = pad_t + hole_h + pad_b
        base_w = pad_l + hole_w + pad_r
        canvas_h = base_h + ext_t + ext_b
        canvas_w = base_w + ext_l + ext_r
        canvas = np.zeros((canvas_h, canvas_w), dtype=int)

        rect_r0 = ext_t
        rect_c0 = ext_l
        canvas[rect_r0 : rect_r0 + base_h, rect_c0 : rect_c0 + base_w] = object_color

        def add_bar(side: str, depth: int):
            if depth == 0:
                return
            if side in ("top", "bottom"):
                min_len = max(2, base_w // 3)
                seg_len = rng.randint(min_len, base_w)
                seg_c0 = rect_c0 + rng.randint(0, base_w - seg_len)
                if side == "top":
                    canvas[0:rect_r0, seg_c0 : seg_c0 + seg_len] = object_color
                else:
                    canvas[rect_r0 + base_h :, seg_c0 : seg_c0 + seg_len] = object_color
            else:
                min_len = max(2, base_h // 3)
                seg_len = rng.randint(min_len, base_h)
                seg_r0 = rect_r0 + rng.randint(0, base_h - seg_len)
                if side == "left":
                    canvas[seg_r0 : seg_r0 + seg_len, 0:rect_c0] = object_color
                else:
                    canvas[seg_r0 : seg_r0 + seg_len, rect_c0 + base_w :] = object_color

        if rng.random() < 0.8:
            add_bar("top", ext_t)
        if rng.random() < 0.8:
            add_bar("bottom", ext_b)
        if rng.random() < 0.8:
            add_bar("left", ext_l)
        if rng.random() < 0.8:
            add_bar("right", ext_r)

        hole_r0 = rect_r0 + pad_t
        hole_c0 = rect_c0 + pad_l
        hole_cells = [(hole_r0 + rr, hole_c0 + cc) for rr, cc in np.argwhere(hole_mask)]
        for rr, cc in hole_cells:
            canvas[rr, cc] = BG
        marker = rng.choice(hole_cells)
        canvas[marker[0], marker[1]] = marker_color

        patch, hole_cells_local, marker_local = _crop_nonzero(canvas, hole_cells, marker)
        object_area = int(np.sum(patch == object_color))
        if object_area < 9:
            continue

        test_comps = [comp for comp in _color_components(patch, bg=BG) if int(comp["area"]) >= 9]
        if len(test_comps) != 1:
            continue
        info = _analyze_large_object(patch, test_comps[0])
        if int(info["marker_color"]) != marker_color:
            continue

        rels = tuple(sorted(info["stamp_offsets"]))
        return {
            "patch": patch,
            "object_color": object_color,
            "marker_color": marker_color,
            "stamp_offsets": rels,
            "stamp_bbox_rel": (
                min(dr for dr, _ in rels),
                min(dc for _, dc in rels),
                max(dr for dr, _ in rels),
                max(dc for _, dc in rels),
            ),
        }
    raise RuntimeError("failed to build a valid object patch")


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    height, width = patch.shape
    sub = grid[top : top + height, left : left + width]
    mask = patch != BG
    sub[mask] = patch[mask]


def _expected_output(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for item in placed:
        color = int(item["object_color"])
        anchor_r, anchor_c = item["anchor"]
        for dr, dc in item["stamp_offsets"]:
            rr = anchor_r + dr
            cc = anchor_c + dc
            prev = int(out[rr, cc])
            assert prev in (BG, color)
            out[rr, cc] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choice((1, 2, 3))
    palette = list(COLORS)
    rng.shuffle(palette)
    object_colors = palette[:object_count]
    marker_colors = palette[object_count : 2 * object_count]

    objects = [
        _build_object_patch(rng, object_color, marker_color)
        for object_color, marker_color in zip(object_colors, marker_colors)
    ]
    objects.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    min_side = {1: 18, 2: 22, 3: 25}[object_count]
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)
    grid = np.zeros((height, width), dtype=int)

    object_boxes: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for item in objects:
        patch = item["patch"]
        ph, pw = patch.shape
        if ph > height or pw > width:
            return None
        for _ in range(PLACE_ATTEMPTS):
            top = rng.randint(0, height - ph)
            left = rng.randint(0, width - pw)
            bbox = (top, left, top + ph - 1, left + pw - 1)
            if any(_boxes_conflict(bbox, other, gap=1) for other in object_boxes):
                continue
            _place_patch(grid, patch, top, left)
            object_boxes.append(bbox)
            placed.append(
                {
                    "object_color": int(item["object_color"]),
                    "marker_color": int(item["marker_color"]),
                    "stamp_offsets": item["stamp_offsets"],
                    "stamp_bbox_rel": item["stamp_bbox_rel"],
                }
            )
            break
        else:
            return None

    stamp_boxes: list[tuple[int, int, int, int]] = []
    rng.shuffle(placed)
    for item in placed:
        min_dr, min_dc, max_dr, max_dc = item["stamp_bbox_rel"]
        r_min = -min_dr
        r_max = height - 1 - max_dr
        c_min = -min_dc
        c_max = width - 1 - max_dc
        if r_min > r_max or c_min > c_max:
            return None

        for _ in range(PLACE_ATTEMPTS):
            anchor_r = rng.randint(r_min, r_max)
            anchor_c = rng.randint(c_min, c_max)
            bbox = (
                anchor_r + min_dr,
                anchor_c + min_dc,
                anchor_r + max_dr,
                anchor_c + max_dc,
            )
            if any(_boxes_conflict(bbox, other, gap=1) for other in object_boxes):
                continue
            if any(_boxes_conflict(bbox, other, gap=1) for other in stamp_boxes):
                continue
            if int(grid[anchor_r, anchor_c]) != BG:
                continue
            grid[anchor_r, anchor_c] = int(item["marker_color"])
            item["anchor"] = (anchor_r, anchor_c)
            stamp_boxes.append(bbox)
            break
        else:
            return None

    expected = _expected_output((height, width), placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
