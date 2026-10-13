from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
INDICATOR = 9
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRECTIONS = ("top", "bottom", "left", "right")
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_canvas_frame(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        side_h = int(r1 - r0 + 1)
        side_w = int(c1 - c0 + 1)
        if side_h != side_w or side_h < 4:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not bool(np.all(patch[border] == FRAME)):
            continue
        if bool(np.any(patch[~border] == FRAME)):
            continue
        center_offset = abs((r0 + r1) - (h - 1)) + abs((c0 + c1) - (w - 1))
        candidates.append((int(side_h), int(center_offset), (int(r0), int(c0), int(r1), int(c1))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _project_origin(direction: str, stamp_h: int, stamp_w: int, canvas_size: int) -> tuple[int, int] | None:
    if stamp_h > canvas_size or stamp_w > canvas_size:
        return None
    if direction == "top":
        return 0, (canvas_size - stamp_w) // 2
    if direction == "bottom":
        return canvas_size - stamp_h, (canvas_size - stamp_w) // 2
    if direction == "left":
        return (canvas_size - stamp_h) // 2, 0
    if direction == "right":
        return (canvas_size - stamp_h) // 2, canvas_size - stamp_w
    return None


def _extract_stamp_records(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    h, w = grid.shape
    stamp_mask = (grid != BG) & (grid != FRAME) & (grid != INDICATOR)
    comps = _components(stamp_mask)
    labels = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(comps):
        for r, c in comp:
            labels[r, c] = int(idx)

    indicators_by_comp: dict[int, list[tuple[int, int]]] = {}
    for ir, ic in map(tuple, np.argwhere(grid == INDICATOR)):
        neighbor_ids = set()
        for nr, nc in _neighbors4(int(ir), int(ic)):
            if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) >= 0:
                neighbor_ids.add(int(labels[nr, nc]))
        if len(neighbor_ids) == 1:
            idx = next(iter(neighbor_ids))
            indicators_by_comp.setdefault(int(idx), []).append((int(ir), int(ic)))

    fr0, fc0, fr1, fc1 = frame_bbox
    center_r2 = int(fr0 + fr1)
    center_c2 = int(fc0 + fc1)
    records: list[dict[str, object]] = []
    for idx, comp in enumerate(comps):
        indicator_list = indicators_by_comp.get(int(idx), [])
        if len(indicator_list) != 1:
            continue
        coords = np.asarray(comp, dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = int(grid[r, c])

        ir, ic = indicator_list[0]
        if ir < r0:
            direction = "top"
        elif ir > r1:
            direction = "bottom"
        elif ic < c0:
            direction = "left"
        else:
            direction = "right"

        distance = max(abs((r0 + r1) - center_r2), abs((c0 + c1) - center_c2))
        records.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "direction": direction,
                "distance": int(distance),
            }
        )
    return records


def _compose_output(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    records: list[dict[str, object]],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0, fc0 : fc1 + 1] = FRAME
    out[fr1, fc0 : fc1 + 1] = FRAME
    out[fr0 : fr1 + 1, fc0] = FRAME
    out[fr0 : fr1 + 1, fc1] = FRAME

    canvas_size = int(fr1 - fr0 - 1)
    if canvas_size <= 0:
        return out
    canvas = np.full((canvas_size, canvas_size), BG, dtype=int)
    ordered = sorted(
        records,
        key=lambda item: (
            int(item["distance"]),
            int(item["bbox"][0]),
            int(item["bbox"][1]),
        ),
    )
    for record in ordered:
        patch = np.asarray(record["patch"], dtype=int)
        stamp_h, stamp_w = patch.shape
        origin = _project_origin(str(record["direction"]), int(stamp_h), int(stamp_w), int(canvas_size))
        if origin is None:
            continue
        rr, cc = origin
        mask = patch != BG
        block = canvas[rr : rr + stamp_h, cc : cc + stamp_w]
        block[mask] = patch[mask]
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = canvas
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each scattered stamp into the central canvas using its maroon direction
    marker, ordering layers from nearer-to-center first to farther last.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _find_canvas_frame(g)
    if frame_bbox is None:
        return g.copy()
    records = _extract_stamp_records(g, frame_bbox)
    return _compose_output(g.shape, frame_bbox, records)


def _mask_is_connected(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) > 0 and len(_components(mask)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    area = int(height * width)
    min_area = max(4, (area * 3 + 4) // 5)
    target_area = random.randint(min_area, area)
    while int(mask.sum()) > target_area:
        removable: list[tuple[int, int]] = []
        for r, c in map(tuple, np.argwhere(mask)):
            test = mask.copy()
            test[int(r), int(c)] = False
            if _touches_all_sides(test) and _mask_is_connected(test):
                removable.append((int(r), int(c)))
        if not removable:
            break
        rr, cc = random.choice(removable)
        mask[rr, cc] = False
    return mask


def _biased_dim(low: int, high: int, preferred_low: int) -> int:
    if high <= preferred_low:
        return random.randint(low, high)
    if random.random() < 0.7:
        return random.randint(preferred_low, high)
    return random.randint(low, high)


def _random_stamp_patch(direction: str, canvas_size: int) -> np.ndarray:
    max_dim = min(6, canvas_size - 2)
    min_dim = 3
    if direction in ("top", "bottom"):
        height = _biased_dim(min_dim, max_dim, min(4, max_dim))
        width = random.randint(min_dim, max_dim)
    else:
        height = random.randint(min_dim, max_dim)
        width = _biased_dim(min_dim, max_dim, min(4, max_dim))

    mask = _random_mask(int(height), int(width))
    coords = [(int(r), int(c)) for r, c in map(tuple, np.argwhere(mask))]
    random.shuffle(coords)

    patch = np.full((height, width), BG, dtype=int)
    n_colors = random.choice((2, 2, 3))
    colors = random.sample(STAMP_COLORS, n_colors)
    for color, (r, c) in zip(colors, coords):
        patch[r, c] = int(color)
    for r, c in coords[n_colors:]:
        patch[r, c] = int(random.choice(colors))
    return patch


def _indicator_for_patch(patch: np.ndarray, direction: str, r0: int, c0: int) -> tuple[int, int]:
    mask = patch != BG
    h, w = patch.shape
    if direction == "top":
        cols = [int(c) for c in range(w) if bool(mask[0, c])]
        return int(r0 - 1), int(c0 + random.choice(cols))
    if direction == "bottom":
        cols = [int(c) for c in range(w) if bool(mask[h - 1, c])]
        return int(r0 + h), int(c0 + random.choice(cols))
    if direction == "left":
        rows = [int(r) for r in range(h) if bool(mask[r, 0])]
        return int(r0 + random.choice(rows)), int(c0 - 1)
    rows = [int(r) for r in range(h) if bool(mask[r, w - 1])]
    return int(r0 + random.choice(rows)), int(c0 + w)


def _union_bbox(
    stamp_bbox: tuple[int, int, int, int],
    indicator: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = stamp_bbox
    ir, ic = indicator
    return min(r0, ir), min(c0, ic), max(r1, ir), max(c1, ic)


def _bbox_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_stamp(
    grid_h: int,
    grid_w: int,
    frame_bbox: tuple[int, int, int, int],
    used_boxes: list[tuple[int, int, int, int]],
    patch: np.ndarray,
    direction: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int, int, int]] | None:
    stamp_h, stamp_w = patch.shape
    r_min = 1 if direction == "top" else 0
    r_max = grid_h - stamp_h - (1 if direction == "bottom" else 0)
    c_min = 1 if direction == "left" else 0
    c_max = grid_w - stamp_w - (1 if direction == "right" else 0)
    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(300):
        r0 = random.randint(int(r_min), int(r_max))
        c0 = random.randint(int(c_min), int(c_max))
        stamp_bbox = (int(r0), int(c0), int(r0 + stamp_h - 1), int(c0 + stamp_w - 1))
        indicator = _indicator_for_patch(patch, direction, int(r0), int(c0))
        union_box = _union_bbox(stamp_bbox, indicator)
        if _bbox_close(union_box, frame_bbox, gap=1):
            continue
        if any(_bbox_close(union_box, other, gap=1) for other in used_boxes):
            continue
        return stamp_bbox, indicator, union_box
    return None


def _physical_side(stamp_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    sr0, sc0, sr1, sc1 = stamp_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    dr = (sr0 + sr1) - (fr0 + fr1)
    dc = (sc0 + sc1) - (fc0 + fc1)
    if abs(dr) > abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _projected_pixels(record: dict[str, object], canvas_size: int) -> dict[tuple[int, int], int]:
    patch = np.asarray(record["patch"], dtype=int)
    stamp_h, stamp_w = patch.shape
    origin = _project_origin(str(record["direction"]), int(stamp_h), int(stamp_w), int(canvas_size))
    assert origin is not None
    rr0, cc0 = origin
    pixels: dict[tuple[int, int], int] = {}
    for rr, cc in map(tuple, np.argwhere(patch != BG)):
        pixels[(int(rr0 + rr), int(cc0 + cc))] = int(patch[rr, cc])
    return pixels


def _has_order_sensitive_overlap(records: list[dict[str, object]], canvas_size: int) -> bool:
    projected = [_projected_pixels(record, canvas_size) for record in records]
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if int(records[i]["distance"]) == int(records[j]["distance"]):
                continue
            common = projected[i].keys() & projected[j].keys()
            for pos in common:
                if int(projected[i][pos]) != int(projected[j][pos]):
                    return True
    return False


def _build_input(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    records: list[dict[str, object]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME
    for record in records:
        patch = np.asarray(record["patch"], dtype=int)
        r0, c0, r1, c1 = record["bbox"]
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != BG
        block[mask] = patch[mask]
        ir, ic = record["indicator"]
        grid[ir, ic] = INDICATOR
    return grid


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(22, 30)
    grid_w = random.randint(22, 30)
    max_canvas = min(14, min(grid_h, grid_w) - 10)
    if max_canvas < 10:
        return None
    canvas_size = random.randint(10, max_canvas)
    outer_side = int(canvas_size + 2)
    fr0 = (grid_h - outer_side) // 2
    fc0 = (grid_w - outer_side) // 2
    frame_bbox = (int(fr0), int(fc0), int(fr0 + outer_side - 1), int(fc0 + outer_side - 1))

    stamp_count = random.randint(4, 6)
    directions = list(DIRECTIONS)
    random.shuffle(directions)
    while len(directions) < stamp_count:
        directions.append(random.choice(DIRECTIONS))
    random.shuffle(directions)

    used_boxes: list[tuple[int, int, int, int]] = []
    records: list[dict[str, object]] = []
    center_r2 = int(frame_bbox[0] + frame_bbox[2])
    center_c2 = int(frame_bbox[1] + frame_bbox[3])

    for direction in directions:
        patch = _random_stamp_patch(direction, canvas_size)
        placed = _place_stamp(grid_h, grid_w, frame_bbox, used_boxes, patch, direction)
        if placed is None:
            return None
        stamp_bbox, indicator, union_box = placed
        distance = max(
            abs((stamp_bbox[0] + stamp_bbox[2]) - center_r2),
            abs((stamp_bbox[1] + stamp_bbox[3]) - center_c2),
        )
        records.append(
            {
                "bbox": stamp_bbox,
                "indicator": indicator,
                "patch": patch,
                "direction": direction,
                "distance": int(distance),
            }
        )
        used_boxes.append(union_box)

    distances = [int(record["distance"]) for record in records]
    if len(set(distances)) != len(distances):
        return None

    mismatch_count = sum(
        1
        for record in records
        if _physical_side(record["bbox"], frame_bbox) != str(record["direction"])
    )
    if mismatch_count < 2:
        return None

    if not _has_order_sensitive_overlap(records, canvas_size):
        return None

    grid = _build_input((grid_h, grid_w), frame_bbox, records)
    expected = _compose_output((grid_h, grid_w), frame_bbox, records)
    if bool(np.array_equal(grid, expected)):
        return None
    if not bool(np.array_equal(main(grid), expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
