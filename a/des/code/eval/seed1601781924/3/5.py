import random
from collections import deque

import numpy as np


BACKGROUND = 0
DEFAULT_SLOT_COLOR = 5
FOREGROUND_COLORS = tuple(c for c in range(10) if c not in (BACKGROUND, DEFAULT_SLOT_COLOR))
MAX_GENERATION_ATTEMPTS = 400
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": local,
                    "area": int(local.sum()),
                }
            )
    return comps


def _bool_component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _touches_occupied(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in _NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            return True
    return False


def _is_rectangular_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    frame = np.zeros_like(mask, dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return bool(np.array_equal(mask, frame))


def _detect_slots(grid: np.ndarray, comps_by_color: dict[int, list[dict]]) -> tuple[int, list[dict]]:
    h, _w = grid.shape
    candidates: list[tuple[int, int, int, list[dict]]] = []
    for color, comps in comps_by_color.items():
        if color == BACKGROUND or not comps:
            continue
        if not all(_is_rectangular_frame(comp["mask"]) for comp in comps):
            continue
        heights = {comp["mask"].shape[0] for comp in comps}
        widths = {comp["mask"].shape[1] for comp in comps}
        row_starts = {comp["bbox"][0] for comp in comps}
        row_ends = {comp["bbox"][2] for comp in comps}
        if len(heights) != 1 or len(widths) != 1 or len(row_starts) != 1 or len(row_ends) != 1:
            continue
        row_start = next(iter(row_starts))
        if row_start < h // 2:
            continue
        ordered = sorted(comps, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
        if any(ordered[i]["bbox"][1] <= ordered[i - 1]["bbox"][3] for i in range(1, len(ordered))):
            continue
        candidates.append((len(comps), row_start, int(color), ordered))

    assert candidates, "failed to find slot frames"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _count, _row, slot_color, slots = candidates[0]
    return int(slot_color), slots


def _overlap_score(mask: np.ndarray, orientation: str, axis2: int) -> int:
    coords = np.argwhere(mask)
    h, w = mask.shape
    score = 0
    if orientation == "h":
        for r, c in coords:
            rr = int(axis2 - r)
            if 0 <= rr < h and mask[rr, c]:
                score += 1
    else:
        for r, c in coords:
            cc = int(axis2 - c)
            if 0 <= cc < w and mask[r, cc]:
                score += 1
    return int(score)


def _fold_mask(mask: np.ndarray, orientation: str, axis2: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if orientation == "h":
        rows = coords[:, 0]
        cols = coords[:, 1]
        proj_rows = np.minimum(rows, axis2 - rows)
        proj_cols = cols
    else:
        rows = coords[:, 0]
        cols = coords[:, 1]
        proj_rows = rows
        proj_cols = np.minimum(cols, axis2 - cols)

    min_r = int(proj_rows.min())
    min_c = int(proj_cols.min())
    max_r = int(proj_rows.max())
    max_c = int(proj_cols.max())
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    out[proj_rows - min_r, proj_cols - min_c] = True
    return out


def _best_fold_info(mask: np.ndarray) -> dict:
    h, w = mask.shape
    candidates: list[dict] = []
    for axis2 in range(2 * h - 1):
        candidates.append(
            {
                "orientation": "h",
                "axis2": int(axis2),
                "score": _overlap_score(mask, "h", axis2),
                "center_dist": abs(axis2 - (h - 1)),
            }
        )
    for axis2 in range(2 * w - 1):
        candidates.append(
            {
                "orientation": "v",
                "axis2": int(axis2),
                "score": _overlap_score(mask, "v", axis2),
                "center_dist": abs(axis2 - (w - 1)),
            }
        )

    best = max(
        candidates,
        key=lambda item: (
            item["score"],
            -item["center_dist"],
            1 if item["orientation"] == "h" else 0,
            -item["axis2"],
        ),
    )
    best_count = sum(item["score"] == best["score"] for item in candidates)
    folded = _fold_mask(mask, best["orientation"], best["axis2"])
    return {
        "orientation": best["orientation"],
        "axis2": int(best["axis2"]),
        "score": int(best["score"]),
        "best_count": int(best_count),
        "folded": folded,
        "fold_area": int(folded.sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each shape's best horizontal/vertical symmetry axis, fold it onto
    the upper or left half, then place the folded shapes into the bottom slots
    ordered by folded area from large to small.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    comps_by_color = {color: _components_of_color(g, color) for color in colors}
    slot_color, slots = _detect_slots(g, comps_by_color)

    shapes: list[dict] = []
    for color, comps in comps_by_color.items():
        if color == slot_color:
            continue
        for comp in comps:
            info = _best_fold_info(comp["mask"])
            shapes.append(
                {
                    "color": int(color),
                    "bbox": comp["bbox"],
                    "folded": info["folded"],
                    "fold_area": int(info["fold_area"]),
                }
            )

    assert len(shapes) == len(slots), "shape count must match slot count"
    shapes.sort(
        key=lambda item: (
            -item["fold_area"],
            item["bbox"][0],
            item["bbox"][1],
            item["color"],
        )
    )

    out = np.full(g.shape, BACKGROUND, dtype=int)
    for slot in slots:
        r0, c0, r1, c1 = slot["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][slot["mask"]] = slot_color

    for shape, slot in zip(shapes, slots):
        folded = shape["folded"]
        h0, w0 = folded.shape
        r0, c0, r1, c1 = slot["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert h0 <= inner_h and w0 <= inner_w, "folded shape must fit in the slot interior"
        target = out[r0 + 1 : r0 + 1 + h0, c0 + 1 : c0 + 1 + w0]
        target[folded] = shape["color"]

    return out


def _random_connected_mask(h: int, w: int, area: int, boundary: str) -> np.ndarray | None:
    if not (1 <= area < h * w):
        return None
    if boundary == "bottom":
        start = (h - 1, random.randrange(w))
    else:
        start = (random.randrange(h), w - 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    for dr, dc in _NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while int(mask.sum()) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for dr, dc in _NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _mirror_half(half: np.ndarray, orientation: str, through_cell: bool) -> tuple[np.ndarray, int]:
    coords = np.argwhere(half)
    if orientation == "h":
        height = half.shape[0] * 2 - 1 if through_cell else half.shape[0] * 2
        full = np.zeros((height, half.shape[1]), dtype=bool)
        axis2 = 2 * (half.shape[0] - 1) if through_cell else 2 * half.shape[0] - 1
        full[coords[:, 0], coords[:, 1]] = True
        full[axis2 - coords[:, 0], coords[:, 1]] = True
    else:
        width = half.shape[1] * 2 - 1 if through_cell else half.shape[1] * 2
        full = np.zeros((half.shape[0], width), dtype=bool)
        axis2 = 2 * (half.shape[1] - 1) if through_cell else 2 * half.shape[1] - 1
        full[coords[:, 0], coords[:, 1]] = True
        full[coords[:, 0], axis2 - coords[:, 1]] = True
    return full, int(axis2)


def _extra_candidates(mask: np.ndarray, orientation: str, axis2: int, keep_side: bool) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or not _touches_occupied(mask, r, c):
                continue
            if orientation == "h":
                rr = axis2 - r
                if not (0 <= rr < h) or rr == r:
                    continue
                if keep_side and not (r < rr):
                    continue
                if not keep_side and not (r > rr):
                    continue
                if mask[rr, c]:
                    continue
            else:
                cc = axis2 - c
                if not (0 <= cc < w) or cc == c:
                    continue
                if keep_side and not (c < cc):
                    continue
                if not keep_side and not (c > cc):
                    continue
                if mask[r, cc]:
                    continue
            candidates.append((r, c))
    return candidates


def _add_extra(mask: np.ndarray, orientation: str, axis2: int, keep_side: bool, required: bool) -> bool:
    candidates = _extra_candidates(mask, orientation, axis2, keep_side)
    if not candidates:
        return not required
    r, c = random.choice(candidates)
    mask[r, c] = True
    return True


def _build_shape(orientation: str) -> dict | None:
    for _ in range(80):
        if orientation == "h":
            keep_h = random.randint(2, 5)
            width = random.randint(3, 6)
            max_area = keep_h * width - 1
            area = random.randint(max(3, keep_h), max_area)
            half = _random_connected_mask(keep_h, width, area, "bottom")
        else:
            height = random.randint(3, 6)
            keep_w = random.randint(2, 5)
            max_area = height * keep_w - 1
            area = random.randint(max(3, keep_w), max_area)
            half = _random_connected_mask(height, keep_w, area, "right")
        if half is None:
            continue
        if _is_rectangular_frame(half):
            continue

        full, axis2 = _mirror_half(half, orientation, through_cell=bool(random.getrandbits(1)))
        if not _add_extra(full, orientation, axis2, keep_side=True, required=True):
            continue
        if not _add_extra(full, orientation, axis2, keep_side=False, required=True):
            continue
        if random.random() < 0.5:
            _add_extra(full, orientation, axis2, keep_side=True, required=False)
        if random.random() < 0.5:
            _add_extra(full, orientation, axis2, keep_side=False, required=False)

        info = _best_fold_info(full)
        if info["orientation"] != orientation or info["best_count"] != 1:
            continue
        if info["score"] < int(full.sum()) - 4:
            continue
        if _bool_component_count(full) != 1 or _bool_component_count(info["folded"]) != 1:
            continue
        return {
            "full_mask": full,
            "folded_mask": info["folded"],
            "source_area": int(full.sum()),
            "fold_area": int(info["fold_area"]),
        }
    return None


def _sample_orientations(count: int) -> list[str]:
    if count == 2:
        orientations = ["h", "v"]
    elif count == 3:
        orientations = ["h", "h", "v"] if random.random() < 0.5 else ["h", "v", "v"]
    else:
        orientations = ["h", "h", "v", "v"]
    random.shuffle(orientations)
    return orientations


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _place_shapes(shapes: list[dict], grid_h: int, grid_w: int, slot_top: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda idx: -int(shapes[idx]["full_mask"].sum()))
    for idx in order:
        mask = shapes[idx]["full_mask"]
        h, w = mask.shape
        row_min = 1 if slot_top - h > 2 else 0
        row_max = slot_top - h - 1
        col_min = 1 if grid_w - w > 2 else 0
        col_max = grid_w - w - 1
        if row_max < row_min or col_max < col_min:
            return None

        dilated = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for r0 in range(row_min, row_max + 1):
            for c0 in range(col_min, col_max + 1):
                if not bool((dilated[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions[idx] = pos
        r0, c0 = pos
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _slot_bboxes(count: int, slot_h: int, slot_w: int, grid_w: int, slot_top: int) -> list[tuple[int, int, int, int]]:
    total_w = count * slot_w + (count - 1) * random.randint(1, 3)
    if total_w > grid_w - 2:
        raise AssertionError
    gap = (grid_w - total_w) // (count + 1)
    gap = max(gap, 1)
    extra = grid_w - (count * slot_w + gap * (count + 1))
    c = gap
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(count):
        if i == count - 1:
            c += extra
        boxes.append((slot_top, c, slot_top + slot_h - 1, c + slot_w - 1))
        c += slot_w + gap
    return boxes


def _render_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_expected(shape_infos: list[dict], boxes: list[tuple[int, int, int, int]], grid_shape: tuple[int, int], slot_color: int) -> np.ndarray:
    out = np.full(grid_shape, BACKGROUND, dtype=int)
    for box in boxes:
        _render_frame(out, box, slot_color)
    ordered = sorted(shape_infos, key=lambda item: (-item["fold_area"], item["source_area"]))
    for item, box in zip(ordered, boxes):
        folded = item["folded_mask"]
        r0, c0, _r1, _c1 = box
        target = out[r0 + 1 : r0 + 1 + folded.shape[0], c0 + 1 : c0 + 1 + folded.shape[1]]
        target[folded] = item["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    colors = random.sample(list(FOREGROUND_COLORS), count)
    shapes: list[dict] = []
    for orientation, color in zip(_sample_orientations(count), colors):
        shape = _build_shape(orientation)
        if shape is None:
            return None
        shape["color"] = int(color)
        shapes.append(shape)

    fold_areas = [shape["fold_area"] for shape in shapes]
    if len(set(fold_areas)) != count:
        return None

    max_fold_h = max(shape["folded_mask"].shape[0] for shape in shapes)
    max_fold_w = max(shape["folded_mask"].shape[1] for shape in shapes)
    slot_h = max_fold_h + 2 + random.randint(0, 1)
    slot_w = max_fold_w + 2 + random.randint(0, 2)

    max_shape_h = max(shape["full_mask"].shape[0] for shape in shapes)
    slot_top = max_shape_h + random.randint(5, 9)
    grid_h = slot_top + slot_h + random.randint(1, 3)
    min_grid_w = count * slot_w + (count + 1)
    grid_w = max(min_grid_w + random.randint(0, 6), max(shape["full_mask"].shape[1] for shape in shapes) + 4)
    grid_w = min(grid_w, 40)
    if min_grid_w > grid_w:
        return None

    try:
        boxes = _slot_bboxes(count, slot_h, slot_w, grid_w, slot_top)
    except AssertionError:
        return None
    positions = _place_shapes(shapes, grid_h, grid_w, slot_top)
    if positions is None or len(positions) != count:
        return None

    inp = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    for shape, (r0, c0) in zip(shapes, positions):
        mask = shape["full_mask"]
        inp[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = shape["color"]
    for box in boxes:
        _render_frame(inp, box, DEFAULT_SLOT_COLOR)

    expected = _render_expected(shapes, boxes, inp.shape, DEFAULT_SLOT_COLOR)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
