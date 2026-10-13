from __future__ import annotations

import numpy as np


BG = 0
OBJ_COLOR = 5
FRAME_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_GENERATION_ATTEMPTS = 400
PADDING = 1


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _rounded_centroid(coords: np.ndarray) -> tuple[int, int]:
    return (
        _round_half_up(float(coords[:, 0].mean())),
        _round_half_up(float(coords[:, 1].mean())),
    )


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(np.asarray(cells, dtype=int))
    return components


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    components: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_from_mask(grid == color):
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    components.sort(key=lambda comp: (comp["bbox"], comp["color"]))
    return components


def _is_rect_frame(component: dict) -> bool:
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border_area = 2 * (h + w) - 4
    if int(component["area"]) != border_area:
        return False
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr not in (r0, r1) and cc not in (c0, c1):
            return False
    return True


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _build_frame_hierarchy(frames: list[dict], shape: tuple[int, int]) -> list[dict]:
    for frame in frames:
        frame["parent"] = None
        frame["children"] = []

    for i, child in enumerate(frames):
        parents = [j for j, parent in enumerate(frames) if i != j and _strictly_contains(parent["bbox"], child["bbox"])]
        if parents:
            parent_idx = min(parents, key=lambda idx: _bbox_area(frames[idx]["bbox"]))
            child["parent"] = int(parent_idx)
            frames[parent_idx]["children"].append(i)

    def depth_of(idx: int) -> int:
        frame = frames[idx]
        cached = frame.get("depth")
        if cached is not None:
            return int(cached)
        parent = frame["parent"]
        depth = 1 if parent is None else depth_of(int(parent)) + 1
        frame["depth"] = depth
        return depth

    for idx in range(len(frames)):
        depth_of(idx)

    for frame in frames:
        mask = np.zeros(shape, dtype=bool)
        r0, c0, r1, c1 = frame["bbox"]
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            mask[r0 + 1 : r1, c0 + 1 : c1] = True
        for child_idx in frame["children"]:
            cr0, cc0, cr1, cc1 = frames[int(child_idx)]["bbox"]
            mask[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
        region_cells = np.argwhere(mask)
        if region_cells.size == 0:
            frame["region_center"] = ((r0 + r1) // 2, (c0 + c1) // 2)
        else:
            frame["region_center"] = _rounded_centroid(region_cells)
        frame["region_mask"] = mask

    frames.sort(key=lambda frame: (-int(frame["depth"]), frame["bbox"]))
    return frames


def _coords_fit_region(coords: np.ndarray, region_mask: np.ndarray) -> bool:
    h, w = region_mask.shape
    if coords.size == 0:
        return True
    if (
        int(coords[:, 0].min()) < 0
        or int(coords[:, 1].min()) < 0
        or int(coords[:, 0].max()) >= h
        or int(coords[:, 1].max()) >= w
    ):
        return False
    return bool(region_mask[coords[:, 0], coords[:, 1]].all())


def _best_shift(
    coords: np.ndarray,
    region_mask: np.ndarray,
    row_sign: int,
    col_sign: int,
    depth: int,
) -> tuple[int, int]:
    want_row = depth if row_sign != 0 else 0
    want_col = depth if col_sign != 0 else 0
    active_axes = int(row_sign != 0) + int(col_sign != 0)

    best: tuple[int, int] | None = None
    best_score: tuple[int, int, int, int] | None = None
    for row_steps in range(want_row, -1, -1):
        for col_steps in range(want_col, -1, -1):
            shifted = coords.copy()
            if row_steps:
                shifted[:, 0] += row_sign * row_steps
            if col_steps:
                shifted[:, 1] += col_sign * col_steps
            if not _coords_fit_region(shifted, region_mask):
                continue
            balance = min(row_steps, col_steps) if active_axes == 2 else row_steps + col_steps
            score = (row_steps + col_steps, balance, row_steps, col_steps)
            if best_score is None or score > best_score:
                best_score = score
                best = (row_steps, col_steps)
    return (0, 0) if best is None else best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested rectangular frame levels and pull each small object inward by its depth.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _foreground_components(g, bg)
    frames = [dict(comp) for comp in components if _is_rect_frame(comp)]
    objects = [dict(comp) for comp in components if not _is_rect_frame(comp)]

    if not frames or not objects:
        return g.copy()

    frames = _build_frame_hierarchy(frames, g.shape)
    out = g.copy()
    for obj in objects:
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = bg

    for obj in objects:
        owner = None
        for frame in frames:
            if _coords_fit_region(obj["coords"], frame["region_mask"]):
                owner = frame
                break
        if owner is None:
            out[obj["coords"][:, 0], obj["coords"][:, 1]] = obj["color"]
            continue

        obj_center = _rounded_centroid(obj["coords"])
        region_center = owner["region_center"]
        row_sign = _sign(region_center[0] - obj_center[0])
        col_sign = _sign(region_center[1] - obj_center[1])
        row_steps, col_steps = _best_shift(
            obj["coords"],
            owner["region_mask"],
            row_sign,
            col_sign,
            int(owner["depth"]),
        )
        shifted = obj["coords"].copy()
        if row_steps:
            shifted[:, 0] += row_sign * row_steps
        if col_steps:
            shifted[:, 1] += col_sign * col_steps
        out[shifted[:, 0], shifted[:, 1]] = obj["color"]

    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_depth(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array([1, 2, 3, 4]), p=np.array([0.15, 0.35, 0.30, 0.20])))


def _sample_inner_size(rng: np.random.Generator, depth: int) -> tuple[int, int]:
    if depth == 1:
        low, high = 8, 13
    elif depth == 2:
        low, high = 7, 11
    elif depth == 3:
        low, high = 6, 9
    else:
        low, high = 5, 8
    return int(rng.integers(low, high)), int(rng.integers(low, high))


def _sample_nested_bboxes(rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    depth = _sample_depth(rng)
    inner_h, inner_w = _sample_inner_size(rng, depth)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    gap_high = 4 if depth <= 3 else 3
    for _ in range(depth - 1):
        top = int(rng.integers(2, gap_high + 1))
        bottom = int(rng.integers(2, gap_high + 1))
        left = int(rng.integers(2, gap_high + 1))
        right = int(rng.integers(2, gap_high + 1))
        r0, c0, r1, c1 = boxes[-1]
        boxes.append((r0 - top - 1, c0 - left - 1, r1 + bottom + 1, c1 + right + 1))

    outer = boxes[-1]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1
    if outer_h > 30 or outer_w > 30:
        return None

    top_margin = int(rng.integers(1, 5))
    left_margin = int(rng.integers(1, 5))
    bottom_margin = int(rng.integers(1, 5))
    right_margin = int(rng.integers(1, 5))
    if outer_h + top_margin + bottom_margin > 32 or outer_w + left_margin + right_margin > 32:
        return None

    dr = top_margin - outer[0]
    dc = left_margin - outer[1]
    translated = [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for (r0, c0, r1, c1) in reversed(boxes)]
    return translated


def _grow_connected_shape(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add((int(pick[0]), int(pick[1])))
    coords = np.asarray(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    area = int(rng.integers(2, 7))
    dims = [(h, w) for h in range(1, 4) for w in range(1, 4) if h * w >= area]
    h, w = dims[int(rng.integers(0, len(dims)))]
    return _grow_connected_shape(rng, h, w, area)


def _has_padding(coords: np.ndarray, occupied: np.ndarray, pad: int = PADDING) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _movement_info(
    coords: np.ndarray,
    frame: dict,
) -> dict:
    obj_center = _rounded_centroid(coords)
    region_center = frame["region_center"]
    row_sign = _sign(region_center[0] - obj_center[0])
    col_sign = _sign(region_center[1] - obj_center[1])
    depth = int(frame["depth"])
    row_steps, col_steps = _best_shift(coords, frame["region_mask"], row_sign, col_sign, depth)
    shifted = coords.copy()
    if row_steps:
        shifted[:, 0] += row_sign * row_steps
    if col_steps:
        shifted[:, 1] += col_sign * col_steps
    want_row = depth if row_sign != 0 else 0
    want_col = depth if col_sign != 0 else 0
    return {
        "shifted": shifted,
        "row_sign": row_sign,
        "col_sign": col_sign,
        "row_steps": row_steps,
        "col_steps": col_steps,
        "moving": bool(row_steps + col_steps > 0),
        "axis": bool((row_sign == 0) ^ (col_sign == 0)),
        "clipped": bool(row_steps < want_row or col_steps < want_col),
    }


def _enumerate_candidates(
    shape: np.ndarray,
    frame: dict,
    occupied_in: np.ndarray,
    occupied_out: np.ndarray,
) -> list[dict]:
    region_mask = frame["region_mask"]
    h, w = region_mask.shape
    ph, pw = shape.shape
    shape_coords = np.argwhere(shape)
    candidates: list[dict] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            coords = shape_coords + np.array([r0, c0], dtype=int)
            if not bool(region_mask[coords[:, 0], coords[:, 1]].all()):
                continue
            if not _has_padding(coords, occupied_in):
                continue
            info = _movement_info(coords, frame)
            if not info["moving"]:
                continue
            if not _has_padding(info["shifted"], occupied_out):
                continue
            candidates.append(
                {
                    "coords": coords,
                    "shifted": info["shifted"],
                    "row_steps": info["row_steps"],
                    "col_steps": info["col_steps"],
                    "axis": info["axis"],
                    "clipped": info["clipped"],
                }
            )
    return candidates


def _pick_candidate(
    rng: np.random.Generator,
    frame: dict,
    occupied_in: np.ndarray,
    occupied_out: np.ndarray,
    need_axis: bool,
    need_clipped: bool,
) -> dict | None:
    candidates: list[dict] = []
    for _ in range(12):
        shape = _random_shape(rng)
        candidates.extend(_enumerate_candidates(shape, frame, occupied_in, occupied_out))
    if not candidates:
        return None

    rng.shuffle(candidates)
    best_priority = None
    best_group: list[dict] = []
    for cand in candidates:
        priority = int(need_axis and cand["axis"]) + int(need_clipped and cand["clipped"])
        score = (priority, cand["row_steps"] + cand["col_steps"], int(cand["clipped"]), int(cand["axis"]))
        if best_priority is None or score > best_priority:
            best_priority = score
            best_group = [cand]
        elif score == best_priority:
            best_group.append(cand)
    return best_group[int(rng.integers(0, len(best_group)))] if best_group else None


def _place_objects(
    rng: np.random.Generator,
    grid: np.ndarray,
    frames: list[dict],
) -> tuple[np.ndarray, dict] | None:
    occupied_in = np.zeros(grid.shape, dtype=bool)
    occupied_out = np.zeros(grid.shape, dtype=bool)
    counts: list[int] = []
    ordered_frames = sorted(frames, key=lambda frame: (int(frame["depth"]), frame["bbox"]))
    for frame in ordered_frames:
        area = int(frame["region_mask"].sum())
        max_objects = min(3, max(1, area // 28))
        counts.append(int(rng.integers(1, max_objects + 1)))

    need_axis = bool(rng.random() < 0.40)
    need_clipped = bool(rng.random() < 0.45)
    total_slots = int(sum(counts))
    placed = 0
    saw_axis = False
    saw_clipped = False

    for frame, count in zip(ordered_frames, counts):
        for _ in range(count):
            placed += 1
            remaining = total_slots - placed + 1
            unmet = int(need_axis and not saw_axis) + int(need_clipped and not saw_clipped)
            if remaining < unmet:
                return None
            force_axis = need_axis and not saw_axis and remaining <= unmet
            force_clipped = need_clipped and not saw_clipped and remaining <= unmet
            candidate = _pick_candidate(rng, frame, occupied_in, occupied_out, force_axis, force_clipped)
            if candidate is None:
                return None
            grid[candidate["coords"][:, 0], candidate["coords"][:, 1]] = OBJ_COLOR
            occupied_in[candidate["coords"][:, 0], candidate["coords"][:, 1]] = True
            occupied_out[candidate["shifted"][:, 0], candidate["shifted"][:, 1]] = True
            saw_axis = saw_axis or bool(candidate["axis"])
            saw_clipped = saw_clipped or bool(candidate["clipped"])

    if not bool(occupied_out.any()):
        return None
    if need_axis and not saw_axis:
        return None
    if need_clipped and not saw_clipped:
        return None
    return grid, {"axis": saw_axis, "clipped": saw_clipped, "total_objects": int(occupied_in.sum())}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    boxes = _sample_nested_bboxes(rng)
    if boxes is None:
        return None

    h = max(box[2] for box in boxes) + 1
    w = max(box[3] for box in boxes) + 1
    grid = np.full((h, w), BG, dtype=int)

    colors = [int(x) for x in rng.choice(np.asarray(FRAME_COLOR_POOL, dtype=int), size=len(boxes), replace=False)]
    frame_records = []
    for box, color in zip(boxes, colors):
        _draw_frame(grid, box, color)
        coords: list[tuple[int, int]] = []
        r0, c0, r1, c1 = box
        for c in range(c0, c1 + 1):
            coords.append((r0, c))
            coords.append((r1, c))
        for r in range(r0 + 1, r1):
            coords.append((r, c0))
            coords.append((r, c1))
        coord_arr = np.asarray(coords, dtype=int)
        frame_records.append(
            {
                "color": color,
                "coords": coord_arr,
                "bbox": box,
                "area": int(coord_arr.shape[0]),
            }
        )

    frames = _build_frame_hierarchy(frame_records, grid.shape)
    placed = _place_objects(rng, grid.copy(), frames)
    if placed is None:
        return None
    inp, meta = placed
    out = main(inp)

    if np.array_equal(inp, out):
        return None
    if meta["total_objects"] < len(boxes):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
