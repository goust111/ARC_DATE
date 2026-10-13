from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _interval_overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _boxes_separated(a: BBox, b: BBox, pad: int = 0) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paint_frame(grid: np.ndarray, box: BBox) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _fill_frame_interior(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(color)


def _corridor_bounds(a: BBox, b: BBox, info: dict[str, int | str]) -> tuple[int, int, int, int]:
    axis = str(info["axis"])
    band = int(info["band"])
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if axis == "h":
        if ac1 < bc0:
            return band, ac1 + 1, band + 1, bc0 - 1
        return band, bc1 + 1, band + 1, ac0 - 1
    if br1 < ar0:
        return br1 + 1, band, ar0 - 1, band + 1
    return ar1 + 1, band, br0 - 1, band + 1


def _paint_corridor(grid: np.ndarray, a: BBox, b: BBox, info: dict[str, int | str]) -> None:
    r0, c0, r1, c1 = _corridor_bounds(a, b, info)
    if r0 <= r1 and c0 <= c1:
        grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE


def _horizontal_visibility(grid: np.ndarray, a: BBox, b: BBox) -> dict[str, int | str] | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    overlap_r0 = max(ar0, br0)
    overlap_r1 = min(ar1, br1)
    if overlap_r1 - overlap_r0 + 1 < 2:
        return None
    if ac1 < bc0:
        gap_c0, gap_c1 = ac1 + 1, bc0 - 1
    elif bc1 < ac0:
        gap_c0, gap_c1 = bc1 + 1, ac0 - 1
    else:
        return None

    for band_r in range(overlap_r0, overlap_r1):
        if gap_c0 <= gap_c1 and not np.all(grid[band_r : band_r + 2, gap_c0 : gap_c1 + 1] == BLACK):
            continue
        return {"axis": "h", "band": int(band_r), "distance": int(max(0, gap_c1 - gap_c0 + 1))}
    return None


def _vertical_visibility(grid: np.ndarray, a: BBox, b: BBox) -> dict[str, int | str] | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    overlap_c0 = max(ac0, bc0)
    overlap_c1 = min(ac1, bc1)
    if overlap_c1 - overlap_c0 + 1 < 2:
        return None
    if ar1 < br0:
        gap_r0, gap_r1 = ar1 + 1, br0 - 1
    elif br1 < ar0:
        gap_r0, gap_r1 = br1 + 1, ar0 - 1
    else:
        return None

    for band_c in range(overlap_c0, overlap_c1):
        if gap_r0 <= gap_r1 and not np.all(grid[gap_r0 : gap_r1 + 1, band_c : band_c + 2] == BLACK):
            continue
        return {"axis": "v", "band": int(band_c), "distance": int(max(0, gap_r1 - gap_r0 + 1))}
    return None


def _visibility_info(grid: np.ndarray, a: BBox, b: BBox) -> dict[str, int | str] | None:
    h_info = _horizontal_visibility(grid, a, b)
    v_info = _vertical_visibility(grid, a, b)
    if h_info is not None and v_info is not None:
        if int(h_info["distance"]) <= int(v_info["distance"]):
            return h_info
        return v_info
    return h_info if h_info is not None else v_info


def _parse_input(grid: np.ndarray) -> tuple[BBox, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_components = _connected_components(g == RED)
    assert len(red_components) == 1, "expected exactly one red component"
    start_box = _bbox_coords(red_components[0])
    assert red_components[0].shape[0] == 4 and _bbox_h(start_box) == 2 and _bbox_w(start_box) == 2

    frame_components = _connected_components(g == GREY)
    frames: list[dict[str, object]] = []
    for coords in frame_components:
        box = _bbox_coords(coords)
        h = _bbox_h(box)
        w = _bbox_w(box)
        r0, c0, r1, c1 = box
        assert h >= 4 and w >= 4, "frame interior must be at least 2x2"
        assert int(coords.shape[0]) == 2 * h + 2 * w - 4, "frame must be single-pixel hollow rectangle"
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border:
                    assert int(g[r, c]) == GREY
                else:
                    assert int(g[r, c]) == BLACK
        frames.append({"bbox": box})

    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))  # type: ignore[index]
    for idx, frame in enumerate(frames):
        frame["id"] = idx
    assert frames, "expected at least one gray frame"
    return start_box, frames


def _simulate_walk(grid: np.ndarray, dynamic: bool) -> dict[str, object]:
    start_box, frames = _parse_input(grid)
    out = np.asarray(grid, dtype=int).copy()
    current_box = start_box
    unvisited = [dict(frame) for frame in frames]
    visited_ids: list[int] = []
    visited_axes: list[str] = []
    sequence: list[BBox] = [start_box]
    frame_map = {int(frame["id"]): frame["bbox"] for frame in frames}

    while True:
        source = out if dynamic else grid
        candidates = []
        for idx, frame in enumerate(unvisited):
            box = frame["bbox"]
            info = _visibility_info(source, current_box, box)
            if info is None:
                continue
            sort_key = (
                int(info["distance"]),
                int(box[0]),
                int(box[1]),
                0 if info["axis"] == "h" else 1,
            )
            candidates.append((sort_key, idx, frame, info))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0])
        _sort_key, idx, frame, info = candidates[0]
        box = frame["bbox"]
        _paint_corridor(out, current_box, box, info)
        _fill_frame_interior(out, box, GREEN if info["axis"] == "h" else YELLOW)
        visited_ids.append(int(frame["id"]))
        visited_axes.append(str(info["axis"]))
        current_box = box
        sequence.append(box)
        unvisited.pop(idx)

    return {
        "output": out,
        "visited": visited_ids,
        "axes": visited_axes,
        "sequence": sequence,
        "frame_map": frame_map,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Greedily connect the red start block to visible gray hollow frames, with each new corridor changing later visibility.

    Input structure:
    - The grid has a black background, exactly one red 2x2 start block, and several gray single-pixel hollow rectangular frames.
    - A frame's interior is black in the input, and different objects are separated by background.
    - Two objects can interact only through a shared 2-row or 2-column band between their bounding boxes.

    Rule:
    - Start at the red block and repeatedly find every unvisited gray frame that is visible in some shared 2-row or 2-column band.
    - A frame is visible only if every pixel in the gap segment of that band is black; any non-black pixel, including earlier orange corridors, blocks sight.
    - Choose the nearest visible frame by edge-to-edge gap distance, breaking ties by the frame whose bounding box is higher, then lefter.
    - Color the black gap pixels of the chosen band orange, then fill that frame's interior green if the corridor was horizontal or yellow if it was vertical.
    - Continue from that frame until no unvisited gray frame remains visible; unreachable frames stay hollow.

    Key insight:
    - Visibility must be recomputed after every move because the newly painted orange corridor becomes a fresh obstacle that can block later greedy choices.
    """
    return _simulate_walk(np.asarray(grid, dtype=int), dynamic=True)["output"]  # type: ignore[index]


GEN_MIN_GRID = 17
GEN_MAX_GRID = 26
GEN_MAX_ATTEMPTS = 500


def _shift_box(box: BBox, dr: int, dc: int) -> BBox:
    return box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc


def _paint_start(grid: np.ndarray, box: BBox) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = RED


def _sample_core_boxes(rng: np.random.Generator) -> list[BBox] | None:
    start = (0, 0, 1, 1)

    h_a = int(rng.integers(4, 7))
    w_a = int(rng.integers(4, 7))
    gap_sa = int(rng.integers(5, 8))
    a_top = int(rng.integers(0, h_a - 1))
    a_r0 = -a_top
    a_c0 = 2 + gap_sa
    a = (a_r0, a_c0, a_r0 + h_a - 1, a_c0 + w_a - 1)

    h_b = int(rng.integers(4, 7))
    gap_ab = int(rng.integers(2, 6))
    b_r0 = a[2] + gap_ab + 1
    b = (b_r0, a[1], b_r0 + h_b - 1, a[3])

    c = (b[0], a[1] - 5, b[2], a[1] - 2)

    h_d = int(rng.integers(4, 7))
    gap_cd = h_a + gap_ab + int(rng.integers(1, 4))
    d_c0 = c[3] - 1
    d_r1 = c[0] - gap_cd - 1
    d = (d_r1 - h_d + 1, d_c0, d_r1, d_c0 + 3)

    boxes = [start, a, b, c, d]
    if not all(_boxes_separated(boxes[i], boxes[j], pad=1) for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
        return None
    return boxes


def _sample_grid_and_shift(
    rng: np.random.Generator,
    core_boxes: list[BBox],
    extra_count: int,
) -> tuple[int, int, list[BBox]] | None:
    min_r = min(box[0] for box in core_boxes)
    min_c = min(box[1] for box in core_boxes)
    max_r = max(box[2] for box in core_boxes)
    max_c = max(box[3] for box in core_boxes)
    core_h = max_r - min_r + 1
    core_w = max_c - min_c + 1

    h_min = max(GEN_MIN_GRID, core_h + 2 + extra_count)
    w_min = max(GEN_MIN_GRID, core_w + 2 + extra_count)
    h_max = min(GEN_MAX_GRID, core_h + 8 + 2 * extra_count)
    w_max = min(GEN_MAX_GRID, core_w + 8 + 2 * extra_count)
    if h_min > h_max or w_min > w_max:
        return None

    h = int(rng.integers(h_min, h_max + 1))
    w = int(rng.integers(w_min, w_max + 1))
    dr_lo = 1 - min_r
    dr_hi = h - 2 - max_r
    dc_lo = 1 - min_c
    dc_hi = w - 2 - max_c
    if dr_lo > dr_hi or dc_lo > dc_hi:
        return None
    dr = int(rng.integers(dr_lo, dr_hi + 1))
    dc = int(rng.integers(dc_lo, dc_hi + 1))
    return h, w, [_shift_box(box, dr, dc) for box in core_boxes]


def _isolated_from_all(candidate: BBox, boxes: list[BBox]) -> bool:
    for box in boxes:
        if not _boxes_separated(candidate, box, pad=1):
            return False
        row_overlap = _interval_overlap_len(candidate[0], candidate[2], box[0], box[2])
        col_overlap = _interval_overlap_len(candidate[1], candidate[3], box[1], box[3])
        if row_overlap >= 2 or col_overlap >= 2:
            return False
    return True


def _place_extra_frames(
    rng: np.random.Generator,
    h: int,
    w: int,
    boxes: list[BBox],
    extra_count: int,
) -> list[BBox] | None:
    extras: list[BBox] = []
    all_boxes = list(boxes)
    tries = 0
    while len(extras) < extra_count and tries < 500:
        tries += 1
        hh = int(rng.integers(4, 7))
        ww = int(rng.integers(4, 7))
        if hh >= h - 1 or ww >= w - 1:
            continue
        r0 = int(rng.integers(1, h - hh))
        c0 = int(rng.integers(1, w - ww))
        cand = (r0, c0, r0 + hh - 1, c0 + ww - 1)
        if not _isolated_from_all(cand, all_boxes):
            continue
        extras.append(cand)
        all_boxes.append(cand)
    if len(extras) != extra_count:
        return None
    return extras


def _apply_random_symmetry(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    code = int(rng.integers(0, 8))
    out = np.array(grid, copy=True)
    if code == 0:
        return out
    if code == 1:
        return np.flipud(out).copy()
    if code == 2:
        return np.fliplr(out).copy()
    if code == 3:
        return np.flipud(np.fliplr(out)).copy()
    if code == 4:
        return out.T.copy()
    if code == 5:
        return np.flipud(out.T).copy()
    if code == 6:
        return np.fliplr(out.T).copy()
    return np.flipud(np.fliplr(out.T)).copy()


def _build_grid_from_boxes(h: int, w: int, boxes: list[BBox]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    _paint_start(grid, boxes[0])
    for box in boxes[1:]:
        _paint_frame(grid, box)
    return grid


def _candidate_obstructed_by_corridor(
    stopped_grid: np.ndarray,
    current_box: BBox,
    target_box: BBox,
    original_info: dict[str, int | str],
) -> bool:
    r0, c0, r1, c1 = _corridor_bounds(current_box, target_box, original_info)
    if r0 > r1 or c0 > c1:
        return False
    return bool(np.any(stopped_grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE))


def _semantic_valid(grid: np.ndarray) -> bool:
    try:
        start_box, frames = _parse_input(grid)
        dynamic = _simulate_walk(grid, dynamic=True)
        static = _simulate_walk(grid, dynamic=False)
        out = dynamic["output"]
        if not np.array_equal(out, main(grid)):
            return False
        if len(frames) < 4 or len(frames) > 6:
            return False
        if len(static["visited"]) < 4:
            return False
        if len(dynamic["visited"]) >= len(static["visited"]):
            return False
        if static["visited"][: len(dynamic["visited"])] != dynamic["visited"]:
            return False
        if set(dynamic["axes"]) != {"h", "v"}:
            return False

        stop_box = dynamic["sequence"][-1] if dynamic["sequence"] else start_box
        next_id = static["visited"][len(dynamic["visited"])]
        next_box = static["frame_map"][next_id]
        original_info = _visibility_info(grid, stop_box, next_box)
        blocked_info = _visibility_info(out, stop_box, next_box)
        if original_info is None or blocked_info is not None:
            return False
        if not _candidate_obstructed_by_corridor(out, stop_box, next_box, original_info):
            return False

        dynamic_set = set(dynamic["visited"])
        for frame in frames:
            box = frame["bbox"]
            frame_id = int(frame["id"])
            interior = out[box[0] + 1 : box[2], box[1] + 1 : box[3]]
            if frame_id in dynamic_set:
                if not np.all(np.isin(interior, [GREEN, YELLOW])):
                    return False
            else:
                if interior.size and not np.all(interior == BLACK):
                    return False
        return True
    except Exception:
        return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    core = _sample_core_boxes(rng)
    if core is None:
        return None

    extra_count = int(rng.integers(0, 3))
    shifted = _sample_grid_and_shift(rng, core, extra_count)
    if shifted is None:
        return None
    h, w, core_boxes = shifted

    extras = _place_extra_frames(rng, h, w, core_boxes, extra_count)
    if extras is None:
        return None

    grid = _build_grid_from_boxes(h, w, core_boxes + extras)
    grid = _apply_random_symmetry(grid, rng)
    if not _semantic_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dynamic-occlusion corridor instance")
