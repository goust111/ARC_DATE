from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    ext = ~mask
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if ext[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and ext[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(ext & ~seen))


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    out[rows + int(top), cols + int(left)] = int(color)


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(mask)
    return {(int(r + top), int(c + left)) for r, c in zip(rows, cols)}


def _mask_dims(mask: np.ndarray) -> tuple[int, int]:
    return int(mask.shape[0]), int(mask.shape[1])


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    objects = []
    for idx, comp in enumerate(_find_components(g)):
        mask = _coords_to_mask(comp["coords"], comp["bbox"])
        has_hole = _has_hole(mask)
        r0, c0, r1, c1 = comp["bbox"]
        objects.append(
            {
                "id": idx,
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "mask": mask,
                "has_hole": has_hole,
                "direction": -1 if has_hole else 1,
                "input_top": int(r0),
                "input_left": int(c0),
                "input_top_edge": int(r0),
                "input_bottom_edge": int(r1),
                "input_area": int(comp["area"]),
            }
        )

    down = [obj for obj in objects if obj["direction"] == 1]
    up = [obj for obj in objects if obj["direction"] == -1]
    down.sort(key=lambda obj: (-obj["input_bottom_edge"], obj["input_top_edge"], obj["input_left"]))
    up.sort(key=lambda obj: (obj["input_top_edge"], obj["input_left"], obj["input_bottom_edge"]))

    out = np.zeros_like(g)
    owner = np.full(g.shape, -1, dtype=int)
    trace: list[dict] = []

    for obj in down + up:
        mask = obj["mask"]
        mh, mw = _mask_dims(mask)
        top = int(obj["input_top"])
        left = int(obj["input_left"])
        dr = int(obj["direction"])

        blocker_ids: set[int] = set()
        while True:
            next_top = top + dr
            next_cells = _mask_cells(mask, next_top, left)
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in next_cells):
                stop = "boundary"
                break
            overlaps = {int(owner[r, c]) for r, c in next_cells if owner[r, c] != -1}
            if overlaps:
                blocker_ids = overlaps
                stop = "object"
                break
            top = next_top

        color = GRAY if (dr == 1 and top + mh - 1 == h - 1) or (dr == -1 and top == 0) else obj["color"]
        _paint_mask(out, mask, top, left, color)
        for r, c in _mask_cells(mask, top, left):
            owner[r, c] = int(obj["id"])

        trace.append(
            {
                "id": int(obj["id"]),
                "color": int(obj["color"]),
                "has_hole": bool(obj["has_hole"]),
                "direction": int(dr),
                "input_top": int(obj["input_top"]),
                "input_left": int(obj["input_left"]),
                "final_top": int(top),
                "final_left": int(left),
                "mask": mask.copy(),
                "stop_type": stop,
                "blocker_ids": sorted(blocker_ids),
                "output_color": int(color),
            }
        )

    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each object vertically by topology, then gray only the ones stopped by the grid edge.

    Input structure:
    - A black background contains 4-connected single-color objects in distinct colors.
    - Some objects are solid, while others are frame-like and enclose at least one black 4-connected hole.
    - Objects are initially separated and keep their own local shape and holes.

    Rule:
    - Classify every object by topology: solid objects slide downward, hole-containing frame objects slide upward.
    - Process all downward objects first, ordered by lowest input row from largest to smallest; then process all upward objects, ordered by highest input row from smallest to largest.
    - Slide each object one row at a time until its next step would leave the grid or overlap already fixed non-background pixels.
    - Recolor an object to gray iff its final position touches the stopping boundary in its travel direction; otherwise keep its original color. Holes stay black.

    Key insight:
    - Topology determines motion direction, but the recolor decision is independent: it depends on whether the stop was caused by the outer boundary rather than by another already placed object.
    """
    return _solve_with_trace(np.asarray(grid, dtype=int))[0]


def _is_connected(mask: np.ndarray) -> bool:
    cells = list(zip(*np.nonzero(mask)))
    if not cells:
        return False
    cell_set = {(int(r), int(c)) for r, c in cells}
    seen = {next(iter(cell_set))}
    q = deque(seen)
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr in cell_set and nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return seen == cell_set


def _random_solid_mask() -> np.ndarray:
    for _ in range(200):
        ch = random.randint(3, 5)
        cw = random.randint(3, 4)
        target = random.randint(6, min(11, ch * cw))
        mask = np.zeros((ch, cw), dtype=bool)
        sr = random.randrange(ch)
        sc = random.randrange(cw)
        mask[sr, sc] = True
        while int(mask.sum()) < target:
            frontier = []
            for r, c in zip(*np.nonzero(mask)):
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < ch and 0 <= nc < cw and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            mask[random.choice(frontier)] = True
        rows, cols = np.nonzero(mask)
        mask = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        if 6 <= int(mask.sum()) <= 11 and _is_connected(mask) and not _has_hole(mask):
            return mask
    raise RuntimeError("failed to generate solid mask")


def _random_frame_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 4)
        canvas = np.zeros((h + 2, w + 2), dtype=bool)
        canvas[1 : h + 1, 1] = True
        canvas[1 : h + 1, w] = True
        canvas[1, 1 : w + 1] = True
        canvas[h, 1 : w + 1] = True
        bumps = random.randint(0, 2)
        for _ in range(bumps):
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                c = random.randint(1, w)
                canvas[0, c] = True
            elif side == "bottom":
                c = random.randint(1, w)
                canvas[h + 1, c] = True
            elif side == "left":
                r = random.randint(1, h)
                canvas[r, 0] = True
            else:
                r = random.randint(1, h)
                canvas[r, w + 1] = True
        rows, cols = np.nonzero(canvas)
        mask = canvas[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        if 8 <= int(mask.sum()) <= 15 and _is_connected(mask) and _has_hole(mask):
            return mask
    raise RuntimeError("failed to generate frame mask")


def _first_overlap_stop_down(mask: np.ndarray, left: int, blocker: set[tuple[int, int]], H: int) -> int | None:
    mh = int(mask.shape[0])
    for top in range(0, H - mh + 1):
        cells = _mask_cells(mask, top, left)
        if cells & blocker:
            return top - 1 if top > 0 else None
    return None


def _first_overlap_stop_up(mask: np.ndarray, left: int, blocker: set[tuple[int, int]], H: int) -> int | None:
    mh = int(mask.shape[0])
    for top in range(H - mh, -1, -1):
        cells = _mask_cells(mask, top, left)
        if cells & blocker:
            return top + 1 if top < H - mh else None
    return None


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    vals = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return vals


def _tint(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place_input(mask: np.ndarray, top: int, left: int, H: int, W: int, occupied: set[tuple[int, int]]) -> bool:
    cells = _mask_cells(mask, top, left)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W) or (r, c) in occupied:
            return False
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr in occupied and nbr not in cells:
                return False
    return True


def _pick_unique_value(candidates: list[int], used: set[int]) -> int | None:
    choices = [v for v in candidates if v not in used]
    return random.choice(choices) if choices else None


def _build_expected(specs: list[dict], H: int, W: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for spec in specs:
        color = GRAY if spec["stop_type"] == "boundary" else spec["color"]
        _paint_mask(out, spec["mask"], spec["final_top"], spec["left"], color)
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, expected_meta: list[dict]) -> bool:
    solved, trace = _solve_with_trace(inp)
    if not np.array_equal(solved, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    kind_by_id = {meta["color"]: meta["kind"] for meta in expected_meta}
    stop_by_id = {meta["color"]: meta["stop_type"] for meta in expected_meta}
    seen = {"solid_boundary": 0, "solid_object": 0, "frame_boundary": 0, "frame_object": 0}

    for item in trace:
        kind = "frame" if item["has_hole"] else "solid"
        if kind_by_id.get(item["color"]) != kind:
            return False
        if stop_by_id.get(item["color"]) != item["stop_type"]:
            return False
        if item["stop_type"] == "object" and len(item["blocker_ids"]) != 1:
            return False
        key = f"{kind}_{item['stop_type']}"
        if key in seen:
            seen[key] += 1

    return all(seen[key] >= 1 for key in seen)


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(4, 6)
    extras = object_count - 4

    solid_pair = [
        {"kind": "solid", "mask": _random_solid_mask()},
        {"kind": "solid", "mask": _random_solid_mask()},
    ]
    frame_pair = [
        {"kind": "frame", "mask": _random_frame_mask()},
        {"kind": "frame", "mask": _random_frame_mask()},
    ]
    extra_objs = []
    for _ in range(extras):
        kind = random.choice(("solid", "frame"))
        extra_objs.append({"kind": kind, "mask": _random_solid_mask() if kind == "solid" else _random_frame_mask()})

    lanes = [
        {"direction": "down", "objects": solid_pair, "pair": True},
        {"direction": "up", "objects": frame_pair, "pair": True},
    ]
    for obj in extra_objs:
        lanes.append({"direction": "down" if obj["kind"] == "solid" else "up", "objects": [obj], "pair": False})

    widths = [max(int(obj["mask"].shape[1]) for obj in lane["objects"]) for lane in lanes]
    min_width = sum(widths) + 2 * (len(lanes) - 1)
    if min_width > 24:
        return None
    min_W = max(15, min_width)
    max_W = max(min_W, min(24, min_width + 6))
    W = random.randint(min_W, max_W)
    H = random.randint(15, 22)

    slack = W - min_width
    slots = _random_composition(slack, len(lanes) + 1)
    lane_lefts = []
    cur = slots[0]
    for i, lane_w in enumerate(widths):
        lane_lefts.append(cur)
        cur += lane_w
        if i + 1 < len(widths):
            cur += 2 + slots[i + 1]

    colors = random.sample(INPUT_COLORS, object_count)
    color_iter = iter(colors)

    specs: list[dict] = []
    used_bottoms: set[int] = set()
    used_tops: set[int] = set()

    for lane_idx, lane in enumerate(lanes):
        left = lane_lefts[lane_idx]
        if lane["pair"] and lane["direction"] == "down":
            lower, upper = lane["objects"]
            lower_h = int(lower["mask"].shape[0])
            lower_final = H - lower_h
            blocker = _mask_cells(lower["mask"], lower_final, left)
            upper_final = _first_overlap_stop_down(upper["mask"], left, blocker, H)
            if upper_final is None or upper_final >= H - int(upper["mask"].shape[0]):
                return None

            lower_bottom = _pick_unique_value(list(range(lower_final - 1, max(-1, lower_final - 6), -1)), used_bottoms)
            if lower_bottom is None:
                return None
            lower_input = lower_bottom - lower_h + 1

            upper_bottom_max = min(
                upper_final + int(upper["mask"].shape[0]) - 2,
                lower_input - 2,
            )
            upper_bottom = _pick_unique_value(list(range(upper_bottom_max, int(upper["mask"].shape[0]) - 2, -1)), used_bottoms)
            if upper_bottom is None:
                return None
            upper_input = upper_bottom - int(upper["mask"].shape[0]) + 1
            if upper_input < 0:
                return None

            for obj, color, final_top, input_top, stop in (
                (lower, next(color_iter), lower_final, lower_input, "boundary"),
                (upper, next(color_iter), upper_final, upper_input, "object"),
            ):
                used_bottoms.add(input_top + int(obj["mask"].shape[0]) - 1)
                specs.append(
                    {
                        "kind": obj["kind"],
                        "mask": obj["mask"],
                        "color": color,
                        "left": left,
                        "input_top": input_top,
                        "final_top": final_top,
                        "stop_type": stop,
                    }
                )

        elif lane["pair"] and lane["direction"] == "up":
            upper, lower = lane["objects"]
            blocker = _mask_cells(upper["mask"], 0, left)
            lower_final = _first_overlap_stop_up(lower["mask"], left, blocker, H)
            if lower_final is None or lower_final <= 0:
                return None

            upper_top = _pick_unique_value(list(range(1, min(H - int(upper["mask"].shape[0]), 6) + 1)), used_tops)
            if upper_top is None:
                return None
            lower_top_min = max(lower_final + 1, upper_top + int(upper["mask"].shape[0]) + 1)
            lower_top = _pick_unique_value(list(range(lower_top_min, H - int(lower["mask"].shape[0]) + 1)), used_tops)
            if lower_top is None:
                return None

            for obj, color, final_top, input_top, stop in (
                (upper, next(color_iter), 0, upper_top, "boundary"),
                (lower, next(color_iter), lower_final, lower_top, "object"),
            ):
                used_tops.add(input_top)
                specs.append(
                    {
                        "kind": obj["kind"],
                        "mask": obj["mask"],
                        "color": color,
                        "left": left,
                        "input_top": input_top,
                        "final_top": final_top,
                        "stop_type": stop,
                    }
                )
        else:
            obj = lane["objects"][0]
            mh = int(obj["mask"].shape[0])
            color = next(color_iter)
            if lane["direction"] == "down":
                final_top = H - mh
                bottom = _pick_unique_value(list(range(final_top - 1, max(-1, final_top - 7), -1)), used_bottoms)
                if bottom is None:
                    return None
                input_top = bottom - mh + 1
                if input_top < 0:
                    return None
                used_bottoms.add(bottom)
            else:
                final_top = 0
                input_top = _pick_unique_value(list(range(1, H - mh + 1)), used_tops)
                if input_top is None:
                    return None
                used_tops.add(input_top)
            specs.append(
                {
                    "kind": obj["kind"],
                    "mask": obj["mask"],
                    "color": color,
                    "left": left,
                    "input_top": input_top,
                    "final_top": final_top,
                    "stop_type": "boundary",
                }
            )

    inp = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        if not _can_place_input(spec["mask"], spec["input_top"], spec["left"], H, W, occupied):
            return None
        patch = _tint(spec["mask"], spec["color"])
        _paint_mask(inp, spec["mask"], spec["input_top"], spec["left"], spec["color"])
        occupied |= _mask_cells(spec["mask"], spec["input_top"], spec["left"])

    expected = _build_expected(specs, H, W)
    if not _validate_semantics(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
