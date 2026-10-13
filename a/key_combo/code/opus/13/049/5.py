from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RECT = 1
GREY = 5
FRAME_COLORS = [2, 3, 4, 6, 7, 8, 9]
RAY_SPECS = [
    ("tl", (-1, -1), "left"),
    ("tr", (-1, 1), "right"),
    ("bl", (1, -1), "left"),
    ("br", (1, 1), "right"),
]


def _neighbors(r: int, c: int, connectivity: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _frame_corners(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _frame_border_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _is_on_ray(src: tuple[int, int], dr: int, dc: int, pt: tuple[int, int]) -> bool:
    drow = int(pt[0] - src[0])
    dcol = int(pt[1] - src[1])
    if drow == 0 or dcol == 0:
        return False
    if drow % dr != 0 or dcol % dc != 0:
        return False
    step_r = drow // dr
    step_c = dcol // dc
    return step_r == step_c and step_r > 0


def _rect_corner_info(rect_box: tuple[int, int, int, int]) -> list[tuple[str, tuple[int, int], tuple[int, int], str]]:
    r0, c0, r1, c1 = rect_box
    return [
        ("tl", (r0, c0), (-1, -1), "left"),
        ("tr", (r0, c1), (-1, 1), "right"),
        ("bl", (r1, c0), (1, -1), "left"),
        ("br", (r1, c1), (1, 1), "right"),
    ]


def _corner_for_ray(ray_name: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    mapping = {
        "tl": (r1, c1),
        "tr": (r1, c0),
        "bl": (r0, c1),
        "br": (r0, c0),
    }
    return mapping[ray_name]


def _box_from_target_corner(ray_name: str, corner: tuple[int, int], h: int, w: int) -> tuple[int, int, int, int]:
    cr, cc = corner
    if ray_name == "tl":
        return cr - h + 1, cc - w + 1, cr, cc
    if ray_name == "tr":
        return cr - h + 1, cc, cr, cc + w - 1
    if ray_name == "bl":
        return cr, cc - w + 1, cr + h - 1, cc
    if ray_name == "br":
        return cr, cc, cr + h - 1, cc + w - 1
    raise ValueError(ray_name)


def _parse_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid == RECT, connectivity=4)
    assert len(comps) == 1
    box = _bbox(comps[0])
    h = box[2] - box[0] + 1
    w = box[3] - box[1] + 1
    assert (h, w) in {(2, 3), (3, 2)}
    assert comps[0].shape[0] == h * w
    return box


def _parse_template(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid == GREY, connectivity=8)
    assert len(comps) == 1
    box = _bbox(comps[0])
    h = box[2] - box[0] + 1
    w = box[3] - box[1] + 1
    assert (h, w) == (3, 3)
    patch = np.zeros((3, 3), dtype=int)
    patch[grid[box[0] : box[2] + 1, box[1] : box[3] + 1] == GREY] = GREY
    assert 4 <= int((patch == GREY).sum()) <= 6
    assert not np.all(patch == GREY)
    return patch


def _parse_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BG, RECT, GREY}):
        comps = _components(grid == color, connectivity=4)
        assert len(comps) == 1
        coords = comps[0]
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert h >= 6 and w >= 6
        assert np.all(patch[border] == color)
        assert np.all(patch[~border] != color)
        assert coords.shape[0] == int(border.sum())
        frames.append(
            {
                "color": int(color),
                "bbox": box,
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "corners": set(_frame_corners(box)),
            }
        )
    assert 2 <= len(frames) <= 4
    return sorted(frames, key=lambda item: item["bbox"])


def _trace_activations(
    rect_box: tuple[int, int, int, int],
    frames: list[dict],
    shape: tuple[int, int],
) -> dict[tuple[int, int, int, int], str]:
    corner_lookup: dict[tuple[int, int], dict] = {}
    for frame in frames:
        for pt in frame["corners"]:
            corner_lookup[pt] = frame

    activations: dict[tuple[int, int, int, int], str] = {}
    h, w = shape
    for ray_name, src, (dr, dc), side in _rect_corner_info(rect_box):
        rr, cc = src[0] + dr, src[1] + dc
        while 0 <= rr < h and 0 <= cc < w:
            frame = corner_lookup.get((rr, cc))
            if frame is not None:
                activations[frame["bbox"]] = side
                break
            rr += dr
            cc += dc
    return activations


def _shear_template(template: np.ndarray, color: int, side: str) -> np.ndarray:
    h, w = template.shape
    out = np.zeros((h, w + h - 1), dtype=int)
    for r in range(h):
        start = h - 1 - r if side == "left" else r
        for c in range(w):
            if template[r, c] == GREY:
                out[r, start + c] = color
    return out


def _stamp_into_frame(out: np.ndarray, frame: dict, stamp: np.ndarray) -> None:
    ir0, ic0, ir1, ic1 = frame["inner_bbox"]
    for r in range(stamp.shape[0]):
        for c in range(stamp.shape[1]):
            color = int(stamp[r, c])
            if color == BG:
                continue
            rr = ir0 + r
            cc = ic0 + c
            if ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
                out[rr, cc] = color


def _render_from_scene(grid: np.ndarray, rect_box: tuple[int, int, int, int], template: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    activations = _trace_activations(rect_box, frames, grid.shape)
    for frame in frames:
        side = activations.get(frame["bbox"])
        if side is None:
            continue
        stamp = _shear_template(template, frame["color"], side)
        _stamp_into_frame(out, frame, stamp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate hollow frames whose corners lie on diagonal rays from the blue rectangle, then stamp a direction-dependent sheared copy of the grey template into them.

    Input structure:
    - Black background with one small blue solid rectangle, one grey 3x3 template motif, and 2-4 differently colored one-cell-thick hollow rectangular frames.
    - The rectangle is separated from the frames; each frame interior is large enough to contain a 3-row stamp.
    - Some frame corners lie exactly on outward 45-degree diagonals from rectangle corners, while other frames are merely distractors.

    Rule:
    - From the rectangle's four outer corners, trace the four outward diagonals and activate the first frame corner hit on each ray.
    - Recolor the grey template to the activated frame's color, shear it on a black canvas of width W+H-1, and anchor that canvas to the frame interior's top-left corner.
    - Rays from left rectangle corners use left shear (row r shifted left by r via descending row starts); rays from right rectangle corners use right shear (row r shifted right by r). Only non-black stamp pixels are written.

    Key insight:
    - The diagonals are invisible control signals rather than drawn output; a frame is selected only by a corner-ray hit, and the emitting corner's left/right side encodes the shear direction while the frame color encodes the recolor.
    """
    g = np.asarray(grid, dtype=int)
    rect_box = _parse_rect(g)
    template = _parse_template(g)
    frames = _parse_frames(g)
    return _render_from_scene(g, rect_box, template, frames)


def _random_template(rng: random.Random) -> np.ndarray:
    coords = [(r, c) for r in range(3) for c in range(3)]
    while True:
        start = rng.choice(coords)
        cells = {start}
        target = rng.randint(4, 6)
        while len(cells) < target:
            frontier = []
            for r, c in list(cells):
                for nr, nc in _neighbors(r, c, 8):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(rng.choice(frontier))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != 2 or min(cols) != 0 or max(cols) != 2:
            continue
        patch = np.zeros((3, 3), dtype=int)
        for r, c in cells:
            patch[r, c] = GREY
        if np.all(patch == GREY):
            continue
        if np.array_equal(_shear_template(patch, 9, "left"), _shear_template(patch, 9, "right")):
            continue
        return patch


def _ray_candidates(
    rect_box: tuple[int, int, int, int],
    shape: tuple[int, int],
    ray_name: str,
    placed_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    ray_info = {name: (src, drdc) for name, src, drdc, _side in _rect_corner_info(rect_box)}
    src, (dr, dc) = ray_info[ray_name]
    candidates: list[tuple[int, int, int, int]] = []
    dims = [(fh, fw) for fh in (6, 7) for fw in (7, 8)]
    random.shuffle(dims)
    for fh, fw in dims:
        for step in range(2, max(h, w)):
            corner = (src[0] + dr * step, src[1] + dc * step)
            box = _box_from_target_corner(ray_name, corner, fh, fw)
            r0, c0, r1, c1 = box
            if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                continue
            if any(_bbox_overlap(box, other, pad=1) for other in placed_boxes):
                continue
            candidates.append(box)
    return candidates


def _box_corner_on_any_ray(rect_box: tuple[int, int, int, int], box: tuple[int, int, int, int]) -> bool:
    for _ray_name, src, (dr, dc), _side in _rect_corner_info(rect_box):
        for pt in _frame_corners(box):
            if _is_on_ray(src, dr, dc, pt):
                return True
    return False


def _box_noncorner_on_any_ray(rect_box: tuple[int, int, int, int], box: tuple[int, int, int, int]) -> bool:
    corner_set = set(_frame_corners(box))
    for _ray_name, src, (dr, dc), _side in _rect_corner_info(rect_box):
        for pt in _frame_border_cells(box):
            if pt in corner_set:
                continue
            if _is_on_ray(src, dr, dc, pt):
                return True
    return False


def _random_box(
    rng: random.Random,
    shape: tuple[int, int],
    placed_boxes: list[tuple[int, int, int, int]],
    rect_box: tuple[int, int, int, int],
    *,
    allow_corner_on_ray: bool,
    require_noncorner_on_ray: bool = False,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    candidates = []
    for fh in (6, 7):
        for fw in (7, 8):
            for r0 in range(1, h - fh):
                for c0 in range(1, w - fw):
                    box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                    if any(_bbox_overlap(box, other, pad=1) for other in placed_boxes):
                        continue
                    corner_hit = _box_corner_on_any_ray(rect_box, box)
                    if corner_hit and not allow_corner_on_ray:
                        continue
                    noncorner_hit = _box_noncorner_on_any_ray(rect_box, box)
                    if require_noncorner_on_ray and (corner_hit or not noncorner_hit):
                        continue
                    candidates.append(box)
    return rng.choice(candidates) if candidates else None


def _template_positions(
    shape: tuple[int, int],
    placed_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    corners = [
        [(r, c) for r in range(1, 4) for c in range(1, 4)],
        [(r, c) for r in range(1, 4) for c in range(w - 5, w - 2)],
        [(r, c) for r in range(h - 5, h - 2) for c in range(1, 4)],
        [(r, c) for r in range(h - 5, h - 2) for c in range(w - 5, w - 2)],
    ]
    out: list[tuple[int, int]] = []
    for group in corners:
        for r0, c0 in group:
            box = (r0, c0, r0 + 2, c0 + 2)
            if any(_bbox_overlap(box, other, pad=1) for other in placed_boxes):
                continue
            out.append((r0, c0))
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_expected(inp: np.ndarray, template: np.ndarray, frames: list[dict]) -> np.ndarray:
    rect_box = _parse_rect(inp)
    return _render_from_scene(inp, rect_box, template, frames)


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 24)
    w = rng.randint(20, 24)
    grid = np.zeros((h, w), dtype=int)

    rect_h, rect_w = rng.choice([(2, 3), (3, 2)])
    rect_r = rng.randint(8, h - rect_h - 9)
    rect_c = rng.randint(8, w - rect_w - 9)
    rect_box = (rect_r, rect_c, rect_r + rect_h - 1, rect_c + rect_w - 1)

    template = _random_template(rng)

    placed_boxes = [rect_box]
    active_count = rng.choice([1, 2, 2, 3])
    total_frames = min(4, max(2, active_count + rng.choice([0, 1, 1])))
    ray_pool = ["tl", "tr", "bl", "br"]
    if active_count >= 2 and rng.random() < 0.75:
        left_ray = rng.choice(["tl", "bl"])
        right_ray = rng.choice(["tr", "br"])
        remaining = [r for r in ray_pool if r not in {left_ray, right_ray}]
        chosen_rays = [left_ray, right_ray] + rng.sample(remaining, active_count - 2)
    else:
        chosen_rays = rng.sample(ray_pool, active_count)
    rng.shuffle(chosen_rays)

    frames_meta: list[dict] = []
    active_steps: dict[str, int] = {}
    for ray_name in chosen_rays:
        candidates = _ray_candidates(rect_box, (h, w), ray_name, placed_boxes)
        if not candidates:
            return None
        box = rng.choice(candidates)
        target_corner = _corner_for_ray(ray_name, box)
        src = {name: source for name, source, _drdc, _side in _rect_corner_info(rect_box)}[ray_name]
        active_steps[ray_name] = abs(target_corner[0] - src[0])
        placed_boxes.append(box)
        frames_meta.append({"bbox": box, "ray": ray_name, "special": "active"})

    inactive_needed = total_frames - active_count
    special_modes = []
    if inactive_needed > 0:
        special_modes = ["ordinary"] * inactive_needed
        if rng.random() < 0.35:
            special_modes[0] = "shadow"
        elif rng.random() < 0.45:
            special_modes[0] = "side_hit"
        rng.shuffle(special_modes)

    for mode in special_modes:
        box = None
        if mode == "shadow" and chosen_rays:
            ray_name = rng.choice(chosen_rays)
            src, (dr, dc) = {
                name: (source, delta) for name, source, delta, _side in _rect_corner_info(rect_box)
            }[ray_name]
            dims = [(fh, fw) for fh in (6, 7) for fw in (7, 8)]
            rng.shuffle(dims)
            for fh, fw in dims:
                for step in range(active_steps[ray_name] + 2, max(h, w)):
                    corner = (src[0] + dr * step, src[1] + dc * step)
                    cand = _box_from_target_corner(ray_name, corner, fh, fw)
                    if cand[0] < 0 or cand[1] < 0 or cand[2] >= h or cand[3] >= w:
                        continue
                    if any(_bbox_overlap(cand, other, pad=1) for other in placed_boxes):
                        continue
                    box = cand
                    break
                if box is not None:
                    break
        elif mode == "side_hit":
            box = _random_box(
                rng,
                (h, w),
                placed_boxes,
                rect_box,
                allow_corner_on_ray=False,
                require_noncorner_on_ray=True,
            )
        if box is None:
            box = _random_box(
                rng,
                (h, w),
                placed_boxes,
                rect_box,
                allow_corner_on_ray=False,
            )
        if box is None:
            return None
        placed_boxes.append(box)
        frames_meta.append({"bbox": box, "ray": None, "special": mode})

    template_pos_candidates = _template_positions((h, w), placed_boxes)
    if not template_pos_candidates:
        return None
    template_r, template_c = rng.choice(template_pos_candidates)
    template_box = (template_r, template_c, template_r + 2, template_c + 2)
    placed_boxes.append(template_box)

    colors = rng.sample(FRAME_COLORS, len(frames_meta))
    rng.shuffle(frames_meta)
    for meta, color in zip(frames_meta, colors):
        meta["color"] = color

    grid[rect_box[0] : rect_box[2] + 1, rect_box[1] : rect_box[3] + 1] = RECT
    for meta in frames_meta:
        _draw_frame(grid, meta["bbox"], meta["color"])
    for r in range(3):
        for c in range(3):
            if template[r, c] == GREY:
                grid[template_r + r, template_c + c] = GREY

    frames = _parse_frames(grid)
    expected = _build_expected(grid, template, frames)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    activations = _trace_activations(rect_box, frames, grid.shape)
    intended = {}
    for meta in frames_meta:
        if meta["ray"] is None:
            continue
        side = "left" if meta["ray"] in {"tl", "bl"} else "right"
        intended[meta["bbox"]] = side
    if activations != intended:
        return None
    if len(activations) != active_count:
        return None

    if not any(meta["ray"] is None for meta in frames_meta) and total_frames > active_count:
        return None

    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        before = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
        after = actual[ir0 : ir1 + 1, ic0 : ic1 + 1]
        is_active = frame["bbox"] in activations
        changed = bool(np.any(before != after))
        if is_active != changed:
            return None

    sides = set(activations.values())
    if active_count >= 2 and rng.random() < 0.5 and len(sides) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray stamp puzzle")
