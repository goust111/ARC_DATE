from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TEAL = 8
FG_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 600


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _bbox_in_bounds(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(bbox: tuple[int, int, int, int], gap: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - gap), max(0, c0 - gap), min(h - 1, r1 + gap), min(w - 1, c1 + gap)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _marker_corner(region: np.ndarray) -> tuple[int, int]:
    marks = np.argwhere(region == TEAL)
    assert marks.shape == (1, 2), "each rectangle must contain exactly one teal marker"
    mr, mc = map(int, marks[0])
    h, w = region.shape
    assert mr in (0, h - 1) and mc in (0, w - 1), "marker must sit on a rectangle corner"
    return mr, mc


def _corner_to_dir(corner: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    mr, mc = corner
    h, w = shape
    assert mr in (0, h - 1) and mc in (0, w - 1)
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def _rect_info_from_component(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict:
    bbox = _bbox_from_cells(cells)
    r0, c0, r1, c1 = bbox
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = region.shape
    corner = _marker_corner(region)
    colors = set(map(int, region.flatten()))
    colors.discard(BG)
    assert TEAL in colors, "rectangle must contain teal marker"
    colors.discard(TEAL)
    assert len(colors) == 1, "rectangle body must have exactly one color"
    color = colors.pop()
    expected = np.full((h, w), color, dtype=int)
    expected[corner] = TEAL
    assert np.array_equal(region, expected), "component must be a filled rectangle with one replaced corner"
    return {
        "bbox": bbox,
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "color": color,
        "corner": corner,
        "marker_abs": (r0 + corner[0], c0 + corner[1]),
    }


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    rects = [_rect_info_from_component(g, comp) for comp in comps]
    assert len(rects) == 4, "expected exactly four marked rectangles"
    return rects


def _split_mode(dr: int, dc: int) -> str:
    return "vertical" if dr == dc else "horizontal"


def _rect_cell_values(rect: dict, full: bool) -> dict[tuple[int, int], int]:
    cells: dict[tuple[int, int], int] = {}
    mr, mc = rect["corner"]
    for rr in range(rect["h"]):
        for cc in range(rect["w"]):
            value = rect["color"]
            if not full and (rr, cc) == (mr, mc):
                value = TEAL
            cells[(rect["r0"] + rr, rect["c0"] + cc)] = value
    return cells


def _simulate_motion(
    mover_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    other_rects: list[dict],
    shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], dict | None, tuple[int, int, int, int] | None]:
    dr, dc = direction
    current = mover_bbox
    while True:
        nxt = _shift_bbox(current, dr, dc)
        if not _bbox_in_bounds(nxt, shape):
            return current, None, None
        hits = [rect for rect in other_rects if _bboxes_overlap(nxt, rect["bbox"])]
        assert len(hits) <= 1, "movement path may hit at most one rectangle"
        if hits:
            return current, hits[0], nxt
        current = nxt


def _split_target_cells(
    rect: dict,
    collision_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> dict[tuple[int, int], int]:
    dr, dc = direction
    moved: dict[tuple[int, int], int] = {}
    if dr == dc:
        coord = lambda r, c: r + c
        neg_shift = (-1, 1)
        pos_shift = (1, -1)
    else:
        coord = lambda r, c: r - c
        neg_shift = (-1, -1)
        pos_shift = (1, 1)

    cr0, cc0, cr1, cc1 = collision_bbox
    strip_vals = [
        coord(r, c)
        for r in range(cr0, cr1 + 1)
        for c in range(cc0, cc1 + 1)
    ]
    low = min(strip_vals)
    high = max(strip_vals)
    neg_count = 0
    pos_count = 0

    for (r, c), value in _rect_cell_values(rect, full=False).items():
        k = coord(r, c)
        if k < low:
            nr, nc = r + neg_shift[0], c + neg_shift[1]
            neg_count += 1
        elif k > high:
            nr, nc = r + pos_shift[0], c + pos_shift[1]
            pos_count += 1
        else:
            continue
        assert 0 <= nr < shape[0] and 0 <= nc < shape[1]
        assert (nr, nc) not in moved
        moved[(nr, nc)] = value
    assert neg_count > 0 and pos_count > 0, "collision strip must leave two non-empty halves"
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the uniquely colored marked rectangle diagonally, then split the rectangle it collides with.

    Input structure:
    - A black background contains exactly four filled rectangles, each with one teal corner marker replacing a corner cell.
    - Three rectangles share one body color and the fourth rectangle has a different body color.
    - Rectangles are separated so each marked rectangle is its own 4-connected component.

    Rule:
    - Treat each teal marker as part of its rectangle and recover the rectangle's body color from the other cells.
    - Identify the unique-color rectangle, restore its teal corner to the body color, erase its start position, and move it one step at a time along the diagonal indicated by that corner.
    - Stop when the next step would leave the grid or overlap another rectangle; on collision, keep the mover at the last legal position and split the struck rectangle along the mover's projected collision slice, shifting the two remaining halves one cell outward on the perpendicular axis.

    Key insight:
    - Only the color outlier moves; the teal corner chooses the diagonal direction, and that same direction determines how a collision cuts the struck rectangle.
    """
    g = np.asarray(grid, dtype=int)
    rects = _parse_rectangles(g)
    color_counts = Counter(rect["color"] for rect in rects)
    movers = [rect for rect in rects if color_counts[rect["color"]] == 1]
    assert len(movers) == 1, "there must be exactly one color outlier"
    mover = movers[0]
    other_rects = [rect for rect in rects if rect is not mover]

    direction = _corner_to_dir(mover["corner"], (mover["h"], mover["w"]))
    final_bbox, collision_rect, collision_bbox = _simulate_motion(
        mover["bbox"], direction, other_rects, g.shape
    )

    out = g.copy()
    mr0, mc0, mr1, mc1 = mover["bbox"]
    out[mr0 : mr1 + 1, mc0 : mc1 + 1] = BG

    fr0, fc0, fr1, fc1 = final_bbox
    assert np.all(out[fr0 : fr1 + 1, fc0 : fc1 + 1] == BG), "mover destination must be empty"
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = mover["color"]

    if collision_rect is not None:
        tr0, tc0, tr1, tc1 = collision_rect["bbox"]
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BG
        split_cells = _split_target_cells(collision_rect, collision_bbox, direction, g.shape)
        for (r, c), value in split_cells.items():
            assert out[r, c] == BG, "split target must land on empty cells"
            out[r, c] = value

    return out


def _make_rect(r0: int, c0: int, h: int, w: int, color: int, corner: tuple[int, int]) -> dict:
    return {"r0": r0, "c0": c0, "h": h, "w": w, "color": color, "corner": corner, "bbox": (r0, c0, r0 + h - 1, c0 + w - 1)}


def _render_input(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        for (r, c), value in _rect_cell_values(rect, full=False).items():
            assert grid[r, c] == BG
            grid[r, c] = value
    return grid


def _scene_output(scene: dict) -> np.ndarray:
    rects = scene["rects"]
    mover_idx = scene["mover_idx"]
    mover = rects[mover_idx]
    direction = _corner_to_dir(mover["corner"], (mover["h"], mover["w"]))
    others = [rect for i, rect in enumerate(rects) if i != mover_idx]
    final_bbox, collision_rect, collision_bbox = _simulate_motion(mover["bbox"], direction, others, scene["shape"])

    out = _render_input(rects, scene["shape"])
    r0, c0, r1, c1 = mover["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    fr0, fc0, fr1, fc1 = final_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = mover["color"]

    if collision_rect is not None:
        tr0, tc0, tr1, tc1 = collision_rect["bbox"]
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BG
        for (r, c), value in _split_target_cells(collision_rect, collision_bbox, direction, scene["shape"]).items():
            assert out[r, c] == BG
            out[r, c] = value
    return out


def _path_bboxes(scene: dict) -> list[tuple[int, int, int, int]]:
    rects = scene["rects"]
    mover = rects[scene["mover_idx"]]
    direction = _corner_to_dir(mover["corner"], (mover["h"], mover["w"]))
    other_rects = [rect for i, rect in enumerate(rects) if i != scene["mover_idx"]]
    boxes = [mover["bbox"]]
    current = mover["bbox"]
    while True:
        nxt = _shift_bbox(current, *direction)
        if not _bbox_in_bounds(nxt, scene["shape"]):
            return boxes
        if any(_bboxes_overlap(nxt, rect["bbox"]) for rect in other_rects):
            return boxes
        boxes.append(nxt)
        current = nxt


def _collision_support_bboxes(scene: dict) -> list[tuple[int, int, int, int]]:
    rects = scene["rects"]
    mover = rects[scene["mover_idx"]]
    direction = _corner_to_dir(mover["corner"], (mover["h"], mover["w"]))
    others = [rect for i, rect in enumerate(rects) if i != scene["mover_idx"]]
    _final, collision_rect, collision_bbox = _simulate_motion(mover["bbox"], direction, others, scene["shape"])
    if collision_rect is None:
        return []
    cells = _split_target_cells(collision_rect, collision_bbox, direction, scene["shape"])
    left = _components(_render_cells_mask(cells, scene["shape"]))
    return [_bbox_from_cells(comp) for comp in left]


def _render_cells_mask(cells: dict[tuple[int, int], int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _collision_candidates(
    shape: tuple[int, int],
    direction: tuple[int, int],
    mover_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[tuple[dict, dict]]:
    h, w = shape
    dr, dc = direction
    mh, mw = mover_size
    th, tw = target_size
    out: list[tuple[dict, dict]] = []
    for tr0 in range(h - th + 1):
        for tc0 in range(w - tw + 1):
            target_rect = _make_rect(tr0, tc0, th, tw, 2, (0, 0))
            for nr0 in range(tr0 - mh + 1, tr0 + th):
                for nc0 in range(tc0 - mw + 1, tc0 + tw):
                    next_bbox = (nr0, nc0, nr0 + mh - 1, nc0 + mw - 1)
                    if not _bbox_in_bounds(next_bbox, shape):
                        continue
                    if not _bboxes_overlap(next_bbox, target_rect["bbox"]):
                        continue
                    final_bbox = _shift_bbox(next_bbox, -dr, -dc)
                    if not _bbox_in_bounds(final_bbox, shape) or _bboxes_overlap(final_bbox, target_rect["bbox"]):
                        continue
                    try:
                        split_cells = _split_target_cells(target_rect, next_bbox, direction, shape)
                    except AssertionError:
                        continue
                    if any(fr0 <= r <= fr1 and fc0 <= c <= fc1 for (r, c) in split_cells for fr0, fc0, fr1, fc1 in [final_bbox]):
                        continue
                    split_comps = _components(_render_cells_mask(split_cells, shape))
                    if len(split_comps) != 2 or min(len(comp) for comp in split_comps) < 2:
                        continue
                    for steps in range(1, 7):
                        start_bbox = _shift_bbox(final_bbox, -steps * dr, -steps * dc)
                        if not _bbox_in_bounds(start_bbox, shape):
                            continue
                        ok = True
                        current = start_bbox
                        for _ in range(steps + 1):
                            if _bboxes_overlap(current, target_rect["bbox"]):
                                ok = False
                                break
                            current = _shift_bbox(current, dr, dc)
                        if ok:
                            mover_rect = {"bbox": start_bbox, "final_bbox": final_bbox}
                            out.append((mover_rect, target_rect))
    return out


def _boundary_candidates(
    shape: tuple[int, int],
    direction: tuple[int, int],
    mover_size: tuple[int, int],
) -> list[dict]:
    h, w = shape
    dr, dc = direction
    mh, mw = mover_size
    out: list[dict] = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            final_bbox = (r0, c0, r0 + mh - 1, c0 + mw - 1)
            if _bbox_in_bounds(_shift_bbox(final_bbox, dr, dc), shape):
                continue
            for steps in range(1, 7):
                start_bbox = _shift_bbox(final_bbox, -steps * dr, -steps * dc)
                if _bbox_in_bounds(start_bbox, shape):
                    out.append({"bbox": start_bbox, "final_bbox": final_bbox})
    return out


def _random_corner(h: int, w: int) -> tuple[int, int]:
    return random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])


def _sample_mover_and_target(shape: tuple[int, int], branch: str, direction: tuple[int, int], common_color: int, outlier_color: int) -> dict | None:
    if branch == "collision":
        sizes = [
            ((mh, mw), (th, tw))
            for mh in range(2, 5)
            for mw in range(2, 5)
            for th in range(3, 5)
            for tw in range(3, 5)
        ]
        random.shuffle(sizes)
        for (mh, mw), (th, tw) in sizes:
            candidates = _collision_candidates(shape, direction, (mh, mw), (th, tw))
            if not candidates:
                continue
            mover_layout, target_layout = random.choice(candidates)
            mr0, mc0, _, _ = mover_layout["bbox"]
            tr0, tc0, tr1, tc1 = target_layout["bbox"]
            mover = _make_rect(mr0, mc0, mh, mw, outlier_color, _dir_to_corner(direction, (mh, mw)))
            target = _make_rect(tr0, tc0, th, tw, common_color, _random_corner(th, tw))
            return {"mover": mover, "target": target}
        return None

    mh = random.randint(2, 4)
    mw = random.randint(2, 4)
    candidates = _boundary_candidates(shape, direction, (mh, mw))
    if not candidates:
        return None
    mover_layout = random.choice(candidates)
    mr0, mc0, _, _ = mover_layout["bbox"]
    mover = _make_rect(mr0, mc0, mh, mw, outlier_color, _dir_to_corner(direction, (mh, mw)))
    return {"mover": mover}


def _dir_to_corner(direction: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    h, w = shape
    return (0 if dr < 0 else h - 1, 0 if dc < 0 else w - 1)


def _candidate_decoy_rect(
    shape: tuple[int, int],
    color: int,
    reserved: list[tuple[int, int, int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(150):
        rh = random.randint(2, 4)
        rw = random.randint(2, 4)
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        rect = _make_rect(r0, c0, rh, rw, color, _random_corner(rh, rw))
        expanded = _expand_bbox(rect["bbox"], 1, shape)
        if any(_bboxes_overlap(expanded, other) for other in reserved):
            continue
        return rect
    return None


def _validate_scene(scene: dict) -> None:
    rects = scene["rects"]
    assert len(rects) == 4
    shape = scene["shape"]
    for i, rect in enumerate(rects):
        assert _bbox_in_bounds(rect["bbox"], shape)
        for other in rects[i + 1 :]:
            assert not _bboxes_overlap(rect["bbox"], other["bbox"])

    colors = [rect["color"] for rect in rects]
    counts = Counter(colors)
    assert sorted(counts.values()) == [1, 3]

    mover = rects[scene["mover_idx"]]
    direction = _corner_to_dir(mover["corner"], (mover["h"], mover["w"]))
    others = [rect for i, rect in enumerate(rects) if i != scene["mover_idx"]]
    final_bbox, collision_rect, collision_bbox = _simulate_motion(mover["bbox"], direction, others, shape)
    path_boxes = _path_bboxes(scene)
    assert path_boxes[0] == mover["bbox"]
    assert path_boxes[-1] == final_bbox

    path_hits = [
        rect
        for rect in others
        if any(_bboxes_overlap(box, rect["bbox"]) for box in path_boxes)
    ]
    assert not path_hits, "no other rectangle may occupy a legal mover position"

    off_path_common = [
        rect
        for rect in others
        if all(not _bboxes_overlap(box, rect["bbox"]) for box in path_boxes)
    ]
    assert off_path_common, "at least one common-color rectangle must stay off the mover path"

    if scene["branch"] == "collision":
        assert collision_rect is not None and collision_bbox is not None
        future_hits = [rect for rect in others if _bboxes_overlap(collision_bbox, rect["bbox"])]
        assert len(future_hits) == 1
        split_cells = _split_target_cells(collision_rect, collision_bbox, direction, shape)
        split_comps = _components(_render_cells_mask(split_cells, shape))
        assert len(split_comps) == 2, "collision must create two separated halves"
        assert min(len(comp) for comp in split_comps) >= 2, "each split half must remain visually non-degenerate"
        fr0, fc0, fr1, fc1 = final_bbox
        assert all(not (fr0 <= r <= fr1 and fc0 <= c <= fc1) for r, c in split_cells), "split target must not overlap the stopped mover"
    else:
        assert collision_rect is None and collision_bbox is None
        assert not _bbox_in_bounds(_shift_bbox(final_bbox, *direction), shape)

    inp = _render_input(rects, shape)
    parsed_bboxes = sorted(rect["bbox"] for rect in _parse_rectangles(inp))
    scene_bboxes = sorted(rect["bbox"] for rect in rects)
    assert parsed_bboxes == scene_bboxes


def _generate_scene() -> dict | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    common_color, outlier_color = random.sample(FG_COLORS, 2)
    direction = random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    branch = random.choice(["collision", "boundary"])

    base = _sample_mover_and_target(shape, branch, direction, common_color, outlier_color)
    if base is None:
        return None

    rects = [base["mover"]]
    if branch == "collision":
        rects.append(base["target"])

    temp_scene = {"shape": shape, "rects": rects.copy(), "mover_idx": 0, "branch": branch}
    reserved = [_expand_bbox(box, 1, shape) for box in _path_bboxes(temp_scene)]
    reserved.extend(_expand_bbox(rect["bbox"], 1, shape) for rect in rects)
    reserved.extend(_expand_bbox(box, 1, shape) for box in _collision_support_bboxes(temp_scene))

    needed_decoys = 4 - len(rects)
    for _ in range(needed_decoys):
        decoy = _candidate_decoy_rect(shape, common_color, reserved)
        if decoy is None:
            return None
        rects.append(decoy)
        reserved.append(_expand_bbox(decoy["bbox"], 1, shape))

    scene = {"shape": shape, "rects": rects, "mover_idx": 0, "branch": branch}
    _validate_scene(scene)
    inp = _render_input(rects, shape)
    expected = _scene_output(scene)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return scene


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return _render_input(scene["rects"], scene["shape"])
    raise RuntimeError("failed to generate a valid input for puzzle 5")
