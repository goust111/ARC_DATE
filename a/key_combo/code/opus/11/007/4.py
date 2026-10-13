from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
MARKER_COLORS = (1, 2, 3)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = outer
    rr0, cc0, rr1, cc1 = inner
    return r0 < rr0 and c0 < cc0 and r1 > rr1 and c1 > cc1


def _inside_frame_interior(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _frame_direct_region(
    bbox: tuple[int, int, int, int], child_bbox: tuple[int, int, int, int] | None
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (
                child_bbox is not None
                and child_bbox[0] <= r <= child_bbox[2]
                and child_bbox[1] <= c <= child_bbox[3]
            ):
                continue
            cells.append((r, c))
    return cells


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME_COLOR):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        expected = set(_border_cells(bbox))
        comp_set = set(comp)
        if comp_set != expected:
            continue
        frames.append(
            {
                "bbox": bbox,
                "border": list(expected),
                "parent": None,
                "children": [],
            }
        )

    for idx, frame in enumerate(frames):
        parents = [
            j
            for j, other in enumerate(frames)
            if j != idx and _contains_bbox(other["bbox"], frame["bbox"])
        ]
        if parents:
            parent = min(parents, key=lambda j: _bbox_area(frames[j]["bbox"]))
            frame["parent"] = parent
            frames[parent]["children"].append(idx)
    return frames


def _assign_direct_markers(
    grid: np.ndarray, frames: list[dict]
) -> tuple[list[list[tuple[int, int, int]]], list[tuple[int, int, int]]]:
    owned: list[list[tuple[int, int, int]]] = [[] for _ in frames]
    all_markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != BACKGROUND) & (grid != FRAME_COLOR)):
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        all_markers.append((rr, cc, color))
        containers = [
            idx for idx, frame in enumerate(frames) if _inside_frame_interior(frame["bbox"], rr, cc)
        ]
        if not containers:
            continue
        owner = min(containers, key=lambda idx: _bbox_area(frames[idx]["bbox"]))
        owned[owner].append((rr, cc, color))
    return owned, all_markers


def _frame_output_color(bbox: tuple[int, int, int, int], markers: list[tuple[int, int, int]]) -> int:
    if not markers:
        return FRAME_COLOR
    if len(markers) == 1:
        return int(markers[0][2])
    r0, c0, _r1, _c1 = bbox
    winner = min(markers, key=lambda item: (abs(item[0] - r0) + abs(item[1] - c0), item[0], item[1]))
    return int(winner[2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested gray frame by counting only the markers in its direct interior region.

    Input structure:
    - The grid has a black background, multiple gray 1-pixel-thick rectangular frame components,
      and sparse colored marker pixels inside those frame systems.
    - Frames may be nested into 2-3 layer chains; separate chains do not overlap.
    - Marker pixels lie strictly inside frames and never on frame borders.

    Rule:
    - Detect every gray rectangular frame and build the containment tree of nested frames.
    - Assign each marker to the smallest frame whose interior contains it; this is exactly that
      frame's direct interior, because markers inside child-frame interiors do not count for ancestors.
    - Recolor each frame: keep gray for count 0; use the unique marker's color for count 1; for
      count >= 2 use the color of the direct-interior marker nearest the frame's top-left border
      corner by Manhattan distance, breaking ties by smaller row then smaller column. Remove all
      marker pixels to background.

    Key insight:
    - Counting must happen on direct interiors defined by the containment tree, not on the whole
      bounding box of a frame, or outer frames would incorrectly absorb markers belonging to inner ones.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    owned_markers, all_markers = _assign_direct_markers(g, frames)

    out = g.copy()
    for r, c, _color in all_markers:
        out[r, c] = BACKGROUND

    for frame, markers in zip(frames, owned_markers):
        color = _frame_output_color(frame["bbox"], markers)
        for r, c in frame["border"]:
            out[r, c] = color
    return out


def _metric(anchor: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int, int]:
    ar, ac = anchor
    r, c = cell
    return abs(r - ar) + abs(c - ac), r, c


def _sample_group_profiles() -> list[list[int]]:
    groups: list[list[int]] = []

    high_a = random.choice([2, 3])
    middle_pair = [1, high_a]
    random.shuffle(middle_pair)
    groups.append([0, *middle_pair])

    if random.random() < 0.6:
        high_b = random.choice([2, 3])
        other = random.choice([0, 1])
        counts = [high_b, other]
        if random.random() < 0.5:
            counts.reverse()
        groups.append(counts)
    else:
        high_b = random.choice([2, 3])
        others = random.sample([value for value in (0, 1, 2, 3) if value != high_b], 2)
        counts = [high_b, *others]
        random.shuffle(counts)
        groups.append(counts)

    if random.random() < 0.35:
        depth = random.choice([2, 3])
        counts = random.sample([0, 1, 2, 3], depth)
        groups.append(counts)

    assert len(groups) in (2, 3)
    assert any(count == 1 for counts in groups for count in counts)
    assert sum(count >= 2 for counts in groups for count in counts) >= 2
    assert groups[0][0] == 0 and any(count > 0 for count in groups[0][1:])
    return groups


def _sample_group_boxes(counts: list[int]) -> list[tuple[int, int, int, int]] | None:
    depth = len(counts)
    outer_min = 11 if depth == 3 else 7
    outer_max = 14 if depth == 3 else 12

    for _ in range(200):
        h = random.randint(outer_min, outer_max)
        w = random.randint(outer_min, outer_max)
        boxes = [(0, 0, h - 1, w - 1)]
        ok = True
        for level in range(1, depth):
            pr0, pc0, pr1, pc1 = boxes[-1]
            ph = pr1 - pr0 + 1
            pw = pc1 - pc0 + 1
            remaining = depth - level - 1
            min_child_h = 3 + 4 * remaining
            min_child_w = 3 + 4 * remaining
            max_child_h = ph - 4
            max_child_w = pw - 4
            if min_child_h > max_child_h or min_child_w > max_child_w:
                ok = False
                break
            ch = random.randint(min_child_h, max_child_h)
            cw = random.randint(min_child_w, max_child_w)
            top = random.randint(2, ph - ch - 2)
            left = random.randint(2, pw - cw - 2)
            bottom = ph - ch - top
            right = pw - cw - left
            if bottom < 2 or right < 2:
                ok = False
                break
            child = (pr0 + top, pc0 + left, pr1 - bottom, pc1 - right)
            boxes.append(child)
        if not ok:
            continue

        capacities = []
        for idx, bbox in enumerate(boxes):
            child = boxes[idx + 1] if idx + 1 < depth else None
            capacities.append(len(_frame_direct_region(bbox, child)))
        if any(capacity < need for capacity, need in zip(capacities, counts)):
            continue

        inner = boxes[-1]
        ih = inner[2] - inner[0] - 1
        iw = inner[3] - inner[1] - 1
        if ih * iw < counts[-1]:
            continue
        return boxes
    return None


def _translate_boxes(
    boxes: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _outer_size(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = boxes[0]
    return r1 - r0 + 1, c1 - c0 + 1


def _place_groups(
    shape: tuple[int, int], group_boxes: list[list[tuple[int, int, int, int]]]
) -> list[list[tuple[int, int, int, int]]] | None:
    h, w = shape
    order = sorted(range(len(group_boxes)), key=lambda idx: _bbox_area(group_boxes[idx][0]), reverse=True)
    placed: dict[int, list[tuple[int, int, int, int]]] = {}
    outer_bboxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        gh, gw = _outer_size(group_boxes[idx])
        positions = [(r, c) for r in range(h - gh + 1) for c in range(w - gw + 1)]
        random.shuffle(positions)
        chosen: tuple[int, int] | None = None
        for r, c in positions:
            bbox = (r, c, r + gh - 1, c + gw - 1)
            ok = True
            for or0, oc0, or1, oc1 in outer_bboxes:
                if not (bbox[2] + 2 < or0 or or1 + 2 < bbox[0] or bbox[3] + 2 < oc0 or oc1 + 2 < bbox[1]):
                    ok = False
                    break
            if ok:
                chosen = (r, c)
                break
        if chosen is None:
            return None
        placed[idx] = _translate_boxes(group_boxes[idx], chosen[0], chosen[1])
        outer_bboxes.append(placed[idx][0])

    return [placed[idx] for idx in range(len(group_boxes))]


def _plan_single_marker(cells: list[tuple[int, int]]) -> dict:
    pos = random.choice(cells)
    color = random.choice(MARKER_COLORS)
    return {
        "markers": [(pos[0], pos[1], color)],
        "winner": (pos[0], pos[1], color),
        "mode": "single",
    }


def _plan_tie_markers(
    bbox: tuple[int, int, int, int], child_bbox: tuple[int, int, int, int] | None, count: int
) -> dict | None:
    anchor = (bbox[0], bbox[1])
    cells = _frame_direct_region(bbox, child_bbox)
    by_dist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cell in cells:
        by_dist[_metric(anchor, cell)[0]].append(cell)

    choices = []
    for dist, pts in by_dist.items():
        pts = sorted(pts)
        if len(pts) < 2:
            continue
        varied_rows = [pair for pair in ((a, b) for a in pts for b in pts if a < b) if pair[0][0] != pair[1][0]]
        pairs = varied_rows or [(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))]
        for first, second in pairs:
            winner_cell = min((first, second), key=lambda cell: _metric(anchor, cell))
            loser_cell = second if winner_cell == first else first
            extras = [
                cell
                for cell in cells
                if cell not in (winner_cell, loser_cell) and _metric(anchor, cell) > _metric(anchor, winner_cell)
            ]
            if count == 2 or extras:
                choices.append((winner_cell, loser_cell, extras))
    if not choices:
        return None

    winner_cell, loser_cell, extras = random.choice(choices)
    winner_color = random.choice(MARKER_COLORS)
    loser_color = random.choice([color for color in MARKER_COLORS if color != winner_color])
    markers = [(winner_cell[0], winner_cell[1], winner_color), (loser_cell[0], loser_cell[1], loser_color)]
    if count == 3:
        extra_cell = random.choice(extras)
        markers.append((extra_cell[0], extra_cell[1], loser_color))
    return {
        "markers": markers,
        "winner": (winner_cell[0], winner_cell[1], winner_color),
        "mode": "tie",
    }


def _plan_clear_markers(
    bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int] | None,
    count: int,
    *,
    require_trap: bool,
) -> dict | None:
    anchor = (bbox[0], bbox[1])
    cells = _frame_direct_region(bbox, child_bbox)
    scored = sorted(((_metric(anchor, cell), cell) for cell in cells), key=lambda item: item[0])

    candidates = []
    fallback = []
    for winner_metric, winner_cell in scored:
        farther = [cell for metric, cell in scored if metric > winner_metric]
        if not farther:
            continue
        trap_cells = [
            cell
            for cell in farther
            if cell[0] < winner_cell[0] or (cell[0] == winner_cell[0] and cell[1] < winner_cell[1])
        ]
        if trap_cells:
            candidates.append((winner_cell, random.choice(trap_cells)))
        fallback.append((winner_cell, random.choice(farther)))

    pairs = candidates if require_trap else (candidates or fallback)
    if not pairs:
        return None

    winner_cell, loser_cell = random.choice(pairs)
    winner_metric = _metric(anchor, winner_cell)
    extras = [cell for cell in cells if cell not in (winner_cell, loser_cell) and _metric(anchor, cell) > winner_metric]
    if count == 3 and not extras:
        return None

    winner_color = random.choice(MARKER_COLORS)
    loser_color = random.choice([color for color in MARKER_COLORS if color != winner_color])
    markers = [(winner_cell[0], winner_cell[1], winner_color), (loser_cell[0], loser_cell[1], loser_color)]
    if count == 3:
        extra_cell = random.choice(extras)
        markers.append((extra_cell[0], extra_cell[1], loser_color))
    return {
        "markers": markers,
        "winner": (winner_cell[0], winner_cell[1], winner_color),
        "mode": "clear_trap" if require_trap else "clear_any",
    }


def _expected_output(grid: np.ndarray, blueprint: dict) -> np.ndarray:
    out = grid.copy()
    out[(grid != BACKGROUND) & (grid != FRAME_COLOR)] = BACKGROUND
    for frame in blueprint["frames"]:
        for r, c in _border_cells(frame["bbox"]):
            out[r, c] = frame["expected_color"]
    return out


def _validate_blueprint(grid: np.ndarray, blueprint: dict) -> None:
    frames = _extract_frames(grid)
    expected_boxes = {tuple(frame["bbox"]) for frame in blueprint["frames"]}
    actual_boxes = {tuple(frame["bbox"]) for frame in frames}
    assert actual_boxes == expected_boxes

    owned, markers = _assign_direct_markers(grid, frames)
    frame_by_box = {tuple(frame["bbox"]): frame for frame in blueprint["frames"]}
    for frame, marks in zip(frames, owned):
        spec = frame_by_box[tuple(frame["bbox"])]
        assert len(marks) == spec["count"]
        actual_color = _frame_output_color(frame["bbox"], marks)
        assert actual_color == spec["expected_color"]

    counts = [frame["count"] for frame in blueprint["frames"]]
    assert 0 in counts and 1 in counts and sum(count >= 2 for count in counts) >= 2
    assert len({color for _r, _c, color in markers}) >= 2
    assert any(
        group["counts"][0] == 0 and any(count > 0 for count in group["counts"][1:])
        for group in blueprint["groups"]
    )

    tie_frames = [frame for frame in blueprint["frames"] if frame.get("mode") == "tie"]
    clear_frames = [frame for frame in blueprint["frames"] if frame.get("mode") == "clear_trap"]
    assert tie_frames and clear_frames
    tie_frame = tie_frames[0]
    clear_frame = clear_frames[0]

    tie_anchor = (tie_frame["bbox"][0], tie_frame["bbox"][1])
    tie_metrics = sorted(_metric(tie_anchor, (r, c)) for r, c, _color in tie_frame["markers"])
    assert len(tie_metrics) >= 2 and tie_metrics[0][0] == tie_metrics[1][0]

    clear_anchor = (clear_frame["bbox"][0], clear_frame["bbox"][1])
    winner = clear_frame["winner"]
    distractors = [marker for marker in clear_frame["markers"] if marker != winner]
    assert distractors
    assert any(
        _metric(clear_anchor, (marker[0], marker[1])) > _metric(clear_anchor, (winner[0], winner[1]))
        and (marker[0] < winner[0] or (marker[0] == winner[0] and marker[1] < winner[1]))
        for marker in distractors
    )

    expected = _expected_output(grid, blueprint)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert np.all(actual[(grid != BACKGROUND) & (grid != FRAME_COLOR)] == BACKGROUND)


def _generate_attempt() -> tuple[np.ndarray, dict] | None:
    groups = _sample_group_profiles()
    group_boxes_rel = []
    for counts in groups:
        boxes = _sample_group_boxes(counts)
        if boxes is None:
            return None
        group_boxes_rel.append(boxes)

    h = random.randint(16, 24)
    w = random.randint(16, 24)
    group_boxes = _place_groups((h, w), group_boxes_rel)
    if group_boxes is None:
        return None

    frame_specs = []
    for counts, boxes in zip(groups, group_boxes):
        assert len(counts) == len(boxes)
        for idx, (count, bbox) in enumerate(zip(counts, boxes)):
            child_bbox = boxes[idx + 1] if idx + 1 < len(boxes) else None
            frame_specs.append(
                {
                    "bbox": bbox,
                    "child_bbox": child_bbox,
                    "count": count,
                    "markers": [],
                    "expected_color": FRAME_COLOR,
                    "mode": None,
                    "winner": None,
                }
            )

    high_indices = [idx for idx, frame in enumerate(frame_specs) if frame["count"] >= 2]
    tie_candidates = [idx for idx in high_indices if _plan_tie_markers(frame_specs[idx]["bbox"], frame_specs[idx]["child_bbox"], frame_specs[idx]["count"]) is not None]
    clear_candidates = [
        idx
        for idx in high_indices
        if _plan_clear_markers(
            frame_specs[idx]["bbox"],
            frame_specs[idx]["child_bbox"],
            frame_specs[idx]["count"],
            require_trap=True,
        )
        is not None
    ]
    if not tie_candidates or not clear_candidates:
        return None

    tie_idx = random.choice(tie_candidates)
    clear_pool = [idx for idx in clear_candidates if idx != tie_idx]
    if not clear_pool:
        return None
    clear_idx = random.choice(clear_pool)

    for idx, frame in enumerate(frame_specs):
        count = frame["count"]
        if count == 0:
            continue
        if count == 1:
            plan = _plan_single_marker(_frame_direct_region(frame["bbox"], frame["child_bbox"]))
        elif idx == tie_idx:
            plan = _plan_tie_markers(frame["bbox"], frame["child_bbox"], count)
        elif idx == clear_idx:
            plan = _plan_clear_markers(frame["bbox"], frame["child_bbox"], count, require_trap=True)
        else:
            plan = _plan_clear_markers(frame["bbox"], frame["child_bbox"], count, require_trap=False)
            if plan is None:
                plan = _plan_tie_markers(frame["bbox"], frame["child_bbox"], count)
        if plan is None:
            return None
        frame["markers"] = plan["markers"]
        frame["winner"] = plan["winner"]
        frame["mode"] = plan["mode"]
        frame["expected_color"] = int(plan["winner"][2])

    grid = np.zeros((h, w), dtype=int)
    for frame in frame_specs:
        for r, c in _border_cells(frame["bbox"]):
            grid[r, c] = FRAME_COLOR
        for r, c, color in frame["markers"]:
            grid[r, c] = color

    blueprint = {
        "groups": [
            {"boxes": boxes, "counts": counts}
            for boxes, counts in zip(group_boxes, groups)
        ],
        "frames": frame_specs,
    }
    _validate_blueprint(grid, blueprint)
    return grid, blueprint


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid nested-frame ARC instance")
