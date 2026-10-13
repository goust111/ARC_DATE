from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
BLUE = 1
ROLES = ("TL", "TR", "BL", "BR")

_ROLE_BY_REL = {
    ((0, 0), (0, 1), (1, 0)): "TL",
    ((0, 0), (0, 1), (1, 1)): "TR",
    ((0, 0), (1, 0), (1, 1)): "BL",
    ((0, 1), (1, 0), (1, 1)): "BR",
}

_ROT_ROLE = {
    0: {"TL": "TL", "TR": "TR", "BR": "BR", "BL": "BL"},
    1: {"TL": "BL", "TR": "TL", "BR": "TR", "BL": "BR"},
    2: {"TL": "BR", "TR": "BL", "BR": "TL", "BL": "TR"},
    3: {"TL": "TR", "TR": "BR", "BR": "BL", "BL": "TL"},
}


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_same_color(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color in (BG, BLUE) or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox_coords(coords), "area": len(cells)})
    return comps


def _corner_vertex(role: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if role == "TL":
        return r0, c0
    if role == "TR":
        return r0, c1
    if role == "BL":
        return r1, c0
    return r1, c1


def _parse_corners(g: np.ndarray) -> list[dict]:
    markers = []
    by_role: dict[str, dict[tuple[int, int], int]] = {role: {} for role in ROLES}
    for comp in _components_same_color(g):
        assert comp["area"] == 3, "every non-blue foreground component must be a 2x2 L corner"
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0, c1 - c0) == (1, 1), "corner marker must fit in a 2x2 box"
        rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in comp["coords"]))
        assert rel in _ROLE_BY_REL, "corner marker must be one of the four L orientations"
        role = _ROLE_BY_REL[rel]
        vertex = _corner_vertex(role, (r0, c0, r1, c1))
        assert vertex not in by_role[role], "corner vertices must be unique per role"
        marker = {
            "id": len(markers),
            "role": role,
            "vertex": vertex,
            "color": int(comp["color"]),
            "coords": comp["coords"],
        }
        by_role[role][vertex] = marker["id"]
        markers.append(marker)
    assert len(markers) >= 8 and len(markers) % 4 == 0, "expected several complete corner frames"
    return markers


def _find_candidate_frames(markers: list[dict]) -> list[dict]:
    by_role: dict[str, dict[tuple[int, int], dict]] = {role: {} for role in ROLES}
    for marker in markers:
        by_role[marker["role"]][marker["vertex"]] = marker

    candidates = []
    for tl in by_role["TL"].values():
        top, left = tl["vertex"]
        for tr in by_role["TR"].values():
            top2, right = tr["vertex"]
            if top2 != top or right - left < 4:
                continue
            for bl in by_role["BL"].values():
                bottom, left2 = bl["vertex"]
                if left2 != left or bottom - top < 4:
                    continue
                br = by_role["BR"].get((bottom, right))
                if br is None:
                    continue
                frame_markers = [tl, tr, bl, br]
                ids = frozenset(marker["id"] for marker in frame_markers)
                if len(ids) != 4:
                    continue
                counts = Counter(marker["color"] for marker in frame_markers)
                if sorted(counts.values()) != [1, 3]:
                    continue
                anchor_color = next(color for color, count in counts.items() if count == 1)
                anchor_roles = [marker["role"] for marker in frame_markers if marker["color"] == anchor_color]
                assert len(anchor_roles) == 1
                candidates.append(
                    {
                        "ids": ids,
                        "box": (int(top), int(left), int(bottom), int(right)),
                        "anchor_color": int(anchor_color),
                        "anchor_role": anchor_roles[0],
                    }
                )
    return candidates


def _solve_frame_cover(markers: list[dict], candidates: list[dict]) -> list[dict]:
    all_ids = frozenset(marker["id"] for marker in markers)
    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_anchor[candidate["anchor_color"]].append(candidate)

    solutions: list[list[dict]] = []
    for anchor_color, pool in by_anchor.items():
        by_marker: dict[int, list[dict]] = defaultdict(list)
        for candidate in pool:
            for marker_id in candidate["ids"]:
                by_marker[marker_id].append(candidate)

        def backtrack(used: frozenset[int], chosen: list[dict]) -> None:
            if len(solutions) > 1:
                return
            if used == all_ids:
                solutions.append(chosen.copy())
                return
            remaining = [marker_id for marker_id in all_ids if marker_id not in used]
            marker_id = min(remaining, key=lambda item: len(by_marker[item]))
            for candidate in by_marker[marker_id]:
                if used & candidate["ids"]:
                    continue
                backtrack(used | candidate["ids"], chosen + [candidate])

        if pool:
            backtrack(frozenset(), [])

    assert len(solutions) == 1, "corner markers must determine one uniform-anchor frame grouping"
    frames = sorted(solutions[0], key=lambda frame: frame["box"])
    assert len({frame["anchor_color"] for frame in frames}) == 1, "all frames must share one anchor color"
    return frames


def _find_frames(g: np.ndarray) -> list[dict]:
    markers = _parse_corners(g)
    candidates = _find_candidate_frames(markers)
    assert candidates, "expected structurally valid corner frames"
    return _solve_frame_cover(markers, candidates)


def _interior(box: tuple[int, int, int, int]) -> tuple[slice, slice]:
    top, left, bottom, right = box
    return slice(top + 1, bottom), slice(left + 1, right)


def _rotation_from_master(master: dict, target: dict) -> int:
    mt, ml, mb, mr = master["box"]
    tt, tl, tb, tr = target["box"]
    mh, mw = mb - mt + 1, mr - ml + 1
    th, tw = tb - tt + 1, tr - tl + 1
    options = []
    for k in range(4):
        rh, rw = (mh, mw) if k % 2 == 0 else (mw, mh)
        if (rh, rw) == (th, tw) and _ROT_ROLE[k][master["anchor_role"]] == target["anchor_role"]:
            options.append(k)
    assert len(options) == 1, "target frame must have one anchor-compatible rotation"
    return options[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the one patterned invisible corner frame into the empty frames after rotating it to each anchor corner.

    Input structure:
    - A black background contains several invisible rectangles marked only by four 2x2 L-shaped corner triominoes.
    - In each rectangle, three corner markers use that frame's color and exactly one corner uses the global anchor color.
    - All rectangles have the same outer size or its transpose, and exactly one rectangle contains blue pixels in its interior.

    Rule:
    - Keep every corner marker fixed.
    - Use the only frame with interior blue pixels as the master stamp.
    - For each empty frame, rotate the master's interior blue pattern by the unique 0, 90, 180, or 270 degree rotation that maps the master outer size and anchor corner to that frame.
    - Paint the rotated blue pattern into the target frame's interior.

    Key insight:
    - The L corners reveal only invisible boxes; the differently colored anchor corner is the shared reference point that determines how the interior stamp rotates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _find_frames(g)
    assert len(frames) >= 2, "expected a master frame and at least one target frame"

    patterned = []
    frame_masks = []
    for frame in frames:
        rr, cc = _interior(frame["box"])
        interior = g[rr, cc]
        assert not np.any((interior != BG) & (interior != BLUE)), "frame interiors may contain only blue pattern pixels"
        mask = interior == BLUE
        frame_masks.append(mask)
        if bool(mask.any()):
            patterned.append(frame)
    assert len(patterned) == 1, "exactly one frame must contain the blue master pattern"
    master = patterned[0]
    master_index = frames.index(master)
    master_rr, master_cc = _interior(master["box"])
    master_patch = np.full_like(g[master_rr, master_cc], BG)
    master_patch[g[master_rr, master_cc] == BLUE] = BLUE

    outside = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        rr, cc = _interior(frame["box"])
        outside[rr, cc] = True
    assert not np.any((g == BLUE) & ~outside), "blue pixels must be inside the master frame"

    out = g.copy()
    for idx, target in enumerate(frames):
        if idx == master_index:
            continue
        assert not frame_masks[idx].any(), "non-master frames must start empty"
        k = _rotation_from_master(master, target)
        stamp = np.rot90(master_patch, k)
        rr, cc = _interior(target["box"])
        assert stamp.shape == out[rr, cc].shape, "rotated stamp must fit the target interior"
        out[rr, cc][stamp == BLUE] = BLUE
    return out


def _corner_cells(role: str, box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    top, left, bottom, right = box
    if role == "TL":
        return ((top, left), (top, left + 1), (top + 1, left))
    if role == "TR":
        return ((top, right - 1), (top, right), (top + 1, right))
    if role == "BL":
        return ((bottom - 1, left), (bottom, left), (bottom, left + 1))
    return ((bottom - 1, right), (bottom, right - 1), (bottom, right))


def _random_pattern(h: int, w: int) -> np.ndarray | None:
    area = h * w
    lo = max(4, area // 5)
    hi = min(max(lo, area // 2), 12)
    if lo > hi:
        return None
    count = random.randint(lo, hi)
    for _ in range(100):
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < count:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier or random.random() < 0.25:
                free = np.argwhere(~mask)
                r, c = random.choice(free.tolist())
            else:
                r, c = random.choice(tuple(frontier))
            mask[r, c] = True
        if np.count_nonzero(mask.any(axis=0)) < 2 or np.count_nonzero(mask.any(axis=1)) < 2:
            continue
        if h == w and any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if h != w and np.array_equal(mask, np.rot90(mask, 2)):
            continue
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = BLUE
        return patch
    return None


def _place_boxes(shapes: list[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int, int, int]] | None:
    used = np.zeros((rows, cols), dtype=bool)
    used_tops: set[int] = set()
    used_bottoms: set[int] = set()
    used_lefts: set[int] = set()
    used_rights: set[int] = set()
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)
    for idx in order:
        h, w = shapes[idx]
        candidates = []
        for top in range(1, rows - h):
            bottom = top + h - 1
            if top in used_tops or bottom in used_bottoms:
                continue
            for left in range(1, cols - w):
                right = left + w - 1
                if left in used_lefts or right in used_rights:
                    continue
                rr0, cc0 = max(0, top - 1), max(0, left - 1)
                rr1, cc1 = min(rows - 1, bottom + 1), min(cols - 1, right + 1)
                if not used[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    candidates.append((top, left, bottom, right))
        if not candidates:
            return None
        box = random.choice(candidates)
        boxes[idx] = box
        top, left, bottom, right = box
        used[max(0, top - 1) : min(rows, bottom + 2), max(0, left - 1) : min(cols, right + 2)] = True
        used_tops.add(top)
        used_bottoms.add(bottom)
        used_lefts.add(left)
        used_rights.add(right)
    return [box for box in boxes if box is not None]


def _render_input_and_expected() -> tuple[np.ndarray, np.ndarray] | None:
    outer_h = random.randint(6, 9)
    outer_w = random.randint(7, 11)
    if outer_h == outer_w:
        outer_w += 1
    master_anchor = random.choice(ROLES)
    if random.random() < 0.65:
        rotations = [0, 1, 2, 3]
    else:
        rotations = [k for k in range(4) if k != random.randrange(4)]
    random.shuffle(rotations)

    frame_count = 1 + len(rotations)
    palette = list(range(2, 10))
    anchor_color = random.choice(palette)
    frame_colors = random.sample([color for color in palette if color != anchor_color], frame_count)

    shapes = [(outer_h, outer_w)]
    anchors = [master_anchor]
    for k in rotations:
        shapes.append((outer_h, outer_w) if k % 2 == 0 else (outer_w, outer_h))
        anchors.append(_ROT_ROLE[k][master_anchor])

    rows = random.randint(24, 30)
    cols = random.randint(24, 30)
    boxes = _place_boxes(shapes, rows, cols)
    if boxes is None:
        return None

    pattern = _random_pattern(outer_h - 2, outer_w - 2)
    if pattern is None:
        return None

    inp = np.zeros((rows, cols), dtype=int)
    expected = np.zeros((rows, cols), dtype=int)
    for i, (box, anchor_role, frame_color) in enumerate(zip(boxes, anchors, frame_colors)):
        for role in ROLES:
            color = anchor_color if role == anchor_role else frame_color
            for r, c in _corner_cells(role, box):
                if inp[r, c] != BG:
                    return None
                inp[r, c] = color
                expected[r, c] = color

    rr, cc = _interior(boxes[0])
    inp[rr, cc][pattern == BLUE] = BLUE
    expected[rr, cc][pattern == BLUE] = BLUE

    for box, k in zip(boxes[1:], rotations):
        stamp = np.rot90(pattern, k)
        rr, cc = _interior(box)
        if stamp.shape != expected[rr, cc].shape:
            return None
        expected[rr, cc][stamp == BLUE] = BLUE

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if len({hash(stamp.tobytes()) for stamp in [pattern, np.rot90(pattern, 1), np.rot90(pattern, 2), np.rot90(pattern, 3)]}) < 3:
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        rendered = _render_input_and_expected()
        if rendered is not None:
            return rendered[0]
    raise RuntimeError("failed to generate a valid invisible-frame anchor-stamp input")
