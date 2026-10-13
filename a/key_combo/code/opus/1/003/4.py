from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAME = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_full_width_bar(coords: np.ndarray, h: int, w: int, row: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return (
        len(coords) == w
        and r0 == row
        and r1 == row
        and c0 == 0
        and c1 == w - 1
    )


def _frame_bbox(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return None

    comp_mask = np.zeros(shape, dtype=bool)
    comp_mask[coords[:, 0], coords[:, 1]] = True

    ring = np.zeros((hh, ww), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(comp_mask[r0 : r1 + 1, c0 : c1 + 1], ring):
        return None
    return r0, c0, r1, c1


def _checkerboard_capacity(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    return (ih * iw + 1) // 2


def _checkerboard_fill(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if ((r - (r0 + 1)) + (c - (c0 + 1))) % 2 == 0:
                cells.append((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each border-colored object to the hollow frame whose checkerboard count equals its virtual slide distance.

    Input structure:
    - The background is black, the top row is a full-width red bar, and the bottom row is a full-width blue bar.
    - The interior contains small 4-connected monochrome red or blue objects and several gray one-cell-thick hollow rectangles.
    - Each rectangle has an empty black interior, and different rectangles have different upper-left-starting checkerboard counts.

    Rule:
    - For each red object, measure the upward slide distance from its bbox top to row 1; for each blue object, measure the downward slide distance from its bbox bottom to row h-2.
    - Find the unique gray rectangle whose checkerboard cell count equals that distance.
    - Fill that rectangle's checkerboard cells with the object's color, leave the other interior cells black, and erase the original object pixels.

    Key insight:
    - The border bars are not destinations to draw toward; they define a virtual distance that is re-used as the numeric key for checkerboard-capacity matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5
    assert np.all(g[0, :] == g[0, 0])
    assert np.all(g[h - 1, :] == g[h - 1, 0])

    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])
    assert top_color != bottom_color

    frames: dict[int, tuple[int, int, int, int]] = {}
    for coords in _components(g == FRAME):
        bbox = _frame_bbox(coords, g.shape)
        assert bbox is not None, "gray components must be rectangular frames"
        cap = _checkerboard_capacity(bbox)
        assert cap not in frames, "frame checkerboard counts must be unique"
        frames[cap] = bbox

    assert len(frames) >= 2

    objects: list[tuple[int, np.ndarray, int]] = []
    for color, side in ((top_color, "top"), (bottom_color, "bottom")):
        comps = _components(g == color)
        seen_non_bar = 0
        for coords in comps:
            if _is_full_width_bar(coords, h, w, 0 if side == "top" else h - 1):
                continue
            seen_non_bar += 1
            r0, _c0, r1, _c1 = _bbox(coords)
            if side == "top":
                dist = r0 - 1
            else:
                dist = (h - 2) - r1
            assert dist > 0
            objects.append((int(color), coords, int(dist)))
        assert seen_non_bar <= 1, "expected at most one non-bar object per border color"

    assert len(objects) >= 1
    distances = [dist for _color, _coords, dist in objects]
    assert len(distances) == len(set(distances)), "object distances must be unique"
    assert set(distances).issubset(frames), "every object distance must match a frame"

    out = g.copy()
    for _color, coords, _dist in objects:
        out[coords[:, 0], coords[:, 1]] = BG

    for color, _coords, dist in objects:
        for r, c in _checkerboard_fill(frames[dist]):
            assert out[r, c] == BG
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (18, 24)
MATCH_CAP_MIN = 2
MATCH_CAP_MAX = 9
INTERIOR_H_RANGE = range(1, 6)
INTERIOR_W_RANGE = range(1, 7)
GEN_RETRIES = 400
OBJECT_RETRIES = 80


def _build_capacity_map() -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for ih in INTERIOR_H_RANGE:
        for iw in INTERIOR_W_RANGE:
            cap = (ih * iw + 1) // 2
            out.setdefault(cap, []).append((ih, iw))
    return out


CAPACITY_TO_SIZES = _build_capacity_map()


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    hh, ww = patch.shape
    for r in range(hh):
        for c in range(ww):
            value = int(patch[r, c])
            if transparent is not None and value == transparent:
                continue
            grid[r0 + r, c0 + c] = value


def _random_object_patch(color: int) -> np.ndarray:
    for _ in range(OBJECT_RETRIES):
        area = random.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    frontier.add((r + dr, c + dc))
            frontier.difference_update(cells)
            cells.add(random.choice(tuple(frontier)))

        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        hh = 1 + max(r for r, _c in norm)
        ww = 1 + max(c for _r, c in norm)
        if hh > 3 or ww > 4:
            continue
        patch = np.full((hh, ww), BG, dtype=int)
        for r, c in norm:
            patch[r, c] = color
        return patch
    raise RuntimeError("failed to build a small connected object")


def _bbox_from_origin(patch: np.ndarray, r0: int, c0: int) -> tuple[int, int, int, int]:
    hh, ww = patch.shape
    return r0, c0, r0 + hh - 1, c0 + ww - 1


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _frame_patch(ih: int, iw: int) -> np.ndarray:
    patch = np.full((ih + 2, iw + 2), FRAME, dtype=int)
    patch[1:-1, 1:-1] = BG
    return patch


def _place_frames(h: int, w: int, frames: list[dict[str, int]]) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(frames)), key=lambda i: (frames[i]["ih"] + 2) * (frames[i]["iw"] + 2), reverse=True)
    for idx in order:
        frame = frames[idx]
        oh = frame["ih"] + 2
        ow = frame["iw"] + 2
        candidates: list[tuple[int, int]] = []
        for r0 in range(2, h - oh - 1):
            for c0 in range(1, w - ow):
                bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                if any(_boxes_too_close(bbox, other, pad=1) for other in placed):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        frame["r0"] = r0
        frame["c0"] = c0
        placed.append((r0, c0, r0 + oh - 1, c0 + ow - 1))
    return True


def _place_object(
    h: int,
    w: int,
    patch: np.ndarray,
    fixed_r0: int,
    blocked_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    hh, ww = patch.shape
    cols = list(range(1, w - ww))
    random.shuffle(cols)
    for c0 in cols:
        bbox = _bbox_from_origin(patch, fixed_r0, c0)
        if any(_boxes_too_close(bbox, other, pad=1) for other in blocked_boxes):
            continue
        return fixed_r0, c0
    return None


def _absolute_coords(patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    return coords + np.array([r0, c0], dtype=int)


def _choose_distances(
    h: int,
    red_patch: np.ndarray,
    blue_patch: np.ndarray,
) -> tuple[int, int, int] | None:
    red_area = int(np.count_nonzero(red_patch))
    blue_area = int(np.count_nonzero(blue_patch))
    red_h = red_patch.shape[0]
    blue_h = blue_patch.shape[0]

    red_options = [
        cap
        for cap in CAPACITY_TO_SIZES
        if MATCH_CAP_MIN <= cap <= MATCH_CAP_MAX
        and cap <= h - 2 - red_h
        and cap not in {red_area, blue_area}
    ]
    blue_options = [
        cap
        for cap in CAPACITY_TO_SIZES
        if MATCH_CAP_MIN <= cap <= MATCH_CAP_MAX
        and cap <= h - 2 - blue_h
        and cap not in {red_area, blue_area}
    ]
    pairs = [(dr, db) for dr in red_options for db in blue_options if dr != db]
    random.shuffle(pairs)

    for red_dist, blue_dist in pairs:
        red_wrong_bottom = h - red_dist - red_h - 2
        blue_wrong_top = h - blue_dist - blue_h - 2
        tempting = []
        for cap in (red_wrong_bottom, blue_wrong_top, red_area, blue_area):
            if cap in CAPACITY_TO_SIZES and cap not in {red_dist, blue_dist}:
                tempting.append(int(cap))
        if tempting:
            return red_dist, blue_dist, random.choice(tempting)
    return None


def _build_expected(inp: np.ndarray, frames: list[dict[str, int]], objects: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        coords = obj["coords"]
        assert isinstance(coords, np.ndarray)
        out[coords[:, 0], coords[:, 1]] = BG
    by_capacity = {
        int(_checkerboard_capacity((frame["r0"], frame["c0"], frame["r0"] + frame["ih"] + 1, frame["c0"] + frame["iw"] + 1))): frame
        for frame in frames
    }
    for obj in objects:
        color = int(obj["color"])
        dist = int(obj["distance"])
        frame = by_capacity[dist]
        bbox = (frame["r0"], frame["c0"], frame["r0"] + frame["ih"] + 1, frame["c0"] + frame["iw"] + 1)
        for r, c in _checkerboard_fill(bbox):
            out[r, c] = color
    return out


def _semantic_guard(h: int, frames: list[dict[str, int]], objects: list[dict[str, object]]) -> bool:
    matched_caps: set[int] = set()
    matched_parities: list[int] = []
    object_areas: set[int] = set()
    wrong_distances: set[int] = set()

    for obj in objects:
        coords = obj["coords"]
        patch = obj["patch"]
        color = int(obj["color"])
        dist = int(obj["distance"])
        assert isinstance(coords, np.ndarray)
        assert isinstance(patch, np.ndarray)

        matched_caps.add(dist)
        object_areas.add(int(np.count_nonzero(patch)))
        r0, _c0, r1, _c1 = _bbox(coords)
        if color == RED:
            wrong_distances.add((h - 2) - r1)
        else:
            wrong_distances.add(r0 - 1)

    frame_caps = []
    for frame in frames:
        bbox = (frame["r0"], frame["c0"], frame["r0"] + frame["ih"] + 1, frame["c0"] + frame["iw"] + 1)
        cap = _checkerboard_capacity(bbox)
        frame_caps.append(cap)
        if cap in matched_caps:
            matched_parities.append((frame["r0"] + frame["c0"]) % 2)

    unused_caps = set(frame_caps) - matched_caps
    if len(unused_caps) != 1:
        return False
    unused_cap = next(iter(unused_caps))

    if not any(parity == 1 for parity in matched_parities):
        return False
    if not matched_caps.isdisjoint(object_areas):
        return False
    if unused_cap not in object_areas | wrong_distances:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    red_patch = _random_object_patch(RED)
    blue_patch = _random_object_patch(BLUE)

    picked = _choose_distances(h, red_patch, blue_patch)
    if picked is None:
        return None
    red_dist, blue_dist, distractor = picked

    red_size = random.choice(CAPACITY_TO_SIZES[red_dist])
    blue_size = random.choice(CAPACITY_TO_SIZES[blue_dist])
    distractor_size = random.choice(CAPACITY_TO_SIZES[distractor])
    frames = [
        {"ih": int(red_size[0]), "iw": int(red_size[1])},
        {"ih": int(blue_size[0]), "iw": int(blue_size[1])},
        {"ih": int(distractor_size[0]), "iw": int(distractor_size[1])},
    ]
    random.shuffle(frames)
    if not _place_frames(h, w, frames):
        return None

    blocked = [
        (frame["r0"], frame["c0"], frame["r0"] + frame["ih"] + 1, frame["c0"] + frame["iw"] + 1)
        for frame in frames
    ]
    red_origin = _place_object(h, w, red_patch, red_dist + 1, blocked)
    if red_origin is None:
        return None
    blocked.append(_bbox_from_origin(red_patch, red_origin[0], red_origin[1]))

    blue_bottom = h - 2 - blue_dist
    blue_origin = _place_object(h, w, blue_patch, blue_bottom - blue_patch.shape[0] + 1, blocked)
    if blue_origin is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[0, :] = RED
    inp[h - 1, :] = BLUE
    for frame in frames:
        _paste(inp, _frame_patch(frame["ih"], frame["iw"]), frame["r0"], frame["c0"])
    _paste(inp, red_patch, red_origin[0], red_origin[1], transparent=BG)
    _paste(inp, blue_patch, blue_origin[0], blue_origin[1], transparent=BG)

    red_coords = _absolute_coords(red_patch, red_origin[0], red_origin[1])
    blue_coords = _absolute_coords(blue_patch, blue_origin[0], blue_origin[1])
    objects: list[dict[str, object]] = [
        {"color": RED, "patch": red_patch, "coords": red_coords, "distance": int(red_dist)},
        {"color": BLUE, "patch": blue_patch, "coords": blue_coords, "distance": int(blue_dist)},
    ]

    if not _semantic_guard(h, frames, objects):
        return None

    expected = _build_expected(inp, frames, objects)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid virtual-slide checkerboard instance")
