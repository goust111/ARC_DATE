from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
FRAME = 5
SOURCE = 7

MAX_GENERATION_ATTEMPTS = 500
GRID_SIZE_RANGE = (22, 30)
SOURCE_HEIGHTS = (4, 5)
SOURCE_WIDTHS = (4, 5)
SOURCE_AREA_RANGE = (8, 15)
FRAME_COUNT_WEIGHTS = [(2, 5), (3, 4), (4, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(patch, np.ones((factor, factor), dtype=patch.dtype))


def _frame_from_component(grid: np.ndarray, comp: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = bbox_coords(comp)
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1
    if oh < 3 or ow < 3:
        return None
    if comp.shape[0] != 2 * oh + 2 * ow - 4:
        return None
    for r, c in comp:
        rr, cc = int(r), int(c)
        if rr not in (r0, r1) and cc not in (c0, c1):
            return None
        if int(grid[rr, cc]) != FRAME:
            return None
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0:
        return None
    if np.any((interior != BG) & (interior != RED)):
        return None
    red_pts = np.argwhere(interior == RED)
    if red_pts.shape[0] != 1:
        return None
    fr, fc = map(int, red_pts[0])
    return {
        "bbox": (r0, c0, r1, c1),
        "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "marker": (r0 + 1 + fr, c0 + 1 + fc),
        "inner_h": oh - 2,
        "inner_w": ow - 2,
    }


def _discover_scene(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == BG, "background must be black"

    frames = []
    for comp in component_coords(g == FRAME):
        frame = _frame_from_component(g, comp)
        assert frame is not None, "grey components must be hollow rectangular frames"
        frames.append(frame)
    assert 2 <= len(frames) <= 4, "expected 2-4 frames"

    comps = component_coords((g != bg) & (g != FRAME))
    source_candidates = []
    singletons = 0
    for comp in comps:
        rows, cols = comp[:, 0], comp[:, 1]
        vals = g[rows, cols]
        if comp.shape[0] == 1:
            assert int(vals[0]) == RED, "non-source singleton components must be red markers"
            singletons += 1
            continue
        source_candidates.append(comp)
    assert len(source_candidates) == 1, "expected exactly one multi-cell source component"
    assert singletons == len(frames), "each frame must contribute one isolated red marker"

    source_comp = source_candidates[0]
    r0, c0, r1, c1 = bbox_coords(source_comp)
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert 4 <= patch.shape[0] <= 5 and 4 <= patch.shape[1] <= 5, "source bbox must be 4x4 to 5x5"

    red_pts = np.argwhere(patch == RED)
    assert red_pts.shape[0] == 1, "source must contain exactly one red marker"
    assert np.any(patch == SOURCE), "source must contain orange pixels"
    assert np.all((patch == BG) | (patch == RED) | (patch == SOURCE)), "unexpected color inside source bbox"
    mr, mc = map(int, red_pts[0])

    for frame in frames:
        ih, iw = int(frame["inner_h"]), int(frame["inner_w"])
        sh, sw = patch.shape
        assert ih % sh == 0 and iw % sw == 0, "frame interior must be an integer multiple of source size"
        k_h = ih // sh
        k_w = iw // sw
        assert k_h == k_w and k_h in (2, 3), "frame scale must be 2 or 3"
        frame["k"] = k_h

    return patch, (r0, c0, r1, c1), (mr, mc), frames


def _render(grid: np.ndarray, patch: np.ndarray, source_bbox: tuple[int, int, int, int], marker_rc: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = source_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    mr, mc = marker_rc

    for frame in frames:
        k = int(frame["k"])
        scaled = scale_patch(patch, k)
        fr, fc = frame["marker"]
        place_r0 = fr - mr * k
        place_c0 = fc - mc * k
        ir0, ic0, ir1, ic1 = frame["interior_bbox"]
        for rr, cc in np.argwhere(scaled != BG):
            tr = int(place_r0 + rr)
            tc = int(place_c0 + cc)
            assert ir0 <= tr <= ir1 and ic0 <= tc <= ic1, "scaled motif must fit fully inside the frame interior"
            if out[tr, tc] == BG:
                out[tr, tc] = int(scaled[rr, cc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the top-left source motif to each frame's interior size and stamp it by the shared red marker.

    Input structure:
    - The grid has black background, one orange source motif with exactly one red pixel, and 2-4 grey hollow rectangular frames.
    - The source motif is a single 4-connected irregular component with a 4x4 to 5x5 tight bbox.
    - Each frame interior contains exactly one red pixel on otherwise empty background.

    Rule:
    - Find the source bbox and the red marker offset inside it.
    - For each frame, compute the integer scale factor from the ratio between the frame's interior size and the source bbox size.
    - Nearest-neighbor scale the full source bbox by that factor, align the top-left of the scaled red block to the frame's red pixel, and stamp only onto background cells inside the frame interior.
    - Keep frame borders and pre-existing red marker pixels unchanged, then erase the original source bbox to background.

    Key insight:
    - The frame interior dimensions encode the scale factor, while the shared red marker fixes the placement of the scaled copy.
    """
    patch, source_bbox, marker_rc, frames = _discover_scene(np.asarray(grid, dtype=int))
    return _render(np.asarray(grid, dtype=int), patch, source_bbox, marker_rc, frames)


def _random_connected_shape(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]] | None:
    area_hi = min(SOURCE_AREA_RANGE[1], h * w - 2)
    if area_hi < SOURCE_AREA_RANGE[0]:
        return None
    area = rng.randint(SOURCE_AREA_RANGE[0], area_hi)
    red_r = rng.randint(1, h - 2)
    red_c = rng.randint(1, w - 2)
    for _ in range(200):
        cells = {(red_r, red_c)}
        frontier = set()
        for nr, nc in neighbors4(red_r, red_c):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < area and frontier:
            pick = rng.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            pr, pc = pick
            nbrs = list(neighbors4(pr, pc))
            rng.shuffle(nbrs)
            for nr, nc in nbrs:
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows != set(range(h)) or cols != set(range(w)):
            continue
        if len(cells) > h * w - 2:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in cells:
            patch[r, c] = SOURCE
        patch[red_r, red_c] = RED
        if np.sum(patch == BG) < 2:
            continue
        return patch, (red_r, red_c)
    return None


def _choose_frame_count_and_scales(source_shape: tuple[int, int], rng: random.Random) -> tuple[int, list[int]]:
    h, w = source_shape
    choices: list[tuple[int, list[int]]] = [
        (2, [2, 3]),
        (3, [2, 2, 3]),
        (3, [2, 3, 3]),
    ]
    if max(h, w) == 4:
        choices.append((4, [2, 2, 2, 3]))
    weighted = []
    for n, ks in choices:
        weight = dict(FRAME_COUNT_WEIGHTS).get(n, 1)
        weighted.extend([(n, ks)] * weight)
    n, ks = rng.choice(weighted)
    rng.shuffle(ks)
    return n, ks


def _place_frames(grid: np.ndarray, blocked: np.ndarray, frames: list[dict], rng: random.Random) -> bool:
    h, w = grid.shape
    for frame in sorted(frames, key=lambda x: -(x["outer_h"] * x["outer_w"])):
        oh = int(frame["outer_h"])
        ow = int(frame["outer_w"])
        candidates = []
        for r0 in range(1, h - oh):
            for c0 in range(1, w - ow):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + oh + 1)
                cc1 = min(w, c0 + ow + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        oh, ow = int(frame["outer_h"]), int(frame["outer_w"])
        grid[r0, c0 : c0 + ow] = FRAME
        grid[r0 + oh - 1, c0 : c0 + ow] = FRAME
        grid[r0 : r0 + oh, c0] = FRAME
        grid[r0 : r0 + oh, c0 + ow - 1] = FRAME
        marker_r = r0 + 1 + int(frame["marker_offset"][0])
        marker_c = c0 + 1 + int(frame["marker_offset"][1])
        grid[marker_r, marker_c] = RED
        frame["bbox"] = (r0, c0, r0 + oh - 1, c0 + ow - 1)
        frame["interior_bbox"] = (r0 + 1, c0 + 1, r0 + oh - 2, c0 + ow - 2)
        frame["marker"] = (marker_r, marker_c)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + oh + 1)
        cc1 = min(w, c0 + ow + 1)
        blocked[rr0:rr1, cc0:cc1] = True
    return True


def _validate_generated_instance(inp: np.ndarray, patch: np.ndarray, source_bbox: tuple[int, int, int, int], marker_rc: tuple[int, int], frames: list[dict], expected: np.ndarray) -> bool:
    try:
        found_patch, found_bbox, found_marker, found_frames = _discover_scene(inp)
    except Exception:
        return False
    if not np.array_equal(found_patch, patch):
        return False
    if found_bbox != source_bbox or found_marker != marker_rc:
        return False
    if sorted((f["bbox"], f["marker"], f["k"]) for f in found_frames) != sorted((f["bbox"], f["marker"], f["k"]) for f in frames):
        return False
    if {int(f["k"]) for f in frames} != {2, 3}:
        return False
    if np.array_equal(inp, expected):
        return False
    r0, c0, r1, c1 = source_bbox
    if np.count_nonzero(expected[r0 : r1 + 1, c0 : c1 + 1]) != 0:
        return False
    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["interior_bbox"]
        interior_in = inp[ir0 : ir1 + 1, ic0 : ic1 + 1]
        interior_out = expected[ir0 : ir1 + 1, ic0 : ic1 + 1]
        if np.sum(interior_in == RED) != 1:
            return False
        if np.array_equal(interior_in, interior_out):
            return False
        if np.sum(interior_out == SOURCE) == 0:
            return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    src_h = rng.choice(SOURCE_HEIGHTS)
    src_w = rng.choice(SOURCE_WIDTHS)
    built = _random_connected_shape(src_h, src_w, rng)
    if built is None:
        return None
    patch, marker_rc = built
    mr, mc = marker_rc

    _n_frames, ks = _choose_frame_count_and_scales((src_h, src_w), rng)
    frames = []
    for k in ks:
        frames.append(
            {
                "k": k,
                "outer_h": src_h * k + 2,
                "outer_w": src_w * k + 2,
                "marker_offset": (mr * k, mc * k),
            }
        )

    load = sum((f["outer_h"] + 2) * (f["outer_w"] + 2) for f in frames) + (src_h + 2) * (src_w + 2)
    side_lo = max(
        GRID_SIZE_RANGE[0],
        max(f["outer_h"] for f in frames) + 3,
        max(f["outer_w"] for f in frames) + 3,
        int(np.ceil(np.sqrt(load))),
    )
    if side_lo > GRID_SIZE_RANGE[1]:
        return None
    h = rng.randint(side_lo, GRID_SIZE_RANGE[1])
    w = rng.randint(side_lo, GRID_SIZE_RANGE[1])

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    src_r0 = rng.randint(0, min(2, h - src_h))
    src_c0 = rng.randint(0, min(2, w - src_w))
    grid[src_r0 : src_r0 + src_h, src_c0 : src_c0 + src_w] = patch
    rr0 = max(0, src_r0 - 1)
    cc0 = max(0, src_c0 - 1)
    rr1 = min(h, src_r0 + src_h + 1)
    cc1 = min(w, src_c0 + src_w + 1)
    blocked[rr0:rr1, cc0:cc1] = True

    if not _place_frames(grid, blocked, frames, rng):
        return None

    source_bbox = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
    expected = _render(grid, patch, source_bbox, marker_rc, frames)
    if not _validate_generated_instance(grid, patch, source_bbox, marker_rc, frames, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
