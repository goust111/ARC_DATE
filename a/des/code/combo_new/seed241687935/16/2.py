from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(10))
GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(pts: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_to_fg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return grid.copy()
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _trace_arm(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    corner: tuple[int, int],
    start: tuple[int, int],
) -> int | None:
    prev = corner
    cur = start
    length = 1
    while True:
        nxt = [p for p in adj[cur] if p != prev]
        if not nxt:
            return length
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
        length += 1


def _analyze_arrow(pts: list[tuple[int, int]]) -> dict[str, object] | None:
    if not 3 <= len(pts) <= 5:
        return None
    pt_set = set(pts)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pt_set:
        adj[(r, c)] = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in pt_set]
    degrees = sorted(len(v) for v in adj.values())
    if degrees.count(1) != 2 or any(d not in (1, 2) for d in degrees):
        return None

    corners = []
    for cell, neigh in adj.items():
        if len(neigh) != 2:
            continue
        (r1, c1), (r2, c2) = neigh
        if r1 != r2 and c1 != c2:
            corners.append(cell)
    if len(corners) != 1:
        return None

    corner = corners[0]
    n1, n2 = adj[corner]
    l1 = _trace_arm(adj, corner, n1)
    l2 = _trace_arm(adj, corner, n2)
    if l1 is None or l2 is None:
        return None
    if sorted([l1, l2]) != [1, len(pts) - 2]:
        return None

    directions = []
    if l1 == 1:
        directions.append((n1[0] - corner[0], n1[1] - corner[1]))
    if l2 == 1:
        cand = (n2[0] - corner[0], n2[1] - corner[1])
        if cand not in directions:
            directions.append(cand)
    if any(direction not in DIR4 for direction in directions):
        return None
    return {"cells": pt_set, "corner": corner, "directions": tuple(directions)}


def _analyze_frame(pts: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(pts) < 10:
        return None
    pt_set = set(pts)
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    if not pt_set <= border:
        return None

    missing = sorted(border - pt_set)
    if not (1 <= len(missing) <= 3):
        return None

    rows = {r for r, _ in missing}
    cols = {c for _, c in missing}
    side = None
    if len(cols) == 1 and next(iter(cols)) == c0:
        rs = sorted(r for r, _ in missing)
        if rs[0] <= r0 or rs[-1] >= r1 or rs != list(range(rs[0], rs[-1] + 1)):
            return None
        side = "left"
    elif len(cols) == 1 and next(iter(cols)) == c1:
        rs = sorted(r for r, _ in missing)
        if rs[0] <= r0 or rs[-1] >= r1 or rs != list(range(rs[0], rs[-1] + 1)):
            return None
        side = "right"
    elif len(rows) == 1 and next(iter(rows)) == r0:
        cs = sorted(c for _, c in missing)
        if cs[0] <= c0 or cs[-1] >= c1 or cs != list(range(cs[0], cs[-1] + 1)):
            return None
        side = "top"
    elif len(rows) == 1 and next(iter(rows)) == r1:
        cs = sorted(c for _, c in missing)
        if cs[0] <= c0 or cs[-1] >= c1 or cs != list(range(cs[0], cs[-1] + 1)):
            return None
        side = "bottom"
    else:
        return None

    return {"cells": pt_set, "bbox": (r0, c0, r1, c1), "gap_side": side, "gap_cells": set(missing)}


def _extract_roles(grid: np.ndarray, color: int, bg: int) -> tuple[dict[str, object], dict[str, object], set[tuple[int, int]]]:
    comps = _components(grid == color)
    frame_candidates = []
    arrow_candidates = []
    for comp in comps:
        frame_info = _analyze_frame(comp)
        if frame_info is not None:
            frame_candidates.append(frame_info)
            continue
        arrow_info = _analyze_arrow(comp)
        if arrow_info is not None:
            arrow_candidates.append(arrow_info)

    assert len(frame_candidates) == 1, "expected one gapped frame per color"
    assert len(arrow_candidates) == 1, "expected one arrow per color"

    frame_cells = frame_candidates[0]["cells"]
    arrow_cells = arrow_candidates[0]["cells"]
    fragments = [set(comp) for comp in comps if set(comp) != frame_cells and set(comp) != arrow_cells]
    assert len(fragments) == 1, "expected one movable fragment per color"
    return frame_candidates[0], arrow_candidates[0], fragments[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment through its matching frame gap until it rests against the opposite inner wall.

    Input structure:
    - The background is the dominant color.
    - Several distinct colors each form one 1-pixel-thick hollow rectangular frame whose border has a single contiguous gap on one side.
    - For each such color there is also one small 4-connected fragment outside the frame and one same-colored L-shaped arrow marker outside the frame.

    Rule:
    - For each color, identify the matching frame, fragment, and arrow.
    - Read the arrow direction from the arrow's 1-cell head arm, then translate the fragment only along that axis.
    - Move the fragment until its leading edge reaches the inner wall opposite the frame gap; erase the arrow and the fragment's old position, and keep the frame unchanged.

    Key insight:
    - The color groups the three objects, the frame gap determines which side is the entry side, and the arrow contributes only the translation axis; together they uniquely determine the fragment's final offset.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        frame, arrow, fragment = _extract_roles(g, color, bg)
        r0, c0, r1, c1 = frame["bbox"]
        gap_side = frame["gap_side"]
        expected_dir = {
            "left": (0, 1),
            "right": (0, -1),
            "top": (1, 0),
            "bottom": (-1, 0),
        }[gap_side]
        assert expected_dir in arrow["directions"], "arrow direction and frame gap must agree"
        dr, dc = expected_dir

        for r, c in arrow["cells"]:
            out[r, c] = bg
        for r, c in fragment:
            out[r, c] = bg

        if dc == 1:
            shift = (c1 - 1) - max(c for _, c in fragment)
        elif dc == -1:
            shift = min(c for _, c in fragment) - (c0 + 1)
        elif dr == 1:
            shift = (r1 - 1) - max(r for r, _ in fragment)
        else:
            shift = min(r for r, _ in fragment) - (r0 + 1)
        assert shift >= 0, "fragment must start outside the frame and move inward"

        for r, c in fragment:
            nr = r + dr * shift
            nc = c + dc * shift
            out[nr, nc] = color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


@lru_cache(maxsize=None)
def _fragment_masks(h: int, w: int) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    area = h * w
    for bits in range(1, 1 << area):
        mask = np.zeros((h, w), dtype=bool)
        for idx in range(area):
            if (bits >> idx) & 1:
                rr = idx // w
                cc = idx % w
                mask[rr, cc] = True
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _mask_connected(mask):
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        if _analyze_arrow(pts) is not None:
            continue
        masks.append(mask)
    return tuple(masks)


def _make_right_patch(color: int, bg: int) -> np.ndarray:
    frag_h = random.randint(1, 3)
    frag_w = random.randint(1, 3)
    masks = _fragment_masks(frag_h, frag_w)
    assert masks, "expected at least one usable fragment mask"
    frag_mask = random.choice(masks)

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    interior_w = frag_w + random.randint(2, 5)
    frame_h = frag_h + top_pad + bottom_pad + 2
    frame_w = interior_w + 2
    outer_gap = random.randint(2, 4)
    shaft_len = random.randint(2, 4)

    frame_top = 1
    frag_top = frame_top + 1 + top_pad
    frag_left = 3
    frame_left = frag_left + frag_w + outer_gap

    patch_h = max(frame_top + frame_h, frag_top + shaft_len) + 1
    patch_w = frame_left + frame_w + 1
    patch = np.full((patch_h, patch_w), bg, dtype=int)

    r0, c0 = frame_top, frame_left
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    patch[r0, c0 : c1 + 1] = color
    patch[r1, c0 : c1 + 1] = color
    patch[r0 : r1 + 1, c0] = color
    patch[r0 : r1 + 1, c1] = color
    patch[frag_top : frag_top + frag_h, c0] = bg

    frag_view = patch[frag_top : frag_top + frag_h, frag_left : frag_left + frag_w]
    frag_view[frag_mask] = color

    patch[frag_top : frag_top + shaft_len, 0] = color
    patch[frag_top, 1] = color
    return _crop_to_fg(patch, bg)


def _make_patch(color: int, bg: int) -> np.ndarray:
    patch = _make_right_patch(color, bg)
    rotations = random.randrange(4)
    return _crop_to_fg(np.rot90(patch, rotations), bg)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _assemble_grid(bg: int, patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    min_h = max(14, max_h + 2)
    min_w = max(14, max_w + 2)
    for _ in range(80):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        grid = np.full((h, w), bg, dtype=int)
        placed_rects: list[tuple[int, int, int, int]] = []
        placed = True
        for patch in random.sample(patches, len(patches)):
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, h - ph):
                for c0 in range(1, w - pw):
                    rect = (r0 - 1, c0 - 1, r0 + ph, c0 + pw)
                    if any(_rects_overlap(rect, other) for other in placed_rects):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                placed = False
                break
            r0, c0 = random.choice(candidates)
            mask = patch != bg
            target = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(target[mask] != bg):
                placed = False
                break
            target[mask] = patch[mask]
            placed_rects.append((r0 - 1, c0 - 1, r0 + ph, c0 + pw))
        if placed:
            return grid
    return None


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 2, 3])
    colors = [c for c in PALETTE if c != bg]
    n_objects = random.randint(1, 3)
    chosen = random.sample(colors, n_objects)
    patches = [_make_patch(color, bg) for color in chosen]
    grid = _assemble_grid(bg, patches)
    if grid is None:
        return None
    if _get_bg(grid) != bg:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
