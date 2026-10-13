from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FILL = 3
INPUT_COLORS = (1, 2, 4, 6)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500
GEN_GRID_ROWS = (16, 24)
GEN_GRID_COLS = (16, 24)
GEN_RECT_COUNT = (3, 4)
GEN_HEIGHTS = (3, 5)
GEN_WIDTHS = (2, 4)
GEN_GAPS = (1, 3)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rects: list[dict] = []
    for coords in _component_coords(g != BG):
        rr = coords[:, 0]
        cc = coords[:, 1]
        r0 = int(rr.min())
        r1 = int(rr.max())
        c0 = int(cc.min())
        c1 = int(cc.max())
        color = int(g[r0, c0])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "every foreground object must be a solid rectangle"
        rects.append(
            {
                "top": r0,
                "left": c0,
                "bottom": r1,
                "right": c1,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "color": color,
            }
        )
    rects.sort(key=lambda item: (item["left"], item["top"]))
    return rects


def _stack_patch(rects: list[dict]) -> tuple[np.ndarray, list[dict]]:
    stack_h = sum(rect["h"] for rect in rects) - len(rects) + 1
    stack_w = sum(rect["w"] for rect in rects) - len(rects) + 1
    patch = np.zeros((stack_h, stack_w), dtype=int)
    placed: list[dict] = []
    r0 = 0
    c0 = 0
    for i, rect in enumerate(rects):
        h = int(rect["h"])
        w = int(rect["w"])
        color = int(rect["color"])
        patch[r0 : r0 + h, c0 : c0 + w] = color
        placed.append({"r0": r0, "c0": c0, "h": h, "w": w, "color": color})
        if i + 1 < len(rects):
            r0 += h - 1
            c0 += w - 1
    return patch, placed


def _mirror_patch(src: np.ndarray) -> np.ndarray:
    """
    Reflect the stacked raster across its bbox main diagonal by swapping
    normalized row/column coordinates and sampling back onto the same grid.
    """
    h, w = map(int, src.shape)
    out = np.zeros_like(src)
    for r in range(h):
        for c in range(w):
            src_r = min(h - 1, int((c + 0.5) * h / w))
            src_c = min(w - 1, int((r + 0.5) * w / h))
            out[r, c] = int(src[src_r, src_c])
    return out


def _fill_enclosed_gap(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    out = grid.copy()
    gap = (grid == BG) & ~outside
    out[gap] = FILL
    return out


def _solve_local(rects: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack, _placed = _stack_patch(rects)
    mirrored = _mirror_patch(stack)
    combined = stack.copy()
    mirror_only = (combined == BG) & (mirrored != BG)
    combined[mirror_only] = mirrored[mirror_only]
    final = _fill_enclosed_gap(combined)
    return final, stack, mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diagonally restack bottom rectangles, reflect the stack across its emergent bbox diagonal, and fill the wedge gap in green.

    Input structure:
    - The background is black.
    - Foreground consists of 3 or 4 different-colored solid rectangles touching the bottom edge.
    - The rectangles are separated horizontally in the input and are identified only by their left-to-right order.

    Rule:
    - Sort the rectangles by left edge, clear the canvas, and rebuild them as a diagonal corner-overlap chain starting at the top-left.
    - Reflect that stacked chain across the main diagonal of its own bounding box, keeping original stack pixels when the mirror overlaps them.
    - Fill any background cells enclosed between the original chain and its mirror with green; all remaining cells stay black.

    Key insight:
    - The mirror axis is not drawn in the input: it appears only after the rectangles are reordered into the diagonal stack, so the stack geometry must be built before the reflection can be inferred.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _extract_rectangles(g)
    assert GEN_RECT_COUNT[0] <= len(rects) <= GEN_RECT_COUNT[1], "expected 3 or 4 rectangles"
    assert len({rect["color"] for rect in rects}) == len(rects), "rectangle colors must be distinct"
    assert all(rect["bottom"] == g.shape[0] - 1 for rect in rects), "every rectangle must touch the bottom edge"
    assert FILL not in {rect["color"] for rect in rects}, "green is reserved for the output gap fill"

    local, _stack, _mirrored = _solve_local(rects)
    out = np.zeros_like(g)
    lh, lw = local.shape
    assert lh <= g.shape[0] and lw <= g.shape[1], "restacked output must fit inside the canvas"
    out[:lh, :lw] = local
    return out


def _render_input(rects: list[dict], shape: tuple[int, int], gaps: list[int], left_pad: int) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    c = int(left_pad)
    for i, rect in enumerate(rects):
        rh = int(rect["h"])
        rw = int(rect["w"])
        r0 = h - rh
        grid[r0:h, c : c + rw] = int(rect["color"])
        if i + 1 < len(rects):
            c += rw + int(gaps[i])
    return grid


def _sample_rectangles() -> list[dict] | None:
    n = random.randint(*GEN_RECT_COUNT)
    branch = random.random()
    require_square = branch < 0.30
    require_nonsquare = branch < 0.85 and not require_square

    dims: list[tuple[int, int]] = []
    sum_h = 0
    sum_w = 0
    for _ in range(n):
        rh = random.randint(*GEN_HEIGHTS)
        rw = random.randint(*GEN_WIDTHS)
        dims.append((rh, rw))
        sum_h += rh
        sum_w += rw

    if require_square and sum_h != sum_w:
        return None
    if require_nonsquare and sum_h == sum_w:
        return None
    if len({rh for rh, _rw in dims}) == 1 and len({rw for _rh, rw in dims}) == 1:
        return None

    colors = random.sample(INPUT_COLORS, n)
    rects = [{"h": rh, "w": rw, "color": color} for (rh, rw), color in zip(dims, colors)]
    return rects


def _validate_case(inp: np.ndarray) -> bool:
    try:
        rects = _extract_rectangles(inp)
        local, stack, mirrored = _solve_local(rects)
    except Exception:
        return False

    if local.shape[0] > inp.shape[0] or local.shape[1] > inp.shape[1]:
        return False
    if np.count_nonzero(local == FILL) < 2:
        return False
    if np.array_equal(stack, mirrored):
        return False
    if np.count_nonzero((stack == BG) & (mirrored != BG)) < 3:
        return False

    alt_rects = list(reversed(rects))
    alt_local, _alt_stack, _alt_mirrored = _solve_local(alt_rects)
    if np.array_equal(local, alt_local):
        return False

    if stack.shape[0] != stack.shape[1]:
        pad = max(stack.shape)
        trivial = np.zeros((pad, pad), dtype=int)
        trivial[: stack.shape[0], : stack.shape[1]] = stack
        trivial = trivial.T[: stack.shape[0], : stack.shape[1]]
        if np.array_equal(mirrored, trivial):
            return False

    out = main(inp)
    probe = np.zeros_like(inp)
    probe[: local.shape[0], : local.shape[1]] = local
    return np.array_equal(out, probe)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rects = _sample_rectangles()
        if rects is None:
            continue

        stack_h = sum(rect["h"] for rect in rects) - len(rects) + 1
        stack_w = sum(rect["w"] for rect in rects) - len(rects) + 1
        gaps = [random.randint(*GEN_GAPS) for _ in range(len(rects) - 1)]
        input_w = sum(rect["w"] for rect in rects) + sum(gaps)

        min_rows = max(GEN_GRID_ROWS[0], stack_h)
        min_cols = max(GEN_GRID_COLS[0], stack_w, input_w)
        if min_rows > GEN_GRID_ROWS[1] or min_cols > GEN_GRID_COLS[1]:
            continue

        grid_h = random.randint(min_rows, GEN_GRID_ROWS[1])
        grid_w = random.randint(min_cols, GEN_GRID_COLS[1])
        left_pad = random.randint(0, grid_w - input_w)
        inp = _render_input(rects, (grid_h, grid_w), gaps, left_pad)

        if _validate_case(inp):
            return inp

    raise RuntimeError("failed to generate a valid diagonal-stack mirror sample")
