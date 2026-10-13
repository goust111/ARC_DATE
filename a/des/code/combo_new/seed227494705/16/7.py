from collections import defaultdict, deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)
DIRS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
ORDER = ["up", "right", "down", "left"]


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _binary_components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _perimeter(r0, c0, r1, c1):
    cells = []
    for c in range(c0, c1 + 1):
        cells.extend([(r0, c), (r1, c)])
    for r in range(r0 + 1, r1):
        cells.extend([(r, c0), (r, c1)])
    return cells


def _frame_gap(grid, box):
    r0, c0, r1, c1 = box
    missing = []
    for r, c in _perimeter(r0, c0, r1, c1):
        if int(grid[r, c]) != MAROON:
            side = "up" if r == r0 else "down" if r == r1 else "left" if c == c0 else "right"
            missing.append((side, r, c))
    return missing[0] if len(missing) == 1 else None


def _portal_frames(grid):
    frames = []
    for cells in _binary_components(grid == MAROON):
        box = _bbox(cells)
        if box[2] - box[0] >= 3 and box[3] - box[1] >= 3:
            gap = _frame_gap(grid, box)
            if gap:
                r0, c0, r1, c1 = box
                inside = [(r, c, int(grid[r, c])) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(grid[r, c]) not in (BLACK, MAROON)]
                if inside:
                    frames.append({"box": box, "gap": gap, "marker": inside[0][2], "marker_cell": (inside[0][0], inside[0][1])})
    return frames


def _colored_components_without_markers(grid, marker_cells):
    mask = (grid != BLACK) & (grid != MAROON)
    for r, c in marker_cells:
        mask[r, c] = False
    return _binary_components(mask)


def _array_from_cells(grid, cells):
    r0, c0, r1, c1 = _bbox(cells)
    arr = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
    for r, c in cells:
        arr[r - r0, c - c0] = int(grid[r, c])
    return arr


def _paste_at_gap(out, arr, gap):
    side, gr, gc = gap
    ah, aw = arr.shape
    if side == "up":
        top, left = gr - ah, gc - aw // 2
    elif side == "down":
        top, left = gr + 1, gc - aw // 2
    elif side == "left":
        top, left = gr - ah // 2, gc - aw
    else:
        top, left = gr - ah // 2, gc + 1
    for r, c in np.argwhere(arr >= 0):
        rr, cc = top + int(r), left + int(c)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = int(arr[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """Copy a source sprite through each paired portal, rotating by gap direction.

    Input structure: Maroon hollow portal frames occur in same-marker-color pairs.
    Every frame has one side gap; exactly one portal in a pair has a multicolor
    source sprite adjacent to the outside of its gap.
    Rule: Pair portals by their interior marker color.  For the portal that has a
    source sprite, rotate that sprite clockwise by the direction difference from
    source gap to target gap, then paste it outside the target gap.  The original
    sprite and all portal frames remain.
    Key insight: The marker color pairs the portals, while the two gap directions
    jointly encode the rotation and target placement.
    """
    out = grid.copy()
    frames = _portal_frames(grid)
    marker_cells = [f["marker_cell"] for f in frames]
    sprites = _colored_components_without_markers(grid, marker_cells)
    by_marker = defaultdict(list)
    for f in frames:
        by_marker[f["marker"]].append(f)

    for group in by_marker.values():
        if len(group) != 2:
            continue
        source = target = source_cells = None
        for f in group:
            side, gr, gc = f["gap"]
            dr, dc = DIRS[side]
            probe = (gr + dr, gc + dc)
            for cells in sprites:
                if probe in cells or any(abs(r - probe[0]) + abs(c - probe[1]) <= 1 for r, c in cells):
                    source, source_cells = f, cells
                    break
            if source is not None:
                break
        if source is None:
            continue
        target = group[0] if group[1] is source else group[1]
        k = (ORDER.index(target["gap"][0]) - ORDER.index(source["gap"][0])) % 4
        arr = np.rot90(_array_from_cells(grid, source_cells), -k)
        _paste_at_gap(out, arr, target["gap"])
    return out


def _draw_frame(grid, r0, c0, side, marker):
    h = w = 5
    r1, c1 = r0 + h - 1, c0 + w - 1
    for r, c in _perimeter(r0, c0, r1, c1):
        grid[r, c] = MAROON
    if side in ("up", "down"):
        gr, gc = (r0 if side == "up" else r1), c0 + 2
    else:
        gr, gc = r0 + 2, (c0 if side == "left" else c1)
    grid[gr, gc] = BLACK
    grid[r0 + 2, c0 + 2] = marker
    return (side, gr, gc)


def _source_position(arr, gap):
    side, gr, gc = gap
    ah, aw = arr.shape
    if side == "up":
        return gr - ah, gc - aw // 2
    if side == "down":
        return gr + 1, gc - aw // 2
    if side == "left":
        return gr - ah // 2, gc - aw
    return gr - ah // 2, gc + 1


def generate_input() -> np.ndarray:
    grid = np.zeros((28, 28), dtype=int)
    sprite = np.array([[2, 3, -1], [-1, 4, 6], [7, -1, -1]], dtype=int)
    positions = [((4, 4), (4, 19)), ((18, 4), (18, 19))]
    markers = random.sample([2, 3, 4, 6, 8], random.randint(1, 2))
    for i, marker in enumerate(markers):
        src_pos, dst_pos = positions[i]
        src_side = random.choice(["right", "down", "up"])
        dst_side = random.choice(list(DIRS))
        g1 = _draw_frame(grid, src_pos[0], src_pos[1], src_side, marker)
        _draw_frame(grid, dst_pos[0], dst_pos[1], dst_side, marker)
        top, left = _source_position(sprite, g1)
        for r, c in np.argwhere(sprite >= 0):
            rr, cc = top + int(r), left + int(c)
            if 0 <= rr < 28 and 0 <= cc < 28:
                grid[rr, cc] = int(sprite[r, c])
    return grid
