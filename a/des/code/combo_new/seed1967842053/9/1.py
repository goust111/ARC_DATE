from collections import Counter
import random

import numpy as np


BG = 0
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _side_color(values):
    vals = [int(v) for v in values if int(v) != BG]
    if not vals:
        return None
    counts = Counter(vals)
    return max(counts.items(), key=lambda item: item[1])[0]


def _frame_info(grid, cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub[0, :] == BG) or np.any(sub[-1, :] == BG):
        return None
    if np.any(sub[:, 0] == BG) or np.any(sub[:, -1] == BG):
        return None

    sides = [
        _side_color(sub[0, 1:-1]),
        _side_color(sub[-1, 1:-1]),
        _side_color(sub[1:-1, 0]),
        _side_color(sub[1:-1, -1]),
    ]
    if any(color is None for color in sides):
        return None
    counts = Counter(sides)
    odd_colors = [color for color, count in counts.items() if count == 1]
    common_colors = [color for color, count in counts.items() if count == 3]
    if len(odd_colors) != 1 or len(common_colors) != 1:
        return None

    interior_zeros = sub[1:-1, 1:-1] == BG
    if not np.any(interior_zeros):
        return None
    holes = _components(interior_zeros)
    if len(holes) != 1:
        return None

    gap = [(r0 + 1 + r, c0 + 1 + c) for r, c in holes[0]]
    return {"bbox": (r0, r1, c0, c1), "gap": gap, "target": odd_colors[0]}


def _split_objects(grid):
    frames = []
    pieces = []
    for comp in _components(grid != BG):
        info = _frame_info(grid, comp)
        if info is None:
            pieces.append(comp)
        else:
            frames.append(info)
    return frames, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each framed gap with the matching loose shape recolored by the frame's odd border side.

    Input structure:
    The background is black.  The grid contains one or more filled rectangular frames whose outer border is intact; one of the four sides has a different color from the other three.  Inside each frame is one connected black irregular gap, and elsewhere in the grid are small single-color loose fragments whose translated shapes match those gaps.

    Rule:
    Find every rectangular frame, identify the unique side color that appears on only one border side, and identify the connected black gap inside the frame.  Remove the loose fragments from the grid.  For each frame, locate the loose fragment with the same normalized cell shape as the gap, translate that shape into the gap, and color all filled gap cells with the frame's unique side color while leaving the original frame unchanged.

    Key insight:
    Shape matching supplies the placement, but the final fill color comes from the visually odd frame side rather than from the fragment's original color.
    """
    arr = np.array(grid, copy=True)
    out = arr.copy()
    frames, pieces = _split_objects(arr)

    pieces_by_shape = {}
    for comp in pieces:
        pieces_by_shape.setdefault(_normalize(comp), []).append(comp)
        for r, c in comp:
            out[r, c] = BG

    used = set()
    for frame in frames:
        shape = _normalize(frame["gap"])
        match = None
        for comp in pieces_by_shape.get(shape, []):
            key = tuple(sorted(comp))
            if key not in used:
                used.add(key)
                match = comp
                break
        if match is None:
            continue
        for r, c in frame["gap"]:
            out[r, c] = frame["target"]
    return out


def _is_filled_rectangle(shape):
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return len(shape) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _random_shape(max_h, max_w):
    for _ in range(200):
        size = random.randint(4, min(9, max_h * max_w - 1))
        cells = {(random.randrange(max_h), random.randrange(max_w))}
        while len(cells) < size:
            cr, cc = random.choice(tuple(cells))
            random.shuffle_dirs = list(NEIGHBORS)
            random.shuffle(random.shuffle_dirs)
            for dr, dc in random.shuffle_dirs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < max_h and 0 <= nc < max_w:
                    cells.add((nr, nc))
                    break
        shape = _normalize(cells)
        if not _is_filled_rectangle(shape):
            return shape
    return ((0, 0), (0, 1), (1, 0), (2, 0))


def _shape_size(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _reserve_rect(reserved, r0, c0, h, w, margin=1):
    rows, cols = reserved.shape
    rr0, cc0 = max(0, r0 - margin), max(0, c0 - margin)
    rr1, cc1 = min(rows, r0 + h + margin), min(cols, c0 + w + margin)
    reserved[rr0:rr1, cc0:cc1] = True


def _can_place_rect(reserved, r0, c0, h, w, margin=1):
    rows, cols = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    rr0, cc0 = max(0, r0 - margin), max(0, c0 - margin)
    rr1, cc1 = min(rows, r0 + h + margin), min(cols, c0 + w + margin)
    return not np.any(reserved[rr0:rr1, cc0:cc1])


def generate_input() -> np.ndarray:
    for _ in range(300):
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)
        grid = np.zeros((rows, cols), dtype=int)
        reserved = np.zeros((rows, cols), dtype=bool)
        colors = list(range(1, 10))
        base_color = random.choice(colors)
        count = random.randint(1, 3)
        frames = []
        shapes_seen = set()

        ok = True
        for _frame_index in range(count):
            placed = False
            for _attempt in range(300):
                h = random.randint(6, min(10, rows - 2))
                w = random.randint(6, min(10, cols - 2))
                r0 = random.randrange(1, rows - h) if rows - h > 1 else 0
                c0 = random.randrange(1, cols - w) if cols - w > 1 else 0
                if not _can_place_rect(reserved, r0, c0, h, w, margin=1):
                    continue

                shape = _random_shape(h - 2, w - 2)
                shape_h, shape_w = _shape_size(shape)
                if shape in shapes_seen or shape_h > h - 2 or shape_w > w - 2:
                    continue
                hole_r = random.randint(0, h - 2 - shape_h)
                hole_c = random.randint(0, w - 2 - shape_w)
                target_choices = [color for color in colors if color != base_color]
                target = random.choice(target_choices)
                side = random.randrange(4)

                grid[r0 : r0 + h, c0 : c0 + w] = base_color
                if side == 0:
                    grid[r0, c0 : c0 + w] = target
                elif side == 1:
                    grid[r0 + h - 1, c0 : c0 + w] = target
                elif side == 2:
                    grid[r0 : r0 + h, c0] = target
                else:
                    grid[r0 : r0 + h, c0 + w - 1] = target

                gap = []
                for sr, sc in shape:
                    gr, gc = r0 + 1 + hole_r + sr, c0 + 1 + hole_c + sc
                    grid[gr, gc] = BG
                    gap.append((gr, gc))
                _reserve_rect(reserved, r0, c0, h, w, margin=1)
                frames.append({"shape": shape, "target": target, "gap": gap})
                shapes_seen.add(shape)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        frame_order = list(frames)
        random.shuffle(frame_order)
        for frame in frame_order:
            shape = frame["shape"]
            shape_h, shape_w = _shape_size(shape)
            placed = False
            for _attempt in range(500):
                r0 = random.randrange(0, rows - shape_h + 1)
                c0 = random.randrange(0, cols - shape_w + 1)
                if not _can_place_rect(reserved, r0, c0, shape_h, shape_w, margin=1):
                    continue
                piece_color = random.choice([color for color in colors if color != frame["target"]])
                for sr, sc in shape:
                    grid[r0 + sr, c0 + sc] = piece_color
                _reserve_rect(reserved, r0, c0, shape_h, shape_w, margin=1)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid

    grid = np.zeros((18, 18), dtype=int)
    grid[2:9, 2:9] = 5
    grid[2, 2:9] = 2
    for r, c in [(4, 4), (4, 5), (5, 4), (6, 4)]:
        grid[r, c] = BG
    for r, c in [(11, 12), (11, 13), (12, 12), (13, 12)]:
        grid[r, c] = 7
    return grid
