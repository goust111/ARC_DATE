import random

import numpy as np


DIRS = ("up", "down", "left", "right")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _detect_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []
    if np.all(grid[:, 0] != bg):
        candidates.append(("left", grid[:, 0].copy()))
    if np.all(grid[:, w - 1] != bg):
        candidates.append(("right", grid[:, w - 1].copy()))
    if np.all(grid[0, :] != bg):
        candidates.append(("top", grid[0, :].copy()))
    if np.all(grid[h - 1, :] != bg):
        candidates.append(("bottom", grid[h - 1, :].copy()))
    assert candidates, "expected a full non-background legend strip on a grid edge"

    side, colors = max(candidates, key=lambda item: (len(set(map(int, item[1]))), len(item[1])))
    mask = np.zeros_like(grid, dtype=bool)
    if side == "left":
        mask[:, 0] = True
    elif side == "right":
        mask[:, -1] = True
    elif side == "top":
        mask[0, :] = True
    else:
        mask[-1, :] = True
    return side, np.asarray(colors, dtype=int), mask


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _marker_gap_side(mask: np.ndarray) -> str | None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    if h < 3 or w < 3:
        return None
    if np.any(sub[1:-1, 1:-1]):
        return None
    if not (sub[0, 0] and sub[0, -1] and sub[-1, 0] and sub[-1, -1]):
        return None

    misses = {
        "up": int(np.sum(~sub[0, 1:-1])),
        "down": int(np.sum(~sub[-1, 1:-1])),
        "left": int(np.sum(~sub[1:-1, 0])),
        "right": int(np.sum(~sub[1:-1, -1])),
    }
    if sorted(misses.values()) != [0, 0, 0, 1]:
        return None

    if misses["up"] == 0 and not np.all(sub[0, 1:-1]):
        return None
    if misses["down"] == 0 and not np.all(sub[-1, 1:-1]):
        return None
    if misses["left"] == 0 and not np.all(sub[1:-1, 0]):
        return None
    if misses["right"] == 0 and not np.all(sub[1:-1, -1]):
        return None

    for side, count in misses.items():
        if count == 1:
            return side
    return None


def _find_body_and_direction(comp_mask: np.ndarray):
    pts = np.argwhere(comp_mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    sub = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    prefix = _prefix_sum(sub)

    candidates = []
    for rr0 in range(h):
        for rr1 in range(rr0, h):
            for cc0 in range(w):
                for cc1 in range(cc0, w):
                    area = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
                    if _rect_sum(prefix, rr0, cc0, rr1, cc1) == area:
                        candidates.append((area, rr0, cc0, rr1, cc1))
    candidates.sort(reverse=True)

    for _area, rr0, cc0, rr1, cc1 in candidates:
        remainder = sub.copy()
        remainder[rr0 : rr1 + 1, cc0 : cc1 + 1] = False
        side = _marker_gap_side(remainder)
        if side is not None:
            return (int(r0 + rr0), int(c0 + cc0), int(r0 + rr1), int(c0 + cc1)), side

    raise AssertionError("failed to recover rectangle body and marker direction")


def _projection_length(grid: np.ndarray, bg: int, rect: tuple[int, int, int, int], direction: str) -> int:
    r0, c0, r1, c1 = rect
    if direction == "up":
        length = 0
        for r in range(r0 - 1, -1, -1):
            if np.all(grid[r, c0 : c1 + 1] == bg):
                length += 1
            else:
                break
        return length
    if direction == "down":
        length = 0
        for r in range(r1 + 1, grid.shape[0]):
            if np.all(grid[r, c0 : c1 + 1] == bg):
                length += 1
            else:
                break
        return length
    if direction == "left":
        length = 0
        for c in range(c0 - 1, -1, -1):
            if np.all(grid[r0 : r1 + 1, c] == bg):
                length += 1
            else:
                break
        return length

    length = 0
    for c in range(c1 + 1, grid.shape[1]):
        if np.all(grid[r0 : r1 + 1, c] == bg):
            length += 1
        else:
            break
    return length


def _paint_projection(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    direction: str,
    length: int,
    color: int,
) -> None:
    if length <= 0:
        return
    r0, c0, r1, c1 = rect
    if direction == "up":
        out[r0 - length : r0, c0 : c1 + 1] = color
    elif direction == "down":
        out[r1 + 1 : r1 + 1 + length, c0 : c1 + 1] = color
    elif direction == "left":
        out[r0 : r1 + 1, c0 - length : c0] = color
    else:
        out[r0 : r1 + 1, c1 + 1 : c1 + 1 + length] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the marker's border-gap side as the projection direction, then use the
    projection length as an index into the edge legend to choose the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    _legend_side, legend, legend_mask = _detect_legend(g, bg)
    object_mask = (g != bg) & ~legend_mask
    out = g.copy()

    for comp in _components(object_mask):
        comp_mask = np.zeros_like(g, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        rect, direction = _find_body_and_direction(comp_mask)
        length = _projection_length(g, bg, rect, direction)
        if length <= 0:
            continue
        assert length <= len(legend), "projection length must fit inside the legend strip"
        fill_color = int(legend[length - 1])
        _paint_projection(out, rect, direction, length, fill_color)

    return out


GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (12, 17)
OBJECT_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (3, 5)
RECT_W_RANGE = (3, 5)
GEN_RETRIES = 400


def _legend_sequence(length: int) -> np.ndarray:
    palette = random.sample(range(1, 10), random.randint(3, 6))
    seq: list[int] = []
    while len(seq) < length:
        block = palette[:]
        random.shuffle(block)
        if seq and len(block) > 1 and block[0] == seq[-1]:
            block[0], block[1] = block[1], block[0]
        seq.extend(block)
    return np.asarray(seq[:length], dtype=int)


def _boundary_ring_cells(top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(left, left + w):
        cells.add((top, c))
        cells.add((top + h - 1, c))
    for r in range(top + 1, top + h - 1):
        cells.add((r, left))
        cells.add((r, left + w - 1))
    return cells


def _component_cells(
    rect_top: int,
    rect_left: int,
    rect_h: int,
    rect_w: int,
    attach_side: str,
    gap_side: str,
    gap_pos: int,
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
    rect_cells = {
        (r, c)
        for r in range(rect_top, rect_top + rect_h)
        for c in range(rect_left, rect_left + rect_w)
    }
    if attach_side == "left":
        tag_top = rect_top - 1
        tag_left = rect_left - 3
        tag_h, tag_w = rect_h + 2, 3
    elif attach_side == "right":
        tag_top = rect_top - 1
        tag_left = rect_left + rect_w
        tag_h, tag_w = rect_h + 2, 3
    elif attach_side == "up":
        tag_top = rect_top - 3
        tag_left = rect_left - 1
        tag_h, tag_w = 3, rect_w + 2
    else:
        tag_top = rect_top + rect_h
        tag_left = rect_left - 1
        tag_h, tag_w = 3, rect_w + 2

    tag_cells = _boundary_ring_cells(tag_top, tag_left, tag_h, tag_w)
    if gap_side == "up":
        gap = (tag_top, tag_left + gap_pos)
    elif gap_side == "down":
        gap = (tag_top + tag_h - 1, tag_left + gap_pos)
    elif gap_side == "left":
        gap = (tag_top + gap_pos, tag_left)
    else:
        gap = (tag_top + gap_pos, tag_left + tag_w - 1)
    tag_cells.remove(gap)

    comp = rect_cells | tag_cells
    bbox = (
        min(r for r, _ in comp),
        min(c for _, c in comp),
        max(r for r, _ in comp),
        max(c for _, c in comp),
    )
    return comp, bbox


def _projection_cells(rect: tuple[int, int, int, int], direction: str, grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    if direction == "up":
        for r in range(r0 - 1, -1, -1):
            seg = [(r, c) for c in range(c0, c1 + 1)]
            if any(grid[rr, cc] != bg for rr, cc in seg):
                break
            cells.extend(seg)
    elif direction == "down":
        for r in range(r1 + 1, grid.shape[0]):
            seg = [(r, c) for c in range(c0, c1 + 1)]
            if any(grid[rr, cc] != bg for rr, cc in seg):
                break
            cells.extend(seg)
    elif direction == "left":
        for c in range(c0 - 1, -1, -1):
            seg = [(r, c) for r in range(r0, r1 + 1)]
            if any(grid[rr, cc] != bg for rr, cc in seg):
                break
            cells.extend(seg)
    else:
        for c in range(c1 + 1, grid.shape[1]):
            seg = [(r, c) for r in range(r0, r1 + 1)]
            if any(grid[rr, cc] != bg for rr, cc in seg):
                break
            cells.extend(seg)
    return cells


def _mark_reserved(mask: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]], pad: int) -> None:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        mask[r0:r1, c0:c1] = True


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    rect_h: int,
    rect_w: int,
    direction: str,
    attach_side: str,
) -> bool:
    bg = 0
    if attach_side in ("left", "right"):
        tag_h, tag_w = rect_h + 2, 3
    else:
        tag_h, tag_w = 3, rect_w + 2

    if direction in ("up", "down"):
        gap_pos = random.randint(1, tag_w - 2)
    else:
        gap_pos = random.randint(1, tag_h - 2)

    candidates = []
    h, w = grid.shape
    for rect_top in range(h):
        for rect_left in range(w):
            comp, bbox = _component_cells(
                rect_top, rect_left, rect_h, rect_w, attach_side, direction, gap_pos
            )
            br0, bc0, br1, bc1 = bbox
            if br0 < 0 or bc0 < 0 or br1 >= h or bc1 >= w:
                continue
            if any(reserved[r, c] or grid[r, c] != bg for r, c in comp):
                continue
            rect = (rect_top, rect_left, rect_top + rect_h - 1, rect_left + rect_w - 1)
            band = _projection_cells(rect, direction, grid, bg)
            length = len(band) // (rect_w if direction in ("up", "down") else rect_h)
            if length <= 0:
                continue
            if any(reserved[r, c] for r, c in band):
                continue
            candidates.append((rect, comp, band))

    if not candidates:
        return False

    rect, comp, band = random.choice(candidates)
    for r, c in comp:
        grid[r, c] = color

    _mark_reserved(reserved, comp, pad=1)
    _mark_reserved(reserved, band, pad=1)
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), bg, dtype=int)

    legend_side = random.choice(["left", "right"])
    legend = _legend_sequence(h)
    if legend_side == "left":
        grid[:, 0] = legend
    else:
        grid[:, -1] = legend

    reserved = np.zeros((h, w), dtype=bool)
    if legend_side == "left":
        reserved[:, 0] = True
    else:
        reserved[:, -1] = True

    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    object_pool = [c for c in range(1, 10) if c not in set(map(int, legend))]
    if len(object_pool) < n_objects:
        object_pool = list(range(1, 10))
    random.shuffle(object_pool)
    colors = object_pool[:n_objects]

    directions = [random.choice(DIRS) for _ in range(n_objects)]
    if n_objects >= 2 and len(set(directions)) == 1:
        directions[-1] = random.choice([d for d in DIRS if d != directions[0]])

    specs = []
    for i in range(n_objects):
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        direction = directions[i]
        attach_side = random.choice([side for side in DIRS if side != direction])
        specs.append((rect_h, rect_w, direction, attach_side, colors[i]))
    random.shuffle(specs)

    for rect_h, rect_w, direction, attach_side, color in specs:
        if not _place_object(grid, reserved, color, rect_h, rect_w, direction, attach_side):
            return None

    if int(np.sum(grid == bg)) <= int(np.sum(grid != bg)):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
