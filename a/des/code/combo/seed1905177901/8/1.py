import random
from collections import deque

import numpy as np


BG = 0
MARKER = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _detect_line(grid: np.ndarray) -> tuple[str, int]:
    row_hits = np.where(np.all(grid == MARKER, axis=1))[0]
    col_hits = np.where(np.all(grid == MARKER, axis=0))[0]
    if len(row_hits) == 1 and len(col_hits) == 0:
        return "h", int(row_hits[0])
    if len(col_hits) == 1 and len(row_hits) == 0:
        return "v", int(col_hits[0])
    raise AssertionError("expected exactly one full red marker line")


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, MARKER) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords})
    return comps


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask)
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - radius)
        r1 = min(h, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(w, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _erode_square(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return out
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    for r in range(int(r0) + radius, int(r1) - radius + 1):
        for c in range(int(c0) + radius, int(c1) - radius + 1):
            if mask[r, c] and np.all(mask[r - radius : r + radius + 1, c - radius : c + radius + 1]):
                out[r, c] = True
    return out


def _solve_horizontal(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_hits = np.where(np.all(g == MARKER, axis=1))[0]
    assert len(row_hits) == 1, "expected a single horizontal marker row"
    line = int(row_hits[0])

    out = g.copy()
    out[(g != BG) & (g != MARKER)] = BG

    for comp in _components(g):
        coords = comp["coords"]
        color = int(comp["color"])
        mask = _mask_from_coords((h, w), coords)
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

        if r1 < line:
            distance = int(line - r1 - 1)
            transformed = _dilate_square(mask, distance)
        elif r0 > line:
            distance = int(r0 - line - 1)
            transformed = _erode_square(mask, distance)
        else:
            raise AssertionError("foreground object must lie entirely on one side of the marker")

        transformed[line, :] = False
        out[transformed] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the full red line as a separator.

    Objects above or left of the line are dilated by the distance from their
    nearest bounding-box edge to the line; objects below or right are eroded by
    the same amount. The line itself stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    orientation, _ = _detect_line(g)
    if orientation == "h":
        return _solve_horizontal(g)
    return _solve_horizontal(g.T).T


def _make_shape(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    modes = ["rect", "rect", "rect", "corner"]
    mode = random.choice(modes)
    if mode == "corner" and h >= 3 and w >= 3:
        corner = random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
        mask[corner] = False
    return mask


def _pad_box(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _box_free(reserved: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_box(reserved: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint_shape(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _place_above(
    grid: np.ndarray,
    input_reserved: np.ndarray,
    output_reserved: np.ndarray,
    line: int,
    color: int,
) -> bool:
    h, w = grid.shape
    valid_distances = []
    for d in range(1, 4):
        max_h = min(5, line - 2 * d - 1)
        max_w = min(6, w - 2 * d - 2)
        if max_h >= 2 and max_w >= 2:
            valid_distances.append(d)
    if not valid_distances:
        return False

    for _ in range(200):
        d = random.choice(valid_distances)
        obj_h = random.randint(2, min(5, line - 2 * d - 1))
        obj_w = random.randint(2, min(6, w - 2 * d - 2))
        shape = _make_shape(obj_h, obj_w)

        r0 = line - d - obj_h
        r1 = r0 + obj_h - 1
        c_min = d + 1
        c_max = w - obj_w - d - 1
        if c_min > c_max:
            continue
        candidates = list(range(c_min, c_max + 1))
        random.shuffle(candidates)

        for c0 in candidates:
            c1 = c0 + obj_w - 1
            input_box = _pad_box(r0, c0, r1, c1, h, w, pad=1)
            expanded_box = (
                max(0, r0 - d - 1),
                max(0, c0 - d - 1),
                line - 1,
                min(w - 1, c1 + d + 1),
            )
            if not _box_free(input_reserved, input_box):
                continue
            if not _box_free(output_reserved, expanded_box):
                continue
            _mark_box(input_reserved, input_box)
            _mark_box(output_reserved, expanded_box)
            _paint_shape(grid, r0, c0, shape, color)
            return True
    return False


def _place_below(
    grid: np.ndarray,
    input_reserved: np.ndarray,
    line: int,
    color: int,
    must_survive: bool,
) -> bool:
    h, w = grid.shape
    valid_distances = []
    for d in range(1, 4):
        max_h = min(6, h - line - d - 2)
        max_w = min(6, w - 2)
        if max_h < 2 or max_w < 2:
            continue
        if must_survive:
            if max_h >= 2 * d + 1 and max_w >= 2 * d + 1:
                valid_distances.append(d)
        else:
            if max_h >= 2:
                valid_distances.append(d)
    if not valid_distances:
        return False

    for _ in range(200):
        d = random.choice(valid_distances)
        max_h = min(6, h - line - d - 2)
        max_w = min(6, w - 2)
        if must_survive:
            obj_h = random.randint(2 * d + 1, min(max_h, 2 * d + 4))
            obj_w = random.randint(2 * d + 1, min(max_w, 2 * d + 4))
        else:
            obj_h = random.randint(2, min(max_h, max(2, 2 * d)))
            obj_w = random.randint(2, max_w)
        shape = _make_shape(obj_h, obj_w)

        r0 = line + d + 1
        r1 = r0 + obj_h - 1
        c_min = 1
        c_max = w - obj_w - 1
        if c_min > c_max:
            continue
        candidates = list(range(c_min, c_max + 1))
        random.shuffle(candidates)

        for c0 in candidates:
            c1 = c0 + obj_w - 1
            input_box = _pad_box(r0, c0, r1, c1, h, w, pad=1)
            if not _box_free(input_reserved, input_box):
                continue
            _mark_box(input_reserved, input_box)
            _paint_shape(grid, r0, c0, shape, color)
            return True
    return False


def _generate_horizontal() -> np.ndarray | None:
    h = random.randint(13, 20)
    w = random.randint(15, 22)
    line = random.randint(5, h - 6)

    grid = np.full((h, w), BG, dtype=int)
    grid[line, :] = MARKER

    top_count = 1 + int(w >= 18 and random.random() < 0.6)
    bottom_count = 1 + int(w >= 16 and random.random() < 0.6)
    total_objects = top_count + bottom_count
    colors = random.sample(PALETTE, total_objects)

    input_reserved = np.zeros((h, w), dtype=bool)
    output_reserved = np.zeros((h, w), dtype=bool)
    input_reserved[line, :] = True
    output_reserved[line, :] = True

    idx = 0
    for _ in range(top_count):
        if not _place_above(grid, input_reserved, output_reserved, line, colors[idx]):
            return None
        idx += 1

    survive_slots = [True] + [random.random() < 0.5 for _ in range(bottom_count - 1)]
    random.shuffle(survive_slots)
    for must_survive in survive_slots:
        if not _place_below(grid, input_reserved, line, colors[idx], must_survive):
            return None
        idx += 1

    return grid


def _nontrivial_change(inp: np.ndarray, out: np.ndarray) -> bool:
    return not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _generate_horizontal()
        if inp is None:
            continue
        if random.random() < 0.5:
            inp = inp.T.copy()
        try:
            out = main(inp)
        except Exception:
            continue
        if _nontrivial_change(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid ARC instance")
