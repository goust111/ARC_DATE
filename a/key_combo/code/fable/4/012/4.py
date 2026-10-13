from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors8(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors8(r, c, (h, w)):
            if not mask[rr, cc] or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    records: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc, (h, w)):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(pts, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            sym_lr = bool(np.array_equal(mask, np.fliplr(mask)))
            sym_ud = bool(np.array_equal(mask, np.flipud(mask)))
            records.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "sym_lr": sym_lr,
                    "sym_ud": sym_ud,
                }
            )
    return records


def _mirror_coords(coords: np.ndarray, axis_kind: str, axis_value: int) -> np.ndarray:
    mirrored = coords.copy()
    if axis_kind == "vertical":
        mirrored[:, 1] = 2 * int(axis_value) - mirrored[:, 1]
    else:
        mirrored[:, 0] = 2 * int(axis_value) - mirrored[:, 0]
    return mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror every doubly symmetric sprite across the axis encoded by the unique singly symmetric sprite.

    Input structure:
    - A dominant background contains several pairwise separated monochrome sprites.
    - Exactly one sprite is symmetric in only one direction, while every other sprite is symmetric both left-right and up-down in its tight bbox.
    - The exceptional sprite has odd bbox width when it breaks left-right symmetry, or odd bbox height when it breaks up-down symmetry.
    - Non-exception sprites stay entirely on one side of the exceptional sprite's center row or center column.

    Rule:
    - Split the foreground into monochrome 8-connected sprites and test each tight crop for left-right and up-down self-symmetry.
    - Find the unique sprite that is symmetric in exactly one direction.
    - If that sprite is not left-right symmetric, use its bbox center column as a vertical mirror axis; otherwise use its bbox center row as a horizontal mirror axis.
    - Erase the exceptional sprite, keep every other sprite, and add its reflection across the chosen global axis.

    Key insight:
    - The odd sprite is not content to preserve; its missing symmetry direction defines the orientation and position of the mirror axis used for every other sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _majority_color(g)
    components = _component_records(g, bg)
    assert len(components) >= 2, "expected multiple sprites"

    traitors = [comp for comp in components if comp["sym_lr"] ^ comp["sym_ud"]]
    assert len(traitors) == 1, "expected exactly one singly symmetric sprite"
    traitor = traitors[0]

    others = [comp for comp in components if comp is not traitor]
    assert all(comp["sym_lr"] and comp["sym_ud"] for comp in others), "other sprites must be doubly symmetric"

    r0, c0, r1, c1 = traitor["bbox"]
    if not traitor["sym_lr"]:
        assert traitor["sym_ud"], "traitor must still be symmetric up-down"
        assert (c1 - c0 + 1) % 2 == 1, "vertical axis must pass through an odd-width bbox"
        axis_kind = "vertical"
        axis_value = (c0 + c1) // 2
    else:
        assert not traitor["sym_ud"], "traitor must break exactly one symmetry"
        assert (r1 - r0 + 1) % 2 == 1, "horizontal axis must pass through an odd-height bbox"
        axis_kind = "horizontal"
        axis_value = (r0 + r1) // 2

    out = g.copy()
    out[traitor["coords"][:, 0], traitor["coords"][:, 1]] = bg

    h, w = g.shape
    for comp in others:
        mirrored = _mirror_coords(comp["coords"], axis_kind, axis_value)
        assert np.all(0 <= mirrored[:, 0]) and np.all(mirrored[:, 0] < h)
        assert np.all(0 <= mirrored[:, 1]) and np.all(mirrored[:, 1] < w)
        out[mirrored[:, 0], mirrored[:, 1]] = int(comp["color"])

    return out


def _orbit_cells(h: int, w: int, r: int, c: int, *, sym_lr: bool, sym_ud: bool) -> list[Coord]:
    rows = {int(r)}
    cols = {int(c)}
    if sym_ud:
        rows.add(h - 1 - int(r))
    if sym_lr:
        cols.add(w - 1 - int(c))
    return [(rr, cc) for rr in sorted(rows) for cc in sorted(cols)]


def _canonical_domain(h: int, w: int, *, sym_lr: bool, sym_ud: bool) -> list[Coord]:
    row_stop = (h + 1) // 2 if sym_ud else h
    col_stop = (w + 1) // 2 if sym_lr else w
    return [(r, c) for r in range(row_stop) for c in range(col_stop)]


def _seed_cell(h: int, w: int, *, sym_lr: bool, sym_ud: bool) -> Coord:
    r = (h - 1) // 2 if sym_ud else random.randrange(h)
    c = (w - 1) // 2 if sym_lr else random.randrange(w)
    return int(r), int(c)


def _grow_symmetric_mask(
    h: int,
    w: int,
    *,
    sym_lr: bool,
    sym_ud: bool,
    min_fill: float,
    max_fill: float,
    attempts: int = 80,
) -> np.ndarray | None:
    min_area = max(4, int(np.ceil(min_fill * h * w)))
    max_area = max(min_area, int(np.floor(max_fill * h * w)))
    domain = _canonical_domain(h, w, sym_lr=sym_lr, sym_ud=sym_ud)

    for _ in range(attempts):
        mask = np.zeros((h, w), dtype=bool)
        used: set[Coord] = set()
        seed = _seed_cell(h, w, sym_lr=sym_lr, sym_ud=sym_ud)
        used.add(seed)
        for rr, cc in _orbit_cells(h, w, *seed, sym_lr=sym_lr, sym_ud=sym_ud):
            mask[rr, cc] = True

        while True:
            area = int(mask.sum())
            candidates: list[Coord] = []
            for cell in domain:
                if cell in used:
                    continue
                orbit = _orbit_cells(h, w, *cell, sym_lr=sym_lr, sym_ud=sym_ud)
                add_area = sum(1 for rr, cc in orbit if not mask[rr, cc])
                if area + add_area > max_area:
                    continue
                test = mask.copy()
                for rr, cc in orbit:
                    test[rr, cc] = True
                if _is_connected(test):
                    candidates.append(cell)

            if area >= min_area and (not candidates or random.random() < 0.35):
                break
            if not candidates:
                break

            chosen = random.choice(candidates)
            used.add(chosen)
            for rr, cc in _orbit_cells(h, w, *chosen, sym_lr=sym_lr, sym_ud=sym_ud):
                mask[rr, cc] = True

        mask = _crop_mask(mask)
        if min(mask.shape) < 2 or int(mask.sum()) < min_area or not _is_connected(mask):
            continue
        got_lr = bool(np.array_equal(mask, np.fliplr(mask)))
        got_ud = bool(np.array_equal(mask, np.flipud(mask)))
        if got_lr == sym_lr and got_ud == sym_ud:
            return mask
    return None


def _boxes_far_enough(a: BBox, b: BBox, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return bool(
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _bbox_fits(shape: tuple[int, int], box: BBox) -> bool:
    h, w = shape
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _mirror_bbox(box: BBox, axis_kind: str, axis_value: int) -> BBox:
    r0, c0, r1, c1 = box
    if axis_kind == "vertical":
        return (r0, 2 * axis_value - c1, r1, 2 * axis_value - c0)
    return (2 * axis_value - r1, c0, 2 * axis_value - r0, c1)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _candidate_exception_boxes(
    shape: tuple[int, int],
    mask_shape: tuple[int, int],
    axis_kind: str,
    axis_value: int,
) -> list[BBox]:
    h, w = shape
    mh, mw = mask_shape
    boxes: list[BBox] = []

    if axis_kind == "vertical":
        left = axis_value - mw // 2
        if left < 0 or left + mw > w:
            return []
        row_start = 1 if h - mh >= 2 else 0
        row_stop = h - mh if h - mh >= 2 else h - mh
        for top in range(row_start, row_stop + 1):
            boxes.append((top, left, top + mh - 1, left + mw - 1))
    else:
        top = axis_value - mh // 2
        if top < 0 or top + mh > h:
            return []
        col_start = 1 if w - mw >= 2 else 0
        col_stop = w - mw if w - mw >= 2 else w - mw
        for left in range(col_start, col_stop + 1):
            boxes.append((top, left, top + mh - 1, left + mw - 1))

    random.shuffle(boxes)
    return boxes


def _candidate_regular_pairs(
    shape: tuple[int, int],
    mask_shape: tuple[int, int],
    axis_kind: str,
    axis_value: int,
) -> list[tuple[BBox, BBox]]:
    h, w = shape
    mh, mw = mask_shape
    pairs: list[tuple[BBox, BBox]] = []

    for top in range(h - mh + 1):
        for left in range(w - mw + 1):
            box = (top, left, top + mh - 1, left + mw - 1)
            if axis_kind == "vertical":
                if not (box[3] < axis_value or box[1] > axis_value):
                    continue
            else:
                if not (box[2] < axis_value or box[0] > axis_value):
                    continue
            mirror = _mirror_bbox(box, axis_kind, axis_value)
            if _bbox_fits(shape, mirror):
                pairs.append((box, mirror))

    random.shuffle(pairs)
    return pairs


def _make_exception_mask(axis_kind: str) -> np.ndarray | None:
    if axis_kind == "vertical":
        h = random.randint(3, 7)
        w = random.choice((3, 5, 7))
        mask = _grow_symmetric_mask(h, w, sym_lr=False, sym_ud=True, min_fill=0.28, max_fill=0.58)
        if mask is not None and mask.shape[1] % 2 == 1 and not np.array_equal(mask, np.fliplr(mask)) and np.array_equal(mask, np.flipud(mask)):
            return mask
        return None

    h = random.choice((3, 5, 7))
    w = random.randint(3, 7)
    mask = _grow_symmetric_mask(h, w, sym_lr=True, sym_ud=False, min_fill=0.28, max_fill=0.58)
    if mask is not None and mask.shape[0] % 2 == 1 and np.array_equal(mask, np.fliplr(mask)) and not np.array_equal(mask, np.flipud(mask)):
        return mask
    return None


def _make_regular_mask() -> np.ndarray | None:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    mask = _grow_symmetric_mask(h, w, sym_lr=True, sym_ud=True, min_fill=0.24, max_fill=0.55)
    if mask is not None and np.array_equal(mask, np.fliplr(mask)) and np.array_equal(mask, np.flipud(mask)):
        return mask
    return None


def _render_expected(
    shape: tuple[int, int],
    bg: int,
    regulars: list[dict],
    axis_kind: str,
    axis_value: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for item in regulars:
        mask = item["mask"]
        color = int(item["color"])
        box = item["box"]
        _paint_mask(out, mask, box[0], box[1], color)
        mirror = _mirror_bbox(box, axis_kind, axis_value)
        _paint_mask(out, mask, mirror[0], mirror[1], color)
    return out


def _scene_valid(inp: np.ndarray, expected: np.ndarray, bg: int) -> bool:
    if _majority_color(inp) != bg:
        return False
    try:
        observed = main(inp)
    except Exception:
        return False
    if not np.array_equal(observed, expected):
        return False

    components = _component_records(inp, bg)
    if len([comp for comp in components if comp["sym_lr"] ^ comp["sym_ud"]]) != 1:
        return False
    return True


def _attempt_input() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    axis_kind = random.choice(("vertical", "horizontal"))

    exception_mask = _make_exception_mask(axis_kind)
    if exception_mask is None:
        return None

    eh, ew = exception_mask.shape
    if axis_kind == "vertical":
        lo = ew // 2 + 4
        hi = w - ew // 2 - 5
    else:
        lo = eh // 2 + 4
        hi = h - eh // 2 - 5
    if lo > hi:
        return None
    axis_value = random.randint(lo, hi)

    bg = random.randrange(10)
    n_regular = random.randint(2, 4)
    colors = random.sample([c for c in range(10) if c != bg], n_regular + 1)
    exception_color = colors[0]
    regular_colors = colors[1:]

    exception_boxes = _candidate_exception_boxes((h, w), exception_mask.shape, axis_kind, axis_value)
    if not exception_boxes:
        return None
    exception_box = random.choice(exception_boxes)

    inp = np.full((h, w), bg, dtype=int)
    _paint_mask(inp, exception_mask, exception_box[0], exception_box[1], exception_color)
    reserved = [exception_box]
    regulars: list[dict] = []

    for color in regular_colors:
        placed = False
        for _ in range(40):
            mask = _make_regular_mask()
            if mask is None:
                continue
            pairs = _candidate_regular_pairs((h, w), mask.shape, axis_kind, axis_value)
            for box, mirror in pairs:
                if not all(_boxes_far_enough(box, other, pad=1) and _boxes_far_enough(mirror, other, pad=1) for other in reserved):
                    continue
                _paint_mask(inp, mask, box[0], box[1], color)
                reserved.extend([box, mirror])
                regulars.append({"mask": mask, "box": box, "color": color})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _render_expected((h, w), bg, regulars, axis_kind, axis_value)
    return inp if _scene_valid(inp, expected, bg) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
