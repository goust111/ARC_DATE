from __future__ import annotations

import random

import numpy as np


GREY = 5
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

_GEN_ATTEMPTS = 600
_PLACEMENT_ATTEMPTS = 300


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _mirror_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return c0, r0, c1, r1


def _bboxes_touch_or_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    margin: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _collect_grey_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    rects: list[dict] = []

    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) != GREY:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(g[nr, nc]) == GREY:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            r0, c0, r1, c1 = bbox
            h, w = _bbox_dims(bbox)
            area = len(coords)
            if h < 3 or w < 3:
                raise AssertionError("each grey component must be a rectangle with height and width >= 3")
            if area != h * w:
                raise AssertionError("each grey component must be a filled rectangle")
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY):
                raise AssertionError("rectangle bbox must be fully grey")

            rects.append({"bbox": bbox, "area": area, "dims": (h, w)})

    return sorted(rects, key=lambda rect: rect["bbox"])


def _classify_rectangles(rects: list[dict]) -> list[str]:
    bbox_to_index = {rect["bbox"]: idx for idx, rect in enumerate(rects)}
    classes: list[str] = []
    for idx, rect in enumerate(rects):
        bbox = rect["bbox"]
        mirrored = _mirror_bbox(bbox)
        partner = bbox_to_index.get(mirrored)
        if mirrored == bbox:
            classes.append("B")
        elif partner is not None and partner != idx:
            classes.append("A")
        else:
            classes.append("C")
    return classes


def _expected_output_from_rectangles(grid: np.ndarray, rects: list[dict], classes: list[str]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    fill = {"A": RED, "B": BLUE, "C": GREEN}
    for rect, cls in zip(rects, classes):
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill[cls]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangle interior by its main-diagonal mirror class while keeping all borders grey.

    Input structure:
    - The grid is square with black background and several non-touching filled grey rectangles.
    - Every grey object is an axis-aligned rectangle with a one-pixel border and a non-empty strict interior.
    - Rectangles do not touch the canvas edge.

    Rule:
    - Detect every grey rectangle and mirror its bbox across the main diagonal, mapping
      `(r0,c0)-(r1,c1)` to `(c0,r0)-(c1,r1)`.
    - If that mirrored bbox matches another rectangle, both rectangles are class A.
    - If the mirrored bbox matches the rectangle itself, it is class B.
    - Otherwise it is class C.
    - Preserve the black background and every rectangle's one-pixel grey border, and recolor
      only the strict interior: A -> red, B -> blue, C -> green.

    Key insight:
    - A rectangle's new interior color is not determined locally by its own size or position;
      it depends on the global pattern of which other rectangles appear at its diagonal mirror location.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("input must be a 2D grid")
    H, W = g.shape
    if H != W:
        raise AssertionError("input grid must be square")
    if not set(map(int, np.unique(g))) <= {BLACK, GREY}:
        raise AssertionError("input may contain only black and grey")

    rects = _collect_grey_rectangles(g)
    if not rects:
        raise AssertionError("expected at least one grey rectangle")

    return _expected_output_from_rectangles(g, rects, _classify_rectangles(rects))


def _empty_grid(size: int) -> np.ndarray:
    return np.full((size, size), BLACK, dtype=int)


def _paint_rectangles(size: int, bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = _empty_grid(size)
    for r0, c0, r1, c1 in bboxes:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid


def _can_add(existing: list[tuple[int, int, int, int]], candidates: list[tuple[int, int, int, int]]) -> bool:
    for bbox in candidates:
        r0, c0, r1, c1 = bbox
        if r0 < 1 or c0 < 1:
            return False
        if any(v < 0 for v in bbox):
            return False
        if r1 >= _current_size - 1 or c1 >= _current_size - 1:
            return False
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            if _bboxes_touch_or_overlap(a, b, margin=1):
                return False
        for b in existing:
            if _bboxes_touch_or_overlap(a, b, margin=1):
                return False
    return True


def _place_a_pair(existing: list[tuple[int, int, int, int]], dims: tuple[int, int]) -> list[tuple[int, int, int, int]] | None:
    h, w = dims
    for _ in range(_PLACEMENT_ATTEMPTS):
        r0 = random.randint(1, _current_size - h - 1)
        c0 = random.randint(1, _current_size - w - 1)
        bbox1 = (r0, c0, r0 + h - 1, c0 + w - 1)
        bbox2 = _mirror_bbox(bbox1)
        if bbox1 == bbox2:
            continue
        if _can_add(existing, [bbox1, bbox2]):
            return [bbox1, bbox2]
    return None


def _place_b(existing: list[tuple[int, int, int, int]], side: int) -> tuple[int, int, int, int] | None:
    for _ in range(_PLACEMENT_ATTEMPTS):
        d = random.randint(1, _current_size - side - 1)
        bbox = (d, d, d + side - 1, d + side - 1)
        if _can_add(existing, [bbox]):
            return bbox
    return None


def _place_c(existing: list[tuple[int, int, int, int]], dims: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = dims
    for _ in range(_PLACEMENT_ATTEMPTS):
        r0 = random.randint(1, _current_size - h - 1)
        c0 = random.randint(1, _current_size - w - 1)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        mirrored = _mirror_bbox(bbox)
        if mirrored == bbox:
            continue
        if _bboxes_touch_or_overlap(bbox, mirrored):
            continue
        if not _can_add(existing, [bbox]):
            continue
        if any(_bboxes_touch_or_overlap(mirrored, other) for other in existing + [bbox]):
            continue
        return bbox
    return None


def _sample_dims(max_side: int = 5) -> tuple[int, int]:
    h = random.randint(3, max_side)
    w = random.randint(3, max_side)
    return h, w


def _build_plan(total_rects: int) -> list[str]:
    plan = ["A", "A", "B", "C"]
    if total_rects == 5:
        plan.append(random.choice(["B", "C"]))
    elif total_rects == 6:
        extras = random.choice(
            [
                ["A", "A"],
                ["B", "C"],
                ["B", "B"],
                ["C", "C"],
            ]
        )
        plan.extend(extras)
    return plan


def _validate_generation_semantics(grid: np.ndarray) -> None:
    rects = _collect_grey_rectangles(grid)
    if not (4 <= len(rects) <= 6):
        raise AssertionError("generator must create 4-6 rectangles")

    bboxes = [rect["bbox"] for rect in rects]
    for i, a in enumerate(bboxes):
        r0, c0, r1, c1 = a
        if r0 <= 0 or c0 <= 0 or r1 >= grid.shape[0] - 1 or c1 >= grid.shape[1] - 1:
            raise AssertionError("rectangles may not touch the border")
        for b in bboxes[i + 1 :]:
            if _bboxes_touch_or_overlap(a, b, margin=1):
                raise AssertionError("rectangles must be visually separated")

    classes = _classify_rectangles(rects)
    counts = {cls: classes.count(cls) for cls in ("A", "B", "C")}
    if counts["A"] < 2 or counts["A"] % 2 != 0:
        raise AssertionError("must contain at least one mirror pair")
    if counts["B"] < 1:
        raise AssertionError("must contain at least one self-symmetric rectangle")
    if counts["C"] < 1:
        raise AssertionError("must contain at least one unpaired rectangle")

    bbox_set = set(bboxes)
    for rect, cls in zip(rects, classes):
        mirrored = _mirror_bbox(rect["bbox"])
        if cls == "A":
            if mirrored == rect["bbox"] or mirrored not in bbox_set:
                raise AssertionError("A rectangles must mirror to another rectangle")
        elif cls == "B":
            if mirrored != rect["bbox"]:
                raise AssertionError("B rectangles must be self-symmetric")
        else:
            if mirrored in bbox_set:
                raise AssertionError("C rectangles must have no mirrored partner")
            if any(_bboxes_touch_or_overlap(mirrored, other) for other in bboxes):
                raise AssertionError("C mirror position should remain visibly empty")

    dims_with_class = [(cls, rect["dims"]) for rect, cls in zip(rects, classes)]
    if not any(cls1 != cls2 and dims1 == dims2 for cls1, dims1 in dims_with_class for cls2, dims2 in dims_with_class):
        raise AssertionError("need at least one same-shape pair across different classes to block shortcuts")

    out = main(grid)
    if not set(map(int, np.unique(out))) <= {BLACK, GREY, BLUE, RED, GREEN}:
        raise AssertionError("unexpected output colors")
    if not {BLUE, RED, GREEN} <= set(map(int, np.unique(out))):
        raise AssertionError("output must exhibit all three recolor branches")

    expected = _expected_output_from_rectangles(grid, rects, classes)
    if not np.array_equal(out, expected):
        raise AssertionError("output does not match the rule-derived expectation")


def _attempt_generate() -> np.ndarray | None:
    global _current_size

    _current_size = random.randint(12, 18)
    total_rects = random.randint(4, 6)
    plan = _build_plan(total_rects)
    bboxes: list[tuple[int, int, int, int]] = []
    kinds: list[str] = []
    stored_dims: list[tuple[str, tuple[int, int]]] = []

    a_dims = _sample_dims()
    pair = _place_a_pair(bboxes, a_dims)
    if pair is None:
        return None
    bboxes.extend(pair)
    kinds.extend(["A", "A"])
    stored_dims.extend([("A", a_dims), ("A", (a_dims[1], a_dims[0]))])

    b_side = random.randint(3, 5)
    b_box = _place_b(bboxes, b_side)
    if b_box is None:
        return None
    bboxes.append(b_box)
    kinds.append("B")
    stored_dims.append(("B", (b_side, b_side)))

    c_dims = random.choice([a_dims, (a_dims[1], a_dims[0]), (b_side, b_side), _sample_dims()])
    c_box = _place_c(bboxes, c_dims)
    if c_box is None:
        return None
    bboxes.append(c_box)
    kinds.append("C")
    stored_dims.append(("C", c_dims))

    extras = plan[4:]
    for kind in extras:
        if kind == "A":
            dims = random.choice([a_dims, (a_dims[1], a_dims[0]), _sample_dims()])
            pair = _place_a_pair(bboxes, dims)
            if pair is None:
                return None
            bboxes.extend(pair)
            kinds.extend(["A", "A"])
            stored_dims.extend([("A", dims), ("A", (dims[1], dims[0]))])
        elif kind == "B":
            side = random.choice([b_side, random.randint(3, 5)])
            bbox = _place_b(bboxes, side)
            if bbox is None:
                return None
            bboxes.append(bbox)
            kinds.append("B")
            stored_dims.append(("B", (side, side)))
        else:
            dims = random.choice([stored_dims[-1][1], a_dims, (a_dims[1], a_dims[0]), (b_side, b_side), _sample_dims()])
            bbox = _place_c(bboxes, dims)
            if bbox is None:
                return None
            bboxes.append(bbox)
            kinds.append("C")
            stored_dims.append(("C", dims))

    if len(bboxes) != total_rects:
        return None

    grid = _paint_rectangles(_current_size, bboxes)
    rects = _collect_grey_rectangles(grid)
    actual_classes = _classify_rectangles(rects)

    if sorted(actual_classes) != sorted(kinds):
        return None

    try:
        _validate_generation_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-mirror rectangle task input")


_current_size = 0
