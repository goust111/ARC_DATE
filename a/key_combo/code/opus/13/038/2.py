import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = (BLUE, RED, GREEN, YELLOW)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(shape_key: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape_key), 1 + max(c for _, c in shape_key)


def _overlay_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = right.copy()
    out[left != BLACK] = left[left != BLACK]
    return out


def _extract_boxes(scene: np.ndarray) -> list[dict]:
    boxes = []
    for coords in _components_from_mask(scene == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        actual = set(coords)
        assert actual == expected, "grey components must be hollow rectangles"
        boxes.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )
    boxes.sort(key=lambda box: box["bbox"])
    return boxes


def _extract_shapes(scene: np.ndarray) -> list[dict]:
    shapes = []
    mask = (scene != BLACK) & (scene != GREY)
    for coords in _components_from_mask(mask):
        colors = {int(scene[r, c]) for r, c in coords}
        assert len(colors) == 1, "every colored shape must be monochromatic after overlay"
        color = next(iter(colors))
        shapes.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "color": color,
                "key": _normalize(coords),
            }
        )
    shapes.sort(key=lambda shape: shape["bbox"])
    return shapes


def _shape_inside_box(shape_coords: list[tuple[int, int]], box: dict) -> bool:
    r0, c0, r1, c1 = box["bbox"]
    return all(r0 < r < r1 and c0 < c < c1 for r, c in shape_coords)


def _partition_shapes(shapes: list[dict], boxes: list[dict]) -> tuple[list[dict], list[dict]]:
    inside: list[dict] = []
    outside: list[dict] = []
    for shape in shapes:
        containing = [idx for idx, box in enumerate(boxes) if _shape_inside_box(shape["coords"], box)]
        assert len(containing) <= 1, "a shape cannot belong to multiple boxes"
        if containing:
            tagged = dict(shape)
            tagged["box_index"] = containing[0]
            inside.append(tagged)
        else:
            outside.append(dict(shape))
    return inside, outside


def _solve_overlay_scene(scene: np.ndarray) -> np.ndarray:
    boxes = _extract_boxes(scene)
    shapes = _extract_shapes(scene)
    inside, outside = _partition_shapes(shapes, boxes)

    assert len(boxes) in (2, 3), "expected 2 or 3 boxes"
    assert 4 <= len(shapes) <= 6, "expected 4 to 6 colored shapes"
    assert len(inside) == len(boxes), "each box must contain exactly one shape"
    assert len(outside) >= 2, "expected at least two outside shapes"

    reference_colors: dict[tuple[tuple[int, int], ...], int] = {}
    for shape in inside:
        key = shape["key"]
        assert key not in reference_colors, "inside reference shapes must be unique"
        reference_colors[key] = int(shape["color"])

    out = scene.copy()
    recolored = 0
    for shape in outside:
        key = shape["key"]
        assert key in reference_colors, "every outside shape must match one inside reference"
        target = reference_colors[key]
        assert target != int(shape["color"]), "outside shape must start with a different color"
        for r, c in shape["coords"]:
            out[r, c] = target
        recolored += 1

    assert recolored == len(outside)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay two split panels, recover boxed reference shapes, and recolor outside copies to the boxed colors.

    Input structure:
    - The grid has a solid grey separator column in the middle, splitting it into equal left and right panels.
    - Each panel has black background plus scattered grey box fragments and colored shape fragments.
    - Only after aligning the two panels and transparently overlaying them do complete grey hollow rectangles and complete colored 4-connected shapes appear.

    Rule:
    - Remove the middle separator column and overlay the left and right panels by relative position.
    - Use the non-black pixel when only one side is non-black, and use the left pixel when both sides are non-black.
    - In the overlaid scene, find grey hollow rectangles, identify the colored shapes strictly inside them, and use those boxed shapes as same-shape color references.
    - Recolor every outside shape whose translated occupancy mask exactly matches a boxed reference shape, while leaving the grey boxes and boxed shapes unchanged.

    Key insight:
    - The containment and shape-matching logic is impossible on either panel alone; the aligned left-priority overlay is the step that reconstructs the real objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 3 and w % 2 == 1, "input must have two equal panels and a middle separator"
    mid = w // 2
    assert np.all(g[:, mid] == GREY), "expected a solid grey separator column"
    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape
    scene = _overlay_panels(left, right)
    return _solve_overlay_scene(scene)


def _shape_key(*coords: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(coords))


SHAPE_LIBRARY = (
    _shape_key((0, 0), (0, 1), (1, 0)),
    _shape_key((0, 0), (0, 1), (1, 1)),
    _shape_key((0, 0), (1, 0), (1, 1)),
    _shape_key((0, 1), (1, 0), (1, 1)),
    _shape_key((0, 0), (0, 1), (0, 2)),
    _shape_key((0, 0), (1, 0), (2, 0)),
    _shape_key((0, 0), (0, 1), (1, 0), (1, 1)),
    _shape_key((0, 0), (0, 1), (0, 2), (1, 1)),
    _shape_key((0, 1), (1, 0), (1, 1), (1, 2)),
    _shape_key((0, 0), (1, 0), (2, 0), (1, 1)),
    _shape_key((0, 1), (1, 0), (1, 1), (2, 1)),
    _shape_key((0, 0), (0, 1), (1, 1), (1, 2)),
    _shape_key((0, 1), (1, 0), (1, 1), (2, 0)),
    _shape_key((0, 0), (1, 0), (2, 0), (2, 1)),
    _shape_key((0, 1), (1, 1), (2, 1), (2, 0)),
    _shape_key((0, 0), (0, 1), (0, 2), (1, 0)),
    _shape_key((0, 0), (0, 1), (0, 2), (1, 2)),
    _shape_key((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    _shape_key((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    _shape_key((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    _shape_key((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    _shape_key((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    _shape_key((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    _shape_key((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
)


def _fits(shape_key: tuple[tuple[int, int], ...], max_h: int, max_w: int) -> bool:
    h, w = _shape_dims(shape_key)
    return h <= max_h and w <= max_w


def _draw_box(scene: np.ndarray, top: int, left: int, size: int) -> None:
    bottom = top + size - 1
    right = left + size - 1
    scene[top, left : right + 1] = GREY
    scene[bottom, left : right + 1] = GREY
    scene[top : bottom + 1, left] = GREY
    scene[top : bottom + 1, right] = GREY


def _paint_shape(scene: np.ndarray, top: int, left: int, shape_key: tuple[tuple[int, int], ...], color: int) -> list[tuple[int, int]]:
    coords = []
    for dr, dc in shape_key:
        rr = top + dr
        cc = left + dc
        scene[rr, cc] = int(color)
        coords.append((rr, cc))
    return coords


def _blocked_for_boxes(h: int, w: int, boxes: list[tuple[int, int]], size: int) -> np.ndarray:
    blocked = np.zeros((h, w), dtype=bool)
    for top, left in boxes:
        blocked[top : top + size, left : left + size] = True
    return blocked


def _can_place_shape(scene: np.ndarray, blocked: np.ndarray, top: int, left: int, shape_key: tuple[tuple[int, int], ...]) -> bool:
    h, w = scene.shape
    cells = [(top + dr, left + dc) for dr, dc in shape_key]
    for rr, cc in cells:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if blocked[rr, cc] or scene[rr, cc] != BLACK:
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w and scene[nr, nc] != BLACK:
                    return False
    return True


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for rr, cc in coords:
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _candidate_positions(h: int, w: int, shape_key: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    sh, sw = _shape_dims(shape_key)
    positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(positions)
    return positions


def _layout_spec(box_count: int, box_size: int) -> tuple[int, list[tuple[int, int]]]:
    if box_count == 3:
        return 13, [(0, 0), (7, 0), (3, 7)]
    if box_size == 7:
        base = 14
    else:
        base = 13
    options = (
        [(0, 0), (base - box_size, base - box_size)],
        [(0, base - box_size), (base - box_size, 0)],
    )
    return base, random.choice(options)


def _transformed_scene(scene: np.ndarray) -> np.ndarray:
    out = np.rot90(scene, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _pad_scene(scene: np.ndarray) -> np.ndarray:
    h, w = scene.shape
    max_extra_h = 15 - h
    max_extra_w = 14 - w
    extra_h = random.randint(0, max_extra_h)
    extra_w = random.randint(0, max_extra_w)
    top = random.randint(0, extra_h)
    bottom = extra_h - top
    left = random.randint(0, extra_w)
    right = extra_w - left
    return np.pad(scene, ((top, bottom), (left, right)), constant_values=BLACK)


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    box_count = 3 if random.random() < 0.5 else 2
    box_size = 6 if box_count == 3 else random.choice((6, 7))
    base, boxes = _layout_spec(box_count, box_size)
    scene = np.zeros((base, base), dtype=int)
    safe = box_size - 4

    candidates = [shape for shape in SHAPE_LIBRARY if _fits(shape, safe, safe)]
    assert len(candidates) >= box_count
    inside_shapes = random.sample(candidates, box_count)
    inside_colors = random.sample(PALETTE, box_count)

    for top, left in boxes:
        _draw_box(scene, top, left, box_size)

    for (top, left), shape_key, color in zip(boxes, inside_shapes, inside_colors):
        sh, sw = _shape_dims(shape_key)
        safe_top = top + 2
        safe_left = left + 2
        rr = safe_top + random.randint(0, safe - sh)
        cc = safe_left + random.randint(0, safe - sw)
        _paint_shape(scene, rr, cc, shape_key, color)

    outside_count = 3 if box_count == 3 or random.random() < 0.5 else 2
    mapping = list(range(box_count))
    if outside_count > box_count:
        mapping.append(random.randrange(box_count))
    random.shuffle(mapping)

    blocked = _blocked_for_boxes(base, base, boxes, box_size)
    for idx in mapping:
        shape_key = inside_shapes[idx]
        target_color = inside_colors[idx]
        source_choices = [color for color in PALETTE if color != target_color]
        source_color = random.choice(source_choices)
        placed = False
        for top, left in _candidate_positions(base, base, shape_key):
            if _can_place_shape(scene, blocked, top, left, shape_key):
                coords = _paint_shape(scene, top, left, shape_key, source_color)
                _mark_blocked(blocked, coords)
                placed = True
                break
        if not placed:
            raise ValueError("failed to place an outside shape")

    pre_scene = _pad_scene(_transformed_scene(scene))
    expected = _solve_overlay_scene(pre_scene)
    return pre_scene, expected


def _split_object(coords: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = list(coords)
    for _ in range(40):
        left_cov: set[tuple[int, int]] = set()
        right_cov: set[tuple[int, int]] = set()
        for cell in cells:
            tag = random.random()
            if tag < 0.35:
                left_cov.add(cell)
            elif tag < 0.70:
                right_cov.add(cell)
            else:
                left_cov.add(cell)
                right_cov.add(cell)
        if left_cov and right_cov and left_cov != set(cells) and right_cov != set(cells):
            return left_cov, right_cov
    left_cov = set(cells[1:])
    right_cov = set(cells[:-1])
    return left_cov, right_cov


def _split_scene(scene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros_like(scene)
    right = np.zeros_like(scene)

    objects = []
    for box in _extract_boxes(scene):
        objects.append({"kind": "box", "coords": box["coords"], "color": GREY})
    for shape in _extract_shapes(scene):
        objects.append({"kind": "shape", "coords": shape["coords"], "color": int(shape["color"])})

    intersections: list[tuple[list[tuple[int, int]], int]] = []
    for obj in objects:
        left_cov, right_cov = _split_object(obj["coords"])
        for r, c in left_cov:
            left[r, c] = obj["color"]
        for r, c in right_cov:
            right[r, c] = obj["color"]
        shared = sorted(left_cov & right_cov)
        if shared:
            intersections.append((shared, obj["color"]))

    assert intersections, "need at least one overlapping non-black cell"
    conflict_cells = random.sample(random.choice(intersections)[0], k=1)
    for r, c in conflict_cells:
        choices = [color for color in PALETTE if color != int(scene[r, c])]
        right[r, c] = random.choice(choices)

    return left, right


def _assemble_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = left.shape
    grid = np.zeros((h, 2 * w + 1), dtype=int)
    grid[:, :w] = left
    grid[:, w] = GREY
    grid[:, w + 1 :] = right
    return grid


def _validate_scene(scene: np.ndarray, expected: np.ndarray, grid: np.ndarray) -> bool:
    h, w = scene.shape
    if scene.shape != expected.shape:
        return False
    if grid.shape != (h, 2 * w + 1):
        return False
    left = grid[:, :w]
    right = grid[:, w + 1 :]
    if not np.all(grid[:, w] == GREY):
        return False
    if not np.array_equal(_overlay_panels(left, right), scene):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    conflict_count = int(np.count_nonzero((left != BLACK) & (right != BLACK) & (left != right)))
    if conflict_count < 1:
        return False

    boxes = _extract_boxes(scene)
    shapes = _extract_shapes(scene)
    inside, outside = _partition_shapes(shapes, boxes)
    if len(boxes) not in (2, 3):
        return False
    if len(shapes) < 4 or len(shapes) > 6:
        return False
    if len(inside) != len(boxes):
        return False
    if len(outside) < 2:
        return False

    target_colors = {}
    for shape in inside:
        if shape["key"] in target_colors:
            return False
        target_colors[shape["key"]] = shape["color"]

    recolor_targets = set()
    for shape in outside:
        if shape["key"] not in target_colors:
            return False
        if target_colors[shape["key"]] == shape["color"]:
            return False
        recolor_targets.add(target_colors[shape["key"]])
    if len(recolor_targets) < 2:
        return False

    objects = [box["coords"] for box in boxes] + [shape["coords"] for shape in shapes]
    for coords in objects:
        mask_left = all(left[r, c] != BLACK for r, c in coords)
        mask_right = all(right[r, c] != BLACK for r, c in coords)
        if mask_left or mask_right:
            return False
        if not any(left[r, c] != BLACK for r, c in coords):
            return False
        if not any(right[r, c] != BLACK for r, c in coords):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        scene, expected = _build_scene()
        left, right = _split_scene(scene)
        grid = _assemble_input(left, right)
    except (AssertionError, ValueError):
        return None
    return grid if _validate_scene(scene, expected, grid) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the overlay containment recolor task")
