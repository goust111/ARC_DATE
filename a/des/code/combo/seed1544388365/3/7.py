from __future__ import annotations

import random

import numpy as np


BG = 0
STAMP_COLOR = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_to_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rotations(mask: np.ndarray) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    out: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    cur = np.asarray(mask, dtype=bool)
    for _ in range(4):
        key = (tuple(map(int, cur.shape)), tuple(int(v) for v in cur.reshape(-1)))
        if key not in seen:
            seen.add(key)
            out.append((_mask_to_coords(cur), int(cur.shape[0]), int(cur.shape[1])))
        cur = np.rot90(cur)
    return out


def _template_patch(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    gray = g == STAMP_COLOR
    comps = _component_coords(gray)
    assert len(comps) == 1, "expected one gray stamp template"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    h, w = g.shape
    touches = [
        r0 == 0 and c0 == 0,
        r0 == 0 and c1 == w - 1,
        r1 == h - 1 and c0 == 0,
        r1 == h - 1 and c1 == w - 1,
    ]
    assert sum(touches) == 1, "template bbox should be anchored in exactly one corner"
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BG, STAMP_COLOR}):
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        assert coords, f"missing pixels for color {color}"
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((sub == color) | (sub == BG)), "rectangle region should only contain its color and hint dots"
        hints = _mask_to_coords(sub == BG)
        assert len(hints) == 2, "each rectangle must contain exactly two hint dots"
        rects.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": tuple(map(int, sub.shape)),
                "hints": hints,
            }
        )
    assert rects, "expected at least one colored rectangle"
    return rects


def _candidate_stamps(
    rect_shape: tuple[int, int],
    hints: tuple[tuple[int, int], ...],
    rotations: list[tuple[tuple[tuple[int, int], ...], int, int]],
) -> set[tuple[tuple[int, int], ...]]:
    assert len(hints) == 2
    (h0r, h0c), (h1r, h1c) = hints
    delta = (int(h1r - h0r), int(h1c - h0c))
    rect_h, rect_w = map(int, rect_shape)
    candidates: set[tuple[tuple[int, int], ...]] = set()

    for coords, _, _ in rotations:
        for ar, ac in coords:
            for br, bc in coords:
                if ar == br and ac == bc:
                    continue
                if (int(br - ar), int(bc - ac)) != delta:
                    continue
                off_r = int(h0r - ar)
                off_c = int(h0c - ac)
                placed = tuple(sorted((int(r + off_r), int(c + off_c)) for r, c in coords))
                if all(0 <= r < rect_h and 0 <= c < rect_w for r, c in placed):
                    candidates.add(placed)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored rectangle's black stamp using the gray corner template.

    Input structure:
    - Black background.
    - One disconnected gray template in a corner.
    - Several unique-color filled rectangles, each with exactly two black hint pixels.

    Rule:
    - Keep the gray template unchanged.
    - For each rectangle, rotate/translate the template so two template pixels land on the
      two black hint pixels.
    - Paint every aligned template pixel black inside that rectangle.
    """
    g = np.asarray(grid, dtype=int)
    template = _template_patch(g)
    rotations = _rotations(template)
    rects = _find_rectangles(g)

    out = g.copy()
    for rect in rects:
        candidates = _candidate_stamps(rect["shape"], rect["hints"], rotations)
        assert len(candidates) == 1, "hint pair should determine a unique stamp placement"
        r0, c0, _, _ = rect["bbox"]
        for rr, cc in next(iter(candidates)):
            out[r0 + rr, c0 + cc] = BG

    return out


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _random_connected_template() -> np.ndarray | None:
    raw_h = random.randint(2, 4)
    raw_w = random.randint(2, 5)
    max_area = raw_h * raw_w - 1
    if max_area < 4:
        return None
    target = random.randint(4, min(8, max_area))
    mask = np.zeros((raw_h, raw_w), dtype=bool)
    seed = (random.randrange(raw_h), random.randrange(raw_w))
    mask[seed] = True

    while int(mask.sum()) < target:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIRS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < raw_h and 0 <= nc < raw_w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    coords = _mask_to_coords(mask)
    r0, c0, r1, c1 = _bbox(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    if np.all(cropped):
        return None
    if int(cropped.sum()) < 4:
        return None
    return cropped


def _actual_stamp(
    rot_coords: tuple[tuple[int, int], ...], offset: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    off_r, off_c = offset
    return tuple(sorted((int(r + off_r), int(c + off_c)) for r, c in rot_coords))


def _valid_hint_pairs(
    rect_shape: tuple[int, int],
    actual: tuple[tuple[int, int], ...],
    rotations: list[tuple[tuple[tuple[int, int], ...], int, int]],
) -> list[tuple[tuple[int, int], ...]]:
    cells = list(actual)
    good: list[tuple[tuple[int, int], ...]] = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            hints = tuple(sorted((cells[i], cells[j])))
            if _candidate_stamps(rect_shape, hints, rotations) == {actual}:
                good.append(hints)
    return good


def _sample_rectangle_specs(
    rotations: list[tuple[tuple[tuple[int, int], ...], int, int]]
) -> list[dict] | None:
    count = random.randint(1, 4)
    colors = random.sample(list(RECT_COLORS), count)
    specs: list[dict] = []

    for color in colors:
        for _ in range(80):
            rot_coords, rot_h, rot_w = random.choice(rotations)
            top = random.randint(0, 3)
            left = random.randint(0, 3)
            bottom = random.randint(0, 3)
            right = random.randint(0, 3)
            rect_h = rot_h + top + bottom
            rect_w = rot_w + left + right
            actual = _actual_stamp(rot_coords, (top, left))
            hints_list = _valid_hint_pairs((rect_h, rect_w), actual, rotations)
            if not hints_list:
                continue
            specs.append(
                {
                    "color": color,
                    "shape": (int(rect_h), int(rect_w)),
                    "stamp": actual,
                    "hints": random.choice(hints_list),
                }
            )
            break
        else:
            return None
    return specs


def _corner_bbox(corner: str, grid_h: int, grid_w: int, obj_h: int, obj_w: int) -> tuple[int, int, int, int]:
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, grid_w - obj_w
    elif corner == "BL":
        r0, c0 = grid_h - obj_h, 0
    elif corner == "BR":
        r0, c0 = grid_h - obj_h, grid_w - obj_w
    else:
        raise AssertionError(corner)
    return int(r0), int(c0), int(r0 + obj_h - 1), int(c0 + obj_w - 1)


def _place_objects(
    template_shape: tuple[int, int],
    rect_specs: list[dict],
) -> tuple[int, int, tuple[int, int, int, int], list[tuple[int, int, int, int]]] | None:
    t_h, t_w = map(int, template_shape)
    for _ in range(120):
        h = random.randint(max(12, t_h + 6), 26)
        w = random.randint(max(12, t_w + 6), 26)
        corner = random.choice(("TL", "TR", "BL", "BR"))
        template_bbox = _corner_bbox(corner, h, w, t_h, t_w)
        placed = [template_bbox]
        rect_bboxes: list[tuple[int, int, int, int]] = []
        ok = True
        for spec in rect_specs:
            rh, rw = map(int, spec["shape"])
            if rh > h - 2 or rw > w - 2:
                ok = False
                break
            candidates = []
            for r0 in range(1, h - rh):
                for c0 in range(1, w - rw):
                    bbox = (int(r0), int(c0), int(r0 + rh - 1), int(c0 + rw - 1))
                    if all(_separated(bbox, other, gap=1) for other in placed):
                        candidates.append(bbox)
            if not candidates:
                ok = False
                break
            bbox = random.choice(candidates)
            rect_bboxes.append(bbox)
            placed.append(bbox)
        if ok:
            return h, w, template_bbox, rect_bboxes
    return None


def _render_case(
    template: np.ndarray,
    rect_specs: list[dict],
    template_bbox: tuple[int, int, int, int],
    rect_bboxes: list[tuple[int, int, int, int]],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, grid_shape)
    grid = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    tr0, tc0, _, _ = template_bbox
    for r, c in _mask_to_coords(template):
        grid[tr0 + r, tc0 + c] = STAMP_COLOR
        out[tr0 + r, tc0 + c] = STAMP_COLOR

    for spec, bbox in zip(rect_specs, rect_bboxes, strict=True):
        r0, c0, r1, c1 = bbox
        color = int(spec["color"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        for hr, hc in spec["hints"]:
            grid[r0 + hr, c0 + hc] = BG
        for sr, sc in spec["stamp"]:
            out[r0 + sr, c0 + sc] = BG

    return grid, out


def _gen_attempt() -> np.ndarray | None:
    template = _random_connected_template()
    if template is None:
        return None
    rotations = _rotations(template)
    rect_specs = _sample_rectangle_specs(rotations)
    if rect_specs is None:
        return None
    placed = _place_objects(template.shape, rect_specs)
    if placed is None:
        return None
    h, w, template_bbox, rect_bboxes = placed
    inp, expected = _render_case(template, rect_specs, template_bbox, rect_bboxes, (h, w))
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-placement puzzle")
