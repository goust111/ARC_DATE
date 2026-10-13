from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


TILE_COLORS = (1, 3)
RECT_COLORS = (2, 4, 5, 6, 7, 8, 9)
CORNERS = ("TL", "TR", "BL", "BR")
GEN_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fill_bbox(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _corner_from_bbox(box: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = box
    touches_top = r0 == 0
    touches_bottom = r1 == h - 1
    touches_left = c0 == 0
    touches_right = c1 == w - 1
    assert touches_top != touches_bottom
    assert touches_left != touches_right
    if touches_top and touches_left:
        return "TL"
    if touches_top and touches_right:
        return "TR"
    if touches_bottom and touches_left:
        return "BL"
    return "BR"


def _shift_bbox(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    corner = _corner_from_bbox(box, shape)
    dr = height if corner[0] == "T" else -height
    dc = width if corner[1] == "L" else -width
    nr0, nc0, nr1, nc1 = r0 + dr, c0 + dc, r1 + dr, c1 + dc
    assert 0 <= nr0 <= nr1 < h
    assert 0 <= nc0 <= nc1 < w
    return nr0, nc0, nr1, nc1


def _find_corner_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.ravel()))
    tile_colors = {color for color, _ in counts.most_common(2)}

    rects: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - tile_colors):
        for coords in _components_of_color(g, color):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            area = int(coords.shape[0])
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            if area != bh * bw:
                continue
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            touches_tb = (r0 == 0) ^ (r1 == h - 1)
            touches_lr = (c0 == 0) ^ (c1 == w - 1)
            if not (touches_tb and touches_lr):
                continue
            rects.append({"color": int(color), "bbox": box, "coords": coords})

    assert len(rects) in (2, 3)
    return rects


def _infer_template(grid: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    observed = np.argwhere(visible_mask)
    assert observed.size > 0

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for ph in range(1, min(3, g.shape[0]) + 1):
        for pw in range(1, min(3, g.shape[1]) + 1):
            templ = np.full((ph, pw), -1, dtype=int)
            ok = True
            for r, c in observed:
                rr, cc = int(r % ph), int(c % pw)
                val = int(g[r, c])
                if templ[rr, cc] == -1:
                    templ[rr, cc] = val
                elif templ[rr, cc] != val:
                    ok = False
                    break
            if ok and np.all(templ != -1):
                candidates.append((ph * pw, ph, pw, templ))

    assert candidates
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_area = candidates[0][0]
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1
    return best[0][3]


def _tile_grid(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rr = np.arange(shape[0]) % template.shape[0]
    cc = np.arange(shape[1]) % template.shape[1]
    return template[np.ix_(rr, cc)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the repeated corner rectangles inward, then restore the vacated holes with the inferred repeating tile background.

    Input structure:
    - The grid has a full-background repeating 2D tile made from blue and green cells.
    - Two or three solid monochrome rectangles sit flush against grid corners and cover the tile beneath them.
    - Exactly one foreground color appears on two separate corner rectangles; every other rectangle color appears once.

    Rule:
    - Identify the repeated-color corner rectangles and translate each one diagonally inward by its own height and width.
    - Keep singleton-color rectangles fixed at their original corner positions.
    - Infer the smallest repeating background tile from cells that remain visible after excluding current rectangles and translated destinations.
    - Re-tile the whole grid with that inferred pattern, then render the fixed rectangles and translated repeated-color rectangles on top.

    Key insight:
    - The repeated color determines which corner rectangles move, while the background must be reconstructed from only those tile pixels that stay visible once both the old holes and the new covered locations are treated as unknown.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _find_corner_rectangles(g)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for rect in rects:
        by_color[int(rect["color"])].append(rect)

    moving_colors = [color for color, comps in by_color.items() if len(comps) == 2]
    assert len(moving_colors) == 1
    moving_color = int(moving_colors[0])
    assert sum(len(comps) for comps in by_color.values()) == len(rects)

    unknown_mask = np.zeros(g.shape, dtype=bool)
    final_rects: list[tuple[int, tuple[int, int, int, int]]] = []
    for rect in rects:
        color = int(rect["color"])
        box = rect["bbox"]
        _fill_bbox(unknown_mask, box)
        if color == moving_color:
            target_box = _shift_bbox(box, g.shape)
            _fill_bbox(unknown_mask, target_box)
            final_rects.append((color, target_box))
        else:
            final_rects.append((color, box))
    visible_mask = ~unknown_mask

    template = _infer_template(g, visible_mask)
    out = _tile_grid(template, g.shape)
    for color, box in final_rects:
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    assert np.array_equal(out[visible_mask], g[visible_mask])
    return out


def _template_has_exact_period(template: np.ndarray) -> bool:
    th, tw = template.shape
    for ph in range(1, th + 1):
        for pw in range(1, tw + 1):
            rr = np.arange(th) % ph
            cc = np.arange(tw) % pw
            tiled = template[:ph, :pw][np.ix_(rr, cc)]
            if np.array_equal(tiled, template):
                return (ph, pw) == (th, tw)
    return True


def _sample_template() -> np.ndarray:
    side = random.choice((2, 3))
    while True:
        template = np.array(
            [[random.choice(TILE_COLORS) for _ in range(side)] for _ in range(side)],
            dtype=int,
        )
        values = set(map(int, template.ravel()))
        if values != set(TILE_COLORS):
            continue
        if side == 3:
            counts = Counter(map(int, template.ravel()))
            if min(counts.values()) < 2:
                continue
        if _template_has_exact_period(template):
            return template


def _render_background(template: np.ndarray, shape: tuple[int, int], phase_r: int, phase_c: int) -> np.ndarray:
    th, tw = template.shape
    out = np.empty(shape, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            out[r, c] = int(template[(r + phase_r) % th, (c + phase_c) % tw])
    return out


def _corner_box(corner: str, rh: int, rw: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = 0 if corner[0] == "T" else h - rh
    c0 = 0 if corner[1] == "L" else w - rw
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _all_boxes_disjoint(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            if _box_overlap(box_a, box_b):
                return False
    return True


def _all_residues_visible(mask: np.ndarray, ph: int, pw: int) -> bool:
    for r in range(ph):
        for c in range(pw):
            if not np.any(mask[r::ph, c::pw]):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    template = _sample_template()
    th, tw = template.shape
    phase_r = random.randrange(th)
    phase_c = random.randrange(tw)

    rect_count = random.choice((2, 3))
    moving_corners = random.sample(CORNERS, 2)
    corners = list(moving_corners)
    if rect_count == 3:
        corners.append(random.choice([corner for corner in CORNERS if corner not in moving_corners]))

    sizes = {corner: (random.randint(2, 5), random.randint(2, 5)) for corner in corners}
    current_boxes = {corner: _corner_box(corner, *sizes[corner], (h, w)) for corner in corners}
    final_boxes = {}
    for corner in corners:
        if corner in moving_corners:
            final_boxes[corner] = _shift_bbox(current_boxes[corner], (h, w))
        else:
            final_boxes[corner] = current_boxes[corner]

    if not _all_boxes_disjoint(list(current_boxes.values())):
        return None
    if not _all_boxes_disjoint(list(final_boxes.values())):
        return None

    bg = _render_background(template, (h, w), phase_r, phase_c)
    inp = bg.copy()
    out = bg.copy()

    colors = random.sample(RECT_COLORS, rect_count - 1 if rect_count == 3 else 1)
    moving_color = colors[0]
    singleton_colors = colors[1:]
    corner_colors = {corner: moving_color for corner in moving_corners}
    for corner, color in zip([c for c in corners if c not in moving_corners], singleton_colors):
        corner_colors[corner] = color

    for corner in corners:
        r0, c0, r1, c1 = current_boxes[corner]
        inp[r0 : r1 + 1, c0 : c1 + 1] = corner_colors[corner]
    for corner in corners:
        r0, c0, r1, c1 = final_boxes[corner]
        out[r0 : r1 + 1, c0 : c1 + 1] = corner_colors[corner]

    unknown = np.zeros((h, w), dtype=bool)
    for corner in corners:
        _fill_bbox(unknown, current_boxes[corner])
        if corner in moving_corners:
            _fill_bbox(unknown, final_boxes[corner])
    visible = ~unknown
    if not _all_residues_visible(visible, th, tw):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(pred, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
