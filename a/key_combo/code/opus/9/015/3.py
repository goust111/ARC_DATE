import random
from collections import Counter, defaultdict, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
MOTIF_LIBRARY = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
]
MAX_GENERATION_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0:r1 + 1, c0:c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if int(connectivity) == 4 else DIRS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 8,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    out: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in components_mask(g == int(color), connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    bbox = comp["bbox"]
    patch = crop_bbox(grid, bbox)
    return bool(np.all(patch == color))


def _center_offset(delta: int, mode: str) -> int:
    if mode == "ceil":
        return int((delta + 1) // 2)
    if mode == "floor":
        return int(delta // 2)
    raise ValueError(f"unsupported centering mode: {mode}")


def _extract_scene(grid: np.ndarray) -> tuple[tuple[int, int], int, list[dict], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = int(get_bg(g))
    assert bg == 0
    h, w = g.shape

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in find_components(g, bg=bg, connectivity=8):
        by_color[int(comp["color"])].append(comp)

    rectangles: list[dict] = []
    motifs: dict[int, np.ndarray] = {}
    for color, comps in by_color.items():
        rects = [comp for comp in comps if comp["bbox"][2] == h - 1 and _is_solid_rectangle(g, comp)]
        assert len(rects) == 1, f"expected one bottom solid rectangle for color {color}"
        rect = rects[0]
        others = [comp for comp in comps if comp is not rect]
        assert len(others) == 1, f"expected one same-color motif for color {color}"
        motif_comp = others[0]
        motif_patch = crop_bbox(g, motif_comp["bbox"])
        motif_mask = motif_patch == int(color)
        assert 2 <= int(motif_mask.sum()) <= 4
        rectangles.append(
            {
                "color": int(color),
                "input_bbox": tuple(map(int, rect["bbox"])),
                "h": bbox_h(rect["bbox"]),
                "w": bbox_w(rect["bbox"]),
            }
        )
        motifs[int(color)] = motif_mask

    rectangles.sort(key=lambda item: (item["input_bbox"][1], item["input_bbox"][0]))
    left_edges = [int(item["input_bbox"][1]) for item in rectangles]
    assert len(rectangles) in (3, 4)
    assert len(left_edges) == len(set(left_edges))
    return (int(h), int(w)), int(bg), rectangles, motifs


def _place_rectangles(
    shape: tuple[int, int],
    rectangles: list[dict],
    motifs: dict[int, np.ndarray],
    *,
    order: list[int] | None = None,
    center_mode: str = "ceil",
) -> tuple[list[dict], int]:
    h, w = map(int, shape)
    order = list(range(len(rectangles))) if order is None else list(order)
    assert order
    offset = int(rectangles[order[0]]["w"] - 2)
    assert offset > 0

    placed: list[dict] = []
    for layer_idx, rect_idx in enumerate(order):
        rect = rectangles[rect_idx]
        color = int(rect["color"])
        rect_h = int(rect["h"])
        rect_w = int(rect["w"])
        top = 0
        left = int(layer_idx * offset)
        assert top + rect_h <= h and left + rect_w <= w

        motif = np.asarray(motifs[color], dtype=bool)
        motif_h, motif_w = map(int, motif.shape)
        assert motif_h <= rect_h and motif_w <= rect_w
        off_r = _center_offset(rect_h - motif_h, center_mode)
        off_c = _center_offset(rect_w - motif_w, center_mode)
        holes = [
            (int(top + off_r + rr), int(left + off_c + cc))
            for rr, cc in np.argwhere(motif)
        ]
        placed.append(
            {
                "color": color,
                "h": rect_h,
                "w": rect_w,
                "top": top,
                "left": left,
                "holes": holes,
            }
        )
    return placed, offset


def _build_stacks(shape: tuple[int, int], placed: list[dict]) -> list[list[list[int]]]:
    h, w = map(int, shape)
    stacks = [[[] for _ in range(w)] for _ in range(h)]
    for rect in placed:
        color = int(rect["color"])
        for r in range(int(rect["top"]), int(rect["top"]) + int(rect["h"])):
            for c in range(int(rect["left"]), int(rect["left"]) + int(rect["w"])):
                stacks[r][c].append(color)
    return stacks


def _copy_stacks(stacks: list[list[list[int]]]) -> list[list[list[int]]]:
    return [[cell.copy() for cell in row] for row in stacks]


def _stacks_to_grid(stacks: list[list[list[int]]], bg: int = 0) -> np.ndarray:
    h = len(stacks)
    w = len(stacks[0]) if h else 0
    out = np.full((h, w), int(bg), dtype=int)
    for r in range(h):
        for c in range(w):
            if stacks[r][c]:
                out[r, c] = int(stacks[r][c][-1])
    return out


def _render_scene(
    shape: tuple[int, int],
    rectangles: list[dict],
    motifs: dict[int, np.ndarray],
    *,
    order: list[int] | None = None,
    center_mode: str = "ceil",
    hole_mode: str = "expose",
    skip_hole_colors: set[int] | None = None,
) -> tuple[np.ndarray, list[dict], list[list[list[int]]]]:
    placed, _offset = _place_rectangles(shape, rectangles, motifs, order=order, center_mode=center_mode)
    base_stacks = _build_stacks(shape, placed)
    skip_hole_colors = set() if skip_hole_colors is None else set(map(int, skip_hole_colors))

    if hole_mode == "expose":
        stacks = _copy_stacks(base_stacks)
        for rect in placed:
            color = int(rect["color"])
            if color in skip_hole_colors:
                continue
            for r, c in rect["holes"]:
                cell = stacks[r][c]
                if color in cell:
                    cell.remove(color)
        return _stacks_to_grid(stacks, bg=0), placed, base_stacks

    if hole_mode == "background":
        out = _stacks_to_grid(base_stacks, bg=0)
        for rect in placed:
            color = int(rect["color"])
            if color in skip_hole_colors:
                continue
            for r, c in rect["holes"]:
                out[r, c] = 0
        return out, placed, base_stacks

    raise ValueError(f"unsupported hole mode: {hole_mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack the bottom rectangles at the top and punch same-colored motif holes that reveal lower layers.

    Input structure:
    - Black background with 3-4 large solid monochrome rectangles touching the bottom edge.
    - Each rectangle color also appears once as a small disconnected motif in the upper area.
    - Rectangle left edges are all distinct, and the motifs act as same-color stencils.

    Rule:
    - Identify the bottom solid rectangles and sort them by their input left edge.
    - Clear the canvas, then place the sorted rectangles at row 0 with horizontal offset `first_width - 2`, so later rectangles overlap earlier ones and later placements cover earlier ones.
    - For each placed rectangle, center its same-colored motif inside the placed bbox; when exact centering is impossible, bias the motif one cell down and right.
    - Punch holes at those motif cells by removing only that rectangle's color layer at each pixel; if lower rectangles remain there, their color becomes visible, otherwise the pixel becomes black.

    Key insight:
    - The carved cells do not simply become background: they expose the next visible layer in the overlap stack, so the solver must combine left-edge stacking order with same-color motif carving.
    """
    shape, _bg, rectangles, motifs = _extract_scene(np.asarray(grid, dtype=int))
    out, _placed, _stacks = _render_scene(shape, rectangles, motifs, center_mode="ceil", hole_mode="expose")
    return out


def _random_motif() -> np.ndarray:
    motif = random.choice(MOTIF_LIBRARY).copy()
    for _ in range(random.randrange(4)):
        motif = np.rot90(motif)
    if random.choice([False, True]):
        motif = np.fliplr(motif)
    return np.asarray(motif, dtype=bool)


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _build_input(shape: tuple[int, int], rectangles: list[dict], motif_placements: list[dict]) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["input_bbox"]
        grid[r0:r1 + 1, c0:c1 + 1] = int(rect["color"])
    for motif in motif_placements:
        r0, c0 = int(motif["top"]), int(motif["left"])
        mask = np.asarray(motif["mask"], dtype=bool)
        color = int(motif["color"])
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = color
    return grid


def _scene_has_bias_case(rectangles: list[dict], motifs: dict[int, np.ndarray]) -> bool:
    for rect in rectangles:
        mask = motifs[int(rect["color"])]
        if (int(rect["h"]) - int(mask.shape[0])) % 2 == 1:
            return True
        if (int(rect["w"]) - int(mask.shape[1])) % 2 == 1:
            return True
    return False


def _semantic_checks(shape: tuple[int, int], rectangles: list[dict], motifs: dict[int, np.ndarray], inp: np.ndarray) -> bool:
    expected, placed, base_stacks = _render_scene(shape, rectangles, motifs, center_mode="ceil", hole_mode="expose")
    if not np.array_equal(main(inp), expected):
        return False

    wrong_bg, _placed_bg, _stacks_bg = _render_scene(shape, rectangles, motifs, center_mode="ceil", hole_mode="background")
    if np.array_equal(expected, wrong_bg):
        return False

    if not _scene_has_bias_case(rectangles, motifs):
        return False
    wrong_floor, _placed_floor, _stacks_floor = _render_scene(shape, rectangles, motifs, center_mode="floor", hole_mode="expose")
    if np.array_equal(expected, wrong_floor):
        return False

    try:
        wrong_reverse, _placed_rev, _stacks_rev = _render_scene(
            shape,
            rectangles,
            motifs,
            order=list(reversed(range(len(rectangles)))),
            center_mode="ceil",
            hole_mode="expose",
        )
    except AssertionError:
        wrong_reverse = None
    if wrong_reverse is not None and np.array_equal(expected, wrong_reverse):
        return False

    for rect in rectangles:
        color = int(rect["color"])
        without_color_holes, _placed_skip, _stacks_skip = _render_scene(
            shape,
            rectangles,
            motifs,
            center_mode="ceil",
            hole_mode="expose",
            skip_hole_colors={color},
        )
        if np.array_equal(expected, without_color_holes):
            return False

    reveal_lower = False
    reveal_background = False
    for rect in placed:
        color = int(rect["color"])
        for r, c in rect["holes"]:
            stack_before = base_stacks[r][c]
            if stack_before and stack_before[-1] == color:
                final_color = int(expected[r, c])
                if final_color == 0:
                    reveal_background = True
                elif final_color != color:
                    reveal_lower = True

    return reveal_lower and reveal_background


def _gen_attempt() -> np.ndarray | None:
    num_rects = random.choice([3, 4])
    base_w = random.choice([5, 6])
    widths = [base_w]
    for _ in range(num_rects - 1):
        widths.append(random.choice([base_w, min(base_w + 1, 7)]))
    heights = [random.randint(5, 7) for _ in range(num_rects)]
    colors = random.sample([1, 2, 3, 4], num_rects)

    gaps = [random.randint(1, 3) for _ in range(num_rects - 1)]
    left_margin = random.randint(0, 2)
    total_width = left_margin + sum(widths) + sum(gaps)
    if total_width > 28:
        return None
    width = random.randint(max(18, total_width), min(30, total_width + 3))

    max_height = max(heights)
    height = random.randint(max(14, max_height + 5), 20)

    rectangles: list[dict] = []
    cursor = left_margin
    for idx in range(num_rects):
        rect_w = int(widths[idx])
        rect_h = int(heights[idx])
        r0 = int(height - rect_h)
        c0 = int(cursor)
        rectangles.append(
            {
                "color": int(colors[idx]),
                "input_bbox": (r0, c0, height - 1, c0 + rect_w - 1),
                "h": rect_h,
                "w": rect_w,
            }
        )
        cursor += rect_w
        if idx + 1 < num_rects:
            cursor += gaps[idx]

    motif_masks = {int(rect["color"]): _random_motif() for rect in rectangles}
    min_rect_top = min(int(rect["input_bbox"][0]) for rect in rectangles)
    motif_placements: list[dict] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    for rect in random.sample(rectangles, len(rectangles)):
        color = int(rect["color"])
        mask = motif_masks[color]
        mh, mw = map(int, mask.shape)
        max_top = int(min_rect_top - mh - 1)
        if max_top < 0 or mw > width:
            return None
        placed = False
        for _ in range(80):
            top = random.randint(0, max_top)
            left = random.randint(0, width - mw)
            box = (top, left, top + mh - 1, left + mw - 1)
            if any(_bbox_overlaps(box, other, pad=1) for other in used_boxes):
                continue
            motif_placements.append({"color": color, "mask": mask, "top": top, "left": left})
            used_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    rectangles.sort(key=lambda item: item["input_bbox"][1])
    inp = _build_input((height, width), rectangles, motif_placements)
    if not _semantic_checks((height, width), rectangles, motif_masks, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacking-hole input")
