from __future__ import annotations

from collections import Counter, deque
from itertools import permutations
import math
import random

import numpy as np


LEGEND_COLOR = 5
SHAPE_COLOR = 7
DIRECTIONS = ("up", "right", "down", "left")
DIR_TO_VEC = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
PALETTE = [0, 1, 2, 3, 4, 6, 8, 9]
TEMPLATES = [
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
]
MAX_GENERATION_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _rotate(arr: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(arr, k=-(turns_cw % 4))


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _normalize_points(
    points: list[tuple[float, float]],
    box: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    r0, c0, r1, c1 = box
    rh = max(1, r1 - r0)
    cw = max(1, c1 - c0)
    return [((r - r0) / rh, (c - c0) / cw) for r, c in points]


def _assignment_cost(
    shape_points: list[tuple[float, float]],
    mark_points: list[tuple[float, float]],
    perm: tuple[int, ...],
    *,
    squared: bool,
) -> float:
    total = 0.0
    for idx, mark_idx in enumerate(perm):
        sr, sc = shape_points[idx]
        mr, mc = mark_points[mark_idx]
        dr = sr - mr
        dc = sc - mc
        if squared:
            total += dr * dr + dc * dc
        else:
            total += abs(dr) + abs(dc)
    return total


def _best_assignment(
    shape_points: list[tuple[float, float]],
    mark_points: list[tuple[float, float]],
    *,
    squared: bool,
) -> tuple[tuple[int, ...], float, float]:
    best_perm: tuple[int, ...] | None = None
    best_cost = float("inf")
    second_cost = float("inf")
    for perm in permutations(range(len(mark_points))):
        cost = _assignment_cost(shape_points, mark_points, perm, squared=squared)
        if cost < best_cost - 1e-9:
            second_cost = best_cost
            best_cost = cost
            best_perm = perm
        elif abs(cost - best_cost) <= 1e-9:
            second_cost = best_cost
            best_perm = None
        elif cost < second_cost:
            second_cost = cost
    assert best_perm is not None, "layout pairing is ambiguous"
    return best_perm, best_cost, second_cost


def _orientation_from_mask(mask: np.ndarray) -> str:
    counts = {
        "up": int(mask[0, :].sum()),
        "down": int(mask[-1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, -1].sum()),
    }
    tips = [name for name, count in counts.items() if count == 1]
    assert len(tips) == 1, f"expected one tip side, got {counts}"
    return tips[0]


def _rotation_needed(current: str, target: str) -> int:
    return (DIRECTIONS.index(target) - DIRECTIONS.index(current)) % 4


def _find_legend_box(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    mask = (grid != int(bg)) & (grid != SHAPE_COLOR)
    box = bbox(mask)
    assert box is not None, "legend not found"
    r0, c0, r1, c1 = box
    crop = crop_bbox(grid, box)
    assert np.all(crop[0, :] == LEGEND_COLOR)
    assert np.all(crop[-1, :] == LEGEND_COLOR)
    assert np.all(crop[:, 0] == LEGEND_COLOR)
    assert np.all(crop[:, -1] == LEGEND_COLOR)
    return r0, c0, r1, c1


def _extract_marks(
    grid: np.ndarray,
    legend_box: tuple[int, int, int, int],
    bg: int,
) -> list[dict]:
    r0, c0, r1, c1 = legend_box
    crop = crop_bbox(grid, legend_box)
    center = ((r1 - r0) // 2, (c1 - c0) // 2)
    marks: list[dict] = []
    for rr, cc in np.argwhere((crop != LEGEND_COLOR) & (crop != bg)):
        rr = int(rr)
        cc = int(cc)
        if rr == center[0]:
            direction = "left" if cc < center[1] else "right"
        else:
            assert cc == center[1], "mark must be on a cardinal axis"
            direction = "up" if rr < center[0] else "down"
        assert (rr, cc) != center
        marks.append(
            {
                "pos_abs": (r0 + rr, c0 + cc),
                "pos_rel": (rr, cc),
                "color": int(crop[rr, cc]),
                "direction": direction,
            }
        )
    assert 2 <= len(marks) <= 4
    assert len({m["direction"] for m in marks}) == len(marks)
    assert len({m["color"] for m in marks}) == len(marks)
    return sorted(marks, key=lambda m: (m["pos_abs"][0], m["pos_abs"][1]))


def _extract_shapes(
    grid: np.ndarray,
    legend_box: tuple[int, int, int, int],
    bg: int,
) -> list[dict]:
    shapes: list[dict] = []
    for coords in components_mask(grid == SHAPE_COLOR, connectivity=4):
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        inside_legend = not (r1 < legend_box[0] or legend_box[2] < r0 or c1 < legend_box[1] or legend_box[3] < c0)
        assert not inside_legend, "shape overlaps legend"
        crop = crop_bbox(grid, box)
        mask = crop != bg
        assert crop.shape[0] == crop.shape[1], "shape bbox must be square"
        shapes.append(
            {
                "bbox": box,
                "crop": crop,
                "mask": mask,
                "orientation": _orientation_from_mask(mask),
            }
        )
    shapes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert 2 <= len(shapes) <= 4
    return shapes


def _match_shapes_to_marks(
    shapes: list[dict],
    marks: list[dict],
    legend_box: tuple[int, int, int, int],
) -> tuple[int, ...]:
    r0 = min(shape["bbox"][0] for shape in shapes)
    c0 = min(shape["bbox"][1] for shape in shapes)
    r1 = max(shape["bbox"][2] for shape in shapes)
    c1 = max(shape["bbox"][3] for shape in shapes)
    shape_box = (r0, c0, r1, c1)

    shape_points = _normalize_points([_bbox_center(shape["bbox"]) for shape in shapes], shape_box)
    mark_points = _normalize_points([mark["pos_abs"] for mark in marks], legend_box)
    perm, _best, _second = _best_assignment(shape_points, mark_points, squared=False)
    return perm


def _render_output_from_parsed(
    grid: np.ndarray,
    shapes: list[dict],
    marks: list[dict],
    assignment: tuple[int, ...],
    bg: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for shape_idx, mark_idx in enumerate(assignment):
        shape = shapes[shape_idx]
        mark = marks[mark_idx]
        turns = _rotation_needed(shape["orientation"], mark["direction"])
        rotated_mask = _rotate(shape["mask"], turns)
        colored_patch = _make_patch(rotated_mask, mark["color"])
        r0, c0, r1, c1 = shape["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        paste(out, colored_patch, r0, c0, transparent=0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each orange placeholder shape with the corresponding legend mark, then rotate and recolor it.

    Input structure:
    - A dominant background contains one solid gray legend rectangle whose interior holds several colored single-pixel marks.
    - Every mark lies exactly above, below, left, or right of the legend center.
    - Outside the legend are equally many orange connected shapes with square tight bboxes and one asymmetric one-pixel tip.

    Rule:
    - Treat the union bbox of all orange shapes as one layout frame and the legend bbox as the second layout frame.
    - Match shapes to marks by the unique minimum total distance between their normalized relative positions inside those two frames.
    - For each matched pair, rotate the shape around its own bbox center so its tip points toward the matched mark direction, recolor the whole shape to the mark color, and write it back into the same bbox.

    Key insight:
    - The legend gives two kinds of instructions at once: mark positions decide which shape matches which mark, while each mark's direction and color decide the rotation target and the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    legend_box = _find_legend_box(g, bg)
    marks = _extract_marks(g, legend_box, bg)
    shapes = _extract_shapes(g, legend_box, bg)
    assert len(shapes) == len(marks), "shape count must equal mark count"
    assignment = _match_shapes_to_marks(shapes, marks, legend_box)
    return _render_output_from_parsed(g, shapes, marks, assignment, bg)


def _legend_mark_positions(h: int, w: int) -> dict[str, tuple[int, int]]:
    cr = h // 2
    cc = w // 2
    return {
        "up": (cr - max(1, h // 2 - 1), cc),
        "down": (cr + max(1, h // 2 - 1), cc),
        "left": (cr, cc - max(1, w // 2 - 1)),
        "right": (cr, cc + max(1, w // 2 - 1)),
    }


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _choose_canvas(mark_count: int, shape_size: int, legend_h: int, legend_w: int) -> tuple[int, int]:
    if mark_count == 4:
        min_dim = max(24, 3 * shape_size + 4, max(legend_h, legend_w) + 8)
    elif mark_count == 3:
        min_dim = max(20, 2 * shape_size + 6, max(legend_h, legend_w) + 6)
    else:
        min_dim = max(18, 2 * shape_size + 4, max(legend_h, legend_w) + 4)
    min_h = min(30, min_dim)
    min_w = min(30, min_dim)
    return random.randint(min_h, 30), random.randint(min_w, 30)


def _plan_shape_offsets(
    dirs: list[str],
    shape_size: int,
    canvas_h: int,
    canvas_w: int,
    legend_box: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    gap = random.randint(2, 3)
    step = shape_size + gap
    base = {
        "up": (0, step),
        "down": (2 * step, step),
        "left": (step, 0),
        "right": (step, 2 * step),
    }

    raw_boxes = []
    for d in dirs:
        rr, cc = base[d]
        if d in ("up", "down"):
            cc += random.randint(-1, 1)
        else:
            rr += random.randint(-1, 1)
        raw_boxes.append((rr, cc, rr + shape_size - 1, cc + shape_size - 1))

    min_r = min(box[0] for box in raw_boxes)
    min_c = min(box[1] for box in raw_boxes)
    all_boxes = [
        (box[0] - min_r, box[1] - min_c, box[2] - min_r, box[3] - min_c)
        for box in raw_boxes
    ]
    layout_h = max(box[2] for box in all_boxes) + 1
    layout_w = max(box[3] for box in all_boxes) + 1
    if layout_h > canvas_h or layout_w > canvas_w:
        return None

    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            if _rects_overlap(all_boxes[i], all_boxes[j], padding=1):
                return None

    for _ in range(200):
        top = random.randint(0, canvas_h - layout_h)
        left = random.randint(0, canvas_w - layout_w)
        placed = []
        for box in all_boxes:
            rr, cc, br, bc = box
            box = (top + rr, left + cc, top + br, left + bc)
            if _rects_overlap(box, legend_box, padding=1):
                break
            placed.append((box[0], box[1]))
        else:
            return placed
    return None


def _render_case_from_meta(meta: dict) -> np.ndarray:
    out = np.array(meta["input"], copy=True)
    bg = meta["bg"]
    for shape in meta["shapes"]:
        r0, c0 = shape["top_left"]
        size = meta["template"].shape[0]
        out[r0 : r0 + size, c0 : c0 + size] = bg
        turns = _rotation_needed(shape["current_direction"], shape["target_direction"])
        mask = _rotate(meta["template"], turns)
        patch = _make_patch(mask, shape["target_color"])
        paste(out, patch, r0, c0, transparent=0)
    return out


def _render_with_permutation(meta: dict, perm: tuple[int, ...]) -> np.ndarray:
    out = np.array(meta["input"], copy=True)
    bg = meta["bg"]
    marks = meta["marks"]
    for shape_idx, shape in enumerate(meta["shapes"]):
        r0, c0 = shape["top_left"]
        size = meta["template"].shape[0]
        out[r0 : r0 + size, c0 : c0 + size] = bg
        mark = marks[perm[shape_idx]]
        turns = _rotation_needed(shape["current_direction"], mark["direction"])
        mask = _rotate(meta["template"], turns)
        patch = _make_patch(mask, mark["color"])
        paste(out, patch, r0, c0, transparent=0)
    return out


def _semantic_checks(meta: dict, output: np.ndarray) -> None:
    expected = _render_case_from_meta(meta)
    assert np.array_equal(output, expected), "solver output disagrees with generation metadata"

    legend_box = meta["legend_box"]
    shape_boxes = []
    for shape in meta["shapes"]:
        r0, c0 = shape["top_left"]
        size = meta["template"].shape[0]
        shape_boxes.append((r0, c0, r0 + size - 1, c0 + size - 1))
    shape_union = (
        min(b[0] for b in shape_boxes),
        min(b[1] for b in shape_boxes),
        max(b[2] for b in shape_boxes),
        max(b[3] for b in shape_boxes),
    )
    shape_points = _normalize_points([_bbox_center(b) for b in shape_boxes], shape_union)
    mark_points = _normalize_points([mark["abs_pos"] for mark in meta["marks"]], legend_box)

    intended = tuple(meta["shape_to_mark"])
    perm_l1, best_l1, second_l1 = _best_assignment(shape_points, mark_points, squared=False)
    perm_l2, best_l2, second_l2 = _best_assignment(shape_points, mark_points, squared=True)
    assert perm_l1 == intended
    assert perm_l2 == intended
    assert second_l1 - best_l1 > 0.08
    assert second_l2 - best_l2 > 0.01

    assert any(shape["current_direction"] != shape["target_direction"] for shape in meta["shapes"])
    assert not np.array_equal(meta["input"], expected)

    alt_outputs = set()
    for perm in permutations(range(len(meta["marks"]))):
        rendered = _render_with_permutation(meta, perm)
        if perm == intended:
            assert np.array_equal(rendered, expected)
        else:
            sig = (rendered.shape, rendered.tobytes())
            alt_outputs.add(sig)
            assert not np.array_equal(rendered, expected), "different pairings must change the output"
    assert len(alt_outputs) == math.factorial(len(meta["marks"])) - 1


def _generate_case(with_meta: bool = False):
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mark_count = random.randint(2, 4)
        template_choices = [tpl for tpl in TEMPLATES if mark_count < 4 or tpl.shape[0] <= 5]
        template = random.choice(template_choices)
        shape_size = template.shape[0]
        legend_h = random.choice([5, 7])
        legend_w = random.choice([5, 7])
        canvas_h, canvas_w = _choose_canvas(mark_count, shape_size, legend_h, legend_w)
        bg = random.choice([c for c in PALETTE if c != LEGEND_COLOR and c != SHAPE_COLOR])
        grid = np.full((canvas_h, canvas_w), bg, dtype=int)

        directions = random.sample(list(DIRECTIONS), k=mark_count)
        mark_colors = random.sample([c for c in PALETTE if c not in {bg, LEGEND_COLOR, SHAPE_COLOR}], k=mark_count)
        random.shuffle(directions)

        legend_top = random.randint(0, canvas_h - legend_h)
        legend_left = random.randint(0, canvas_w - legend_w)
        legend_box = (
            legend_top,
            legend_left,
            legend_top + legend_h - 1,
            legend_left + legend_w - 1,
        )

        shape_tops = _plan_shape_offsets(directions, shape_size, canvas_h, canvas_w, legend_box)
        if shape_tops is None:
            continue

        grid[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w] = LEGEND_COLOR
        mark_pos_map = _legend_mark_positions(legend_h, legend_w)
        marks: list[dict] = []
        for idx, (direction, color) in enumerate(zip(directions, mark_colors)):
            rr, cc = mark_pos_map[direction]
            grid[legend_top + rr, legend_left + cc] = color
            marks.append(
                {
                    "direction": direction,
                    "color": color,
                    "abs_pos": (legend_top + rr, legend_left + cc),
                    "list_index": idx,
                }
            )

        shapes: list[dict] = []
        need_rotation = False
        for shape_idx, (top, left) in enumerate(shape_tops):
            current_direction = random.choice(list(DIRECTIONS))
            target_direction = directions[shape_idx]
            need_rotation |= current_direction != target_direction
            mask = _rotate(template, _rotation_needed("up", current_direction))
            patch = _make_patch(mask, SHAPE_COLOR)
            paste(grid, patch, top, left, transparent=0)
            shapes.append(
                {
                    "top_left": (top, left),
                    "current_direction": current_direction,
                    "target_direction": target_direction,
                    "target_color": mark_colors[shape_idx],
                }
            )
        if not need_rotation:
            continue

        parsed_bg = get_bg(grid)
        if parsed_bg != bg:
            continue
        parsed_legend = _find_legend_box(grid, bg)
        parsed_marks = _extract_marks(grid, parsed_legend, bg)
        parsed_shapes = _extract_shapes(grid, parsed_legend, bg)
        if len(parsed_marks) != mark_count or len(parsed_shapes) != mark_count:
            continue

        intended_shape_to_mark = tuple(range(mark_count))
        meta = {
            "input": grid,
            "bg": bg,
            "template": template,
            "legend_box": parsed_legend,
            "marks": marks,
            "shapes": shapes,
            "shape_to_mark": intended_shape_to_mark,
        }

        out = main(grid)
        try:
            _semantic_checks(meta, out)
        except AssertionError:
            continue

        if with_meta:
            meta["expected"] = _render_case_from_meta(meta)
            return grid, meta
        return grid
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case(with_meta=False)
