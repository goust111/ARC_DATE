from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import find_components, get_bg  # noqa: E402


DIR_TO_SIDE = {
    "up": "top",
    "down": "bottom",
    "left": "left",
    "right": "right",
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
ROT90_DIR = {"up": "right", "right": "down", "down": "left", "left": "up"}
FLIP_H_DIR = {"up": "up", "down": "down", "left": "right", "right": "left"}
FLIP_V_DIR = {"up": "down", "down": "up", "left": "left", "right": "right"}
PALETTE = list(range(1, 10))
GEN_RETRIES = 400


def _edge_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], border: int, fill: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(fill)
    grid[r0, c0 : c1 + 1] = int(border)
    grid[r1, c0 : c1 + 1] = int(border)
    grid[r0 : r1 + 1, c0] = int(border)
    grid[r0 : r1 + 1, c1] = int(border)


def _bbox_from_rcwh(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _transform_point(pt: tuple[int, int], shape: tuple[int, int], op: str) -> tuple[int, int]:
    r, c = map(int, pt)
    h, w = map(int, shape)
    if op == "id":
        return r, c
    if op == "rot90":
        return c, h - 1 - r
    if op == "rot180":
        return h - 1 - r, w - 1 - c
    if op == "rot270":
        return w - 1 - c, r
    if op == "flip_h":
        return r, w - 1 - c
    if op == "flip_v":
        return h - 1 - r, c
    raise ValueError(op)


def _transform_shape(shape: tuple[int, int], op: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if op in {"rot90", "rot270"}:
        return w, h
    return h, w


def _transform_dir(direction: str, op: str) -> str:
    if op == "id":
        return direction
    if op == "rot90":
        return ROT90_DIR[direction]
    if op == "rot180":
        return ROT90_DIR[ROT90_DIR[direction]]
    if op == "rot270":
        return ROT90_DIR[ROT90_DIR[ROT90_DIR[direction]]]
    if op == "flip_h":
        return FLIP_H_DIR[direction]
    if op == "flip_v":
        return FLIP_V_DIR[direction]
    raise ValueError(op)


def _transform_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], op: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts = [_transform_point(pt, shape, op) for pt in corners]
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _transform_specs(specs: list[dict], shape: tuple[int, int], op: str) -> tuple[list[dict], tuple[int, int]]:
    new_shape = _transform_shape(shape, op)
    out = []
    for spec in specs:
        out.append(
            {
                "bbox": _transform_bbox(spec["bbox"], shape, op),
                "marker": _transform_point(spec["marker"], shape, op),
                "direction": _transform_dir(spec["direction"], op),
                "border": int(spec["border"]),
                "fill": int(spec["fill"]),
            }
        )
    return out, new_shape


def _classify_rect_component(
    grid: np.ndarray,
    comp: dict,
    bg: int,
) -> dict | None:
    bbox = tuple(map(int, comp["bbox"]))
    r0, c0, r1, c1 = bbox
    h, w = _bbox_hw(bbox)
    if h < 3 or w < 3:
        return None
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    edge = _edge_mask(h, w)
    border = int(comp["color"])
    if not np.all(patch[edge] == border):
        return None
    inner = patch[1:-1, 1:-1]
    inner_colors = {int(v) for v in inner.ravel()}
    if len(inner_colors) != 1:
        return None
    fill = inner_colors.pop()
    if fill == bg or fill == border:
        return None
    if not np.all(patch[~edge] == fill):
        return None
    return {
        "bbox": bbox,
        "border": border,
        "fill": fill,
        "marker": None,
        "direction": None,
        "budget": int(inner.size),
    }


def _aligned_direction(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    mr, mc = map(int, marker)
    r0, c0, r1, c1 = map(int, bbox)
    matches = []
    if mr < r0 and c0 <= mc <= c1:
        matches.append("up")
    if mr > r1 and c0 <= mc <= c1:
        matches.append("down")
    if mc < c0 and r0 <= mr <= r1:
        matches.append("left")
    if mc > c1 and r0 <= mr <= r1:
        matches.append("right")
    if len(matches) != 1:
        return None
    return matches[0]


def _parse_problem(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    rects: list[dict] = []
    used_ids: set[int] = set()

    for color, color_comps in by_color.items():
        rect_candidates = []
        markers = []
        for comp in color_comps:
            classified = _classify_rect_component(g, comp, bg)
            if classified is not None:
                rect_candidates.append(classified)
            elif int(comp["area"]) == 1:
                coord = tuple(map(int, comp["coords"][0]))
                markers.append(coord)
        for rect in rect_candidates:
            aligned = [(m, _aligned_direction(m, rect["bbox"])) for m in markers]
            aligned = [(m, d) for m, d in aligned if d is not None]
            assert len(aligned) == 1, f"border color {color} must have one aligned marker"
            rect["marker"], rect["direction"] = aligned[0]
            rects.append(rect)
        if rect_candidates:
            assert len(rect_candidates) == 1, f"border color {color} should define one rectangle"
            assert len(markers) == 1, f"border color {color} should have one marker"
            used_ids.add(color)

    assert rects, "expected at least one two-color rectangle"
    assert len(rects) >= 2, "expected multiple rectangles"
    assert len({rect["border"] for rect in rects}) == len(rects), "border colors must be unique"
    return bg, rects


def _can_expand(
    bbox: tuple[int, int, int, int],
    side: str,
    obstacles: np.ndarray,
) -> bool:
    h, w = obstacles.shape
    r0, c0, r1, c1 = map(int, bbox)
    if side == "up":
        if r0 == 0:
            return False
        return not bool(np.any(obstacles[r0 - 1, c0 : c1 + 1]))
    if side == "down":
        if r1 == h - 1:
            return False
        return not bool(np.any(obstacles[r1 + 1, c0 : c1 + 1]))
    if side == "left":
        if c0 == 0:
            return False
        return not bool(np.any(obstacles[r0 : r1 + 1, c0 - 1]))
    if side == "right":
        if c1 == w - 1:
            return False
        return not bool(np.any(obstacles[r0 : r1 + 1, c1 + 1]))
    raise ValueError(side)


def _expand_once(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if side == "up":
        return r0 - 1, c0, r1, c1
    if side == "down":
        return r0, c0, r1 + 1, c1
    if side == "left":
        return r0, c0 - 1, r1, c1
    if side == "right":
        return r0, c0, r1, c1 + 1
    raise ValueError(side)


def _stretch_bbox(
    bbox: tuple[int, int, int, int],
    direction: str,
    budget: int,
    obstacles: np.ndarray,
) -> tuple[tuple[int, int, int, int], dict]:
    primary = str(direction)
    secondary = OPPOSITE[primary]
    remaining = int(budget)
    current = tuple(map(int, bbox))
    grew = {primary: 0, secondary: 0}

    while remaining > 0 and _can_expand(current, primary, obstacles):
        current = _expand_once(current, primary)
        grew[primary] += 1
        remaining -= 1

    while remaining > 0 and _can_expand(current, secondary, obstacles):
        current = _expand_once(current, secondary)
        grew[secondary] += 1
        remaining -= 1

    info = {
        "direction": primary,
        "primary_growth": int(grew[primary]),
        "secondary_growth": int(grew[secondary]),
        "used_budget": int(budget - remaining),
        "unused_budget": int(remaining),
        "bounced": bool(grew[secondary] > 0),
        "moved": bool(current != bbox),
    }
    return current, info


def _solve_specs(shape: tuple[int, int], specs: list[dict], bg: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    h, w = map(int, shape)
    grid = np.full((h, w), int(bg), dtype=int)
    for spec in specs:
        _paint_rect(grid, spec["bbox"], spec["border"], spec["fill"])
        mr, mc = map(int, spec["marker"])
        assert grid[mr, mc] == bg
        grid[mr, mc] = int(spec["border"])

    occupied = grid != bg
    out = np.full_like(grid, bg)
    analyses = []
    claimed = np.zeros_like(grid, dtype=bool)

    for spec in specs:
        self_mask = np.zeros_like(grid, dtype=bool)
        r0, c0, r1, c1 = map(int, spec["bbox"])
        self_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        mr, mc = map(int, spec["marker"])
        self_mask[mr, mc] = True
        final_bbox, info = _stretch_bbox(
            spec["bbox"],
            spec["direction"],
            (r1 - r0 - 1) * (c1 - c0 - 1),
            occupied & ~self_mask,
        )
        fr0, fc0, fr1, fc1 = map(int, final_bbox)
        patch_mask = np.zeros_like(grid, dtype=bool)
        patch_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
        assert not np.any(claimed & patch_mask), "generated outputs must stay non-overlapping"
        claimed |= patch_mask
        _paint_rect(out, final_bbox, spec["border"], spec["fill"])
        analyses.append(
            {
                "initial_bbox": tuple(map(int, spec["bbox"])),
                "final_bbox": final_bbox,
                "marker": tuple(map(int, spec["marker"])),
                "border": int(spec["border"]),
                "fill": int(spec["fill"]),
                **info,
            }
        )

    return grid, out, analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each framed rectangle by its interior area, bouncing leftover growth to the opposite side when blocked.

    Input structure
    - The grid has a dominant background and several two-color filled rectangles whose border colors are all distinct.
    - Each rectangle has a lone marker pixel of the same border color placed somewhere on one side's extended row or column, indicating only the stretch direction.
    - Other rectangles may sit in that direction and act as blockers, and some rectangles may start close to a grid edge.

    Rule
    - For each rectangle, compute a stretch budget equal to the number of interior fill pixels inside its border.
    - Expand one whole row or column at a time along the marked direction; when the next step would hit another input object or leave the grid, send all remaining budget to the opposite side.
    - If both sides are blocked, stop and discard leftover budget; then redraw the expanded rectangle with the same border and fill colors, and remove every marker pixel.

    Key insight
    - The marker chooses only the first side, while the actual amount comes from the rectangle's own interior area, so blocked examples reveal a one-time rebound rather than a stretch-to-marker rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, rects = _parse_problem(g)
    occupied = g != bg
    out = np.full_like(g, bg)
    claimed = np.zeros_like(g, dtype=bool)

    for rect in rects:
        self_mask = np.zeros_like(g, dtype=bool)
        r0, c0, r1, c1 = map(int, rect["bbox"])
        self_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        mr, mc = map(int, rect["marker"])
        self_mask[mr, mc] = True
        final_bbox, _info = _stretch_bbox(rect["bbox"], rect["direction"], rect["budget"], occupied & ~self_mask)
        fr0, fc0, fr1, fc1 = map(int, final_bbox)
        final_mask = np.zeros_like(g, dtype=bool)
        final_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
        assert not np.any(claimed & final_mask), "output rectangles should not overlap"
        claimed |= final_mask
        _paint_rect(out, final_bbox, rect["border"], rect["fill"])

    return out


def _sample_axis_spaces(length: int, budget: int, branch: str) -> tuple[int, int]:
    max_room = 30
    if branch == "free":
        primary = random.randint(max(2, budget), min(max_room, max(2, budget) + 5))
        secondary = random.randint(0, 4)
    elif branch == "bounce":
        primary = random.randint(2, budget - 1)
        need = budget - primary
        secondary = random.randint(need, min(max_room, need + 4))
    elif branch == "discard":
        primary = random.randint(2, budget - 1)
        max_secondary = budget - primary - 1
        secondary = random.randint(0, max(0, max_secondary))
    else:
        raise ValueError(branch)
    return primary, secondary


def _make_single_scene() -> tuple[list[dict], tuple[int, int], list[str]]:
    direction = random.choice(list(OPPOSITE))
    branch = random.choices(["free", "bounce", "discard"], weights=[2, 3, 2], k=1)[0]
    while True:
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        budget = (h - 2) * (w - 2)
        if branch == "free" or budget >= 3:
            break
    if direction in {"up", "down"}:
        primary, secondary = _sample_axis_spaces(h, budget, branch)
        left = random.randint(0, 3)
        right = random.randint(0, 3)
        if direction == "up":
            top, bottom = primary, secondary
        else:
            top, bottom = secondary, primary
        scene_h = top + h + bottom
        scene_w = left + w + right
        bbox = _bbox_from_rcwh(top, left, h, w)
        gap = random.randint(2, primary)
        marker_c = random.randint(left, left + w - 1)
        marker_r = top - gap if direction == "up" else top + h - 1 + gap
    else:
        primary, secondary = _sample_axis_spaces(w, budget, branch)
        top = random.randint(0, 3)
        bottom = random.randint(0, 3)
        if direction == "left":
            left, right = primary, secondary
        else:
            left, right = secondary, primary
        scene_h = top + h + bottom
        scene_w = left + w + right
        bbox = _bbox_from_rcwh(top, left, h, w)
        gap = random.randint(2, primary)
        marker_r = random.randint(top, top + h - 1)
        marker_c = left - gap if direction == "left" else left + w - 1 + gap
    spec = {
        "bbox": bbox,
        "marker": (marker_r, marker_c),
        "direction": direction,
        "border": -1,
        "fill": -1,
    }
    return [spec], (scene_h, scene_w), [branch]


def _make_blocked_scene() -> tuple[list[dict], tuple[int, int], list[str]]:
    a_branch = random.choices(["bounce", "discard"], weights=[3, 2], k=1)[0]
    b_branch = random.choices(["free", "bounce", "discard"], weights=[2, 2, 1], k=1)[0]
    while True:
        ah = random.randint(3, 6)
        aw = random.randint(3, 6)
        bh = random.randint(3, 5)
        bw = random.randint(max(3, aw), 6)
        budget_a = (ah - 2) * (aw - 2)
        budget_b = (bh - 2) * (bw - 2)
        if a_branch == "discard" and budget_a < 2:
            continue
        if b_branch != "free" and budget_b < 3:
            continue
        break

    left_b, right_b = _sample_axis_spaces(bw, budget_b, b_branch)
    b_dir = random.choice(["left", "right"])
    if b_dir == "left":
        b_left, b_right = left_b, right_b
    else:
        b_left, b_right = right_b, left_b
    scene_w = b_left + bw + b_right
    b_c0 = b_left
    b_bbox = _bbox_from_rcwh(1, b_c0, bh, bw)

    a_down = random.randint(budget_a if a_branch == "bounce" else 1, budget_a + 4)
    if a_branch == "discard":
        a_down = random.randint(1, budget_a - 1)
    scene_h = 1 + bh + ah + a_down
    a_c0 = random.randint(b_c0, b_c0 + bw - aw)
    a_bbox = _bbox_from_rcwh(1 + bh, a_c0, ah, aw)

    a_marker = (0, random.randint(a_c0, a_c0 + aw - 1))
    if b_dir == "left":
        b_gap = random.randint(2, b_left)
        b_marker = (random.randint(1, bh), b_c0 - b_gap)
    else:
        b_gap = random.randint(2, b_right)
        b_marker = (random.randint(1, bh), b_c0 + bw - 1 + b_gap)

    specs = [
        {"bbox": a_bbox, "marker": a_marker, "direction": "up", "border": -1, "fill": -1},
        {"bbox": b_bbox, "marker": b_marker, "direction": b_dir, "border": -1, "fill": -1},
    ]
    op = random.choice(["id", "rot90", "rot180", "rot270", "flip_h", "flip_v"])
    specs, shape = _transform_specs(specs, (scene_h, scene_w), op)
    return specs, shape, [f"blocked_{a_branch}", f"aux_{b_branch}"]


def _assign_colors(specs: list[dict]) -> None:
    needed = 2 * len(specs)
    colors = random.sample(PALETTE, needed)
    for i, spec in enumerate(specs):
        spec["border"] = int(colors[2 * i])
        spec["fill"] = int(colors[2 * i + 1])


def _translate_specs(specs: list[dict], dr: int, dc: int) -> list[dict]:
    out = []
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        mr, mc = spec["marker"]
        out.append(
            {
                "bbox": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
                "marker": (mr + dr, mc + dc),
                "direction": spec["direction"],
                "border": spec["border"],
                "fill": spec["fill"],
            }
        )
    return out


def _compose_scenes(
    scene_defs: list[tuple[list[dict], tuple[int, int], list[str]]]
) -> tuple[list[dict], tuple[int, int], list[str]] | None:
    gap = random.randint(2, 4)
    if len(scene_defs) == 1:
        specs, shape, tags = scene_defs[0]
        if shape[0] > 30 or shape[1] > 30:
            return None
        extra_h = 30 - shape[0]
        extra_w = 30 - shape[1]
        top = random.randint(0, extra_h)
        bottom = random.randint(0, extra_h - top)
        left = random.randint(0, extra_w)
        right = random.randint(0, extra_w - left)
        translated = _translate_specs(specs, top, left)
        return translated, (top + shape[0] + bottom, left + shape[1] + right), tags

    (specs1, shape1, tags1), (specs2, shape2, tags2) = scene_defs
    horizontal_fits = shape1[1] + gap + shape2[1] <= 30 and max(shape1[0], shape2[0]) <= 30
    vertical_fits = shape1[0] + gap + shape2[0] <= 30 and max(shape1[1], shape2[1]) <= 30
    if horizontal_fits and (not vertical_fits or random.random() < 0.5):
        total_h = max(shape1[0], shape2[0])
        total_w = shape1[1] + gap + shape2[1]
        extra_h = 30 - total_h
        extra_w = 30 - total_w
        top1 = random.randint(0, total_h - shape1[0])
        top2 = random.randint(0, total_h - shape2[0])
        top_pad = random.randint(0, extra_h)
        bottom_pad = random.randint(0, extra_h - top_pad)
        left_pad = random.randint(0, extra_w)
        right_pad = random.randint(0, extra_w - left_pad)
        specs = _translate_specs(specs1, top_pad + top1, left_pad)
        specs += _translate_specs(specs2, top_pad + top2, left_pad + shape1[1] + gap)
        return specs, (top_pad + total_h + bottom_pad, left_pad + total_w + right_pad), tags1 + tags2

    if not vertical_fits:
        return None
    total_h = shape1[0] + gap + shape2[0]
    total_w = max(shape1[1], shape2[1])
    extra_h = 30 - total_h
    extra_w = 30 - total_w
    left1 = random.randint(0, total_w - shape1[1])
    left2 = random.randint(0, total_w - shape2[1])
    top_pad = random.randint(0, extra_h)
    bottom_pad = random.randint(0, extra_h - top_pad)
    left_pad = random.randint(0, extra_w)
    right_pad = random.randint(0, extra_w - left_pad)
    specs = _translate_specs(specs1, top_pad, left_pad + left1)
    specs += _translate_specs(specs2, top_pad + shape1[0] + gap, left_pad + left2)
    return specs, (top_pad + total_h + bottom_pad, left_pad + total_w + right_pad), tags1 + tags2


def _valid_generated(specs: list[dict], shape: tuple[int, int], tags: list[str]) -> np.ndarray | None:
    _assign_colors(specs)
    try:
        grid, expected, analyses = _solve_specs(shape, specs)
        out = main(grid)
        bg, parsed = _parse_problem(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    if bg != 0 or len(parsed) != len(specs):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not all(item["moved"] for item in analyses):
        return None
    if not any(item["bounced"] for item in analyses):
        return None
    if "blocked" not in " ".join(tags) and not any(item["unused_budget"] > 0 for item in analyses):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        scene_count = random.choices([1, 2], weights=[2, 5], k=1)[0]
        scene_defs = []
        total_rects = 0
        for _scene_idx in range(scene_count):
            if total_rects == 0:
                kind = random.choices(["blocked", "single"], weights=[4, 2], k=1)[0]
            else:
                kind = random.choices(["blocked", "single"], weights=[3, 3], k=1)[0]
            if kind == "blocked":
                scene = _make_blocked_scene()
            else:
                scene = _make_single_scene()
            scene_defs.append(scene)
            total_rects += len(scene[0])
        if total_rects < 2 or total_rects > 4:
            continue
        composed = _compose_scenes(scene_defs)
        if composed is None:
            continue
        specs, shape, tags = composed
        if shape[0] > 30 or shape[1] > 30:
            continue
        candidate = _valid_generated(specs, shape, tags)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
