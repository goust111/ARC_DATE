from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shadow_border",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BORDER_COLOR = 5
SAFE_BACKGROUNDS = (0, 1, 8)
OBJECT_COLORS = tuple(c for c in range(10) if c != BORDER_COLOR)
EDGE_NAMES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400
LOCAL_ATTEMPTS = 200


def _edge_distances(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]) -> dict[str, int]:
    h, w = map(int, shape)
    return {
        "top": int(r0),
        "bottom": int(h - 1 - r1),
        "left": int(c0),
        "right": int(w - 1 - c1),
    }


def _nearest_edge(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]) -> str:
    distances = _edge_distances(r0, c0, r1, c1, shape)
    best = min(distances.values())
    winners = [edge for edge, dist in distances.items() if dist == best]
    assert len(winners) == 1, "every object must have a unique nearest edge"
    return winners[0]


def _nearest_edge_or_none(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]) -> str | None:
    distances = _edge_distances(r0, c0, r1, c1, shape)
    best = min(distances.values())
    winners = [edge for edge, dist in distances.items() if dist == best]
    return winners[0] if len(winners) == 1 else None


def _make_rect(color: int, r0: int, c0: int, r1: int, c1: int) -> dict[str, int]:
    return {
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
    }


def _too_close(a: dict[str, int], b: dict[str, int], gap: int = 1) -> bool:
    return not (
        a["r1"] + gap < b["r0"]
        or b["r1"] + gap < a["r0"]
        or a["c1"] + gap < b["c0"]
        or b["c1"] + gap < a["c0"]
    )


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one foreground rectangle"

    objects: list[dict[str, int]] = []
    seen_colors: set[int] = set()
    h, w = g.shape
    for comp in comps:
        color = int(comp["color"])
        assert color not in seen_colors, "foreground rectangle colors must be distinct"
        seen_colors.add(color)

        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert 0 < r0 < r1 < h - 1, "rectangles must stay away from the top/bottom border"
        assert 0 < c0 < c1 < w - 1, "rectangles must stay away from the left/right border"

        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert int(comp["area"]) == area, "every component must be a solid filled rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be monochromatic"

        objects.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "edge": _nearest_edge(r0, c0, r1, c1, g.shape),
            }
        )

    return bg, objects


def _ray_objects(objects: list[dict[str, int]], edge: str, idx: int) -> list[dict[str, int]]:
    if edge == "top":
        ray = [obj for obj in objects if obj["c0"] <= idx <= obj["c1"]]
        ray.sort(key=lambda obj: obj["r0"])
    elif edge == "bottom":
        ray = [obj for obj in objects if obj["c0"] <= idx <= obj["c1"]]
        ray.sort(key=lambda obj: -obj["r1"])
    elif edge == "left":
        ray = [obj for obj in objects if obj["r0"] <= idx <= obj["r1"]]
        ray.sort(key=lambda obj: obj["c0"])
    else:
        ray = [obj for obj in objects if obj["r0"] <= idx <= obj["r1"]]
        ray.sort(key=lambda obj: -obj["c1"])
    return ray


def _compute_output(shape: tuple[int, int], bg: int, objects: list[dict[str, int]]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    out[0, :] = BORDER_COLOR
    out[-1, :] = BORDER_COLOR
    out[:, 0] = BORDER_COLOR
    out[:, -1] = BORDER_COLOR

    for c in range(1, w - 1):
        top_ray = _ray_objects(objects, "top", c)
        if top_ray and any(obj["edge"] == "top" for obj in top_ray):
            out[0, c] = top_ray[0]["color"]

        bottom_ray = _ray_objects(objects, "bottom", c)
        if bottom_ray and any(obj["edge"] == "bottom" for obj in bottom_ray):
            out[h - 1, c] = bottom_ray[0]["color"]

    for r in range(1, h - 1):
        left_ray = _ray_objects(objects, "left", r)
        if left_ray and any(obj["edge"] == "left" for obj in left_ray):
            out[r, 0] = left_ray[0]["color"]

        right_ray = _ray_objects(objects, "right", r)
        if right_ray and any(obj["edge"] == "right" for obj in right_ray):
            out[r, w - 1] = right_ray[0]["color"]

    return out


def _visible_from_every_edge(shape: tuple[int, int], bg: int, objects: list[dict[str, int]]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    out[0, :] = BORDER_COLOR
    out[-1, :] = BORDER_COLOR
    out[:, 0] = BORDER_COLOR
    out[:, -1] = BORDER_COLOR

    for c in range(1, w - 1):
        top_ray = _ray_objects(objects, "top", c)
        if top_ray:
            out[0, c] = top_ray[0]["color"]
        bottom_ray = _ray_objects(objects, "bottom", c)
        if bottom_ray:
            out[h - 1, c] = bottom_ray[0]["color"]

    for r in range(1, h - 1):
        left_ray = _ray_objects(objects, "left", r)
        if left_ray:
            out[r, 0] = left_ray[0]["color"]
        right_ray = _ray_objects(objects, "right", r)
        if right_ray:
            out[r, w - 1] = right_ray[0]["color"]

    return out


def _own_projection_only(shape: tuple[int, int], bg: int, objects: list[dict[str, int]]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    out[0, :] = BORDER_COLOR
    out[-1, :] = BORDER_COLOR
    out[:, 0] = BORDER_COLOR
    out[:, -1] = BORDER_COLOR

    orders = {
        "top": sorted((obj for obj in objects if obj["edge"] == "top"), key=lambda obj: -obj["r0"]),
        "bottom": sorted((obj for obj in objects if obj["edge"] == "bottom"), key=lambda obj: obj["r1"]),
        "left": sorted((obj for obj in objects if obj["edge"] == "left"), key=lambda obj: -obj["c0"]),
        "right": sorted((obj for obj in objects if obj["edge"] == "right"), key=lambda obj: obj["c1"]),
    }

    for obj in orders["top"]:
        out[0, obj["c0"] : obj["c1"] + 1] = obj["color"]
    for obj in orders["bottom"]:
        out[h - 1, obj["c0"] : obj["c1"] + 1] = obj["color"]
    for obj in orders["left"]:
        out[obj["r0"] : obj["r1"] + 1, 0] = obj["color"]
    for obj in orders["right"]:
        out[obj["r0"] : obj["r1"] + 1, w - 1] = obj["color"]

    return out


def _span_colors(out: np.ndarray, obj: dict[str, int]) -> set[int]:
    if obj["edge"] == "top":
        vals = out[0, obj["c0"] : obj["c1"] + 1]
    elif obj["edge"] == "bottom":
        vals = out[-1, obj["c0"] : obj["c1"] + 1]
    elif obj["edge"] == "left":
        vals = out[obj["r0"] : obj["r1"] + 1, 0]
    else:
        vals = out[obj["r0"] : obj["r1"] + 1, -1]
    return {int(v) for v in vals.tolist()}


def _has_foreign_occlusion(shape: tuple[int, int], objects: list[dict[str, int]]) -> bool:
    for edge in EDGE_NAMES:
        limit = shape[1] - 1 if edge in ("top", "bottom") else shape[0] - 1
        for idx in range(1, limit):
            ray = _ray_objects(objects, edge, idx)
            if ray and any(obj["edge"] == edge for obj in ray) and ray[0]["edge"] != edge:
                return True
    return False


def _has_visible_but_noncasting_ray(shape: tuple[int, int], objects: list[dict[str, int]]) -> bool:
    for edge in EDGE_NAMES:
        limit = shape[1] - 1 if edge in ("top", "bottom") else shape[0] - 1
        for idx in range(1, limit):
            ray = _ray_objects(objects, edge, idx)
            if ray and not any(obj["edge"] == edge for obj in ray):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw only the border shadows of the interior rectangles, recoloring blocked shadow slices by the nearer occluder.

    Input structure:
    - The background grid contains several separated, solid, monochromatic rectangles.
    - Each rectangle stays strictly inside the canvas and has a unique nearest outer edge.
    - Rectangles may sit between another rectangle and that other rectangle's nearest edge.

    Rule:
    - Clear the whole interior to the background and draw a one-cell grey border around the canvas.
    - Each rectangle contributes only toward its own nearest edge, over the full row-span or column-span it covers on that edge.
    - Along each edge ray, a shadow exists only if some rectangle on that ray belongs to that edge; the border cell then takes the color of the first rectangle seen from that edge, even if that nearer blocker belongs to another edge.
    - If no rectangle assigned to that edge lies on a ray, keep that border cell grey.

    Key insight:
    - The border is not painted by simple visibility; nearest-edge ownership decides whether a shadow exists, and edge-side occluders then overwrite the shadow color slice by slice.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _extract_objects(g)
    return _compute_output(g.shape, bg, objects)


def _draw_grid(shape: tuple[int, int], bg: int, rects: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return grid


def _sample_edge_rect(
    shape: tuple[int, int],
    edge: str,
    color: int,
    existing: list[dict[str, int]],
) -> dict[str, int] | None:
    h, w = map(int, shape)
    for _ in range(LOCAL_ATTEMPTS):
        rect_h = random.randint(2, min(5, h - 4))
        rect_w = random.randint(2, min(6, w - 4))
        max_d = min(5, h - rect_h - 2, w - rect_w - 2)
        if max_d < 1:
            return None
        d = random.randint(1, max_d)

        if edge == "top":
            r0 = d
            c_lo = d + 1
            c_hi = w - rect_w - d - 1
            if c_lo > c_hi:
                continue
            c0 = random.randint(c_lo, c_hi)
        elif edge == "bottom":
            r0 = h - rect_h - d
            c_lo = d + 1
            c_hi = w - rect_w - d - 1
            if c_lo > c_hi:
                continue
            c0 = random.randint(c_lo, c_hi)
        elif edge == "left":
            c0 = d
            r_lo = d + 1
            r_hi = h - rect_h - d - 1
            if r_lo > r_hi:
                continue
            r0 = random.randint(r_lo, r_hi)
        else:
            c0 = w - rect_w - d
            r_lo = d + 1
            r_hi = h - rect_h - d - 1
            if r_lo > r_hi:
                continue
            r0 = random.randint(r_lo, r_hi)

        rect = _make_rect(color, r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if _nearest_edge_or_none(rect["r0"], rect["c0"], rect["r1"], rect["c1"], shape) != edge:
            continue
        if any(_too_close(rect, other, gap=1) for other in existing):
            continue
        return rect
    return None


def _build_top_scenario(shape: tuple[int, int], colors: list[int], use_multilayer: bool) -> list[dict[str, int]] | None:
    h, w = map(int, shape)
    for _ in range(LOCAL_ATTEMPTS):
        rect_h = random.randint(2, 4)
        rect_w = random.randint(4, min(7, w - 7))
        max_top = min(8, h - rect_h - 7)
        if max_top < 4:
            return None

        r0_a = random.randint(4, max_top)
        c_lo = r0_a + 2
        c_hi = w - rect_w - r0_a - 2
        if c_lo > c_hi:
            continue
        c0_a = random.randint(c_lo, c_hi)
        caster = _make_rect(colors[0], r0_a, c0_a, r0_a + rect_h - 1, c0_a + rect_w - 1)
        if _nearest_edge_or_none(caster["r0"], caster["c0"], caster["r1"], caster["c1"], shape) != "top":
            continue

        overlap_w = random.randint(2, min(4, rect_w - 1))
        overlap_start = random.randint(1, rect_w - overlap_w)
        overlap_left = c0_a + overlap_start
        overlap_right = overlap_left + overlap_w - 1

        blocker_h = random.randint(2, 3)
        r0_b_min = 3
        r0_b_max = r0_a - blocker_h - 2
        if r0_b_min > r0_b_max:
            continue
        r0_b = random.randint(r0_b_min, r0_b_max)
        blocker = _make_rect(colors[1], r0_b, 1, r0_b + blocker_h - 1, overlap_right)
        if _nearest_edge_or_none(blocker["r0"], blocker["c0"], blocker["r1"], blocker["c1"], shape) != "left":
            continue
        if _too_close(caster, blocker, gap=1):
            continue

        rects = [caster, blocker]
        if use_multilayer:
            if overlap_w < 2:
                continue
            blocker2_h = random.randint(2, 3)
            r0_c_min = 3
            r0_c_max = r0_b - blocker2_h - 2
            if r0_c_min > r0_c_max:
                continue
            sub_w = random.randint(1, overlap_w - 1)
            r0_c = random.randint(r0_c_min, r0_c_max)
            blocker2 = _make_rect(
                colors[2],
                r0_c,
                1,
                r0_c + blocker2_h - 1,
                overlap_left + sub_w - 1,
            )
            if _nearest_edge_or_none(blocker2["r0"], blocker2["c0"], blocker2["r1"], blocker2["c1"], shape) != "left":
                continue
            if any(_too_close(blocker2, other, gap=1) for other in rects):
                continue
            rects.append(blocker2)

        return rects
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bg, objects = _extract_objects(grid)
        out = _compute_output(grid.shape, bg, objects)
        if not np.array_equal(out, main(grid)):
            return False

        visible_any = _visible_from_every_edge(grid.shape, bg, objects)
        own_only = _own_projection_only(grid.shape, bg, objects)
        if np.array_equal(out, visible_any):
            return False
        if np.array_equal(out, own_only):
            return False

        if len({obj["edge"] for obj in objects}) < 4:
            return False
        if not _has_foreign_occlusion(grid.shape, objects):
            return False
        if not _has_visible_but_noncasting_ray(grid.shape, objects):
            return False
        if not any(
            obj["color"] in _span_colors(out, obj)
            and any(color not in (obj["color"], BORDER_COLOR) for color in _span_colors(out, obj))
            for obj in objects
            if obj["edge"] in EDGE_NAMES
        ):
            return False

        border_values = np.concatenate((out[0], out[-1], out[1:-1, 0], out[1:-1, -1]))
        if len(set(map(int, border_values.tolist())) - {BORDER_COLOR}) < 3:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(18, 26)
        w = random.randint(18, 26)
        shape = (h, w)
        bg = random.choice(SAFE_BACKGROUNDS)
        use_multilayer = random.random() < 0.55
        extra_edges = random.sample(EDGE_NAMES, k=random.randint(0, 2))

        needed = 4 + int(use_multilayer) + len(extra_edges)
        palette = [color for color in OBJECT_COLORS if color != bg]
        colors = random.sample(palette, needed)

        rects = _build_top_scenario(shape, colors, use_multilayer)
        if rects is None:
            continue

        next_color = len(rects)
        bottom = _sample_edge_rect(shape, "bottom", colors[next_color], rects)
        if bottom is None:
            continue
        rects.append(bottom)

        right = _sample_edge_rect(shape, "right", colors[next_color + 1], rects)
        if right is None:
            continue
        rects.append(right)

        color_idx = next_color + 2
        ok = True
        for edge in extra_edges:
            extra = _sample_edge_rect(shape, edge, colors[color_idx], rects)
            color_idx += 1
            if extra is None:
                ok = False
                break
            rects.append(extra)
        if not ok:
            continue

        grid = _draw_grid(shape, bg, rects)
        grid = np.rot90(grid, random.randrange(4)).copy()
        if random.random() < 0.5:
            grid = np.fliplr(grid).copy()

        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid shadow-border puzzle input")
