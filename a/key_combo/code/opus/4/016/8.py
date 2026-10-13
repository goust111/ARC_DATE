from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, get_bg


BG = 0
TEAL = 8
INPUT_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_ATTEMPTS = 400


def _area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _draw_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _sorted_rectangles(rects: list[dict]) -> list[dict]:
    return sorted(
        rects,
        key=lambda rect: (-int(rect["area"]), *map(int, rect["bbox"])),
    )


def _infer_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rects: list[dict] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        assert coords.size > 0, "every foreground color must stay visible"
        box = bbox_coords(coords)
        rects.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, box)),
                "coords": coords,
                "area": _area(box),
            }
        )

    areas = [int(rect["area"]) for rect in rects]
    assert len(areas) == len(set(areas)), "rectangle areas must be unique so occlusion order is determined"
    return _sorted_rectangles(rects)


def _restore_scene(rects: list[dict], shape: tuple[int, int] | None = None) -> np.ndarray:
    if shape is None:
        h = max(int(rect["bbox"][2]) for rect in rects) + 1
        w = max(int(rect["bbox"][3]) for rect in rects) + 1
    else:
        h, w = map(int, shape)
    out = np.full((int(h), int(w)), BG, dtype=int)
    for rect in _sorted_rectangles(rects):
        _draw_rect(out, rect["bbox"], int(rect["color"]))
    return out


def _container_colors(rects: list[dict]) -> set[int]:
    container_colors: set[int] = set()
    for outer in rects:
        obox = outer["bbox"]
        if any(
            _strictly_contains(obox, inner["bbox"])
            for inner in rects
            if int(inner["color"]) != int(outer["color"])
        ):
            container_colors.add(int(outer["color"]))
    return container_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the occluded rectangles, then recolor every restored container rectangle to teal.

    Input structure:
    - The background is black.
    - Each non-black, non-teal color belongs to one axis-aligned solid rectangle, but smaller rectangles may overlap and hide parts of larger ones.
    - Every rectangle keeps enough visible extremal pixels that its full bbox can be recovered from the visible fragment.

    Rule:
    - For each foreground color, infer the rectangle's full bbox from the visible pixels' min/max row and column.
    - Sort the inferred rectangles by decreasing bbox area and redraw them as solid rectangles on a blank black canvas, letting smaller rectangles overwrite larger ones.
    - In that restored rectangle set, mark any rectangle whose bbox strictly contains at least one other rectangle's bbox as a container.
    - Recolor only the finally visible pixels of those container rectangles to teal; keep all other visible pixels unchanged.

    Key insight:
    - Containment is defined on the hidden full rectangles rather than on the broken visible fragments, so the scene must be reconstructed from recovered bboxes before deciding what to recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "expected black background"
    assert TEAL not in set(map(int, g.ravel())), "teal must be output-only"

    rects = _infer_rectangles(g)
    assert 4 <= len(rects) <= 5, "expected 4-5 rectangles"

    restored = _restore_scene(rects, g.shape)
    containers = _container_colors(rects)
    assert containers, "expected at least one containing rectangle"
    assert len(containers) < len(rects), "expected at least one non-container rectangle"

    out = restored.copy()
    for color in containers:
        out[out == int(color)] = TEAL

    assert np.any(out != g), "expected a visible transformation"
    return out


def _edge_masks(box: tuple[int, int, int, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    r0, c0, r1, c1 = map(int, box)
    top = (np.full(c1 - c0 + 1, r0, dtype=int), np.arange(c0, c1 + 1, dtype=int))
    bottom = (np.full(c1 - c0 + 1, r1, dtype=int), np.arange(c0, c1 + 1, dtype=int))
    left = (np.arange(r0, r1 + 1, dtype=int), np.full(r1 - r0 + 1, c0, dtype=int))
    right = (np.arange(r0, r1 + 1, dtype=int), np.full(r1 - r0 + 1, c1, dtype=int))
    return [top, bottom, left, right]


def _visible_complete_edge_count(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> int:
    count = 0
    for rr, cc in _edge_masks(box):
        if np.all(grid[rr, cc] == int(color)):
            count += 1
    return int(count)


def _ring_has_foreign_occluder(
    grid: np.ndarray,
    outer_color: int,
    inner_color: int,
    outer_box: tuple[int, int, int, int],
    inner_box: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = map(int, outer_box)
    ring = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    ir0, ic0, ir1, ic1 = map(int, inner_box)
    ring[ir0 - r0 : ir1 - r0 + 1, ic0 - c0 : ic1 - c0 + 1] = False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    foreign = (patch != BG) & (patch != int(outer_color)) & (patch != int(inner_color))
    return bool(np.any(ring & foreign))


def _pair_visually_hidden(
    grid: np.ndarray,
    outer_color: int,
    inner_color: int,
    outer_box: tuple[int, int, int, int],
    inner_box: tuple[int, int, int, int],
) -> bool:
    edge_broken = _visible_complete_edge_count(grid, outer_color, outer_box) < 4
    return bool(edge_broken and _ring_has_foreign_occluder(grid, outer_color, inner_color, outer_box, inner_box))


def _find_free_box(
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    forbidden: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    tries = 120
    for _ in range(tries):
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if any(_boxes_overlap(box, other) for other in forbidden):
            continue
        return box
    return None


def _choose_dims(candidates: list[tuple[int, int]], used_areas: set[int], rng: random.Random) -> tuple[int, int] | None:
    rng.shuffle(candidates)
    for hh, ww in candidates:
        area = int(hh * ww)
        if area not in used_areas:
            used_areas.add(area)
            return int(hh), int(ww)
    return None


def _place_inner(
    outer: tuple[int, int, int, int],
    inner_h: int,
    inner_w: int,
    side: str,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = map(int, outer)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h - inner_h < 2 or outer_w - inner_w < 2:
        return None

    tries = 120
    for _ in range(tries):
        if side == "top":
            mt = rng.randint(1, min(2, outer_h - inner_h - 1))
            mb = outer_h - inner_h - mt
            if mb < 1:
                continue
            ml = rng.randint(1, outer_w - inner_w - 1)
        elif side == "bottom":
            mb = rng.randint(1, min(2, outer_h - inner_h - 1))
            mt = outer_h - inner_h - mb
            if mt < 1:
                continue
            ml = rng.randint(1, outer_w - inner_w - 1)
        elif side == "left":
            ml = rng.randint(1, min(2, outer_w - inner_w - 1))
            mr = outer_w - inner_w - ml
            if mr < 1:
                continue
            mt = rng.randint(1, outer_h - inner_h - 1)
        else:
            mr = rng.randint(1, min(2, outer_w - inner_w - 1))
            ml = outer_w - inner_w - mr
            if ml < 1:
                continue
            mt = rng.randint(1, outer_h - inner_h - 1)

        if side in {"top", "bottom"}:
            mr = outer_w - inner_w - ml
            if mr < 1:
                continue
        else:
            mb = outer_h - inner_h - mt
            if mb < 1:
                continue

        ir0 = r0 + mt
        ic0 = c0 + ml
        box = (ir0, ic0, ir0 + inner_h - 1, ic0 + inner_w - 1)
        if _strictly_contains(outer, box):
            return box
    return None


def _make_occluder(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    side: str,
    used_areas: set[int],
    rng: random.Random,
    limit_area: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    or0, oc0, or1, oc1 = map(int, outer)
    ir0, ic0, ir1, ic1 = map(int, inner)
    hh, ww = map(int, shape)
    if hh * ww >= limit_area or hh * ww in used_areas:
        return None

    if side == "top":
        if ir1 - hh + 1 <= ir0:
            return None
        r0 = ir0 - hh + 1
        c_min = max(oc0 + 1, ic0 - ww + 2)
        c_max = min(oc1 - ww, ic1 - 1)
        if c_min > c_max or r0 >= or0:
            return None
        c0 = rng.randint(int(c_min), int(c_max))
    elif side == "bottom":
        r0 = ir1
        c_min = max(oc0 + 1, ic0 - ww + 2)
        c_max = min(oc1 - ww, ic1 - 1)
        if c_min > c_max or r0 + hh - 1 <= or1:
            return None
        c0 = rng.randint(int(c_min), int(c_max))
    elif side == "left":
        if ic1 - ww + 1 <= ic0:
            return None
        c0 = ic0 - ww + 1
        r_min = max(or0 + 1, ir0 - hh + 2)
        r_max = min(or1 - hh, ir1 - 1)
        if r_min > r_max or c0 >= oc0:
            return None
        r0 = rng.randint(int(r_min), int(r_max))
    else:
        c0 = ic1
        r_min = max(or0 + 1, ir0 - hh + 2)
        r_max = min(or1 - hh, ir1 - 1)
        if r_min > r_max or c0 + ww - 1 <= oc1:
            return None
        r0 = rng.randint(int(r_min), int(r_max))

    box = (int(r0), int(c0), int(r0 + hh - 1), int(c0 + ww - 1))
    if not _boxes_overlap(box, outer) or not _boxes_overlap(box, inner):
        return None
    if _strictly_contains(outer, box) or _strictly_contains(inner, box):
        return None
    used_areas.add(int(hh * ww))
    return box


def _expected_container_colors(rects: list[dict]) -> set[int]:
    return {int(rect["color"]) for rect in rects if any(_strictly_contains(rect["bbox"], other["bbox"]) for other in rects if rect is not other)}


def _validate_scene(inp: np.ndarray, rects: list[dict], hidden_pair: tuple[int, int]) -> bool:
    if int(get_bg(inp)) != BG:
        return False
    if TEAL in set(map(int, inp.ravel())):
        return False
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    color_to_box = {int(rect["color"]): tuple(map(int, rect["bbox"])) for rect in rects}
    for rect in rects:
        color = int(rect["color"])
        visible = np.argwhere(inp == color)
        if visible.size == 0:
            return False
        if tuple(map(int, bbox_coords(visible))) != tuple(map(int, rect["bbox"])):
            return False
        if _visible_complete_edge_count(inp, color, rect["bbox"]) < 2:
            return False

    expected = _expected_container_colors(rects)
    if not expected or len(expected) >= len(rects):
        return False

    outer_color, inner_color = map(int, hidden_pair)
    if not _pair_visually_hidden(inp, outer_color, inner_color, color_to_box[outer_color], color_to_box[inner_color]):
        return False

    occluded = False
    for rect in rects:
        box = rect["bbox"]
        full_area = _area(box)
        visible_area = int(np.count_nonzero(inp == int(rect["color"])))
        if visible_area < full_area:
            occluded = True
            break
    if not occluded:
        return False

    try:
        out = main(inp)
    except Exception:
        return False

    restored = _restore_scene(_infer_rectangles(inp), inp.shape)
    if TEAL not in set(map(int, colors_present(out, exclude={BG}))):
        return False
    if np.array_equal(out, inp):
        return False
    if out.shape != inp.shape or restored.shape != out.shape:
        return False

    visible_container_colors = {
        int(rect["color"])
        for rect in rects
        if np.any(restored == int(rect["color"])) and int(rect["color"]) in expected
    }
    if not visible_container_colors:
        return False
    if not np.any(out == TEAL):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 18)
    w = rng.randint(14, 18)
    chain = rng.random() < 0.5
    side = rng.choice(["top", "bottom", "left", "right"])
    colors = rng.sample(INPUT_COLORS, 5 if chain else 4)
    used_areas: set[int] = set()

    outer_dims = _choose_dims(
        [(8, 9), (8, 10), (9, 9), (9, 10), (10, 9), (10, 10), (8, 11), (11, 8)],
        used_areas,
        rng,
    )
    if outer_dims is None:
        return None
    outer_h, outer_w = outer_dims

    min_top = 2 if side == "top" else 1
    min_left = 2 if side == "left" else 1
    max_top = h - outer_h - (2 if side == "bottom" else 1)
    max_left = w - outer_w - (2 if side == "right" else 1)
    if max_top < min_top or max_left < min_left:
        return None
    outer_r0 = rng.randint(min_top, max_top)
    outer_c0 = rng.randint(min_left, max_left)
    outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)

    if chain:
        inner_dims = _choose_dims(
            [(5, 6), (6, 5), (5, 7), (7, 5), (6, 6)],
            used_areas,
            rng,
        )
    else:
        inner_dims = _choose_dims(
            [(4, 5), (5, 4), (4, 6), (6, 4), (5, 5)],
            used_areas,
            rng,
        )
    if inner_dims is None:
        return None
    inner = _place_inner(outer, inner_dims[0], inner_dims[1], side, rng)
    if inner is None:
        return None

    rects = [
        {"color": int(colors[0]), "bbox": outer, "area": _area(outer)},
        {"color": int(colors[1]), "bbox": inner, "area": _area(inner)},
    ]
    hidden_pair = (int(colors[0]), int(colors[1]))

    if chain:
        core_dims = _choose_dims(
            [(3, 4), (4, 3), (3, 5), (5, 3), (4, 4)],
            used_areas,
            rng,
        )
        if core_dims is None:
            return None
        core_side = rng.choice([s for s in ["top", "bottom", "left", "right"] if s != side] or [side])
        core = _place_inner(inner, core_dims[0], core_dims[1], core_side, rng)
        if core is None:
            return None
        rects.append({"color": int(colors[2]), "bbox": core, "area": _area(core)})

    limit_area = min(_area(inner), _area(rects[-1]["bbox"]) if chain else _area(inner))
    occluder_shapes = [(4, 2), (5, 2), (2, 4), (2, 5), (3, 2), (2, 3)]
    rng.shuffle(occluder_shapes)
    occluder = None
    for shape in occluder_shapes:
        trial_used = set(used_areas)
        occluder = _make_occluder(outer, inner, side, trial_used, rng, limit_area, shape)
        if occluder is not None:
            used_areas = trial_used
            break
    if occluder is None:
        return None

    next_color_idx = 3 if chain else 2
    rects.append({"color": int(colors[next_color_idx]), "bbox": occluder, "area": _area(occluder)})

    indep_dims = _choose_dims(
        [(3, 3), (3, 4), (4, 3), (2, 5), (5, 2)],
        used_areas,
        rng,
    )
    if indep_dims is None:
        return None
    indep = _find_free_box(h, w, indep_dims[0], indep_dims[1], [rect["bbox"] for rect in rects], rng)
    if indep is None:
        return None
    rects.append({"color": int(colors[-1]), "bbox": indep, "area": _area(indep)})

    areas = [int(rect["area"]) for rect in rects]
    if len(areas) != len(set(areas)):
        return None

    inp = np.full((h, w), BG, dtype=int)
    for rect in _sorted_rectangles(rects):
        _draw_rect(inp, rect["bbox"], int(rect["color"]))

    if not _validate_scene(inp, rects, hidden_pair):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
