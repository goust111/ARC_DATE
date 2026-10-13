from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, find_components, get_bg, paste


BACKGROUND = Color.BLACK
TEMPLATE_COLOR = Color.GREEN
PAIR_COLORS = (Color.BLUE, Color.RED, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON)
SIDES = ("left", "right", "above", "below")

GRID_RANGE = (18, 26)
PAIR_COUNT_CHOICES = (2, 3)
TEMPLATE_AREA_RANGE = (5, 8)
TEMPLATE_HEIGHT_CHOICES = (3, 4)
TEMPLATE_WIDTH_CHOICES = (3, 4, 5)
FRAGMENT_GAP_RANGE = (3, 6)

TEMPLATE_RETRIES = 400
PLACEMENT_RETRIES = 400
INSTANCE_RETRIES = 400
SEPARATION_PADDING = 1


Coord = tuple[int, int]


def _coords_set(coords: np.ndarray) -> frozenset[Coord]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _bbox_of_cells(cells: frozenset[Coord]) -> tuple[int, int, int, int]:
    coords = np.array(sorted(cells), dtype=int)
    return bbox_coords(coords)


def _normalize_cells(cells: frozenset[Coord]) -> tuple[Coord, ...]:
    r0, c0, _r1, _c1 = _bbox_of_cells(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _place_offsets(offsets: tuple[Coord, ...], r0: int, c0: int) -> frozenset[Coord]:
    return frozenset((r0 + dr, c0 + dc) for dr, dc in offsets)


def _translate(cells: frozenset[Coord], dr: int, dc: int) -> frozenset[Coord]:
    return frozenset((r + dr, c + dc) for r, c in cells)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _is_solid_anchor(comp: dict, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == 4
        and r1 - r0 + 1 == 2
        and c1 - c0 + 1 == 2
        and bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])).all())
    )


def _cardinal_relation(cells: frozenset[Coord], anchor_bbox: tuple[int, int, int, int]) -> tuple[str, int, int]:
    r0, c0, r1, c1 = _bbox_of_cells(cells)
    ar0, ac0, ar1, ac1 = map(int, anchor_bbox)

    candidates: list[tuple[str, int, int]] = []
    if c1 < ac0 and _intervals_overlap(r0, r1, ar0, ar1):
        candidates.append(("left", 0, ac0 - c1 - 1))
    if c0 > ac1 and _intervals_overlap(r0, r1, ar0, ar1):
        candidates.append(("right", 0, ac1 + 1 - c0))
    if r1 < ar0 and _intervals_overlap(c0, c1, ac0, ac1):
        candidates.append(("above", ar0 - r1 - 1, 0))
    if r0 > ar1 and _intervals_overlap(c0, c1, ac0, ac1):
        candidates.append(("below", ar1 + 1 - r0, 0))
    assert len(candidates) == 1, "fragment/template placement must have a unique cardinal relation to the anchor"
    return candidates[0]


def _mirror_across_anchor(cells: frozenset[Coord], anchor_bbox: tuple[int, int, int, int], side: str) -> frozenset[Coord]:
    ar0, ac0, ar1, ac1 = map(int, anchor_bbox)
    mirrored = set()
    for r, c in cells:
        if side == "left":
            mirrored.add((r, 2 * ac0 - 1 - c))
        elif side == "right":
            mirrored.add((r, 2 * ac1 + 1 - c))
        elif side == "above":
            mirrored.add((2 * ar0 - 1 - r, c))
        else:
            mirrored.add((2 * ar1 + 1 - r, c))
    return frozenset(mirrored)


def _move_to_anchor(cells: frozenset[Coord], anchor_bbox: tuple[int, int, int, int]) -> tuple[str, frozenset[Coord]]:
    side, dr, dc = _cardinal_relation(cells, anchor_bbox)
    return side, _translate(cells, dr, dc)


def _solve_color(
    grid: np.ndarray,
    bg: int,
    color: int,
    template_offsets: tuple[Coord, ...],
    components: list[dict],
) -> dict:
    color_components = [comp for comp in components if int(comp["color"]) == int(color)]
    anchor_candidates = [comp for comp in color_components if _is_solid_anchor(comp, grid)]
    assert len(anchor_candidates) == 1, "each non-template color must contain exactly one 2x2 anchor"
    anchor = anchor_candidates[0]
    anchor_cells = _coords_set(anchor["coords"])

    fragment_cells = set()
    for comp in color_components:
        if comp is anchor:
            continue
        fragment_cells.update((int(r), int(c)) for r, c in comp["coords"])
    fragment_cells = frozenset(fragment_cells)
    assert len(fragment_cells) >= 2, "each fragment must expose at least two pixels"

    H, W = map(int, grid.shape)
    seen_origins: set[Coord] = set()
    placements: list[frozenset[Coord]] = []
    for pr, pc in fragment_cells:
        for tr, tc in template_offsets:
            r0 = pr - tr
            c0 = pc - tc
            if (r0, c0) in seen_origins:
                continue
            seen_origins.add((r0, c0))
            cells = _place_offsets(template_offsets, r0, c0)
            if not all(0 <= r < H and 0 <= c < W for r, c in cells):
                continue
            if not fragment_cells.issubset(cells):
                continue
            if cells & anchor_cells:
                continue
            ok = True
            for r, c in cells:
                val = int(grid[r, c])
                if val != bg and (r, c) not in fragment_cells:
                    ok = False
                    break
            if ok:
                placements.append(cells)

    unique_placements = {frozenset(p) for p in placements}
    assert len(unique_placements) == 1, f"fragment color {color} must map to a unique completed template placement"
    completed = next(iter(unique_placements))
    side, moved = _move_to_anchor(completed, anchor["bbox"])
    mirrored = _mirror_across_anchor(moved, anchor["bbox"], side)
    assert moved.isdisjoint(mirrored), "moved copy and mirror must stay disjoint"
    assert all(0 <= r < H and 0 <= c < W for r, c in moved | mirrored), "output cells must stay in bounds"

    return {
        "color": int(color),
        "anchor_bbox": tuple(map(int, anchor["bbox"])),
        "anchor_cells": anchor_cells,
        "fragment_cells": fragment_cells,
        "completed_cells": completed,
        "side": side,
        "moved_cells": moved,
        "mirrored_cells": mirrored,
        "final_cells": moved | mirrored,
    }


def _extract_scene(grid: np.ndarray) -> tuple[dict, tuple[Coord, ...], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected foreground objects"

    largest_area = max(int(comp["area"]) for comp in components)
    template_candidates = [comp for comp in components if int(comp["area"]) == largest_area]
    assert len(template_candidates) == 1, "template must be the unique largest foreground object"
    template = template_candidates[0]
    template_offsets = _normalize_cells(_coords_set(template["coords"]))

    scene: list[dict] = []
    colors = sorted({int(comp["color"]) for comp in components if int(comp["color"]) != int(template["color"])})
    for color in colors:
        scene.append(_solve_color(g, bg, color, template_offsets, components))
    return template, template_offsets, scene, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored fragment into the template, slide the completed copy to its anchor, and mirror it across the anchor-facing edge.

    Input structure:
    - The background is black.
    - One unique largest monochromatic 4-connected object is a full template sprite.
    - Each other foreground color has one nearby fragment plus one solid 2x2 anchor square of the same color.

    Rule:
    - Use the unique largest object as the template mask.
    - For each non-template color, uniquely place a translated template copy whose cells contain exactly that color's fragment pixels.
    - Slide the completed copy straight toward its 2x2 anchor along the unique cardinal direction until it touches the anchor's near edge, then reflect the touched copy across that edge.
    - Erase the original template, the original fragments, and the anchors; keep only the slid completed copy and its mirror in that color.

    Key insight:
    - The fragment is only a partial locator for a hidden full template, while the anchor determines both the sliding target and the mirror axis, so completion must happen before sliding and mirroring.
    """
    template, _template_offsets, scene, bg = _extract_scene(grid)
    out = np.full_like(np.asarray(grid, dtype=int), bg)

    all_cells: set[Coord] = set()
    for info in scene:
        assert not (all_cells & info["final_cells"]), "different colors must not overlap in the output"
        for r, c in info["final_cells"]:
            out[r, c] = int(info["color"])
        all_cells.update(info["final_cells"])

    assert int(template["color"]) not in {int(v) for v in np.unique(out) if int(v) != bg}
    return out


def _cells_to_patch(cells: frozenset[Coord], color: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox_of_cells(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = color
    return patch, r0, c0


def _paint_cells(grid: np.ndarray, cells: frozenset[Coord], color: int, bg: int) -> None:
    patch, r0, c0 = _cells_to_patch(cells, color, bg)
    paste(grid, patch, r0, c0, transparent=bg)


def _random_template_mask() -> np.ndarray:
    for _ in range(TEMPLATE_RETRIES):
        target_area = random.randint(*TEMPLATE_AREA_RANGE)
        h = random.choice(TEMPLATE_HEIGHT_CHOICES)
        w = random.choice(TEMPLATE_WIDTH_CHOICES)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in tuple(cells):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        hh, ww = mask.shape
        if hh not in TEMPLATE_HEIGHT_CHOICES or ww not in TEMPLATE_WIDTH_CHOICES:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric 4-connected template")


def _mask_offsets(mask: np.ndarray) -> tuple[Coord, ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _sample_pair_blueprint(
    template_offsets: tuple[Coord, ...],
    template_shape: tuple[int, int],
    H: int,
    W: int,
    reserved_input: list[frozenset[Coord]],
    reserved_output: list[frozenset[Coord]],
) -> dict | None:
    h, w = map(int, template_shape)

    for _ in range(PLACEMENT_RETRIES):
        side = random.choice(SIDES)
        gap = random.randint(*FRAGMENT_GAP_RANGE)

        if side == "left":
            if W - 2 * w < gap:
                continue
            r0 = random.randint(0, H - h)
            c0 = random.randint(gap, W - 2 * w)
            ar = random.randint(r0, r0 + h - 2)
            ac = c0 + w
            away = (0, -gap)
        elif side == "right":
            if W - w - gap < w:
                continue
            r0 = random.randint(0, H - h)
            c0 = random.randint(w, W - w - gap)
            ar = random.randint(r0, r0 + h - 2)
            ac = c0 - 2
            away = (0, gap)
        elif side == "above":
            if H - 2 * h < gap:
                continue
            r0 = random.randint(gap, H - 2 * h)
            c0 = random.randint(0, W - w)
            ar = r0 + h
            ac = random.randint(c0, c0 + w - 2)
            away = (-gap, 0)
        else:
            if H - h - gap < h:
                continue
            r0 = random.randint(h, H - h - gap)
            c0 = random.randint(0, W - w)
            ar = r0 - 2
            ac = random.randint(c0, c0 + w - 2)
            away = (gap, 0)

        moved = _place_offsets(template_offsets, r0, c0)
        anchor_bbox = (ar, ac, ar + 1, ac + 1)
        anchor_cells = frozenset((rr, cc) for rr in range(ar, ar + 2) for cc in range(ac, ac + 2))
        initial = _translate(moved, away[0], away[1])
        mirrored = _mirror_across_anchor(moved, anchor_bbox, side)
        if moved & mirrored:
            continue

        sweep = set()
        step_r = 0 if away[0] == 0 else (1 if away[0] > 0 else -1)
        step_c = 0 if away[1] == 0 else (1 if away[1] > 0 else -1)
        steps = max(abs(away[0]), abs(away[1]))
        for k in range(steps + 1):
            sweep.update(_translate(moved, step_r * k, step_c * k))
        sweep = frozenset(sweep)

        output_cells = moved | mirrored
        input_cells = sweep | anchor_cells

        if any(_too_close(output_cells, other, SEPARATION_PADDING) for other in reserved_output):
            continue
        if any(_too_close(input_cells, other, SEPARATION_PADDING) for other in reserved_input):
            continue

        return {
            "side": side,
            "gap": gap,
            "anchor_bbox": anchor_bbox,
            "anchor_cells": anchor_cells,
            "initial_cells": initial,
            "moved_cells": moved,
            "mirrored_cells": mirrored,
            "final_cells": output_cells,
            "input_reservation": input_cells,
        }

    return None


def _too_close(a: frozenset[Coord], b: frozenset[Coord], padding: int) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= padding:
                return True
    return False


def _template_origin_options(
    template_offsets: tuple[Coord, ...],
    H: int,
    W: int,
    reservations: list[frozenset[Coord]],
) -> list[Coord]:
    r_span = max(r for r, _c in template_offsets)
    c_span = max(c for _r, c in template_offsets)
    options = []
    for r0 in range(H - r_span):
        for c0 in range(W - c_span):
            cells = _place_offsets(template_offsets, r0, c0)
            if all(not _too_close(cells, other, SEPARATION_PADDING + 1) for other in reservations):
                options.append((r0, c0))
    return options


def _connected_fragment_subsets(mask: np.ndarray) -> list[tuple[Coord, ...]]:
    offsets = _mask_offsets(mask)
    variants: list[tuple[Coord, ...]] = []
    for size in range(2, len(offsets)):
        for idxs in combinations(range(len(offsets)), size):
            chosen = tuple(offsets[i] for i in idxs)
            submask = np.zeros(mask.shape, dtype=bool)
            for r, c in chosen:
                submask[r, c] = True
            if len(components_mask(submask, connectivity=4)) == 1:
                variants.append(chosen)
    random.shuffle(variants)
    return variants


def _shortcut_final(fragment: frozenset[Coord], anchor_bbox: tuple[int, int, int, int]) -> frozenset[Coord] | None:
    try:
        side, moved = _move_to_anchor(fragment, anchor_bbox)
    except AssertionError:
        return None
    return moved | _mirror_across_anchor(moved, anchor_bbox, side)


def _choose_fragment_for_blueprint(
    blueprint: dict,
    grid_without_fragment: np.ndarray,
    bg: int,
    color: int,
    template_mask: np.ndarray,
    template_offsets: tuple[Coord, ...],
) -> frozenset[Coord] | None:
    full_final = blueprint["final_cells"]
    r0, c0, _r1, _c1 = _bbox_of_cells(blueprint["initial_cells"])

    for subset_offsets in _connected_fragment_subsets(template_mask):
        fragment = _place_offsets(subset_offsets, r0, c0)
        draft = grid_without_fragment.copy()
        _paint_cells(draft, fragment, color, bg)
        try:
            info = _solve_color(draft, bg, color, template_offsets, find_components(draft, bg=bg, connectivity=4, monochromatic=True))
        except AssertionError:
            continue
        if info["completed_cells"] != blueprint["initial_cells"]:
            continue
        if info["final_cells"] != full_final:
            continue
        shortcut = _shortcut_final(fragment, blueprint["anchor_bbox"])
        if shortcut == full_final:
            continue
        return fragment
    return None


def _validate_instance(inp: np.ndarray) -> None:
    template, template_offsets, scene, bg = _extract_scene(inp)
    assert int(template["color"]) == TEMPLATE_COLOR
    assert len(scene) in PAIR_COUNT_CHOICES
    assert len(scene) == len({info["color"] for info in scene})

    final_by_color: dict[int, frozenset[Coord]] = {}
    all_final: set[Coord] = set()
    for info in scene:
        assert len(info["fragment_cells"]) < len(template_offsets), "fragment must be a proper subset of the template"
        shortcut = _shortcut_final(info["fragment_cells"], info["anchor_bbox"])
        assert shortcut != info["final_cells"], "completion must be necessary; fragment-only mirroring cannot match the true output"
        assert not (all_final & info["final_cells"])
        final_by_color[info["color"]] = info["final_cells"]
        all_final.update(info["final_cells"])

    out = main(inp)
    for color, cells in final_by_color.items():
        for r, c in cells:
            assert int(out[r, c]) == int(color)
    assert int(template["color"]) not in {int(v) for v in np.unique(out) if int(v) != bg}


def _attempt_instance() -> np.ndarray | None:
    H = random.randint(*GRID_RANGE)
    W = random.randint(*GRID_RANGE)
    bg = BACKGROUND

    template_mask = _random_template_mask()
    template_offsets = _mask_offsets(template_mask)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PAIR_COLORS, pair_count)

    blueprints: list[dict] = []
    reserved_input: list[frozenset[Coord]] = []
    reserved_output: list[frozenset[Coord]] = []
    for color in colors:
        blueprint = _sample_pair_blueprint(template_offsets, template_mask.shape, H, W, reserved_input, reserved_output)
        if blueprint is None:
            return None
        blueprint["color"] = color
        blueprints.append(blueprint)
        reserved_input.append(blueprint["input_reservation"])
        reserved_output.append(blueprint["final_cells"])

    template_options = _template_origin_options(template_offsets, H, W, reserved_input)
    if not template_options:
        return None
    template_origin = random.choice(template_options)
    template_cells = _place_offsets(template_offsets, *template_origin)

    base = np.full((H, W), bg, dtype=int)
    _paint_cells(base, template_cells, TEMPLATE_COLOR, bg)
    for blueprint in blueprints:
        _paint_cells(base, blueprint["anchor_cells"], blueprint["color"], bg)

    fragments: list[frozenset[Coord]] = []
    for blueprint in blueprints:
        fragment = _choose_fragment_for_blueprint(blueprint, base, bg, blueprint["color"], template_mask, template_offsets)
        if fragment is None:
            return None
        fragments.append(fragment)

    inp = base.copy()
    for blueprint, fragment in zip(blueprints, fragments):
        _paint_cells(inp, fragment, blueprint["color"], bg)

    try:
        _validate_instance(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(INSTANCE_RETRIES):
        inp = _attempt_instance()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid anchor-edge-completion ARC input")
