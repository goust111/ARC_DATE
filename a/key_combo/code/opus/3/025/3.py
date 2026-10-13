from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


for parent in Path(__file__).resolve().parents:
    candidate = parent / "new"
    if (candidate / "arc_utils.py").exists():
        sys.path.append(str(candidate))
        break
else:
    raise ImportError("could not locate ARC utility module")

from arc_utils import (  # type: ignore
    Color,
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    neighbors4,
    paste,
    random_free_location,
)


BG = Color.BLACK
TEMPLATE_COLOR = Color.GREEN
INDICATOR_COLOR = Color.ORANGE
FRAGMENT_COLORS = (Color.BLUE, Color.RED, Color.YELLOW)
ROT_TO_SIDE = {0: "top", 1: "right", 2: "bottom", 3: "left"}
SIDE_TO_ROT = {side: rot for rot, side in ROT_TO_SIDE.items()}

GRID_SIDE_RANGE = (22, 28)
GEN_RETRIES = 300
TEMPLATE_RETRIES = 200
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
TEMPLATE_AREA_RANGE = (10, 15)
FRAGMENT_SIZE_RANGE = (3, 6)


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    arr = np.array(list(coords), dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(arr)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))


def _offsets_to_mask(offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not offsets:
        return np.zeros((0, 0), dtype=bool)
    arr = np.array(offsets, dtype=int)
    _r0, _c0, r1, c1 = bbox_coords(arr)
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in offsets:
        mask[r, c] = True
    return mask


def _mask_to_offsets(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _rotate_offsets(offsets: tuple[tuple[int, int], ...], turns_cw: int) -> tuple[tuple[int, int], ...]:
    mask = _offsets_to_mask(offsets)
    rot = np.rot90(mask, k=(-int(turns_cw)) % 4)
    return _mask_to_offsets(rot)


def _translated(
    offsets: tuple[tuple[int, int], ...], origin: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    dr, dc = map(int, origin)
    return tuple(sorted((dr + r, dc + c) for r, c in offsets))


def _is_connected(offsets) -> bool:
    cells = {tuple(map(int, cell)) for cell in offsets}
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in neighbors4(*cell):
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return len(seen) == len(cells)


def _matching_origins(
    shape_offsets: tuple[tuple[int, int], ...],
    fragment_offsets,
    grid_shape: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    shape = tuple(sorted((int(r), int(c)) for r, c in shape_offsets))
    fragment = {tuple(map(int, cell)) for cell in fragment_offsets}
    candidate_origins = set()
    for fr, fc in fragment:
        for sr, sc in shape:
            candidate_origins.add((fr - sr, fc - sc))

    matches = []
    for origin in sorted(candidate_origins):
        placed = {(origin[0] + r, origin[1] + c) for r, c in shape}
        if not fragment.issubset(placed):
            continue
        if grid_shape is not None:
            h, w = map(int, grid_shape)
            if any(r < 0 or c < 0 or r >= h or c >= w for r, c in placed):
                continue
        matches.append((int(origin[0]), int(origin[1])))
    return matches


def _component_offsets(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize((int(r), int(c)) for r, c in coords)


def _side_from_indicator(bbox_vals, indicator: tuple[int, int]) -> str:
    r0, c0, r1, c1 = map(int, bbox_vals)
    ir, ic = map(int, indicator)
    if ir == r0 - 1 and c0 <= ic <= c1:
        return "top"
    if ic == c1 + 1 and r0 <= ir <= r1:
        return "right"
    if ir == r1 + 1 and c0 <= ic <= c1:
        return "bottom"
    if ic == c0 - 1 and r0 <= ir <= r1:
        return "left"
    raise AssertionError("indicator is not aligned to a fragment bbox side")


def _paint_offsets(
    grid: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    color: int,
) -> None:
    for r, c in _translated(offsets, origin):
        grid[r, c] = int(color)


def _largest_template_component(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    comps = [comp for comp in comps if int(comp["color"]) != INDICATOR_COLOR]
    assert len(comps) >= 2, "expected one template and at least one fragment"
    comps.sort(key=lambda comp: int(comp["area"]), reverse=True)
    assert int(comps[0]["area"]) > int(comps[1]["area"]), "template must be uniquely largest"
    return comps[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored fragment with the rotation of the template encoded by its adjacent indicator pixel.

    Input structure:
    - A black background contains one full green template component and two or three smaller colored fragments.
    - Each fragment has exactly one adjacent orange indicator pixel placed just outside one side of the fragment bbox.
    - Fragments use colors distinct from the template and from each other, and the template is the unique largest monochromatic foreground component.

    Rule:
    - Identify the template as the largest non-orange monochromatic component.
    - For each fragment, read the indicator side relative to the fragment bbox: top/right/bottom/left means rotate the template by 0/90/180/270 degrees clockwise.
    - Find the unique translation where that rotated template contains all fragment pixels, then fill the missing cells with the fragment color.
    - Remove every orange indicator pixel and leave the original green template unchanged.

    Key insight:
    - The orange pixel does not point to a growth direction; its side encodes which template rotation must be used before the fragment can be uniquely completed.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    template_comp = _largest_template_component(g)
    template_offsets = _component_offsets(template_comp["coords"])
    rotated_templates = {rot: _rotate_offsets(template_offsets, rot) for rot in range(4)}

    out = g.copy()
    out[out == INDICATOR_COLOR] = bg

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    fragment_comps = []
    for comp in comps:
        color = int(comp["color"])
        if color in (bg, INDICATOR_COLOR):
            continue
        if np.array_equal(comp["coords"], template_comp["coords"]):
            continue
        fragment_comps.append(comp)
    assert 2 <= len(fragment_comps) <= 3, "expected two or three colored fragments"

    for frag in fragment_comps:
        frag_color = int(frag["color"])
        frag_cells = {tuple(map(int, cell)) for cell in frag["coords"]}
        indicator_neighbors = set()
        for r, c in frag_cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == INDICATOR_COLOR:
                    indicator_neighbors.add((int(nr), int(nc)))
        assert len(indicator_neighbors) == 1, "each fragment must have exactly one adjacent indicator"
        indicator = next(iter(indicator_neighbors))
        side = _side_from_indicator(frag["bbox"], indicator)
        rotation = SIDE_TO_ROT[side]

        full_offsets = rotated_templates[rotation]
        origins = _matching_origins(full_offsets, frag_cells, grid_shape=g.shape)
        assert len(origins) == 1, "fragment must determine a unique placement for the indicated rotation"
        origin = origins[0]

        for rr, cc in _translated(full_offsets, origin):
            base = int(g[rr, cc])
            assert base in (bg, frag_color, INDICATOR_COLOR), "completed fragment overlaps another object"
            out[rr, cc] = frag_color

    return out


def _generate_template() -> tuple[tuple[int, int], ...] | None:
    for _ in range(TEMPLATE_RETRIES):
        target_area = random.randint(*TEMPLATE_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    arr = np.array(list(trial), dtype=int)
                    r0, c0, r1, c1 = bbox_coords(arr)
                    if r1 - r0 + 1 > 6 or c1 - c0 + 1 > 6:
                        continue
                    frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        if len(cells) != target_area:
            continue

        offsets = _normalize(cells)
        arr = np.array(offsets, dtype=int)
        _r0, _c0, r1, c1 = bbox_coords(arr)
        h = int(r1 + 1)
        w = int(c1 + 1)
        if h < 3 or w < 3:
            continue
        if len(offsets) == h * w:
            continue
        if h == 1 or w == 1:
            continue
        if len({_rotate_offsets(offsets, rot) for rot in range(4)}) != 4:
            continue
        return offsets
    return None


def _indicator_positions(
    fragment_offsets: tuple[tuple[int, int], ...],
    side: str,
    shape_offsets: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    arr = np.array(fragment_offsets, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    if side == "top":
        options = {(r - 1, c) for r, c in fragment_offsets if r == r0}
    elif side == "right":
        options = {(r, c + 1) for r, c in fragment_offsets if c == c1}
    elif side == "bottom":
        options = {(r + 1, c) for r, c in fragment_offsets if r == r1}
    else:
        options = {(r, c - 1) for r, c in fragment_offsets if c == c0}

    shape_cells = set(shape_offsets)
    clean = sorted(pos for pos in options if pos not in shape_cells)
    return clean if clean else sorted(options)


def _fragment_candidates(
    template_offsets: tuple[tuple[int, int], ...]
) -> dict[int, list[dict]]:
    rotated_templates = {rot: _rotate_offsets(template_offsets, rot) for rot in range(4)}
    by_rotation: dict[int, list[dict]] = {rot: [] for rot in range(4)}

    for rot, shape_offsets in rotated_templates.items():
        cells = list(shape_offsets)
        max_size = min(FRAGMENT_SIZE_RANGE[1], len(cells))
        for size in range(FRAGMENT_SIZE_RANGE[0], max_size + 1):
            for combo in combinations(cells, size):
                subset = tuple(sorted(combo))
                if not _is_connected(subset):
                    continue
                own_origins = _matching_origins(shape_offsets, subset)
                if own_origins != [(0, 0)]:
                    continue

                cross_matches = {}
                for other_rot, other_shape in rotated_templates.items():
                    origins = _matching_origins(other_shape, subset)
                    if origins:
                        cross_matches[other_rot] = origins
                if len(cross_matches) < 2:
                    continue

                side = ROT_TO_SIDE[rot]
                indicator_options = _indicator_positions(subset, side, shape_offsets)
                if not indicator_options:
                    continue

                by_rotation[rot].append(
                    {
                        "rotation": rot,
                        "shape_offsets": shape_offsets,
                        "fragment_offsets": subset,
                        "side": side,
                        "cross_matches": cross_matches,
                        "indicator_options": indicator_options,
                    }
                )
    return by_rotation


def _choose_fragments(template_offsets: tuple[tuple[int, int], ...]) -> list[dict] | None:
    candidates = _fragment_candidates(template_offsets)
    available = [rot for rot, items in candidates.items() if items]
    if len(available) < 3:
        return None

    fragment_count = 3 if random.random() < 0.55 else 2
    chosen_rotations = random.sample(available, fragment_count)
    chosen_colors = random.sample(FRAGMENT_COLORS, fragment_count)

    fragments = []
    used_signatures = set()
    for rot, color in zip(chosen_rotations, chosen_colors):
        pool = candidates[rot][:]
        random.shuffle(pool)
        picked = None
        for cand in pool:
            signature = (rot, _normalize(cand["fragment_offsets"]))
            if signature in used_signatures:
                continue
            picked = cand
            used_signatures.add(signature)
            break
        if picked is None:
            return None

        fragments.append(
            {
                "rotation": rot,
                "color": int(color),
                "shape_offsets": picked["shape_offsets"],
                "fragment_offsets": picked["fragment_offsets"],
                "side": picked["side"],
                "cross_matches": picked["cross_matches"],
                "indicator_local": random.choice(picked["indicator_options"]),
            }
        )
    return fragments


def _shape_patch(offsets: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    mask = _offsets_to_mask(offsets)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _fragment_patches(fragment: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    shape_offsets = fragment["shape_offsets"]
    fragment_offsets = fragment["fragment_offsets"]
    indicator_local = fragment["indicator_local"]
    all_cells = list(shape_offsets) + [indicator_local]
    rs = [r for r, _c in all_cells]
    cs = [c for _r, c in all_cells]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    reserve = np.zeros((h, w), dtype=int)
    input_patch = np.zeros((h, w), dtype=int)
    for r, c in shape_offsets:
        reserve[r - r0, c - c0] = 9
    for r, c in fragment_offsets:
        input_patch[r - r0, c - c0] = int(fragment["color"])
    reserve[indicator_local[0] - r0, indicator_local[1] - c0] = 9
    input_patch[indicator_local[0] - r0, indicator_local[1] - c0] = INDICATOR_COLOR
    return reserve, input_patch, (-r0, -c0)


def _expected_output(grid: np.ndarray, fragments: list[dict]) -> np.ndarray:
    out = grid.copy()
    out[out == INDICATOR_COLOR] = BG
    for fragment in fragments:
        _paint_offsets(out, fragment["shape_offsets"], fragment["shape_origin"], fragment["color"])
    return out


def _validate_instance(grid: np.ndarray, template_offsets, fragments: list[dict]) -> bool:
    try:
        template_comp = _largest_template_component(grid)
        if int(template_comp["color"]) != TEMPLATE_COLOR:
            return False
        if _component_offsets(template_comp["coords"]) != template_offsets:
            return False

        for fragment in fragments:
            rot = int(fragment["rotation"])
            observed = _translated(fragment["fragment_offsets"], fragment["shape_origin"])
            own_origins = _matching_origins(fragment["shape_offsets"], observed, grid_shape=grid.shape)
            if own_origins != [fragment["shape_origin"]]:
                return False

            distinct_rotations = 0
            for other_rot in range(4):
                other_shape = _rotate_offsets(template_offsets, other_rot)
                if _matching_origins(other_shape, observed, grid_shape=grid.shape):
                    distinct_rotations += 1
            if distinct_rotations < 2:
                return False

            side = _side_from_indicator(
                bbox_coords(np.array(observed, dtype=int)),
                fragment["indicator_global"],
            )
            if SIDE_TO_ROT[side] != rot:
                return False

        expected = _expected_output(grid, fragments)
        actual = main(grid)
        return np.array_equal(actual, expected)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    template_offsets = _generate_template()
    if template_offsets is None:
        return None

    fragments = _choose_fragments(template_offsets)
    if fragments is None:
        return None

    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserve_canvas = np.zeros((h, w), dtype=int)

    template_patch = _shape_patch(template_offsets, TEMPLATE_COLOR)
    try:
        template_loc = random_free_location(
            reserve_canvas,
            template_patch,
            transparent=BG,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(reserve_canvas, template_patch, *template_loc, transparent=BG)
    paste(grid, template_patch, *template_loc, transparent=BG)

    for fragment in fragments:
        reserve_patch, input_patch, shape_offset_in_patch = _fragment_patches(fragment)
        try:
            patch_loc = random_free_location(
                reserve_canvas,
                reserve_patch,
                transparent=BG,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(reserve_canvas, reserve_patch, *patch_loc, transparent=BG)
        paste(grid, input_patch, *patch_loc, transparent=BG)

        fragment["shape_origin"] = (
            int(patch_loc[0] + shape_offset_in_patch[0]),
            int(patch_loc[1] + shape_offset_in_patch[1]),
        )
        fragment["indicator_global"] = (
            int(patch_loc[0] + shape_offset_in_patch[0] + fragment["indicator_local"][0]),
            int(patch_loc[1] + shape_offset_in_patch[1] + fragment["indicator_local"][1]),
        )

    if not _validate_instance(grid, template_offsets, fragments):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-indicator completion puzzle")
