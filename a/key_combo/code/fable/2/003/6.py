from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_fable_2_003_6", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_arc_utils = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
bbox = _arc_utils.bbox
crop_bbox = _arc_utils.crop_bbox
find_components = _arc_utils.find_components
get_bg = _arc_utils.get_bg
paste = _arc_utils.paste
random_free_location = _arc_utils.random_free_location


PLACEHOLDER_COLOR = Color.BLUE
DEFAULT_BAND_COLOR = Color.GREY
TEMPLATE_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
BAND_THICKNESS_RANGE = (2, 4)
TEMPLATE_COUNT_RANGE = (2, 4)
PLACEHOLDER_COUNT_MIN = 2
TEMPLATE_H_RANGE = (4, 7)
TEMPLATE_W_RANGE = (4, 7)
TEMPLATE_FILL_RANGE = (0.42, 0.68)
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 300


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _crop_true(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask.astype(int), box).astype(bool)


def _find_band(grid: np.ndarray) -> tuple[int, int, int]:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]
    candidates = [r for r in full_rows if int(grid[r, 0]) != int(get_bg(grid))]
    assert candidates, "expected a full-width non-background band"
    color = int(grid[candidates[0], 0])
    assert all(int(grid[r, 0]) == color for r in candidates), "band rows must share one color"
    expected = list(range(candidates[0], candidates[-1] + 1))
    assert candidates == expected, "band rows must be contiguous"
    return candidates[0], candidates[-1], color


def _template_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _shift_template(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple((int(top + r), int(left + c)) for r, c in coords)


def _split_visible(coords: tuple[tuple[int, int], ...], band0: int, band1: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    visible = frozenset((r, c) for r, c in coords if not (band0 <= r <= band1))
    hidden = frozenset((r, c) for r, c in coords if band0 <= r <= band1)
    return visible, hidden


def _extract_templates(grid: np.ndarray, bg: int, band0: int, band1: int, band_color: int) -> list[dict]:
    template_colors = sorted(set(map(int, np.unique(grid))) - {int(bg), PLACEHOLDER_COLOR, int(band_color)})
    templates = find_components(grid, bg=bg, colors=template_colors, connectivity=4, monochromatic=True)
    assert templates, "expected at least one full template object"

    out = []
    signatures = set()
    for comp in templates:
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 < band0 or r0 > band1, "templates must lie completely outside the band"
        local = crop_bbox((grid == int(comp["color"])).astype(int), comp["bbox"]).astype(bool)
        sig = _mask_signature(local)
        assert sig not in signatures, "template shapes must be distinct"
        signatures.add(sig)
        out.append(
            {
                "color": int(comp["color"]),
                "mask": local,
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": tuple((int(r), int(c)) for r, c in np.argwhere(local)),
            }
        )
    return out


def _build_candidates(grid: np.ndarray, templates: list[dict], band0: int, band1: int) -> tuple[frozenset[tuple[int, int]], list[dict]]:
    height, width = map(int, grid.shape)
    blue_pixels = frozenset((int(r), int(c)) for r, c in np.argwhere(grid == PLACEHOLDER_COLOR))
    assert blue_pixels, "expected at least one blue remnant"

    candidates: list[dict] = []
    for template_index, template in enumerate(templates):
        mask = template["mask"]
        h, w = map(int, mask.shape)
        row_start = max(0, band0 - h + 1)
        row_end = min(height - h, band1)
        for top in range(row_start, row_end + 1):
            for left in range(0, width - w + 1):
                full_coords = _shift_template(mask, top, left)
                visible, hidden = _split_visible(full_coords, band0, band1)
                if not visible or not hidden:
                    continue
                if any(int(grid[r, c]) != PLACEHOLDER_COLOR for r, c in visible):
                    continue
                candidates.append(
                    {
                        "template_index": template_index,
                        "color": template["color"],
                        "mask": mask,
                        "top": int(top),
                        "left": int(left),
                        "visible_set": visible,
                        "full_set": frozenset(full_coords),
                    }
                )
    assert candidates, "expected at least one compatible template placement"
    return blue_pixels, candidates


def _unique_cover(blue_pixels: frozenset[tuple[int, int]], candidates: list[dict]) -> list[dict]:
    by_pixel: dict[tuple[int, int], list[int]] = {pixel: [] for pixel in blue_pixels}
    for idx, candidate in enumerate(candidates):
        for pixel in candidate["visible_set"]:
            if pixel in by_pixel:
                by_pixel[pixel].append(idx)

    assert all(by_pixel[pixel] for pixel in blue_pixels), "every blue pixel must be explained by some template"

    covered: set[tuple[int, int]] = set()
    used_full: set[tuple[int, int]] = set()
    chosen: list[int] = []
    solutions: list[list[int]] = []

    def options_for(pixel: tuple[int, int]) -> list[int]:
        opts = []
        for idx in by_pixel[pixel]:
            candidate = candidates[idx]
            if candidate["visible_set"] & covered:
                continue
            if candidate["full_set"] & used_full:
                continue
            opts.append(idx)
        return opts

    def search() -> None:
        if len(solutions) > 1:
            return
        if len(covered) == len(blue_pixels):
            solutions.append(chosen.copy())
            return

        remaining = [pixel for pixel in blue_pixels if pixel not in covered]
        pivot = min(remaining, key=lambda pixel: len(options_for(pixel)))
        options = options_for(pivot)
        if not options:
            return

        options.sort(key=lambda idx: (-len(candidates[idx]["visible_set"]), candidates[idx]["template_index"], candidates[idx]["top"], candidates[idx]["left"]))
        for idx in options:
            candidate = candidates[idx]
            chosen.append(idx)
            covered.update(candidate["visible_set"])
            used_full.update(candidate["full_set"])
            search()
            for pixel in candidate["visible_set"]:
                covered.remove(pixel)
            for pixel in candidate["full_set"]:
                used_full.remove(pixel)
            chosen.pop()
            if len(solutions) > 1:
                return

    search()
    assert len(solutions) == 1, "blue remnants must admit a unique compatible reconstruction"
    return [candidates[idx] for idx in sorted(solutions[0], key=lambda idx: (candidates[idx]["top"], candidates[idx]["left"], candidates[idx]["color"]))]


def _infer_solution(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    band0, band1, band_color = _find_band(g)
    templates = _extract_templates(g, bg, band0, band1, band_color)
    blue_pixels, candidates = _build_candidates(g, templates, band0, band1)
    placements = _unique_cover(blue_pixels, candidates)
    return {
        "bg": bg,
        "band0": int(band0),
        "band1": int(band1),
        "templates": templates,
        "placements": placements,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each blue remnant by matching it to the unique full template compatible with the grey occluder band.

    Input structure:
    - A black-background-like grid is crossed by one full-width horizontal grey band of positive thickness.
    - Several blue placeholder remnants lie where full objects intersect that band, so only the band-exterior cells remain visible.
    - Elsewhere outside the band sit complete monochrome template shapes in distinct colors.

    Rule:
    - Treat each colored template as a full shape that may be translated onto a blue remnant position without rotation.
    - A template matches a remnant exactly when the template cells outside the grey band coincide with the visible blue cells there.
    - Remove the grey band, replace every blue remnant with the full matched template recolored to that template's color, and keep the original templates unchanged.

    Key insight:
    - The visible blue pieces are not full-shape copies, so direct shape equality fails; the correct match is the unique template whose band-masked residue equals the remnant.
    """
    g = np.asarray(grid, dtype=int)
    info = _infer_solution(g)
    bg = int(info["bg"])
    band0 = int(info["band0"])
    band1 = int(info["band1"])

    out = g.copy()
    out[band0 : band1 + 1, :] = bg
    out[g == PLACEHOLDER_COLOR] = bg

    for placement in info["placements"]:
        patch = _template_patch(placement["mask"], placement["color"], bg)
        paste(out, patch, placement["top"], placement["left"], transparent=bg)
    return out


def _random_connected_mask(height: int, width: int) -> np.ndarray | None:
    min_fill = max(height + width - 1, int(np.ceil(TEMPLATE_FILL_RANGE[0] * height * width)))
    max_fill = min(height * width - 1, int(np.floor(TEMPLATE_FILL_RANGE[1] * height * width)))
    if min_fill > max_fill:
        return None

    for _ in range(SHAPE_RETRIES):
        target = random.randint(min_fill, max_fill)
        mask = np.zeros((height, width), dtype=bool)

        col = random.randrange(width)
        mask[0, col] = True
        for row in range(1, height):
            col = max(0, min(width - 1, col + random.choice((-1, 0, 1))))
            mask[row, col] = True

        bridge_row = random.randrange(height)
        current_col = int(np.flatnonzero(mask[bridge_row])[0])
        missing_cols = [c for c in range(width) if not mask[:, c].any()]
        random.shuffle(missing_cols)
        for goal_col in missing_cols:
            c0, c1 = sorted((current_col, goal_col))
            mask[bridge_row, c0 : c1 + 1] = True
            current_col = goal_col

        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.add((rr, cc))

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        return mask
    return None


def _has_clearance(occupied: np.ndarray, coords: frozenset[tuple[int, int]], band0: int, band1: int) -> bool:
    height, width = map(int, occupied.shape)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < height and 0 <= cc < width):
                    continue
                if band0 <= rr <= band1:
                    continue
                if occupied[rr, cc] and (rr, cc) not in coords:
                    return False
    return True


def _place_templates(canvas: np.ndarray, templates: list[dict], band0: int, band1: int, bg: int) -> list[dict] | None:
    height = int(canvas.shape[0])
    placements = []
    for template in templates:
        patch = _template_patch(template["mask"], template["color"], bg)
        options = []
        if patch.shape[0] <= band0:
            options.append((0, canvas[:band0, :]))
        if patch.shape[0] <= height - band1 - 1:
            options.append((band1 + 1, canvas[band1 + 1 :, :]))
        random.shuffle(options)
        placed = False
        for row_offset, region in options:
            try:
                sub_r, sub_c = random_free_location(region, patch, transparent=bg, border=1, padding=1)
            except ValueError:
                continue
            paste(canvas, patch, row_offset + sub_r, sub_c, transparent=bg)
            full_coords = frozenset((row_offset + sub_r + dr, sub_c + dc) for dr, dc in template["coords"])
            placements.append(
                {
                    "color": template["color"],
                    "mask": template["mask"],
                    "top": int(row_offset + sub_r),
                    "left": int(sub_c),
                    "full_set": full_coords,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return placements


def _placeholder_options(
    template: dict,
    grid_shape: tuple[int, int],
    band0: int,
    band1: int,
    occupied_full: np.ndarray,
    occupied_visible: np.ndarray,
) -> dict[str, list[dict]]:
    height, width = map(int, grid_shape)
    mask = template["mask"]
    h, w = map(int, mask.shape)
    options = {"both": [], "top": [], "bottom": []}
    for top in range(max(1, band0 - h + 1), min(height - h - 1, band1) + 1):
        for left in range(1, width - w):
            full_coords = frozenset((top + dr, left + dc) for dr, dc in template["coords"])
            visible_set, hidden_set = _split_visible(tuple(full_coords), band0, band1)
            if not visible_set or not hidden_set:
                continue
            if any(occupied_full[r, c] for r, c in full_coords):
                continue
            if not _has_clearance(occupied_visible, visible_set, band0, band1):
                continue
            has_top = any(r < band0 for r, _ in visible_set)
            has_bottom = any(r > band1 for r, _ in visible_set)
            kind = "both" if has_top and has_bottom else "top" if has_top else "bottom"
            options[kind].append(
                {
                    "template_index": int(template["index"]),
                    "color": int(template["color"]),
                    "mask": mask,
                    "top": int(top),
                    "left": int(left),
                    "full_set": full_coords,
                    "visible_set": visible_set,
                    "kind": kind,
                }
            )
    return options


def _visible_local_mask(mask: np.ndarray, top: int, band0: int, band1: int) -> np.ndarray:
    visible = np.asarray(mask, dtype=bool).copy()
    for row in range(visible.shape[0]):
        if band0 <= top + row <= band1:
            visible[row, :] = False
    return _crop_true(visible)


def _expected_output(inp: np.ndarray, truth: list[dict], band0: int, band1: int, bg: int) -> np.ndarray:
    out = inp.copy()
    out[band0 : band1 + 1, :] = bg
    out[inp == PLACEHOLDER_COLOR] = bg
    for placement in truth:
        patch = _template_patch(placement["mask"], placement["color"], bg)
        paste(out, patch, placement["top"], placement["left"], transparent=bg)
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray, templates: list[dict], truth: list[dict]) -> bool:
    try:
        solved = _infer_solution(inp)
        produced = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(produced, expected):
        return False

    solved_key = sorted((int(p["color"]), int(p["top"]), int(p["left"])) for p in solved["placements"])
    truth_key = sorted((int(p["color"]), int(p["top"]), int(p["left"])) for p in truth)
    if solved_key != truth_key:
        return False

    full_template_signatures = {_mask_signature(t["mask"]) for t in templates}
    band0, band1, _ = _find_band(inp)
    for placement in truth:
        visible_local = _visible_local_mask(placement["mask"], placement["top"], band0, band1)
        if _mask_signature(visible_local) in full_template_signatures:
            return False

        matches = []
        for template in templates:
            h, w = map(int, template["mask"].shape)
            if placement["top"] + h > inp.shape[0] or placement["left"] + w > inp.shape[1]:
                continue
            candidate_visible = _split_visible(_shift_template(template["mask"], placement["top"], placement["left"]), band0, band1)[0]
            if candidate_visible == placement["visible_set"]:
                matches.append(int(template["index"]))
        if matches != [int(placement["template_index"])]:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    bg = int(Color.BLACK)
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    band_thickness = random.randint(*BAND_THICKNESS_RANGE)
    if height < band_thickness + 10:
        return None

    band0 = random.randint(6, height - band_thickness - 6)
    band1 = band0 + band_thickness - 1

    n_templates = random.randint(*TEMPLATE_COUNT_RANGE)
    n_placeholders = random.randint(PLACEHOLDER_COUNT_MIN, n_templates)
    colors = random.sample(TEMPLATE_COLORS, n_templates)

    templates: list[dict] = []
    signatures = set()
    for idx, color in enumerate(colors):
        mask = None
        for _ in range(SHAPE_RETRIES):
            trial = _random_connected_mask(random.randint(*TEMPLATE_H_RANGE), random.randint(*TEMPLATE_W_RANGE))
            if trial is None:
                continue
            sig = _mask_signature(trial)
            if sig in signatures:
                continue
            mask = trial
            signatures.add(sig)
            break
        if mask is None:
            return None
        templates.append({"index": idx, "color": int(color), "mask": mask, "coords": tuple((int(r), int(c)) for r, c in np.argwhere(mask))})

    inp = np.full((height, width), bg, dtype=int)
    inp[band0 : band1 + 1, :] = DEFAULT_BAND_COLOR

    template_placements = _place_templates(inp, templates, band0, band1, bg)
    if template_placements is None:
        return None

    occupied_full = np.zeros_like(inp, dtype=bool)
    occupied_visible = np.zeros_like(inp, dtype=bool)
    for placement in template_placements:
        for r, c in placement["full_set"]:
            occupied_full[r, c] = True
            occupied_visible[r, c] = True

    chosen_indices = random.sample(range(n_templates), n_placeholders)
    random.shuffle(chosen_indices)
    truth: list[dict] = []
    kinds = set()
    for template_index in chosen_indices:
        template = templates[template_index]
        picked = None
        for _ in range(PLACEMENT_RETRIES):
            options = _placeholder_options(template, inp.shape, band0, band1, occupied_full, occupied_visible)
            pools = []
            if options["both"]:
                pools.extend(["both"] * 4)
            if options["top"]:
                pools.extend(["top"] * 2)
            if options["bottom"]:
                pools.extend(["bottom"] * 2)
            if not pools:
                return None
            kind = random.choice(pools)
            candidate = random.choice(options[kind])
            picked = candidate
            break
        if picked is None:
            return None

        for r, c in picked["visible_set"]:
            inp[r, c] = PLACEHOLDER_COLOR
            occupied_visible[r, c] = True
        for r, c in picked["full_set"]:
            occupied_full[r, c] = True
        truth.append(picked)
        kinds.add(picked["kind"])

    if len(kinds) == 1 and n_placeholders >= 3 and random.random() < 0.5:
        return None

    expected = _expected_output(inp, truth, band0, band1, bg)
    if not _validate_instance(inp, expected, templates, truth):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/2/003/6")
