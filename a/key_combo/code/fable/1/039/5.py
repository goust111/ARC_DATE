from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, neighbors4, paste, random_free_location


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
GRID_MIN = 26
GRID_MAX = 30
PALETTE = tuple(range(10))


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
    )


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_mask(coords: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(grid_shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _boundary_component_is_connected(grid: np.ndarray, boundary: set[tuple[int, int]], color: int) -> bool:
    start = next(iter(boundary))
    h, w = grid.shape
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != int(color):
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return boundary <= seen


def _uniform_surround_color(grid: np.ndarray, coords: np.ndarray, cavity_color: int) -> int | None:
    h, w = grid.shape
    boundary: set[tuple[int, int]] = set()
    colors: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(grid[nr, nc]) == int(cavity_color):
                continue
            boundary.add((nr, nc))
            colors.add(int(grid[nr, nc]))
    if len(colors) != 1 or not boundary:
        return None
    surround = next(iter(colors))
    if not _boundary_component_is_connected(grid, boundary, surround):
        return None
    return surround


def _collect_cavity_summary(grid: np.ndarray, color: int) -> dict | None:
    comps = components_mask(grid == int(color), connectivity=4)
    border_comps = []
    records = []
    for coords in comps:
        if _touches_border(coords, grid.shape):
            border_comps.append(coords)
            continue
        surround = _uniform_surround_color(grid, coords, color)
        if surround is None:
            continue
        records.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "shape": _normalize_coords(coords),
                "surround": int(surround),
                "area": int(coords.shape[0]),
            }
        )
    if not border_comps or len(records) < 3:
        return None
    shape_counts = Counter(record["shape"] for record in records)
    singletons = [shape for shape, count in shape_counts.items() if count == 1]
    if len(singletons) != 1 or len(shape_counts) < 2:
        return None
    return {
        "color": int(color),
        "border_area": int(sum(comp.shape[0] for comp in border_comps)),
        "records": records,
        "shape_counts": shape_counts,
        "unique_shape": singletons[0],
    }


def _select_answer_record(grid: np.ndarray) -> tuple[int, dict]:
    candidates = []
    for color in sorted(map(int, np.unique(grid))):
        summary = _collect_cavity_summary(grid, color)
        if summary is None:
            continue
        score = (
            int(summary["border_area"]),
            len(summary["records"]),
            sum(record["area"] for record in summary["records"]),
        )
        candidates.append((score, summary))
    assert candidates, "expected one color to act as background with enclosed cavities"
    best_score = max(score for score, _summary in candidates)
    best = [summary for score, summary in candidates if score == best_score]
    assert len(best) == 1, "background color selection is ambiguous"
    summary = best[0]
    answer = [record for record in summary["records"] if record["shape"] == summary["unique_shape"]]
    assert len(answer) == 1, "expected exactly one outlier cavity"
    return summary["color"], answer[0]


def _render_cavity_crop(bg: int, record: dict, grid_shape: tuple[int, int]) -> np.ndarray:
    mask = _shape_mask(record["coords"], grid_shape)
    cropped = crop_bbox(mask.astype(int), record["bbox"]).astype(bool)
    out = np.full(cropped.shape, int(bg), dtype=int)
    out[cropped] = int(record["surround"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the uniquely shaped enclosed background cavity and recolor it with its enclosing object.

    Input structure:
    - One color serves as background both outside the foreground clutter and inside sealed cavities.
    - Several background components do not touch the grid border because monochrome foreground objects enclose them.
    - The enclosed cavities are compared by translation-equivalent shape; exactly one shape occurs once and every other cavity shape repeats.

    Rule:
    - Enumerate all non-border connected components of the background color and keep the ones whose entire 4-neighborhood belongs to one monochrome enclosing foreground component.
    - Group those enclosed cavities by normalized shape, considering only translation and ignoring position.
    - Select the cavity shape that appears exactly once and return its tight crop, painting the cavity cells with the enclosing object's color and leaving the rest of the crop as background.

    Key insight:
    - The target object is negative space: the decisive comparison is among enclosed holes in the background, not among the visible foreground blobs that surround them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, answer = _select_answer_record(g)
    return _render_cavity_crop(bg, answer, g.shape)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_coords(np.argwhere(mask))


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    out = np.rot90(out, int(rng.integers(0, 4)))
    return np.array(out, dtype=bool, copy=True)


def _random_connected_mask(
    rng: np.random.Generator,
    area_lo: int,
    area_hi: int,
    max_h: int,
    max_w: int,
    banned: set[tuple[tuple[int, int], ...]],
) -> np.ndarray | None:
    for _ in range(300):
        area = int(rng.integers(area_lo, area_hi + 1))
        cells = {(0, 0)}
        min_r = max_r = min_c = max_c = 0
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    nr0 = min(min_r, nr)
                    nr1 = max(max_r, nr)
                    nc0 = min(min_c, nc)
                    nc1 = max(max_c, nc)
                    if nr1 - nr0 + 1 <= max_h and nc1 - nc0 + 1 <= max_w:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(0, len(frontier)))]
            cells.add((nr, nc))
            min_r = min(min_r, nr)
            max_r = max(max_r, nr)
            min_c = min(min_c, nc)
            max_c = max(max_c, nc)
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h < 2 or w < 2 or area == h * w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        mask = _random_transform(mask, rng)
        sig = _mask_signature(mask)
        if sig in banned:
            continue
        banned.add(sig)
        return mask
    return None


def _build_host_patch(mask: np.ndarray, color: int, bg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))
    patch = np.full((h + top + bottom, w + left + right), int(color), dtype=int)
    cavity = np.zeros_like(patch, dtype=bool)
    cavity[top : top + h, left : left + w] = mask
    patch[cavity] = int(bg)
    return patch, cavity


def _make_open_bay_patch(
    rng: np.random.Generator, color: int, bg: int, min_area: int
) -> tuple[np.ndarray, int, int]:
    for _ in range(100):
        interior_h = int(rng.integers(4, 7))
        interior_w = int(rng.integers(5, 9))
        if interior_h * interior_w <= min_area:
            continue
        thickness = int(rng.integers(1, 3))
        patch = np.full((interior_h + thickness, interior_w + 2 * thickness), int(bg), dtype=int)
        patch[:, :thickness] = int(color)
        patch[:, -thickness:] = int(color)
        patch[-thickness:, :] = int(color)
        turns = int(rng.integers(0, 4))
        patch = np.rot90(patch, turns)
        return patch, interior_h * interior_w, turns
    raise RuntimeError("failed to build open bay")


def _place_open_bay(
    grid: np.ndarray, layout: np.ndarray, rng: np.random.Generator, color: int, bg: int, min_area: int
) -> int | None:
    patch, bay_area, turns = _make_open_bay_patch(rng, color, bg, min_area)
    ph, pw = patch.shape
    h, w = grid.shape
    if turns == 0:
        r0 = 0
        c0 = int(rng.integers(0, w - pw + 1))
    elif turns == 1:
        r0 = int(rng.integers(0, h - ph + 1))
        c0 = 0
    elif turns == 2:
        r0 = h - ph
        c0 = int(rng.integers(0, w - pw + 1))
    else:
        r0 = int(rng.integers(0, h - ph + 1))
        c0 = w - pw
    if bool(np.any(layout[r0 : r0 + ph, c0 : c0 + pw])):
        return None
    footprint = np.ones_like(patch, dtype=int)
    paste(layout, footprint, r0, c0, transparent=None)
    paste(grid, patch, r0, c0, transparent=None)
    return bay_area


def _multicolor_blob_patch(rng: np.random.Generator, colors: list[int], bg: int) -> np.ndarray | None:
    mask = _random_connected_mask(rng, 6, 12, 5, 6, set())
    if mask is None:
        return None
    coords = np.argwhere(mask)
    use = list(map(int, rng.choice(colors, size=int(rng.integers(2, min(4, len(colors)) + 1)), replace=False)))
    patch = np.full(mask.shape, int(bg), dtype=int)
    assigned = []
    for color in use:
        assigned.append(color)
    while len(assigned) < len(coords):
        assigned.append(int(rng.choice(use)))
    rng.shuffle(assigned)
    for (r, c), color in zip(coords, assigned):
        patch[int(r), int(c)] = int(color)
    return patch


def _noise_patch(rng: np.random.Generator, colors: list[int], bg: int) -> np.ndarray:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        patch = np.array([[int(rng.choice(colors))]], dtype=int)
    elif kind == 1:
        patch = np.full((1, 2), int(bg), dtype=int)
        vals = rng.choice(colors, size=2, replace=True)
        patch[0, 0] = int(vals[0])
        patch[0, 1] = int(vals[1])
    else:
        patch = np.full((2, 2), int(bg), dtype=int)
        vals = rng.choice(colors, size=3, replace=True)
        patch[0, 0] = int(vals[0])
        patch[1, 0] = int(vals[1])
        patch[1, 1] = int(vals[2])
        if bool(rng.integers(0, 2)):
            patch = np.rot90(patch)
    return patch


def _largest_foreground_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    comps = components_mask(grid != int(bg), connectivity=4)
    largest = max(comps, key=lambda coords: int(coords.shape[0]))
    bbox = bbox_coords(largest)
    r0, c0, r1, c1 = bbox
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    out[largest[:, 0] - r0, largest[:, 1] - c0] = grid[largest[:, 0], largest[:, 1]]
    return out


def _count_multicolor_components(grid: np.ndarray, bg: int) -> int:
    total = 0
    for coords in components_mask(grid != int(bg), connectivity=4):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) > 1:
            total += 1
    return total


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    bg = int(rng.choice(PALETTE))
    fg_colors = [color for color in PALETTE if color != bg]
    bay_color = int(rng.choice(fg_colors))
    inner_colors = [color for color in fg_colors if color != bay_color]

    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), int(bg), dtype=int)
    layout = np.zeros((h, w), dtype=int)

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    answer_mask = _random_connected_mask(rng, 4, 6, 4, 5, used_shapes)
    if answer_mask is None:
        return None
    answer_area = int(answer_mask.sum())

    large_mask = _random_connected_mask(rng, max(answer_area + 1, 7), 10, 5, 6, used_shapes)
    if large_mask is None:
        return None
    extra_group_count = int(rng.integers(1, 3))
    group_specs = [{"mask": answer_mask, "count": 1, "answer": True}, {"mask": large_mask, "count": int(rng.integers(2, 4)), "answer": False}]
    for _ in range(extra_group_count):
        extra_mask = _random_connected_mask(rng, 3, 8, 5, 6, used_shapes)
        if extra_mask is None:
            return None
        group_specs.append({"mask": extra_mask, "count": 2, "answer": False})

    bay_area = _place_open_bay(grid, layout, rng, bay_color, bg, answer_area + 4)
    if bay_area is None or bay_area <= answer_area:
        return None

    instances = []
    for spec in group_specs:
        for _ in range(spec["count"]):
            instances.append(spec)
    rng.shuffle(instances)

    answer_record = None
    for spec in instances:
        host_color = int(rng.choice(inner_colors))
        patch, cavity = _build_host_patch(spec["mask"], host_color, bg, rng)
        footprint = np.ones_like(patch, dtype=int)
        try:
            r0, c0 = random_free_location(
                layout,
                footprint,
                transparent=0,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(layout, footprint, r0, c0, transparent=None)
        paste(grid, patch, r0, c0, transparent=None)
        coords = np.argwhere(cavity)
        coords = coords + np.array([r0, c0], dtype=int)
        record = {
            "coords": coords,
            "bbox": bbox_coords(coords),
            "shape": _mask_signature(spec["mask"]),
            "surround": host_color,
            "area": int(coords.shape[0]),
        }
        if spec["answer"]:
            answer_record = record

    assert answer_record is not None

    for _ in range(int(rng.integers(1, 3))):
        patch = _multicolor_blob_patch(rng, inner_colors, bg)
        if patch is None:
            return None
        footprint = np.ones_like(patch, dtype=int)
        try:
            r0, c0 = random_free_location(
                layout,
                footprint,
                transparent=0,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(layout, footprint, r0, c0, transparent=None)
        paste(grid, patch, r0, c0, transparent=None)

    for _ in range(int(rng.integers(6, 12))):
        patch = _noise_patch(rng, inner_colors, bg)
        footprint = np.ones_like(patch, dtype=int)
        try:
            r0, c0 = random_free_location(
                layout,
                footprint,
                transparent=0,
                border=1,
                padding=0,
                padding_connectivity=8,
            )
        except ValueError:
            break
        paste(layout, footprint, r0, c0, transparent=None)
        paste(grid, patch, r0, c0, transparent=None)

    meta = {
        "bg": bg,
        "answer": answer_record,
        "open_bay_area": int(bay_area),
    }
    if not _generated_case_is_valid(grid, meta):
        return None
    return grid, meta


def _generated_case_is_valid(grid: np.ndarray, meta: dict) -> bool:
    bg = int(meta["bg"])
    summary = _collect_cavity_summary(grid, bg)
    if summary is None:
        return False
    shape_counts = summary["shape_counts"]
    if sum(count == 1 for count in shape_counts.values()) != 1:
        return False
    answer = meta["answer"]
    expected = _render_cavity_crop(bg, answer, grid.shape)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    closed_areas = [record["area"] for record in summary["records"]]
    if max(closed_areas) <= int(answer["area"]):
        return False
    if int(meta["open_bay_area"]) <= int(answer["area"]):
        return False
    if np.array_equal(_largest_foreground_crop(grid, bg), expected):
        return False
    if _count_multicolor_components(grid, bg) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid cavity-outlier input")
