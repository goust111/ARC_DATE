from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
FRAME = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
SEED_COLORS = (Color.PINK, Color.ORANGE)

TIE_OUTER_SHAPES = [(6, 7), (6, 8), (7, 6), (7, 8), (8, 6), (8, 7)]
NO_TIE_OUTER_SHAPES = [(8, 8)]

GRID_SHAPES = {
    2: ((18, 24), (18, 26)),
    3: ((22, 30), (22, 30)),
}

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_frame_component(comp: dict) -> bool:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 6 or h > 8 or w < 6 or w > 8:
        return False
    local = np.zeros((h, w), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    local[shifted[:, 0], shifted[:, 1]] = True
    return np.array_equal(local, _border_mask(h, w))


def _extract_side_colors(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = map(int, bbox)

    def marker_on(values: np.ndarray) -> int:
        vals = [int(v) for v in values if int(v) != FRAME]
        assert len(vals) == 1, "each frame side must contain exactly one non-grey marker"
        assert vals[0] in MARKER_COLORS, "side marker must use one of the legend colors"
        return vals[0]

    top = marker_on(grid[r0, c0 : c1 + 1])
    bottom = marker_on(grid[r1, c0 : c1 + 1])
    left = marker_on(grid[r0 : r1 + 1, c0])
    right = marker_on(grid[r0 : r1 + 1, c1])
    assert len({top, bottom, left, right}) == 4, "a frame must use four distinct side markers"
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _nearest_side_or_tie(
    centroid_r: float,
    centroid_c: float,
    bbox: tuple[int, int, int, int],
    side_colors: dict[str, int],
    seed_color: int,
) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    distances = {
        "top": centroid_r - r0,
        "bottom": r1 - centroid_r,
        "left": centroid_c - c0,
        "right": c1 - centroid_c,
    }
    nearest = min(distances.values())
    sides = [side for side, dist in distances.items() if abs(dist - nearest) < 1e-9]
    if len(sides) != 1:
        return int(seed_color)
    return int(side_colors[sides[0]])


def _paint_frame(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    seed_color: int,
    side_colors: dict[str, int],
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    seed_r, seed_c = map(int, seed)
    out[r0 + 1 : r1, seed_c] = int(seed_color)
    out[seed_r, c0 + 1 : c1] = int(seed_color)

    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    seed_rr = seed_r - (r0 + 1)
    seed_cc = seed_c - (c0 + 1)
    region_mask = np.ones((inner_h, inner_w), dtype=bool)
    region_mask[seed_rr, :] = False
    region_mask[:, seed_cc] = False
    regions = components_mask(region_mask, connectivity=4)
    assert len(regions) == 4, "the seed cross must split the interior into four regions"

    for coords in regions:
        global_rows = coords[:, 0].astype(int) + (r0 + 1)
        global_cols = coords[:, 1].astype(int) + (c0 + 1)
        centroid_r = float(global_rows.mean())
        centroid_c = float(global_cols.mean())
        color = _nearest_side_or_tie(centroid_r, centroid_c, bbox, side_colors, seed_color)
        out[global_rows, global_cols] = int(color)


def _parse_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frame_components = [comp for comp in components if _is_frame_component(comp)]
    frame_ids = {id(comp) for comp in frame_components}
    seed_components = []
    for comp in components:
        if id(comp) in frame_ids:
            continue
        coords = np.asarray(comp["coords"], dtype=int)
        if int(comp["area"]) != 1:
            continue
        value = int(g[coords[0, 0], coords[0, 1]])
        if value in SEED_COLORS:
            seed_components.append(comp)

    assert 2 <= len(frame_components) <= 3, "expected two or three frames"
    assert len(seed_components) == len(frame_components), "each frame must contain one seed"

    frames = []
    for comp in sorted(frame_components, key=lambda item: tuple(map(int, item["bbox"]))):
        bbox = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = bbox
        side_colors = _extract_side_colors(g, bbox)

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        interior_colors = set(map(int, np.unique(interior)))
        assert interior_colors <= {BG, *SEED_COLORS}, "frame interior may only contain background and its seed"

        matching_seeds = []
        for seed_comp in seed_components:
            seed_r, seed_c = map(int, seed_comp["coords"][0])
            if r0 < seed_r < r1 and c0 < seed_c < c1:
                matching_seeds.append((seed_r, seed_c))
        assert len(matching_seeds) == 1, "each frame must enclose exactly one seed"

        seed = matching_seeds[0]
        seed_color = int(g[seed])
        frames.append(
            {
                "bbox": bbox,
                "seed": seed,
                "seed_color": seed_color,
                "side_colors": side_colors,
            }
        )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each boxed seed into a cross and color the four induced regions by their centroid's nearest frame side.

    Input structure:
    - The background is black.
    - Two or three non-touching grey hollow rectangular frames appear on the canvas.
    - Each frame has one colored marker embedded in each of its four sides, and one isolated pink or orange seed inside.

    Rule:
    - For every frame, extend the interior seed upward, downward, leftward, and rightward until the rays reach the frame wall; keep the seed and all ray pixels in the seed's color.
    - Treat the rays as separators, so the interior splits into four rectangular regions that exclude the ray row and ray column.
    - Compute each region's centroid and compare its distances to the frame's top, bottom, left, and right sides.
    - Fill the region with the marker color of the unique nearest side; if the nearest distance is tied, fill that region with the seed color instead.

    Key insight:
    - The quadrant colors are not fixed by position: the seed offset changes the region centroids, and the nearest frame side to each centroid determines the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    out = g.copy()
    for frame in _parse_frames(g):
        _paint_frame(out, frame["bbox"], frame["seed"], frame["seed_color"], frame["side_colors"])
    return out


def _seed_candidates(inner_h: int, inner_w: int, require_tie: bool) -> list[tuple[int, int]]:
    candidates = []
    for seed_r in range(1, inner_h - 1):
        for seed_c in range(1, inner_w - 1):
            up = seed_r
            down = inner_h - seed_r - 1
            left = seed_c
            right = inner_w - seed_c - 1
            areas = [up * left, up * right, down * left, down * right]
            if len(set(areas)) != 4:
                continue
            has_tie = any((up == left, up == right, down == left, down == right))
            if has_tie == require_tie:
                candidates.append((seed_r, seed_c))
    return candidates


def _frame_input_patch(
    outer_h: int,
    outer_w: int,
    side_colors: dict[str, int],
    seed: tuple[int, int],
    seed_color: int,
) -> np.ndarray:
    patch = np.full((outer_h, outer_w), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME

    patch[0, outer_w // 2] = int(side_colors["top"])
    patch[-1, outer_w // 2] = int(side_colors["bottom"])
    patch[outer_h // 2, 0] = int(side_colors["left"])
    patch[outer_h // 2, -1] = int(side_colors["right"])

    seed_r, seed_c = seed
    patch[seed_r + 1, seed_c + 1] = int(seed_color)
    return patch


def _frame_output_patch(
    outer_h: int,
    outer_w: int,
    side_colors: dict[str, int],
    seed: tuple[int, int],
    seed_color: int,
) -> np.ndarray:
    patch = _frame_input_patch(outer_h, outer_w, side_colors, seed, seed_color)
    seed_r, seed_c = seed
    absolute_seed = (seed_r + 1, seed_c + 1)
    patch[1:-1, absolute_seed[1]] = int(seed_color)
    patch[absolute_seed[0], 1:-1] = int(seed_color)

    quadrants = [
        (slice(1, absolute_seed[0]), slice(1, absolute_seed[1])),
        (slice(1, absolute_seed[0]), slice(absolute_seed[1] + 1, outer_w - 1)),
        (slice(absolute_seed[0] + 1, outer_h - 1), slice(1, absolute_seed[1])),
        (slice(absolute_seed[0] + 1, outer_h - 1), slice(absolute_seed[1] + 1, outer_w - 1)),
    ]

    for row_slice, col_slice in quadrants:
        rows = np.arange(row_slice.start, row_slice.stop, dtype=float)
        cols = np.arange(col_slice.start, col_slice.stop, dtype=float)
        assert rows.size > 0 and cols.size > 0
        centroid_r = float(rows.mean())
        centroid_c = float(cols.mean())
        color = _nearest_side_or_tie(
            centroid_r,
            centroid_c,
            (0, 0, outer_h - 1, outer_w - 1),
            side_colors,
            seed_color,
        )
        patch[row_slice, col_slice] = int(color)

    patch[1:-1, absolute_seed[1]] = int(seed_color)
    patch[absolute_seed[0], 1:-1] = int(seed_color)
    return patch


def _make_frame_spec(require_tie: bool) -> dict:
    shapes = TIE_OUTER_SHAPES if require_tie else NO_TIE_OUTER_SHAPES
    outer_h, outer_w = random.choice(shapes)
    inner_h = outer_h - 2
    inner_w = outer_w - 2
    candidates = _seed_candidates(inner_h, inner_w, require_tie=require_tie)
    assert candidates, "selected frame shape must admit at least one valid seed"

    seed = random.choice(candidates)
    top, bottom, left, right = random.sample(MARKER_COLORS, 4)
    side_colors = {"top": top, "bottom": bottom, "left": left, "right": right}
    seed_color = random.choice(SEED_COLORS)
    input_patch = _frame_input_patch(outer_h, outer_w, side_colors, seed, seed_color)
    output_patch = _frame_output_patch(outer_h, outer_w, side_colors, seed, seed_color)
    return {
        "outer_h": outer_h,
        "outer_w": outer_w,
        "seed": seed,
        "seed_color": int(seed_color),
        "side_colors": side_colors,
        "require_tie": bool(require_tie),
        "input_patch": input_patch,
        "output_patch": output_patch,
    }


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 + 1 < br0 or br1 + 1 < ar0:
        return True
    if ac1 + 1 < bc0 or bc1 + 1 < ac0:
        return True
    return False


def _frame_has_tie(spec: dict) -> bool:
    inner_h = int(spec["outer_h"]) - 2
    inner_w = int(spec["outer_w"]) - 2
    seed_r, seed_c = map(int, spec["seed"])
    up = seed_r
    down = inner_h - seed_r - 1
    left = seed_c
    right = inner_w - seed_c - 1
    return any((up == left, up == right, down == left, down == right))


def _place_specs(specs: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    n_frames = len(specs)
    row_range, col_range = GRID_SHAPES[n_frames]
    height = random.randint(*row_range)
    width = random.randint(*col_range)
    grid = np.full((height, width), BG, dtype=int)
    expected = np.full_like(grid, BG)
    reserved = np.full_like(grid, BG)

    placed = []
    order = sorted(specs, key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    for spec in order:
        block = np.ones((spec["outer_h"], spec["outer_w"]), dtype=int)
        try:
            r0, c0 = random_free_location(
                reserved,
                block,
                transparent=BG,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(grid, spec["input_patch"], r0, c0, transparent=BG)
        paste(expected, spec["output_patch"], r0, c0, transparent=BG)
        paste(reserved, block, r0, c0)

        placed_spec = dict(spec)
        placed_spec["bbox"] = (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1)
        placed.append(placed_spec)

    return grid, expected, placed


def _validate_instance(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    tie_flags = [_frame_has_tie(spec) for spec in specs]
    if not any(tie_flags):
        return False
    if not any(not flag for flag in tie_flags):
        return False

    for i, spec in enumerate(specs):
        bbox = spec["bbox"]
        r0, c0, r1, c1 = bbox
        if not (
            0 <= r0 < r1 < grid.shape[0]
            and 0 <= c0 < c1 < grid.shape[1]
        ):
            return False
        for other in specs[i + 1 :]:
            if not _boxes_separate(bbox, other["bbox"]):
                return False

        inner_h = int(spec["outer_h"]) - 2
        inner_w = int(spec["outer_w"]) - 2
        seed_r, seed_c = map(int, spec["seed"])
        if not (1 <= seed_r <= inner_h - 2 and 1 <= seed_c <= inner_w - 2):
            return False
        up = seed_r
        down = inner_h - seed_r - 1
        left = seed_c
        right = inner_w - seed_c - 1
        if min(up, down, left, right) < 1:
            return False
        if len({up * left, up * right, down * left, down * right}) != 4:
            return False
        if len(set(spec["side_colors"].values())) != 4:
            return False

        actual_patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(actual_patch, spec["input_patch"]):
            return False

    try:
        parsed = _parse_frames(grid)
        actual = main(grid)
    except Exception:
        return False
    if len(parsed) != len(specs):
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice((2, 3))
    requirements = [True, False]
    if n_frames == 3:
        requirements.append(random.choice((True, False)))
    random.shuffle(requirements)

    specs = [_make_frame_spec(require_tie=req) for req in requirements]
    placed = _place_specs(specs)
    if placed is None:
        return None
    grid, expected, placed_specs = placed
    if not _validate_instance(grid, expected, placed_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
