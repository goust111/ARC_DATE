from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


FRAME_COLOR = Color.GREY
BG_COLOR = Color.BLACK
MARKER_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
FRAME_DIM_OPTIONS = [(2, 2), (2, 3), (2, 4), (3, 3), (2, 5), (3, 4)]
GRID_ROW_RANGE = (16, 24)
GRID_COL_RANGE = (16, 24)
FRAME_COUNT_OPTIONS = [2, 3, 4]
FRAME_COUNT_WEIGHTS = [1, 4, 3]
GAP_BBOX_PADDING = 1
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 300


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _local_component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return local


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in find_components(grid, bg=get_bg(grid), connectivity=8, colors=[FRAME_COLOR], monochromatic=True):
        local = _local_component_mask(comp)
        h, w = local.shape
        border = _border_mask(h, w)
        assert h >= 3 and w >= 3, "frame must enclose a non-empty interior"
        assert not bool(local[1:-1, 1:-1].any()), "grey pixels must stay on the bbox perimeter"
        assert bool(local[0, :].any()) and bool(local[-1, :].any())
        assert bool(local[:, 0].any()) and bool(local[:, -1].any())
        assert not bool(local[~border].any())
        frames.append(
            {
                "bbox": comp["bbox"],
                "area": int((h - 2) * (w - 2)),
                "complete": bool(np.array_equal(local, border)),
            }
        )
    assert frames, "expected at least one grey frame"
    frame_areas = [frame["area"] for frame in frames]
    assert len(frame_areas) == len(set(frame_areas)), "frame interior areas must be unique"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Pair marker counts with frame interior areas, then fill each grey frame by border completeness.

    Input structure:
    - The background is black.
    - Grey pixels form 2-4 hollow rectangular frames with one-pixel borders; some borders are complete and some have 1-3 missing edge pixels.
    - Other non-black, non-grey pixels are isolated marker pixels, and each marker color appears a number of times equal to exactly one frame's interior area.

    Rule:
    - For each grey frame, use its grey component bbox to measure the interior area `(height-2) * (width-2)`.
    - Count how many marker pixels appear in each non-grey color and pair each color to the frame whose interior area matches that count.
    - Keep the grey border unchanged.
    - If a frame border is complete, fill every interior cell with the paired color.
    - If a frame border has any missing perimeter pixel, fill only interior cells whose local `(r+c)` parity is even, leaving the alternating cells black.
    - Erase every marker pixel elsewhere.

    Key insight:
    - The marker colors identify which frame gets which fill color by count matching, while the frame's own topology decides whether that fill is solid or checkerboard.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == BG_COLOR

    marker_components = [
        comp
        for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True)
        if int(comp["color"]) not in (bg, FRAME_COLOR)
    ]
    assert marker_components, "expected colored marker pixels"
    assert all(int(comp["area"]) == 1 for comp in marker_components), "markers must be isolated single pixels"

    frames = _extract_frames(g)
    marker_colors = sorted({int(comp["color"]) for comp in marker_components})
    color_counts = {color: int(np.sum(g == color)) for color in marker_colors}
    assert len(color_counts) == len(set(color_counts.values())), "marker counts must be unique"

    areas = {frame["area"] for frame in frames}
    assert areas == set(color_counts.values()), "marker counts must match frame interior areas exactly"
    area_to_color = {count: color for color, count in color_counts.items()}

    out = g.copy()
    out[(out != bg) & (out != FRAME_COLOR)] = bg

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        fill_color = area_to_color[frame["area"]]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if frame["complete"]:
            interior[:, :] = fill_color
            continue
        rr, cc = np.indices(interior.shape)
        interior[:, :] = np.where((rr + cc) % 2 == 0, fill_color, bg)

    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = GAP_BBOX_PADDING) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _choose_frame_specs() -> list[dict]:
    n = random.choices(FRAME_COUNT_OPTIONS, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    base_dims = random.sample(FRAME_DIM_OPTIONS, n)
    specs = []
    complete_count = random.randint(1, n - 1)
    complete_ids = set(random.sample(range(n), complete_count))
    for idx, (ih, iw) in enumerate(base_dims):
        if ih != iw and random.random() < 0.5:
            ih, iw = iw, ih
        specs.append(
            {
                "inner_h": ih,
                "inner_w": iw,
                "outer_h": ih + 2,
                "outer_w": iw + 2,
                "area": ih * iw,
                "complete": idx in complete_ids,
            }
        )

    complete_areas = [spec["area"] for spec in specs if spec["complete"]]
    broken_areas = [spec["area"] for spec in specs if not spec["complete"]]
    if n >= 3:
        assert complete_areas and broken_areas
        if not (min(complete_areas) < max(broken_areas) and max(complete_areas) > min(broken_areas)):
            return []
    return specs


def _place_frames(h: int, w: int, specs: list[dict]) -> bool:
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"], reverse=True)
    placed: list[tuple[int, int, int, int]] = []

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        spec = specs[idx]
        oh, ow = spec["outer_h"], spec["outer_w"]
        candidates = [(r0, c0) for r0 in range(1, h - oh) for c0 in range(1, w - ow)]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            if any(_boxes_too_close(bbox, other) for other in placed):
                continue
            spec["bbox"] = bbox
            placed.append(bbox)
            if rec(pos + 1):
                return True
            placed.pop()
            spec.pop("bbox", None)
        return False

    return rec(0)


def _frame_patch(spec: dict) -> np.ndarray:
    patch = np.zeros((spec["outer_h"], spec["outer_w"]), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    if spec["complete"]:
        return patch

    side_cells = []
    h, w = patch.shape
    side_cells.extend((0, c) for c in range(1, w - 1))
    side_cells.extend((h - 1, c) for c in range(1, w - 1))
    side_cells.extend((r, 0) for r in range(1, h - 1))
    side_cells.extend((r, w - 1) for r in range(1, h - 1))
    for _ in range(50):
        candidate = patch.copy()
        gap_count = random.randint(1, min(3, len(side_cells)))
        for r, c in random.sample(side_cells, gap_count):
            candidate[r, c] = BG_COLOR
        comps = find_components(candidate, bg=BG_COLOR, connectivity=8, colors=[FRAME_COLOR], monochromatic=True)
        if len(comps) != 1:
            continue
        if comps[0]["bbox"] != (0, 0, h - 1, w - 1):
            continue
        return candidate
    raise RuntimeError("failed to sample a connected broken frame")


def _paint_frames(grid: np.ndarray, specs: list[dict]) -> None:
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        patch = _frame_patch(spec)
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _marker_candidates(shape: tuple[int, int], specs: list[dict]) -> dict[int, list[tuple[int, int]]]:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    parity_to_cells = {0: [], 1: []}
    for r in range(h):
        for c in range(w):
            if forbidden[r, c]:
                continue
            parity_to_cells[(r + c) % 2].append((r, c))
    return parity_to_cells


def _scatter_markers(grid: np.ndarray, specs: list[dict], colors: list[int]) -> bool:
    total = sum(spec["area"] for spec in specs)
    parity_to_cells = _marker_candidates(grid.shape, specs)
    viable = [parity for parity, cells in parity_to_cells.items() if len(cells) >= total]
    if not viable:
        return False
    parity = random.choice(viable)
    cells = random.sample(parity_to_cells[parity], total)
    random.shuffle(cells)

    cursor = 0
    for spec, color in zip(specs, colors):
        for r, c in cells[cursor : cursor + spec["area"]]:
            grid[r, c] = color
        cursor += spec["area"]
    return True


def _expected_from_specs(inp: np.ndarray, specs: list[dict], colors: list[int]) -> np.ndarray:
    out = inp.copy()
    out[(out != BG_COLOR) & (out != FRAME_COLOR)] = BG_COLOR
    for spec, color in zip(specs, colors):
        r0, c0, r1, c1 = spec["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if spec["complete"]:
            interior[:, :] = color
        else:
            rr, cc = np.indices(interior.shape)
            interior[:, :] = np.where((rr + cc) % 2 == 0, color, BG_COLOR)
    return out


def _semantic_checks(inp: np.ndarray, specs: list[dict], colors: list[int]) -> bool:
    frames = _extract_frames(inp)
    if len(frames) != len(specs):
        return False
    if not any(spec["complete"] for spec in specs):
        return False
    if not any(not spec["complete"] for spec in specs):
        return False

    expected_areas = {spec["area"] for spec in specs}
    if expected_areas != {frame["area"] for frame in frames}:
        return False

    actual_counts = {color: int(np.sum(inp == color)) for color in colors}
    if set(actual_counts.values()) != expected_areas:
        return False

    complete_areas = [spec["area"] for spec in specs if spec["complete"]]
    broken_areas = [spec["area"] for spec in specs if not spec["complete"]]
    if len(specs) >= 3 and not (min(complete_areas) < max(broken_areas) and max(complete_areas) > min(broken_areas)):
        return False

    expected = _expected_from_specs(inp, specs, colors)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _attempt_generate() -> np.ndarray | None:
    specs = _choose_frame_specs()
    if not specs:
        return None

    colors = random.sample(MARKER_COLORS, len(specs))
    random.shuffle(specs)
    specs.sort(key=lambda spec: spec["area"])
    colors = [color for _area, color in sorted(zip([spec["area"] for spec in specs], colors))]

    for _ in range(PLACEMENT_RETRIES):
        h = random.randint(*GRID_ROW_RANGE)
        w = random.randint(*GRID_COL_RANGE)
        for spec in specs:
            spec.pop("bbox", None)
        if not _place_frames(h, w, specs):
            continue

        grid = np.zeros((h, w), dtype=int)
        _paint_frames(grid, specs)
        if not _scatter_markers(grid, specs, colors):
            continue
        if _semantic_checks(grid, specs, colors):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
