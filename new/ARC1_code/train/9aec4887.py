from __future__ import annotations

from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, bbox_h, bbox_w, components_mask as _gen_components_mask, find_components, get_bg, paste as _gen_paste
import random as _gen_random


def _solid_bar(component: dict) -> dict | None:
    b = component["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    area = int(component["area"])
    if min(h, w) != 1 or area != max(h, w) or area <= 1:
        return None

    orientation = "h" if h == 1 else "v"
    return {
        "index": int(component["index"]),
        "color": int(component["color"]),
        "bbox": b,
        "orientation": orientation,
        "length": area,
    }


def _find_frame(bars: list[dict]) -> tuple[dict, dict, dict, dict, int, set[int]]:
    frames: list[tuple[dict, dict, dict, dict, int, set[int]]] = []

    for combo in combinations(bars, 4):
        lengths = {int(bar["length"]) for bar in combo}
        if len(lengths) != 1:
            continue
        n = lengths.pop()

        horizontals = [bar for bar in combo if bar["orientation"] == "h"]
        verticals = [bar for bar in combo if bar["orientation"] == "v"]
        if len(horizontals) != 2 or len(verticals) != 2:
            continue

        h_rows = {int(bar["bbox"][0]) for bar in horizontals}
        v_cols = {int(bar["bbox"][1]) for bar in verticals}
        if len(h_rows) != 2 or len(v_cols) != 2:
            continue

        top, bottom = sorted(horizontals, key=lambda bar: int(bar["bbox"][0]))
        left, right = sorted(verticals, key=lambda bar: int(bar["bbox"][1]))

        top_row = int(top["bbox"][0])
        bottom_row = int(bottom["bbox"][0])
        left_col = int(left["bbox"][1])
        right_col = int(right["bbox"][1])
        if bottom_row - top_row - 1 != n or right_col - left_col - 1 != n:
            continue

        if top["bbox"] != (top_row, left_col + 1, top_row, right_col - 1):
            continue
        if bottom["bbox"] != (bottom_row, left_col + 1, bottom_row, right_col - 1):
            continue
        if left["bbox"] != (top_row + 1, left_col, bottom_row - 1, left_col):
            continue
        if right["bbox"] != (top_row + 1, right_col, bottom_row - 1, right_col):
            continue

        frame_indices = {int(bar["index"]) for bar in combo}
        frames.append((top, bottom, left, right, n, frame_indices))

    assert len(frames) == 1, "expected a unique four-bar rectangular frame"
    return frames[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the colored frame around the mask, recoloring each mask pixel by its unique nearest side.

    Input structure
    - The background is the dominant color.
    - Four same-length solid bars form the top, bottom, left, and right sides of
      a rectangular frame, with the frame corners left as background.
    - All remaining foreground components share one mask color and occupy an
      n by n bbox, matching the frame interior size.

    Rule
    - Output the compact n+2 by n+2 frame with the four bars on the border.
    - Normalize the mask bbox into the frame interior.
    - For each mask pixel, compare its distances to the top, left, bottom, and
      right edges of the mask bbox; recolor it with the corresponding side color
      only when the nearest side is unique.
    - Keep tied mask pixels in the original mask color and leave all other
      interior cells as background.

    Key insight
    - The mask is not simply copied: side colors claim exactly the pixels whose
      nearest frame side is unambiguous, while equidistant pixels remain mask-colored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    components = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"
    )
    assert components, "expected foreground components"
    for i, component in enumerate(components):
        component["index"] = i

    candidate_bars = [
        bar
        for component in components
        if (bar := _solid_bar(component)) is not None
    ]
    top, bottom, left, right, n, frame_indices = _find_frame(candidate_bars)

    mask_components = [
        component for component in components if int(component["index"]) not in frame_indices
    ]
    assert mask_components, "expected non-frame mask components"
    mask_colors = {int(component["color"]) for component in mask_components}
    assert len(mask_colors) == 1, "expected one mask color"
    mask_color = mask_colors.pop()
    side_colors = [top["color"], left["color"], bottom["color"], right["color"]]
    assert mask_color not in side_colors, "mask color must be distinct from frame side colors"

    mask_coords = np.argwhere(g == mask_color)
    assert mask_coords.size > 0, "expected mask pixels"
    mask_bbox = bbox_coords(mask_coords)
    assert bbox_h(mask_bbox) == n and bbox_w(mask_bbox) == n, "mask bbox must match frame interior"
    r0, c0, _r1, _c1 = mask_bbox

    out = np.full((n + 2, n + 2), bg, dtype=int)
    out[0, 1:-1] = top["color"]
    out[-1, 1:-1] = bottom["color"]
    out[1:-1, 0] = left["color"]
    out[1:-1, -1] = right["color"]

    for r, c in mask_coords:
        rr = int(r - r0)
        cc = int(c - c0)
        distances = [rr, cc, n - 1 - rr, n - 1 - cc]
        nearest = min(distances)
        if distances.count(nearest) == 1:
            color = side_colors[distances.index(nearest)]
        else:
            color = mask_color
        out[rr + 1, cc + 1] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and object size bounds.
GEN_INTERIOR_SIZES = [3, 4, 5]
GEN_CANVAS_HEIGHTS = [14, 15, 16]
GEN_CANVAS_WIDTHS = [15, 16]
GEN_MIN_FRAME_MASK_GAP = 3

# Palette.
GEN_BACKGROUND = 0
GEN_MASK_COLOR = 8
GEN_SIDE_COLOR_POOL = list(range(1, 8))

# Mask structure.
GEN_TARGET_MASK_CELLS = {3: 6, 4: 10, 5: 16}
GEN_REQUIRED_EDGES = {"top", "bottom", "left", "right"}
GEN_CONNECTIVITY = 4
GEN_MIN_COMPONENTS = 1
GEN_MAX_COMPONENTS = 4

# Retry budget.
GEN_MAX_ATTEMPTS = 300


def _gen_edge_tags(r: int, c: int, n: int) -> set[str]:
    tags: set[str] = set()
    if r == 0:
        tags.add("top")
    if r == n - 1:
        tags.add("bottom")
    if c == 0:
        tags.add("left")
    if c == n - 1:
        tags.add("right")
    return tags


def _gen_coord_classes(n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    tied: list[tuple[int, int]] = []
    unique: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            distances = [r, c, n - 1 - r, n - 1 - c]
            if distances.count(min(distances)) == 1:
                unique.append((r, c))
            else:
                tied.append((r, c))
    return tied, unique


def _gen_fill_missing_edges(mask: np.ndarray, n: int) -> None:
    """Mutate mask until at least one selected cell touches each frame side."""
    missing_edges = set(GEN_REQUIRED_EDGES)
    for r, c in np.argwhere(mask):
        missing_edges -= _gen_edge_tags(int(r), int(c), n)

    edge_cells = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if r in (0, n - 1) or c in (0, n - 1)
    ]
    while missing_edges:
        candidates = [
            (r, c)
            for r, c in edge_cells
            if not mask[r, c] and (_gen_edge_tags(r, c, n) & missing_edges)
        ]
        best_cover = max(
            len(_gen_edge_tags(r, c, n) & missing_edges) for r, c in candidates
        )
        r, c = _gen_random.choice(
            [
                (r, c)
                for r, c in candidates
                if len(_gen_edge_tags(r, c, n) & missing_edges) == best_cover
            ]
        )
        mask[r, c] = True
        missing_edges -= _gen_edge_tags(r, c, n)


def _gen_make_mask(n: int) -> np.ndarray:
    """Return an n by n mask with target density, edge coverage, and tie/unique cells."""
    target_cells = GEN_TARGET_MASK_CELLS[n]
    tied, unique = _gen_coord_classes(n)
    mask = np.zeros((n, n), dtype=bool)

    for r, c in (_gen_random.choice(tied), _gen_random.choice(unique)):
        mask[r, c] = True

    _gen_fill_missing_edges(mask, n)
    open_cells = [
        (r, c) for r in range(n) for c in range(n) if not bool(mask[r, c])
    ]
    for r, c in _gen_random.sample(open_cells, target_cells - int(mask.sum())):
        mask[r, c] = True
    return mask


def _gen_frame_patch(n: int, side_colors: tuple[int, int, int, int]) -> np.ndarray:
    top_color, left_color, bottom_color, right_color = side_colors
    frame = np.full((n + 2, n + 2), GEN_BACKGROUND, dtype=int)
    frame[0, 1:-1] = top_color
    frame[-1, 1:-1] = bottom_color
    frame[1:-1, 0] = left_color
    frame[1:-1, -1] = right_color
    return frame


def _gen_recolor_output(
    mask: np.ndarray, side_colors: tuple[int, int, int, int]
) -> np.ndarray:
    """Render the expected output, recoloring only mask pixels with a unique nearest side."""
    n = int(mask.shape[0])
    out = _gen_frame_patch(n, side_colors)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return out

    rr = coords[:, 0]
    cc = coords[:, 1]
    distances = np.stack([rr, cc, n - 1 - rr, n - 1 - cc], axis=1)
    nearest = distances.min(axis=1)
    unique = (distances == nearest[:, None]).sum(axis=1) == 1

    out[rr[~unique] + 1, cc[~unique] + 1] = GEN_MASK_COLOR
    side_lookup = np.array(side_colors, dtype=int)
    out[rr[unique] + 1, cc[unique] + 1] = side_lookup[
        distances[unique].argmin(axis=1)
    ]
    return out


def _gen_bbox_gap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 < br0:
        row_gap = br0 - ar1
    elif br1 < ar0:
        row_gap = ar0 - br1
    else:
        row_gap = 0

    if ac1 < bc0:
        col_gap = bc0 - ac1
    elif bc1 < ac0:
        col_gap = ac0 - bc1
    else:
        col_gap = 0
    return max(row_gap, col_gap)


def _gen_valid_placements(h: int, w: int, n: int) -> list[tuple[int, int, int, int]]:
    frame_size = n + 2
    placements: list[tuple[int, int, int, int]] = []
    for fr in range(h - frame_size + 1):
        for fc in range(w - frame_size + 1):
            frame_bbox = (fr, fc, fr + frame_size - 1, fc + frame_size - 1)
            for mr in range(h - n + 1):
                for mc in range(w - n + 1):
                    mask_bbox = (mr, mc, mr + n - 1, mc + n - 1)
                    if _gen_bbox_gap(frame_bbox, mask_bbox) >= GEN_MIN_FRAME_MASK_GAP:
                        placements.append((fr, fc, mr, mc))
    return placements


def _gen_place_input(
    h: int,
    w: int,
    mask: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray | None:
    """Place frame and raw mask with the required bbox gap."""
    n = int(mask.shape[0])
    placements = _gen_valid_placements(h, w, n)
    if not placements:
        return None

    fr, fc, mr, mc = _gen_random.choice(placements)
    grid = np.full((h, w), GEN_BACKGROUND, dtype=int)
    _gen_paste(grid, frame, fr, fc, transparent=GEN_BACKGROUND)
    raw_mask = np.where(mask, GEN_MASK_COLOR, GEN_BACKGROUND)
    _gen_paste(grid, raw_mask, mr, mc, transparent=GEN_BACKGROUND)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = _gen_random.choice(GEN_INTERIOR_SIZES)
    colors = _gen_random.sample(GEN_SIDE_COLOR_POOL, 4)
    side_colors = (colors[0], colors[1], colors[2], colors[3])
    mask = _gen_make_mask(n)
    components = len(_gen_components_mask(mask, connectivity=GEN_CONNECTIVITY))
    if not (GEN_MIN_COMPONENTS <= components <= GEN_MAX_COMPONENTS):
        return None

    output = _gen_recolor_output(mask, side_colors)
    frame = _gen_frame_patch(n, side_colors)
    h = _gen_random.choice(GEN_CANVAS_HEIGHTS)
    w = _gen_random.choice(GEN_CANVAS_WIDTHS)
    inp = _gen_place_input(h, w, mask, frame)
    if inp is None:
        return None

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9aec4887 input")
