from __future__ import annotations

from itertools import combinations

import numpy as np


BG = 0
GREY = 5
PANEL_ACTIVE_W = 3
DIVIDER_COL = 4
WORKSPACE_COL0 = DIVIDER_COL + 1
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 600


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _divider_column(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert cols, "expected a full-height grey divider"
    return int(cols[-1])


def _parse_bands(grid: np.ndarray, divider: int) -> list[dict]:
    panel = grid[:, divider - PANEL_ACTIVE_W : divider]
    colors = sorted(set(map(int, panel.ravel())) - {BG, GREY})
    assert colors, "expected at least one colored band in the reference panel"

    bands = []
    occupied = np.zeros(grid.shape[0], dtype=bool)
    for color in colors:
        rows = np.where(np.any(panel == color, axis=1))[0]
        assert rows.size > 0
        assert np.all(np.diff(rows) == 1), "band rows must be contiguous"
        r0, r1 = int(rows[0]), int(rows[-1])
        for r in rows:
            cols = tuple(np.where(panel[r] == color)[0].tolist())
            if r0 == r1 or (r0 < r < r1):
                assert cols == (0, 1, 2), "single-row and interior rows must be full-width"
            else:
                assert cols in {(0,), (2,)}, "end rows must use one endpoint marker"
        assert not occupied[r0 : r1 + 1].any(), "bands must not overlap"
        occupied[r0 : r1 + 1] = True
        bands.append({"color": int(color), "r0": r0, "r1": r1})
    return sorted(bands, key=lambda b: (b["r0"], b["r1"], b["color"]))


def _band_color_at_row(bands: list[dict], row: int) -> int:
    hits = [band for band in bands if band["r0"] <= row <= band["r1"]]
    assert len(hits) == 1, "each center row must land in exactly one band"
    return int(hits[0]["color"])


def _parse_frames(grid: np.ndarray, divider: int) -> tuple[dict[int, dict], list[int]]:
    workspace = grid[:, divider + 1 :]
    colors = sorted(set(map(int, workspace.ravel())) - {BG, GREY})
    assert len(colors) >= 3, "expected 3-5 frame colors in the workspace"

    frames: dict[int, dict] = {}
    for color in colors:
        coords = np.argwhere(workspace == color)
        assert coords.size > 0
        bbox = _bbox_coords(coords)
        mask = _frame_mask(workspace.shape, bbox)
        assert np.all(mask[workspace == color]), "visible pixels must lie on the inferred frame"
        r0, c0, r1, c1 = bbox
        frames[int(color)] = {
            "color": int(color),
            "bbox": bbox,
            "mask": mask,
            "center_row": int((r0 + r1) // 2),
            "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        }

    edges: set[tuple[int, int]] = set()
    for a, b in combinations(colors, 2):
        crossing = frames[a]["mask"] & frames[b]["mask"]
        for r_raw, c_raw in np.argwhere(crossing):
            visible = int(workspace[int(r_raw), int(c_raw)])
            if visible == a:
                edges.add((b, a))
            elif visible == b:
                edges.add((a, b))

    order = _unique_path(colors, edges)
    return frames, order


def _unique_path(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: [other for other in colors if (color, other) in edges] for color in colors}
    paths: list[tuple[int, ...]] = []

    def search(path: tuple[int, ...], remaining: set[int]) -> None:
        if len(paths) > 1:
            return
        if not remaining:
            paths.append(path)
            return
        for nxt in outgoing[path[-1]]:
            if nxt in remaining:
                search((*path, nxt), remaining - {nxt})

    all_colors = set(colors)
    for start in colors:
        search((start,), all_colors - {start})
        if len(paths) > 1:
            break

    assert len(paths) == 1, "crossings must determine a unique full order"
    return list(paths[0])


def _render_nested(size: int, colors: list[int]) -> np.ndarray:
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(colors):
        inset = 2 * layer
        r0 = c0 = inset
        r1 = c1 = size - 1 - inset
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
    return out


def _solve_components(grid: np.ndarray) -> tuple[list[dict], list[int], list[int], int]:
    divider = _divider_column(grid)
    bands = _parse_bands(grid, divider)
    frames, order = _parse_frames(grid, divider)
    ordered_frames = [frames[color] for color in order]
    band_colors = [_band_color_at_row(bands, frame["center_row"]) for frame in ordered_frames]
    outer = ordered_frames[0]["bbox"]
    size = max(outer[2] - outer[0] + 1, outer[3] - outer[1] + 1)
    return ordered_frames, order, band_colors, size


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a depth-ordered stack of overlapping frames using the stretched band color at each frame's center row.

    Input structure:
    - The left side contains a thin grey-framed reference panel with three active columns and one full-height grey divider.
    - Colored rows in that panel encode compressed horizontal bands: single-row bands are full-width, while taller bands use endpoint markers on the first and last row and full-width connector rows in between.
    - To the right of the divider, several one-cell-thick rectangular frames overlap on a black workspace, and their visible crossings encode a unique depth order.

    Rule:
    - Recover every colored band from the left panel and treat it as a full-width horizontal stripe over the workspace rows.
    - Reconstruct each workspace frame from the bounding box of its visible pixels and infer the unique back-to-front order from pairwise crossing visibility.
    - For each frame in that order, look up the color of the stretched stripe containing the frame's center row, then redraw those colors as concentric square rings with one black gap between successive rings.

    Key insight:
    - The workspace crossings determine only the ring order, while the left panel determines only the ring colors; both sources are required because the correct color sequence is attached to frames through their center rows, not through the input frame colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    ordered_frames, _order, band_colors, size = _solve_components(g)
    assert len(ordered_frames) == len(band_colors)
    assert size >= 4 * len(ordered_frames) - 1, "outer frame must support spaced concentric rings"
    return _render_nested(size, band_colors)


def _is_monotonic(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _sample_center_rows(rng: np.random.Generator, h: int, n: int, outer_size: int) -> list[int] | None:
    valid_outer = list(range(outer_size // 2, h - outer_size // 2))
    if not valid_outer:
        return None

    for _ in range(200):
        outer_center = int(rng.choice(valid_outer))
        others: list[int] = []
        candidates = list(range(1, h - 1))
        rng.shuffle(candidates)
        for row in candidates:
            row = int(row)
            if row == outer_center or abs(row - outer_center) < 3:
                continue
            if all(abs(row - prev) >= 3 for prev in others):
                others.append(row)
                if len(others) == n - 1:
                    break
        if len(others) != n - 1:
            continue
        rng.shuffle(others)
        centers = [outer_center] + others
        if not _is_monotonic(centers):
            return centers
    return None


def _build_band_specs(
    rng: np.random.Generator, h: int, centers: list[int], colors: list[int]
) -> list[dict] | None:
    center_to_color = dict(zip(centers, colors, strict=True))
    sorted_centers = sorted(centers)
    specs: list[dict] = []
    eligible_for_growth: list[int] = []

    for idx, center in enumerate(sorted_centers):
        prev_center = sorted_centers[idx - 1] if idx > 0 else None
        next_center = sorted_centers[idx + 1] if idx + 1 < len(sorted_centers) else None

        options = [(0, 0)]
        can_top = center - 1 >= 1 and (prev_center is None or center - 1 > prev_center)
        can_bottom = center + 1 <= h - 2 and (next_center is None or center + 1 < next_center)
        if can_top:
            options.append((1, 0))
        if can_bottom:
            options.append((0, 1))
        if can_top and can_bottom:
            options.append((1, 1))
            eligible_for_growth.append(len(specs))

        if len(options) == 1:
            top, bottom = options[0]
        else:
            top, bottom = options[int(rng.integers(len(options)))]
            if (top, bottom) == (0, 0) and rng.random() < 0.6:
                top, bottom = options[int(rng.integers(1, len(options)))]

        specs.append(
            {
                "center": int(center),
                "color": int(center_to_color[center]),
                "r0": int(center - top),
                "r1": int(center + bottom),
                "flip": bool(rng.integers(2)),
            }
        )

    if all(spec["r0"] == spec["r1"] for spec in specs):
        growth_choices = [i for i, spec in enumerate(specs) if spec["center"] > 1 or spec["center"] < h - 2]
        if not growth_choices:
            return None
        idx = int(rng.choice(growth_choices))
        center = specs[idx]["center"]
        prev_end = specs[idx - 1]["r1"] if idx > 0 else 0
        next_start = specs[idx + 1]["r0"] if idx + 1 < len(specs) else h - 1
        if center - 1 > prev_end and center + 1 < next_start:
            specs[idx]["r0"] = center - 1
            specs[idx]["r1"] = center + 1
        elif center - 1 > prev_end:
            specs[idx]["r0"] = center - 1
        elif center + 1 < next_start:
            specs[idx]["r1"] = center + 1
        else:
            return None

    return specs


def _paint_panel(grid: np.ndarray, band_specs: list[dict]) -> None:
    panel = grid[:, DIVIDER_COL - PANEL_ACTIVE_W : DIVIDER_COL]
    for spec in band_specs:
        color = int(spec["color"])
        r0, r1 = int(spec["r0"]), int(spec["r1"])
        if r0 == r1:
            panel[r0, :] = color
            continue
        if spec["flip"]:
            panel[r0, 2] = color
            panel[r1, 0] = color
        else:
            panel[r0, 0] = color
            panel[r1, 2] = color
        if r1 - r0 > 1:
            panel[r0 + 1 : r1, :] = color


def _random_box_for_center(
    rng: np.random.Generator, center_row: int, h: int, w: int
) -> tuple[int, int, int, int] | None:
    max_half = min(center_row, h - 1 - center_row)
    heights = [value for value in range(3, min(15, 2 * max_half + 1) + 1, 2)]
    widths = [value for value in range(3, min(19, w) + 1, 2)]
    if not heights or not widths:
        return None
    box_h = int(rng.choice(heights))
    box_w = int(rng.choice(widths))
    r0 = int(center_row - box_h // 2)
    c0 = int(rng.integers(0, w - box_w + 1))
    return (r0, c0, r0 + box_h - 1, c0 + box_w - 1)


def _perimeters_overlap(shape: tuple[int, int], a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return bool(np.any(_frame_mask(shape, a) & _frame_mask(shape, b)))


def _paint_workspace(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    workspace = np.zeros(shape, dtype=int)
    for color, box in zip(colors, boxes, strict=True):
        workspace[_frame_mask(shape, box)] = int(color)
    return workspace


def _visible_spans_full_bbox(workspace: np.ndarray, colors: list[int], boxes: list[tuple[int, int, int, int]]) -> bool:
    for color, box in zip(colors, boxes, strict=True):
        coords = np.argwhere(workspace == color)
        if coords.size == 0:
            return False
        if _bbox_coords(coords) != box:
            return False
    return True


def _build_frame_boxes(
    rng: np.random.Generator, h: int, w: int, centers: list[int]
) -> tuple[list[tuple[int, int, int, int]], int] | None:
    n = len(centers)
    outer_size = 4 * n - 1
    outer_center = centers[0]
    outer_r0 = outer_center - outer_size // 2
    outer_c0 = int(rng.integers(0, w - outer_size + 1))
    boxes = [(outer_r0, outer_c0, outer_r0 + outer_size - 1, outer_c0 + outer_size - 1)]

    for center in centers[1:]:
        candidate = None
        for _ in range(300):
            box = _random_box_for_center(rng, center, h, w)
            if box is None or box in boxes:
                continue
            if not _perimeters_overlap((h, w), box, boxes[-1]):
                continue
            candidate = box
            break
        if candidate is None:
            return None
        boxes.append(candidate)
    return boxes, outer_size


def _wrong_output_by_order(frames: list[dict], bands: list[dict], size: int, key) -> np.ndarray:
    wrong_frames = sorted(frames, key=key)
    colors = [_band_color_at_row(bands, frame["center_row"]) for frame in wrong_frames]
    return _render_nested(size, colors)


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        divider = _divider_column(grid)
        bands = _parse_bands(grid, divider)
        frame_map, order = _parse_frames(grid, divider)
        frames = [frame_map[color] for color in order]
        if len(frames) != len(bands):
            return False

        center_rows = [frame["center_row"] for frame in frames]
        if _is_monotonic(center_rows):
            return False

        band_colors = [_band_color_at_row(bands, row) for row in center_rows]
        if band_colors == order:
            return False

        size = max(frames[0]["bbox"][2] - frames[0]["bbox"][0] + 1, frames[0]["bbox"][3] - frames[0]["bbox"][1] + 1)
        correct = _render_nested(size, band_colors)
        if not np.array_equal(main(grid), correct):
            return False

        wrong_row = _wrong_output_by_order(frames, bands, size, key=lambda frame: frame["center_row"])
        if np.array_equal(wrong_row, correct):
            return False

        wrong_area = _wrong_output_by_order(frames, bands, size, key=lambda frame: (-frame["area"], frame["center_row"]))
        if np.array_equal(wrong_area, correct):
            return False

        wrong_input_colors = _render_nested(size, [frame["color"] for frame in frames])
        if np.array_equal(wrong_input_colors, correct):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(3, 6))
    outer_size = 4 * n - 1

    h_min = max(outer_size + 2, 19)
    h = int(rng.integers(h_min, 27))
    w_workspace = int(rng.integers(max(outer_size + 1, 18), 26))
    w_total = WORKSPACE_COL0 + w_workspace
    if max(h, w_total) > 30:
        return None

    centers = _sample_center_rows(rng, h, n, outer_size)
    if centers is None:
        return None

    frame_colors = [int(c) for c in rng.choice(PALETTE, size=n, replace=False)]
    band_palette = [int(c) for c in rng.choice(PALETTE, size=n, replace=False)]
    band_specs = _build_band_specs(rng, h, centers, band_palette)
    if band_specs is None:
        return None

    built = _build_frame_boxes(rng, h, w_workspace, centers)
    if built is None:
        return None
    boxes, expected_outer_size = built
    if expected_outer_size != outer_size:
        return None

    workspace = _paint_workspace((h, w_workspace), boxes, frame_colors)
    if not _visible_spans_full_bbox(workspace, frame_colors, boxes):
        return None

    grid = np.zeros((h, w_total), dtype=int)
    grid[:, 0] = GREY
    grid[:, DIVIDER_COL] = GREY
    grid[0, : WORKSPACE_COL0] = GREY
    grid[-1, : WORKSPACE_COL0] = GREY
    _paint_panel(grid, band_specs)
    grid[:, WORKSPACE_COL0:] = workspace

    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {GEN_ATTEMPTS} attempts")
