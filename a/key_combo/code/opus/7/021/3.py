from __future__ import annotations

import numpy as np


BG = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, ignore: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in ignore:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return comps


def _pack_top_aligned(crops: list[np.ndarray]) -> np.ndarray:
    assert crops
    out_h = max(crop.shape[0] for crop in crops)
    out_w = sum(crop.shape[1] for crop in crops)
    out = np.zeros((out_h, out_w), dtype=int)
    c0 = 0
    for crop in crops:
        h, w = crop.shape
        out[:h, c0 : c0 + w] = crop
        c0 += w
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    grey_cells = np.argwhere(g == MARKER)
    assert grey_cells.size > 0
    assert all(int(r) in {0, h - 1} for r, _ in grey_cells), "markers must lie on the top or bottom edge"

    components = _find_components(g, ignore={BG, MARKER})
    assert components, "expected at least one sprite"

    markers = []
    used_components: set[int] = set()
    for edge, row in (("top", 0), ("bottom", h - 1)):
        for col in np.where(g[row] == MARKER)[0]:
            matches = []
            for idx, comp in enumerate(components):
                _r0, c0, _r1, c1 = comp["bbox"]
                if c0 <= int(col) <= c1:
                    matches.append(idx)
            assert len(matches) == 1, "each marker column must select exactly one sprite"
            comp_idx = matches[0]
            assert comp_idx not in used_components, "each sprite may be selected by at most one marker"
            used_components.add(comp_idx)
            markers.append({"edge": edge, "col": int(col), "component_idx": comp_idx})

    assert markers, "expected at least one marker"
    markers.sort(key=lambda item: item["col"])

    component_order = sorted(range(len(components)), key=lambda idx: components[idx]["bbox"][1])
    x_rank = {idx: rank for rank, idx in enumerate(component_order)}
    for comp in components:
        bbox = comp["bbox"]
        comp["crop"] = _crop_bbox(g, bbox)

    return {
        "grid": g,
        "shape": (h, w),
        "components": components,
        "markers": markers,
        "component_order": component_order,
        "x_rank": x_rank,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select sprites by top/bottom boundary markers, vertically mirror only the bottom-marked ones, and pack the crops left-to-right.

    Input structure:
    - A black background contains 4-6 separated monochrome 8-connected sprites with distinct colors.
    - Grey marker pixels appear only on the top row and bottom row.
    - Each marker column overlaps exactly one sprite bbox, while some unmarked sprites may act as distractors.

    Rule:
    - Find every grey marker on the top and bottom boundary.
    - For each marker, select the unique sprite whose bbox column range contains that marker column.
    - Crop the selected sprite to its bbox; keep top-marked crops unchanged and vertically flip bottom-marked crops.
    - Sort the selected crops by marker column from left to right.
    - Concatenate the resulting crops horizontally with no gaps, top-aligning them and padding shorter crops with black.

    Key insight:
    - The boundary marker does two jobs at once: its column chooses which sprite to extract, and its edge encodes whether that crop should be mirrored.
    """
    analysis = _analyze(grid)
    crops: list[np.ndarray] = []
    for marker in analysis["markers"]:
        crop = analysis["components"][marker["component_idx"]]["crop"]
        crops.append(crop if marker["edge"] == "top" else np.flipud(crop))
    return _pack_top_aligned(crops)


def _mask_connected(mask: np.ndarray) -> bool:
    cells = list(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _random_mask(height: int, width: int, *, require_non_ud: bool) -> np.ndarray:
    area = height * width
    min_cells = max(4, (area + 1) // 2)
    for _ in range(400):
        fill_prob = RNG.uniform(0.45, 0.72)
        mask = RNG.random((height, width)) < fill_prob
        count = int(mask.sum())
        if count < min_cells or count >= area:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if require_non_ud and np.array_equal(mask, np.flipud(mask)):
            continue
        if not _mask_connected(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a valid sprite mask")


def _build_sprite(height: int, width: int, color: int, *, require_non_ud: bool) -> np.ndarray:
    return _random_mask(height, width, require_non_ud=require_non_ud).astype(int) * int(color)


def _make_selected_slots(total: int, selected_count: int) -> list[bool]:
    distractor_count = total - selected_count
    assert distractor_count >= 1
    while True:
        slots = [True] * total
        interior = int(RNG.integers(1, total - 1))
        slots[interior] = False
        remaining = distractor_count - 1
        if remaining > 0:
            candidates = [idx for idx in range(total) if idx != interior]
            for idx in RNG.choice(candidates, size=remaining, replace=False):
                slots[int(idx)] = False
        if any(slots[:interior]) and any(slots[interior + 1 :]):
            return slots


def _assign_edges(selected_slots: list[bool]) -> list[str | None]:
    selected_indices = [idx for idx, flag in enumerate(selected_slots) if flag]
    start_top = bool(RNG.integers(0, 2))
    edges: list[str | None] = [None] * len(selected_slots)
    for rank, idx in enumerate(selected_indices):
        edge = "top" if (rank % 2 == 0) == start_top else "bottom"
        edges[idx] = edge
    return edges


def _distribute_slack(parts: list[int], slack: int) -> list[int]:
    extra = parts[:]
    for _ in range(slack):
        extra[int(RNG.integers(0, len(extra)))] += 1
    return extra


def _pack_markers(markers: list[dict], components: list[dict]) -> np.ndarray:
    crops: list[np.ndarray] = []
    for marker in markers:
        crop = components[marker["component_idx"]]["crop"]
        crops.append(crop if marker["edge"] == "top" else np.flipud(crop))
    return _pack_top_aligned(crops)


def _wrong_outputs(analysis: dict) -> list[np.ndarray]:
    markers = analysis["markers"]
    components = analysis["components"]

    ordered = [components[m["component_idx"]]["crop"] for m in markers]
    wrong = [
        _pack_top_aligned(ordered),
        _pack_top_aligned([np.flipud(crop) for crop in ordered]),
        _pack_top_aligned([components[idx]["crop"] for idx in analysis["component_order"]]),
    ]

    top_first = [m for m in markers if m["edge"] == "top"] + [m for m in markers if m["edge"] == "bottom"]
    bottom_first = [m for m in markers if m["edge"] == "bottom"] + [m for m in markers if m["edge"] == "top"]
    wrong.append(_pack_markers(top_first, components))
    wrong.append(_pack_markers(bottom_first, components))
    return wrong


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    analysis = _analyze(grid)
    components = analysis["components"]
    markers = analysis["markers"]

    assert 4 <= len(components) <= 6
    assert 3 <= len(markers) <= 5
    assert len(markers) < len(components)
    assert {m["edge"] for m in markers} == {"top", "bottom"}

    edge_seq = [m["edge"] for m in markers]
    assert edge_seq != sorted(edge_seq)
    assert edge_seq != sorted(edge_seq, reverse=True)

    selected_indices = {m["component_idx"] for m in markers}
    selected_heights = [components[idx]["crop"].shape[0] for idx in selected_indices]
    assert len(set(selected_heights)) >= 2, "selected crops must have different heights for top-alignment to matter"

    for idx in selected_indices:
        crop = components[idx]["crop"]
        assert not np.array_equal(crop, np.flipud(crop)), "selected shapes must make vertical mirroring visible"

    x_ranks = sorted(analysis["x_rank"][idx] for idx in selected_indices)
    assert any(rank not in x_ranks for rank in range(min(x_ranks), max(x_ranks) + 1)), "a distractor must sit between selected sprites"

    out = main(grid)
    assert np.array_equal(out, expected)
    for wrong in _wrong_outputs(analysis):
        assert not np.array_equal(out, wrong), "generated sample allows a shortcut interpretation"


def _gen_attempt() -> np.ndarray | None:
    selected_count = int(RNG.integers(3, 6))
    total = int(RNG.integers(max(4, selected_count + 1), 7))
    selected_slots = _make_selected_slots(total, selected_count)
    edges = _assign_edges(selected_slots)

    widths = [int(RNG.integers(3, 5)) for _ in range(total)]
    heights = [int(RNG.integers(3, 5)) for _ in range(total)]
    selected_idx = [idx for idx, flag in enumerate(selected_slots) if flag]
    if {heights[idx] for idx in selected_idx} == {3} or {heights[idx] for idx in selected_idx} == {4}:
        heights[selected_idx[0]] = 3
        heights[selected_idx[-1]] = 4

    min_width = 2 + sum(widths) + (total - 1)
    if min_width > 30:
        return None
    slack = int(RNG.integers(0, 31 - min_width))
    extras = _distribute_slack([1] + [1] * (total - 1) + [1], slack)
    left_margin = extras[0]
    gaps = extras[1:-1]
    right_margin = extras[-1]
    grid_w = left_margin + sum(widths) + sum(gaps) + right_margin
    assert grid_w <= 30

    max_h = max(heights)
    grid_h_low = max(15, max_h + 6)
    grid_h_high = min(21, 31)
    if grid_h_low > grid_h_high:
        return None
    grid_h = int(RNG.integers(grid_h_low, grid_h_high))

    starts: list[int] = []
    c0 = left_margin
    for idx, width in enumerate(widths):
        starts.append(c0)
        if idx < total - 1:
            c0 += width + gaps[idx]

    colors = [int(c) for c in RNG.choice(PALETTE, size=total, replace=False)]
    grid = np.zeros((grid_h, grid_w), dtype=int)
    placed_components: list[dict] = []
    expected_markers: list[dict] = []

    for idx in range(total):
        h = heights[idx]
        w = widths[idx]
        row = int(RNG.integers(2, grid_h - h - 1))
        col = starts[idx]
        sprite = _build_sprite(h, w, colors[idx], require_non_ud=bool(selected_slots[idx]))
        grid[row : row + h, col : col + w] = sprite
        placed_components.append(
            {
                "bbox": (row, col, row + h - 1, col + w - 1),
                "crop": sprite.copy(),
                "color": colors[idx],
            }
        )
        if selected_slots[idx]:
            marker_col = int(RNG.integers(col, col + w))
            marker_row = 0 if edges[idx] == "top" else grid_h - 1
            grid[marker_row, marker_col] = MARKER
            expected_markers.append({"edge": edges[idx], "col": marker_col, "component_idx": idx})

    if np.any(grid[1:-1] == MARKER):
        return None

    expected_markers.sort(key=lambda item: item["col"])
    expected_crops = [
        placed_components[m["component_idx"]]["crop"]
        if m["edge"] == "top"
        else np.flipud(placed_components[m["component_idx"]]["crop"])
        for m in expected_markers
    ]
    expected = _pack_top_aligned(expected_crops)

    try:
        _validate_generated(grid, expected)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
