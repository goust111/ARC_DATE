from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
FILL_COLORS = (1, 2, 3, 4, 7)
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

PANEL_H_RANGE = (8, 11)
WIDTH_RANGE = (10, 15)
RECT_SIDE_RANGE = (3, 5)
OBJECT_COUNT_CHOICES = (2, 2, 3)
MAX_MARKERS = 5
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(comp, dtype=int))
    return out


def _dominant_fill(patch: np.ndarray) -> int:
    vals, counts = np.unique(patch[patch != BG], return_counts=True)
    assert len(vals) > 0, "expected foreground pixels inside the crop"
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "expected a unique dominant fill color"
    return int(vals[int(np.argmax(counts))])


def _score_patch(patch: np.ndarray) -> tuple[int, int]:
    fill = _dominant_fill(patch)
    markers = int(np.count_nonzero(patch != fill))
    return fill, markers


def _panel_objects(panel: np.ndarray) -> list[dict]:
    objects = []
    for coords in components_mask(panel != BG):
        bbox = bbox_coords(coords)
        patch = crop_bbox(panel, bbox)
        fill, markers = _score_patch(patch)
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "fill": fill,
                "markers": markers,
                "bbox_area": int(patch.shape[0] * patch.shape[1]),
            }
        )
    assert objects, "expected at least one foreground object per panel"
    return objects


def _choose_sparse_patch(panel: np.ndarray) -> np.ndarray:
    objects = _panel_objects(panel)
    min_markers = min(obj["markers"] for obj in objects)
    winners = [obj["patch"] for obj in objects if obj["markers"] == min_markers]
    assert len(winners) == 1, "expected a unique sparsest rectangle"
    return winners[0]


def _overlay(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    assert top.shape == bottom.shape, "selected crops must have the same shape"
    out = bottom.copy()
    top_fg = top != BG
    out[top_fg] = top[top_fg]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the marker-sparsest rectangle from each panel and transparently overlay the two crops.

    Input structure:
    - The grid has black background and one full-width grey separator row.
    - The separator splits the input into two equal-height panels.
    - Each panel contains several separated rectangular patches that are mostly one fill color, with a few differently colored marker cells or black cutout cells inside the rectangle's bbox.

    Rule:
    - Find the grey separator row and split the grid into the top and bottom panels.
    - In each panel, detect every 4-connected non-black object and crop its bounding box.
    - For each crop, infer the dominant non-black fill color and count every cell that differs from that fill color as a marker/cutout.
    - Select the unique crop with the smallest such count in each panel.
    - Overlay the top crop over the bottom crop, treating black as transparent.

    Key insight:
    - The divider row means "solve the two panels independently, then merge them"; the deciding feature is marker sparsity inside each rectangle, not size, position, or color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r] == SEP)]
    assert len(divider_rows) == 1, "expected exactly one grey separator row"
    divider = divider_rows[0]

    top = g[:divider, :]
    bottom = g[divider + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "separator must split two equal nonempty panels"

    top_patch = _choose_sparse_patch(top)
    bottom_patch = _choose_sparse_patch(bottom)
    return _overlay(top_patch, bottom_patch)


def _is_connected_nonzero(patch: np.ndarray) -> bool:
    return len(components_mask(patch != BG)) == 1


def _interior_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]


def _all_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _make_patch(
    h: int,
    w: int,
    fill: int,
    marker_count: int,
    *,
    require_hole: bool = False,
) -> np.ndarray:
    assert marker_count >= 0
    cells = _all_cells(h, w)
    interior = _interior_cells(h, w)
    palette = [color for color in MARKER_COLORS if color != fill]
    for _ in range(200):
        patch = np.full((h, w), fill, dtype=int)
        if marker_count == 0:
            return patch

        min_holes = 1 if require_hole else 0
        max_holes = min(marker_count, len(interior))
        if min_holes > max_holes:
            continue
        hole_count = random.randint(min_holes, max_holes)
        hole_positions = set(random.sample(interior, hole_count))
        colorable = [pos for pos in cells if pos not in hole_positions]
        colored_count = marker_count - hole_count
        if colored_count > len(colorable):
            continue
        colored_positions = random.sample(colorable, colored_count)

        for r, c in hole_positions:
            patch[r, c] = BG

        if colored_positions:
            distinct_count = random.randint(1, min(3, len(palette), len(colored_positions)))
            chosen_palette = random.sample(palette, distinct_count)
        for idx, (r, c) in enumerate(colored_positions):
            patch[r, c] = chosen_palette[idx % len(chosen_palette)] if random.random() < 0.5 else random.choice(chosen_palette)

        if not _is_connected_nonzero(patch):
            continue
        vals, counts = np.unique(patch[patch != BG], return_counts=True)
        max_count = int(counts.max())
        if int(np.count_nonzero(counts == max_count)) != 1:
            continue
        if int(vals[int(np.argmax(counts))]) != fill:
            continue
        if _score_patch(patch)[1] != marker_count:
            continue
        return patch
    raise RuntimeError("failed to build a valid patch")


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_patches(panel_h: int, width: int, patches: list[np.ndarray]) -> list[tuple[int, int]]:
    for _ in range(200):
        positions: list[tuple[int, int] | None] = [None] * len(patches)
        boxes: list[tuple[int, int, int, int]] = []
        order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
        ok = True
        for idx in order:
            ph, pw = patches[idx].shape
            candidates = [(r, c) for r in range(panel_h - ph + 1) for c in range(width - pw + 1)]
            random.shuffle(candidates)
            placed = False
            for r, c in candidates:
                box = (r, c, r + ph - 1, c + pw - 1)
                if any(_boxes_too_close(box, other) for other in boxes):
                    continue
                positions[idx] = (r, c)
                boxes.append(box)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return [(int(r), int(c)) for r, c in positions]
    raise RuntimeError("failed to place patches")


def _build_panel(objects: list[dict], panel_h: int, width: int) -> np.ndarray:
    panel = np.zeros((panel_h, width), dtype=int)
    positions = _place_patches(panel_h, width, [obj["patch"] for obj in objects])
    for obj, (r0, c0) in zip(objects, positions):
        patch = obj["patch"]
        ph, pw = patch.shape
        panel[r0 : r0 + ph, c0 : c0 + pw] = patch
        obj["bbox"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
    return panel


def _sample_panel_objects(
    panel_h: int,
    width: int,
    winner_patch: np.ndarray,
    winner_fill: int,
    winner_markers: int,
) -> tuple[list[dict], int]:
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    extra_fills = random.sample([color for color in FILL_COLORS if color != winner_fill], object_count - 1)
    objects = [{"patch": winner_patch, "markers": winner_markers, "fill": winner_fill}]

    for fill in extra_fills:
        dh = random.randint(*RECT_SIDE_RANGE)
        dw = random.randint(*RECT_SIDE_RANGE)
        max_count = min(MAX_MARKERS, dh * dw - 1)
        marker_count = random.randint(winner_markers + 1, max_count)
        patch = _make_patch(dh, dw, fill, marker_count, require_hole=random.random() < 0.35)
        objects.append({"patch": patch, "markers": marker_count, "fill": fill})

    random.shuffle(objects)
    winner_idx = next(idx for idx, obj in enumerate(objects) if obj["patch"] is winner_patch)
    return objects, winner_idx


def _render_grid(
    panel_h: int,
    width: int,
    top_objects: list[dict],
    bottom_objects: list[dict],
) -> np.ndarray:
    top_panel = _build_panel(top_objects, panel_h, width)
    bottom_panel = _build_panel(bottom_objects, panel_h, width)
    grid = np.zeros((2 * panel_h + 1, width), dtype=int)
    grid[:panel_h] = top_panel
    grid[panel_h] = SEP
    grid[panel_h + 1 :] = bottom_panel
    return grid


def _pick_by_min_area(objects: list[dict]) -> np.ndarray:
    return min(objects, key=lambda obj: (obj["patch"].shape[0] * obj["patch"].shape[1], obj["bbox"][0], obj["bbox"][1]))["patch"]


def _pick_by_topleft(objects: list[dict]) -> np.ndarray:
    return min(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["patch"].shape[0] * obj["patch"].shape[1]))["patch"]


def _alt_overlay_or_none(top_patch: np.ndarray, bottom_patch: np.ndarray) -> np.ndarray | None:
    if top_patch.shape != bottom_patch.shape:
        return None
    return _overlay(top_patch, bottom_patch)


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    width = random.randint(*WIDTH_RANGE)

    win_h = random.randint(*RECT_SIDE_RANGE)
    win_w = random.randint(*RECT_SIDE_RANGE)
    top_fill, bottom_fill = random.sample(FILL_COLORS, 2)
    top_markers = random.randint(1, 3)
    bottom_markers = random.randint(0, 3)

    try:
        top_winner = _make_patch(win_h, win_w, top_fill, top_markers, require_hole=True)
    except RuntimeError:
        return None

    for _ in range(80):
        try:
            bottom_winner = _make_patch(win_h, win_w, bottom_fill, bottom_markers, require_hole=random.random() < 0.35)
        except RuntimeError:
            continue
        visible_bottom = (top_winner == BG) & (bottom_winner != BG)
        if np.any(visible_bottom) and np.any(bottom_winner[visible_bottom] == bottom_fill):
            break
    else:
        return None

    try:
        top_objects, _top_winner_idx = _sample_panel_objects(panel_h, width, top_winner, top_fill, top_markers)
        bottom_objects, _bottom_winner_idx = _sample_panel_objects(panel_h, width, bottom_winner, bottom_fill, bottom_markers)
        grid = _render_grid(panel_h, width, top_objects, bottom_objects)
    except RuntimeError:
        return None

    expected = _overlay(top_winner, bottom_winner)
    if not np.any((top_winner == BG) & (bottom_winner != BG)):
        return None
    if np.array_equal(expected, top_winner) or np.array_equal(expected, bottom_winner):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    alt_area = _alt_overlay_or_none(_pick_by_min_area(top_objects), _pick_by_min_area(bottom_objects))
    if alt_area is not None and np.array_equal(alt_area, expected):
        return None

    alt_pos = _alt_overlay_or_none(_pick_by_topleft(top_objects), _pick_by_topleft(bottom_objects))
    if alt_pos is not None and np.array_equal(alt_pos, expected):
        return None

    detected_top = _panel_objects(grid[:panel_h])
    detected_bottom = _panel_objects(grid[panel_h + 1 :])
    if len(detected_top) != len(top_objects) or len(detected_bottom) != len(bottom_objects):
        return None
    if sum(obj["markers"] == top_markers for obj in detected_top) != 1:
        return None
    if sum(obj["markers"] == bottom_markers for obj in detected_bottom) != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
