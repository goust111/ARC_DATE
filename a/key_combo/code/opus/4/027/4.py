from __future__ import annotations

import random

import numpy as np


BG = 0
MOTIF = 2
FRAME = 5
MARKER_COLORS = (7, 8)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("patch placement went out of bounds")
    mask = patch != BG
    dst[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"coords": coords, "bbox": _bbox_coords(coords), "area": int(coords.shape[0])})
    return comps


def _frame_components(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _components_of_color(grid, FRAME):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        expected_area = 2 * (h + w) - 4
        if comp["area"] != expected_area:
            continue
        patch = _crop_bbox(grid, comp["bbox"])
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == FRAME):
            continue
        if np.any(patch[1:-1, 1:-1] == FRAME):
            continue
        frames.append(comp)
    frames.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    assert len(frames) == 2, "expected exactly two rectangular frames"
    return frames


def _marker_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = []
    for color in MARKER_COLORS:
        pts = np.argwhere(grid == color)
        assert pts.shape[0] == 1, "expected one marker per marker color"
        coords.append((int(pts[0, 0]), int(pts[0, 1])))
    coords.sort(key=lambda rc: (rc[1], rc[0]))
    return coords


def _mirror_complete(mask: np.ndarray, axis: str) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        if axis == "vertical":
            out[r, w - 1 - c] = True
        else:
            out[h - 1 - r, c] = True
    return out


def _mask_to_patch(mask: np.ndarray, color: int = MOTIF) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "motif must be non-empty"
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return out


def _original_and_completed_patches(
    grid: np.ndarray, frame_box: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, str]:
    r0, c0, r1, c1 = frame_box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    h, w = inner.shape
    assert h != w, "frame interior must have a unique aspect ratio"
    mask = inner == MOTIF
    assert bool(mask.any()), "frame must contain a motif"
    axis = "vertical" if w > h else "horizontal"
    completed = _mirror_complete(mask, axis)
    return _mask_to_patch(mask), _mask_to_patch(completed), axis


def _stamped_output(
    height: int,
    width: int,
    bottom_row: int,
    motifs: list[np.ndarray],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((height - bottom_row, width), dtype=int)
    for patch, (mr, mc) in zip(motifs, markers, strict=True):
        pr, pc = patch.shape[0] // 2, patch.shape[1] // 2
        _paste(out, patch, mr - bottom_row - pr, mc - pc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each framed half-motif using the frame aspect ratio, then stamp the completed motifs onto the lower markers.

    Input structure:
    - A black background contains two grey hollow rectangular frames in the upper area and two isolated marker pixels in the lower area.
    - Each frame contains a red 4-connected motif drawn only on one side of the frame interior.
    - One frame is wider than tall and the other is taller than wide, so their interior aspect ratios determine different mirror axes.

    Rule:
    - For each frame, look at its interior size after removing the grey border.
    - If the interior is wider than tall, mirror the red motif left-to-right inside that interior; if it is taller than wide, mirror it top-to-bottom.
    - Crop the completed motif to its tight bounding box, pair frames and markers by left-to-right order, and stamp each cropped motif so its bbox center lands on the paired marker.
    - Return only the rows strictly below the lowest frame, keeping stamped red pixels on black and discarding the original markers.

    Key insight:
    - The frame geometry decides how to complete each motif, while the spatial left-to-right ordering separately decides where each completed motif must be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _frame_components(g)
    markers = _marker_coords(g)
    bottom_row = max(frame["bbox"][2] for frame in frames) + 1

    motifs = []
    for frame in frames:
        _original, completed, _axis = _original_and_completed_patches(g, frame["bbox"])
        motifs.append(completed)

    return _stamped_output(g.shape[0], g.shape[1], bottom_row, motifs, markers)


WIDE_INNERS = [(3, 7), (3, 9), (5, 9)]
TALL_INNERS = [(7, 3), (9, 3), (9, 5)]
MAX_ATTEMPTS = 500


def _grow_connected_cells(cells: list[tuple[int, int]], count: int) -> set[tuple[int, int]] | None:
    chosen = {random.choice(cells)}
    cell_set = set(cells)
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cell_set and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        chosen.add(random.choice(frontier))
    return chosen


def _make_half_mask(kind: str, inner_h: int, inner_w: int) -> np.ndarray | None:
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    count = random.randint(3, 6)
    if kind == "wide":
        axis = inner_w // 2
        allowed = [(r, c) for r in range(inner_h) for c in range(axis + 1)]
    else:
        axis = inner_h // 2
        allowed = [(r, c) for r in range(axis + 1) for c in range(inner_w)]
    chosen = _grow_connected_cells(allowed, count)
    if chosen is None:
        return None
    for r, c in chosen:
        mask[r, c] = True

    coords = np.argwhere(mask)
    if kind == "wide":
        if not np.any(coords[:, 1] < axis):
            return None
        row_span = int(coords[:, 0].max() - coords[:, 0].min() + 1)
        if row_span % 2 == 0:
            return None
    else:
        if not np.any(coords[:, 0] < axis):
            return None
        col_span = int(coords[:, 1].max() - coords[:, 1].min() + 1)
        if col_span % 2 == 0:
            return None
    return mask


def _semantic_ok(mask: np.ndarray, kind: str) -> bool:
    correct_axis = "vertical" if kind == "wide" else "horizontal"
    wrong_axis = "horizontal" if kind == "wide" else "vertical"

    original = _mask_to_patch(mask)
    completed = _mask_to_patch(_mirror_complete(mask, correct_axis))
    wrong = _mask_to_patch(_mirror_complete(mask, wrong_axis))

    if completed.shape[0] % 2 == 0 or completed.shape[1] % 2 == 0:
        return False
    if np.array_equal(original, completed):
        return False
    if np.array_equal(completed, wrong):
        return False
    return True


def _generate_motif(kind: str, inner_h: int, inner_w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        mask = _make_half_mask(kind, inner_h, inner_w)
        if mask is None or not _semantic_ok(mask, kind):
            continue
        original = np.zeros((inner_h, inner_w), dtype=int)
        original[mask] = MOTIF
        completed = _mask_to_patch(_mirror_complete(mask, "vertical" if kind == "wide" else "horizontal"))
        return original, completed
    raise RuntimeError(f"failed to generate a valid {kind} motif")


def _paint_frame(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int) -> None:
    grid[top, left : left + outer_w] = FRAME
    grid[top + outer_h - 1, left : left + outer_w] = FRAME
    grid[top : top + outer_h, left] = FRAME
    grid[top : top + outer_h, left + outer_w - 1] = FRAME


def _try_marker_positions(
    out_h: int,
    width: int,
    patches: list[np.ndarray],
    frame_centers: list[int],
) -> list[tuple[int, int]] | None:
    def can_place(occupied: np.ndarray, patch: np.ndarray, center_r: int, center_c: int) -> bool:
        hh, hw = patch.shape[0] // 2, patch.shape[1] // 2
        top = center_r - hh
        left = center_c - hw
        if top < 0 or left < 0 or top + patch.shape[0] > out_h or left + patch.shape[1] > width:
            return False
        return not bool((occupied[top : top + patch.shape[0], left : left + patch.shape[1]] & (patch != BG)).any())

    def place(occupied: np.ndarray, patch: np.ndarray, center_r: int, center_c: int) -> None:
        hh, hw = patch.shape[0] // 2, patch.shape[1] // 2
        top = center_r - hh
        left = center_c - hw
        occupied[top : top + patch.shape[0], left : left + patch.shape[1]] |= patch != BG

    half_sizes = [(p.shape[0] // 2, p.shape[1] // 2) for p in patches]
    left_ranges = [
        range(hw + 1, width - hw - 1)
        for _hh, hw in half_sizes
    ]
    left_choices = [c for c in left_ranges[0] if c < width - half_sizes[1][1] - 1]
    right_choices = [c for c in left_ranges[1] if c > half_sizes[0][1] + 1]
    random.shuffle(left_choices)
    random.shuffle(right_choices)
    for c0 in left_choices[:80]:
        for c1 in right_choices[:80]:
            if c0 >= c1:
                continue
            if c1 - c0 <= half_sizes[0][1] + half_sizes[1][1] + 3:
                continue
            if abs(c0 - frame_centers[0]) + abs(c1 - frame_centers[1]) < 2:
                continue
            rows = []
            ok = True
            for hh, _hw in half_sizes:
                lo = hh + 1
                hi = out_h - hh - 2
                if lo > hi:
                    ok = False
                    break
                rows.append(random.randint(lo, hi))
            if not ok:
                continue
            occupied = np.zeros((out_h, width), dtype=bool)
            if not can_place(occupied, patches[0], rows[0], c0):
                continue
            place(occupied, patches[0], rows[0], c0)
            if not can_place(occupied, patches[1], rows[1], c1):
                continue
            swapped = np.zeros((out_h, width), dtype=bool)
            if not can_place(swapped, patches[1], rows[0], c0):
                continue
            place(swapped, patches[1], rows[0], c0)
            if not can_place(swapped, patches[0], rows[1], c1):
                continue
            return [(rows[0], c0), (rows[1], c1)]
    return None


def _build_input() -> np.ndarray | None:
    order = random.sample(["wide", "tall"], 2)
    inners = {
        "wide": random.choice(WIDE_INNERS),
        "tall": random.choice(TALL_INNERS),
    }
    originals: dict[str, np.ndarray] = {}
    completed: dict[str, np.ndarray] = {}
    for kind in ("wide", "tall"):
        inner_h, inner_w = inners[kind]
        originals[kind], completed[kind] = _generate_motif(kind, inner_h, inner_w)

    if np.array_equal(completed["wide"], completed["tall"]):
        return None

    outer_sizes = {kind: (inners[kind][0] + 2, inners[kind][1] + 2) for kind in ("wide", "tall")}
    left_kind, right_kind = order
    left_h, left_w = outer_sizes[left_kind]
    right_h, right_w = outer_sizes[right_kind]

    top = random.randint(1, 3)
    gap = random.randint(2, 5)
    left_margin = random.randint(1, 3)
    frame_bottom = top + max(left_h, right_h) - 1
    out_h = random.randint(9, 14)
    height = frame_bottom + 1 + out_h
    width = max(20, left_margin + left_w + gap + right_w + 2)
    width = random.randint(width, min(30, width + 6))

    left_col = left_margin
    right_col_min = left_col + left_w + gap
    right_col_max = width - right_w - 1
    if right_col_min > right_col_max or height > 30:
        return None
    right_col = random.randint(right_col_min, right_col_max)

    patches = [completed[left_kind], completed[right_kind]]
    frame_centers = [left_col + left_w // 2, right_col + right_w // 2]
    marker_local = _try_marker_positions(out_h, width, patches, frame_centers)
    if marker_local is None:
        return None

    marker_colors = list(MARKER_COLORS)
    random.shuffle(marker_colors)

    grid = np.zeros((height, width), dtype=int)
    placements = [(left_kind, top, left_col), (right_kind, top, right_col)]
    for kind, r0, c0 in placements:
        outer_h, outer_w = outer_sizes[kind]
        _paint_frame(grid, r0, c0, outer_h, outer_w)
        inner = originals[kind]
        _paste(grid, inner, r0 + 1, c0 + 1)

    bottom_row = frame_bottom + 1
    marker_global = []
    for color, (r, c) in zip(marker_colors, marker_local, strict=True):
        grid[bottom_row + r, c] = color
        marker_global.append((bottom_row + r, c))

    correct = main(grid)
    marker_sorted = sorted(marker_global, key=lambda rc: (rc[1], rc[0]))
    expected = _stamped_output(height, width, bottom_row, patches, marker_sorted)
    if not np.array_equal(correct, expected):
        return None

    original_patches = [_mask_to_patch(originals[left_kind] == MOTIF), _mask_to_patch(originals[right_kind] == MOTIF)]
    if np.array_equal(correct, _stamped_output(height, width, bottom_row, original_patches, marker_sorted)):
        return None
    if np.array_equal(correct, _stamped_output(height, width, bottom_row, list(reversed(patches)), marker_sorted)):
        return None

    wrong_left = _mask_to_patch(
        _mirror_complete(originals[left_kind] == MOTIF, "horizontal" if left_kind == "wide" else "vertical")
    )
    wrong_right = _mask_to_patch(
        _mirror_complete(originals[right_kind] == MOTIF, "horizontal" if right_kind == "wide" else "vertical")
    )
    try:
        wrong_outputs = [
            _stamped_output(height, width, bottom_row, [wrong_left, patches[1]], marker_sorted),
            _stamped_output(height, width, bottom_row, [patches[0], wrong_right], marker_sorted),
        ]
    except ValueError:
        return None
    if any(np.array_equal(correct, wrong_out) for wrong_out in wrong_outputs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
