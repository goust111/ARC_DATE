import hashlib
from typing import Iterable

import numpy as np


BASE_COLOR = 1
FRAME_COLORS = (5, 7, 8, 9)
MARKER_POOL = (2, 3, 4, 6, 8, 9)
GEN_ATTEMPTS = 500
_SEEN_OUTPUT_SIGS: set[str] = set()


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 - 1) * (c1 - c0 - 1)


def _iter_border_cells(r0: int, c0: int, r1: int, c1: int) -> Iterable[tuple[int, int]]:
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in _iter_border_cells(*bbox):
        mask[r, c] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            color = int(g[r0, c0])
            if color == 0:
                continue
            for r1 in range(r0 + 2, h):
                if int(g[r1, c0]) != color:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(g[r0, c1]) != color or int(g[r1, c1]) != color:
                        continue
                    if not np.all(g[r0, c0 : c1 + 1] == color):
                        continue
                    if not np.all(g[r1, c0 : c1 + 1] == color):
                        continue
                    if not np.all(g[r0 : r1 + 1, c0] == color):
                        continue
                    if not np.all(g[r0 : r1 + 1, c1] == color):
                        continue
                    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == color):
                        continue
                    frames.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "color": color,
                            "area": (r1 - r0 - 1) * (c1 - c0 - 1),
                        }
                    )
    return frames


def _largest_frame(grid: np.ndarray) -> dict:
    frames = _find_frames(grid)
    assert frames, "expected at least one hollow rectangular frame"
    max_area = max(frame["area"] for frame in frames)
    winners = [frame for frame in frames if frame["area"] == max_area]
    assert len(winners) == 1, "expected a unique largest frame"
    return winners[0]


def _expand_markers(crop: np.ndarray) -> np.ndarray:
    out = np.full(crop.shape, BASE_COLOR, dtype=int)
    marker_coords = np.argwhere(crop != BASE_COLOR)
    assert 2 <= len(marker_coords) <= 3, "expected 2-3 interior markers"
    for r, c in marker_coords:
        color = int(crop[int(r), int(c)])
        row_mask = out[int(r), :] == BASE_COLOR
        out[int(r), row_mask] = color
        col_mask = out[:, int(c)] == BASE_COLOR
        out[col_mask, int(c)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the interior of the unique largest hollow frame, then expand its markers into row-and-column crosses.

    Input structure:
    - The grid is mostly black background with 2-3 non-overlapping one-cell-thick hollow rectangular frames.
    - The unique largest frame encloses a blue interior containing 2-3 isolated non-blue marker pixels on distinct rows and columns.
    - Smaller frames and the outside background may also contain colored distractor pixels.

    Rule:
    - Detect every closed monochrome hollow rectangle and choose the one with unique maximum interior area.
    - Crop to the cells strictly inside that frame, discarding the frame border and everything outside it.
    - Starting from blue everywhere in the cropped subgrid, expand each non-blue marker to its full row and full column within the cropped bounds only.
    - If expanded lines cross, the crossing keeps the color of the marker that appears earlier in row-major order inside the cropped subgrid.

    Key insight:
    - The largest frame determines both which markers are valid and the coordinate system of the expansion, so outside pixels and smaller-frame contents must be ignored completely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _largest_frame(g)
    r0, c0, r1, c1 = frame["bbox"]
    crop = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert np.all((crop == BASE_COLOR) | (crop != frame["color"])), "frame color must stay on the border"
    return _expand_markers(crop)


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _choose_box(
    rng: np.random.Generator,
    shape: tuple[int, int],
    box_shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    margin: int,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    bh, bw = box_shape
    if bh + 2 * margin > h or bw + 2 * margin > w:
        return None
    positions = [
        (r0, c0)
        for r0 in range(margin, h - bh - margin + 1)
        for c0 in range(margin, w - bw - margin + 1)
    ]
    rng.shuffle(positions)
    for r0, c0 in positions:
        bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        if all(_boxes_separate(bbox, other, gap=1) for other in existing):
            return bbox
    return None


def _sample_marker_positions(
    rng: np.random.Generator,
    h: int,
    w: int,
    count: int,
) -> list[tuple[int, int]]:
    for _ in range(200):
        rows = rng.choice(h, size=count, replace=False)
        cols = rng.choice(w, size=count, replace=False)
        order = rng.permutation(count)
        positions = [(int(rows[i]), int(cols[order[i]])) for i in range(count)]
        ok = True
        for i, (r0, c0) in enumerate(positions):
            for r1, c1 in positions[i + 1 :]:
                if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return positions
    raise RuntimeError("failed to place marker pixels")


def _isolated_candidates(
    grid: np.ndarray,
    forbidden: np.ndarray,
    preferred_rows: set[int] | None = None,
    preferred_cols: set[int] | None = None,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if forbidden[r, c] or grid[r, c] != 0:
                continue
            good = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if grid[rr, cc] != 0:
                        good = False
                        break
                if not good:
                    break
            if not good:
                continue
            aligned = (
                (preferred_rows is not None and r in preferred_rows)
                or (preferred_cols is not None and c in preferred_cols)
            )
            if aligned:
                cells.insert(0, (r, c))
            else:
                cells.append((r, c))
    return cells


def _output_from_meta(meta: dict) -> np.ndarray:
    h, w = meta["largest_interior_shape"]
    crop = np.full((h, w), BASE_COLOR, dtype=int)
    for r, c, color in meta["markers_local"]:
        crop[r, c] = color
    return _expand_markers(crop)


def _wrong_global_projection(grid: np.ndarray, meta: dict) -> np.ndarray:
    r0, c0, r1, c1 = meta["largest_bbox"]
    h, w = meta["largest_interior_shape"]
    out = np.full((h, w), BASE_COLOR, dtype=int)

    excluded = np.zeros(grid.shape, dtype=bool)
    for bbox in meta["frame_bboxes"]:
        excluded |= _border_mask(grid.shape, bbox)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (0, BASE_COLOR) or excluded[r, c]:
                continue
            local_r = r - (r0 + 1)
            local_c = c - (c0 + 1)
            if 0 <= local_r < h:
                row_mask = out[local_r, :] == BASE_COLOR
                out[local_r, row_mask] = color
            if 0 <= local_c < w:
                col_mask = out[:, local_c] == BASE_COLOR
                out[col_mask, local_c] = color
    return out


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    frames = _find_frames(grid)
    if len(frames) < len(meta["frame_bboxes"]):
        return False

    intended = meta["largest_bbox"]
    areas = [frame["area"] for frame in frames]
    if areas.count(max(areas)) != 1:
        return False
    if not any(frame["bbox"] == intended for frame in frames):
        return False
    if _largest_frame(grid)["bbox"] != intended:
        return False

    rows = [r for r, _c, _color in meta["markers_local"]]
    cols = [c for _r, c, _color in meta["markers_local"]]
    if len(rows) != len(set(rows)) or len(cols) != len(set(cols)):
        return False

    expected = _output_from_meta(meta)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, grid[intended[0] + 1 : intended[2], intended[1] + 1 : intended[3]]):
        return False
    if np.array_equal(_wrong_global_projection(grid, meta), expected):
        return False
    if len(np.unique(expected)) < 3:
        return False
    return True


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    h = int(rng.integers(15, 20))
    w = int(rng.integers(15, 20))
    grid = np.zeros((h, w), dtype=int)

    largest_hi = int(rng.integers(6, 10))
    largest_wi = int(rng.integers(6, 10))
    largest_outer = (largest_hi + 2, largest_wi + 2)
    n_frames = int(rng.choice([2, 3]))

    small_interiors: list[tuple[int, int]] = []
    largest_area = largest_hi * largest_wi
    for _ in range(n_frames - 1):
        for _try in range(100):
            shi = int(rng.integers(3, 7))
            swi = int(rng.integers(3, 7))
            if shi * swi < largest_area:
                small_interiors.append((shi, swi))
                break
        else:
            return None

    frame_colors = list(rng.choice(FRAME_COLORS, size=n_frames, replace=False))
    boxes: list[tuple[int, int, int, int]] = []
    largest_bbox = _choose_box(rng, grid.shape, largest_outer, boxes, margin=2)
    if largest_bbox is None:
        return None
    boxes.append(largest_bbox)
    for hi, wi in small_interiors:
        bbox = _choose_box(rng, grid.shape, (hi + 2, wi + 2), boxes, margin=1)
        if bbox is None:
            return None
        boxes.append(bbox)

    for bbox, color in zip(boxes, frame_colors):
        _draw_frame(grid, bbox, int(color))

    lr0, lc0, lr1, lc1 = largest_bbox
    grid[lr0 + 1 : lr1, lc0 + 1 : lc1] = BASE_COLOR

    marker_count = int(rng.choice([2, 3]))
    marker_palette = [color for color in MARKER_POOL if color not in frame_colors]
    marker_colors = list(rng.choice(marker_palette, size=marker_count, replace=False))
    marker_positions = _sample_marker_positions(rng, largest_hi, largest_wi, marker_count)
    markers_local: list[tuple[int, int, int]] = []
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[lr0 + 1 + r, lc0 + 1 + c] = color
        markers_local.append((r, c, int(color)))

    used_inner_distractor = False
    distractor_palette = [color for color in range(2, 10) if color not in frame_colors]
    for bbox in boxes[1:]:
        r0, c0, r1, c1 = bbox
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if rng.random() < 0.8:
            count = 1 if min(inner_h, inner_w) <= 3 else int(rng.choice([1, 2]))
            cells = _sample_marker_positions(rng, inner_h, inner_w, count)
            colors = list(rng.choice(distractor_palette, size=count, replace=False))
            for (r, c), color in zip(cells, colors):
                rr = r0 + 1 + r
                cc = c0 + 1 + c
                grid[rr, cc] = color
                used_inner_distractor = True

    forbidden = np.zeros(grid.shape, dtype=bool)
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        forbidden[r0 : r1 + 1, c0 : c1 + 1] = True

    preferred_rows = set(range(lr0 + 1, lr1))
    preferred_cols = set(range(lc0 + 1, lc1))
    candidates = _isolated_candidates(grid, forbidden, preferred_rows, preferred_cols)
    if not candidates:
        return None

    aligned = [cell for cell in candidates if cell[0] in preferred_rows or cell[1] in preferred_cols]
    if not aligned:
        return None

    rng.shuffle(aligned)
    outside_count = min(len(candidates), int(rng.choice([1, 2, 3])))
    outside_cells = [aligned[0]]
    remaining = [cell for cell in candidates if cell != aligned[0]]
    rng.shuffle(remaining)
    for cell in remaining:
        if len(outside_cells) >= outside_count:
            break
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in outside_cells):
            outside_cells.append(cell)

    outside_colors = list(
        rng.choice(
            [color for color in distractor_palette if color != BASE_COLOR],
            size=len(outside_cells),
            replace=len(outside_cells) > len(set(distractor_palette)),
        )
    )
    for (r, c), color in zip(outside_cells, outside_colors):
        grid[r, c] = int(color)

    meta = {
        "largest_bbox": largest_bbox,
        "largest_interior_shape": (largest_hi, largest_wi),
        "markers_local": markers_local,
        "frame_bboxes": boxes,
        "outside_distractors": outside_cells,
        "used_inner_distractor": used_inner_distractor,
    }
    return grid, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is None:
            continue
        grid, meta = result
        if not meta["used_inner_distractor"]:
            continue
        if not _validate_instance(grid, meta):
            continue
        out_sig = _signature(main(grid))
        if out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
