from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("TL", "TR", "BL", "BR")
CORNER_DIR = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for sr, sc in np.argwhere(grid == color):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if _in_bounds((h, w), rr, cc) and not seen[rr, cc] and int(grid[rr, cc]) == color:
                    seen[rr, cc] = True
                    q.append((rr, cc))

        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        components.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": len(coords),
            }
        )

    return components


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _corner_cells(bbox: tuple[int, int, int, int], corner: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if corner == "TL":
        return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0)]
    if corner == "TR":
        return [(r0, c1), (r0, c1 - 1), (r0 + 1, c1)]
    if corner == "BL":
        return [(r1, c0), (r1, c0 + 1), (r1 - 1, c0)]
    if corner == "BR":
        return [(r1, c1), (r1, c1 - 1), (r1 - 1, c1)]
    raise ValueError(corner)


def _marker_corner(bbox: tuple[int, int, int, int], marker_cells: set[tuple[int, int]]) -> str | None:
    for corner in CORNERS:
        allowed = set(_corner_cells(bbox, corner))
        if marker_cells and marker_cells.issubset(allowed) and _corner_cells(bbox, corner)[0] in marker_cells:
            return corner
    return None


def _parse_frame_component(grid: np.ndarray, bg: int, frame_color: int, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    coord_set = set(comp["coords"])
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coord_set):
        return None

    border = _border_cells((r0, c0, r1, c1))
    if not coord_set.issubset(border):
        return None

    marker_cells = border - coord_set
    if not 1 <= len(marker_cells) <= 3:
        return None

    marker_values = {int(grid[r, c]) for r, c in marker_cells}
    if len(marker_values) != 1:
        return None
    marker_color = next(iter(marker_values))
    if marker_color in (bg, frame_color):
        return None

    corner = _marker_corner((r0, c0, r1, c1), marker_cells)
    if corner is None:
        return None

    stamp_cells: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            v = int(grid[r, c])
            if v == bg:
                continue
            if v in (frame_color, marker_color):
                return None
            stamp_cells.append((r, c, v))
    if not stamp_cells:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border": border,
        "color": frame_color,
        "marker_color": marker_color,
        "marker_cells": marker_cells,
        "corner": corner,
        "spacing": len(marker_cells),
        "stamp_cells": stamp_cells,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    best: list[dict] = []
    best_score = (-1, -1)

    for color in [int(c) for c in np.unique(grid) if int(c) != bg]:
        frames: list[dict] = []
        for comp in _components_of_color(grid, color):
            parsed = _parse_frame_component(grid, bg, color, comp)
            if parsed is not None:
                frames.append(parsed)
        if not frames:
            continue

        perimeter = sum(len(f["border"]) for f in frames)
        score = (len(frames), perimeter)
        if score > best_score:
            best = frames
            best_score = score

    assert best, "expected at least one marked rectangular frame"
    best.sort(key=lambda f: f["bbox"])
    return best


def _stamp_patch(frame: dict) -> tuple[int, int, list[tuple[int, int, int]]]:
    cells = frame["stamp_cells"]
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    patch_cells = [(r - r0, c - c0, color) for r, c, color in cells]
    return r1 - r0 + 1, c1 - c0 + 1, patch_cells


def _copy_origins(
    frame: dict,
    frames: list[dict],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    r0, c0, r1, c1 = frame["bbox"]
    dr, dc = CORNER_DIR[frame["corner"]]
    spacing = int(frame["spacing"])
    patch_h, patch_w, _ = _stamp_patch(frame)

    if dr < 0:
        top = r0 - spacing - patch_h + 1
    else:
        top = r1 + spacing
    if dc < 0:
        left = c0 - spacing - patch_w + 1
    else:
        left = c1 + spacing

    other_borders: set[tuple[int, int]] = set()
    for other in frames:
        if other is not frame:
            other_borders.update(other["border"])

    origins: list[tuple[int, int]] = []
    while True:
        bottom = top + patch_h - 1
        right = left + patch_w - 1
        if top < 0 or left < 0 or bottom >= h or right >= w:
            return origins, "edge"

        blocked = False
        for rr in range(top, bottom + 1):
            for cc in range(left, right + 1):
                if (rr, cc) in other_borders:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            return origins, "frame"

        origins.append((top, left))
        top += dr * (patch_h + spacing)
        left += dc * (patch_w + spacing)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each framed interior motif outward in the direction and spacing encoded by its colored corner tag.

    Input structure:
    - The dominant background contains several same-color one-cell-thick rectangular frames.
    - Each frame has a small non-background motif inside it, separated from the frame border.
    - Exactly one corner of each frame is replaced by a unique marker color; the marker occupies one to three border cells at that corner.

    Rule:
    - Locate every marked rectangular frame and read the marker corner as the diagonal stamping direction.
    - Use the number of marker cells as the spacing between the frame/copies and the next motif copy.
    - Copy the tight motif pattern repeatedly outside the frame along that diagonal, stopping before a copy would leave the grid or overlap another frame border.
    - Preserve all original frames, markers, and interior motifs.

    Key insight:
    - The corner tag is a two-part instruction: its position gives the direction, while its visible size gives the repeat spacing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    frames = _find_frames(g, bg)

    out = g.copy()
    for frame in frames:
        patch_h, patch_w, patch_cells = _stamp_patch(frame)
        assert patch_h > 0 and patch_w > 0
        origins, _ = _copy_origins(frame, frames, g.shape)
        for top, left in origins:
            for rr, cc, color in patch_cells:
                out[top + rr, left + cc] = color

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 250
MIN_SIZE = 18
MAX_SIZE = 30
MIN_FRAMES = 2
MAX_FRAMES = 4
FRAME_H_RANGE = (6, 10)
FRAME_W_RANGE = (6, 10)
STAMP_MAX_DIM = 3
FRAME_PADDING = 1


def _expanded_bbox(bbox: tuple[int, int, int, int], h: int, w: int, pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _random_stamp_mask(max_h: int, max_w: int) -> np.ndarray:
    ph = random.randint(1, min(STAMP_MAX_DIM, max_h))
    pw = random.randint(1, min(STAMP_MAX_DIM, max_w))
    target = random.randint(1, min(ph * pw, 5))

    mask = np.zeros((ph, pw), dtype=bool)
    sr = random.randrange(ph)
    sc = random.randrange(pw)
    mask[sr, sc] = True
    frontier = [(sr, sc)]
    while int(mask.sum()) < target:
        r, c = random.choice(frontier)
        nbrs = []
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ph and 0 <= cc < pw and not mask[rr, cc]:
                nbrs.append((rr, cc))
        if not nbrs:
            all_seen = [(int(rr), int(cc)) for rr, cc in np.argwhere(mask)]
            frontier = all_seen
            if not any(
                0 <= r + dr < ph and 0 <= c + dc < pw and not mask[r + dr, c + dc]
                for r, c in frontier
                for dr, dc in DIR4
            ):
                break
            continue
        nr, nc = random.choice(nbrs)
        mask[nr, nc] = True
        frontier.append((nr, nc))
    return mask


def _marker_cells_for_size(bbox: tuple[int, int, int, int], corner: str, size: int) -> list[tuple[int, int]]:
    cells = _corner_cells(bbox, corner)
    if size == 1:
        return [cells[0]]
    if size == 2:
        return [cells[0], random.choice(cells[1:])]
    return cells


def _first_copy_fits(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    corner: str,
    spacing: int,
    patch_shape: tuple[int, int],
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    ph, pw = patch_shape
    dr, dc = CORNER_DIR[corner]
    top = r0 - spacing - ph + 1 if dr < 0 else r1 + spacing
    left = c0 - spacing - pw + 1 if dc < 0 else c1 + spacing
    return 0 <= top and top + ph <= h and 0 <= left and left + pw <= w


def _paint_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_color: int,
    marker_color: int,
    marker_cells: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    for r, c in marker_cells:
        grid[r, c] = marker_color


def _paint_stamp(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
) -> None:
    for rr, cc in np.argwhere(mask):
        grid[top + int(rr), left + int(cc)] = color


def _validate_generated(inp: np.ndarray, expected_frame_count: int) -> bool:
    try:
        bg = _background_color(inp)
        frames = _find_frames(inp, bg)
        if len(frames) != expected_frame_count:
            return False

        planned: set[tuple[int, int]] = set()
        stop_reasons: set[str] = set()
        for frame in frames:
            _, _, patch_cells = _stamp_patch(frame)
            origins, reason = _copy_origins(frame, frames, inp.shape)
            if not origins:
                return False
            stop_reasons.add(reason)
            for top, left in origins:
                for rr, cc, _ in patch_cells:
                    cell = (top + rr, left + cc)
                    if inp[cell] != bg or cell in planned:
                        return False
                    planned.add(cell)

        out = main(inp)
        if not np.array_equal(out[inp != bg], inp[inp != bg]):
            return False
        if not planned or not all(out[cell] != bg for cell in planned):
            return False
        return bool(stop_reasons)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(MIN_SIZE, MAX_SIZE)
    cols = random.randint(MIN_SIZE, MAX_SIZE)
    n_frames = random.randint(MIN_FRAMES, MAX_FRAMES)

    palette = random.sample(range(10), 2 + 2 * n_frames)
    bg = palette[0]
    frame_color = palette[1]
    stamp_colors = palette[2 : 2 + n_frames]
    marker_colors = palette[2 + n_frames :]

    grid = np.full((rows, cols), bg, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    directions = list(CORNERS)
    random.shuffle(directions)
    while len(directions) < n_frames:
        directions.append(random.choice(CORNERS))
    random.shuffle(directions)

    for idx in range(n_frames):
        placed_this = False
        for _ in range(PLACE_ATTEMPTS):
            fh = random.randint(*FRAME_H_RANGE)
            fw = random.randint(*FRAME_W_RANGE)
            if fh >= rows - 1 or fw >= cols - 1:
                continue

            r0 = random.randint(0, rows - fh)
            c0 = random.randint(0, cols - fw)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            bbox = (r0, c0, r1, c1)
            padded = _expanded_bbox(bbox, rows, cols, FRAME_PADDING)
            if any(_bbox_overlaps(padded, other) for other in placed):
                continue

            corner = directions[idx]
            spacing = random.randint(1, 3)
            stamp_mask = _random_stamp_mask(fh - 4, fw - 4)
            ph, pw = stamp_mask.shape
            if not _first_copy_fits((rows, cols), bbox, corner, spacing, (ph, pw)):
                continue

            stamp_top = random.randint(r0 + 2, r1 - ph - 1)
            stamp_left = random.randint(c0 + 2, c1 - pw - 1)
            marker_cells = _marker_cells_for_size(bbox, corner, spacing)

            _paint_frame(grid, bbox, frame_color, marker_colors[idx], marker_cells)
            _paint_stamp(grid, stamp_top, stamp_left, stamp_mask, stamp_colors[idx])
            placed.append(padded)
            placed_this = True
            break

        if not placed_this:
            return None

    if not _validate_generated(grid, n_frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-frame input")
