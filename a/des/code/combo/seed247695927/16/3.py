from __future__ import annotations

import random
from collections import Counter

import numpy as np


Corner = str

_RNG = random.Random()
_NEXT_CORNER: dict[Corner, Corner] = {
    "TL": "TR",
    "TR": "BR",
    "BR": "BL",
    "BL": "TL",
}
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_BG = 0
_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _bbox_corner(bbox: tuple[int, int, int, int], height: int, width: int) -> Corner | None:
    r0, c0, r1, c1 = bbox
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == width - 1:
        return "TR"
    if r1 == height - 1 and c1 == width - 1:
        return "BR"
    if r1 == height - 1 and c0 == 0:
        return "BL"
    return None


def _corner_anchor(corner: Corner, height: int, width: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, width - obj_w
    if corner == "BR":
        return height - obj_h, width - obj_w
    if corner == "BL":
        return height - obj_h, 0
    raise AssertionError(f"unknown corner {corner}")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    bg = _get_bg(g)

    out = g.copy()
    planned: dict[tuple[int, int], int] = {}
    by_corner: dict[Corner, dict] = {}

    for comp in _find_components(g, bg):
        corner = _bbox_corner(comp["bbox"], height, width)
        if corner is None:
            continue
        assert corner not in by_corner, "at most one corner object per corner"
        by_corner[corner] = comp

    assert by_corner, "expected at least one true-corner object"

    for src_corner, comp in by_corner.items():
        r0, c0, r1, c1 = comp["bbox"]
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        dst_corner = _NEXT_CORNER[src_corner]
        dst_r0, dst_c0 = _corner_anchor(dst_corner, height, width, obj_h, obj_w)
        color = int(comp["color"])
        for rr, cc in comp["coords"]:
            dr = dst_r0 + (rr - r0)
            dc = dst_c0 + (cc - c0)
            assert 0 <= dr < height and 0 <= dc < width
            assert (dr, dc) not in planned, "generated corner copies must not overlap"
            planned[(dr, dc)] = color

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray:
    h = _RNG.randint(1, max_h)
    w = _RNG.randint(1, max_w)
    area = h * w
    target = _RNG.randint(1, area)
    mask = np.zeros((h, w), dtype=bool)
    start = (_RNG.randrange(h), _RNG.randrange(w))
    mask[start] = True
    frontier = set()
    while int(mask.sum()) < target:
        frontier.clear()
        for r, c in np.argwhere(mask):
            for dr, dc in _DIR4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = _RNG.choice(tuple(frontier))
        mask[nr, nc] = True
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _anchored_corner_mask(corner: Corner, max_h: int, max_w: int) -> np.ndarray:
    while True:
        mask = _random_connected_mask(max_h, max_w)
        h, w = mask.shape
        if corner in ("TL", "TR"):
            top_ok = bool(mask[0, :].any())
        else:
            top_ok = bool(mask[h - 1, :].any())
        if corner in ("TL", "BL"):
            side_ok = bool(mask[:, 0].any())
        else:
            side_ok = bool(mask[:, w - 1].any())
        if top_ok and side_ok:
            return mask


def _anchor_for_corner(corner: Corner, height: int, width: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    return _corner_anchor(corner, height, width, obj_h, obj_w)


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    sub[mask] = patch[mask]
    grid[r0 : r0 + h, c0 : c0 + w] = sub


def _cells_with_gap_ok(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    cell_set = set(cells)
    for r, c in cells:
        if occupied[r, c]:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cell_set:
                    continue
                if occupied[nr, nc]:
                    return False
    return True


def _touches_true_corner(r0: int, c0: int, obj_h: int, obj_w: int, height: int, width: int) -> bool:
    top = r0 == 0
    left = c0 == 0
    bottom = r0 + obj_h - 1 == height - 1
    right = c0 + obj_w - 1 == width - 1
    return (top and left) or (top and right) or (bottom and right) or (bottom and left)


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _plan_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _random_corner_set() -> list[Corner]:
    count = _RNG.randint(1, 4)
    return _RNG.sample(["TL", "TR", "BR", "BL"], count)


def _build_corner_objects(height: int, width: int, corners: list[Corner]) -> tuple[np.ndarray, dict[Corner, tuple[np.ndarray, int]]] | None:
    grid = np.full((height, width), _BG, dtype=int)
    corner_objects: dict[Corner, tuple[np.ndarray, int]] = {}
    max_h = min(5, height - 1)
    max_w = min(5, width - 1)
    used_colors = _RNG.sample(_COLOR_POOL, len(corners))
    for corner, color in zip(corners, used_colors, strict=False):
        for _ in range(60):
            mask = _anchored_corner_mask(corner, max_h, max_w)
            patch = _render_mask(mask, color, _BG)
            obj_h, obj_w = patch.shape
            r0, c0 = _anchor_for_corner(corner, height, width, obj_h, obj_w)
            cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]
            if not _cells_with_gap_ok(grid != _BG, cells):
                continue
            _place_patch(grid, patch, r0, c0, _BG)
            corner_objects[corner] = (mask, color)
            break
        else:
            return None
    return grid, corner_objects


def _planned_stamp_cells(
    corner_objects: dict[Corner, tuple[np.ndarray, int]], height: int, width: int
) -> dict[tuple[int, int], int] | None:
    planned: dict[tuple[int, int], int] = {}
    for corner, (mask, color) in corner_objects.items():
        obj_h, obj_w = mask.shape
        dst_corner = _NEXT_CORNER[corner]
        r0, c0 = _anchor_for_corner(dst_corner, height, width, obj_h, obj_w)
        for rr, cc in np.argwhere(mask):
            cell = (r0 + int(rr), c0 + int(cc))
            if cell in planned:
                return None
            planned[cell] = color
    return planned


def _random_distractor_kind() -> str:
    return _RNG.choice(["interior", "edge"])


def _candidate_positions(height: int, width: int, obj_h: int, obj_w: int, kind: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(height - obj_h + 1):
        for c0 in range(width - obj_w + 1):
            top = r0 == 0
            left = c0 == 0
            bottom = r0 + obj_h - 1 == height - 1
            right = c0 + obj_w - 1 == width - 1
            if (top and left) or (top and right) or (bottom and left) or (bottom and right):
                continue
            edge_hits = sum((top, bottom, left, right))
            if kind == "interior" and edge_hits != 0:
                continue
            if kind == "edge" and edge_hits != 1:
                continue
            positions.append((r0, c0))
    _RNG.shuffle(positions)
    return positions


def _place_distractors(inp: np.ndarray, out: np.ndarray) -> bool:
    height, width = inp.shape
    occupied = (inp != _BG) | (out != _BG)
    count = _RNG.randint(2, 6)
    for _ in range(count):
        success = False
        for _ in range(80):
            kind = _random_distractor_kind()
            mask = _random_connected_mask(min(4, height - 1), min(4, width - 1))
            obj_h, obj_w = mask.shape
            color = _RNG.choice(_COLOR_POOL)
            patch = _render_mask(mask, color, _BG)
            for r0, c0 in _candidate_positions(height, width, obj_h, obj_w, kind):
                cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]
                if not _cells_with_gap_ok(occupied, cells):
                    continue
                _place_patch(inp, patch, r0, c0, _BG)
                _place_patch(out, patch, r0, c0, _BG)
                occupied = (inp != _BG) | (out != _BG)
                success = True
                break
            if success:
                break
        if not success:
            return False
    return True


def _bg_is_dominant(grid: np.ndarray) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = counts.get(_BG, 0)
    return all(bg_count > count for color, count in counts.items() if color != _BG)


def _gen_attempt() -> np.ndarray | None:
    height = _RNG.randint(8, 16)
    width = _RNG.randint(8, 16)
    corners = _random_corner_set()

    built = _build_corner_objects(height, width, corners)
    if built is None:
        return None
    source_grid, corner_objects = built

    planned = _planned_stamp_cells(corner_objects, height, width)
    if planned is None:
        return None

    out = source_grid.copy()
    for (r, c), color in planned.items():
        out[r, c] = color

    inp = source_grid.copy()
    if not _place_distractors(inp, out):
        return None
    if not _bg_is_dominant(inp):
        return None
    try:
        solved = _plan_output(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
