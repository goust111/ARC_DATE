from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.append(utils_dir)
            break
    from arc_utils import Color, crop_bbox, find_components, in_bounds

    return Color, crop_bbox, find_components, in_bounds


Color, crop_bbox, find_components, in_bounds = _load_arc_utils()

BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREY = Color.GREY
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_GEN_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 500
_GEN_CANVAS_MIN = 13
_GEN_CANVAS_MAX = 18
_SAFE_LAYOUTS = (
    (8, 7, 2, 1, (("top", 1), ("top", 3)), (("top", 2), ("left", 2))),
    (8, 7, 2, 1, (("top", 1), ("top", 3)), (("top", 2), ("right", 2))),
    (8, 7, 2, 1, (("top", 1), ("top", 3)), (("bottom", 2), ("left", 2))),
    (8, 7, 2, 1, (("top", 1), ("top", 3)), (("bottom", 2), ("right", 2))),
    (8, 7, 1, 1, (("bottom", 1), ("bottom", 3)), (("top", 2), ("left", 2))),
    (8, 7, 1, 1, (("bottom", 1), ("bottom", 3)), (("top", 2), ("right", 2))),
    (8, 7, 1, 1, (("bottom", 1), ("bottom", 3)), (("bottom", 2), ("left", 2))),
    (8, 7, 1, 1, (("bottom", 1), ("bottom", 3)), (("bottom", 2), ("right", 2))),
    (8, 8, 2, 2, (("top", 1), ("left", 1)), (("top", 2), ("left", 2))),
    (8, 8, 2, 2, (("top", 1), ("left", 1)), (("top", 2), ("right", 2))),
    (8, 8, 2, 2, (("top", 1), ("left", 1)), (("bottom", 2), ("left", 2))),
    (8, 8, 2, 2, (("top", 1), ("left", 1)), (("bottom", 2), ("right", 2))),
    (8, 8, 2, 1, (("top", 1), ("right", 1)), (("top", 2), ("left", 2))),
    (8, 8, 2, 1, (("top", 1), ("right", 1)), (("top", 2), ("right", 2))),
    (8, 8, 2, 1, (("top", 1), ("right", 1)), (("bottom", 2), ("left", 2))),
    (8, 8, 2, 1, (("top", 1), ("right", 1)), (("bottom", 2), ("right", 2))),
    (8, 8, 1, 2, (("bottom", 1), ("left", 1)), (("top", 2), ("left", 2))),
    (8, 8, 1, 2, (("bottom", 1), ("left", 1)), (("top", 2), ("right", 2))),
    (8, 8, 1, 2, (("bottom", 1), ("left", 1)), (("bottom", 2), ("left", 2))),
    (8, 8, 1, 2, (("bottom", 1), ("left", 1)), (("bottom", 2), ("right", 2))),
    (8, 8, 1, 1, (("bottom", 1), ("right", 1)), (("top", 2), ("left", 2))),
    (8, 8, 1, 1, (("bottom", 1), ("right", 1)), (("top", 2), ("right", 2))),
    (8, 8, 1, 1, (("bottom", 1), ("right", 1)), (("bottom", 2), ("left", 2))),
    (8, 8, 1, 1, (("bottom", 1), ("right", 1)), (("bottom", 2), ("right", 2))),
)


def _fg_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr, dc in DIRS4
        if in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc])
    ]


def _opposite_edge(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int]:
    if dr:
        rows = np.flatnonzero(mask[:, c])
        assert rows.size
        return int(rows.max() if dr > 0 else rows.min()), int(c)

    cols = np.flatnonzero(mask[r])
    assert cols.size
    return int(r), int(cols.max() if dc > 0 else cols.min())


def _outside_opposite_edge(
    mask: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[int, int] | None:
    if dr:
        rows = np.flatnonzero(mask[:, c])
        assert rows.size
        rr = int(rows.max() + 1 if dr > 0 else rows.min() - 1)
        return (rr, int(c)) if in_bounds(mask.shape, rr, c) else None

    cols = np.flatnonzero(mask[r])
    assert cols.size
    cc = int(cols.max() + 1 if dc > 0 else cols.min() - 1)
    return (int(r), cc) if in_bounds(mask.shape, r, cc) else None


def _collect_tabs(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    tabs: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]):
                continue
            neighbors = _fg_neighbors(mask, r, c)
            if len(neighbors) != 1:
                continue
            dr, dc = neighbors[0]
            target = _opposite_edge(mask, r, c, dr, dc)
            if target != (r, c):
                tabs.append(((r, c), (dr, dc), target))
    return tabs


def _collect_notches(
    mask: np.ndarray, tab_cells: set[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    notches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if bool(mask[r, c]):
                continue

            fg_dirs: list[tuple[int, int]] = []
            bg_dirs: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
                    fg_dirs.append((dr, dc))
                else:
                    bg_dirs.append((dr, dc))

            if len(fg_dirs) != 3 or len(bg_dirs) != 1:
                continue

            outside_dr, outside_dc = bg_dirs[0]
            dr, dc = -outside_dr, -outside_dc
            side_a = (r - dc, c + dr)
            side_b = (r + dc, c - dr)
            if side_a in tab_cells or side_b in tab_cells:
                continue

            target = _outside_opposite_edge(mask, r, c, dr, dc)
            if target is not None and not bool(mask[target]):
                notches.append(((r, c), (dr, dc), target))
    return notches


def _crop_interior(grid: np.ndarray) -> np.ndarray:
    corners = np.argwhere(grid == RED)
    assert corners.shape == (4, 2), "expected exactly four red corner markers"
    r0, c0 = corners.min(axis=0)
    r1, c1 = corners.max(axis=0)
    expected = {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}
    assert {tuple(map(int, p)) for p in corners} == expected, "red markers must be rectangle corners"
    return crop_bbox(grid, (int(r0) + 1, int(c0) + 1, int(r1) - 1, int(c1) - 1))


def _transform_crop(crop: np.ndarray) -> np.ndarray:
    recolored = np.where(crop != BLACK, RED, BLACK)
    mask = recolored == RED
    tabs = _collect_tabs(mask)
    tab_cells = {cell for cell, _direction, _target in tabs}
    notches = _collect_notches(mask, tab_cells)

    out = recolored.copy()
    for _cell, _direction, target in tabs:
        out[target] = BLACK
    for _cell, _direction, target in notches:
        out[target] = RED

    assert tabs or notches, "pattern must contain at least one tab or notch source"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the corner-marked inner pattern and complete opposite-side tab/notch complements after recoloring it red.

    Input structure
    - A black background contains one single-pixel grey rectangular frame whose four corners are red markers.
    - Inside that frame there is one separated blue 4-connected pattern; the pattern does not touch the frame.

    Rule
    - Use the four red corner markers to identify the frame and crop only its interior.
    - Recolor every non-black cell in the crop to red.
    - Treat each red one-neighbor boundary pixel as a tab and delete the opposite edge pixel on the same row or column span.
    - Treat each black three-neighbor boundary gap as a notch and add a red pixel one cell outside the opposite edge on the same row or column span, collecting all changes before applying them.

    Key insight
    - The red corners do two jobs at once: they define the crop window, and they also reveal the color that the inner shape must be unified to before opposite-side tab/notch completion becomes detectable by simple 4-neighbor counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _transform_crop(_crop_interior(g))


def _feature_axis(side: str, pos: int) -> tuple[str, int]:
    return ("col", pos) if side in {"top", "bottom"} else ("row", pos)


def _source_cell(
    side: str, pos: int, top: int, left: int, height: int, width: int, kind: str
) -> tuple[int, int]:
    if side == "top":
        return (top - 1, left + pos) if kind == "tab" else (top, left + pos)
    if side == "bottom":
        return (top + height, left + pos) if kind == "tab" else (top + height - 1, left + pos)
    if side == "left":
        return (top + pos, left - 1) if kind == "tab" else (top + pos, left)
    return (top + pos, left + width) if kind == "tab" else (top + pos, left + width - 1)


def _apply_feature_layout(
    inner_h: int,
    inner_w: int,
    rect_h: int,
    rect_w: int,
    tabs: list[tuple[str, int]],
    notches: list[tuple[str, int]],
    top: int,
    left: int,
) -> np.ndarray:
    crop = np.zeros((inner_h, inner_w), dtype=int)
    crop[top : top + rect_h, left : left + rect_w] = BLUE
    for side, pos in notches:
        r, c = _source_cell(side, pos, top, left, rect_h, rect_w, "notch")
        crop[r, c] = BLACK
    for side, pos in tabs:
        r, c = _source_cell(side, pos, top, left, rect_h, rect_w, "tab")
        crop[r, c] = BLUE
    return crop


def _expected_from_layout(
    crop: np.ndarray,
    rect_h: int,
    rect_w: int,
    tabs: list[tuple[str, int]],
    notches: list[tuple[str, int]],
    top: int,
    left: int,
) -> np.ndarray:
    out = np.where(crop != BLACK, RED, BLACK)
    for side, pos in tabs:
        if side == "top":
            out[top + rect_h - 1, left + pos] = BLACK
        elif side == "bottom":
            out[top, left + pos] = BLACK
        elif side == "left":
            out[top + pos, left + rect_w - 1] = BLACK
        else:
            out[top + pos, left] = BLACK

    for side, pos in notches:
        if side == "top":
            out[top + rect_h, left + pos] = RED
        elif side == "bottom":
            out[top - 1, left + pos] = RED
        elif side == "left":
            out[top + pos, left + rect_w] = RED
        else:
            out[top + pos, left - 1] = RED
    return out


def _build_grid(crop: np.ndarray) -> np.ndarray:
    inner_h, inner_w = crop.shape
    frame_h, frame_w = inner_h + 2, inner_w + 2
    canvas_h = int(_GEN_RNG.integers(max(_GEN_CANVAS_MIN, frame_h + 2), _GEN_CANVAS_MAX + 1))
    canvas_w = int(_GEN_RNG.integers(max(_GEN_CANVAS_MIN, frame_w + 2), _GEN_CANVAS_MAX + 1))
    top = int(_GEN_RNG.integers(1, canvas_h - frame_h))
    left = int(_GEN_RNG.integers(1, canvas_w - frame_w))

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    grid[top : top + frame_h, left] = GREY
    grid[top : top + frame_h, left + frame_w - 1] = GREY
    grid[top, left : left + frame_w] = GREY
    grid[top + frame_h - 1, left : left + frame_w] = GREY
    for rr, cc in (
        (top, left),
        (top, left + frame_w - 1),
        (top + frame_h - 1, left),
        (top + frame_h - 1, left + frame_w - 1),
    ):
        grid[rr, cc] = RED
    grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = crop
    return grid


def _symmetry(arr: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return arr.copy()
    if code == 1:
        return np.rot90(arr, 1).copy()
    if code == 2:
        return np.rot90(arr, 2).copy()
    if code == 3:
        return np.rot90(arr, 3).copy()
    if code == 4:
        return np.fliplr(arr).copy()
    if code == 5:
        return np.flipud(arr).copy()
    if code == 6:
        return arr.T.copy()
    return np.fliplr(arr.T).copy()


def _valid_component(crop: np.ndarray) -> bool:
    comps = find_components(
        crop,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    return len(comps) == 1


def _gen_attempt() -> np.ndarray | None:
    inner_h, inner_w, top, left, tabs, notches = _SAFE_LAYOUTS[int(_GEN_RNG.integers(len(_SAFE_LAYOUTS)))]
    rect_h = rect_w = 5
    crop = _apply_feature_layout(inner_h, inner_w, rect_h, rect_w, list(tabs), list(notches), top, left)
    expected = _expected_from_layout(crop, rect_h, rect_w, list(tabs), list(notches), top, left)
    code = int(_GEN_RNG.integers(8))
    crop = _symmetry(crop, code)
    expected = _symmetry(expected, code)
    if not (15 <= int(np.count_nonzero(crop == BLUE)) <= 25):
        return None
    if np.any(crop[0] == BLUE) or np.any(crop[-1] == BLUE) or np.any(crop[:, 0] == BLUE) or np.any(crop[:, -1] == BLUE):
        return None
    if not _valid_component(crop):
        return None

    mask = crop == BLUE
    detected_tabs = _collect_tabs(mask)
    tab_cells = {cell for cell, _direction, _target in detected_tabs}
    detected_notches = _collect_notches(mask, tab_cells)
    if len(detected_tabs) != 2 or len(detected_notches) != 2:
        return None
    recolored = np.where(crop != BLACK, RED, BLACK)
    if int(np.count_nonzero(expected != recolored)) != 4:
        return None
    if len({target for _cell, _direction, target in detected_tabs}) != 2:
        return None
    if len({target for _cell, _direction, target in detected_notches}) != 2:
        return None

    grid = _build_grid(crop)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 4")
