from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if not candidate.exists():
            continue
        util_dir = str(candidate.parent)
        if util_dir not in sys.path:
            sys.path.insert(0, util_dir)
        try:
            import arc_utils  # type: ignore
        except Exception:
            return None
        return arc_utils
    return None


_ARC = _load_arc_utils()


if _ARC is not None:
    bbox = _ARC.bbox
    crop_bbox = _ARC.crop_bbox
    find_components = _ARC.find_components
    get_bg = _ARC.get_bg
else:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(values[np.argmax(counts)])

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box):
        r0, c0, r1, c1 = map(int, box)
        return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        if bg is None:
            bg = get_bg(grid)
        if connectivity != 4:
            raise ValueError("fallback find_components only supports 4-connectivity")
        if return_format != "dict":
            raise ValueError("fallback find_components only supports return_format='dict'")

        g = np.asarray(grid, dtype=int)
        if colors is None:
            color_list = [int(c) for c in np.unique(g) if int(c) != int(bg)]
        else:
            color_list = [int(c) for c in colors if int(c) != int(bg)]

        if not monochromatic:
            raise ValueError("fallback find_components only supports monochromatic=True")

        h, w = g.shape
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for color in color_list:
            coords = np.argwhere(g == color)
            for r, c in coords:
                r = int(r)
                c = int(c)
                if seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                if len(comp) < int(min_area):
                    continue
                comp_coords = np.array(comp, dtype=int)
                r0, c0 = comp_coords.min(axis=0)
                r1, c1 = comp_coords.max(axis=0)
                out.append(
                    {
                        "color": color,
                        "coords": comp_coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(len(comp)),
                    }
                )
        return out


BG = 0
SOURCE = 1
MARKER = 2
DIVIDER = 5
ROW_COUNT = 2
QUADRANTS = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 400


@dataclass(frozen=True)
class MarkerSpec:
    slot_row: int
    slot_col: int
    rel_r: int
    rel_c: int
    quadrant: str


@dataclass(frozen=True)
class PuzzleSpec:
    left_width: int
    cols: int
    sprite_top: int
    sprite_left: int
    markers: tuple[MarkerSpec, ...]


def bbox_h(box) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box) -> int:
    return int(box[3] - box[1] + 1)


def _quadrant_slices(h: int, w: int, quadrant: str) -> tuple[slice, slice]:
    row_slice = slice(0, h // 2) if quadrant[0] == "T" else slice(h // 2, h)
    col_slice = slice(0, w // 2) if quadrant[1] == "L" else slice(w // 2, w)
    return row_slice, col_slice


def _quadrant_of(rel_r: int, rel_c: int, h: int, w: int) -> str:
    vertical = "T" if int(rel_r) < h // 2 else "B"
    horizontal = "L" if int(rel_c) < w // 2 else "R"
    return vertical + horizontal


def _slot_origin(slot_row: int, slot_col: int, sep_col: int, slot_h: int, slot_w: int) -> tuple[int, int]:
    return int(slot_row * (slot_h + 1)), int(sep_col + 1 + slot_col * (slot_w + 1))


def _source_patch(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[SOURCE],
        min_area=1,
        return_format="dict",
    )
    assert len(comps) == 1, "expected exactly one blue source sprite"
    box = comps[0]["bbox"]
    patch = np.where(crop_bbox(grid, box) == SOURCE, SOURCE, BG)
    return patch, box


def _separator_col(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    sep_cols = np.flatnonzero(np.all(g == DIVIDER, axis=0))
    assert len(sep_cols) == 1, "expected exactly one full-height grey separator column"
    assert int(np.count_nonzero(g == DIVIDER)) == g.shape[0], "separator must be a single uninterrupted column"
    return int(sep_cols[0])


def _panel_shape(grid: np.ndarray, sep_col: int, slot_h: int, slot_w: int) -> tuple[int, int]:
    height, width = map(int, grid.shape)
    assert (height + 1) % (slot_h + 1) == 0, "grid height must match a uniform slot-row layout"
    rows = (height + 1) // (slot_h + 1)
    right_width = width - sep_col - 1
    assert right_width > 0, "right panel must exist"
    assert (right_width + 1) % (slot_w + 1) == 0, "right panel width must match uniform slot columns"
    cols = (right_width + 1) // (slot_w + 1)
    assert rows == ROW_COUNT, "this task uses exactly two slot rows"
    assert cols in (2, 3), "right panel must be a 2x2 or 2x3 slot grid"
    return int(rows), int(cols)


def _parse_markers(
    grid: np.ndarray,
    sep_col: int,
    slot_h: int,
    slot_w: int,
    rows: int,
    cols: int,
) -> list[MarkerSpec]:
    g = np.asarray(grid, dtype=int)
    panel = g[:, sep_col + 1 :]
    assert np.all(np.isin(panel, [BG, MARKER])), "right panel may only contain black cells and red markers"

    for row_gap in range(1, rows):
        gap_r = row_gap * slot_h + (row_gap - 1)
        assert np.all(g[gap_r, sep_col + 1 :] == BG), "row gaps between slots must stay black"
    for col_gap in range(1, cols):
        gap_c = sep_col + col_gap * slot_w + col_gap
        assert np.all(g[:, gap_c] == BG), "column gaps between slots must stay black"

    markers = []
    for slot_row in range(rows):
        for slot_col in range(cols):
            r0, c0 = _slot_origin(slot_row, slot_col, sep_col, slot_h, slot_w)
            slot = g[r0 : r0 + slot_h, c0 : c0 + slot_w]
            red_coords = np.argwhere(slot == MARKER)
            assert len(red_coords) <= 1, "each slot may contain at most one marker"
            if len(red_coords) == 0:
                assert np.all(slot == BG), "unmarked slots must be blank"
                continue
            rel_r, rel_c = map(int, red_coords[0])
            markers.append(
                MarkerSpec(
                    slot_row=slot_row,
                    slot_col=slot_col,
                    rel_r=rel_r,
                    rel_c=rel_c,
                    quadrant=_quadrant_of(rel_r, rel_c, slot_h, slot_w),
                )
            )
    assert 2 <= len(markers) <= 4, "expected two to four marked slots"
    return markers


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, list[MarkerSpec]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG, "background should stay black"
    sprite, _box = _source_patch(g)
    slot_h, slot_w = sprite.shape
    sep_col = _separator_col(g)
    rows, cols = _panel_shape(g, sep_col, slot_h, slot_w)
    markers = _parse_markers(g, sep_col, slot_h, slot_w, rows, cols)
    return sprite, sep_col, rows, cols, markers


def _apply_semantics(grid: np.ndarray, sprite: np.ndarray, sep_col: int, markers: list[MarkerSpec]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == MARKER] = BG
    slot_h, slot_w = sprite.shape

    for marker in markers:
        r0, c0 = _slot_origin(marker.slot_row, marker.slot_col, sep_col, slot_h, slot_w)
        slot = out[r0 : r0 + slot_h, c0 : c0 + slot_w]
        slot[:, :] = BG
        row_slice, col_slice = _quadrant_slices(slot_h, slot_w, marker.quadrant)
        slot[row_slice, col_slice] = sprite[row_slice, col_slice]
        slot[marker.rel_r, marker.rel_c] = BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the left sprite into marked slots, then keep only the marker-selected quadrant of each copy.

    Input structure:
    - A black background contains one blue 4-connected source sprite on the left and one full-height grey separator column.
    - The right side is a regular 2x2 or 2x3 grid of same-sized black slots separated by one black row/column gap.
    - Some slots contain a single red marker pixel; unmarked slots are otherwise empty.

    Rule:
    - Crop the blue source sprite to its bbox and align that bbox to every marked slot.
    - Determine the marker quadrant relative to the slot center, with center-column cells counting as right and center-row cells counting as bottom.
    - In each marked slot keep only the source pixels inside that quadrant, erase the other three quadrants, and remove the red marker pixel.

    Key insight:
    - The red marker has two coupled roles: it selects which slots receive the stamp and its exact in-slot position decides which quadrant of the stamp survives.
    """
    sprite, sep_col, _rows, _cols, markers = _analyze(np.asarray(grid, dtype=int))
    return _apply_semantics(grid, sprite, sep_col, markers)


def _random_path(rng: np.random.Generator, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    moves = []
    if er != r:
        moves.extend([(1 if er > r else -1, 0)] * abs(er - r))
    if ec != c:
        moves.extend([(0, 1 if ec > c else -1)] * abs(ec - c))
    rng.shuffle(moves)
    out = [(r, c)]
    for dr, dc in moves:
        r += dr
        c += dc
        out.append((r, c))
    return out


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return list(frontier)


def _sprite_ok(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    h, w = mask.shape
    if box != (0, 0, h - 1, w - 1):
        return False
    if np.all(mask):
        return False
    for quadrant in QUADRANTS:
        row_slice, col_slice = _quadrant_slices(h, w, quadrant)
        sub = mask[row_slice, col_slice]
        if not np.any(sub):
            return False
        if np.all(sub):
            return False
    return True


def _random_sprite(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    for _ in range(250):
        center = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        anchors = [
            (0, int(rng.integers(0, w))),
            (h - 1, int(rng.integers(0, w))),
            (int(rng.integers(0, h)), 0),
            (int(rng.integers(0, h)), w - 1),
        ]
        mask = np.zeros((h, w), dtype=bool)
        mask[center] = True
        for anchor in anchors:
            for rr, cc in _random_path(rng, center, anchor):
                mask[rr, cc] = True

        min_target = max(int(mask.sum()), max(7, (h * w) // 2))
        max_target = min(h * w - 2, int(np.ceil(h * w * 0.72)))
        if min_target > max_target:
            continue
        target = int(rng.integers(min_target, max_target + 1))
        while int(mask.sum()) < target:
            frontier = _frontier_cells(mask)
            if not frontier:
                break
            rr, cc = frontier[int(rng.integers(0, len(frontier)))]
            mask[rr, cc] = True

        if not _sprite_ok(mask):
            continue
        return np.where(mask, SOURCE, BG)
    return None


def _quadrant_cells(h: int, w: int, quadrant: str) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            if _quadrant_of(r, c, h, w) == quadrant:
                cells.append((r, c))
    return cells


def _choose_marker_cell(
    rng: np.random.Generator,
    sprite: np.ndarray,
    quadrant: str,
    *,
    force_overlap: bool,
) -> tuple[int, int]:
    h, w = sprite.shape
    candidates = _quadrant_cells(h, w, quadrant)
    overlap = [(r, c) for r, c in candidates if int(sprite[r, c]) == SOURCE]
    empty = [(r, c) for r, c in candidates if int(sprite[r, c]) == BG]
    edge = [
        (r, c)
        for r, c in candidates
        if (h % 2 == 1 and r == h // 2) or (w % 2 == 1 and c == w // 2)
    ]

    if force_overlap and overlap:
        pool = overlap
    else:
        if overlap and empty:
            pool = overlap if rng.random() < 0.5 else empty
        elif overlap:
            pool = overlap
        else:
            pool = empty

    edge_pool = [cell for cell in pool if cell in edge]
    if edge_pool and rng.random() < 0.35:
        pool = edge_pool
    return pool[int(rng.integers(0, len(pool)))]


def _expected_from_spec(grid: np.ndarray, sprite: np.ndarray, sep_col: int, markers: tuple[MarkerSpec, ...]) -> np.ndarray:
    return _apply_semantics(grid, sprite, sep_col, list(markers))


def _validate_generated(grid: np.ndarray, sprite: np.ndarray, spec: PuzzleSpec) -> bool:
    h, w = sprite.shape
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if grid.shape[0] != ROW_COUNT * h + (ROW_COUNT - 1):
        return False
    if len(spec.markers) >= ROW_COUNT * spec.cols:
        return False
    if len({marker.quadrant for marker in spec.markers}) != len(spec.markers):
        return False
    if not any(int(sprite[marker.rel_r, marker.rel_c]) == SOURCE for marker in spec.markers):
        return False

    sep_col = spec.left_width
    parsed_sprite, parsed_sep, parsed_rows, parsed_cols, parsed_markers = _analyze(grid)
    if parsed_sep != sep_col or parsed_rows != ROW_COUNT or parsed_cols != spec.cols:
        return False
    if not np.array_equal(parsed_sprite, sprite):
        return False

    parsed_map = {(m.slot_row, m.slot_col): m for m in parsed_markers}
    for marker in spec.markers:
        key = (marker.slot_row, marker.slot_col)
        if key not in parsed_map or parsed_map[key] != marker:
            return False

    expected = _expected_from_spec(grid, sprite, sep_col, spec.markers)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    marked_slots = {(m.slot_row, m.slot_col) for m in spec.markers}
    slot_signatures = set()
    for slot_row in range(ROW_COUNT):
        for slot_col in range(spec.cols):
            r0, c0 = _slot_origin(slot_row, slot_col, sep_col, h, w)
            out_slot = expected[r0 : r0 + h, c0 : c0 + w]
            if (slot_row, slot_col) in marked_slots:
                if np.array_equal(out_slot, sprite):
                    return False
                slot_signatures.add(out_slot.tobytes())
            elif np.any(out_slot != BG):
                return False
    if len(slot_signatures) != len(spec.markers):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cols = int(rng.choice([2, 3]))
    h = int(rng.integers(4, 7))
    w = int(rng.integers(4, 7))
    right_width = cols * w + (cols - 1)
    max_left_width = 30 - 1 - right_width
    if max_left_width < w:
        return None
    left_low = max(w, 5)
    left_high = min(max_left_width, w + 4)
    if left_high < left_low:
        left_low = w
        left_high = max_left_width
    left_width = int(rng.integers(left_low, left_high + 1))

    total_height = ROW_COUNT * h + (ROW_COUNT - 1)
    grid = np.full((total_height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, left_width] = DIVIDER

    sprite = _random_sprite(rng, h, w)
    if sprite is None:
        return None

    sprite_top = int(rng.integers(0, total_height - h + 1))
    sprite_left = int(rng.integers(0, left_width - w + 1))
    sprite_mask = sprite == SOURCE
    patch = grid[sprite_top : sprite_top + h, sprite_left : sprite_left + w]
    patch[sprite_mask] = SOURCE

    all_slots = [(slot_row, slot_col) for slot_row in range(ROW_COUNT) for slot_col in range(cols)]
    if cols == 2:
        marked_count = int(rng.choice([2, 3]))
    else:
        marked_count = int(rng.choice([2, 3, 4]))
    marked_indices = rng.choice(len(all_slots), size=marked_count, replace=False)
    chosen_slots = [all_slots[int(i)] for i in marked_indices]
    quadrants = list(rng.choice(QUADRANTS, size=marked_count, replace=False))
    rng.shuffle(quadrants)
    force_overlap_index = int(rng.integers(0, marked_count))

    markers = []
    for idx, ((slot_row, slot_col), quadrant) in enumerate(zip(chosen_slots, quadrants)):
        rel_r, rel_c = _choose_marker_cell(
            rng,
            sprite,
            quadrant,
            force_overlap=(idx == force_overlap_index),
        )
        r0, c0 = _slot_origin(slot_row, slot_col, left_width, h, w)
        grid[r0 + rel_r, c0 + rel_c] = MARKER
        markers.append(
            MarkerSpec(
                slot_row=slot_row,
                slot_col=slot_col,
                rel_r=rel_r,
                rel_c=rel_c,
                quadrant=quadrant,
            )
        )

    spec = PuzzleSpec(
        left_width=left_width,
        cols=cols,
        sprite_top=sprite_top,
        sprite_left=sprite_left,
        markers=tuple(markers),
    )
    if not _validate_generated(grid, sprite, spec):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-quadrant stamping input")
