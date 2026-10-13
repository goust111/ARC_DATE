from __future__ import annotations

from collections import Counter
import random
import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --------------- solver helpers ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            parts.append(np.array(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(int(x) for x in np.asarray(grid).ravel())
    return int(counts.most_common(1)[0][0])


def _frame_candidate(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    if h < 5 or w < 5:
        return None

    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    row_lines = [r for r in range(h) if np.all(sub[r, :] == color)]
    col_lines = [c for c in range(w) if np.all(sub[:, c] == color)]
    if len(row_lines) < 2 or len(col_lines) < 2:
        return None
    if row_lines[0] != 0 or row_lines[-1] != h - 1:
        return None
    if col_lines[0] != 0 or col_lines[-1] != w - 1:
        return None
    if any(b - a < 2 for a, b in zip(row_lines, row_lines[1:])):
        return None
    if any(b - a < 2 for a, b in zip(col_lines, col_lines[1:])):
        return None

    expected = np.zeros((h, w), dtype=bool)
    expected[row_lines, :] = True
    expected[:, col_lines] = True
    if not np.array_equal(mask, expected):
        return None

    cells = []
    for i in range(len(row_lines) - 1):
        rr0 = row_lines[i] + 1
        rr1 = row_lines[i + 1] - 1
        for j in range(len(col_lines) - 1):
            cc0 = col_lines[j] + 1
            cc1 = col_lines[j + 1] - 1
            if rr0 > rr1 or cc0 > cc1:
                return None
            block = sub[rr0 : rr1 + 1, cc0 : cc1 + 1]
            vals = np.unique(block)
            if len(vals) != 1:
                return None
            cells.append((i, j))

    if len(cells) < 4:
        return None

    return {
        "sep": int(color),
        "bbox": (r0, c0, r1, c1),
        "row_lines": row_lines,
        "col_lines": col_lines,
        "score": (len(cells), h * w, int(mask.sum())),
    }


def _parse_template(grid: np.ndarray) -> dict:
    bg = _bg_color(grid)
    candidates: list[dict] = []
    for color in sorted(set(int(x) for x in np.unique(grid)) - {bg}):
        for coords in _components(grid == color):
            cand = _frame_candidate(grid, color, coords)
            if cand is not None:
                candidates.append(cand)
    assert candidates, "template frame not found"

    frame = max(candidates, key=lambda item: item["score"])
    sep = int(frame["sep"])
    fr0, fc0, fr1, fc1 = frame["bbox"]
    sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    row_lines = list(frame["row_lines"])
    col_lines = list(frame["col_lines"])

    cells = []
    cell_map: dict[tuple[int, int], dict] = {}
    for i in range(len(row_lines) - 1):
        rr0 = fr0 + row_lines[i] + 1
        rr1 = fr0 + row_lines[i + 1] - 1
        for j in range(len(col_lines) - 1):
            cc0 = fc0 + col_lines[j] + 1
            cc1 = fc0 + col_lines[j + 1] - 1
            block = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
            color = int(block[0, 0])
            cell = {
                "index": (i, j),
                "bbox": (rr0, cc0, rr1, cc1),
                "shape": (int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)),
                "area": int(block.size),
                "filled": color != bg,
                "color": color,
            }
            cells.append(cell)
            cell_map[(i, j)] = cell

    return {
        "bg": bg,
        "sep": sep,
        "bbox": (fr0, fc0, fr1, fc1),
        "subgrid": sub,
        "rows": len(row_lines) - 1,
        "cols": len(col_lines) - 1,
        "cells": cells,
        "cell_map": cell_map,
    }


def _collect_fragments(grid: np.ndarray, bg: int, sep: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    outer = np.ones((h, w), dtype=bool)
    outer[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    mask = outer & (grid != bg) & (grid != sep)

    fragments = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        color_vals = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(color_vals) == 1, "fragments must be monochrome"
        fh = int(r1 - r0 + 1)
        fw = int(c1 - c0 + 1)
        assert int(len(coords)) == fh * fw, "fragments must be solid rectangles"
        fragments.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (fh, fw),
                "area": int(len(coords)),
                "color": int(color_vals[0]),
            }
        )
    return fragments


def _slot_fill_colors(template: dict) -> dict[tuple[int, int], int]:
    cell_map = template["cell_map"]
    rows = int(template["rows"])
    cols = int(template["cols"])
    fills: dict[tuple[int, int], int] = {}

    for cell in template["cells"]:
        if cell["filled"]:
            continue
        i, j = cell["index"]
        neighbors = []
        for dr, dc in DIR4:
            ii, jj = i + dr, j + dc
            if 0 <= ii < rows and 0 <= jj < cols and cell_map[(ii, jj)]["filled"]:
                neighbors.append(cell_map[(ii, jj)])
        assert len(neighbors) == 1, "every empty slot must have exactly one filled neighbor"
        fills[(i, j)] = int(neighbors[0]["color"])
    return fills


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill template slots by matching outside rectangle fragments on area,
    then recolor each inserted fragment using its unique adjacent filled cell.
    """
    g = np.asarray(grid, dtype=int)
    template = _parse_template(g)
    bg = int(template["bg"])
    sep = int(template["sep"])
    empty_cells = [cell for cell in template["cells"] if not cell["filled"]]
    fragments = _collect_fragments(g, bg, sep, template["bbox"])

    empty_areas = [int(cell["area"]) for cell in empty_cells]
    frag_areas = [int(frag["area"]) for frag in fragments]
    assert len(set(empty_areas)) == len(empty_areas), "empty slot areas must be unique"
    assert len(set(frag_areas)) == len(frag_areas), "fragment areas must be unique"
    assert len(empty_cells) == len(fragments), "fragment count must match empty slots"

    frag_by_area = {int(frag["area"]): frag for frag in fragments}
    for cell in empty_cells:
        frag = frag_by_area.get(int(cell["area"]))
        assert frag is not None, "missing area-matched fragment"
        sh, sw = map(int, cell["shape"])
        fh, fw = map(int, frag["shape"])
        assert (fh, fw) == (sh, sw) or (fh, fw) == (sw, sh), "fragment shape mismatch"

    fills = _slot_fill_colors(template)

    out = np.full_like(g, bg)
    fr0, fc0, fr1, fc1 = template["bbox"]
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    for cell in empty_cells:
        r0, c0, r1, c1 = cell["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(fills[cell["index"]])
    return out


# --------------- generator ---------------


SEP_COLOR = 5
GRID_MIN = 20
GRID_MAX = 30
ROW_OPTIONS = (2, 3, 4)
COL_OPTIONS = (2, 3, 4)
CELL_H_CHOICES = (2, 3, 4, 5)
CELL_W_CHOICES = (2, 3, 4, 5, 6)
EMPTY_COUNT_MIN = 2
EMPTY_COUNT_MAX = 4
FRAME_PAD = 1
PLACEMENT_ATTEMPTS = 250
GEN_ATTEMPTS = 400

_VALID_MASK_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _valid_filled_masks(rows: int, cols: int) -> list[np.ndarray]:
    key = (rows, cols)
    if key in _VALID_MASK_CACHE:
        return _VALID_MASK_CACHE[key]

    masks: list[np.ndarray] = []
    cell_count = rows * cols
    max_empty = min(EMPTY_COUNT_MAX, cell_count - 1)

    for bits in range(1, 1 << cell_count):
        filled = np.zeros((rows, cols), dtype=bool)
        for idx in range(cell_count):
            if (bits >> idx) & 1:
                filled[idx // cols, idx % cols] = True

        empty_count = cell_count - int(filled.sum())
        if empty_count < EMPTY_COUNT_MIN or empty_count > max_empty:
            continue

        valid = True
        for r in range(rows):
            for c in range(cols):
                if filled[r, c]:
                    continue
                nfilled = 0
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and filled[rr, cc]:
                        nfilled += 1
                if nfilled != 1:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            masks.append(filled)

    _VALID_MASK_CACHE[key] = masks
    return masks


def _empty_sources(filled: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    rows, cols = filled.shape
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            if filled[r, c]:
                continue
            neighbors = []
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and filled[rr, cc]:
                    neighbors.append((rr, cc))
            if len(neighbors) != 1:
                raise ValueError("invalid template mask")
            mapping[(r, c)] = neighbors[0]
    return mapping


def _sample_cell_sizes(filled: np.ndarray) -> tuple[list[int], list[int]] | None:
    rows, cols = filled.shape
    empty_cells = [tuple(map(int, rc)) for rc in np.argwhere(~filled)]

    for _ in range(200):
        row_heights = random.sample(CELL_H_CHOICES, rows)
        col_widths = random.sample(CELL_W_CHOICES, cols)

        areas = []
        nonsquare = False
        for r, c in empty_cells:
            h = int(row_heights[r])
            w = int(col_widths[c])
            areas.append(h * w)
            nonsquare |= h != w

        frame_h = sum(row_heights) + rows + 1
        frame_w = sum(col_widths) + cols + 1
        if len(set(areas)) != len(areas):
            continue
        if not nonsquare:
            continue
        if frame_h > 22 or frame_w > 24:
            continue
        return [int(x) for x in row_heights], [int(x) for x in col_widths]
    return None


def _cell_bboxes(row_heights: list[int], col_widths: list[int], top: int, left: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    r = int(top) + 1
    for i, h in enumerate(row_heights):
        c = int(left) + 1
        for j, w in enumerate(col_widths):
            boxes[(i, j)] = (r, c, r + int(h) - 1, c + int(w) - 1)
            c += int(w) + 1
        r += int(h) + 1
    return boxes


def _draw_template(
    grid: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
    filled: np.ndarray,
    fill_colors: dict[tuple[int, int], int],
    top: int,
    left: int,
    bg: int,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    frame_h = sum(row_heights) + len(row_heights) + 1
    frame_w = sum(col_widths) + len(col_widths) + 1
    grid[top : top + frame_h, left : left + frame_w] = SEP_COLOR
    boxes = _cell_bboxes(row_heights, col_widths, top, left)
    for key, (r0, c0, r1, c1) in boxes.items():
        if filled[key]:
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(fill_colors[key])
        else:
            grid[r0 : r1 + 1, c0 : c1 + 1] = bg
    return boxes


def _mark_occ(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = occ.shape
    occ[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _place_fragments(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    fragment_specs: list[dict],
    bg: int,
) -> bool:
    h, w = grid.shape
    occ = np.zeros((h, w), dtype=bool)
    _mark_occ(occ, *frame_bbox, pad=FRAME_PAD)

    for spec in sorted(fragment_specs, key=lambda item: item["area"], reverse=True):
        fh, fw = spec["shape"]
        positions = [(r, c) for r in range(h - fh + 1) for c in range(w - fw + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions[:PLACEMENT_ATTEMPTS]:
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            if occ[max(0, r0 - FRAME_PAD) : min(h, r1 + FRAME_PAD + 1), max(0, c0 - FRAME_PAD) : min(w, c1 + FRAME_PAD + 1)].any():
                continue
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
            _mark_occ(occ, r0, c0, r1, c1, pad=FRAME_PAD)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected_output(
    shape: tuple[int, int],
    row_heights: list[int],
    col_widths: list[int],
    filled: np.ndarray,
    fill_colors: dict[tuple[int, int], int],
    empty_fill_colors: dict[tuple[int, int], int],
    top: int,
    left: int,
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    _draw_template(out, row_heights, col_widths, filled, fill_colors, top, left, bg)
    boxes = _cell_bboxes(row_heights, col_widths, top, left)
    for key, color in empty_fill_colors.items():
        r0, c0, r1, c1 = boxes[key]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.choice(ROW_OPTIONS)
    cols = random.choice(COL_OPTIONS)
    masks = _valid_filled_masks(rows, cols)
    if not masks:
        return None
    filled = random.choice(masks).copy()

    sizes = _sample_cell_sizes(filled)
    if sizes is None:
        return None
    row_heights, col_widths = sizes

    frame_h = sum(row_heights) + rows + 1
    frame_w = sum(col_widths) + cols + 1
    min_h = max(GRID_MIN, frame_h + 4)
    min_w = max(GRID_MIN, frame_w + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = random.randint(min_h, GRID_MAX)
    grid_w = random.randint(min_w, GRID_MAX)
    top = random.randint(1, grid_h - frame_h - 1)
    left = random.randint(1, grid_w - frame_w - 1)

    bg_choices = [c for c in range(10) if c != SEP_COLOR]
    bg = random.choice(bg_choices)

    empty_to_source = _empty_sources(filled)
    source_cells = sorted(set(empty_to_source.values()))
    empty_cells = sorted(empty_to_source.keys())

    available = [c for c in range(10) if c not in {bg, SEP_COLOR}]
    random.shuffle(available)
    need = len(source_cells) + len(empty_cells)
    if need > len(available):
        return None
    source_palette = available[: len(source_cells)]
    fragment_palette = available[len(source_cells) : need]
    source_colors = {cell: int(color) for cell, color in zip(source_cells, source_palette)}

    fill_colors: dict[tuple[int, int], int] = {}
    for key in [tuple(map(int, rc)) for rc in np.argwhere(filled)]:
        if key in source_colors:
            fill_colors[key] = int(source_colors[key])
        else:
            fill_colors[key] = int(random.choice(source_palette))

    empty_fill_colors = {cell: int(source_colors[src]) for cell, src in empty_to_source.items()}

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    boxes = _draw_template(grid, row_heights, col_widths, filled, fill_colors, top, left, bg)
    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)

    nonsquare = [cell for cell in empty_cells if boxes[cell][2] - boxes[cell][0] != boxes[cell][3] - boxes[cell][1]]
    rotate_flags = {cell: False for cell in empty_cells}
    if len(nonsquare) == 1:
        rotate_flags[nonsquare[0]] = bool(random.getrandbits(1))
    elif len(nonsquare) >= 2:
        for cell in nonsquare:
            rotate_flags[cell] = bool(random.getrandbits(1))
        if not any(rotate_flags[cell] for cell in nonsquare):
            rotate_flags[random.choice(nonsquare)] = True
        if all(rotate_flags[cell] for cell in nonsquare):
            rotate_flags[random.choice(nonsquare)] = False

    fragment_specs = []
    for color, cell in zip(fragment_palette, empty_cells):
        r0, c0, r1, c1 = boxes[cell]
        sh = int(r1 - r0 + 1)
        sw = int(c1 - c0 + 1)
        if rotate_flags[cell]:
            shape = (sw, sh)
        else:
            shape = (sh, sw)
        fragment_specs.append({"shape": shape, "area": sh * sw, "color": int(color)})

    if not _place_fragments(grid, frame_bbox, fragment_specs, bg):
        return None

    if _bg_color(grid) != bg:
        return None

    expected = _build_expected_output(
        (grid_h, grid_w),
        row_heights,
        col_widths,
        filled,
        fill_colors,
        empty_fill_colors,
        top,
        left,
        bg,
    )

    try:
        result = main(grid)
    except Exception:
        return None
    if np.array_equal(result, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
