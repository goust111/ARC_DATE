from __future__ import annotations

import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray, *, bg: int, connectivity: int = 4, monochromatic: bool = True
) -> list[dict]:
    assert connectivity == 4
    assert monochromatic

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
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return components


def _row_runs(mask: np.ndarray) -> list[tuple[int, int]] | None:
    runs: list[tuple[int, int]] = []
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if len(cols) == 0:
            return None
        if int(cols[-1] - cols[0] + 1) != int(len(cols)):
            return None
        runs.append((int(cols[0]), int(cols[-1])))
    return runs


def _col_runs(mask: np.ndarray) -> list[tuple[int, int]] | None:
    runs: list[tuple[int, int]] = []
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        if len(rows) == 0:
            return None
        if int(rows[-1] - rows[0] + 1) != int(len(rows)):
            return None
        runs.append((int(rows[0]), int(rows[-1])))
    return runs


def _match_vertical_triangle(mask: np.ndarray, kind: str) -> dict | None:
    runs = _row_runs(mask)
    if runs is None or len(runs) < 2:
        return None

    widths = [hi - lo + 1 for lo, hi in runs]
    centers2 = [lo + hi for lo, hi in runs]
    if len(set(centers2)) != 1:
        return None

    if kind == "up":
        if any(widths[i + 1] - widths[i] != 2 for i in range(len(widths) - 1)):
            return None
        base_row_local = mask.shape[0] - 1
        base_lo, base_hi = runs[-1]
    else:
        if any(widths[i] - widths[i + 1] != 2 for i in range(len(widths) - 1)):
            return None
        base_row_local = 0
        base_lo, base_hi = runs[0]

    tip_width = min(widths[0], widths[-1])
    if tip_width not in (1, 2):
        return None

    return {
        "kind": kind,
        "base_row_local": int(base_row_local),
        "base_lo_local": int(base_lo),
        "base_hi_local": int(base_hi),
        "base_len": int(base_hi - base_lo + 1),
    }


def _match_horizontal_triangle(mask: np.ndarray, kind: str) -> dict | None:
    runs = _col_runs(mask)
    if runs is None or len(runs) < 2:
        return None

    heights = [hi - lo + 1 for lo, hi in runs]
    centers2 = [lo + hi for lo, hi in runs]
    if len(set(centers2)) != 1:
        return None

    if kind == "left":
        if any(heights[i + 1] - heights[i] != 2 for i in range(len(heights) - 1)):
            return None
        base_col_local = mask.shape[1] - 1
        base_lo, base_hi = runs[-1]
    else:
        if any(heights[i] - heights[i + 1] != 2 for i in range(len(heights) - 1)):
            return None
        base_col_local = 0
        base_lo, base_hi = runs[0]

    tip_height = min(heights[0], heights[-1])
    if tip_height not in (1, 2):
        return None

    return {
        "kind": kind,
        "base_col_local": int(base_col_local),
        "base_lo_local": int(base_lo),
        "base_hi_local": int(base_hi),
        "base_len": int(base_hi - base_lo + 1),
    }


def _analyze_triangle(grid: np.ndarray, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color

    matches = []
    for kind in ("up", "down"):
        info = _match_vertical_triangle(mask, kind)
        if info is not None:
            matches.append(info)
    for kind in ("left", "right"):
        info = _match_horizontal_triangle(mask, kind)
        if info is not None:
            matches.append(info)

    if len(matches) != 1:
        return None

    info = matches[0]
    kind = info["kind"]
    desc = {
        "color": color,
        "kind": kind,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(comp["area"]),
    }

    if kind in ("up", "down"):
        base_row = int(r0 + info["base_row_local"])
        base_c0 = int(c0 + info["base_lo_local"])
        base_c1 = int(c0 + info["base_hi_local"])
        desc.update(
            {
                "base_row": base_row,
                "base_c0": base_c0,
                "base_c1": base_c1,
                "base_len": int(info["base_len"]),
            }
        )
    else:
        base_col = int(c0 + info["base_col_local"])
        base_r0 = int(r0 + info["base_lo_local"])
        base_r1 = int(r0 + info["base_hi_local"])
        desc.update(
            {
                "base_col": base_col,
                "base_r0": base_r0,
                "base_r1": base_r1,
                "base_len": int(info["base_len"]),
            }
        )

    return desc


def _triangle_patch(kind: str, base_len: int, color: int, bg: int = 0) -> np.ndarray:
    depth = (base_len + 1) // 2
    if kind in ("up", "down"):
        patch = np.full((depth, base_len), bg, dtype=int)
        for t in range(depth):
            width = base_len - 2 * t
            left = t
            row = depth - 1 - t if kind == "up" else t
            patch[row, left : left + width] = color
        return patch

    patch = np.full((base_len, depth), bg, dtype=int)
    for t in range(depth):
        height = base_len - 2 * t
        top = t
        col = depth - 1 - t if kind == "left" else t
        patch[top : top + height, col] = color
    return patch


def _expanded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        max(0, int(r0) - pad),
        max(0, int(c0) - pad),
        min(h - 1, int(r1) + pad),
        min(w - 1, int(c1) + pad),
    )


def _bbox_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = _expanded_bbox(bbox, occupied.shape, pad=pad)
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = _expanded_bbox(bbox, occupied.shape, pad=pad)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _count_pairs(grid: np.ndarray, bg: int) -> int:
    tris = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        tri = _analyze_triangle(grid, comp)
        assert tri is not None, "every foreground component must be a valid residue triangle"
        tris.append(tri)

    h, w = grid.shape
    down_map = {
        (t["color"], t["base_len"], t["base_c0"], t["base_c1"], t["base_row"]): t
        for t in tris
        if t["kind"] == "down"
    }
    right_map = {
        (t["color"], t["base_len"], t["base_r0"], t["base_r1"], t["base_col"]): t
        for t in tris
        if t["kind"] == "right"
    }

    count = 0
    for tri in tris:
        if tri["kind"] == "up":
            gap_row = tri["base_row"] + 1
            start = tri["base_c0"] - 1
            end = tri["base_c1"] + 1
            mate = down_map.get(
                (tri["color"], tri["base_len"], tri["base_c0"], tri["base_c1"], tri["base_row"] + 2)
            )
            if mate is not None and 0 <= start and end < w and np.all(grid[gap_row, start : end + 1] == bg):
                count += 1
        elif tri["kind"] == "left":
            gap_col = tri["base_col"] + 1
            start = tri["base_r0"] - 1
            end = tri["base_r1"] + 1
            mate = right_map.get(
                (tri["color"], tri["base_len"], tri["base_r0"], tri["base_r1"], tri["base_col"] + 2)
            )
            if mate is not None and 0 <= start and end < h and np.all(grid[start : end + 1, gap_col] == bg):
                count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the erased generator bars from paired triangular envelope residues.

    Input structure:
    - A black background contains several solid monochrome isosceles triangles whose row widths or column heights shrink by 2 each step.
    - Some same-color triangles come in horizontal or vertical facing pairs with exactly one black row or one black column between their bases.
    - Other triangles are unmatched distractors and should not survive.

    Rule:
    - Detect every valid axis-aligned triangle residue and classify its facing direction.
    - Pair same-color triangles whose bases are parallel, equal, centered the same way, face each other, and are separated by exactly one black seam.
    - For each pair, fill the black seam with a centered bar of the same color whose length is the visible widest base plus 2, then erase all triangles.

    Key insight:
    - The triangles are not output objects; they are the two halves of a diamond envelope left behind after the original middle bar was erased, so the hidden bar must be inferred as one cell longer on each end than the widest visible residue row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    h, w = g.shape

    triangles = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        tri = _analyze_triangle(g, comp)
        assert tri is not None, "every foreground component must be a valid residue triangle"
        triangles.append(tri)

    down_map = {
        (t["color"], t["base_len"], t["base_c0"], t["base_c1"], t["base_row"]): t
        for t in triangles
        if t["kind"] == "down"
    }
    right_map = {
        (t["color"], t["base_len"], t["base_r0"], t["base_r1"], t["base_col"]): t
        for t in triangles
        if t["kind"] == "right"
    }

    out = np.full_like(g, bg)

    for tri in triangles:
        color = int(tri["color"])
        if tri["kind"] == "up":
            gap_row = tri["base_row"] + 1
            start = tri["base_c0"] - 1
            end = tri["base_c1"] + 1
            mate = down_map.get(
                (color, tri["base_len"], tri["base_c0"], tri["base_c1"], tri["base_row"] + 2)
            )
            if mate is not None and 0 <= start and end < w and np.all(g[gap_row, start : end + 1] == bg):
                out[gap_row, start : end + 1] = color
        elif tri["kind"] == "left":
            gap_col = tri["base_col"] + 1
            start = tri["base_r0"] - 1
            end = tri["base_r1"] + 1
            mate = right_map.get(
                (color, tri["base_len"], tri["base_r0"], tri["base_r1"], tri["base_col"] + 2)
            )
            if mate is not None and 0 <= start and end < h and np.all(g[start : end + 1, gap_col] == bg):
                out[start : end + 1, gap_col] = color

    return out


PAIR_COUNT_CHOICES = (1, 2, 2, 3)
GRID_RANGE = (18, 30)
PAIR_LINE_LENGTHS = tuple(range(5, 12))
NOISE_BASE_LENGTHS = tuple(range(3, 9))
PAIR_ORIENTATIONS = ("H", "V")
TRIANGLE_KINDS = ("up", "down", "left", "right")
GEN_RETRIES = 400


def _sample_pair_orientations(count: int) -> list[str]:
    if count == 1:
        return [random.choice(PAIR_ORIENTATIONS)]
    if random.random() < 0.75:
        out = ["H", "V"] + [random.choice(PAIR_ORIENTATIONS) for _ in range(count - 2)]
        random.shuffle(out)
        return out
    return [random.choice(PAIR_ORIENTATIONS) for _ in range(count)]


def _paint_pair(inp: np.ndarray, out: np.ndarray, orientation: str, color: int, anchor: tuple[int, int], length: int) -> None:
    if orientation == "H":
        row, c0 = anchor
        base_len = length - 2
        depth = (length - 1) // 2
        top_patch = _triangle_patch("up", base_len, color)
        bottom_patch = _triangle_patch("down", base_len, color)
        top_view = inp[row - depth : row, c0 + 1 : c0 + length - 1]
        bottom_view = inp[row + 1 : row + depth + 1, c0 + 1 : c0 + length - 1]
        top_view[top_patch != 0] = color
        bottom_view[bottom_patch != 0] = color
        out[row, c0 : c0 + length] = color
        return

    r0, col = anchor
    base_len = length - 2
    depth = (length - 1) // 2
    left_patch = _triangle_patch("left", base_len, color)
    right_patch = _triangle_patch("right", base_len, color)
    left_view = inp[r0 + 1 : r0 + length - 1, col - depth : col]
    right_view = inp[r0 + 1 : r0 + length - 1, col + 1 : col + depth + 1]
    left_view[left_patch != 0] = color
    right_view[right_patch != 0] = color
    out[r0 : r0 + length, col] = color


def _pair_bbox(shape: tuple[int, int], orientation: str, anchor: tuple[int, int], length: int) -> tuple[int, int, int, int]:
    if orientation == "H":
        row, c0 = anchor
        depth = (length - 1) // 2
        return row - depth, c0, row + depth, c0 + length - 1
    r0, col = anchor
    depth = (length - 1) // 2
    return r0, col - depth, r0 + length - 1, col + depth


def _noise_bbox(kind: str, top: int, left: int, base_len: int) -> tuple[int, int, int, int]:
    patch = _triangle_patch(kind, base_len, 1)
    h, w = patch.shape
    return top, left, top + h - 1, left + w - 1


def _place_pair(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    orientation: str,
    color: int,
) -> bool:
    h, w = inp.shape
    lengths = [L for L in PAIR_LINE_LENGTHS if (L <= w and 2 * ((L - 1) // 2) + 1 <= h)] if orientation == "H" else [
        L for L in PAIR_LINE_LENGTHS if (L <= h and 2 * ((L - 1) // 2) + 1 <= w)
    ]
    random.shuffle(lengths)

    for length in lengths:
        depth = (length - 1) // 2
        if orientation == "H":
            rows = list(range(depth, h - depth))
            cols = list(range(0, w - length + 1))
            random.shuffle(rows)
            random.shuffle(cols)
            for row in rows:
                for c0 in cols:
                    bbox = _pair_bbox(inp.shape, orientation, (row, c0), length)
                    if _bbox_free(occupied, bbox, pad=1):
                        _paint_pair(inp, out, orientation, color, (row, c0), length)
                        _mark_bbox(occupied, bbox, pad=1)
                        return True
        else:
            rows = list(range(0, h - length + 1))
            cols = list(range(depth, w - depth))
            random.shuffle(rows)
            random.shuffle(cols)
            for r0 in rows:
                for col in cols:
                    bbox = _pair_bbox(inp.shape, orientation, (r0, col), length)
                    if _bbox_free(occupied, bbox, pad=1):
                        _paint_pair(inp, out, orientation, color, (r0, col), length)
                        _mark_bbox(occupied, bbox, pad=1)
                        return True
    return False


def _place_noise(inp: np.ndarray, occupied: np.ndarray, kind: str, color: int) -> bool:
    h, w = inp.shape
    base_lengths = list(NOISE_BASE_LENGTHS)
    random.shuffle(base_lengths)

    for base_len in base_lengths:
        patch = _triangle_patch(kind, base_len, color)
        ph, pw = patch.shape
        if ph > h or pw > w:
            continue
        rows = list(range(0, h - ph + 1))
        cols = list(range(0, w - pw + 1))
        random.shuffle(rows)
        random.shuffle(cols)
        for top in rows:
            for left in cols:
                bbox = _noise_bbox(kind, top, left, base_len)
                if _bbox_free(occupied, bbox, pad=1):
                    mask = patch != 0
                    region = inp[top : top + ph, left : left + pw]
                    if bool(region[mask].any()):
                        continue
                    region[mask] = patch[mask]
                    _mark_bbox(occupied, bbox, pad=1)
                    return True
    return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    pair_count = random.choice(PAIR_COUNT_CHOICES)
    max_noise = min(3, 9 - pair_count)
    if max_noise == 0:
        noise_count = 0
    elif random.random() < 0.35:
        noise_count = 0
    else:
        noise_count = random.randint(1, max_noise)

    colors = random.sample(list(range(1, 10)), pair_count + noise_count)
    pair_colors = colors[:pair_count]
    noise_colors = colors[pair_count:]

    for orientation, color in zip(_sample_pair_orientations(pair_count), pair_colors):
        if not _place_pair(inp, expected, occupied, orientation, color):
            return None

    noise_kinds = [random.choice(TRIANGLE_KINDS) for _ in range(noise_count)]
    for kind, color in zip(noise_kinds, noise_colors):
        if not _place_noise(inp, occupied, kind, color):
            return None

    if _count_pairs(inp, 0) != pair_count:
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residue-to-bar puzzle")
