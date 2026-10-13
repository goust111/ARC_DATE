from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


RED = 2
GREEN = 3
PINK = 6
TEAL = 8
BLACK = 0

CELL = 5
TABLE_DIM = 3
QUERY_H = 5
TABLE_H = TABLE_DIM * CELL
SEP1_ROW = QUERY_H
TABLE_ROW0 = SEP1_ROW + 1
SEP2_ROW = TABLE_ROW0 + TABLE_H
VERDICT_H = 1
RED_BAND_H = 2

GLYPH_COLORS = (1, 4, 5, 7, 9)
RECT_COLORS = (1, 4, 5, 7, 9)
MAX_RETRIES = 400


def _shape_pool() -> list[int]:
    pool: list[int] = []
    for mask in range(1, 1 << 9):
        bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3)
        if bits.sum() < 4 or bits.sum() > 7:
            continue
        if not bits[0].any() or not bits[2].any() or not bits[:, 0].any() or not bits[:, 2].any():
            continue
        pool.append(mask)
    return pool


SHAPE_POOL = _shape_pool()


def _render_glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), TEAL, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = int(color)
    return cell


def _cell_signature(cell: np.ndarray, bg: int = TEAL) -> tuple[tuple[int, int, int], ...] | None:
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    coords = np.argwhere(mask)
    return tuple((int(r), int(c), int(cell[r, c])) for r, c in coords)


def _constant_rows(g: np.ndarray, color: int | None = None) -> list[int]:
    rows: list[int] = []
    for r in range(g.shape[0]):
        if not bool(np.all(g[r, :] == g[r, 0])):
            continue
        if color is not None and int(g[r, 0]) != int(color):
            continue
        rows.append(int(r))
    return rows


def _locate_layout(g: np.ndarray) -> dict[str, int]:
    divider_candidates = [
        r
        for r in range(g.shape[0] - 1)
        if bool(np.all(g[r, :] == RED)) and bool(np.all(g[r + 1, :] == RED))
    ]
    assert divider_candidates, "expected a 2-row red divider band"
    red_row0 = max(divider_candidates)

    sep_rows = [r for r in _constant_rows(g[:red_row0, :], PINK)]
    assert len(sep_rows) >= 2, "expected two pink separator rows"
    sep1, sep2 = int(sep_rows[0]), int(sep_rows[1])

    table_region = g[sep1 + 1 : sep2, :]
    fg = np.argwhere(table_region != TEAL)
    assert fg.size > 0, "reference table must contain glyphs"
    first_fg_r = int(fg[:, 0].min()) + sep1 + 1
    first_fg_c = int(fg[:, 1].min())
    table_r0 = first_fg_r - 1
    table_c0 = first_fg_c - 1

    dim = (sep2 - table_r0) // CELL
    assert dim >= 2 and table_r0 + dim * CELL == sep2, "unexpected table geometry"

    return {
        "red_row0": red_row0,
        "sep1": sep1,
        "sep2": sep2,
        "table_r0": table_r0,
        "table_c0": table_c0,
        "dim": dim,
    }


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(sorted(coords), dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _apply_movement(lower: np.ndarray, verdict: str | None) -> np.ndarray:
    if verdict is None:
        return lower.copy()

    h, w = lower.shape
    comps = _find_components(lower, BLACK)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    moving_colors = [color for color, items in by_color.items() if len(items) > 1]
    assert len(moving_colors) == 1, "expected exactly one repeated color in lower region"
    moving_color = int(moving_colors[0])
    moving = sorted(by_color[moving_color], key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    out = lower.copy()
    for comp in moving:
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1]
        out[rr, cc] = BLACK

    for comp in moving:
        r0, c0, r1, c1 = comp["bbox"]
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        assert int(comp["area"]) == hh * ww, "moving object must be a solid rectangle"

        if verdict == "row":
            if c0 == 0:
                dr, dc = 0, ww
            elif c1 == w - 1:
                dr, dc = 0, -ww
            else:
                raise AssertionError("row verdict requires moving rectangles to touch a vertical border")
        else:
            if r0 == 0:
                dr, dc = hh, 0
            elif r1 == h - 1:
                dr, dc = -hh, 0
            else:
                raise AssertionError("column verdict requires moving rectangles to touch a horizontal border")

        nr0, nc0 = int(r0 + dr), int(c0 + dc)
        assert 0 <= nr0 and 0 <= nc0 and nr0 + hh <= h and nc0 + ww <= w
        out[nr0 : nr0 + hh, nc0 : nc0 + ww] = moving_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use query-to-table glyph alignment to decide whether repeated lower rectangles shift horizontally or vertically.

    Input structure:
    - A 2-row full-width red divider splits the grid into an upper glyph-lookup area and a lower rectangle area.
    - In the upper area, teal background surrounds a query row of 5x5 glyph cells, two full-width pink separator rows, and a 3x3 table of 5x5 glyph cells whose centered 3x3 colored shapes identify each cell.
    - In the lower area, black background contains solid monochrome boundary rectangles; exactly one color appears in multiple separate rectangles while all other foreground colors are singletons.

    Rule:
    - Match each query glyph to the unique table cell with the same color and the same 3x3 shape, and green-fill the matched table-cell background.
    - If the matched table cells form one complete table row, green-fill the query band and shift only the repeated-color lower rectangles horizontally inward by their own widths; if they form one complete table column, do the analogous vertical inward shift by their own heights; otherwise leave the lower area unchanged and fill the verdict band red.
    - Singleton lower rectangles never move, and repeated-color rectangles overwrite anything they land on after being erased from their original positions.

    Key insight:
    - The glyph lookup does not directly produce the final answer; it yields a row-vs-column verdict that acts as an implicit control parameter for a separate geometric transform in the lower region.
    """
    g = np.asarray(grid, dtype=int)
    layout = _locate_layout(g)
    red_row0 = layout["red_row0"]
    sep1 = layout["sep1"]
    sep2 = layout["sep2"]
    table_r0 = layout["table_r0"]
    table_c0 = layout["table_c0"]
    dim = layout["dim"]

    col_starts = [table_c0 + CELL * i for i in range(dim)]
    row_starts = [table_r0 + CELL * i for i in range(dim)]

    query_sigs: list[tuple[tuple[int, int, int], ...]] = []
    for c in col_starts:
        sig = _cell_signature(g[:CELL, c : c + CELL])
        assert sig is not None, "expected a query glyph in every query slot"
        query_sigs.append(sig)
    query_set = set(query_sigs)
    assert len(query_set) == dim, "query glyphs should be distinct"

    out = g.copy()
    matched: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g[r : r + CELL, c : c + CELL])
            if sig in query_set:
                matched.append((int(ri), int(ci)))
                cell = out[r : r + CELL, c : c + CELL]
                cell[cell == TEAL] = GREEN

    verdict: str | None
    if len(matched) == dim and len({r for r, _c in matched}) == 1:
        verdict = "row"
    elif len(matched) == dim and len({c for _r, c in matched}) == 1:
        verdict = "col"
    else:
        verdict = None

    if verdict is None:
        out[sep2 + 1 : red_row0, :] = RED
    else:
        query_band = out[:sep1, :]
        query_band[query_band == TEAL] = GREEN
        out[sep2 + 1 : red_row0, :] = GREEN

    lower_in = g[red_row0 + RED_BAND_H :, :]
    out[red_row0 + RED_BAND_H :, :] = _apply_movement(lower_in, verdict)
    return out


def _corner_pos(corner: str, h: int, w: int, lower_h: int, lower_w: int) -> tuple[int, int]:
    r0 = 0 if corner[0] == "T" else lower_h - h
    c0 = 0 if corner[1] == "L" else lower_w - w
    return int(r0), int(c0)


def _move_rect(rect: dict, lower_h: int, lower_w: int, verdict: str) -> tuple[int, int]:
    r0, c0 = int(rect["r0"]), int(rect["c0"])
    h, w = int(rect["h"]), int(rect["w"])
    if verdict == "row":
        return (r0, c0 + w) if c0 == 0 else (r0, c0 - w)
    return (r0 + h, c0) if r0 == 0 else (r0 - h, c0)


def _paint_rect(grid: np.ndarray, rect: dict, *, clear: bool = False) -> None:
    color = BLACK if clear else int(rect["color"])
    r0, c0 = int(rect["r0"]), int(rect["c0"])
    h, w = int(rect["h"]), int(rect["w"])
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _render_lower(lower_h: int, lower_w: int, objects: list[dict], verdict: str | None) -> np.ndarray:
    grid = np.full((lower_h, lower_w), BLACK, dtype=int)
    for rect in objects:
        _paint_rect(grid, rect)
    if verdict is None:
        return grid

    moving = [rect for rect in objects if rect["moving"]]
    for rect in moving:
        _paint_rect(grid, rect, clear=True)
    for rect in sorted(moving, key=lambda item: (item["r0"], item["c0"])):
        nr0, nc0 = _move_rect(rect, lower_h, lower_w, verdict)
        moved = dict(rect)
        moved["r0"] = nr0
        moved["c0"] = nc0
        _paint_rect(grid, moved)
    return grid


def _render_expected(meta: dict) -> np.ndarray:
    upper = meta["upper_input"].copy()
    out = upper.copy()

    for ri, ci in meta["matched_cells"]:
        r = TABLE_ROW0 + ri * CELL
        c = meta["left_margin"] + ci * CELL
        cell = out[r : r + CELL, c : c + CELL]
        cell[cell == TEAL] = GREEN

    query_band = out[:SEP1_ROW, :]
    query_band[query_band == TEAL] = GREEN
    out[SEP2_ROW + 1 : SEP2_ROW + 1 + VERDICT_H, :] = GREEN

    full = np.vstack(
        [
            out,
            np.full((RED_BAND_H, out.shape[1]), RED, dtype=int),
            _render_lower(meta["lower_h"], out.shape[1], meta["objects"], meta["branch"]),
        ]
    )
    return full


def _rects_overlap(a: dict, b: dict) -> bool:
    return (
        int(a["r0"]) < int(b["r0"]) + int(b["h"])
        and int(b["r0"]) < int(a["r0"]) + int(a["h"])
        and int(a["c0"]) < int(b["c0"]) + int(b["w"])
        and int(b["c0"]) < int(a["c0"]) + int(a["w"])
    )


def _sample_upper(width: int) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    left_margin = width - TABLE_DIM * CELL
    left_margin = random.randint(0, left_margin)
    right_margin = width - TABLE_DIM * CELL - left_margin
    assert right_margin >= 0

    grid = np.full((SEP2_ROW + 1 + VERDICT_H, width), TEAL, dtype=int)
    grid[SEP1_ROW, :] = PINK
    grid[SEP2_ROW, :] = PINK

    shapes = random.sample(SHAPE_POOL, TABLE_DIM * TABLE_DIM)
    colors = [random.choice(GLYPH_COLORS) for _ in range(TABLE_DIM * TABLE_DIM)]
    glyphs: dict[tuple[int, int], np.ndarray] = {}
    for idx, (mask, color) in enumerate(zip(shapes, colors, strict=True)):
        ri, ci = divmod(idx, TABLE_DIM)
        glyphs[(ri, ci)] = _render_glyph(mask, color)

    branch = random.choice(("row", "col"))
    line_index = random.randrange(TABLE_DIM)
    matched = (
        [(line_index, ci) for ci in range(TABLE_DIM)]
        if branch == "row"
        else [(ri, line_index) for ri in range(TABLE_DIM)]
    )

    for (ri, ci), glyph in glyphs.items():
        r0 = TABLE_ROW0 + ri * CELL
        c0 = left_margin + ci * CELL
        grid[r0 : r0 + CELL, c0 : c0 + CELL] = glyph

    query_slots = list(range(TABLE_DIM))
    random.shuffle(query_slots)
    for slot, cell_pos in zip(query_slots, matched, strict=True):
        c0 = left_margin + slot * CELL
        grid[:CELL, c0 : c0 + CELL] = glyphs[cell_pos]

    return grid, matched, left_margin, branch


def _sample_repeated_sizes(repeated_corners: list[str], lower_h: int, lower_w: int) -> dict[str, tuple[int, int]] | None:
    sizes: dict[str, tuple[int, int]] = {}
    for corner in repeated_corners:
        h = random.randint(1, max(1, lower_h // 2))
        w = random.randint(1, min(3, lower_w // 2))
        sizes[corner] = (h, w)

    if all(h == w for h, w in sizes.values()):
        corner = random.choice(repeated_corners)
        h, w = sizes[corner]
        if h < max(1, lower_h // 2):
            sizes[corner] = (h + 1, w)
        elif w < min(3, lower_w // 2):
            sizes[corner] = (h, w + 1)
        elif h > 1:
            sizes[corner] = (h - 1, w)
        else:
            return None
    return sizes


def _build_objects(lower_h: int, lower_w: int, branch: str) -> list[dict] | None:
    corners = ["TL", "TR", "BL", "BR"]
    repeated_count = random.choice((2, 3))
    repeated_corners = random.sample(corners, repeated_count)
    sizes = _sample_repeated_sizes(repeated_corners, lower_h, lower_w)
    if sizes is None:
        return None

    colors = list(RECT_COLORS)
    random.shuffle(colors)
    moving_color = colors.pop()

    objects: list[dict] = []
    for corner in repeated_corners:
        h, w = sizes[corner]
        r0, c0 = _corner_pos(corner, h, w, lower_h, lower_w)
        objects.append(
            {
                "kind": "corner",
                "corner": corner,
                "moving": True,
                "color": moving_color,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
            }
        )

    unused = [corner for corner in corners if corner not in repeated_corners]
    random.shuffle(unused)
    singleton_corner_count = random.randint(1, len(unused))
    for corner in unused[:singleton_corner_count]:
        h = random.randint(1, max(1, lower_h // 2))
        w = random.randint(1, min(3, lower_w // 2))
        color = colors.pop()
        r0, c0 = _corner_pos(corner, h, w, lower_h, lower_w)
        objects.append(
            {
                "kind": "corner",
                "corner": corner,
                "moving": False,
                "color": color,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
            }
        )

    if colors and random.random() < 0.6:
        edge = random.choice(("top", "bottom", "left", "right"))
        if edge in ("top", "bottom") and lower_w >= 5:
            h = 1
            w = random.randint(1, 3)
            c0 = random.randint(1, lower_w - w - 1)
            r0 = 0 if edge == "top" else lower_h - h
        elif edge in ("left", "right") and lower_h >= 4:
            h = 1
            w = 1
            r0 = random.randint(1, lower_h - h - 1)
            c0 = 0 if edge == "left" else lower_w - w
        else:
            h = w = 0
            r0 = c0 = 0
        if h > 0:
            objects.append(
                {
                    "kind": "edge",
                    "corner": None,
                    "moving": False,
                    "color": colors.pop(),
                    "r0": r0,
                    "c0": c0,
                    "h": h,
                    "w": w,
                }
            )

    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if _rects_overlap(a, b):
                return None

    lower_input = _render_lower(lower_h, lower_w, objects, None)
    comps = _find_components(lower_input, BLACK)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)
    if by_color.get(moving_color) is None or len(by_color[moving_color]) != repeated_count:
        return None
    if any(color != moving_color and len(items) != 1 for color, items in by_color.items()):
        return None

    chosen = _render_lower(lower_h, lower_w, objects, branch)
    alternate = _render_lower(lower_h, lower_w, objects, "col" if branch == "row" else "row")
    all_moved = _render_lower(
        lower_h,
        lower_w,
        [{**rect, "moving": True} for rect in objects],
        branch,
    )
    if np.array_equal(chosen, alternate) or np.array_equal(chosen, all_moved):
        return None
    return objects


def _sample_case() -> tuple[np.ndarray, dict] | None:
    width = random.randint(TABLE_DIM * CELL + 1, TABLE_DIM * CELL + 4)
    lower_h = random.randint(4, 5)
    total_h = SEP2_ROW + 1 + VERDICT_H + RED_BAND_H + lower_h
    if total_h > 30:
        return None

    upper_input, matched_cells, left_margin, branch = _sample_upper(width)
    objects = _build_objects(lower_h, width, branch)
    if objects is None:
        return None

    red_band = np.full((RED_BAND_H, width), RED, dtype=int)
    lower_input = _render_lower(lower_h, width, objects, None)
    full_input = np.vstack([upper_input, red_band, lower_input])

    meta = {
        "upper_input": upper_input,
        "matched_cells": matched_cells,
        "left_margin": left_margin,
        "branch": branch,
        "lower_h": lower_h,
        "objects": objects,
    }
    expected = _render_expected(meta)
    if np.array_equal(full_input, expected):
        return None
    if not np.array_equal(main(full_input), expected):
        return None
    if not _semantic_checks(full_input, meta):
        return None
    return full_input, meta


def _semantic_checks(inp: np.ndarray, meta: dict) -> bool:
    out = main(inp)
    expected = _render_expected(meta)
    if not np.array_equal(out, expected):
        return False

    lower_in = inp[-meta["lower_h"] :, :]
    chosen_lower = _render_lower(meta["lower_h"], inp.shape[1], meta["objects"], meta["branch"])
    other_branch = "col" if meta["branch"] == "row" else "row"
    other_lower = _render_lower(meta["lower_h"], inp.shape[1], meta["objects"], other_branch)
    move_all_lower = _render_lower(
        meta["lower_h"],
        inp.shape[1],
        [{**rect, "moving": True} for rect in meta["objects"]],
        meta["branch"],
    )
    if np.array_equal(lower_in, chosen_lower):
        return False
    if np.array_equal(chosen_lower, other_lower):
        return False
    if np.array_equal(chosen_lower, move_all_lower):
        return False

    repeated = [rect for rect in meta["objects"] if rect["moving"]]
    if len(repeated) < 2:
        return False
    if not all(rect["corner"] in {"TL", "TR", "BL", "BR"} for rect in repeated):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        sample = _sample_case()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input for puzzle 9")
