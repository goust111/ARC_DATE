from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
TEAL = 8

DIAG_NW_SE = "nw_se"
DIAG_NE_SW = "ne_sw"
CORNER_NW = "NW"
CORNER_NE = "NE"
CORNER_SW = "SW"
CORNER_SE = "SE"

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)

GRID_H_RANGE = (16, 23)  # high is exclusive
GRID_W_RANGE = (16, 23)
MOTIF_COUNT_OPTIONS = (2, 3)
BLOCK_SIZE_OPTIONS = (1, 2)
MARKER_DIST_MIN = 2
MARKER_DIST_MAX = 6
MAX_GENERATION_ATTEMPTS = 500
MAX_LOCAL_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _red_components8(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != RED:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == RED:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _stamp_top_left(corner: str, anchor: tuple[int, int], size: int) -> tuple[int, int]:
    r, c = anchor
    if corner == CORNER_NW:
        return r - size + 1, c - size + 1
    if corner == CORNER_NE:
        return r - size + 1, c
    if corner == CORNER_SW:
        return r, c - size + 1
    assert corner == CORNER_SE
    return r, c


def _stamp_cells(shape: tuple[int, int], corner: str, anchor: tuple[int, int], size: int) -> set[tuple[int, int]]:
    r0, c0 = _stamp_top_left(corner, anchor, size)
    cells = set()
    for r in range(r0, r0 + size):
        for c in range(c0, c0 + size):
            assert _in_bounds(shape, r, c)
            cells.add((r, c))
    return cells


def _safe_dirs(corner: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner == CORNER_NW:
        return DIR_DOWN, DIR_RIGHT
    if corner == CORNER_NE:
        return DIR_DOWN, DIR_LEFT
    if corner == CORNER_SW:
        return DIR_UP, DIR_RIGHT
    assert corner == CORNER_SE
    return DIR_UP, DIR_LEFT


def _motif_info_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h % 2 == 0 and h >= 2
    size = h // 2

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    occupied: set[tuple[int, int]] = set()
    for br in range(2):
        for bc in range(2):
            block = sub[br * size : (br + 1) * size, bc * size : (bc + 1) * size]
            red_mask = block == RED
            assert np.all(red_mask | (block == BG))
            if np.all(red_mask):
                occupied.add((br, bc))
            else:
                assert not np.any(red_mask)

    anchors = []
    if occupied == {(0, 0), (1, 1)}:
        anchors.append({"corner": CORNER_NE, "anchor": (r0 - 1, c1 + 1), "size": size})
        anchors.append({"corner": CORNER_SW, "anchor": (r1 + 1, c0 - 1), "size": size})
    else:
        assert occupied == {(0, 1), (1, 0)}
        anchors.append({"corner": CORNER_NW, "anchor": (r0 - 1, c0 - 1), "size": size})
        anchors.append({"corner": CORNER_SE, "anchor": (r1 + 1, c1 + 1), "size": size})

    for record in anchors:
        ar, ac = record["anchor"]
        assert _in_bounds(grid.shape, ar, ac)
    return {"bbox": (r0, c0, r1, c1), "size": size, "anchors": anchors}


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each blue clue to the complementary outer corner of its red diagonal motif, then stamp a teal square there.

    Input structure:
    - The grid is a black background containing 2-3 red diagonal motifs and several isolated blue clue pixels.
    - Each red motif occupies an even-sized square bounding box split into four equal square blocks, with exactly one filled red block on each diagonal corner.
    - Each blue pixel shares exactly one row or column with one complementary outer-corner position implied by a red motif.

    Rule:
    - For every red motif, detect which diagonal of its 2x2 block layout is filled and compute the two opposite outer-corner positions just outside the motif.
    - Move each blue clue along its shared row or column so it lands in the cell immediately adjacent to its unique target outer-corner position.
    - Erase the original blue locations and stamp a solid teal square of the motif's block size at each target outer corner, preserving the red motifs.

    Key insight:
    - The blue clues do not reveal their destinations directly; their targets must first be inferred from the missing complementary corners of the red diagonal motifs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _red_components8(g)
    assert 2 <= len(components) <= 3

    anchors = []
    for comp in components:
        info = _motif_info_from_component(g, comp)
        anchors.extend(info["anchors"])

    blue_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    assert len(blue_coords) == len(anchors)

    anchor_records = []
    for record in anchors:
        corner = str(record["corner"])
        anchor = tuple(record["anchor"])
        size = int(record["size"])
        stamp = _stamp_cells(g.shape, corner, anchor, size)
        anchor_records.append(
            {
                "corner": corner,
                "anchor": anchor,
                "size": size,
                "stamp": stamp,
            }
        )

    moves: list[tuple[tuple[int, int], tuple[int, int]]] = []
    planned_landings: set[tuple[int, int]] = set()
    for br, bc in blue_coords:
        aligned = []
        for record in anchor_records:
            ar, ac = record["anchor"]
            if (br == ar) ^ (bc == ac):
                aligned.append(record)
        assert len(aligned) == 1
        record = aligned[0]
        ar, ac = record["anchor"]

        if br == ar:
            step = -1 if bc < ac else 1
            landing = (ar, ac + step)
        else:
            step = -1 if br < ar else 1
            landing = (ar + step, ac)

        assert _in_bounds(g.shape, landing[0], landing[1])
        assert int(g[landing]) == BG
        assert landing not in record["stamp"]
        assert landing not in planned_landings
        planned_landings.add(landing)
        moves.append(((br, bc), landing))

    out = g.copy()
    for br, bc in blue_coords:
        out[br, bc] = BG

    for record in anchor_records:
        r0, c0 = _stamp_top_left(record["corner"], record["anchor"], record["size"])
        out[r0 : r0 + record["size"], c0 : c0 + record["size"]] = TEAL

    for _, landing in moves:
        out[landing] = BLUE
    return out


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _expand_cells(shape: tuple[int, int], cells: set[tuple[int, int]], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _build_motif_record(
    shape: tuple[int, int],
    r0: int,
    c0: int,
    size: int,
    diagonal: str,
) -> dict:
    bbox_size = 2 * size
    red_cells: set[tuple[int, int]] = set()
    if diagonal == DIAG_NW_SE:
        red_cells |= _rect_cells(r0, c0, size, size)
        red_cells |= _rect_cells(r0 + size, c0 + size, size, size)
        anchors = [
            {"corner": CORNER_NE, "anchor": (r0 - 1, c0 + bbox_size), "size": size},
            {"corner": CORNER_SW, "anchor": (r0 + bbox_size, c0 - 1), "size": size},
        ]
    else:
        red_cells |= _rect_cells(r0, c0 + size, size, size)
        red_cells |= _rect_cells(r0 + size, c0, size, size)
        anchors = [
            {"corner": CORNER_NW, "anchor": (r0 - 1, c0 - 1), "size": size},
            {"corner": CORNER_SE, "anchor": (r0 + bbox_size, c0 + bbox_size), "size": size},
        ]

    stamp_cells: set[tuple[int, int]] = set()
    for record in anchors:
        stamp_cells |= _stamp_cells(shape, record["corner"], record["anchor"], size)

    full_footprint = red_cells | stamp_cells
    pad_box = _rect_cells(r0 - size, c0 - size, 4 * size, 4 * size)
    return {
        "r0": r0,
        "c0": c0,
        "size": size,
        "diagonal": diagonal,
        "red_cells": red_cells,
        "stamp_cells": stamp_cells,
        "anchors": anchors,
        "footprint": full_footprint,
        "buffer": _expand_cells(shape, pad_box, 0),
    }


def _place_motifs(shape: tuple[int, int], motif_count: int) -> list[dict] | None:
    h, w = shape
    motifs: list[dict] = []
    used_buffer: set[tuple[int, int]] = set()
    anchor_rows: set[int] = set()
    anchor_cols: set[int] = set()

    specs = [
        {
            "size": random.choice(BLOCK_SIZE_OPTIONS),
            "diagonal": random.choice((DIAG_NW_SE, DIAG_NE_SW)),
        }
        for _ in range(motif_count)
    ]
    random.shuffle(specs)

    for spec in specs:
        size = int(spec["size"])
        diagonal = str(spec["diagonal"])
        if h < 4 * size or w < 4 * size:
            return None
        placed = None
        for _ in range(MAX_LOCAL_ATTEMPTS):
            r0 = random.randint(size, h - 3 * size)
            c0 = random.randint(size, w - 3 * size)
            record = _build_motif_record(shape, r0, c0, size, diagonal)
            rows = {int(a["anchor"][0]) for a in record["anchors"]}
            cols = {int(a["anchor"][1]) for a in record["anchors"]}
            if rows & anchor_rows or cols & anchor_cols:
                continue
            if record["buffer"] & used_buffer:
                continue
            placed = record
            break
        if placed is None:
            return None
        motifs.append(placed)
        used_buffer |= placed["buffer"]
        anchor_rows |= {int(a["anchor"][0]) for a in placed["anchors"]}
        anchor_cols |= {int(a["anchor"][1]) for a in placed["anchors"]}
    return motifs


def _marker_options(
    shape: tuple[int, int],
    anchor_records: list[dict],
    record: dict,
    occupied: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    ar, ac = map(int, record["anchor"])
    other_rows = {int(r["anchor"][0]) for r in anchor_records if r is not record}
    other_cols = {int(r["anchor"][1]) for r in anchor_records if r is not record}
    options = []

    for dr, dc in _safe_dirs(str(record["corner"])):
        if dr != 0:
            max_dist = ar if dr < 0 else (h - 1 - ar)
        else:
            max_dist = ac if dc < 0 else (w - 1 - ac)
        for dist in range(MARKER_DIST_MIN, min(MARKER_DIST_MAX, max_dist) + 1):
            mr = ar + dr * dist
            mc = ac + dc * dist
            marker = (mr, mc)
            if marker in occupied:
                continue
            if marker in blue_cells:
                continue
            if dr == 0 and mc in other_cols:
                continue
            if dc == 0 and mr in other_rows:
                continue

            clear = True
            for step in range(1, dist):
                cell = (ar + dr * step, ac + dc * step)
                if cell in occupied or cell in blue_cells:
                    clear = False
                    break
            if not clear:
                continue

            if _expand_cells(shape, {marker}, 1) & blue_cells:
                continue
            options.append(marker)
    random.shuffle(options)
    return options


def _render_input(shape: tuple[int, int], motifs: list[dict], markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for motif in motifs:
        for r, c in motif["red_cells"]:
            grid[r, c] = RED
    for r, c in markers:
        assert grid[r, c] == BG
        grid[r, c] = BLUE
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randrange(*GRID_H_RANGE),
        random.randrange(*GRID_W_RANGE),
    )
    motif_count = random.choice(MOTIF_COUNT_OPTIONS)
    motifs = _place_motifs(shape, motif_count)
    if motifs is None:
        return None

    anchor_records = [anchor for motif in motifs for anchor in motif["anchors"]]
    occupied = set().union(*(motif["red_cells"] for motif in motifs))
    markers: list[tuple[int, int]] = []
    blue_cells: set[tuple[int, int]] = set()

    for record in sorted(anchor_records, key=lambda a: (a["anchor"][0], a["anchor"][1])):
        options = _marker_options(shape, anchor_records, record, occupied, blue_cells)
        if not options:
            return None
        marker = random.choice(options)
        markers.append(marker)
        blue_cells.add(marker)

    grid = _render_input(shape, motifs, markers)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None

    if int(np.sum(out == TEAL)) != sum(2 * motif["size"] * motif["size"] for motif in motifs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
