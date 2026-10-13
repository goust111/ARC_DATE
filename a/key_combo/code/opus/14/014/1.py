from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
MASK = 6
ORANGE = 7
TEAL = 8

RIGHT_CARRIER = BLUE
LEFT_CARRIER = TEAL
OBJECT_COLORS = (GREEN, ORANGE)

TEMPLATE_DATA = (
    ("lr", ("111", "010")),
    ("lr", ("101", "111")),
    ("lr", ("111", "010", "010")),
    ("lr", ("1111", "0110")),
    ("lr", ("101", "111", "010")),
    ("rot", ("110", "011")),
    ("rot", ("110", "010", "011")),
    ("rot", ("1100", "0110", "0011")),
    ("rot", ("1110", "0111")),
)


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _partner(r: int, c: int, height: int, width: int, sym_type: str) -> tuple[int, int]:
    if sym_type == "lr":
        return r, width - 1 - c
    if sym_type == "rot":
        return height - 1 - r, width - 1 - c
    raise ValueError(f"unknown symmetry type: {sym_type}")


def _symmetry_candidates(local: np.ndarray, carrier: int) -> list[str]:
    occupied_box = _bbox(local != carrier)
    assert occupied_box is not None, "lane must contain an embedded object"
    r0, c0, r1, c1 = occupied_box
    obj = local[r0 : r1 + 1, c0 : c1 + 1]
    occupied = obj != carrier
    solids = (obj != carrier) & (obj != MASK)
    masks = obj == MASK
    candidates: list[str] = []
    height, width = obj.shape
    for sym_type in ("lr", "rot"):
        ok = True
        for r, c in np.argwhere(occupied):
            pr, pc = _partner(int(r), int(c), height, width, sym_type)
            if not occupied[pr, pc]:
                ok = False
                break
        if not ok:
            continue
        for r, c in np.argwhere(masks):
            pr, pc = _partner(int(r), int(c), height, width, sym_type)
            if (pr, pc) == (int(r), int(c)) or not solids[pr, pc]:
                ok = False
                break
        if ok:
            candidates.append(sym_type)
    return candidates


def _repair_local(local: np.ndarray, carrier: int) -> tuple[np.ndarray, str, int]:
    colors = {int(v) for v in np.unique(local) if int(v) not in {carrier, MASK}}
    assert len(colors) == 1, "embedded object must use exactly one non-carrier solid color"
    obj_color = next(iter(colors))

    occupied_box = _bbox(local != carrier)
    assert occupied_box is not None
    r0, c0, r1, c1 = occupied_box
    obj = local[r0 : r1 + 1, c0 : c1 + 1].copy()

    candidates = _symmetry_candidates(local, carrier)
    assert len(candidates) == 1, f"expected exactly one symmetry type, got {candidates}"
    sym_type = candidates[0]

    height, width = obj.shape
    for r, c in np.argwhere(obj == MASK):
        pr, pc = _partner(int(r), int(c), height, width, sym_type)
        assert obj[pr, pc] != MASK and obj[pr, pc] != carrier
        obj[r, c] = int(obj[pr, pc])

    repaired = local.copy()
    repaired[r0 : r1 + 1, c0 : c1 + 1] = obj
    assert not np.any(repaired == MASK), "all masks inside a lane must be repaired"
    return repaired, sym_type, obj_color


def _slide_local(local: np.ndarray, carrier: int) -> np.ndarray:
    repaired, _sym_type, _obj_color = _repair_local(local, carrier)
    obj_coords = np.argwhere(repaired != carrier)
    assert obj_coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(obj_coords)
    if carrier == RIGHT_CARRIER:
        delta = local.shape[1] - 1 - c1
    else:
        assert carrier == LEFT_CARRIER
        delta = -c0

    moved = obj_coords.copy()
    moved[:, 1] += delta
    assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < local.shape[1])), "slide must stay inside the carrier"
    out = np.full(local.shape, carrier, dtype=int)
    out[moved[:, 0], moved[:, 1]] = repaired[obj_coords[:, 0], obj_coords[:, 1]]
    return out


def _extract_lanes(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    lanes: list[tuple[tuple[int, int, int, int], int]] = []
    for coords in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        local = grid[r0 : r1 + 1, c0 : c1 + 1]
        carrier_colors = {int(v) for v in np.unique(local) if int(v) in {RIGHT_CARRIER, LEFT_CARRIER}}
        assert len(carrier_colors) == 1, "each connected non-black component must have one carrier color"
        carrier = next(iter(carrier_colors))
        assert np.all(local != BLACK), "carrier bbox must be a solid rectangle with an embedded object"
        lanes.append(((r0, c0, r1, c1), carrier))
    assert 2 <= len(lanes) <= 3, "expected 2-3 carrier lanes"
    lanes.sort(key=lambda item: (item[0][0], item[0][1]))
    return lanes


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    out = g.copy()
    for (r0, c0, r1, c1), carrier in _extract_lanes(g):
        out[r0 : r1 + 1, c0 : c1 + 1] = _slide_local(g[r0 : r1 + 1, c0 : c1 + 1], carrier)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each masked symmetric shape inside a colored carrier, then slide the repaired shape to the carrier edge.

    Input structure:
    - The background is black.
    - The grid contains 2-3 separated horizontal carrier rectangles in blue or teal.
    - Each carrier contains exactly one 4-connected green or orange shape, and 1-2 of that shape's cells are replaced by magenta masks.

    Rule:
    - For each carrier independently, use the visible local symmetry of the embedded shape to decide whether the full shape is left-right reflective or 180-degree rotationally symmetric.
    - Replace every magenta cell with the color of its unique symmetry partner, repairing the shape before any motion.
    - In blue carriers, slide the repaired shape right until its repaired bounding box is flush with the carrier's right edge; in teal carriers, slide it left until flush with the carrier's left edge.
    - Preserve the shape and rows during the slide, and restore vacated cells to the carrier color.

    Key insight:
    - The magenta cells hide real shape pixels, so repairing them first is necessary to recover the true horizontal extent that determines the final slide distance.
    """
    return _solve(grid)


def _visible_slide_delta(local: np.ndarray, carrier: int) -> int:
    visible_coords = np.argwhere((local != carrier) & (local != MASK))
    assert visible_coords.size > 0
    _r0, c0, _r1, c1 = _bbox_from_coords(visible_coords)
    if carrier == RIGHT_CARRIER:
        return local.shape[1] - 1 - c1
    return -c0


def _render_template(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, 0).astype(int)


def _mask_groups(mask: np.ndarray, sym_type: str) -> list[list[tuple[int, int]]]:
    coords = np.argwhere(mask)
    box = _bbox(mask)
    assert box is not None
    _r0, c0, _r1, c1 = box
    groups: list[list[tuple[int, int]]] = []
    for edge_col in (c0, c1):
        cells = [(int(r), int(c)) for r, c in coords if int(c) == edge_col]
        if not (1 <= len(cells) <= 2):
            continue
        partner_cells = {_partner(r, c, mask.shape[0], mask.shape[1], sym_type) for r, c in cells}
        if any(p in cells for p in partner_cells):
            continue
        if all(mask[pr, pc] for pr, pc in partner_cells):
            groups.append(cells)
    return groups


TEMPLATES: list[dict[str, object]] = []
for _sym_type, _rows in TEMPLATE_DATA:
    _mask = _mask_from_rows(_rows)
    _groups = _mask_groups(_mask, _sym_type)
    assert _groups, f"template {_rows} must expose at least one extreme-column mask group"
    TEMPLATES.append({"sym": _sym_type, "mask": _mask, "groups": _groups})


def _lane_info(template: dict[str, object], carrier: int, rng: random.Random) -> dict[str, object]:
    shape_mask = np.array(template["mask"], dtype=bool)
    groups = list(template["groups"])
    obj_h, obj_w = shape_mask.shape
    lane_h = rng.randint(max(3, obj_h), max(5, obj_h))
    lane_w = rng.randint(max(8, obj_w + 2), 12)
    row0 = rng.randint(0, lane_h - obj_h)
    gap = rng.randint(1, lane_w - obj_w - 1)
    if carrier == RIGHT_CARRIER:
        obj_c0 = lane_w - obj_w - gap
    else:
        obj_c0 = gap
    return {
        "sym": template["sym"],
        "mask": shape_mask,
        "mask_cells": list(rng.choice(groups)),
        "lane_h": lane_h,
        "lane_w": lane_w,
        "obj_r0": row0,
        "obj_c0": obj_c0,
        "carrier": carrier,
        "obj_color": rng.choice(OBJECT_COLORS),
    }


def _render_lane_input(info: dict[str, object]) -> np.ndarray:
    carrier = int(info["carrier"])
    obj_color = int(info["obj_color"])
    lane = np.full((int(info["lane_h"]), int(info["lane_w"])), carrier, dtype=int)
    shape = _render_template(np.array(info["mask"], dtype=bool), obj_color)
    r0 = int(info["obj_r0"])
    c0 = int(info["obj_c0"])
    h, w = shape.shape
    patch = lane[r0 : r0 + h, c0 : c0 + w]
    patch[shape != 0] = shape[shape != 0]
    for rr, cc in info["mask_cells"]:
        patch[int(rr), int(cc)] = MASK
    return lane


def _lane_output_from_info(info: dict[str, object]) -> np.ndarray:
    return _slide_local(_render_lane_input(info), int(info["carrier"]))


def _assemble_grid(lanes: list[dict[str, object]], rng: random.Random) -> np.ndarray:
    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    gaps = [rng.randint(1, 3) for _ in range(len(lanes) - 1)]
    height = top_margin + bottom_margin + sum(int(info["lane_h"]) for info in lanes) + sum(gaps)
    width = max(int(info["lane_w"]) + rng.randint(0, 4) for info in lanes)
    width = max(width, 12)

    grid = np.zeros((height, width), dtype=int)
    row = top_margin
    for idx, info in enumerate(lanes):
        lane_h = int(info["lane_h"])
        lane_w = int(info["lane_w"])
        col = rng.randint(0, width - lane_w)
        lane = _render_lane_input(info)
        grid[row : row + lane_h, col : col + lane_w] = lane
        info["grid_r0"] = row
        info["grid_c0"] = col
        row += lane_h
        if idx < len(gaps):
            row += gaps[idx]
    return grid


def _validate_lane_semantics(info: dict[str, object]) -> None:
    lane = _render_lane_input(info)
    carrier = int(info["carrier"])
    candidates = _symmetry_candidates(lane, carrier)
    assert candidates == [info["sym"]], f"lane should expose one symmetry, got {candidates}"

    full_box = _bbox(lane != carrier)
    solid_box = _bbox((lane != carrier) & (lane != MASK))
    assert full_box is not None and solid_box is not None
    assert full_box != solid_box, "masking must change the visible solid bbox"
    assert _visible_slide_delta(lane, carrier) != _slide_local_delta(lane, carrier), "repair must affect the slide distance"

    out = _slide_local(lane, carrier)
    moved_colors = {int(v) for v in np.unique(out) if int(v) not in {carrier}}
    assert moved_colors == {int(info["obj_color"])}


def _slide_local_delta(local: np.ndarray, carrier: int) -> int:
    repaired, _sym, _color = _repair_local(local, carrier)
    coords = np.argwhere(repaired != carrier)
    _r0, c0, _r1, c1 = _bbox_from_coords(coords)
    if carrier == RIGHT_CARRIER:
        return local.shape[1] - 1 - c1
    return -c0


def _validate_grid(grid: np.ndarray, lanes: list[dict[str, object]]) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    extracted = _extract_lanes(grid)
    assert len(extracted) == len(lanes)

    colors = Counter(int(v) for v in grid.ravel())
    assert colors[BLACK] > 0
    assert colors[BLUE] + colors[TEAL] > colors[MASK]
    for info in lanes:
        _validate_lane_semantics(info)

    out = _solve(grid)
    assert out.shape == grid.shape
    assert not np.any(out == MASK)
    assert any(
        np.any(
            out[
                int(info["grid_r0"]) : int(info["grid_r0"]) + int(info["lane_h"]),
                int(info["grid_c0"]) : int(info["grid_c0"]) + int(info["lane_w"]),
            ]
            != grid[
                int(info["grid_r0"]) : int(info["grid_r0"]) + int(info["lane_h"]),
                int(info["grid_c0"]) : int(info["grid_c0"]) + int(info["lane_w"]),
            ]
        )
        for info in lanes
    ), "at least one lane must move"

    expected = np.zeros_like(grid)
    for info in lanes:
        r0 = int(info["grid_r0"])
        c0 = int(info["grid_c0"])
        lane_h = int(info["lane_h"])
        lane_w = int(info["lane_w"])
        expected[r0 : r0 + lane_h, c0 : c0 + lane_w] = _lane_output_from_info(info)
    assert np.array_equal(out, expected), "assembled output must match per-lane construction"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        lane_count = rng.choice((2, 3))
        carrier_seq = [RIGHT_CARRIER, LEFT_CARRIER]
        if lane_count == 3:
            carrier_seq.append(rng.choice((RIGHT_CARRIER, LEFT_CARRIER)))
        rng.shuffle(carrier_seq)

        sym_seq = ["lr", "rot"]
        if lane_count == 3:
            sym_seq.append(rng.choice(("lr", "rot")))
        rng.shuffle(sym_seq)

        lanes: list[dict[str, object]] = []
        for carrier, sym in zip(carrier_seq, sym_seq):
            choices = [tpl for tpl in TEMPLATES if tpl["sym"] == sym]
            info = _lane_info(rng.choice(choices), carrier, rng)
            lanes.append(info)

        grid = _assemble_grid(lanes, rng)
        try:
            _validate_grid(grid, lanes)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
