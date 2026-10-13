from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
TAIL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
START_QUADRANT = {"U": 0, "R": 1, "D": 2, "L": 3}
BODY_HEIGHTS = (4, 6)
BODY_WIDTHS = (4, 6, 8)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, colors: tuple[int, ...] | list[int]) -> list[dict]:
    comps: list[dict] = []
    for color in colors:
        for coords in _components_mask(grid == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_full_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _quadrant_slices(box: tuple[int, int, int, int]) -> list[tuple[slice, slice]]:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    r_mid = r0 + h // 2
    c_mid = c0 + w // 2
    return [
        (slice(r0, r_mid), slice(c0, c_mid)),
        (slice(r0, r_mid), slice(c_mid, c1 + 1)),
        (slice(r_mid, r1 + 1), slice(c_mid, c1 + 1)),
        (slice(r_mid, r1 + 1), slice(c0, c_mid)),
    ]


def _decode_tail(
    grid: np.ndarray,
    rect_box: tuple[int, int, int, int],
    tail_coords: np.ndarray,
) -> tuple[str, int, int]:
    tail_cells = {tuple(map(int, p)) for p in tail_coords}
    assert tail_cells
    r0, c0, r1, c1 = rect_box
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    touching = []
    for tr, tc in tail_cells:
        for nr, nc in _neighbors4(tr, tc):
            if r0 <= nr <= r1 and c0 <= nc <= c1 and int(grid[nr, nc]) == GREY:
                touching.append(((tr, tc), (nr, nc)))
    assert len(touching) == 1, "tail must touch its rectangle at exactly one side cell"
    (tr, tc), (br, bc) = touching[0]
    delta = (tr - br, tc - bc)
    direction = None
    for name, vec in DIRS.items():
        if vec == delta:
            direction = name
            break
    assert direction is not None, "tail must extend orthogonally away from rectangle"

    length = len(tail_cells)
    assert 1 <= length <= 4
    dr, dc = DIRS[direction]
    expected = {(tr + k * dr, tc + k * dc) for k in range(length)}
    assert tail_cells == expected, "tail must be a contiguous straight segment"

    mid_rows = {r0 + h // 2 - 1, r0 + h // 2}
    mid_cols = {c0 + w // 2 - 1, c0 + w // 2}
    if direction == "U":
        assert br == r0 and bc in mid_cols
    elif direction == "D":
        assert br == r1 and bc in mid_cols
    elif direction == "L":
        assert bc == c0 and br in mid_rows
    else:
        assert bc == c1 and br in mid_rows

    return direction, length, int(grid[tr, tc])


def _paint_rect(
    out: np.ndarray,
    rect_box: tuple[int, int, int, int],
    direction: str,
    length: int,
    color: int,
) -> None:
    quadrants = _quadrant_slices(rect_box)
    start = START_QUADRANT[direction]
    selected = {(start + k) % 4 for k in range(length)}
    for idx, (rs, cs) in enumerate(quadrants):
        out[rs, cs] = int(color) if idx in selected else GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangle by using its tail direction as the starting quadrant and its tail length as the clockwise quadrant count.

    Input structure:
    - The background is black.
    - The foreground contains 3-4 solid gray even-sized rectangles.
    - Each rectangle has exactly one single-color, one-pixel-wide straight tail attached at a side midpoint and extending outward in one cardinal direction.

    Rule:
    - Split every gray rectangle into four equal quadrants: TL, TR, BR, BL.
    - Map tail direction to the starting quadrant: up -> TL, right -> TR, down -> BR, left -> BL.
    - Let L be the tail length; color L consecutive quadrants clockwise from that start with the tail color, leaving the remaining quadrants gray.
    - Erase all tail pixels to black.

    Key insight:
    - The same physical tail encodes two independent signals at once: its direction chooses where the quadrant cycle begins, and its length chooses how many clockwise quadrants are recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    rects = _find_monochrome_components(g, [GREY])
    assert rects, "expected gray rectangles"
    rects.sort(key=lambda comp: comp["bbox"])

    rect_index = -np.ones(g.shape, dtype=int)
    for idx, comp in enumerate(rects):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h % 2 == 0 and w % 2 == 0
        assert _is_full_rectangle(g, comp["bbox"], GREY), "gray component must be a solid rectangle"
        rect_index[r0 : r1 + 1, c0 : c1 + 1] = idx

    colors = sorted(int(v) for v in np.unique(g) if int(v) not in {BLACK, GREY})
    tails = _find_monochrome_components(g, colors)
    assert tails, "expected colored tails"

    out = g.copy()
    seen_colors: set[int] = set()
    rect_info: dict[int, tuple[str, int, int, np.ndarray]] = {}

    for tail in tails:
        coords = tail["coords"]
        color = int(tail["color"])
        assert color not in seen_colors, "tail colors must be unique"
        seen_colors.add(color)

        touching_rects = set()
        for tr, tc in coords:
            for nr, nc in _neighbors4(int(tr), int(tc)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and rect_index[nr, nc] >= 0:
                    touching_rects.add(int(rect_index[nr, nc]))
        assert len(touching_rects) == 1, "each tail must belong to exactly one rectangle"
        idx = next(iter(touching_rects))
        assert idx not in rect_info, "each rectangle must have exactly one tail"

        direction, length, decoded_color = _decode_tail(g, rects[idx]["bbox"], coords)
        assert decoded_color == color
        rect_info[idx] = (direction, length, color, coords)
        out[coords[:, 0], coords[:, 1]] = BLACK

    assert len(rect_info) == len(rects), "every rectangle must have one tail"

    for idx, comp in enumerate(rects):
        direction, length, color, _coords = rect_info[idx]
        _paint_rect(out, comp["bbox"], direction, length, color)

    return out


def _bbox_gap_at_least(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _tail_cells(
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: str,
    length: int,
    mid_pick: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if direction in {"U", "D"}:
        attach_c = c0 + w // 2 - 1 + mid_pick
        start = (r0 - 1, attach_c) if direction == "U" else (r1 + 1, attach_c)
    else:
        attach_r = r0 + h // 2 - 1 + mid_pick
        start = (attach_r, c0 - 1) if direction == "L" else (attach_r, c1 + 1)
    dr, dc = DIRS[direction]
    cells = [(start[0] + k * dr, start[1] + k * dc) for k in range(length)]
    return cells, start


def _placement_from_spec(
    spec: dict,
    r0: int,
    c0: int,
    shape: tuple[int, int],
) -> dict | None:
    h, w = int(spec["h"]), int(spec["w"])
    direction = str(spec["direction"])
    length = int(spec["length"])
    color = int(spec["color"])
    mid_pick = int(spec["mid_pick"])
    grid_h, grid_w = shape

    body_cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    tail_list, _start = _tail_cells(r0, c0, h, w, direction, length, mid_pick)
    full_cells = set(body_cells)
    for r, c in tail_list:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return None
        full_cells.add((r, c))
    if len(full_cells) != len(body_cells) + len(tail_list):
        return None

    body_box = (r0, c0, r0 + h - 1, c0 + w - 1)
    return {
        "body_box": body_box,
        "body_cells": body_cells,
        "tail_cells": tuple(tail_list),
        "full_cells": full_cells,
        "direction": direction,
        "length": length,
        "color": color,
    }


def _build_specs(count: int) -> list[dict]:
    colors = random.sample(TAIL_COLORS, count)
    anchor_dir = random.choice(tuple(DIRS))
    anchor_len = random.choice((1, 2, 3))
    other_dir = random.choice([d for d in DIRS if d != anchor_dir])
    other_len = random.choice([l for l in (1, 2, 3, 4) if l != anchor_len])

    specs = [
        {"direction": anchor_dir, "length": anchor_len, "color": colors[0]},
        {"direction": other_dir, "length": anchor_len, "color": colors[1]},
        {"direction": anchor_dir, "length": other_len, "color": colors[2]},
    ]
    if count == 4:
        if other_len != 4 and random.random() < 0.65:
            len4 = 4
        else:
            len4 = random.choice((1, 2, 3, 4))
        specs.append(
            {
                "direction": random.choice(tuple(DIRS)),
                "length": len4,
                "color": colors[3],
            }
        )

    for spec in specs:
        spec["h"] = random.choice(BODY_HEIGHTS)
        spec["w"] = random.choice(BODY_WIDTHS)
        spec["mid_pick"] = random.randint(0, 1)

    random.shuffle(specs)
    return specs


def _search_layout(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    ordered = sorted(specs, key=lambda s: s["h"] * s["w"] + s["length"], reverse=True)

    def rec(i: int, margin_cells: set[tuple[int, int]], body_boxes: list[tuple[int, int, int, int]], placed: list[dict]):
        if i == len(ordered):
            return placed

        spec = ordered[i]
        candidates: list[dict] = []
        for r0 in range(shape[0] - spec["h"] + 1):
            for c0 in range(shape[1] - spec["w"] + 1):
                placement = _placement_from_spec(spec, r0, c0, shape)
                if placement is None:
                    continue
                if any(cell in margin_cells for cell in placement["full_cells"]):
                    continue
                if any(
                    not _bbox_gap_at_least(placement["body_box"], old_box, gap=2)
                    for old_box in body_boxes
                ):
                    continue
                candidates.append(placement)

        if not candidates:
            return None
        random.shuffle(candidates)
        for placement in candidates[:160]:
            next_margin = margin_cells | _expand_cells(placement["full_cells"], shape, radius=1)
            result = rec(i + 1, next_margin, body_boxes + [placement["body_box"]], placed + [placement])
            if result is not None:
                return result
        return None

    return rec(0, set(), [], [])


def _render_input(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for placement in placements:
        r0, c0, r1, c1 = placement["body_box"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        for r, c in placement["tail_cells"]:
            grid[r, c] = int(placement["color"])
    return grid


def _object_pattern(direction: str, length: int) -> tuple[int, ...]:
    start = START_QUADRANT[direction]
    return tuple(sorted((start + k) % 4 for k in range(length)))


def _semantic_checks(specs: list[dict], placements: list[dict], inp: np.ndarray) -> bool:
    if len(placements) != len(specs):
        return False

    dirs = [p["direction"] for p in placements]
    lens = [p["length"] for p in placements]
    if len(set(dirs)) < 2 or len(set(lens)) < 2:
        return False

    same_len_diff_dir = False
    same_dir_diff_len = False
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            pi = placements[i]
            pj = placements[j]
            if pi["length"] == pj["length"] and pi["direction"] != pj["direction"]:
                if _object_pattern(pi["direction"], pi["length"]) != _object_pattern(pj["direction"], pj["length"]):
                    same_len_diff_dir = True
            if pi["direction"] == pj["direction"] and pi["length"] != pj["length"]:
                if _object_pattern(pi["direction"], pi["length"]) != _object_pattern(pj["direction"], pj["length"]):
                    same_dir_diff_len = True
    if not same_len_diff_dir or not same_dir_diff_len:
        return False

    if _get_bg(inp) != BLACK:
        return False
    if Counter(int(v) for v in np.unique(inp) if int(v) not in {BLACK, GREY}).total() != len(placements):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    count = random.choice((3, 4))
    min_side = 16 if count == 3 else 18
    shape = (random.randint(min_side, 24), random.randint(min_side, 24))
    specs = _build_specs(count)
    placements = _search_layout(shape, specs)
    if placements is None:
        return None

    inp = _render_input(shape, placements)
    if not _semantic_checks(specs, placements, inp):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    # Oracle from placement metadata: tails erased, quadrants recolored per object.
    expected = np.full(shape, BLACK, dtype=int)
    for placement in placements:
        r0, c0, r1, c1 = placement["body_box"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for placement in placements:
        _paint_rect(
            expected,
            placement["body_box"],
            placement["direction"],
            placement["length"],
            placement["color"],
        )
    if not np.array_equal(out, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
