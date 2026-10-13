from __future__ import annotations

import numpy as np


BG = 0
RED = 2
YELLOW = 4
FRAME_COLORS = (1, 3, 5, 6, 7, 8, 9)
OPEN_DIRS = ("U", "D", "L", "R")
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
GEN_RETRIES = 500


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _side_from_adjacent_red(red_cell: Coord, border_set: set[Coord]) -> str | None:
    r, c = red_cell
    sides = set()
    if (r + 1, c) in border_set:
        sides.add("U")
    if (r - 1, c) in border_set:
        sides.add("D")
    if (r, c + 1) in border_set:
        sides.add("L")
    if (r, c - 1) in border_set:
        sides.add("R")
    if len(sides) != 1:
        return None
    return next(iter(sides))


def _reflect_cell(cell: Coord, side: str, bbox: tuple[int, int, int, int]) -> Coord:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if side == "U":
        return 2 * r0 - 1 - r, c
    if side == "D":
        return 2 * r1 + 1 - r, c
    if side == "L":
        return r, 2 * c0 - 1 - c
    if side == "R":
        return r, 2 * c1 + 1 - c
    raise ValueError(f"unknown side {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each cup's yellow contents across the red-marked wall so the contents spill outside the frame.

    Input structure:
    - A black background contains 2-3 cup-like frame objects, each in a unique non-black,
      non-red, non-yellow color.
    - Every frame is a 1-cell-thick three-sided groove with one open side, and its cavity
      contains 1-3 yellow pixels.
    - One or two red pixels sit just outside one closed wall of each frame and touch that wall
      by 4-neighborhood; different frames are well separated.

    Rule:
    - For each frame color, find the red pixels that touch that frame and infer which side
      of the frame they mark.
    - Use the frame's outer bbox edge on that side as a mirror axis lying just outside the wall.
    - Reflect every yellow cavity pixel across that axis, draw the reflected yellow pixels in
      the outside background, erase the original cavity yellows, and erase the red markers.
    - Leave the frame color itself unchanged.

    Key insight:
    - The red markers do not move; they only choose which frame wall becomes the reflection axis,
      and that same reflection both ejects the contents outward and determines their final geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    frame_colors = sorted(int(color) for color in np.unique(g) if int(color) not in {BG, RED, YELLOW})
    red_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]

    for color in frame_colors:
        frame_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        assert frame_cells, "each frame color must appear"
        r0, c0, r1, c1 = _bbox(frame_cells)
        frame_bbox = (r0, c0, r1, c1)

        markers = []
        side_votes = []
        for red_cell in red_cells:
            side = _side_from_adjacent_red(red_cell, frame_cells)
            if side is None:
                continue
            markers.append(red_cell)
            side_votes.append(side)
        assert markers, "each frame must have adjacent red markers"
        assert len(set(side_votes)) == 1, "all markers for one frame must lie on the same side"
        marker_side = side_votes[0]

        content = [
            (r, c)
            for r, c in yellow_cells
            if r0 <= r <= r1 and c0 <= c <= c1 and (r, c) not in frame_cells
        ]
        assert content, "each frame must contain yellow pixels"

        for r, c in content:
            out[r, c] = BG
            rr, cc = _reflect_cell((r, c), marker_side, frame_bbox)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "reflected pixel left the grid"
            out[rr, cc] = YELLOW

        for r, c in markers:
            out[r, c] = BG

    return out


def _build_canonical_frame(
    inner_w: int,
    depth: int,
    style: str,
    leg_side: str | None,
    leg_len: int,
) -> tuple[set[Coord], set[Coord], tuple[int, int]]:
    frame = {(r, 0) for r in range(depth + 1)}
    frame |= {(r, inner_w + 1) for r in range(depth + 1)}
    frame |= {(depth, c) for c in range(inner_w + 2)}
    height = depth + 1
    if style == "l":
        assert leg_side in {"left", "right"} and leg_len >= 1
        leg_col = 0 if leg_side == "left" else inner_w + 1
        for r in range(depth + 1, depth + 1 + leg_len):
            frame.add((r, leg_col))
        height += leg_len
    cavity = {(r, c) for r in range(depth) for c in range(1, inner_w + 1)}
    return frame, cavity, (height, inner_w + 2)


def _transform_cells(cells: set[Coord], shape: tuple[int, int], open_dir: str) -> tuple[set[Coord], tuple[int, int]]:
    h, w = shape
    if open_dir == "U":
        return set(cells), (h, w)
    if open_dir == "D":
        return {(h - 1 - r, c) for r, c in cells}, (h, w)
    if open_dir == "L":
        return {(w - 1 - c, r) for r, c in cells}, (w, h)
    if open_dir == "R":
        return {(c, h - 1 - r) for r, c in cells}, (w, h)
    raise ValueError(f"unknown direction {open_dir}")


def _side_candidates(frame_cells: set[Coord], shape: tuple[int, int], side: str) -> list[Coord]:
    h, w = shape
    if side == "U":
        return sorted((r, c) for r, c in frame_cells if r == 0)
    if side == "D":
        return sorted((r, c) for r, c in frame_cells if r == h - 1)
    if side == "L":
        return sorted((r, c) for r, c in frame_cells if c == 0)
    if side == "R":
        return sorted((r, c) for r, c in frame_cells if c == w - 1)
    raise ValueError(f"unknown side {side}")


def _marker_from_edge_cell(cell: Coord, side: str) -> Coord:
    r, c = cell
    if side == "U":
        return r - 1, c
    if side == "D":
        return r + 1, c
    if side == "L":
        return r, c - 1
    if side == "R":
        return r, c + 1
    raise ValueError(f"unknown side {side}")


def _pick_marker_cells(rng: np.random.Generator, edge_cells: list[Coord]) -> list[Coord]:
    if len(edge_cells) == 1:
        return [edge_cells[0]]

    if bool(rng.integers(0, 2)):
        return [edge_cells[int(rng.integers(0, len(edge_cells)))]]

    adjacent_pairs = []
    for first, second in zip(edge_cells, edge_cells[1:]):
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1:
            adjacent_pairs.append([first, second])
    if adjacent_pairs:
        idx = int(rng.integers(0, len(adjacent_pairs)))
        return adjacent_pairs[idx]

    picks = rng.choice(len(edge_cells), size=2, replace=False)
    return [edge_cells[int(picks[0])], edge_cells[int(picks[1])]]


def _local_reflections(content: list[Coord], side: str, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    bbox = (0, 0, h - 1, w - 1)
    return [_reflect_cell(cell, side, bbox) for cell in content]


def _build_object_spec(
    rng: np.random.Generator,
    color: int,
    forced_style: str | None = None,
    force_two_markers: bool = False,
) -> dict[str, object]:
    inner_w = int(rng.integers(3, 6))
    depth = int(rng.integers(2, 5))

    style = forced_style or ("l" if rng.random() < 0.45 else "u")
    leg_side = None
    leg_len = 0
    if style == "l":
        leg_side = "left" if bool(rng.integers(0, 2)) else "right"
        leg_len = int(rng.integers(1, 3))

    frame_canonical, cavity_canonical, canonical_shape = _build_canonical_frame(
        inner_w, depth, style, leg_side, leg_len
    )
    open_dir = OPEN_DIRS[int(rng.integers(0, len(OPEN_DIRS)))]
    frame_cells, shape = _transform_cells(frame_canonical, canonical_shape, open_dir)
    cavity_cells, _ = _transform_cells(cavity_canonical, canonical_shape, open_dir)

    content_count = int(rng.integers(1, min(3, len(cavity_cells)) + 1))
    cavity_list = sorted(cavity_cells)
    content_pick = rng.choice(len(cavity_list), size=content_count, replace=False)
    content_cells = [cavity_list[int(idx)] for idx in np.atleast_1d(content_pick)]

    closed_sides = [side for side in OPEN_DIRS if side != open_dir]
    back_side = OPPOSITE[open_dir]
    if style == "l":
        marker_sides = [side for side in closed_sides if side != back_side]
    else:
        marker_sides = closed_sides
    marker_side = marker_sides[int(rng.integers(0, len(marker_sides)))]

    edge_cells = _side_candidates(frame_cells, shape, marker_side)
    marker_edge_cells = _pick_marker_cells(rng, edge_cells)
    if force_two_markers and len(edge_cells) >= 2:
        adjacent_pairs = []
        for first, second in zip(edge_cells, edge_cells[1:]):
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1:
                adjacent_pairs.append([first, second])
        if adjacent_pairs:
            marker_edge_cells = adjacent_pairs[int(rng.integers(0, len(adjacent_pairs)))]
        else:
            picks = rng.choice(len(edge_cells), size=2, replace=False)
            marker_edge_cells = [edge_cells[int(picks[0])], edge_cells[int(picks[1])]]
    marker_cells = [_marker_from_edge_cell(cell, marker_side) for cell in marker_edge_cells]

    reflected_cells = _local_reflections(content_cells, marker_side, shape)
    h, w = shape
    for r, c in reflected_cells:
        assert r < 0 or r >= h or c < 0 or c >= w, "reflected cells must leave the frame bbox"

    return {
        "color": int(color),
        "style": style,
        "open_dir": open_dir,
        "marker_side": marker_side,
        "shape": shape,
        "frame_cells": frame_cells,
        "cavity_cells": cavity_cells,
        "content_cells": content_cells,
        "marker_cells": marker_cells,
        "reflected_cells": reflected_cells,
    }


def _cell_with_buffer(block_mask: np.ndarray, cell: Coord) -> bool:
    r, c = cell
    return bool(block_mask[r, c])


def _mark_with_buffer(block_mask: np.ndarray, cell: Coord) -> None:
    h, w = block_mask.shape
    r, c = cell
    r0 = max(0, r - 1)
    r1 = min(h, r + 2)
    c0 = max(0, c - 1)
    c1 = min(w, c + 2)
    block_mask[r0:r1, c0:c1] = True


def _render_input(shape: tuple[int, int], placed_specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in placed_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        color = int(spec["color"])
        for r, c in spec["frame_cells"]:
            grid[top + r, left + c] = color
        for r, c in spec["content_cells"]:
            grid[top + r, left + c] = YELLOW
        for r, c in spec["marker_cells"]:
            grid[top + r, left + c] = RED
    return grid


def _render_expected_output(shape: tuple[int, int], placed_specs: list[dict[str, object]]) -> np.ndarray:
    out = _render_input(shape, placed_specs)
    for spec in placed_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        for r, c in spec["content_cells"]:
            out[top + r, left + c] = BG
        for r, c in spec["marker_cells"]:
            out[top + r, left + c] = BG
        for r, c in spec["reflected_cells"]:
            out[top + r, left + c] = YELLOW
    return out


def _frame_gap_clear(frame_mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return not bool(frame_mask[top : top + h, left : left + w].any())


def _mark_frame_gap(frame_mask: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int = 3) -> None:
    h, w = shape
    H, W = frame_mask.shape
    r0 = max(0, top - pad)
    r1 = min(H, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(W, left + w + pad)
    frame_mask[r0:r1, c0:c1] = True


def _place_specs(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    H, W = canvas_shape
    frame_gap_mask = np.zeros(canvas_shape, dtype=bool)
    scene_block_mask = np.zeros(canvas_shape, dtype=bool)
    placed: list[dict[str, object]] = []

    for spec in sorted(specs, key=lambda item: item["shape"][0] * item["shape"][1], reverse=True):
        h, w = spec["shape"]
        all_cells = list(spec["frame_cells"]) + list(spec["content_cells"]) + list(spec["marker_cells"]) + list(spec["reflected_cells"])
        min_r = min(r for r, _c in all_cells)
        max_r = max(r for r, _c in all_cells)
        min_c = min(c for _r, c in all_cells)
        max_c = max(c for _r, c in all_cells)

        top_lo = -min_r
        top_hi = H - 1 - max_r
        left_lo = -min_c
        left_hi = W - 1 - max_c
        if top_lo > top_hi or left_lo > left_hi:
            return None

        candidates = [(top, left) for top in range(top_lo, top_hi + 1) for left in range(left_lo, left_hi + 1)]
        rng.shuffle(candidates)

        placed_here = False
        for top, left in candidates:
            if not _frame_gap_clear(frame_gap_mask, top, left, spec["shape"]):
                continue

            abs_cells = [(top + r, left + c) for r, c in all_cells]
            if any(_cell_with_buffer(scene_block_mask, cell) for cell in abs_cells):
                continue

            spec["top"] = int(top)
            spec["left"] = int(left)
            _mark_frame_gap(frame_gap_mask, top, left, spec["shape"])
            for cell in abs_cells:
                _mark_with_buffer(scene_block_mask, cell)
            placed.append(spec)
            placed_here = True
            break

        if not placed_here:
            return None

    return placed


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap, col_gap


def _semantic_ok(inp: np.ndarray, placed_specs: list[dict[str, object]]) -> bool:
    out = main(inp)
    expected = _render_expected_output(inp.shape, placed_specs)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, inp):
        return False
    if np.count_nonzero(inp == RED) < len(placed_specs):
        return False
    if np.any(out == RED):
        return False
    if np.count_nonzero(inp == YELLOW) != np.count_nonzero(out == YELLOW):
        return False
    if not any(len(spec["content_cells"]) >= 2 for spec in placed_specs):
        return False
    if len(placed_specs) >= 2 and len({spec["marker_side"] for spec in placed_specs}) < 2:
        return False
    if len(placed_specs) == 3 and len({spec["style"] for spec in placed_specs}) < 2:
        return False

    frame_bboxes = []
    for spec in placed_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        h, w = spec["shape"]
        frame_bboxes.append((top, left, top + h - 1, left + w - 1))
        for r, c in spec["reflected_cells"]:
            if 0 <= r < h and 0 <= c < w:
                return False
        for mr, mc in spec["marker_cells"]:
            if mr in range(h) and mc in range(w):
                return False
    for idx, first in enumerate(frame_bboxes):
        for second in frame_bboxes[idx + 1 :]:
            row_gap, col_gap = _bbox_distance(first, second)
            if row_gap < 3 and col_gap < 3:
                return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_objects = int(rng.choice(np.array([2, 3], dtype=int)))
    height = int(rng.integers(15, 24))
    width = int(rng.integers(16, 26))

    colors = rng.choice(np.array(FRAME_COLORS, dtype=int), size=num_objects, replace=False)
    forced_styles = [None] * num_objects
    if num_objects == 3:
        forced_styles[0] = "u"
        forced_styles[1] = "l"
    elif num_objects == 2 and rng.random() < 0.65:
        forced_styles = ["u", "l"]

    force_two_marker_idx = int(rng.integers(0, num_objects))

    specs = []
    for idx in range(num_objects):
        specs.append(
            _build_object_spec(
                rng,
                int(colors[idx]),
                forced_style=forced_styles[idx],
                force_two_markers=(idx == force_two_marker_idx),
            )
        )

    placed = _place_specs(rng, (height, width), specs)
    if placed is None:
        return None

    grid = _render_input((height, width), placed)
    if not _semantic_ok(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ejection ARC input")
