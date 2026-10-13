import random
from collections import Counter, deque

import numpy as np


_BG = 0
_MARKER = 5
_BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_DIRECTIONS = ("up", "right", "down", "left")
_DIR_STEP = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
_DIR_TURNS = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}

_SIZE_VALUES = (3, 5, 7, 9)
_COUNT_RANGE = (1, 5)
_MIN_SIDE_BY_COUNT = {
    1: 11,
    2: 13,
    3: 16,
    4: 19,
    5: 22,
}
_MAX_SIDE = 26
_BORDER_MARGIN = 1
_OBJECT_GAP = 1
_MAX_ATTEMPTS = 400


def _border_bg(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_center(cr: int, cc: int, h: int, w: int) -> tuple[int, int, int, int]:
    half_h = h // 2
    half_w = w // 2
    return cr - half_h, cc - half_w, cr + half_h, cc + half_w


def _boxes_separated(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _paint_box(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
    require_bg: bool,
) -> None:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if require_bg:
        assert bool(np.all(patch == _BG)), "boxes must not overlap"
    patch[:, :] = int(color)


def _pointer_direction(rel_cells: list[tuple[int, int]]) -> str:
    rel_set = set(rel_cells)
    assert (0, 0) in rel_set, "marker must include the center cell"

    shaft = [cell for cell in rel_cells if cell != (0, 0)]
    assert 1 <= len(shaft) <= 2, "pointer shaft must have length 1 or 2"

    if all(dc == 0 for dr, dc in shaft):
        steps = sorted(dr for dr, _ in shaft)
        if all(step < 0 for step in steps):
            assert [abs(step) for step in steps] == list(range(len(shaft), 0, -1))
            return "up"
        if all(step > 0 for step in steps):
            assert [abs(step) for step in steps] == list(range(1, len(shaft) + 1))
            return "down"

    if all(dr == 0 for dr, dc in shaft):
        steps = sorted(dc for _, dc in shaft)
        if all(step < 0 for step in steps):
            assert [abs(step) for step in steps] == list(range(len(shaft), 0, -1))
            return "left"
        if all(step > 0 for step in steps):
            assert [abs(step) for step in steps] == list(range(1, len(shaft) + 1))
            return "right"

    raise AssertionError("marker must be a straight center-out pointer")


def _rotated_shape(h: int, w: int, turns: int) -> tuple[int, int]:
    return (h, w) if turns % 2 == 0 else (w, h)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each rectangular block according to the grey pointer embedded at its center.

    Input structure:
    - Several disconnected solid-colored rectangles sit on a black background.
    - Each rectangle contains a grey center dot and a short grey pointer shaft.

    Rule:
    - Up -> 0 degrees, right -> 90 degrees clockwise, down -> 180 degrees,
      left -> 270 degrees clockwise.
    - Rotate each rectangle around its own center cell.
    - Erase the grey pointer in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _border_bg(g)

    out = np.full((h, w), bg, dtype=int)
    comps = _components4(g != bg)
    assert comps, "expected at least one non-background block"

    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        assert bh >= 3 and bw >= 3 and bh % 2 == 1 and bw % 2 == 1

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch != bg)), "each component must fill its bounding box"

        colors = Counter(map(int, patch.ravel()))
        assert len(colors) == 2, "each block must contain exactly one block color and one marker color"
        block_color, _ = colors.most_common(1)[0]
        marker_colors = [color for color in colors if int(color) != int(block_color)]
        assert len(marker_colors) == 1
        marker_color = int(marker_colors[0])

        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        assert int(g[cr, cc]) == marker_color, "marker center must sit at the rectangle center"

        marker_rel = []
        for rr, cc2 in np.argwhere(patch == marker_color):
            marker_rel.append((int(rr + r0 - cr), int(cc2 + c0 - cc)))
        direction = _pointer_direction(marker_rel)
        turns = _DIR_TURNS[direction]
        oh, ow = _rotated_shape(bh, bw, turns)
        box = _bbox_from_center(cr, cc, oh, ow)
        tr0, tc0, tr1, tc1 = box
        assert 0 <= tr0 <= tr1 < h and 0 <= tc0 <= tc1 < w, "rotated block must stay inside the grid"
        _paint_box(out, box, int(block_color), require_bg=True)

    return out


def _direction_sequence(n_obj: int) -> list[str]:
    seq = random.sample(_DIRECTIONS, min(n_obj, len(_DIRECTIONS)))
    while len(seq) < n_obj:
        seq.append(random.choice(_DIRECTIONS))
    random.shuffle(seq)
    return seq


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for spec in specs:
        in_box = _bbox_from_center(spec["cr"], spec["cc"], spec["h"], spec["w"])
        _paint_box(grid, in_box, int(spec["color"]), require_bg=True)
        dr, dc = _DIR_STEP[spec["direction"]]
        grid[spec["cr"], spec["cc"]] = _MARKER
        for step in range(1, int(spec["length"]) + 1):
            grid[spec["cr"] + dr * step, spec["cc"] + dc * step] = _MARKER
    return grid


def _render_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for spec in specs:
        oh, ow = _rotated_shape(int(spec["h"]), int(spec["w"]), int(spec["turns"]))
        out_box = _bbox_from_center(spec["cr"], spec["cc"], oh, ow)
        _paint_box(grid, out_box, int(spec["color"]), require_bg=True)
    return grid


def _candidate_sizes(grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    limit = min(grid_h, grid_w) - 2 * (_BORDER_MARGIN + 1)
    sizes = [size for size in _SIZE_VALUES if size <= limit]
    pairs = [(hh, ww) for hh in sizes for ww in sizes]
    random.shuffle(pairs)
    if random.random() < 0.7:
        pairs.sort(key=lambda pair: pair[0] == pair[1])
    return pairs


def _attempt_generate() -> np.ndarray | None:
    n_obj = random.randint(*_COUNT_RANGE)
    min_side = _MIN_SIDE_BY_COUNT[n_obj]
    grid_h = random.randint(min_side, _MAX_SIDE)
    grid_w = random.randint(min_side, _MAX_SIDE)

    directions = _direction_sequence(n_obj)
    colors = random.sample(_BLOCK_COLORS, n_obj)
    size_pairs = _candidate_sizes(grid_h, grid_w)
    if not size_pairs:
        return None

    specs: list[dict] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    for idx in range(n_obj):
        placed = False
        direction = directions[idx]
        turns = _DIR_TURNS[direction]

        for hh, ww in size_pairs:
            axis_span = hh if direction in ("up", "down") else ww
            lengths = [1] + ([2] if axis_span >= 5 else [])
            random.shuffle(lengths)

            for length in lengths:
                oh, ow = _rotated_shape(hh, ww, turns)
                row_half = max(hh // 2, oh // 2)
                col_half = max(ww // 2, ow // 2)
                min_r = _BORDER_MARGIN + row_half
                max_r = grid_h - _BORDER_MARGIN - 1 - row_half
                min_c = _BORDER_MARGIN + col_half
                max_c = grid_w - _BORDER_MARGIN - 1 - col_half
                if min_r > max_r or min_c > max_c:
                    continue

                centers = [
                    (cr, cc)
                    for cr in range(min_r, max_r + 1)
                    for cc in range(min_c, max_c + 1)
                ]
                random.shuffle(centers)

                for cr, cc in centers:
                    in_box = _bbox_from_center(cr, cc, hh, ww)
                    out_box = _bbox_from_center(cr, cc, oh, ow)
                    if not all(_boxes_separated(in_box, box, _OBJECT_GAP) for box in input_boxes):
                        continue
                    if not all(_boxes_separated(out_box, box, _OBJECT_GAP) for box in output_boxes):
                        continue

                    specs.append(
                        {
                            "cr": int(cr),
                            "cc": int(cc),
                            "h": int(hh),
                            "w": int(ww),
                            "color": int(colors[idx]),
                            "direction": direction,
                            "turns": int(turns),
                            "length": int(length),
                        }
                    )
                    input_boxes.append(in_box)
                    output_boxes.append(out_box)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

        if not placed:
            return None

    candidate = _render_input((grid_h, grid_w), specs)
    expected = _render_output((grid_h, grid_w), specs)

    # Avoid the least informative cases where every block keeps the same footprint.
    if not any(spec["turns"] % 2 == 1 and spec["h"] != spec["w"] for spec in specs):
        if random.random() < 0.8:
            return None

    if not np.array_equal(main(candidate), expected):
        return None
    return candidate


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input failed after {_MAX_ATTEMPTS} attempts")
