from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _paste(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = sprite != bg
    canvas[top : top + sprite.shape[0], left : left + sprite.shape[1]][mask] = sprite[mask]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for rr, cc in coords:
                sprite[rr - r0, cc - c0] = color
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                }
            )
    return components


def _first_true_positions(mask: np.ndarray, axis: int) -> np.ndarray:
    positions = []
    if axis == 0:
        for c in range(mask.shape[1]):
            rows = np.flatnonzero(mask[:, c])
            assert rows.size > 0
            positions.append(int(rows[0]))
    else:
        for r in range(mask.shape[0]):
            cols = np.flatnonzero(mask[r])
            assert cols.size > 0
            positions.append(int(cols[0]))
    return np.array(positions, dtype=int)


def _last_true_positions(mask: np.ndarray, axis: int) -> np.ndarray:
    positions = []
    if axis == 0:
        for c in range(mask.shape[1]):
            rows = np.flatnonzero(mask[:, c])
            assert rows.size > 0
            positions.append(int(rows[-1]))
    else:
        for r in range(mask.shape[0]):
            cols = np.flatnonzero(mask[r])
            assert cols.size > 0
            positions.append(int(cols[-1]))
    return np.array(positions, dtype=int)


def _is_single_run(line: np.ndarray) -> bool:
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return False
    return idx[-1] - idx[0] + 1 == idx.size


def _transitions(line: np.ndarray) -> int:
    if line.size <= 1:
        return 0
    return int(np.sum(line[:-1] != line[1:]))


def _analyze_piece(component: dict, bg: int) -> dict:
    sprite = np.asarray(component["sprite"], dtype=int)
    mask = sprite != bg
    h, w = mask.shape
    assert h >= 2 and w >= 2

    border_lines = {
        "top": mask[0, :],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }
    jagged_sides = [
        side
        for side, line in border_lines.items()
        if line.any() and (~line).any() and _transitions(line) >= 2
    ]
    assert len(jagged_sides) == 1, f"expected exactly one jagged side, got {jagged_sides}"
    side = jagged_sides[0]

    if side in ("left", "right"):
        for r in range(h):
            assert _is_single_run(mask[r, :])
    else:
        for c in range(w):
            assert _is_single_run(mask[:, c])

    row_first = _first_true_positions(mask, axis=1)
    row_last = _last_true_positions(mask, axis=1)
    col_first = _first_true_positions(mask, axis=0)
    col_last = _last_true_positions(mask, axis=0)

    return {
        "color": int(component["color"]),
        "bbox": tuple(map(int, component["bbox"])),
        "sprite": sprite,
        "side": side,
        "border": border_lines[side].astype(int),
        "row_first": row_first,
        "row_last": row_last,
        "col_first": col_first,
        "col_last": col_last,
    }


def _assemble_pair(a: dict, b: dict, bg: int) -> np.ndarray | None:
    sides = {a["side"], b["side"]}
    if sides == {"right", "left"}:
        left_piece = a if a["side"] == "right" else b
        right_piece = b if b["side"] == "left" else a
        if left_piece["sprite"].shape[0] != right_piece["sprite"].shape[0]:
            return None
        if not np.all(left_piece["border"] + right_piece["border"] == 1):
            return None

        shift_values = (left_piece["row_last"] + 1) - right_piece["row_first"]
        if np.any(shift_values != shift_values[0]):
            return None
        shift = int(shift_values[0])
        if shift <= 0:
            return None

        out_h = int(left_piece["sprite"].shape[0])
        out_w = shift + int(right_piece["sprite"].shape[1])
        out = np.full((out_h, out_w), bg, dtype=int)
        _paste(out, left_piece["sprite"], 0, 0, bg)
        _paste(out, right_piece["sprite"], 0, shift, bg)
    elif sides == {"bottom", "top"}:
        top_piece = a if a["side"] == "bottom" else b
        bottom_piece = b if b["side"] == "top" else a
        if top_piece["sprite"].shape[1] != bottom_piece["sprite"].shape[1]:
            return None
        if not np.all(top_piece["border"] + bottom_piece["border"] == 1):
            return None

        shift_values = (top_piece["col_last"] + 1) - bottom_piece["col_first"]
        if np.any(shift_values != shift_values[0]):
            return None
        shift = int(shift_values[0])
        if shift <= 0:
            return None

        out_h = shift + int(bottom_piece["sprite"].shape[0])
        out_w = int(top_piece["sprite"].shape[1])
        out = np.full((out_h, out_w), bg, dtype=int)
        _paste(out, top_piece["sprite"], 0, 0, bg)
        _paste(out, bottom_piece["sprite"], shift, 0, bg)
    else:
        return None

    if np.any(out == bg):
        return None
    return out


def _pair_pieces(pieces: list[dict], bg: int) -> list[dict]:
    candidates: dict[tuple[int, int], dict] = {}
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            rect = _assemble_pair(pieces[i], pieces[j], bg)
            if rect is None:
                continue
            bbox_a = pieces[i]["bbox"]
            bbox_b = pieces[j]["bbox"]
            anchor = (min(bbox_a[0], bbox_b[0]), min(bbox_a[1], bbox_b[1]))
            candidates[(i, j)] = {
                "rect": rect,
                "area": int(rect.shape[0] * rect.shape[1]),
                "anchor": anchor,
            }

    memo: dict[tuple[int, ...], list[list[dict]]] = {}

    def rec(remaining: tuple[int, ...]) -> list[list[dict]]:
        if not remaining:
            return [[]]
        if remaining in memo:
            return memo[remaining]

        i = remaining[0]
        sols: list[list[dict]] = []
        rem_set = set(remaining)
        for j in remaining[1:]:
            key = (i, j)
            if key not in candidates:
                continue
            next_remaining = tuple(sorted(rem_set - {i, j}))
            for tail in rec(next_remaining):
                sols.append([candidates[key], *tail])
        memo[remaining] = sols
        return sols

    solutions = rec(tuple(range(len(pieces))))
    assert len(solutions) == 1, "pairing must be unique"
    return solutions[0]


def _pack_rectangles(rectangles: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = max(int(rect.shape[0]) for rect in rectangles)
    out_w = sum(int(rect.shape[1]) for rect in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for rect in rectangles:
        _paste(out, rect, 0, col, bg)
        col += int(rect.shape[1]) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble complementary jagged fragments into sorted full rectangles.

    Input structure
    - The most common color is the background.
    - There are 4-6 monochrome 4-connected fragments, each in a distinct color.
    - Every fragment is almost rectangular: exactly one side has a 1-cell-deep alternating tooth pattern, and the other three sides are straight runs.

    Rule
    - Crop every fragment to its bounding box and identify the jagged side by the only bbox border whose occupied/empty pattern alternates repeatedly.
    - Pair fragments whose jagged sides face opposite directions, have the same interface length, and have complementary border patterns.
    - Translate each matched pair until the teeth interlock; the union must become a solid rectangle while both fragment colors stay unchanged.
    - Sort the completed rectangles by descending area and pack them left to right, top-aligned, with one background column between neighbors.

    Key insight
    - The pairing signal is not color or proximity but the alternating occupancy pattern on the jagged border itself: complementary border masks reveal the mate, and the row/column edge profiles give the exact translation needed to close the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    components = _find_components(g, bg)
    assert 4 <= len(components) <= 6 and len(components) % 2 == 0

    pieces = [_analyze_piece(component, bg) for component in components]
    paired = _pair_pieces(pieces, bg)
    paired.sort(key=lambda item: (-item["area"], item["anchor"][0], item["anchor"][1]))

    rectangles = [item["rect"] for item in paired]
    return _pack_rectangles(rectangles, bg)


# --------------- generator ---------------

PALETTE = tuple(range(10))
PAIR_COUNT_CHOICES = (2, 3)
AXES = ("vertical", "horizontal")
JAGGED_LENGTH_RANGE = (4, 7)
BASE_SIDE_RANGE = (2, 4)
OUTPUT_MAX = 30
INPUT_SIDE_RANGE = (18, 30)
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _alternating_bits(length: int, start: int) -> np.ndarray:
    return np.array([(start + i) % 2 for i in range(length)], dtype=int)


def _build_pair(axis: str, length: int, start: int, a: int, b: int, colors: tuple[int, int], bg: int) -> tuple[list[np.ndarray], np.ndarray]:
    color_a, color_b = map(int, colors)
    bits = _alternating_bits(length, start)

    if axis == "vertical":
        left = np.full((length, a + 1), bg, dtype=int)
        right = np.full((length, b + 1), bg, dtype=int)
        for r, bit in enumerate(bits):
            left[r, : a + bit] = color_a
            right[r, bit:] = color_b
        rect = np.full((length, a + b + 1), bg, dtype=int)
        _paste(rect, left, 0, 0, bg)
        _paste(rect, right, 0, a, bg)
        pieces = [left, right]
    else:
        top = np.full((a + 1, length), bg, dtype=int)
        bottom = np.full((b + 1, length), bg, dtype=int)
        for c, bit in enumerate(bits):
            top[: a + bit, c] = color_a
            bottom[bit:, c] = color_b
        rect = np.full((a + b + 1, length), bg, dtype=int)
        _paste(rect, top, 0, 0, bg)
        _paste(rect, bottom, a, 0, bg)
        pieces = [top, bottom]

    assert not np.any(rect == bg)
    return pieces, rect


def _sample_pair_specs() -> list[dict] | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    specs: list[dict] = []
    used_signatures: set[tuple[str, int, int]] = set()
    used_areas: set[int] = set()

    for _ in range(200):
        if len(specs) == n_pairs:
            break
        axis = random.choice(AXES)
        length = random.randint(*JAGGED_LENGTH_RANGE)
        start = random.randint(0, 1)
        signature = (axis, length, start)
        if signature in used_signatures:
            continue

        a = random.randint(*BASE_SIDE_RANGE)
        b = random.randint(*BASE_SIDE_RANGE)
        if axis == "vertical":
            rect_h, rect_w = length, a + b + 1
        else:
            rect_h, rect_w = a + b + 1, length
        area = rect_h * rect_w
        if area in used_areas:
            continue

        specs.append(
            {
                "axis": axis,
                "length": length,
                "start": start,
                "a": a,
                "b": b,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "area": area,
            }
        )
        used_signatures.add(signature)
        used_areas.add(area)

    if len(specs) != n_pairs:
        return None

    out_w = sum(spec["rect_w"] for spec in specs) + n_pairs - 1
    out_h = max(spec["rect_h"] for spec in specs)
    if out_w > OUTPUT_MAX or out_h > OUTPUT_MAX:
        return None
    return specs


def _mark_blocked(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    rr0 = max(0, top - 1)
    rr1 = min(blocked.shape[0], top + h + 1)
    cc0 = max(0, left - 1)
    cc1 = min(blocked.shape[1], left + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _scatter_pieces(pieces: list[np.ndarray], bg: int, height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    order = sorted(pieces, key=lambda piece: piece.shape[0] * piece.shape[1], reverse=True)

    for piece in order:
        ph, pw = map(int, piece.shape)
        if ph + 2 > height or pw + 2 > width:
            return None

        placements = []
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, height - ph - 1)
            left = random.randint(1, width - pw - 1)
            if blocked[top : top + ph, left : left + pw].any():
                continue
            placements.append((top, left))
            if len(placements) >= 12:
                break

        if not placements:
            return None

        top, left = random.choice(placements)
        _paste(grid, piece, top, left, bg)
        _mark_blocked(blocked, top, left, ph, pw)

    return grid


def _pack_expected(rectangles: list[np.ndarray], bg: int) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda rect: -(rect.shape[0] * rect.shape[1]))
    return _pack_rectangles(ordered, bg)


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_pair_specs()
    if specs is None:
        return None

    n_pieces = 2 * len(specs)
    colors = random.sample(PALETTE, n_pieces + 1)
    bg = int(colors[0])
    piece_colors = colors[1:]

    pieces: list[np.ndarray] = []
    rectangles: list[np.ndarray] = []
    for spec_idx, spec in enumerate(specs):
        pair_colors = (piece_colors[2 * spec_idx], piece_colors[2 * spec_idx + 1])
        pair_pieces, rect = _build_pair(
            spec["axis"],
            spec["length"],
            spec["start"],
            spec["a"],
            spec["b"],
            pair_colors,
            bg,
        )
        pieces.extend(pair_pieces)
        rectangles.append(rect)

    fg_area = sum(int(np.count_nonzero(piece != bg)) for piece in pieces)
    min_side_h = max(max(piece.shape[0] for piece in pieces) + 2, INPUT_SIDE_RANGE[0])
    min_side_w = max(max(piece.shape[1] for piece in pieces) + 2, INPUT_SIDE_RANGE[0])
    if min_side_h > OUTPUT_MAX or min_side_w > OUTPUT_MAX:
        return None

    candidates = []
    for height in range(min_side_h, OUTPUT_MAX + 1):
        for width in range(min_side_w, OUTPUT_MAX + 1):
            if height * width > 2 * fg_area:
                candidates.append((height, width))
    if not candidates:
        return None

    expected = _pack_expected(rectangles, bg)
    random.shuffle(pieces)

    for _ in range(80):
        height, width = random.choice(candidates)
        inp = _scatter_pieces(pieces, bg, height, width)
        if inp is None:
            continue
        if np.array_equal(main(inp), expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jagged-fragment puzzle")
