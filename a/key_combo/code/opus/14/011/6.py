from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BRIDGE = 5
PALETTE = [2, 3, 4, 1, 7, 6]
LEFT_RIGHT_RANK_ORDER = [2, 1, 4, 3, 6, 5]
SPECIAL_COLUMN_PAIRS = [(2, 1), (4, 3), (6, 5)]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            box = grid[r0 : r1 + 1, c0 : c1 + 1]
            expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == expected_area, "foreground objects must be solid rectangles"
            assert np.all(box == color), "foreground objects must be monochrome rectangles"

            objects.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )

    return objects


def _shift_objects(ordered_objects: list[dict], shape: tuple[int, int]) -> tuple[list[dict], np.ndarray]:
    h, w = shape
    shifted: list[dict] = []
    canvas = np.full(shape, BG, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    for rank, obj in enumerate(ordered_objects, start=1):
        dc = -1 if rank % 2 == 1 else 1
        moved = obj["coords"] + np.array([0, dc], dtype=int)
        in_bounds = (
            (0 <= moved[:, 0]) & (moved[:, 0] < h) &
            (0 <= moved[:, 1]) & (moved[:, 1] < w)
        )
        moved = moved[in_bounds]
        assert len(moved) > 0, "a shifted object disappeared completely"
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "shifted objects must remain disjoint"

        color = int(obj["color"])
        canvas[moved[:, 0], moved[:, 1]] = color
        occupied[moved[:, 0], moved[:, 1]] = True

        r0 = int(moved[:, 0].min())
        r1 = int(moved[:, 0].max())
        c0 = int(moved[:, 1].min())
        c1 = int(moved[:, 1].max())
        shifted.append(
            {
                "rank": rank,
                "color": color,
                "coords": moved,
                "bbox": (r0, c0, r1, c1),
            }
        )

    return shifted, canvas


def _pair_gap(left_bbox: tuple[int, int, int, int], right_bbox: tuple[int, int, int, int]) -> int:
    return int(right_bbox[1] - left_bbox[3] - 1)


def _shifted_relations(shifted_objects: list[dict]) -> list[dict]:
    relations: list[dict] = []
    for i in range(len(shifted_objects)):
        for j in range(i + 1, len(shifted_objects)):
            a = shifted_objects[i]
            b = shifted_objects[j]
            if a["bbox"][3] < b["bbox"][1]:
                left, right = a, b
            elif b["bbox"][3] < a["bbox"][1]:
                left, right = b, a
            else:
                continue

            row0 = max(left["bbox"][0], right["bbox"][0])
            row1 = min(left["bbox"][2], right["bbox"][2])
            relations.append(
                {
                    "left": left,
                    "right": right,
                    "gap": _pair_gap(left["bbox"], right["bbox"]),
                    "row0": row0,
                    "row1": row1,
                }
            )
    return relations


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))

    objects = _find_objects(g)
    assert 4 <= len(objects) <= 6, "expected 4-6 rectangles"
    top_rows = [obj["bbox"][0] for obj in objects]
    assert len(set(top_rows)) == len(top_rows), "top rows must be unique"

    ordered = sorted(objects, key=lambda obj: obj["bbox"][0])
    shifted, out = _shift_objects(ordered, g.shape)

    for relation in _shifted_relations(shifted):
        if relation["gap"] != 1 or relation["row0"] > relation["row1"]:
            continue
        fill_col = relation["left"]["bbox"][3] + 1
        row0, row1 = relation["row0"], relation["row1"]
        column = out[row0 : row1 + 1, fill_col]
        bg_mask = column == BG
        column[bg_mask] = BRIDGE
        out[row0 : row1 + 1, fill_col] = column

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift rectangles by odd/even vertical rank, then bridge one-column gaps that emerge after the shift.

    Input structure:
    - The background is black.
    - The foreground consists of several solid 4-connected monochrome rectangles with distinct colors.
    - Rectangle top rows are all different, so the objects have a unique top-to-bottom order.

    Rule:
    - Sort the rectangles by their bbox top row.
    - Shift ranks 1, 3, 5, ... one column left and ranks 2, 4, 6, ... one column right, clipping any pixels that leave the grid.
    - After shifting, examine every left-right rectangle pair; when their shifted column spans are separated by exactly one column and their row spans overlap, fill that gap column with gray over the overlap rows, but only on background cells.

    Key insight:
    - The gray bridge is not determined from the original layout; it appears only after the parity-driven alternating shift changes which rectangles become one column apart.
    """
    return _solve(grid)


def _make_widths(rng: np.random.Generator) -> list[int]:
    widths = [2] * 6
    extra_units = int(rng.integers(0, 4))
    while extra_units > 0:
        idx = int(rng.integers(0, 6))
        if widths[idx] < 4:
            widths[idx] += 1
            extra_units -= 1
    return widths


def _make_row_layout(rng: np.random.Generator) -> tuple[list[int], list[int]]:
    heights = [int(rng.integers(2, 5)) for _ in range(6)]
    contrast_pair = int(rng.integers(0, 3))

    tops: list[int] = []
    current_top = int(rng.integers(1, 4))
    for pair_idx in range(3):
        top_a = current_top
        h_a = heights[2 * pair_idx]
        bottom_a = top_a + h_a - 1

        h_b = heights[2 * pair_idx + 1]
        if pair_idx == contrast_pair:
            top_b = bottom_a + int(rng.integers(1, 4))
        else:
            top_b = top_a + int(rng.integers(1, h_a))
        bottom_b = top_b + h_b - 1

        tops.extend([top_a, top_b])
        current_top = max(bottom_a, bottom_b) + int(rng.integers(1, 4))

    return tops, heights


def _make_input_attempt(rng: np.random.Generator) -> np.ndarray | None:
    widths = _make_widths(rng)
    tops, heights = _make_row_layout(rng)

    gaps = [3, 3, 3, 3, 3]
    base_width = sum(widths) + sum(gaps)
    if base_width > 30:
        return None

    spare = 30 - base_width
    left_margin = int(rng.integers(0, spare + 1))
    spare -= left_margin
    right_margin = int(rng.integers(0, spare + 1))
    spare -= right_margin
    for gap_idx in (1, 3):
        add = int(rng.integers(0, spare + 1))
        gaps[gap_idx] += add
        spare -= add

    rank_to_color = {rank: color for rank, color in zip(range(1, 7), rng.permutation(PALETTE))}
    rank_to_width = {rank: widths[rank - 1] for rank in range(1, 7)}
    rank_to_top = {rank: tops[rank - 1] for rank in range(1, 7)}
    rank_to_height = {rank: heights[rank - 1] for rank in range(1, 7)}

    x = left_margin
    rank_to_left: dict[int, int] = {}
    for idx, rank in enumerate(LEFT_RIGHT_RANK_ORDER):
        rank_to_left[rank] = x
        x += rank_to_width[rank]
        if idx < len(gaps):
            x += gaps[idx]
    width = x + right_margin

    max_bottom = max(rank_to_top[rank] + rank_to_height[rank] - 1 for rank in range(1, 7))
    height = max_bottom + 1 + int(rng.integers(1, 4))
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for rank in range(1, 7):
        r0 = rank_to_top[rank]
        c0 = rank_to_left[rank]
        h = rank_to_height[rank]
        w = rank_to_width[rank]
        grid[r0 : r0 + h, c0 : c0 + w] = rank_to_color[rank]

    return grid


def _validate_semantics(inp: np.ndarray) -> None:
    objects = _find_objects(inp)
    assert len(objects) == 6
    assert len({obj["color"] for obj in objects}) == 6

    ordered = sorted(objects, key=lambda obj: obj["bbox"][0])
    assert [obj["bbox"][0] for obj in ordered] == sorted({obj["bbox"][0] for obj in ordered})

    for obj in ordered:
        assert 2 <= obj["height"] <= 4
        assert 2 <= obj["width"] <= 4

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a = ordered[i]["bbox"]
            b = ordered[j]["bbox"]
            if a[3] < b[1]:
                gap = _pair_gap(a, b)
            elif b[3] < a[1]:
                gap = _pair_gap(b, a)
            else:
                gap = -1
            assert gap > 2, "input rectangles must not be adjacent before shifting"

    shifted, shifted_canvas = _shift_objects(ordered, inp.shape)
    relations = _shifted_relations(shifted)
    one_col_pairs = [rel for rel in relations if rel["gap"] == 1]
    bridge_pairs = [rel for rel in one_col_pairs if rel["row0"] <= rel["row1"]]
    contrast_pairs = [rel for rel in one_col_pairs if rel["row0"] > rel["row1"]]

    assert len(one_col_pairs) == 3
    assert len(bridge_pairs) == 2
    assert len(contrast_pairs) == 1

    expected = shifted_canvas.copy()
    bridge_cells = set()
    for rel in bridge_pairs:
        col = rel["left"]["bbox"][3] + 1
        for r in range(rel["row0"], rel["row1"] + 1):
            assert int(inp[r, col]) == BG
            if int(expected[r, col]) == BG:
                bridge_cells.add((r, col))
                expected[r, col] = BRIDGE

    assert bridge_cells, "bridging must add visible gray cells"
    out = main(inp)
    assert np.array_equal(out, expected), "solver output must match the independently checked rule"

    special_pairs = {(rel["left"]["rank"], rel["right"]["rank"]) for rel in one_col_pairs}
    assert special_pairs == set(SPECIAL_COLUMN_PAIRS)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_input_attempt(rng)
        if grid is None:
            continue
        try:
            _validate_semantics(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid emergent-gap-bridge input")
