from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
EDGES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def _border_side(cell: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = cell
    h, w = shape
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, f"anchor must lie on exactly one border side, got {sides}"
    return sides[0]


def _segment_length(component: dict) -> int:
    coords = component["coords"]
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if len(rows) == 1:
        sorted_cols = np.sort(coords[:, 1])
        expected = np.arange(int(sorted_cols[0]), int(sorted_cols[-1]) + 1)
        assert np.array_equal(sorted_cols, expected), "horizontal segment must be contiguous"
    elif len(cols) == 1:
        sorted_rows = np.sort(coords[:, 0])
        expected = np.arange(int(sorted_rows[0]), int(sorted_rows[-1]) + 1)
        assert np.array_equal(sorted_rows, expected), "vertical segment must be contiguous"
    else:
        raise AssertionError("non-anchor component must be a straight line segment")

    return int(component["area"])


def _direction_for_edge(edge: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[edge]


def _extract_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    components = _find_components(g, bg=BG)
    assert components, "expected at least one colored object"

    by_color: dict[int, dict] = {}
    for comp in components:
        color = int(comp["color"])
        group = by_color.setdefault(color, {"anchor": None, "lengths": []})
        area = int(comp["area"])

        if area == 1:
            cell = tuple(map(int, comp["coords"][0]))
            side = _border_side(cell, (h, w))
            assert group["anchor"] is None, f"expected one anchor for color {color}"
            group["anchor"] = {"cell": cell, "edge": side}
            continue

        length = _segment_length(comp)
        assert 2 <= length, "fragments must be longer than anchors"
        group["lengths"].append(length)

    groups: list[dict] = []
    for color, group in by_color.items():
        assert group["anchor"] is not None, f"missing anchor for color {color}"
        assert 2 <= len(group["lengths"]) <= 4, f"expected 2-4 fragments for color {color}"
        groups.append(
            {
                "color": int(color),
                "anchor": group["anchor"]["cell"],
                "edge": group["anchor"]["edge"],
                "lengths": sorted(int(v) for v in group["lengths"]),
            }
        )

    return groups


def _render_output(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)

    for group in groups:
        color = int(group["color"])
        r, c = group["anchor"]
        dr, dc = _direction_for_edge(group["edge"])

        assert out[r, c] == BG, "anchors must not overlap"
        out[r, c] = color

        rr, cc = r, c
        for length in group["lengths"]:
            for _ in range(int(length)):
                rr += dr
                cc += dc
                assert 0 <= rr < h and 0 <= cc < w, "assembled line exceeds canvas"
                assert out[rr, cc] == BG, "assembled lines must not overlap"
                out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered fragments into one straight line growing inward from its edge anchor.

    Input structure:
    - The background is black.
    - Several colors each have one single-pixel anchor on exactly one grid edge.
    - Away from the border, every color also has 2-4 disconnected horizontal or vertical line fragments of length at least 2.

    Rule:
    - Group fragments by color and find that color's unique edge anchor.
    - Sort each color's fragment lengths from short to long.
    - Starting at the anchor, extend in the direction perpendicular to the anchor edge and place the sorted lengths back-to-back with no gaps.
    - Remove the original scattered fragments and keep only the rebuilt straight lines.

    Key insight:
    - The anchor determines both which fragments belong together and which inward direction their lengths must be laid out along; the fragments' original positions and orientations are irrelevant apart from their lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    groups = _extract_groups(g)
    return _render_output(g.shape, groups)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _expand_with_margin(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _line_cells_from_spec(spec: dict) -> list[tuple[int, int]]:
    r, c = spec["anchor"]
    dr, dc = _direction_for_edge(spec["edge"])
    cells = [(r, c)]
    rr, cc = r, c
    for length in spec["lengths"]:
        for _ in range(int(length)):
            rr += dr
            cc += dc
            cells.append((rr, cc))
    return cells


def _random_lengths(axis_len: int) -> list[int]:
    count = random.randint(2, 4)
    pool = [2, 3, 4, 5]
    random.shuffle(pool)
    lengths = sorted(pool[:count])
    assert 1 + sum(lengths) <= axis_len
    return lengths


def _candidate_anchor_positions(edge: str, h: int, w: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, c) for c in range(1, w - 1)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if edge == "left":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _sample_specs(h: int, w: int, colors: list[int]) -> list[dict] | None:
    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    for color in colors:
        placed = False
        edges = list(EDGES)
        random.shuffle(edges)

        for edge in edges:
            axis_len = h if edge in ("top", "bottom") else w
            lengths = _random_lengths(axis_len)
            candidates = _candidate_anchor_positions(edge, h, w)
            random.shuffle(candidates)

            for anchor in candidates:
                spec = {"color": int(color), "edge": edge, "anchor": anchor, "lengths": lengths}
                cells = _line_cells_from_spec(spec)
                if any(occupied[r, c] for r, c in cells):
                    continue
                for r, c in cells:
                    occupied[r, c] = True
                specs.append(spec)
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    return specs


def _placement_cells(
    start: tuple[int, int], orientation: str, length: int
) -> list[tuple[int, int]]:
    r, c = start
    if orientation == "h":
        return [(r, c + offset) for offset in range(length)]
    return [(r + offset, c) for offset in range(length)]


def _segment_candidates(
    h: int, w: int, blocked: np.ndarray, length: int, orientation: str
) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    if orientation == "h":
        for r in range(1, h - 1):
            for c in range(1, w - length):
                cells = _placement_cells((r, c), orientation, length)
                if all(not blocked[rr, cc] for rr, cc in cells):
                    candidates.append(cells)
    else:
        for r in range(1, h - length):
            for c in range(1, w - 1):
                cells = _placement_cells((r, c), orientation, length)
                if all(not blocked[rr, cc] for rr, cc in cells):
                    candidates.append(cells)
    return candidates


def _build_input_from_specs(h: int, w: int, specs: list[dict]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for spec in specs:
        cell = [spec["anchor"]]
        _paint_cells(grid, cell, int(spec["color"]))
        _expand_with_margin(blocked, cell)

    fragments: list[tuple[int, int]] = []
    for spec in specs:
        for length in spec["lengths"]:
            fragments.append((int(spec["color"]), int(length)))
    random.shuffle(fragments)

    for color, length in fragments:
        orientations = ["h", "v"]
        random.shuffle(orientations)
        placed = False

        for orientation in orientations:
            candidates = _segment_candidates(h, w, blocked, length, orientation)
            if not candidates:
                continue
            cells = random.choice(candidates)
            _paint_cells(grid, cells, color)
            _expand_with_margin(blocked, cells)
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    color_count = random.randint(2, 4)
    colors = random.sample(COLORS, color_count)

    specs = _sample_specs(h, w, colors)
    if specs is None:
        return None

    grid = _build_input_from_specs(h, w, specs)
    if grid is None:
        return None

    expected = _render_output((h, w), specs)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
