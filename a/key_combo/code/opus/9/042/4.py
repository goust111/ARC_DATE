from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
INTERIOR_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SIDE_DELTAS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

FRAME_COUNT = 3
GRID_MIN = 18
GRID_MAX = 25  # exclusive
FRAME_MIN = 5
FRAME_MAX = 8  # exclusive
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _gap_side_and_delta(
    gap: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top", SIDE_DELTAS["top"]
    if gr == r1 and c0 < gc < c1:
        return "bottom", SIDE_DELTAS["bottom"]
    if gc == c0 and r0 < gr < r1:
        return "left", SIDE_DELTAS["left"]
    if gc == c1 and r0 < gr < r1:
        return "right", SIDE_DELTAS["right"]
    raise AssertionError("gap must be a non-corner border cell")


def _interior_diag_cells(
    bbox: tuple[int, int, int, int], diag_sum: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for r in range(r0 + 1, r1):
        c = diag_sum - r
        if c0 < c < c1:
            cells.append((r, c))
    return cells


def _ray_cells(
    shape: tuple[int, int], gap: tuple[int, int], side: str
) -> list[tuple[int, int]]:
    dr, dc = SIDE_DELTAS[side]
    r, c = gap[0] + dr, gap[1] + dc
    cells = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components_mask(grid == BLUE):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must be at least 5x5"
        border = _border_cells(r0, c0, r1, c1)
        coord_set = {tuple(map(int, rc)) for rc in coords}
        assert coord_set <= border, "blue component must lie on its bbox border"
        gaps = list(border - coord_set)
        assert len(gaps) == 1, "expected exactly one missing border cell"
        gap = gaps[0]
        assert coord_set == border - {gap}, "blue cells must form a one-gap rectangle"
        side, delta = _gap_side_and_delta(gap, bbox)

        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        colored_coords = np.argwhere((inside != BLACK) & (inside != BLUE))
        assert len(colored_coords) >= 2, "need at least two visible interior clues"
        colored_global = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in colored_coords]
        colors = {int(grid[r, c]) for r, c in colored_global}
        assert len(colors) == 1, "interior clues must share one color"
        diag_sums = {r + c for r, c in colored_global}
        assert len(diag_sums) == 1, "interior clues must lie on one anti-diagonal"

        color = next(iter(colors))
        diag_sum = next(iter(diag_sums))
        frames.append(
            {
                "bbox": bbox,
                "gap": gap,
                "side": side,
                "delta": delta,
                "color": color,
                "diag_sum": diag_sum,
            }
        )

    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each one-gap blue frame's hidden anti-diagonal cells, then shoot that color outward through the gap.

    Input structure:
    - The grid has a black background and several blue rectangular frame borders, each with exactly one missing non-corner border pixel.
    - Inside each frame, all non-black non-blue clue pixels share one color and lie on one global anti-diagonal.
    - Some cells on that same anti-diagonal inside the frame may still be black, while all other interior cells are black.

    Rule:
    - For each blue frame, infer the unique clue color C and anti-diagonal sum D from the visible interior pixels.
    - Fill every black interior cell whose global coordinates satisfy row+col=D with color C, and leave all other interior black cells unchanged.
    - Restore the missing border pixel to blue, then from the cell immediately outside that gap extend a straight outward ray in color C until the grid edge.

    Key insight:
    - The interior anti-diagonal fixes what color must be completed inside the frame, while the border gap independently fixes the direction in which that recovered color must propagate outside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    frames = _extract_frames(g)
    assert frames, "expected at least one blue frame"

    out = g.copy()
    for frame in frames:
        bbox = frame["bbox"]
        color = int(frame["color"])
        diag_sum = int(frame["diag_sum"])
        gap = frame["gap"]
        side = frame["side"]

        diag_cells = _interior_diag_cells(bbox, diag_sum)
        assert len(diag_cells) >= 2, "anti-diagonal inside frame must be non-trivial"
        for r, c in diag_cells:
            if out[r, c] == BLACK:
                out[r, c] = color
            else:
                assert int(out[r, c]) == color, "visible clue color must match the inferred fill"

        out[gap] = BLUE
        for r, c in _ray_cells(g.shape, gap, side):
            assert int(g[r, c]) == BLACK, "ray must extend only through background in the input"
            out[r, c] = color

    return out


def _expand_box(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _boxes_disjoint(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _gap_cell(r0: int, c0: int, h: int, w: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r0 + h - 1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c0 + w - 1


def _diag_options(
    bbox: tuple[int, int, int, int], min_len: int
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    buckets: dict[int, list[tuple[int, int]]] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            buckets.setdefault(r + c, []).append((r, c))
    return [sorted(cells) for cells in buckets.values() if len(cells) >= min_len]


def _visibility_split(
    rng: np.random.Generator, diag_cells: list[tuple[int, int]], mode: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    n = len(diag_cells)
    assert n >= 3
    hidden: set[int] = set()
    if mode == "endpoint":
        hidden.add(0 if int(rng.integers(0, 2)) == 0 else n - 1)
        if n >= 4 and float(rng.random()) < 0.35:
            extra_choices = [i for i in range(1, n - 1)]
            hidden.add(int(rng.choice(extra_choices)))
    elif mode == "middle":
        hidden.add(int(rng.integers(1, n - 1)))
        if n >= 5 and float(rng.random()) < 0.35:
            extra_choices = [i for i in range(1, n - 1) if i not in hidden]
            hidden.add(int(rng.choice(extra_choices)))
    else:
        assert mode == "mixed"
        hidden.add(0 if int(rng.integers(0, 2)) == 0 else n - 1)
        hidden.add(int(rng.integers(1, n - 1)))
        if n >= 5 and float(rng.random()) < 0.5:
            extra_choices = [i for i in range(n) if i not in hidden]
            candidate = int(rng.choice(extra_choices))
            if n - (len(hidden) + 1) >= 2:
                hidden.add(candidate)

    visible = [diag_cells[i] for i in range(n) if i not in hidden]
    hidden_cells = [diag_cells[i] for i in range(n) if i in hidden]
    assert len(visible) >= 2 and len(hidden_cells) >= 1
    return visible, hidden_cells


def _sample_frame_spec(
    rng: np.random.Generator,
    shape: tuple[int, int],
    side: str,
    mode: str,
    color: int,
) -> dict | None:
    h, w = shape
    min_hw = 6 if mode == "mixed" else 5
    fh = int(rng.integers(min_hw, FRAME_MAX))
    fw = int(rng.integers(min_hw, FRAME_MAX))
    if h - fh - 1 <= 1 or w - fw - 1 <= 1:
        return None

    r0 = int(rng.integers(1, h - fh))
    c0 = int(rng.integers(1, w - fw))
    bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
    offset_max = fw - 2 if side in {"top", "bottom"} else fh - 2
    offset = int(rng.integers(1, offset_max + 1))
    gap = _gap_cell(r0, c0, fh, fw, side, offset)
    ray = _ray_cells(shape, gap, side)
    if not ray:
        return None

    diag_cells_choices = _diag_options(bbox, 4 if mode == "mixed" else 3)
    if not diag_cells_choices:
        return None
    diag_cells = diag_cells_choices[int(rng.integers(0, len(diag_cells_choices)))]
    visible, hidden = _visibility_split(rng, diag_cells, mode)
    return {
        "bbox": bbox,
        "pad_bbox": _expand_box(bbox, shape, 1),
        "bbox_cells": _box_cells(bbox),
        "pad_cells": _box_cells(_expand_box(bbox, shape, 1)),
        "gap": gap,
        "side": side,
        "ray": ray,
        "ray_set": set(ray),
        "color": int(color),
        "diag_sum": sum(diag_cells[0]),
        "diag_cells": diag_cells,
        "visible": visible,
        "hidden": hidden,
    }


def _compatible(spec: dict, placed: list[dict]) -> bool:
    for other in placed:
        if not _boxes_disjoint(spec["pad_bbox"], other["pad_bbox"]):
            return False
        if spec["ray_set"] & other["pad_cells"]:
            return False
        if other["ray_set"] & spec["pad_cells"]:
            return False
        if spec["ray_set"] & other["ray_set"]:
            return False
    return True


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        gr, gc = spec["gap"]
        grid[gr, gc] = BLACK
        for r, c in spec["visible"]:
            grid[r, c] = spec["color"]
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["diag_cells"]:
            out[r, c] = spec["color"]
        gr, gc = spec["gap"]
        out[gr, gc] = BLUE
        for r, c in spec["ray"]:
            out[r, c] = spec["color"]
    return out


def _validate_semantics(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(inp.max(initial=0)) > 9 or int(inp.min(initial=0)) < 0:
        return False

    try:
        frames = _extract_frames(inp)
    except AssertionError:
        return False

    if len(frames) != FRAME_COUNT:
        return False

    colors = set()
    sides = set()
    saw_hidden_endpoint = False
    saw_hidden_middle = False
    rays: list[set[tuple[int, int]]] = []

    for frame in frames:
        bbox = frame["bbox"]
        gap = frame["gap"]
        side = frame["side"]
        color = frame["color"]
        diag_sum = frame["diag_sum"]

        if color in colors:
            return False
        colors.add(color)
        sides.add(side)

        diag_cells = _interior_diag_cells(bbox, diag_sum)
        colored = [cell for cell in diag_cells if int(inp[cell]) == color]
        hidden = [cell for cell in diag_cells if int(inp[cell]) == BLACK]
        if len(colored) < 2 or len(hidden) < 1:
            return False

        if hidden and hidden[0] in {diag_cells[0], diag_cells[-1]}:
            saw_hidden_endpoint = True
        if any(cell not in {diag_cells[0], diag_cells[-1]} for cell in hidden):
            saw_hidden_middle = True

        r0, c0, r1, c1 = bbox
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                v = int(inp[r, c])
                if (r, c) in diag_cells:
                    if v not in {BLACK, color}:
                        return False
                elif v != BLACK:
                    return False

        ray_set = set(_ray_cells(inp.shape, gap, side))
        if not ray_set:
            return False
        if any(int(inp[r, c]) != BLACK for r, c in ray_set):
            return False
        rays.append(ray_set)

    if len(sides) < 2 or not saw_hidden_endpoint or not saw_hidden_middle:
        return False

    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            if rays[i] & rays[j]:
                return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    shape = (int(rng.integers(GRID_MIN, GRID_MAX)), int(rng.integers(GRID_MIN, GRID_MAX)))
    colors = [int(v) for v in rng.choice(np.array(INTERIOR_COLORS), size=FRAME_COUNT, replace=False)]
    sides = list(rng.choice(np.array(SIDES), size=FRAME_COUNT, replace=False))
    modes = ["endpoint", "middle", "mixed"]
    rng.shuffle(colors)
    rng.shuffle(sides)
    rng.shuffle(modes)

    specs: list[dict] = []
    for color, side, mode in zip(colors, sides, modes, strict=False):
        placed = False
        for _ in range(120):
            spec = _sample_frame_spec(rng, shape, side, mode, color)
            if spec is None:
                continue
            if _compatible(spec, specs):
                specs.append(spec)
                placed = True
                break
        if not placed:
            return None

    inp = _render_input(shape, specs)
    if not _validate_semantics(inp):
        return None

    expected = _expected_output(inp, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anti-diagonal gap-ray puzzle")
