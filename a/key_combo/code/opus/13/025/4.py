from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_PALETTE = (BLUE, RED, GREEN)

_GLYPHS = {
    BLUE: np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    RED: np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    GREEN: np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
}
_CODEBOOK = {
    tuple(tuple(bool(cell) for cell in row) for row in mask): color
    for color, mask in _GLYPHS.items()
}

_GEN_ATTEMPTS = 400
_PAIR_ATTEMPTS = 300
_GRID_MAX = 20


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in _N4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _decode_glyph(patch: np.ndarray) -> int:
    key = tuple(tuple(bool(cell) for cell in row) for row in (patch == GREY))
    if key not in _CODEBOOK:
        raise AssertionError("unrecognized 3x3 glyph")
    return _CODEBOOK[key]


def _extract_glyphs(grid: np.ndarray) -> list[dict]:
    components = _components_mask(grid == GREY)
    glyphs: list[dict] = []
    for coords in components:
        if coords.shape[0] != 5:
            raise AssertionError("each glyph must contain exactly five grey pixels")
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        if (r1 - r0, c1 - c0) != (2, 2):
            raise AssertionError("each glyph must occupy a 3x3 bbox")
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = _decode_glyph(patch)
        glyphs.append(
            {
                "bbox": bbox,
                "center": (r0 + 1, c0 + 1),
                "mask": _GLYPHS[color],
                "color": color,
            }
        )
    if len(glyphs) not in (6, 8):
        raise AssertionError("expected 6 or 8 glyphs")
    return glyphs


def _alignment(a: dict, b: dict) -> str | None:
    same_row = a["center"][0] == b["center"][0]
    same_col = a["center"][1] == b["center"][1]
    if same_row == same_col:
        return None
    return "row" if same_row else "col"


def _pair_glyphs(glyphs: list[dict]) -> list[tuple[int, str, dict, dict]]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for glyph in glyphs:
        by_color[glyph["color"]].append(glyph)

    pairs: list[tuple[int, str, dict, dict]] = []
    for color, items in by_color.items():
        if len(items) not in (2, 4):
            raise AssertionError("each decoded color must appear exactly 2 or 4 times")
        neighbors: dict[int, tuple[int, str]] = {}
        for i, glyph in enumerate(items):
            aligned = []
            for j, other in enumerate(items):
                if i == j:
                    continue
                axis = _alignment(glyph, other)
                if axis is not None:
                    aligned.append((j, axis))
            if len(aligned) != 1:
                raise AssertionError("same-color pairing must be unique")
            neighbors[i] = aligned[0]

        used: set[int] = set()
        for i, (j, axis) in neighbors.items():
            if neighbors.get(j) != (i, axis):
                raise AssertionError("pair alignment must be mutual")
            if i in used or j in used:
                continue
            used.update({i, j})
            pairs.append((color, axis, items[i], items[j]))
        if len(used) != len(items):
            raise AssertionError("every glyph must belong to exactly one pair")
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each grey 3x3 glyph, pair same-color glyphs by aligned centers, then connect and recolor them.

    Input structure:
    - The grid is a black background containing 6 or 8 disconnected 3x3 grey/black glyph bboxes.
    - Every glyph has exactly five grey pixels and matches one of three code shapes: top-T, down-right bend, or down-left bend.
    - Glyphs with the same decoded color occur in even multiplicity, and each glyph has exactly one same-color partner whose bbox center is aligned in either one row or one column.

    Rule:
    - Decode each glyph shape to its hidden color: top-T -> blue, down-right bend -> red, down-left bend -> green.
    - Pair glyphs by decoded color and the unique same-row or same-column center alignment.
    - For each pair, fill the background cells strictly between the two 3x3 bboxes along the shared center row or center column with the decoded color.
    - Recolor every grey pixel inside each glyph to that glyph's decoded color while keeping all black pixels black.

    Key insight:
    - The grey glyphs are encrypted endpoints: their shape determines both which other glyph they connect to and which color the connector must use.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    glyphs = _extract_glyphs(g)
    pairs = _pair_glyphs(glyphs)

    out = g.copy()
    for glyph in glyphs:
        r0, c0, r1, c1 = glyph["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[glyph["mask"]] = glyph["color"]

    for color, axis, a, b in pairs:
        if axis == "row":
            left, right = sorted((a, b), key=lambda item: item["bbox"][1])
            row = left["center"][0]
            start = left["bbox"][3] + 1
            stop = right["bbox"][1]
            out[row, start:stop] = color
        else:
            top, bottom = sorted((a, b), key=lambda item: item["bbox"][0])
            col = top["center"][1]
            start = top["bbox"][2] + 1
            stop = bottom["bbox"][0]
            out[start:stop, col] = color
    return out


def _expanded_bbox_cells(
    bbox: tuple[int, int, int, int], h: int, w: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            cells.append((r, c))
    return cells


def _sample_pair(h: int, w: int, axis: str) -> dict:
    if axis == "row":
        gap = random.randint(2, min(7, w - 6))
        left0 = random.randint(0, w - (6 + gap))
        row0 = random.randint(0, h - 3)
        right0 = left0 + 3 + gap
        bboxes = [(row0, left0, row0 + 2, left0 + 2), (row0, right0, row0 + 2, right0 + 2)]
        centers = [(row0 + 1, left0 + 1), (row0 + 1, right0 + 1)]
        path = [(row0 + 1, c) for c in range(left0 + 3, right0)]
    else:
        gap = random.randint(2, min(7, h - 6))
        top0 = random.randint(0, h - (6 + gap))
        col0 = random.randint(0, w - 3)
        bottom0 = top0 + 3 + gap
        bboxes = [(top0, col0, top0 + 2, col0 + 2), (bottom0, col0, bottom0 + 2, col0 + 2)]
        centers = [(top0 + 1, col0 + 1), (bottom0 + 1, col0 + 1)]
        path = [(r, col0 + 1) for r in range(top0 + 3, bottom0)]

    keepout = []
    for bbox in bboxes:
        keepout.extend(_expanded_bbox_cells(bbox, h, w))
    return {"axis": axis, "bboxes": bboxes, "centers": centers, "path": path, "keepout": keepout}


def _choose_pair_colors(n_pairs: int) -> list[int]:
    if n_pairs == 3:
        colors = list(_PALETTE)
        random.shuffle(colors)
        return colors
    dominant = random.choice(_PALETTE)
    others = [color for color in _PALETTE if color != dominant]
    random.shuffle(others)
    colors = [dominant, dominant, others[0], others[1]]
    random.shuffle(colors)
    return colors


def _choose_axes(n_pairs: int) -> list[str]:
    if random.random() < 0.65:
        row_count = random.randint(1, n_pairs - 1)
        axes = ["row"] * row_count + ["col"] * (n_pairs - row_count)
    else:
        axes = [random.choice(("row", "col"))] * n_pairs
    random.shuffle(axes)
    return axes


def _place_pairs(h: int, w: int, colors: list[int], axes: list[str]) -> list[dict] | None:
    keepout_mask = np.zeros((h, w), dtype=bool)
    path_mask = np.zeros((h, w), dtype=bool)
    used_rows = {color: set() for color in _PALETTE}
    used_cols = {color: set() for color in _PALETTE}
    placements: list[dict] = []

    specs = list(zip(colors, axes))
    random.shuffle(specs)
    for color, axis in specs:
        placed = None
        for _ in range(_PAIR_ATTEMPTS):
            candidate = _sample_pair(h, w, axis)
            rows = {r for r, _ in candidate["centers"]}
            cols = {c for _, c in candidate["centers"]}
            if rows & used_rows[color] or cols & used_cols[color]:
                continue
            if any(keepout_mask[r, c] or path_mask[r, c] for r, c in candidate["keepout"]):
                continue
            if any(keepout_mask[r, c] or path_mask[r, c] for r, c in candidate["path"]):
                continue
            placed = candidate
            break
        if placed is None:
            return None

        used_rows[color].update(r for r, _ in placed["centers"])
        used_cols[color].update(c for _, c in placed["centers"])
        for r, c in placed["keepout"]:
            keepout_mask[r, c] = True
        for r, c in placed["path"]:
            path_mask[r, c] = True
        placed["color"] = color
        placements.append(placed)
    return placements


def _render_input(h: int, w: int, placements: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for placement in placements:
        glyph = _GLYPHS[placement["color"]]
        for r0, c0, r1, c1 in placement["bboxes"]:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            patch[glyph] = GREY
    return grid


def _render_expected(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    for placement in placements:
        color = placement["color"]
        glyph = _GLYPHS[color]
        for r0, c0, r1, c1 in placement["bboxes"]:
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[glyph] = color
        for r, c in placement["path"]:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = 4 if random.random() < 0.6 else 3
    colors = _choose_pair_colors(n_pairs)
    axes = _choose_axes(n_pairs)

    base_min = 12 if n_pairs == 3 else 14
    h_min = max(base_min, 4 * axes.count("row") - 1, 8 if "col" in axes else 0)
    w_min = max(base_min, 4 * axes.count("col") - 1, 8 if "row" in axes else 0)
    if h_min > _GRID_MAX or w_min > _GRID_MAX:
        return None

    h = random.randint(h_min, _GRID_MAX)
    w = random.randint(w_min, _GRID_MAX)
    placements = _place_pairs(h, w, colors, axes)
    if placements is None:
        return None

    inp = _render_input(h, w, placements)
    expected = _render_expected(inp, placements)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid glyph-pairing input")
