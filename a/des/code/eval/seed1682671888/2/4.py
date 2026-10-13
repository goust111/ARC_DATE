from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        if seen[start_r, start_c]:
            continue
        stack = [(int(start_r), int(start_c))]
        seen[start_r, start_c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _proportional_offset(edge_index: int, parent_len: int, child_len: int) -> int:
    slack = parent_len - child_len
    assert slack >= 0, "child must fit inside parent"
    if slack == 0 or parent_len <= 1:
        return 0
    return int(round(edge_index * slack / (parent_len - 1)))


def _placement_from_marker(parent: dict, child: dict) -> tuple[int, int]:
    side = parent["marker_side"]
    edge_index = parent["marker_index"]
    parent_h = parent["h"]
    parent_w = parent["w"]
    child_h = child["h"]
    child_w = child["w"]

    if side == "top":
        return 0, _proportional_offset(edge_index, parent_w, child_w)
    if side == "bottom":
        return parent_h - child_h, _proportional_offset(edge_index, parent_w, child_w)
    if side == "left":
        return _proportional_offset(edge_index, parent_h, child_h), 0
    if side == "right":
        return _proportional_offset(edge_index, parent_h, child_h), parent_w - child_w
    raise AssertionError("unknown marker side")


def _extract_pieces(grid: np.ndarray) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    pieces: dict[int, dict] = {}

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue

        comps = _components(g == color)
        assert comps, "each non-background color must appear at least once"
        areas = [int(comp.shape[0]) for comp in comps]
        largest = max(areas)
        assert areas.count(largest) == 1, "each piece body must be the unique largest component of its color"
        body = comps[areas.index(largest)]
        r0, c0, r1, c1 = _bbox(body)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]

        assert np.all(patch != bg), "piece bbox must be fully filled"
        foreign = np.argwhere(patch != color)
        assert len(foreign) <= 1, "piece bbox may contain at most one marker"

        marker_color = None
        marker_side = None
        marker_index = None
        if len(foreign) == 1:
            mr, mc = map(int, foreign[0])
            marker_color = int(patch[mr, mc])
            side_hits = []
            if mr == 0:
                side_hits.append("top")
            if mr == patch.shape[0] - 1:
                side_hits.append("bottom")
            if mc == 0:
                side_hits.append("left")
            if mc == patch.shape[1] - 1:
                side_hits.append("right")
            assert len(side_hits) == 1, "marker must lie on exactly one non-corner edge"
            marker_side = side_hits[0]
            marker_index = mc if marker_side in ("top", "bottom") else mr

        pieces[color] = {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "h": int(r1 - r0 + 1),
            "w": int(c1 - c0 + 1),
            "marker_color": marker_color,
            "marker_side": marker_side,
            "marker_index": marker_index,
        }

    assert pieces, "expected at least one piece"
    return pieces


def _chain_order(pieces: dict[int, dict]) -> list[int]:
    colors = set(pieces)
    referenced: list[int] = []

    for piece in pieces.values():
        child = piece["marker_color"]
        if child is None:
            continue
        assert child in pieces, "marker color must reference another piece body color"
        referenced.append(int(child))

    counts = Counter(referenced)
    assert all(count == 1 for count in counts.values()), "each piece can only be referenced once"

    roots = sorted(colors - set(referenced))
    leaves = sorted(color for color, piece in pieces.items() if piece["marker_color"] is None)
    assert len(roots) == 1, "expected exactly one outermost base piece"
    assert len(leaves) == 1, "expected exactly one innermost core piece"

    order = []
    current = roots[0]
    seen: set[int] = set()
    while True:
        assert current not in seen, "nesting chain must be acyclic"
        seen.add(current)
        order.append(current)
        child = pieces[current]["marker_color"]
        if child is None:
            break
        current = int(child)

    assert len(order) == len(pieces), "all pieces must lie on one reference chain"
    return order


def _assemble_from_pieces(pieces: dict[int, dict], order: list[int]) -> np.ndarray:
    root = pieces[order[0]]
    out = np.full((root["h"], root["w"]), int(root["color"]), dtype=int)
    anchors = {order[0]: (0, 0)}

    for parent_color, child_color in zip(order, order[1:]):
        parent = pieces[parent_color]
        child = pieces[child_color]
        parent_r, parent_c = anchors[parent_color]
        dr, dc = _placement_from_marker(parent, child)
        child_r = parent_r + dr
        child_c = parent_c + dc
        anchors[child_color] = (child_r, child_c)
        out[child_r : child_r + child["h"], child_c : child_c + child["w"]] = int(child["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular fragments into a nested stack.

    Each piece has a unique body color and may carry one edge marker whose color
    names the next inner piece. The marker side chooses which side of the child
    touches the parent, while the marker's offset along that edge is projected
    proportionally onto the remaining placement slack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pieces = _extract_pieces(g)
    order = _chain_order(pieces)
    return _assemble_from_pieces(pieces, order)


# --------------- generator ---------------

COLOR_VALUES = tuple(range(10))
GRID_MIN = 20
GRID_MAX = 30
PIECE_COUNT_RANGE = (3, 5)
CORE_SIZE_RANGE = (2, 4)
GROWTH_RANGE = (1, 3)
MAX_OUTER_SIZE = 12
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _random_chain_dims(n_pieces: int) -> list[tuple[int, int]] | None:
    h = random.randint(*CORE_SIZE_RANGE)
    w = random.randint(*CORE_SIZE_RANGE)
    dims = [(h, w)]

    for _ in range(n_pieces - 1):
        h += random.randint(*GROWTH_RANGE)
        w += random.randint(*GROWTH_RANGE)
        if h > MAX_OUTER_SIZE or w > MAX_OUTER_SIZE:
            return None
        dims.append((h, w))

    dims.reverse()
    return dims


def _marker_spec(parent_h: int, parent_w: int) -> tuple[str, int]:
    sides = []
    if parent_w >= 3:
        sides.extend(["top", "bottom"])
    if parent_h >= 3:
        sides.extend(["left", "right"])
    side = random.choice(sides)
    if side in ("top", "bottom"):
        return side, random.randint(1, parent_w - 2)
    return side, random.randint(1, parent_h - 2)


def _piece_patch(piece: dict, child_color: int | None) -> np.ndarray:
    patch = np.full((piece["h"], piece["w"]), int(piece["color"]), dtype=int)
    if child_color is None:
        return patch

    side = piece["marker_side"]
    idx = piece["marker_index"]
    if side == "top":
        patch[0, idx] = int(child_color)
    elif side == "bottom":
        patch[-1, idx] = int(child_color)
    elif side == "left":
        patch[idx, 0] = int(child_color)
    elif side == "right":
        patch[idx, -1] = int(child_color)
    else:
        raise AssertionError("unknown marker side")
    return patch


def _place_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    total_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    side_floor = int(np.ceil(np.sqrt(total_area * 2.6)))
    min_h = max(GRID_MIN, max_h + 2, side_floor)
    min_w = max(GRID_MIN, max_w + 2, side_floor)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(height, r0 + ph + PLACEMENT_PADDING)
            cc1 = min(width, c0 + pw + PLACEMENT_PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _specs_to_pieces(colors: list[int], dims: list[tuple[int, int]]) -> dict[int, dict]:
    pieces: dict[int, dict] = {}
    for idx, (color, (h, w)) in enumerate(zip(colors, dims)):
        marker_color = None if idx == len(colors) - 1 else int(colors[idx + 1])
        marker_side = None
        marker_index = None
        if marker_color is not None:
            marker_side, marker_index = _marker_spec(h, w)
        pieces[int(color)] = {
            "color": int(color),
            "bbox": (0, 0, h - 1, w - 1),
            "h": int(h),
            "w": int(w),
            "marker_color": marker_color,
            "marker_side": marker_side,
            "marker_index": marker_index,
        }
    return pieces


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_VALUES)
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample([c for c in COLOR_VALUES if c != bg], n_pieces)
    dims = _random_chain_dims(n_pieces)
    if dims is None:
        return None

    pieces = _specs_to_pieces(colors, dims)
    order = colors[:]
    expected = _assemble_from_pieces(pieces, order)

    patches = []
    for idx, color in enumerate(order):
        child_color = None if idx == len(order) - 1 else int(order[idx + 1])
        patches.append(_piece_patch(pieces[int(color)], child_color))

    grid = _place_patches(patches, bg)
    if grid is None:
        return None

    bg_count = int(np.sum(grid == bg))
    if bg_count <= max(piece["h"] * piece["w"] for piece in pieces.values()):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
