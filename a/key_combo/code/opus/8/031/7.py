from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
PALETTE = tuple(range(1, 10))
GRID_MIN = 17
GRID_MAX = 26
RECT_H_RANGE = (4, 7)
RECT_W_RANGE = (5, 8)
MIN_GAP = 2
MAX_SHAPE_ATTEMPTS = 300
MAX_LAYOUT_ATTEMPTS = 400
MAX_GENERATION_ATTEMPTS = 600
EDGE_CONFIGS = [
    ("top", "left"),
    ("bottom",),
    ("right",),
    ("left", "bottom"),
    ("top",),
    ("right", "bottom"),
    ("left",),
    ("top", "right"),
]
_edge_config_index = 0


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    shifted = arr - np.array([r0, c0], dtype=int)
    h, w = shifted.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask, (int(r0), int(c0))


def _mask_coords(mask: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = origin
    return [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]


def _edge_of_component(coords: np.ndarray, shape: tuple[int, int]) -> str | None:
    arr = np.asarray(coords, dtype=int)
    h, w = shape
    touches: list[str] = []
    if int(arr[:, 0].min()) == 0:
        touches.append("top")
    if int(arr[:, 0].max()) == h - 1:
        touches.append("bottom")
    if int(arr[:, 1].min()) == 0:
        touches.append("left")
    if int(arr[:, 1].max()) == w - 1:
        touches.append("right")
    if len(touches) != 1:
        return None
    return touches[0]


def _flip_mask(mask: np.ndarray, edge: str) -> np.ndarray:
    if edge in {"top", "bottom"}:
        return np.flipud(mask)
    if edge in {"left", "right"}:
        return np.fliplr(mask)
    raise ValueError(f"unknown edge {edge}")


def _wrong_flip_mask(mask: np.ndarray, edge: str) -> np.ndarray:
    if edge in {"top", "bottom"}:
        return np.fliplr(mask)
    return np.flipud(mask)


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    seen = {tuple(map(int, cells[0]))}
    stack = [tuple(map(int, cells[0]))]
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                cell = (rr, cc)
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == int(mask.sum())


def _grow_polyomino(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))
    arr = np.array(sorted(cells), dtype=int)
    arr -= arr.min(axis=0)
    hh, ww = arr.max(axis=0) + 1
    mask = np.zeros((int(hh), int(ww)), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _placement_fits(mask: np.ndarray, top: int, left: int, blocked: np.ndarray) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > blocked.shape[0] or left + w > blocked.shape[1]:
        return False
    patch = blocked[top : top + h, left : left + w]
    return not np.any(mask & patch)


def _mark_blocked(mask: np.ndarray, top: int, left: int, blocked: np.ndarray, pad: int = 1) -> None:
    rows = [top + int(r) for r, _c in np.argwhere(mask)]
    cols = [left + int(c) for _r, c in np.argwhere(mask)]
    r0 = max(0, min(rows) - pad)
    r1 = min(blocked.shape[0] - 1, max(rows) + pad)
    c0 = max(0, min(cols) - pad)
    c1 = min(blocked.shape[1] - 1, max(cols) + pad)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _shape_completion_positions(
    grid: np.ndarray,
    moved_mask: np.ndarray,
    anchor_coords: frozenset[tuple[int, int]],
    bg: int,
) -> list[frozenset[tuple[int, int]]]:
    local = np.argwhere(moved_mask)
    h, w = moved_mask.shape
    gh, gw = grid.shape
    out: list[frozenset[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for top in range(gh - h + 1):
        for left in range(gw - w + 1):
            placed = frozenset((top + int(r), left + int(c)) for r, c in local)
            if any(int(grid[r, c]) != int(bg) for r, c in placed):
                continue
            union = placed | anchor_coords
            rows = [r for r, _c in union]
            cols = [c for _r, c in union]
            rect_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
            if len(union) == rect_area and placed not in seen:
                out.append(placed)
                seen.add(placed)
    return out


def _solve_actions(grid: np.ndarray) -> list[tuple[int, frozenset[tuple[int, int]], frozenset[tuple[int, int]]]]:
    bg = get_bg(grid)
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    assert 1 <= len(by_color) <= 2, "expected 1-2 foreground colors"
    actions: list[tuple[int, frozenset[tuple[int, int]], frozenset[tuple[int, int]]]] = []
    for color in sorted(by_color):
        parts = by_color[color]
        assert len(parts) == 2, "each color must have exactly two components"
        edge_parts = []
        anchor_parts = []
        for part in parts:
            edge = _edge_of_component(part["coords"], grid.shape)
            if edge is None:
                anchor_parts.append(part)
            else:
                edge_parts.append((part, edge))
        assert len(edge_parts) == 1 and len(anchor_parts) == 1, "expected one edge fragment and one inner anchor"
        fragment, edge = edge_parts[0]
        anchor = anchor_parts[0]
        source_mask, _ = _coords_to_mask(fragment["coords"])
        moved_mask = _flip_mask(source_mask, edge)
        anchor_coords = frozenset((int(r), int(c)) for r, c in anchor["coords"])
        candidates = _shape_completion_positions(grid, moved_mask, anchor_coords, bg)
        assert len(candidates) == 1, "expected a unique mirror-completion placement"
        source_coords = frozenset((int(r), int(c)) for r, c in fragment["coords"])
        actions.append((color, source_coords, candidates[0]))
    return actions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each edge fragment toward its same-colored anchor and move it to complete a solid rectangle.

    Input structure:
    - The background is black.
    - There are 1 or 2 foreground colors, and each such color forms exactly two 4-connected objects.
    - One object of each color touches exactly one grid edge and serves as a fragment, while the other same-colored object stays away from all edges as an anchor.

    Rule:
    - For each color, detect which edge the fragment touches.
    - If it touches the top or bottom edge, flip the fragment vertically; if it touches the left or right edge, flip it horizontally.
    - Move the flipped fragment to the unique background location where it and the anchor together form one filled axis-aligned rectangle, then erase the fragment from its original edge position.

    Key insight:
    - The fragment is not translated directly: its edge contact tells you which mirror flip must happen first, and only that flipped version can complete the anchor into a rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    actions = _solve_actions(g)
    out = g.copy()
    bg = get_bg(g)
    for _color, source_coords, _target_coords in actions:
        for r, c in source_coords:
            out[r, c] = int(bg)
    for color, _source_coords, target_coords in actions:
        for r, c in target_coords:
            assert int(out[r, c]) == int(bg), "target placement must land on background"
            out[r, c] = int(color)
    return out


def _choose_edge_config() -> tuple[str, ...]:
    global _edge_config_index
    config = EDGE_CONFIGS[_edge_config_index % len(EDGE_CONFIGS)]
    _edge_config_index += 1
    return config


def _sample_fragment_and_anchor(edge: str) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        frag_h = random.randint(2, rect_h - 1)
        frag_w = random.randint(2, rect_w - 1)
        max_area = frag_h * frag_w - 1
        min_area = max(2, (frag_h * frag_w) // 3)
        area = random.randint(min_area, max_area)
        fragment = _grow_polyomino(frag_h, frag_w, area)
        fh, fw = fragment.shape
        if fh > rect_h or fw > rect_w:
            continue
        if np.array_equal(fragment, _flip_mask(fragment, edge)):
            continue
        top = random.randint(0, rect_h - fh)
        left = random.randint(0, rect_w - fw)
        placed = np.zeros((rect_h, rect_w), dtype=bool)
        placed[top : top + fh, left : left + fw] = fragment
        anchor = ~placed
        if not _is_connected(fragment):
            continue
        if not _is_connected(anchor):
            continue
        if np.array_equal(fragment, _wrong_flip_mask(fragment, edge)):
            continue
        return fragment, anchor
    raise RuntimeError("failed to sample a valid fragment/anchor split")


def _random_inner_origin(blocked: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    tops = list(range(2, blocked.shape[0] - h - 1))
    lefts = list(range(2, blocked.shape[1] - w - 1))
    random.shuffle(tops)
    random.shuffle(lefts)
    for top in tops:
        for left in lefts:
            mask = np.ones((h, w), dtype=bool)
            if _placement_fits(mask, top, left, blocked):
                return top, left
    return None


def _edge_origins(edge: str, grid_shape: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    origins: list[tuple[int, int]] = []
    if edge == "top":
        for left in range(1, gw - w - 1):
            origins.append((0, left))
    elif edge == "bottom":
        for left in range(1, gw - w - 1):
            origins.append((gh - h, left))
    elif edge == "left":
        for top in range(1, gh - h - 1):
            origins.append((top, 0))
    elif edge == "right":
        for top in range(1, gh - h - 1):
            origins.append((top, gw - w))
    random.shuffle(origins)
    return origins


def _build_case(edges: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    num_colors = len(edges)
    gh = random.randint(max(GRID_MIN, 16 + 2 * num_colors), GRID_MAX)
    gw = random.randint(max(GRID_MIN, 16 + 2 * num_colors), GRID_MAX)
    colors = random.sample(PALETTE, num_colors)
    items: list[dict] = []
    blocked = np.zeros((gh, gw), dtype=bool)

    for color, edge in zip(colors, edges):
        fragment, anchor = _sample_fragment_and_anchor(edge)
        rect_h, rect_w = anchor.shape
        anchor_origin = _random_inner_origin(blocked, rect_h, rect_w)
        if anchor_origin is None:
            return None
        source_mask = _flip_mask(fragment, edge)
        source_origin = None
        for origin in _edge_origins(edge, (gh, gw), source_mask.shape[0], source_mask.shape[1]):
            if _placement_fits(source_mask, origin[0], origin[1], blocked):
                source_origin = origin
                break
        if source_origin is None:
            return None
        items.append(
            {
                "color": color,
                "edge": edge,
                "fragment": fragment,
                "anchor": anchor,
                "anchor_origin": anchor_origin,
                "source_mask": source_mask,
                "source_origin": source_origin,
            }
        )
        _mark_blocked(np.ones((rect_h, rect_w), dtype=bool), anchor_origin[0], anchor_origin[1], blocked, pad=MIN_GAP)
        _mark_blocked(source_mask, source_origin[0], source_origin[1], blocked, pad=MIN_GAP)

    inp = np.zeros((gh, gw), dtype=int)
    out = np.zeros((gh, gw), dtype=int)
    for item in items:
        color = int(item["color"])
        anchor = item["anchor"]
        rect_h, rect_w = anchor.shape
        ar, ac = item["anchor_origin"]
        out[ar : ar + rect_h, ac : ac + rect_w] = color
        inp[ar : ar + rect_h, ac : ac + rect_w][anchor] = color
        sr, sc = item["source_origin"]
        source_mask = item["source_mask"]
        inp[sr : sr + source_mask.shape[0], sc : sc + source_mask.shape[1]][source_mask] = color

    if not _validate_generated_case(inp, out):
        return None
    return inp, out


def _validate_generated_case(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        produced = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(produced, expected_out):
        return False

    comps = find_components(inp, bg=BG, connectivity=4, monochromatic=True)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)
    if not (1 <= len(by_color) <= 2):
        return False

    seen_edges: set[str] = set()
    for color, parts in by_color.items():
        if len(parts) != 2:
            return False
        edge_part = None
        edge = None
        anchor_part = None
        for part in parts:
            part_edge = _edge_of_component(part["coords"], inp.shape)
            if part_edge is None:
                if anchor_part is not None:
                    return False
                anchor_part = part
            else:
                if edge_part is not None:
                    return False
                edge_part = part
                edge = part_edge
        if edge_part is None or anchor_part is None or edge is None:
            return False
        if edge in seen_edges:
            return False
        seen_edges.add(edge)

        source_mask, _ = _coords_to_mask(edge_part["coords"])
        anchor_coords = frozenset((int(r), int(c)) for r, c in anchor_part["coords"])
        correct = _shape_completion_positions(inp, _flip_mask(source_mask, edge), anchor_coords, BG)
        direct = _shape_completion_positions(inp, source_mask, anchor_coords, BG)
        wrong = _shape_completion_positions(inp, _wrong_flip_mask(source_mask, edge), anchor_coords, BG)
        if len(correct) != 1 or len(direct) != 0 or len(wrong) != 0:
            return False

    out_comps = find_components(expected_out, bg=BG, connectivity=4, monochromatic=True)
    out_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in out_comps:
        out_by_color[int(comp["color"])].append(comp)
    if set(out_by_color) != set(by_color):
        return False
    for color, parts in out_by_color.items():
        if len(parts) != 1:
            return False
        coords = parts[0]["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        if len(coords) != (int(rows.max()) - int(rows.min()) + 1) * (int(cols.max()) - int(cols.min()) + 1):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        edges = _choose_edge_config()
        case = _build_case(edges)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid mirror-complement puzzle")
