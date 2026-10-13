from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    pts = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r, c in pts:
        if r not in {r0, r1} and c not in {c0, c1}:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if not _is_rect_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            frames.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                    "children": [],
                    "parent": None,
                    "depth": 1,
                }
            )
    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _annotate_depths(frames: list[dict]) -> None:
    for frame in frames:
        containers = [other for other in frames if other is not frame and _contains(other["bbox"], frame["bbox"])]
        if containers:
            parent = min(containers, key=lambda item: item["area"])
            frame["parent"] = parent
            parent["children"].append(frame)

    def depth_of(frame: dict) -> int:
        if frame["parent"] is None:
            return 1
        if frame["depth"] != 1:
            return int(frame["depth"])
        frame["depth"] = depth_of(frame["parent"]) + 1
        return int(frame["depth"])

    for frame in frames:
        frame["depth"] = 1
    for frame in frames:
        depth_of(frame)


def _find_palette(grid: np.ndarray, bg: int, min_len: int) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int, list[int]]] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != c1:
            continue
        if len(comp) != r1 - r0 + 1:
            continue
        if c0 not in {0, w - 1}:
            continue
        if r0 != 0 and r1 != h - 1:
            continue
        inner_col = 1 if c0 == 0 else w - 2
        if w > 1 and np.any(grid[r0 : r1 + 1, inner_col] != bg):
            continue
        colors = [int(grid[r, c0]) for r in range(r0, r1 + 1)]
        if len(colors) < min_len:
            continue
        candidates.append((len(colors), r0, c0, r1, colors))
    assert candidates, "expected a corner palette column"
    _length, r0, c0, r1, colors = max(candidates, key=lambda item: item[0])
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    return mask, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames and their leaf fills by palette depth, then erase the palette.

    Input structure:
    - bg is the dominant color. The foreground contains one-pixel-thick hollow rectangular frames,
      possibly arranged in multiple nesting trees.
    - Each frame is monochrome and may either contain smaller frames or, if it is a leaf, a solid
      fill region inside its border.
    - A corner palette appears as a one-cell-wide vertical column of colors; reading top-to-bottom
      yields the replacement color for depth 1, depth 2, depth 3, and so on.

    Rule:
    - Detect every monochrome rectangular frame and recover the nesting relation by strict spatial
      containment of frame bounding boxes.
    - Assign depth 1 to outermost frames and increment depth for each enclosed level.
    - Replace each frame border with the palette color for its depth, and replace every non-background
      cell inside a leaf frame with that same depth color.
    - Erase the palette itself back to the background color.

    Key insight:
    The transformation is keyed by geometric containment depth rather than by original colors, so the
    solver must first build the frame hierarchy and only then apply the palette lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"
    _annotate_depths(frames)
    max_depth = max(int(frame["depth"]) for frame in frames)
    palette_mask, palette = _find_palette(g, bg, max_depth)
    assert len(palette) >= max_depth, "palette must cover every nesting depth"

    out = g.copy()
    out[palette_mask] = bg

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(palette[int(frame["depth"]) - 1])
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color

    for frame in frames:
        if frame["children"]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        color = int(palette[int(frame["depth"]) - 1])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size:
            interior[interior != bg] = color

    return out


@dataclass
class _GenNode:
    children: list["_GenNode"] = field(default_factory=list)
    border_color: int = 0
    fill_color: int | None = None
    orient: str = "h"
    top: int = 1
    bottom: int = 1
    left: int = 1
    right: int = 1
    gaps: list[int] = field(default_factory=list)
    shifts: list[int] = field(default_factory=list)
    inner_h: int = 0
    inner_w: int = 0
    h: int = 0
    w: int = 0


_BG = 0
_COLORS = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 400


def _rand_color(exclude: set[int]) -> int:
    choices = [color for color in _COLORS if color not in exclude]
    return int(random.choice(choices))


def _build_node(remaining_depth: int) -> _GenNode:
    node = _GenNode()
    if remaining_depth == 1:
        node.inner_h = random.randint(1, 3)
        node.inner_w = random.randint(1, 3)
        node.h = node.inner_h + 2
        node.w = node.inner_w + 2
        return node

    child_options = [1, 2] if remaining_depth >= 4 else [1, 2, 3]
    child_count = random.choice(child_options)
    deep_child = random.randrange(child_count)
    child_depths = []
    for idx in range(child_count):
        if idx == deep_child:
            child_depths.append(remaining_depth - 1)
        else:
            child_depths.append(random.randint(1, remaining_depth - 1))
    node.children = [_build_node(depth) for depth in child_depths]
    node.orient = random.choice(["h", "v"])
    node.top = random.randint(1, 2)
    node.bottom = random.randint(1, 2)
    node.left = random.randint(1, 2)
    node.right = random.randint(1, 2)
    node.gaps = [random.randint(1, 2) for _ in range(max(0, child_count - 1))]

    if node.orient == "h":
        cross_h = max(child.h for child in node.children) + random.randint(0, 1)
        node.shifts = [random.randint(0, cross_h - child.h) for child in node.children]
        node.h = 2 + node.top + node.bottom + cross_h
        node.w = 2 + node.left + node.right + sum(child.w for child in node.children) + sum(node.gaps)
    else:
        cross_w = max(child.w for child in node.children) + random.randint(0, 1)
        node.shifts = [random.randint(0, cross_w - child.w) for child in node.children]
        node.h = 2 + node.top + node.bottom + sum(child.h for child in node.children) + sum(node.gaps)
        node.w = 2 + node.left + node.right + cross_w
    return node


def _assign_colors(node: _GenNode) -> None:
    node.border_color = _rand_color({_BG})
    if not node.children:
        node.fill_color = _rand_color({_BG, node.border_color})
        return
    for child in node.children:
        _assign_colors(child)


def _pack_roots(roots: list[_GenNode], palette_on_left: bool) -> tuple[int, int, list[tuple[int, int]]] | None:
    count = len(roots)
    order = roots[:]
    random.shuffle(order)
    roots[:] = order
    orient = random.choice(["h", "v"])
    gaps = [random.randint(2, 4) for _ in range(max(0, count - 1))]

    if orient == "h":
        used_h = max(root.h for root in roots)
        used_w = sum(root.w for root in roots) + sum(gaps)
    else:
        used_h = sum(root.h for root in roots) + sum(gaps)
        used_w = max(root.w for root in roots)

    min_h = used_h + 2
    min_w = used_w + 3
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(max(12, min_h), 30)
    w = random.randint(max(12, min_w), 30)
    top = random.randint(1, h - used_h - 1)
    if palette_on_left:
        left = random.randint(2, w - used_w - 1)
    else:
        left = random.randint(1, w - used_w - 2)

    positions: list[tuple[int, int]] = []
    if orient == "h":
        cursor = left
        for idx, root in enumerate(roots):
            row = top + random.randint(0, used_h - root.h)
            positions.append((int(row), int(cursor)))
            cursor += root.w
            if idx < len(gaps):
                cursor += gaps[idx]
    else:
        cursor = top
        for idx, root in enumerate(roots):
            col = left + random.randint(0, used_w - root.w)
            positions.append((int(cursor), int(col)))
            cursor += root.h
            if idx < len(gaps):
                cursor += gaps[idx]

    return h, w, positions


def _draw_node(
    grid: np.ndarray,
    node: _GenNode,
    r0: int,
    c0: int,
    depth: int,
    records: list[dict],
) -> None:
    r1 = r0 + node.h - 1
    c1 = c0 + node.w - 1
    grid[r0 : r1 + 1, c0] = node.border_color
    grid[r0 : r1 + 1, c1] = node.border_color
    grid[r0, c0 : c1 + 1] = node.border_color
    grid[r1, c0 : c1 + 1] = node.border_color
    records.append({"bbox": (r0, c0, r1, c1), "depth": depth, "leaf": not node.children})

    if not node.children:
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(node.fill_color)
        return

    if node.orient == "h":
        cursor = c0 + 1 + node.left
        base_r = r0 + 1 + node.top
        for idx, child in enumerate(node.children):
            child_r = base_r + node.shifts[idx]
            _draw_node(grid, child, child_r, cursor, depth + 1, records)
            cursor += child.w
            if idx < len(node.gaps):
                cursor += node.gaps[idx]
    else:
        cursor = r0 + 1 + node.top
        base_c = c0 + 1 + node.left
        for idx, child in enumerate(node.children):
            child_c = base_c + node.shifts[idx]
            _draw_node(grid, child, cursor, child_c, depth + 1, records)
            cursor += child.h
            if idx < len(node.gaps):
                cursor += node.gaps[idx]


def _oracle_output(
    inp: np.ndarray,
    bg: int,
    palette_mask: np.ndarray,
    palette_colors: list[int],
    records: list[dict],
) -> np.ndarray:
    out = inp.copy()
    out[palette_mask] = bg
    for rec in records:
        r0, c0, r1, c1 = rec["bbox"]
        color = int(palette_colors[int(rec["depth"]) - 1])
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        if rec["leaf"]:
            out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    max_depth = random.randint(2, 5)
    root_count = random.randint(1, 2 if max_depth >= 5 else 3)
    root_depths = [max_depth] + [random.randint(1, max_depth) for _ in range(root_count - 1)]
    random.shuffle(root_depths)
    roots = [_build_node(depth) for depth in root_depths]
    for root in roots:
        _assign_colors(root)

    corner = random.choice(["tl", "tr", "bl", "br"])
    palette_on_left = corner in {"tl", "bl"}
    packed = _pack_roots(roots, palette_on_left=palette_on_left)
    if packed is None:
        return None
    h, w, positions = packed

    grid = np.full((h, w), _BG, dtype=int)
    palette_colors = random.sample(list(_COLORS), max_depth)
    palette_mask = np.zeros_like(grid, dtype=bool)
    palette_rows = list(range(max_depth)) if corner in {"tl", "tr"} else list(range(h - max_depth, h))
    palette_col = 0 if palette_on_left else w - 1
    for idx, row in enumerate(palette_rows):
        grid[row, palette_col] = int(palette_colors[idx])
        palette_mask[row, palette_col] = True

    records: list[dict] = []
    for root, (r0, c0) in zip(roots, positions, strict=False):
        _draw_node(grid, root, r0, c0, depth=1, records=records)

    expected = _oracle_output(grid, _BG, palette_mask, palette_colors, records)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, _out = case
        return inp
    raise RuntimeError("failed to generate a valid nested-frame palette case")
