from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


BG = 0
MAROON = 9
FRAME_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
MAX_ATTEMPTS = 500


@dataclass(eq=False)
class Node:
    children: list["Node"] = field(default_factory=list)
    marker: bool = False
    color: int = -1
    h: int = 0
    w: int = 0
    bbox: tuple[int, int, int, int] | None = None
    orientation: str = "h"
    fixed_size: tuple[int, int] | None = None


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _paint_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _frame_area(frame: dict) -> int:
    return int(frame["h"] * frame["w"])


def _contains(outer: dict, inner: dict) -> bool:
    return (
        int(outer["r0"]) < int(inner["r0"])
        and int(outer["c0"]) < int(inner["c0"])
        and int(outer["r1"]) > int(inner["r1"])
        and int(outer["c1"]) > int(inner["c1"])
    )


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in map(int, np.unique(g)):
        if int(color) in (BG, MAROON):
            continue
        for coords in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            hh = int(r1 - r0 + 1)
            ww = int(c1 - c0 + 1)
            if hh < 3 or ww < 3:
                continue
            ring = _ring_mask(hh, ww)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if len(coords) != int(ring.sum()):
                continue
            if not bool((patch[ring] == int(color)).all()):
                continue
            if bool((patch[~ring] == int(color)).any()):
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "h": int(hh),
                    "w": int(ww),
                    "children": [],
                    "parent": None,
                    "has_marker": False,
                    "immune": False,
                }
            )
    return frames


def _detect_maroon_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    for coords in _components(g == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if hh != ww:
            continue
        if len(coords) != int(hh * ww):
            continue
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parent frames pass their current color to the largest direct child frame
    that is not an immune maroon-leaf.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _detect_frames(g)
    if not frames:
        return out

    for i, frame in enumerate(frames):
        parents = [j for j, other in enumerate(frames) if i != j and _contains(other, frame)]
        if parents:
            parent = min(parents, key=lambda j: (_frame_area(frames[j]), frames[j]["r0"], frames[j]["c0"]))
            frames[i]["parent"] = int(parent)
            frames[parent]["children"].append(int(i))

    markers = _detect_maroon_boxes(g)
    for mr0, mc0, mr1, mc1 in markers:
        owners = [
            i
            for i, frame in enumerate(frames)
            if int(frame["r0"]) < int(mr0)
            and int(frame["c0"]) < int(mc0)
            and int(frame["r1"]) > int(mr1)
            and int(frame["c1"]) > int(mc1)
        ]
        if not owners:
            continue
        owner = min(owners, key=lambda j: (_frame_area(frames[j]), frames[j]["r0"], frames[j]["c0"]))
        frames[owner]["has_marker"] = True

    for frame in frames:
        frame["immune"] = bool(frame["has_marker"] and not frame["children"])

    current_colors = [int(frame["color"]) for frame in frames]

    def pick_child(fid: int) -> int | None:
        candidates = [cid for cid in frames[fid]["children"] if not bool(frames[cid]["immune"])]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda cid: (
                _frame_area(frames[cid]),
                -int(frames[cid]["r0"]),
                -int(frames[cid]["c0"]),
            ),
        )

    def dfs(fid: int) -> None:
        chosen = pick_child(int(fid))
        if chosen is not None:
            new_color = int(current_colors[int(fid)])
            current_colors[int(chosen)] = int(new_color)
            _paint_border(out, frames[int(chosen)]["bbox"], int(new_color))
        for cid in frames[int(fid)]["children"]:
            dfs(int(cid))

    roots = [i for i, frame in enumerate(frames) if frame["parent"] is None]
    roots.sort(key=lambda i: (_frame_area(frames[i]), -frames[i]["r0"], -frames[i]["c0"]), reverse=True)
    for rid in roots:
        dfs(int(rid))

    return out


def _walk(node: Node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _leaves(node: Node) -> list[Node]:
    return [cur for cur in _walk(node) if not cur.children]


def _split_nonnegative(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts <= 1:
        return [int(total)]
    marks = sorted(
        int(x) for x in rng.choice(total + parts - 1, size=parts - 1, replace=False).tolist()
    )
    out: list[int] = []
    prev = -1
    for mark in marks + [int(total + parts - 1)]:
        out.append(int(mark - prev - 1))
        prev = int(mark)
    return out


def _sample_leaf_size(rng: np.random.Generator, band: str) -> tuple[int, int]:
    if band == "small":
        lo, hi = 5, 8
    elif band == "mid":
        lo, hi = 6, 9
    else:
        lo, hi = 8, 11
    return int(rng.integers(lo, hi)), int(rng.integers(lo, hi))


def _build_tree(rng: np.random.Generator, total_frames: int) -> tuple[Node, Node]:
    root = Node()
    pivot = Node()
    root.children.append(pivot)

    max_target = min(3, total_frames - 2)
    target_children = 2 if max_target == 2 else int(rng.integers(2, max_target + 1))
    pivot.children = [Node() for _ in range(target_children)]

    remaining = int(total_frames - (2 + target_children))
    while remaining > 0:
        non_pivot = [child for child in root.children if child is not pivot]
        candidates = [child for child in non_pivot if len(child.children) < 3]
        if len(root.children) < 3 and (not candidates or bool(rng.random() < 0.65)):
            branch = Node()
            root.children.append(branch)
            remaining -= 1
            if remaining > 0 and bool(rng.random() < 0.6):
                extra = min(int(rng.integers(1, min(remaining, 2) + 1)), 3)
                branch.children.extend(Node() for _ in range(extra))
                remaining -= int(extra)
        else:
            parent = candidates[int(rng.integers(0, len(candidates)))]
            parent.children.append(Node())
            remaining -= 1

    return root, pivot


def _configure_tree(root: Node, pivot: Node, mode: str, rng: np.random.Generator) -> None:
    for node in _walk(root):
        node.marker = False
        node.fixed_size = None

    pivot_children = list(pivot.children)
    rng.shuffle(pivot_children)

    for _ in range(100):
        areas: list[int] = []
        if mode == "skip_largest":
            immune = pivot_children[0]
            immune.marker = True
            immune.fixed_size = _sample_leaf_size(rng, "large")
            eligible = pivot_children[1]
            eligible.marker = False
            eligible.fixed_size = _sample_leaf_size(rng, "small")
            for child in pivot_children[2:]:
                child.marker = bool(rng.random() < 0.4)
                child.fixed_size = _sample_leaf_size(rng, "small" if child.marker else "mid")
        elif mode == "all_immune":
            for idx, child in enumerate(pivot_children):
                child.marker = True
                band = "large" if idx == 0 and len(pivot_children) > 1 else "mid"
                child.fixed_size = _sample_leaf_size(rng, band)
        else:
            leader = pivot_children[0]
            leader.marker = False
            leader.fixed_size = _sample_leaf_size(rng, "large")
            if len(pivot_children) > 1:
                blocker = pivot_children[1]
                blocker.marker = True
                blocker.fixed_size = _sample_leaf_size(rng, "small")
            for child in pivot_children[2:]:
                child.marker = bool(rng.random() < 0.5)
                child.fixed_size = _sample_leaf_size(rng, "small" if child.marker else "mid")

        areas = [int(child.fixed_size[0] * child.fixed_size[1]) for child in pivot_children if child.fixed_size]
        if len(areas) == len(set(areas)):
            break

    pivot_set = set(pivot.children)
    extra_leaves = [leaf for leaf in _leaves(root) if leaf not in pivot_set]
    for leaf in extra_leaves:
        leaf.marker = bool(rng.random() < 0.35)


def _assign_sizes(node: Node, pivot: Node, rng: np.random.Generator, under_pivot: bool = False) -> None:
    inside_pivot = bool(under_pivot or node is pivot)
    if not node.children:
        if node.fixed_size is not None:
            node.h, node.w = map(int, node.fixed_size)
            return
        lo, hi = (5, 9) if inside_pivot else (5, 8)
        node.h = int(rng.integers(lo, hi))
        node.w = int(rng.integers(lo, hi))
        return

    for child in node.children:
        _assign_sizes(child, pivot, rng, inside_pivot)

    node.orientation = "h" if len(node.children) == 1 else str(rng.choice(np.array(["h", "v"], dtype=object)))
    if node.orientation == "h":
        base_h = max(child.h for child in node.children) + 4
        base_w = sum(child.w for child in node.children) + max(0, len(node.children) - 1) + 4
    else:
        base_h = sum(child.h for child in node.children) + max(0, len(node.children) - 1) + 4
        base_w = max(child.w for child in node.children) + 4

    extra_cap = 6 if node is pivot else 5 if not under_pivot else 4
    node.h = int(base_h + rng.integers(0, extra_cap))
    node.w = int(base_w + rng.integers(0, extra_cap))


def _place(node: Node, r0: int, c0: int, rng: np.random.Generator) -> None:
    node.bbox = (int(r0), int(c0), int(r0 + node.h - 1), int(c0 + node.w - 1))
    if not node.children:
        return

    rng.shuffle(node.children)
    if node.orientation == "h":
        needed = sum(child.w for child in node.children) + max(0, len(node.children) - 1)
        slack = int(node.w - 4 - needed)
        gaps = _split_nonnegative(slack, len(node.children) + 1, rng)
        x = int(c0 + 2 + gaps[0])
        for idx, child in enumerate(node.children):
            y_slack = int(node.h - child.h - 4)
            y = int(r0 + 2 + rng.integers(0, y_slack + 1))
            _place(child, y, x, rng)
            x += int(child.w)
            if idx < len(node.children) - 1:
                x += int(1 + gaps[idx + 1])
    else:
        needed = sum(child.h for child in node.children) + max(0, len(node.children) - 1)
        slack = int(node.h - 4 - needed)
        gaps = _split_nonnegative(slack, len(node.children) + 1, rng)
        y = int(r0 + 2 + gaps[0])
        for idx, child in enumerate(node.children):
            x_slack = int(node.w - child.w - 4)
            x = int(c0 + 2 + rng.integers(0, x_slack + 1))
            _place(child, y, x, rng)
            y += int(child.h)
            if idx < len(node.children) - 1:
                y += int(1 + gaps[idx + 1])


def _draw_marker(grid: np.ndarray, node: Node, rng: np.random.Generator) -> None:
    assert node.bbox is not None
    r0, c0, r1, c1 = map(int, node.bbox)
    max_side = min(2, node.h - 4, node.w - 4)
    side = 1 if max_side <= 1 else int(rng.integers(1, max_side + 1))
    rr = int(rng.integers(r0 + 2, r1 - side + 1))
    cc = int(rng.integers(c0 + 2, c1 - side + 1))
    grid[rr : rr + side, cc : cc + side] = MAROON


def _render_input(root: Node, rng: np.random.Generator) -> np.ndarray:
    top = int(rng.integers(1, 5))
    left = int(rng.integers(1, 5))
    bottom = int(rng.integers(1, 5))
    right = int(rng.integers(1, 5))
    _place(root, top, left, rng)
    grid = np.full((top + root.h + bottom, left + root.w + right), BG, dtype=int)
    for node in _walk(root):
        assert node.bbox is not None
        _paint_border(grid, node.bbox, int(node.color))
        if node.marker and not node.children:
            _draw_marker(grid, node, rng)
    return grid


def _child_key(node: Node) -> tuple[int, int, int]:
    assert node.bbox is not None
    r0, c0, _r1, _c1 = map(int, node.bbox)
    return int(node.h * node.w), -int(r0), -int(c0)


def _expected_output(inp: np.ndarray, root: Node) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    current = {node: int(node.color) for node in _walk(root)}

    def dfs(node: Node) -> None:
        eligible = [child for child in node.children if not bool(child.marker and not child.children)]
        if eligible:
            chosen = max(eligible, key=_child_key)
            current[chosen] = int(current[node])
            assert chosen.bbox is not None
            _paint_border(out, chosen.bbox, int(current[node]))
        for child in node.children:
            dfs(child)

    dfs(root)
    return out


def _assign_colors(root: Node, rng: np.random.Generator) -> None:
    nodes = list(_walk(root))
    colors = rng.choice(FRAME_COLORS, size=len(nodes), replace=False).tolist()
    rng.shuffle(colors)
    for node, color in zip(nodes, colors, strict=False):
        node.color = int(color)


def _all_unique_sibling_areas(root: Node) -> bool:
    for node in _walk(root):
        if len(node.children) < 2:
            continue
        areas = [int(child.h * child.w) for child in node.children]
        if len(areas) != len(set(areas)):
            return False
    return True


def _selected_child(node: Node) -> Node | None:
    eligible = [child for child in node.children if not bool(child.marker and not child.children)]
    if not eligible:
        return None
    return max(eligible, key=_child_key)


def _valid_mode(root: Node, pivot: Node, mode: str) -> bool:
    if not _all_unique_sibling_areas(root):
        return False

    root_choice = _selected_child(root)
    if root_choice is not pivot:
        return False

    pivot_choice = _selected_child(pivot)
    pivot_areas = [int(child.h * child.w) for child in pivot.children]
    max_area = max(pivot_areas)

    if mode == "all_immune":
        return all(child.marker and not child.children for child in pivot.children) and pivot_choice is None

    if pivot_choice is None:
        return False

    chosen_area = int(pivot_choice.h * pivot_choice.w)
    if mode == "skip_largest":
        return chosen_area < max_area and any(
            bool(child.marker and not child.children and child.h * child.w == max_area) for child in pivot.children
        )

    return chosen_area == max_area and any(child.marker for child in pivot.children)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    mode = ["normal", "skip_largest", "all_immune"][int(rng.integers(0, 3))]
    total_frames = int(rng.integers(4, 8))

    root, pivot = _build_tree(rng, total_frames)
    _configure_tree(root, pivot, mode, rng)
    _assign_sizes(root, pivot, rng)
    _assign_colors(root, rng)
    inp = _render_input(root, rng)

    if not _valid_mode(root, pivot, mode):
        return None

    out = _expected_output(inp, root)
    if bool(np.array_equal(inp, out)):
        return None
    if not bool(np.array_equal(main(inp), out)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
