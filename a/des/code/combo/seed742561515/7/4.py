from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


BG = 0


@dataclass
class Frame:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    parent: "Frame | None" = None
    children: list["Frame"] = field(default_factory=list)


@dataclass
class GenNode:
    children: list["GenNode"] = field(default_factory=list)
    parent: "GenNode | None" = None
    rel_r: int = 0
    rel_c: int = 0
    abs_r: int = 0
    abs_c: int = 0
    h: int = 0
    w: int = 0
    color: int = -1


def _legend_ranks(grid: np.ndarray) -> dict[int, int]:
    colors = [int(v) for v in grid[-1] if int(v) != BG]
    assert colors, "expected a non-empty legend row"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return {color: idx for idx, color in enumerate(colors)}


def _is_rectangular_frame(mask: np.ndarray) -> bool:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _extract_frames(body: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for color in sorted(set(map(int, body.flatten())) - {BG}):
        coords = np.argwhere(body == color)
        assert coords.size > 0
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        r0, c0 = int(lo[0]), int(lo[1])
        r1, c1 = int(hi[0]), int(hi[1])
        mask = body[r0 : r1 + 1, c0 : c1 + 1] == color
        assert _is_rectangular_frame(mask), f"color {color} must form a 1-cell-thick rectangular frame"
        frames.append(Frame(color=color, r0=r0, c0=c0, r1=r1, c1=c1))
    return frames


def _contains(outer: Frame, inner: Frame) -> bool:
    return outer.r0 < inner.r0 and outer.c0 < inner.c0 and outer.r1 > inner.r1 and outer.c1 > inner.c1


def _build_containment(frames: list[Frame]) -> None:
    for inner in frames:
        parents = [outer for outer in frames if outer is not inner and _contains(outer, inner)]
        if not parents:
            continue
        inner.parent = min(parents, key=lambda f: (f.r1 - f.r0) * (f.c1 - f.c0))
        inner.parent.children.append(inner)


def _winner_color(frame: Frame, ranks: dict[int, int]) -> int:
    best_color = frame.color
    best_rank = ranks[best_color]
    cur = frame.parent
    while cur is not None:
        cur_rank = ranks[cur.color]
        if cur_rank > best_rank:
            best_color = cur.color
            best_rank = cur_rank
        cur = cur.parent
    return best_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the bottom legend row, then fill each leaf frame's interior with the
    highest-ranked color on its ancestor path as defined by that legend row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a legend row"

    ranks = _legend_ranks(g)
    body = g[:-1].copy()
    frames = _extract_frames(body)
    _build_containment(frames)

    out = body.copy()
    for frame in frames:
        if frame.children:
            continue
        fill = _winner_color(frame, ranks)
        out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] = fill
    return out


# --------------- generator ---------------

TOTAL_FRAMES_RANGE = (2, 5)
MAX_TREE_DEPTH = 3

LEAF_H_RANGE = (3, 5)
LEAF_W_RANGE = (3, 6)
INNER_MARGIN_RANGE = (2, 3)
SIBLING_GAP_RANGE = (2, 3)
ROOT_GAP_RANGE = (3, 5)
ROOT_PAD_RANGE = (1, 4)

BODY_H_RANGE = (8, 24)
BODY_W_RANGE = (11, 30)
EXTRA_SLACK_MAX = 4

GEN_RETRIES = 400


def _make_subtree(budget: int, depth: int, rng: random.Random) -> GenNode:
    node = GenNode()
    if budget == 1:
        return node

    remaining = budget - 1
    if remaining == 1 or depth >= MAX_TREE_DEPTH or rng.random() < 0.45:
        child_budgets = [remaining]
    else:
        cut = rng.randint(1, remaining - 1)
        child_budgets = [cut, remaining - cut]
        rng.shuffle(child_budgets)

    for child_budget in child_budgets:
        child = _make_subtree(child_budget, depth + 1, rng)
        child.parent = node
        node.children.append(child)
    return node


def _make_forest(rng: random.Random) -> list[GenNode]:
    total = rng.randint(*TOTAL_FRAMES_RANGE)
    if total >= 3 and rng.random() < 0.4:
        cut = rng.randint(1, total - 1)
        budgets = [cut, total - cut]
        rng.shuffle(budgets)
    else:
        budgets = [total]
    return [_make_subtree(budget, 0, rng) for budget in budgets]


def _layout_node(node: GenNode, rng: random.Random) -> None:
    for child in node.children:
        _layout_node(child, rng)

    if not node.children:
        node.h = rng.randint(*LEAF_H_RANGE)
        node.w = rng.randint(*LEAF_W_RANGE)
        return

    if len(node.children) == 1:
        child = node.children[0]
        top = rng.randint(*INNER_MARGIN_RANGE)
        bottom = rng.randint(*INNER_MARGIN_RANGE)
        left = rng.randint(*INNER_MARGIN_RANGE)
        right = rng.randint(*INNER_MARGIN_RANGE)
        child.rel_r = top
        child.rel_c = left
        node.h = top + child.h + bottom
        node.w = left + child.w + right
        return

    rng.shuffle(node.children)
    a, b = node.children
    top = rng.randint(*INNER_MARGIN_RANGE)
    bottom = rng.randint(*INNER_MARGIN_RANGE)
    left = rng.randint(*INNER_MARGIN_RANGE)
    right = rng.randint(*INNER_MARGIN_RANGE)
    gap = rng.randint(*SIBLING_GAP_RANGE)

    if rng.random() < 0.5:
        band_h = max(a.h, b.h) + rng.randint(0, 2)
        a.rel_r = top + rng.randint(0, band_h - a.h)
        b.rel_r = top + rng.randint(0, band_h - b.h)
        a.rel_c = left
        b.rel_c = left + a.w + gap
        node.h = top + band_h + bottom
        node.w = left + a.w + gap + b.w + right
    else:
        band_w = max(a.w, b.w) + rng.randint(0, 2)
        a.rel_c = left + rng.randint(0, band_w - a.w)
        b.rel_c = left + rng.randint(0, band_w - b.w)
        a.rel_r = top
        b.rel_r = top + a.h + gap
        node.h = top + a.h + gap + b.h + bottom
        node.w = left + band_w + right


def _set_abs(node: GenNode, abs_r: int, abs_c: int) -> None:
    node.abs_r = abs_r
    node.abs_c = abs_c
    for child in node.children:
        _set_abs(child, abs_r + child.rel_r, abs_c + child.rel_c)


def _arrange_roots(roots: list[GenNode], rng: random.Random) -> tuple[int, int]:
    for root in roots:
        root.rel_r = 0
        root.rel_c = 0

    if len(roots) == 1:
        return roots[0].h, roots[0].w

    rng.shuffle(roots)
    gap = rng.randint(*ROOT_GAP_RANGE)
    if rng.random() < 0.5:
        band_h = max(root.h for root in roots) + rng.randint(0, 3)
        cursor = 0
        for root in roots:
            root.rel_r = rng.randint(0, band_h - root.h)
            root.rel_c = cursor
            cursor += root.w + gap
        return band_h, cursor - gap

    band_w = max(root.w for root in roots) + rng.randint(0, 3)
    cursor = 0
    for root in roots:
        root.rel_r = cursor
        root.rel_c = rng.randint(0, band_w - root.w)
        cursor += root.h + gap
    return cursor - gap, band_w


def _all_nodes(roots: list[GenNode]) -> list[GenNode]:
    out: list[GenNode] = []

    def walk(node: GenNode) -> None:
        out.append(node)
        for child in node.children:
            walk(child)

    for root in roots:
        walk(root)
    return out


def _leaf_nodes(roots: list[GenNode]) -> list[GenNode]:
    return [node for node in _all_nodes(roots) if not node.children]


def _place_forest(roots: list[GenNode], legend_span: int, rng: random.Random) -> tuple[int, int] | None:
    group_h, group_w = _arrange_roots(roots, rng)

    top = rng.randint(*ROOT_PAD_RANGE)
    bottom = rng.randint(*ROOT_PAD_RANGE)
    left = rng.randint(*ROOT_PAD_RANGE)
    right = rng.randint(*ROOT_PAD_RANGE)

    body_h = group_h + top + bottom
    body_w = group_w + left + right

    need_w = max(BODY_W_RANGE[0], legend_span)
    if body_w < need_w:
        extra = need_w - body_w
        extra_left = rng.randint(0, extra)
        left += extra_left
        right += extra - extra_left
        body_w = need_w

    if body_h < BODY_H_RANGE[0]:
        extra = BODY_H_RANGE[0] - body_h
        extra_top = rng.randint(0, extra)
        top += extra_top
        bottom += extra - extra_top
        body_h = BODY_H_RANGE[0]

    if body_h > BODY_H_RANGE[1] or body_w > BODY_W_RANGE[1]:
        return None

    extra_h = rng.randint(0, min(EXTRA_SLACK_MAX, BODY_H_RANGE[1] - body_h))
    extra_top = rng.randint(0, extra_h)
    top += extra_top
    bottom += extra_h - extra_top
    body_h += extra_h

    extra_w = rng.randint(0, min(EXTRA_SLACK_MAX, BODY_W_RANGE[1] - body_w))
    extra_left = rng.randint(0, extra_w)
    left += extra_left
    right += extra_w - extra_left
    body_w += extra_w

    for root in roots:
        _set_abs(root, top + root.rel_r, left + root.rel_c)
    return body_h, body_w


def _assign_colors(roots: list[GenNode], rng: random.Random) -> list[int]:
    nodes = _all_nodes(roots)
    colors = rng.sample(list(range(1, 10)), len(nodes))
    rng.shuffle(colors)
    for node, color in zip(nodes, colors):
        node.color = color
    legend = colors[:]
    rng.shuffle(legend)
    return legend


def _draw_frame(grid: np.ndarray, node: GenNode) -> None:
    r0, c0 = node.abs_r, node.abs_c
    r1, c1 = r0 + node.h - 1, c0 + node.w - 1
    grid[r0, c0 : c1 + 1] = node.color
    grid[r1, c0 : c1 + 1] = node.color
    grid[r0 : r1 + 1, c0] = node.color
    grid[r0 : r1 + 1, c1] = node.color


def _render_body(roots: list[GenNode], body_h: int, body_w: int) -> np.ndarray:
    body = np.full((body_h, body_w), BG, dtype=int)
    for node in _all_nodes(roots):
        _draw_frame(body, node)
    return body


def _winner_for_leaf(node: GenNode, ranks: dict[int, int]) -> int:
    best_color = node.color
    best_rank = ranks[best_color]
    cur = node.parent
    while cur is not None:
        cur_rank = ranks[cur.color]
        if cur_rank > best_rank:
            best_color = cur.color
            best_rank = cur_rank
        cur = cur.parent
    return best_color


def _render_expected(body: np.ndarray, roots: list[GenNode], legend: list[int]) -> np.ndarray:
    ranks = {color: idx for idx, color in enumerate(legend)}
    out = body.copy()
    for leaf in _leaf_nodes(roots):
        fill = _winner_for_leaf(leaf, ranks)
        out[leaf.abs_r + 1 : leaf.abs_r + leaf.h - 1, leaf.abs_c + 1 : leaf.abs_c + leaf.w - 1] = fill
    return out


def _path_depths(roots: list[GenNode]) -> list[int]:
    depths: list[int] = []
    for leaf in _leaf_nodes(roots):
        depth = 0
        cur = leaf.parent
        while cur is not None:
            depth += 1
            cur = cur.parent
        depths.append(depth)
    return depths


def _legend_modes_ok(roots: list[GenNode], legend: list[int]) -> bool:
    ranks = {color: idx for idx, color in enumerate(legend)}
    winners = [_winner_for_leaf(leaf, ranks) for leaf in _leaf_nodes(roots)]
    leaves = _leaf_nodes(roots)
    nontrivial = [winner != leaf.color for winner, leaf in zip(winners, leaves)]
    depths = _path_depths(roots)

    if not depths or max(depths) == 0:
        return False
    if max(depths) >= 2 and not any(nontrivial):
        return False
    if len(leaves) >= 2 and len(set(winners)) == 1 and len(set(legend)) >= 3 and random.random() < 0.5:
        return False
    return True


def _build_input(body: np.ndarray, legend: list[int], rng: random.Random) -> np.ndarray:
    span = 2 * len(legend) - 1
    legend_row = np.full((1, body.shape[1]), BG, dtype=int)
    start = rng.randint(0, body.shape[1] - span)
    for idx, color in enumerate(legend):
        legend_row[0, start + 2 * idx] = color
    return np.vstack([body, legend_row])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    roots = _make_forest(rng)
    for root in roots:
        _layout_node(root, rng)

    legend_span = 2 * sum(1 for _ in _all_nodes(roots)) - 1
    placed = _place_forest(roots, legend_span, rng)
    if placed is None:
        return None
    body_h, body_w = placed

    legend = _assign_colors(roots, rng)
    if not _legend_modes_ok(roots, legend):
        return None

    body = _render_body(roots, body_h, body_w)
    inp = _build_input(body, legend, rng)
    expected = _render_expected(body, roots, legend)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
