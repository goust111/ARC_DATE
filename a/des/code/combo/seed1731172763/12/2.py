from __future__ import annotations

import itertools
import random

import numpy as np


BG = 8
GREY = 5
VISIBLE_COLORS = (0, 1, 2, 3, 4, 6, 7, 9)
TEMPLATES = ("h", "v", "hv", "vh")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(counts.argmax())])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _paint_node(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]
    raise AssertionError("line endpoints must be axis-aligned")


def _find_nodes(grid: np.ndarray, bg: int) -> tuple[dict[int, dict[str, int]], dict[tuple[int, int], int]]:
    nodes: dict[int, dict[str, int]] = {}
    pix_to_node: dict[tuple[int, int], int] = {}
    next_id = 0
    for comp in _components((grid != bg) & (grid != GREY)):
        if len(comp) != 4:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        color = int(grid[r0, c0])
        if color == bg or color == GREY:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        nodes[next_id] = {"color": color}
        for cell in comp:
            pix_to_node[cell] = next_id
        next_id += 1
    assert nodes, "expected at least one 2x2 node"
    return nodes, pix_to_node


def _find_legend(
    grid: np.ndarray, bg: int
) -> tuple[dict[tuple[int, int], int], tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates = []
    for comp in _components((grid != bg) & (grid != GREY)):
        if len(comp) != 3:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        if min(rs) != max(rs):
            continue
        c0, c1 = min(cs), max(cs)
        if c1 - c0 != 2:
            continue
        r = rs[0]
        row = tuple(int(grid[r, c]) for c in range(c0, c1 + 1))
        candidates.append({"r": r, "c0": c0, "c1": c1, "colors": row})

    buckets: dict[str, list[dict[str, object]]] = {"tl": [], "tr": [], "bl": [], "br": []}
    for entry in candidates:
        c0 = int(entry["c0"])
        r = int(entry["r"])
        if c0 == 0:
            key = "tl" if r <= h // 2 else "bl"
            buckets[key].append(entry)
        if c0 == w - 3:
            key = "tr" if r <= h // 2 else "br"
            buckets[key].append(entry)

    def score(item: tuple[str, list[dict[str, object]]]) -> tuple[int, int]:
        key, entries = item
        if not entries:
            return (-1, -10**9)
        if key == "tl":
            dist = sum(int(e["r"]) + int(e["c0"]) for e in entries)
        elif key == "tr":
            dist = sum(int(e["r"]) + (w - 1 - int(e["c1"])) for e in entries)
        elif key == "bl":
            dist = sum((h - 1 - int(e["r"])) + int(e["c0"]) for e in entries)
        else:
            dist = sum((h - 1 - int(e["r"])) + (w - 1 - int(e["c1"])) for e in entries)
        return (len(entries), -dist)

    legend_corner, legend_entries = max(buckets.items(), key=score)
    assert legend_entries, "legend entries must be found in one corner"

    mapping: dict[tuple[int, int], int] = {}
    rmin = min(int(entry["r"]) for entry in legend_entries)
    rmax = max(int(entry["r"]) for entry in legend_entries)
    cmin = min(int(entry["c0"]) for entry in legend_entries)
    cmax = max(int(entry["c1"]) for entry in legend_entries)
    for entry in legend_entries:
        left, mix, right = map(int, entry["colors"])
        mapping[tuple(sorted((left, right)))] = mix
    assert mapping, "legend mapping must be non-empty"

    if legend_corner in ("tl", "tr"):
        bbox = (rmin, cmin, rmax, cmax)
    else:
        bbox = (rmin, cmin, rmax, cmax)
    return mapping, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey path by looking up the colors of the two 2x2 nodes it connects.

    The legend is a corner stack of 1x3 entries [left_color, mixed_color, right_color].
    After recoloring the paths, the legend region is cleared back to background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    nodes, pix_to_node = _find_nodes(g, bg)
    mapping, legend_bbox = _find_legend(g, bg)

    out = g.copy()
    for comp in _components(g == GREY):
        touched = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                node_id = pix_to_node.get((nr, nc))
                if node_id is not None:
                    touched.add(int(node_id))
        assert len(touched) == 2, "each grey path must connect exactly two nodes"
        pair = tuple(sorted(int(nodes[node_id]["color"]) for node_id in touched))
        assert pair in mapping, "legend must define every connected node-color pair"
        mixed = int(mapping[pair])
        for r, c in comp:
            out[r, c] = mixed

    r0, c0, r1, c1 = legend_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out


def _legend_bbox(h: int, w: int, entry_count: int, corner: str) -> tuple[int, int, int, int]:
    height = 2 * entry_count - 1
    if corner in ("tl", "tr"):
        r0 = 0
    else:
        r0 = h - height
    if corner in ("tl", "bl"):
        c0 = 0
    else:
        c0 = w - 3
    return (r0, c0, r0 + height - 1, c0 + 2)


def _reserve_rect(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _box_size(template: str) -> tuple[int, int]:
    if template == "h":
        return random.randint(5, 7), random.randint(8, 11)
    if template == "v":
        return random.randint(8, 11), random.randint(5, 7)
    return random.randint(7, 10), random.randint(7, 10)


def _place_boxes(
    h: int, w: int, specs: list[dict[str, object]], legend_rect: tuple[int, int, int, int]
) -> list[dict[str, object]] | None:
    used = np.zeros((h, w), dtype=bool)
    _reserve_rect(used, legend_rect, pad=1)
    placed: list[dict[str, object]] = []
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["h"]) * int(specs[i]["w"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        bh = int(spec["h"])
        bw = int(spec["w"])
        found = None
        for _ in range(120):
            r0 = random.randint(0, h - bh)
            c0 = random.randint(0, w - bw)
            rect = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            rr0 = max(0, rect[0] - 1)
            cc0 = max(0, rect[1] - 1)
            rr1 = min(h - 1, rect[2] + 1)
            cc1 = min(w - 1, rect[3] + 1)
            if not used[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                found = rect
                used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
                break
        if found is None:
            return None
        placed.append(
            {
                "pair": spec["pair"],
                "mix": spec["mix"],
                "template": spec["template"],
                "rect": found,
            }
        )
    return placed


def _corner_pos(rect: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    top_rows = [r0 + 1, r0 + 2]
    bottom_rows = [r0 + h - 4, r0 + h - 3]
    left_cols = [c0 + 1, c0 + 2]
    right_cols = [c0 + w - 4, c0 + w - 3]
    if corner == "tl":
        return random.choice(top_rows), random.choice(left_cols)
    if corner == "tr":
        return random.choice(top_rows), random.choice(right_cols)
    if corner == "bl":
        return random.choice(bottom_rows), random.choice(left_cols)
    return random.choice(bottom_rows), random.choice(right_cols)


def _render_horizontal(
    inp: np.ndarray, out: np.ndarray, rect: tuple[int, int, int, int], pair: tuple[int, int], mix: int
) -> None:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    node_r = random.randint(r0 + 1, r0 + h - 3)
    left_c = random.randint(c0 + 1, c0 + 2)
    right_c = random.randint(c0 + w - 4, c0 + w - 3)
    colors = list(pair)
    random.shuffle(colors)
    _paint_node(inp, node_r, left_c, colors[0])
    _paint_node(inp, node_r, right_c, colors[1])
    _paint_node(out, node_r, left_c, colors[0])
    _paint_node(out, node_r, right_c, colors[1])
    path_r = node_r + random.randint(0, 1)
    for c in range(left_c + 2, right_c):
        inp[path_r, c] = GREY
        out[path_r, c] = mix


def _render_vertical(
    inp: np.ndarray, out: np.ndarray, rect: tuple[int, int, int, int], pair: tuple[int, int], mix: int
) -> None:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    node_c = random.randint(c0 + 1, c0 + w - 3)
    top_r = random.randint(r0 + 1, r0 + 2)
    bot_r = random.randint(r0 + h - 4, r0 + h - 3)
    colors = list(pair)
    random.shuffle(colors)
    _paint_node(inp, top_r, node_c, colors[0])
    _paint_node(inp, bot_r, node_c, colors[1])
    _paint_node(out, top_r, node_c, colors[0])
    _paint_node(out, bot_r, node_c, colors[1])
    path_c = node_c + random.randint(0, 1)
    for r in range(top_r + 2, bot_r):
        inp[r, path_c] = GREY
        out[r, path_c] = mix


def _render_elbow(
    inp: np.ndarray,
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    pair: tuple[int, int],
    mix: int,
    template: str,
) -> None:
    start_corner, end_corner = random.choice(
        [("tl", "br"), ("tr", "bl"), ("bl", "tr"), ("br", "tl")]
    )
    sr, sc = _corner_pos(rect, start_corner)
    er, ec = _corner_pos(rect, end_corner)
    colors = list(pair)
    random.shuffle(colors)
    _paint_node(inp, sr, sc, colors[0])
    _paint_node(inp, er, ec, colors[1])
    _paint_node(out, sr, sc, colors[0])
    _paint_node(out, er, ec, colors[1])

    if template == "hv":
        step_h = 1 if ec > sc else -1
        step_v = 1 if er > sr else -1
        start = (sr + random.randint(0, 1), sc + 2 if step_h > 0 else sc - 1)
        end = (er - 1 if step_v > 0 else er + 2, ec + random.randint(0, 1))
        bend = (start[0], end[1])
        path = _line(start, bend) + _line(bend, end)[1:]
    else:
        step_v = 1 if er > sr else -1
        step_h = 1 if ec > sc else -1
        start = (sr + 2 if step_v > 0 else sr - 1, sc + random.randint(0, 1))
        end = (er + random.randint(0, 1), ec - 1 if step_h > 0 else ec + 2)
        bend = (end[0], start[1])
        path = _line(start, bend) + _line(bend, end)[1:]

    for r, c in path:
        inp[r, c] = GREY
        out[r, c] = mix


def _render_pair(
    inp: np.ndarray,
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    pair: tuple[int, int],
    mix: int,
    template: str,
) -> None:
    if template == "h":
        _render_horizontal(inp, out, rect, pair, mix)
    elif template == "v":
        _render_vertical(inp, out, rect, pair, mix)
    else:
        _render_elbow(inp, out, rect, pair, mix, template)


def _draw_legend(
    inp: np.ndarray,
    pairs_and_mixes: list[tuple[tuple[int, int], int]],
    corner: str,
) -> tuple[int, int, int, int]:
    h, w = inp.shape
    rect = _legend_bbox(h, w, len(pairs_and_mixes), corner)
    r0, c0, _r1, _c1 = rect
    random.shuffle(pairs_and_mixes)
    for i, (pair, mix) in enumerate(pairs_and_mixes):
        row = r0 + 2 * i
        left, right = pair
        if random.random() < 0.5:
            left, right = right, left
        inp[row, c0 : c0 + 3] = np.array([left, mix, right], dtype=int)
    return rect


def _gen_attempt() -> np.ndarray | None:
    path_count = random.randint(2, 4)
    palette_size = 4 if path_count == 4 else random.choice([3, 4])
    node_colors = random.sample(list(VISIBLE_COLORS), palette_size)
    all_pairs = list(itertools.combinations(node_colors, 2))
    random.shuffle(all_pairs)
    used_pairs = [tuple(sorted(pair)) for pair in all_pairs[:path_count]]
    mix_pool = [color for color in VISIBLE_COLORS if color not in node_colors]
    mix_colors = random.sample(mix_pool, path_count)

    specs = []
    for pair, mix in zip(used_pairs, mix_colors, strict=False):
        template = random.choice(TEMPLATES)
        bh, bw = _box_size(template)
        specs.append({"pair": pair, "mix": int(mix), "template": template, "h": bh, "w": bw})

    h = random.randint(22, 30)
    w = random.randint(22, 30)
    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_rect = _legend_bbox(h, w, path_count, corner)
    placed = _place_boxes(h, w, specs, legend_rect)
    if placed is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    _draw_legend(inp, [(tuple(spec["pair"]), int(spec["mix"])) for spec in placed], corner)

    for spec in placed:
        _render_pair(
            inp,
            out,
            tuple(spec["rect"]),
            tuple(spec["pair"]),
            int(spec["mix"]),
            str(spec["template"]),
        )

    r0, c0, r1, c1 = legend_rect
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
