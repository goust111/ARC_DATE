from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PALETTE = tuple(range(1, 10))
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
SIDE_DELTAS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
DEGREE_COLORS = {1: BLUE, 2: RED, 3: GREEN, 4: YELLOW}
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 and int(np.sum(mask)) == len(comps[0])


def _component_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        colors = set()
        for r, c in coords:
            sub[r - r0, c - c0] = True
            colors.add(int(grid[r, c]))
        objects.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": sub,
                "area": int(len(coords)),
                "colors": colors,
            }
        )
    return objects


def _neg(sig: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-x for x in sig)


def _match_signatures(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return len(a) == len(b) and any(x != 0 or y != 0 for x, y in zip(a, b)) and all(
        x == -y for x, y in zip(a, b)
    )


def _valid_piece_frame(mask: np.ndarray, top: int, bottom: int, left: int, right: int) -> bool:
    h, w = mask.shape
    s = h - top - bottom
    if s < 3 or s != w - left - right:
        return False

    core = mask[top : h - bottom, left : w - right]
    if not bool(core[0, 0] and core[0, -1] and core[-1, 0] and core[-1, -1]):
        return False

    if top:
        row = mask[0]
        if not row.any():
            return False
        allowed = np.zeros(w, dtype=bool)
        allowed[left + 1 : left + s - 1] = True
        if bool((row & ~allowed).any()):
            return False
    else:
        if bool(mask[0, :left].any()) or bool(mask[0, left + s :].any()):
            return False

    if bottom:
        row = mask[-1]
        if not row.any():
            return False
        allowed = np.zeros(w, dtype=bool)
        allowed[left + 1 : left + s - 1] = True
        if bool((row & ~allowed).any()):
            return False
    else:
        if bool(mask[-1, :left].any()) or bool(mask[-1, left + s :].any()):
            return False

    if left:
        col = mask[:, 0]
        if not col.any():
            return False
        allowed = np.zeros(h, dtype=bool)
        allowed[top + 1 : top + s - 1] = True
        if bool((col & ~allowed).any()):
            return False
    else:
        if bool(mask[:top, 0].any()) or bool(mask[top + s :, 0].any()):
            return False

    if right:
        col = mask[:, -1]
        if not col.any():
            return False
        allowed = np.zeros(h, dtype=bool)
        allowed[top + 1 : top + s - 1] = True
        if bool((col & ~allowed).any()):
            return False
    else:
        if bool(mask[:top, -1].any()) or bool(mask[top + s :, -1].any()):
            return False

    return True


def _decode_piece(mask: np.ndarray) -> dict:
    candidates: list[tuple[int, int, int, int, int]] = []
    h, w = mask.shape
    for top, bottom, left, right in itertools.product((0, 1), repeat=4):
        if not _valid_piece_frame(mask, top, bottom, left, right):
            continue
        s = h - top - bottom
        candidates.append((top, bottom, left, right, s))
    assert len(candidates) == 1, "piece frame must decode uniquely"

    top, bottom, left, right, s = candidates[0]
    core = mask[top : h - bottom, left : w - right]

    def side_sig(name: str) -> tuple[int, ...]:
        vals = []
        for pos in range(1, s - 1):
            if name == "top":
                if top and bool(mask[0, left + pos]):
                    vals.append(1)
                elif not bool(core[0, pos]):
                    vals.append(-1)
                else:
                    vals.append(0)
            elif name == "bottom":
                if bottom and bool(mask[-1, left + pos]):
                    vals.append(1)
                elif not bool(core[-1, pos]):
                    vals.append(-1)
                else:
                    vals.append(0)
            elif name == "left":
                if left and bool(mask[top + pos, 0]):
                    vals.append(1)
                elif not bool(core[pos, 0]):
                    vals.append(-1)
                else:
                    vals.append(0)
            else:
                if right and bool(mask[top + pos, -1]):
                    vals.append(1)
                elif not bool(core[pos, -1]):
                    vals.append(-1)
                else:
                    vals.append(0)
        return tuple(vals)

    return {
        "mask": mask.copy(),
        "core_size": int(s),
        "top_tab": int(top),
        "bottom_tab": int(bottom),
        "left_tab": int(left),
        "right_tab": int(right),
        "top": side_sig("top"),
        "right": side_sig("right"),
        "bottom": side_sig("bottom"),
        "left": side_sig("left"),
    }


def _extract_pieces_and_edges(grid: np.ndarray) -> tuple[dict[int, dict], list[tuple[int, int]]]:
    objects = _component_objects(grid)
    legend_edges: list[tuple[int, int]] = []
    pieces: dict[int, dict] = {}

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        colors = list(obj["colors"])
        if obj["area"] == 2 and (r1 - r0 + 1, c1 - c0 + 1) == (1, 2) and len(colors) == 2:
            ordered = sorted(obj["coords"], key=lambda rc: rc[1])
            a = int(grid[ordered[0]])
            b = int(grid[ordered[1]])
            assert a != b
            legend_edges.append((a, b))
            continue

        assert len(colors) == 1, "each fragment must be monochrome"
        color = int(colors[0])
        piece = _decode_piece(obj["mask"])
        piece["color"] = color
        pieces[color] = piece

    assert pieces, "expected at least one fragment"
    assert legend_edges, "expected 1x2 color legend blocks"
    assert set(pieces) == {c for edge in legend_edges for c in edge}, "legend colors must match fragments"
    return pieces, legend_edges


def _build_relations(
    pieces: dict[int, dict], edges: list[tuple[int, int]]
) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple[int, int]]]:
    neighbors = {color: set() for color in pieces}
    relations: dict[tuple[int, int], tuple[int, int]] = {}
    seen_edges: set[frozenset[int]] = set()

    for a, b in edges:
        edge_key = frozenset((int(a), int(b)))
        assert edge_key not in seen_edges, "legend must not repeat an adjacency pair"
        seen_edges.add(edge_key)
        pa = pieces[int(a)]
        pb = pieces[int(b)]
        matches: list[tuple[int, int]] = []
        if _match_signatures(pa["right"], pb["left"]):
            matches.append((0, 1))
        if _match_signatures(pa["left"], pb["right"]):
            matches.append((0, -1))
        if _match_signatures(pa["bottom"], pb["top"]):
            matches.append((1, 0))
        if _match_signatures(pa["top"], pb["bottom"]):
            matches.append((-1, 0))
        assert len(matches) == 1, "each adjacent pair must have exactly one geometric match"
        dr, dc = matches[0]
        relations[(int(a), int(b))] = (int(dr), int(dc))
        relations[(int(b), int(a))] = (-int(dr), -int(dc))
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))

    return neighbors, relations


def _layout_from_relations(
    pieces: dict[int, dict], neighbors: dict[int, set[int]], relations: dict[tuple[int, int], tuple[int, int]]
) -> tuple[dict[int, tuple[int, int]], int, int]:
    start = next(iter(pieces))
    coords = {int(start): (0, 0)}
    stack = [int(start)]
    while stack:
        cur = stack.pop()
        cr, cc = coords[cur]
        for nxt in neighbors[cur]:
            dr, dc = relations[(cur, nxt)]
            want = (cr + dr, cc + dc)
            if nxt in coords:
                assert coords[nxt] == want, "adjacency directions must be globally consistent"
                continue
            coords[nxt] = want
            stack.append(int(nxt))

    assert len(coords) == len(pieces), "adjacency graph must connect all fragments"
    min_r = min(r for r, _c in coords.values())
    min_c = min(c for _r, c in coords.values())
    norm = {color: (r - min_r, c - min_c) for color, (r, c) in coords.items()}
    rows = max(r for r, _c in norm.values()) + 1
    cols = max(c for _r, c in norm.values()) + 1
    expected_cells = {(r, c) for r in range(rows) for c in range(cols)}
    placed_cells = set(norm.values())
    assert placed_cells == expected_cells, "assembled fragments must fill a complete rectangle"

    pos_to_color = {pos: color for color, pos in norm.items()}
    for color, (r, c) in norm.items():
        expected = set()
        for dr, dc in SIDE_DELTAS.values():
            other = (r + dr, c + dc)
            if other in pos_to_color:
                expected.add(pos_to_color[other])
        assert expected == neighbors[color], "legend adjacency must match rectangular grid neighbors"

    return norm, int(rows), int(cols)


def _assemble_output(
    pieces: dict[int, dict], coords: dict[int, tuple[int, int]], rows: int, cols: int, neighbors: dict[int, set[int]]
) -> np.ndarray:
    core_sizes = {piece["core_size"] for piece in pieces.values()}
    assert len(core_sizes) == 1, "all fragments must share one core size"
    s = int(next(iter(core_sizes)))
    out = np.zeros((rows * s, cols * s), dtype=int)
    filled = np.zeros_like(out, dtype=bool)

    for color, piece in pieces.items():
        degree = len(neighbors[color])
        assert degree in DEGREE_COLORS
        paint = DEGREE_COLORS[degree]
        gr = coords[color][0] * s - piece["top_tab"]
        gc = coords[color][1] * s - piece["left_tab"]
        for pr, pc in np.argwhere(piece["mask"]):
            rr = int(gr + pr)
            cc = int(gc + pc)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            assert not filled[rr, cc], "assembled pieces must not overlap"
            filled[rr, cc] = True
            out[rr, cc] = int(paint)

    assert bool(filled.all()), "assembled pieces must cover the whole rectangle"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered keyed fragments into a rectangle and recolor each fragment by its final neighbor count.

    Input structure:
    - bg = black.
    - Several monochrome connected fragments are scattered on the grid; each fragment keeps its orientation.
    - Fragment edges may contain 1-pixel tabs and 1-pixel notches on their four sides.
    - Separate 1x2 multicolor legend blocks list which fragment colors should be adjacent.

    Rule:
    - Read every 1x2 legend block as an unordered adjacency pair between two fragment colors.
    - For each adjacent pair, compare the two fragments' side profiles and find the unique side pairing whose tabs and notches are exactly complementary.
    - Use those pairwise directions to place all fragments into one complete rectangle, then recolor each fragment by how many neighbors it touches there:
      1 neighbor -> blue, 2 -> red, 3 -> green, 4 -> yellow.

    Key insight:
    - The color legend says which pieces must meet, but the actual left/right/up/down placement is hidden in the tiny tab-vs-notch pattern along each edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    pieces, edges = _extract_pieces_and_edges(g)
    neighbors, relations = _build_relations(pieces, edges)
    coords, rows, cols = _layout_from_relations(pieces, neighbors, relations)
    return _assemble_output(pieces, coords, rows, cols, neighbors)


def _pattern_pool(length: int) -> list[tuple[int, ...]]:
    reps = []
    for raw in itertools.product((-1, 0, 1), repeat=length):
        if not any(raw):
            continue
        neg = _neg(raw)
        if raw <= neg:
            reps.append(tuple(int(x) for x in raw))
    return reps


def _choose_layout() -> tuple[int, int]:
    family = random.choice(("strip", "edge", "center"))
    if family == "strip":
        return 1, random.randint(3, 5)
    if family == "edge":
        return random.choice(((2, 3), (3, 2), (2, 4), (4, 2)))
    return 3, 3


def _make_piece_mask(sides: dict[str, tuple[int, ...]], core_size: int) -> dict:
    s = int(core_size)
    top_tab = int(any(v == 1 for v in sides["top"]))
    bottom_tab = int(any(v == 1 for v in sides["bottom"]))
    left_tab = int(any(v == 1 for v in sides["left"]))
    right_tab = int(any(v == 1 for v in sides["right"]))

    mask = np.zeros((s + top_tab + bottom_tab, s + left_tab + right_tab), dtype=bool)
    top0 = top_tab
    left0 = left_tab
    mask[top0 : top0 + s, left0 : left0 + s] = True

    for pos, val in enumerate(sides["top"], start=1):
        if val == -1:
            mask[top0, left0 + pos] = False
        elif val == 1:
            mask[0, left0 + pos] = True
    for pos, val in enumerate(sides["bottom"], start=1):
        if val == -1:
            mask[top0 + s - 1, left0 + pos] = False
        elif val == 1:
            mask[-1, left0 + pos] = True
    for pos, val in enumerate(sides["left"], start=1):
        if val == -1:
            mask[top0 + pos, left0] = False
        elif val == 1:
            mask[top0 + pos, 0] = True
    for pos, val in enumerate(sides["right"], start=1):
        if val == -1:
            mask[top0 + pos, left0 + s - 1] = False
        elif val == 1:
            mask[top0 + pos, -1] = True

    piece = _decode_piece(mask)
    assert piece["core_size"] == s
    return piece


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    sub[patch] = int(color)


def _legend_patch(edges: list[tuple[int, int]], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    block_cols = random.randint(1, min(4, len(edges)))
    block_rows = (len(edges) + block_cols - 1) // block_cols
    legend_h = 2 * block_rows - 1
    legend_w = 3 * block_cols - 1
    patch = np.zeros((legend_h, legend_w), dtype=int)
    fill = np.zeros_like(patch, dtype=bool)
    shuffled = edges[:]
    random.shuffle(shuffled)
    for idx, (a, b) in enumerate(shuffled):
        rr = 2 * (idx // block_cols)
        cc = 3 * (idx % block_cols)
        if random.random() < 0.5:
            patch[rr, cc] = int(a)
            patch[rr, cc + 1] = int(b)
        else:
            patch[rr, cc] = int(b)
            patch[rr, cc + 1] = int(a)
        fill[rr, cc : cc + 2] = True
    return patch, fill


def _place_legend(grid: np.ndarray, legend: np.ndarray, fill: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    lh, lw = legend.shape
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - lw
    elif corner == "bl":
        r0, c0 = h - lh, 0
    else:
        r0, c0 = h - lh, w - lw

    placed = np.zeros_like(grid, dtype=bool)
    for rr, cc in np.argwhere(fill):
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        grid[gr, gc] = int(legend[rr, cc])
        placed[gr, gc] = True
    return placed, _dilate8(placed)


def _assemble_expected(
    pieces: dict[int, dict], coords: dict[int, tuple[int, int]], rows: int, cols: int, neighbors: dict[int, set[int]]
) -> np.ndarray:
    return _assemble_output(pieces, coords, rows, cols, neighbors)


def _random_grid_size(piece_count: int) -> tuple[int, int]:
    if piece_count >= 9:
        lo = 29
    elif piece_count >= 8:
        lo = 27
    elif piece_count >= 6:
        lo = 25
    else:
        lo = 22
    return random.randint(lo, 30), random.randint(lo, 30)


def _place_pieces(grid: np.ndarray, pieces: dict[int, dict], blocked: np.ndarray) -> bool:
    h, w = grid.shape
    order = list(pieces)
    random.shuffle(order)
    order.sort(key=lambda color: int(pieces[color]["mask"].sum()), reverse=True)
    occupied = blocked.copy()

    for color in order:
        patch = pieces[color]["mask"]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            window = occupied[r0 : r0 + ph, c0 : c0 + pw]
            if bool((window & patch).any()):
                continue
            if bool((grid[r0 : r0 + ph, c0 : c0 + pw] != BLACK).any()):
                continue
            _paint_patch(grid, patch, r0, c0, color)
            footprint = np.zeros_like(grid, dtype=bool)
            footprint[r0 : r0 + ph, c0 : c0 + pw] = patch
            occupied |= _dilate8(footprint)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = _choose_layout()
    piece_count = rows * cols
    core_size = 5 if piece_count >= 8 else random.choice((5, 6))
    side_len = core_size - 2
    all_patterns = _pattern_pool(side_len)

    edge_count = rows * (cols - 1) + cols * (rows - 1)
    if len(all_patterns) < edge_count:
        return None

    colors = random.sample(PALETTE, piece_count)
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(colors)
    color_at = {pos: int(color) for pos, color in zip(positions, colors)}
    sides_by_color = {
        color: {side: tuple(0 for _ in range(side_len)) for side in SIDES} for color in colors
    }

    chosen_patterns = random.sample(all_patterns, edge_count)
    pattern_idx = 0
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            color = color_at[(r, c)]
            if c + 1 < cols:
                other = color_at[(r, c + 1)]
                pat = chosen_patterns[pattern_idx]
                pattern_idx += 1
                if random.random() < 0.5:
                    sides_by_color[color]["right"] = pat
                    sides_by_color[other]["left"] = _neg(pat)
                else:
                    sides_by_color[color]["right"] = _neg(pat)
                    sides_by_color[other]["left"] = pat
                edges.append((color, other))
            if r + 1 < rows:
                other = color_at[(r + 1, c)]
                pat = chosen_patterns[pattern_idx]
                pattern_idx += 1
                if random.random() < 0.5:
                    sides_by_color[color]["bottom"] = pat
                    sides_by_color[other]["top"] = _neg(pat)
                else:
                    sides_by_color[color]["bottom"] = _neg(pat)
                    sides_by_color[other]["top"] = pat
                edges.append((color, other))

    pieces = {}
    for color, sides in sides_by_color.items():
        piece = _make_piece_mask(sides, core_size)
        if not _is_connected(piece["mask"]):
            return None
        piece["color"] = int(color)
        pieces[int(color)] = piece

    neighbors, relations = _build_relations(pieces, edges)
    coords, solved_rows, solved_cols = _layout_from_relations(pieces, neighbors, relations)
    if solved_rows != rows or solved_cols != cols:
        return None
    expected = _assemble_expected(pieces, coords, rows, cols, neighbors)

    grid_h, grid_w = _random_grid_size(piece_count)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    legend_patch, legend_fill = _legend_patch(edges, grid_h, grid_w)
    if legend_patch.shape[0] > grid_h or legend_patch.shape[1] > grid_w:
        return None
    legend_cells, blocked = _place_legend(grid, legend_patch, legend_fill)
    blocked |= _dilate8(legend_cells)

    if not _place_pieces(grid, pieces, blocked):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
