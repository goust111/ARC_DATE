from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GENERATION_RETRIES = 400
WORK_CANVAS = 30
MIN_PIECE_SIDE = 3
OUTPUT_H_RANGE = (8, 12)
OUTPUT_W_RANGE = (8, 12)
PIECE_COUNT_RANGE = (3, 6)
PLACEMENT_PAD = 2
CROP_MARGIN_MAX = 3


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(int(r0), int(c0))])
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(cells)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    counts = np.zeros(mask.shape, dtype=int)
    counts[1:, :] += mask[:-1, :]
    counts[:-1, :] += mask[1:, :]
    counts[:, 1:] += mask[:, :-1]
    counts[:, :-1] += mask[:, 1:]
    return counts


def _extract_blocks_and_markers(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    mask = g != BG
    counts = _neighbor_count(mask)
    marker_mask = mask & (counts == 1)

    core = g.copy()
    core[marker_mask] = BG
    comps = _components(core != BG)
    assert comps, "expected at least one block after removing seam markers"

    pieces: list[dict] = []
    cell_to_piece: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = _bbox(comp)
        patch = core[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(patch != BG), "each block must be a dense non-black rectangle"
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "shape": (int(patch.shape[0]), int(patch.shape[1])),
                "area": int(patch.size),
            }
        )
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                cell_to_piece[(int(rr), int(cc))] = idx

    markers: list[dict] = []
    for r, c in np.argwhere(marker_mask):
        core_neighbors = [
            (int(r + dr), int(c + dc))
            for dr, dc in DIR4
            if 0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1] and core[r + dr, c + dc] != BG
        ]
        assert len(core_neighbors) == 1, "each seam marker must touch exactly one block"
        piece_idx = cell_to_piece[core_neighbors[0]]
        r0, c0, r1, c1 = pieces[piece_idx]["bbox"]

        if c == c0 - 1 and r0 <= r <= r1:
            side = "left"
            offset = int(r - r0)
        elif c == c1 + 1 and r0 <= r <= r1:
            side = "right"
            offset = int(r - r0)
        elif r == r0 - 1 and c0 <= c <= c1:
            side = "top"
            offset = int(c - c0)
        elif r == r1 + 1 and c0 <= c <= c1:
            side = "bottom"
            offset = int(c - c0)
        else:
            raise AssertionError("marker must sit one cell outside one block edge")

        markers.append(
            {
                "piece": int(piece_idx),
                "side": side,
                "offset": int(offset),
                "color": int(g[r, c]),
            }
        )

    assert markers, "expected at least one marker pair"
    return pieces, markers


def _relative_position(
    origin: tuple[int, int],
    src_piece: dict,
    src_marker: dict,
    dst_piece: dict,
    dst_marker: dict,
) -> tuple[int, int]:
    top, left = origin
    src_h, src_w = src_piece["shape"]
    dst_h, dst_w = dst_piece["shape"]
    side_a = src_marker["side"]
    side_b = dst_marker["side"]
    off_a = int(src_marker["offset"])
    off_b = int(dst_marker["offset"])

    if side_a == "right" and side_b == "left":
        return top + off_a - off_b, left + src_w
    if side_a == "left" and side_b == "right":
        return top + off_a - off_b, left - dst_w
    if side_a == "bottom" and side_b == "top":
        return top + src_h, left + off_a - off_b
    if side_a == "top" and side_b == "bottom":
        return top - dst_h, left + off_a - off_b
    raise AssertionError("paired markers must lie on opposite sides")


def _assemble_from_constraints(pieces: list[dict], markers: list[dict]) -> np.ndarray:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for marker in markers:
        by_color[int(marker["color"])].append(marker)

    graph: dict[int, list[tuple[int, dict, dict]]] = defaultdict(list)
    for color, pair in by_color.items():
        assert len(pair) == 2, f"marker color {color} must appear exactly twice"
        a, b = pair
        assert a["piece"] != b["piece"], "a marker pair must connect two different blocks"
        graph[a["piece"]].append((int(b["piece"]), a, b))
        graph[b["piece"]].append((int(a["piece"]), b, a))

    assert len(graph) == len(pieces), "every block must participate in the marker-constraint graph"

    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    q = deque([0])
    while q:
        piece_idx = q.popleft()
        origin = positions[piece_idx]
        for other_idx, src_marker, dst_marker in graph[piece_idx]:
            target = _relative_position(origin, pieces[piece_idx], src_marker, pieces[other_idx], dst_marker)
            if other_idx in positions:
                assert positions[other_idx] == target, "marker constraints must be globally consistent"
            else:
                positions[other_idx] = target
                q.append(int(other_idx))

    assert len(positions) == len(pieces), "marker constraints must connect all blocks"

    min_r = min(pos[0] for pos in positions.values())
    min_c = min(pos[1] for pos in positions.values())
    norm_positions = {
        idx: (int(pos[0] - min_r), int(pos[1] - min_c))
        for idx, pos in positions.items()
    }

    out_h = max(norm_positions[idx][0] + pieces[idx]["shape"][0] for idx in range(len(pieces)))
    out_w = max(norm_positions[idx][1] + pieces[idx]["shape"][1] for idx in range(len(pieces)))
    out = np.full((out_h, out_w), BG, dtype=int)

    placed_area = 0
    for idx, piece in enumerate(pieces):
        top, left = norm_positions[idx]
        patch = piece["patch"]
        h, w = piece["shape"]
        target = out[top : top + h, left : left + w]
        assert not np.any(target != BG), "reassembled blocks must not overlap"
        target[:] = patch
        placed_area += int(piece["area"])

    assert placed_area == sum(int(piece["area"]) for piece in pieces)
    assert int(np.count_nonzero(out != BG)) == placed_area, "assembled rectangle must stay dense"
    assert out_h * out_w == placed_area, "assembled blocks must fill one tight rectangle"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular blocks by matching the colored seam markers outside their edges.

    Input structure:
    - The background is black.
    - Several dense rectangular blocks are scattered on the grid; their cells are grey with possible non-black pattern colors inside.
    - One-cell colored seam markers sit exactly one step outside some block edges, and each marker color appears exactly twice.
    - A matched color pair always connects opposite sides of two blocks and its row or column offset tells how the seam must align.

    Rule:
    - Identify the outside marker pixels and remove them temporarily.
    - Recover each dense rectangular block, then pair markers by color to infer which block touches which opposite side.
    - Convert every matched marker pair into an exact translation constraint between the two block top-left corners.
    - Solve all constraints, place the blocks with no gaps, and output only the reassembled block contents with the markers removed.

    Key insight:
    - The colored singletons are not noise: each same-color pair encodes both adjacency direction and exact row or column alignment, so the whole original rectangle can be reconstructed from those seam clues alone.
    """
    pieces, markers = _extract_blocks_and_markers(np.asarray(grid, dtype=int))
    return _assemble_from_constraints(pieces, markers)


def _split_rectangles(
    height: int,
    width: int,
    piece_count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, height, width)]
    while len(rects) < piece_count:
        choices: list[tuple[int, str]] = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            if r1 - r0 >= 2 * MIN_PIECE_SIDE:
                choices.append((idx, "h"))
            if c1 - c0 >= 2 * MIN_PIECE_SIDE:
                choices.append((idx, "v"))
        if not choices:
            return None
        idx, axis = rng.choice(choices)
        r0, c0, r1, c1 = rects.pop(idx)
        if axis == "h":
            cut = rng.randrange(r0 + MIN_PIECE_SIDE, r1 - MIN_PIECE_SIDE + 1)
            rects.extend([(r0, c0, cut, c1), (cut, c0, r1, c1)])
        else:
            cut = rng.randrange(c0 + MIN_PIECE_SIDE, c1 - MIN_PIECE_SIDE + 1)
            rects.extend([(r0, c0, r1, cut), (r0, cut, r1, c1)])
    return rects


def _adjacency_edges(rects: list[tuple[int, int, int, int]]) -> list[dict]:
    edges: list[dict] = []
    n = len(rects)
    for i in range(n):
        ar0, ac0, ar1, ac1 = rects[i]
        for j in range(i + 1, n):
            br0, bc0, br1, bc1 = rects[j]
            if ac1 == bc0:
                lo = max(ar0, br0)
                hi = min(ar1, br1)
                if lo < hi:
                    edges.append({"a": i, "b": j, "side_a": "right", "side_b": "left", "lo": lo, "hi": hi})
            elif bc1 == ac0:
                lo = max(ar0, br0)
                hi = min(ar1, br1)
                if lo < hi:
                    edges.append({"a": i, "b": j, "side_a": "left", "side_b": "right", "lo": lo, "hi": hi})

            if ar1 == br0:
                lo = max(ac0, bc0)
                hi = min(ac1, bc1)
                if lo < hi:
                    edges.append({"a": i, "b": j, "side_a": "bottom", "side_b": "top", "lo": lo, "hi": hi})
            elif br1 == ar0:
                lo = max(ac0, bc0)
                hi = min(ac1, bc1)
                if lo < hi:
                    edges.append({"a": i, "b": j, "side_a": "top", "side_b": "bottom", "lo": lo, "hi": hi})
    return edges


def _spanning_tree(edge_pool: list[dict], node_count: int, rng: random.Random) -> list[dict] | None:
    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    shuffled = edge_pool[:]
    rng.shuffle(shuffled)
    tree: list[dict] = []
    for edge in shuffled:
        if union(int(edge["a"]), int(edge["b"])):
            tree.append(edge.copy())
            if len(tree) == node_count - 1:
                return tree
    return None


def _local_offset(rect: tuple[int, int, int, int], side: str, global_coord: int) -> int:
    r0, c0, _r1, _c1 = rect
    if side in ("left", "right"):
        return int(global_coord - r0)
    return int(global_coord - c0)


def _assign_marker_offsets(
    edges: list[dict],
    rects: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> bool:
    used: dict[tuple[int, str], list[int]] = defaultdict(list)
    order = sorted(range(len(edges)), key=lambda idx: (edges[idx]["hi"] - edges[idx]["lo"], idx))

    def okay(piece_idx: int, side: str, offset: int) -> bool:
        return all(abs(offset - prev) > 1 for prev in used[(piece_idx, side)])

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        edge_idx = order[pos]
        edge = edges[edge_idx]
        coords = list(range(int(edge["lo"]), int(edge["hi"])))
        rng.shuffle(coords)
        for coord in coords:
            off_a = _local_offset(rects[int(edge["a"])], edge["side_a"], coord)
            off_b = _local_offset(rects[int(edge["b"])], edge["side_b"], coord)
            if not okay(int(edge["a"]), edge["side_a"], off_a):
                continue
            if not okay(int(edge["b"]), edge["side_b"], off_b):
                continue
            used[(int(edge["a"]), edge["side_a"])].append(off_a)
            used[(int(edge["b"]), edge["side_b"])].append(off_b)
            edge["coord"] = int(coord)
            if dfs(pos + 1):
                return True
            used[(int(edge["a"]), edge["side_a"])].pop()
            used[(int(edge["b"]), edge["side_b"])].pop()
            edge.pop("coord", None)
        return False

    return dfs(0)


def _decorate_output(height: int, width: int, rng: random.Random) -> np.ndarray:
    arr = np.full((height, width), GREY, dtype=int)
    colors = rng.sample(PALETTE, rng.randint(2, 4))
    mode = rng.choice(["row", "col", "diag", "mixed"])

    if mode in ("row", "mixed"):
        step = rng.randint(2, 4)
        phase = rng.randrange(step)
        arr[phase:height:step, :] = colors[0]

    if mode in ("col", "mixed"):
        step = rng.randint(2, 4)
        phase = rng.randrange(step)
        arr[:, phase:width:step] = colors[min(1, len(colors) - 1)]

    if mode == "diag":
        rows, cols = np.indices((height, width))
        period = rng.randint(2, 4)
        phase = rng.randrange(period)
        arr[(rows + cols + phase) % period == 0] = colors[0]

    for _ in range(rng.randint(2, 4)):
        rh = rng.randint(1, min(3, height))
        rw = rng.randint(1, min(3, width))
        r0 = rng.randint(0, height - rh)
        c0 = rng.randint(0, width - rw)
        arr[r0 : r0 + rh, c0 : c0 + rw] = rng.choice(colors)

    if not np.any(arr != GREY):
        r = rng.randrange(height)
        c = rng.randrange(width)
        arr[r, c] = colors[0]
    return arr


def _markers_for_pieces(
    rects: list[tuple[int, int, int, int]],
    edges: list[dict],
    rng: random.Random,
) -> list[list[dict]]:
    assert len(edges) <= len(PALETTE), "not enough colors for unique marker pairs"
    colors = rng.sample(PALETTE, len(edges))
    per_piece: list[list[dict]] = [[] for _ in rects]
    for color, edge in zip(colors, edges):
        coord = int(edge["coord"])
        off_a = _local_offset(rects[int(edge["a"])], edge["side_a"], coord)
        off_b = _local_offset(rects[int(edge["b"])], edge["side_b"], coord)
        per_piece[int(edge["a"])].append({"side": edge["side_a"], "offset": off_a, "color": int(color)})
        per_piece[int(edge["b"])].append({"side": edge["side_b"], "offset": off_b, "color": int(color)})
    return per_piece


def _reserve_box(top: int, left: int, height: int, width: int) -> tuple[slice, slice]:
    return (
        slice(top - PLACEMENT_PAD, top + height + PLACEMENT_PAD),
        slice(left - PLACEMENT_PAD, left + width + PLACEMENT_PAD),
    )


def _place_scattered(
    patches: list[np.ndarray],
    piece_markers: list[list[dict]],
    rng: random.Random,
) -> np.ndarray | None:
    canvas = np.full((WORK_CANVAS, WORK_CANVAS), BG, dtype=int)
    occupied = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=bool)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].size, reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        candidates = [
            (r, c)
            for r in range(PLACEMENT_PAD, WORK_CANVAS - h - PLACEMENT_PAD + 1)
            for c in range(PLACEMENT_PAD, WORK_CANVAS - w - PLACEMENT_PAD + 1)
        ]
        rng.shuffle(candidates)
        placed = False
        for top, left in candidates:
            rs, cs = _reserve_box(top, left, h, w)
            if occupied[rs, cs].any():
                continue
            placements[idx] = (int(top), int(left))
            occupied[rs, cs] = True
            placed = True
            break
        if not placed:
            return None

    for idx, patch in enumerate(patches):
        top, left = placements[idx]
        assert top is not None and left is not None
        h, w = patch.shape
        canvas[top : top + h, left : left + w] = patch
        for marker in piece_markers[idx]:
            side = marker["side"]
            offset = int(marker["offset"])
            if side == "top":
                mr, mc = top - 1, left + offset
            elif side == "bottom":
                mr, mc = top + h, left + offset
            elif side == "left":
                mr, mc = top + offset, left - 1
            else:
                mr, mc = top + offset, left + w
            assert canvas[mr, mc] == BG, "marker cells must stay isolated"
            canvas[mr, mc] = int(marker["color"])

    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_margin = rng.randint(1, min(CROP_MARGIN_MAX, int(r0)))
    left_margin = rng.randint(1, min(CROP_MARGIN_MAX, int(c0)))
    bottom_margin = rng.randint(1, min(CROP_MARGIN_MAX, WORK_CANVAS - 1 - int(r1)))
    right_margin = rng.randint(1, min(CROP_MARGIN_MAX, WORK_CANVAS - 1 - int(c1)))
    return canvas[
        int(r0) - top_margin : int(r1) + bottom_margin + 1,
        int(c0) - left_margin : int(c1) + right_margin + 1,
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    out_h = rng.randint(*OUTPUT_H_RANGE)
    out_w = rng.randint(*OUTPUT_W_RANGE)
    piece_count = rng.randint(*PIECE_COUNT_RANGE)

    rects = _split_rectangles(out_h, out_w, piece_count, rng)
    if rects is None:
        return None

    edge_pool = _adjacency_edges(rects)
    if len(edge_pool) < piece_count - 1:
        return None

    tree = _spanning_tree(edge_pool, piece_count, rng)
    if tree is None:
        return None
    if not _assign_marker_offsets(tree, rects, rng):
        return None

    output = _decorate_output(out_h, out_w, rng)
    patches = [output[r0:r1, c0:c1].copy() for r0, c0, r1, c1 in rects]
    if not all(np.any(patch == GREY) for patch in patches):
        return None
    if sum(int(np.any(patch != GREY)) for patch in patches) < 2:
        return None

    piece_markers = _markers_for_pieces(rects, tree, rng)
    scattered = _place_scattered(patches, piece_markers, rng)
    if scattered is None:
        return None

    try:
        if np.array_equal(main(scattered), output):
            return scattered
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid jigsaw-marker ARC input")
