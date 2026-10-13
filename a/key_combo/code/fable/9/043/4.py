from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARKER_COLORS = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _prune_to_fixpoint(mask: np.ndarray) -> np.ndarray:
    alive = mask.copy()
    degree = np.zeros(mask.shape, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(alive):
        rr, cc = int(r), int(c)
        deg = 0
        for nr, nc in _neighbors4(rr, cc):
            if _in_bounds(alive.shape, nr, nc) and alive[nr, nc]:
                deg += 1
        degree[rr, cc] = deg
        if deg <= 1:
            q.append((rr, cc))

    while q:
        r, c = q.popleft()
        if not alive[r, c] or degree[r, c] > 1:
            continue
        alive[r, c] = False
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(alive.shape, nr, nc) and alive[nr, nc]:
                degree[nr, nc] -= 1
                if degree[nr, nc] <= 1:
                    q.append((nr, nc))
    return alive


def _enclosed_voids(structure_mask: np.ndarray) -> list[np.ndarray]:
    voids: list[np.ndarray] = []
    for coords in _components(~structure_mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (
            int(rows.min()) == 0
            or int(rows.max()) == structure_mask.shape[0] - 1
            or int(cols.min()) == 0
            or int(cols.max()) == structure_mask.shape[1] - 1
        ):
            continue
        voids.append(coords)
    return voids


def _render_output(
    structure_mask: np.ndarray,
    fg_color: int,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.zeros(structure_mask.shape, dtype=int)
    out[structure_mask] = int(fg_color)
    for r, c, color in markers:
        out[r, c] = int(color)

    marker_map = {(r, c): int(color) for r, c, color in markers}
    used_markers: set[tuple[int, int]] = set()
    for coords in _enclosed_voids(structure_mask):
        local_markers = [(int(r), int(c)) for r, c in coords if (int(r), int(c)) in marker_map]
        assert len(local_markers) == 1, "each enclosed region must contain exactly one marker"
        marker_pos = local_markers[0]
        used_markers.add(marker_pos)
        fill_color = marker_map[marker_pos]
        for r, c in coords:
            rr, cc = int(r), int(c)
            if out[rr, cc] == BG:
                out[rr, cc] = fill_color

    assert len(used_markers) == len(markers), "all markers must lie inside surviving loops"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively prune dangling foreground branches, then fill each surviving loop interior with its marker color.

    Input structure:
    - A black background contains one structural foreground color that forms several 4-connected closed loops.
    - Tree-like branch tendrils hang from the loops, and some pairs of loops are joined by narrow channels whose two ends both attach to loops.
    - Each loop interior contains exactly one non-black marker pixel, while no marker appears outside a loop interior.

    Rule:
    - Repeatedly delete every structural foreground cell that has at most one 4-neighbor of the same structural color until no more deletions are possible.
    - Keep the surviving structural cells, which include the loops and any channel that is supported from both ends.
    - For every background region enclosed by the surviving structure, recolor that whole enclosed region with the color of its unique marker pixel; leave markers unchanged.

    Key insight:
    - The stable survivors are the 2-core of the structural foreground graph, so a bridge between two loops survives even though it is not itself part of any single loop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    nonzero = [int(color) for color in np.unique(g) if int(color) != BG]
    assert len(nonzero) >= 2, "expected one structure color plus marker colors"
    fg_color = max(nonzero, key=lambda color: int(np.count_nonzero(g == color)))
    structure_mask = g == fg_color
    marker_mask = (g != BG) & (g != fg_color)
    markers = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(marker_mask)]

    assert np.count_nonzero(structure_mask) > len(markers), "structure color must dominate markers"
    assert markers, "expected at least one marker"

    alive = _prune_to_fixpoint(structure_mask)
    out = _render_output(alive, fg_color, markers)
    return out


def _draw_box_border(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _choose_layout(rng: np.random.Generator):
    orientation = str(rng.choice(["horizontal", "vertical"]))
    n_loops = _randint(rng, 2, 3)
    loop_heights = [_randint(rng, 5, 7) for _ in range(n_loops)]
    loop_widths = [_randint(rng, 5, 7) for _ in range(n_loops)]
    gaps = [_randint(rng, 2, 5) for _ in range(n_loops - 1)]

    if orientation == "horizontal":
        top_room = _randint(rng, 4, 6)
        bottom_room = _randint(rng, 4, 6)
        left_room = _randint(rng, 3, 5)
        right_room = _randint(rng, 3, 5)
        row_offsets = [_randint(rng, 0, 1) for _ in range(n_loops)]
        height = top_room + max(row_offsets[i] + loop_heights[i] for i in range(n_loops)) + bottom_room
        width = left_room + sum(loop_widths) + sum(gaps) + right_room
        if max(height, width) > 30:
            return None

        boxes = []
        c0 = left_room
        for i in range(n_loops):
            r0 = top_room + row_offsets[i]
            r1 = r0 + loop_heights[i] - 1
            c1 = c0 + loop_widths[i] - 1
            boxes.append((r0, c0, r1, c1))
            if i + 1 < n_loops:
                c0 = c1 + 1 + gaps[i]
        return (height, width), boxes, orientation

    top_room = _randint(rng, 3, 5)
    bottom_room = _randint(rng, 3, 5)
    left_room = _randint(rng, 4, 6)
    right_room = _randint(rng, 4, 6)
    col_offsets = [_randint(rng, 0, 1) for _ in range(n_loops)]
    height = top_room + sum(loop_heights) + sum(gaps) + bottom_room
    width = left_room + max(col_offsets[i] + loop_widths[i] for i in range(n_loops)) + right_room
    if max(height, width) > 30:
        return None

    boxes = []
    r0 = top_room
    for i in range(n_loops):
        c0 = left_room + col_offsets[i]
        r1 = r0 + loop_heights[i] - 1
        c1 = c0 + loop_widths[i] - 1
        boxes.append((r0, c0, r1, c1))
        if i + 1 < n_loops:
            r0 = r1 + 1 + gaps[i]
    return (height, width), boxes, orientation


def _connect_boxes(
    rng: np.random.Generator,
    skeleton: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    orientation: str,
):
    connected_edges: list[list[tuple[int, int]]] = []
    blocked_anchors = [set() for _ in boxes]
    connect_flags = [bool(rng.random() < 0.65) for _ in range(len(boxes) - 1)]
    if connect_flags and rng.random() < 0.75 and not any(connect_flags):
        connect_flags[_randint(rng, 0, len(connect_flags) - 1)] = True

    for i, do_connect in enumerate(connect_flags):
        if not do_connect:
            continue
        a = boxes[i]
        b = boxes[i + 1]
        if orientation == "horizontal":
            rows = sorted(set(range(a[0] + 1, a[2])) & set(range(b[0] + 1, b[2])))
            if not rows:
                return None
            row = int(rng.choice(rows))
            cells = [(row, c) for c in range(a[3], b[1] + 1)]
            blocked_anchors[i].add((row, a[3]))
            blocked_anchors[i + 1].add((row, b[1]))
        else:
            cols = sorted(set(range(a[1] + 1, a[3])) & set(range(b[1] + 1, b[3])))
            if not cols:
                return None
            col = int(rng.choice(cols))
            cells = [(r, col) for r in range(a[2], b[0] + 1)]
            blocked_anchors[i].add((a[2], col))
            blocked_anchors[i + 1].add((b[0], col))

        for r, c in cells:
            skeleton[r, c] = True
        connected_edges.append(cells)
    return connected_edges, blocked_anchors


def _marker_positions(
    rng: np.random.Generator,
    boxes: list[tuple[int, int, int, int]],
    fg_color: int,
) -> list[tuple[int, int, int]]:
    colors = [color for color in MARKER_COLORS if color != fg_color]
    rng.shuffle(colors)
    markers = []
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        r = _randint(rng, r0 + 1, r1 - 1)
        c = _randint(rng, c0 + 1, c1 - 1)
        markers.append((r, c, int(colors[i])))
    return markers


def _perp_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _try_attach_tree(
    rng: np.random.Generator,
    mask: np.ndarray,
    anchor: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]] | None:
    for trunk_len in rng.permutation(np.arange(1, 5)):
        trunk = [
            (anchor[0] + direction[0] * step, anchor[1] + direction[1] * step)
            for step in range(1, int(trunk_len) + 1)
        ]
        if not trunk:
            continue
        cells = list(trunk)
        edges: dict[tuple[int, int], set[tuple[int, int]]] = {}

        def link(a: tuple[int, int], b: tuple[int, int]) -> None:
            edges.setdefault(a, set()).add(b)
            edges.setdefault(b, set()).add(a)

        link(anchor, trunk[0])
        for prev, cur in zip(trunk, trunk[1:]):
            link(prev, cur)

        twig_added = False
        if len(trunk) >= 2 and rng.random() < 0.55:
            pivot = trunk[_randint(rng, 0, len(trunk) - 1)]
            perp_options = _perp_dirs(direction)
            rng.shuffle(perp_options)
            for twig_dir in perp_options:
                for twig_len in rng.permutation(np.arange(1, 3)):
                    twig = [
                        (pivot[0] + twig_dir[0] * step, pivot[1] + twig_dir[1] * step)
                        for step in range(1, int(twig_len) + 1)
                    ]
                    if any(cell in cells for cell in twig):
                        continue
                    ok = True
                    for cell in twig:
                        if not _in_bounds(mask.shape, cell[0], cell[1]) or mask[cell]:
                            ok = False
                            break
                    if not ok:
                        continue
                    cells.extend(twig)
                    link(pivot, twig[0])
                    for prev, cur in zip(twig, twig[1:]):
                        link(prev, cur)
                    twig_added = True
                    break
                if twig_added:
                    break

        cell_set = set(cells)
        ok = True
        for r, c in cells:
            if not _in_bounds(mask.shape, r, c) or mask[r, c]:
                ok = False
                break
            allowed = edges.get((r, c), set())
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(mask.shape, nr, nc):
                    ok = False
                    break
                occupied = mask[nr, nc] or (nr, nc) in cell_set
                if occupied and (nr, nc) not in allowed:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c in cells:
            mask[r, c] = True
        return cells
    return None


def _loop_anchor_candidates(
    bbox: tuple[int, int, int, int],
    blocked_anchors: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in range(c0 + 1, c1):
        candidates.append(((r0, c), (-1, 0)))
        candidates.append(((r1, c), (1, 0)))
    for r in range(r0 + 1, r1):
        candidates.append(((r, c0), (0, -1)))
        candidates.append(((r, c1), (0, 1)))
    return [item for item in candidates if item[0] not in blocked_anchors]


def _channel_anchor_candidates(
    channel_cells: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(channel_cells) <= 2:
        return []
    anchors = channel_cells[1:-1]
    if len({r for r, _ in channel_cells}) == 1:
        return [((r, c), d) for r, c in anchors for d in [(-1, 0), (1, 0)]]
    return [((r, c), d) for r, c in anchors for d in [(0, -1), (0, 1)]]


def _add_branches(
    rng: np.random.Generator,
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    blocked_anchors: list[set[tuple[int, int]]],
    channels: list[list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    branch_cells: list[tuple[int, int]] = []

    for bbox, blocked in zip(boxes, blocked_anchors):
        candidates = _loop_anchor_candidates(bbox, blocked)
        rng.shuffle(candidates)
        target = _randint(rng, 1, 2)
        placed = 0
        for anchor, direction in candidates:
            new_cells = _try_attach_tree(rng, mask, anchor, direction)
            if new_cells is None:
                continue
            branch_cells.extend(new_cells)
            placed += 1
            if placed >= target:
                break
        if placed == 0:
            return []

    for channel in channels:
        if rng.random() >= 0.7:
            continue
        candidates = _channel_anchor_candidates(channel)
        rng.shuffle(candidates)
        for anchor, direction in candidates:
            new_cells = _try_attach_tree(rng, mask, anchor, direction)
            if new_cells is not None:
                branch_cells.extend(new_cells)
                break

    return branch_cells


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    skeleton: np.ndarray,
    branch_cells: list[tuple[int, int]],
    channels: list[list[tuple[int, int]]],
    markers: list[tuple[int, int, int]],
    fg_color: int,
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    structure_mask = inp == fg_color
    alive = _prune_to_fixpoint(structure_mask)
    if not np.array_equal(alive, skeleton):
        return False
    if any(alive[r, c] for r, c in branch_cells):
        return False
    if len(_enclosed_voids(skeleton)) != len(boxes):
        return False
    marker_positions = {(r, c) for r, c, _ in markers}
    for coords in _enclosed_voids(skeleton):
        if sum((int(r), int(c)) in marker_positions for r, c in coords) != 1:
            return False
    if channels:
        loop_border = set()
        for r0, c0, r1, c1 in boxes:
            loop_border.update((r0, c) for c in range(c0, c1 + 1))
            loop_border.update((r1, c) for c in range(c0, c1 + 1))
            loop_border.update((r, c0) for r in range(r0, r1 + 1))
            loop_border.update((r, c1) for r in range(r0, r1 + 1))
        bridge_interior = [
            cell
            for cells in channels
            for cell in cells[1:-1]
            if cell not in loop_border
        ]
        if not bridge_interior or not all(alive[r, c] for r, c in bridge_interior):
            return False
    return np.array_equal(main(inp), expected)


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    layout = _choose_layout(rng)
    if layout is None:
        return None
    shape, boxes, orientation = layout
    fg_color = int(rng.choice(MARKER_COLORS))

    skeleton = np.zeros(shape, dtype=bool)
    for bbox in boxes:
        _draw_box_border(skeleton, bbox)

    connected = _connect_boxes(rng, skeleton, boxes, orientation)
    if connected is None:
        return None
    channels, blocked_anchors = connected

    markers = _marker_positions(rng, boxes, fg_color)
    expected = _render_output(skeleton, fg_color, markers)

    input_mask = skeleton.copy()
    branch_cells = _add_branches(rng, input_mask, boxes, blocked_anchors, channels)
    if not branch_cells:
        return None

    inp = np.zeros(shape, dtype=int)
    inp[input_mask] = fg_color
    for r, c, color in markers:
        inp[r, c] = int(color)

    if not _validate_case(inp, expected, skeleton, branch_cells, channels, markers, fg_color, boxes):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid iterative-pruning loop input")
