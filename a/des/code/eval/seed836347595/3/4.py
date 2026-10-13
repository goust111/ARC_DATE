from __future__ import annotations

from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
PATTERN_COLORS = (0, 7, 8, 9)

BG = YELLOW
LINE_COLORS = (BLUE, RED)
SIDES = ("top", "right", "bottom", "left")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

FRAGMENT_COUNT_RANGE = (3, 6)
INNER_HEIGHT_RANGE = (2, 5)
INNER_WIDTH_RANGE = (2, 5)
SLOT_GAP_RANGE = (2, 5)
TOP_SLACK_RANGE = (10, 17)
WIDTH_SLACK_RANGE = (4, 13)
BOTTOM_MARGIN_RANGE = (1, 3)
PLACEMENT_ATTEMPTS = 250
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_from_component(
    grid: np.ndarray,
    coords: np.ndarray,
    color: int,
    interior_color: int | None = None,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _frame_mask(h, w)
    if not np.all(patch[border] == color):
        return None
    if np.any(patch[~border] == color):
        return None
    if interior_color is not None and not np.all(patch[~border] == interior_color):
        return None
    return {"bbox": (r0, c0, r1, c1), "patch": patch.copy()}


def _find_frames(
    grid: np.ndarray,
    color: int,
    interior_color: int | None = None,
) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for coords in _components(grid == color):
        info = _frame_from_component(grid, coords, color, interior_color=interior_color)
        if info is not None:
            frames.append(info)
    return frames


def _border_owner_map(shape: tuple[int, int], frames: list[dict[str, object]]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        owner[r0, c0 : c1 + 1] = idx
        owner[r1, c0 : c1 + 1] = idx
        owner[r0 : r1 + 1, c0] = idx
        owner[r0 : r1 + 1, c1] = idx
    return owner


def _adjacent_to_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return (r0 <= r <= r1 and c in (c0 - 1, c1 + 1)) or (c0 <= c <= c1 and r in (r0 - 1, r1 + 1))


def _build_fragment_graph(grid: np.ndarray, fragments: list[dict[str, object]]) -> list[list[tuple[int, int]]]:
    owner = _border_owner_map(grid.shape, fragments)
    graph: list[list[tuple[int, int]]] = [[] for _ in fragments]

    for color in LINE_COLORS:
        for coords in _components(grid == color):
            touched: set[int] = set()
            for r, c in coords:
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        idx = int(owner[nr, nc])
                        if idx >= 0:
                            touched.add(idx)
            if len(touched) != 2:
                raise AssertionError("each line component must attach to exactly two fragments")
            a, b = sorted(touched)
            graph[a].append((b, color))
            graph[b].append((a, color))

    return graph


def _traversal_order(
    graph: list[list[tuple[int, int]]],
    start_idx: int,
) -> tuple[list[int], list[int | None]]:
    n = len(graph)
    degrees = sorted(len(neighbors) for neighbors in graph)
    assert degrees == [1, 1] + [2] * (n - 2), "fragments must form a simple chain"

    order = [start_idx]
    incoming: list[int | None] = [None]
    visited = {start_idx}
    prev = -1
    cur = start_idx

    while len(order) < n:
        options = [(nxt, color) for nxt, color in graph[cur] if nxt != prev]
        assert len(options) == 1, "chain traversal must have one forward choice"
        nxt, color = options[0]
        assert nxt not in visited, "chain must not revisit fragments"
        order.append(int(nxt))
        incoming.append(int(color))
        visited.add(int(nxt))
        prev, cur = cur, int(nxt)

    assert len(visited) == n
    return order, incoming


def _assemble_output(
    fragments: list[dict[str, object]],
    slots: list[dict[str, object]],
    order: list[int],
    incoming: list[int | None],
) -> np.ndarray:
    ordered_slots = sorted(slots, key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    assert len(ordered_slots) == len(order)

    frag_shape = np.asarray(fragments[order[0]]["patch"], dtype=int).shape
    for fragment in fragments:
        assert np.asarray(fragment["patch"], dtype=int).shape == frag_shape
    for slot in ordered_slots:
        r0, c0, r1, c1 = map(int, slot["bbox"])
        assert (r1 - r0 + 1, c1 - c0 + 1) == frag_shape

    row0 = min(int(slot["bbox"][0]) for slot in ordered_slots)
    col0 = min(int(slot["bbox"][1]) for slot in ordered_slots)
    row1 = max(int(slot["bbox"][2]) for slot in ordered_slots)
    col1 = max(int(slot["bbox"][3]) for slot in ordered_slots)

    out = np.full((row1 - row0 + 1, col1 - col0 + 1), BG, dtype=int)
    for slot, frag_idx, color in zip(ordered_slots, order, incoming):
        patch = np.asarray(fragments[frag_idx]["patch"], dtype=int)
        if color == RED:
            patch = np.fliplr(patch)
        r0, c0, r1, c1 = map(int, slot["bbox"])
        out[r0 - row0 : r1 - row0 + 1, c0 - col0 : c1 - col0 + 1] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the colored connector chain from the green-marked start fragment,
    flip fragments reached through red edges, and place them left-to-right into
    the bottom magenta slots. The returned grid is the compact crop of that slot row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fragments = _find_frames(g, GREY)
    slots = _find_frames(g, MAGENTA, interior_color=BG)
    assert fragments, "expected grey-framed fragments"
    assert len(slots) == len(fragments), "slot count must match fragment count"

    green_pixels = np.argwhere(g == GREEN)
    assert len(green_pixels) == 1, "expected exactly one green start marker"
    marker = tuple(map(int, green_pixels[0]))

    start_candidates = [idx for idx, fragment in enumerate(fragments) if _adjacent_to_bbox(marker, fragment["bbox"])]
    assert len(start_candidates) == 1, "green marker must identify exactly one fragment"
    start_idx = int(start_candidates[0])

    graph = _build_fragment_graph(g, fragments)
    order, incoming = _traversal_order(graph, start_idx)
    return _assemble_output(fragments, slots, order, incoming)


def _reserve_rect(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _reserve_path(mask: np.ndarray, path: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in path:
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _frame_patch(pattern: np.ndarray) -> np.ndarray:
    out = np.full((pattern.shape[0] + 2, pattern.shape[1] + 2), GREY, dtype=int)
    out[1:-1, 1:-1] = pattern
    return out


def _random_pattern(inner_h: int, inner_w: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        color_count = int(rng.integers(2, min(4, len(PATTERN_COLORS)) + 1))
        colors = rng.choice(PATTERN_COLORS, size=color_count, replace=False)
        pattern = rng.choice(colors, size=(inner_h, inner_w)).astype(int)
        if len(np.unique(pattern)) < 2:
            continue
        if np.array_equal(pattern, np.fliplr(pattern)):
            continue
        return pattern

    pattern = np.full((inner_h, inner_w), PATTERN_COLORS[0], dtype=int)
    pattern[:, -1] = PATTERN_COLORS[1]
    pattern[0, 0] = PATTERN_COLORS[2]
    assert not np.array_equal(pattern, np.fliplr(pattern))
    return pattern


def _place_fragment_bboxes(
    shape: tuple[int, int],
    count: int,
    frag_h: int,
    frag_w: int,
    slot_bboxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    for bbox in slot_bboxes:
        _reserve_rect(blocked, bbox, pad=2)

    max_r0 = min(r0 for r0, _, _, _ in slot_bboxes) - frag_h - 3
    min_r0 = 2
    min_c0 = 2
    max_c0 = w - frag_w - 3
    if max_r0 < min_r0 or max_c0 < min_c0:
        return None

    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(min_r0, max_r0 + 1))
            c0 = int(rng.integers(min_c0, max_c0 + 1))
            bbox = (r0, c0, r0 + frag_h - 1, c0 + frag_w - 1)
            rr0 = max(0, bbox[0] - 2)
            cc0 = max(0, bbox[1] - 2)
            rr1 = min(h - 1, bbox[2] + 2)
            cc1 = min(w - 1, bbox[3] + 2)
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            bboxes.append(bbox)
            _reserve_rect(blocked, bbox, pad=2)
            placed = True
            break
        if not placed:
            return None

    leftmost = min(bbox[1] for bbox in bboxes)
    if sum(int(bbox[1] == leftmost) for bbox in bboxes) != 1:
        return None
    return bboxes


def _ports_for_bbox(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "top": (r0 - 1, (c0 + c1) // 2),
        "right": ((r0 + r1) // 2, c1 + 1),
        "bottom": (r1 + 1, (c0 + c1) // 2),
        "left": ((r0 + r1) // 2, c0 - 1),
    }


def _shortest_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    sr, sc = start
    er, ec = end
    if not (0 <= sr < h and 0 <= sc < w and 0 <= er < h and 0 <= ec < w):
        return None
    if blocked[sr, sc] or blocked[er, ec]:
        return None

    seen = np.zeros((h, w), dtype=bool)
    parent_r = np.full((h, w), -1, dtype=int)
    parent_c = np.full((h, w), -1, dtype=int)
    queue = deque([start])
    seen[sr, sc] = True
    dirs = list(DIRS4)
    rng.shuffle(dirs)

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            break
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                queue.append((nr, nc))

    if not seen[er, ec]:
        return None

    path: list[tuple[int, int]] = []
    cur = (er, ec)
    while True:
        path.append(cur)
        if cur == start:
            path.reverse()
            return path
        pr = int(parent_r[cur[0], cur[1]])
        pc = int(parent_c[cur[0], cur[1]])
        cur = (pr, pc)


def _route_edges(
    grid_shape: tuple[int, int],
    fragment_bboxes: list[tuple[int, int, int, int]],
    slot_bboxes: list[tuple[int, int, int, int]],
    order: list[int],
    edge_colors: list[int],
    start_idx: int,
    rng: np.random.Generator,
) -> list[tuple[list[tuple[int, int]], int]] | None:
    blocked = np.zeros(grid_shape, dtype=bool)
    for bbox in fragment_bboxes:
        _reserve_rect(blocked, bbox, pad=1)
    for bbox in slot_bboxes:
        _reserve_rect(blocked, bbox, pad=1)

    ports = [_ports_for_bbox(bbox) for bbox in fragment_bboxes]
    used_sides: list[set[str]] = [set() for _ in fragment_bboxes]
    used_sides[start_idx].add("left")

    routed: list[tuple[list[tuple[int, int]], int]] = []
    for (src, dst), color in zip(zip(order, order[1:]), edge_colors):
        pairs: list[tuple[str, tuple[int, int], str, tuple[int, int]]] = []
        for src_side, src_port in ports[src].items():
            if src_side in used_sides[src]:
                continue
            for dst_side, dst_port in ports[dst].items():
                if dst_side in used_sides[dst]:
                    continue
                distance = abs(src_port[0] - dst_port[0]) + abs(src_port[1] - dst_port[1])
                pairs.append((src_side, src_port, dst_side, dst_port, distance))
        rng.shuffle(pairs)
        pairs.sort(key=lambda item: int(item[4]))

        chosen: tuple[str, str, list[tuple[int, int]]] | None = None
        for src_side, src_port, dst_side, dst_port, _distance in pairs:
            local_blocked = blocked.copy()
            local_blocked[src_port] = False
            local_blocked[dst_port] = False
            path = _shortest_path(local_blocked, src_port, dst_port, rng)
            if path is None or len(path) < 2:
                continue
            chosen = (src_side, dst_side, path)
            break

        if chosen is None:
            return None

        src_side, dst_side, path = chosen
        used_sides[src].add(src_side)
        used_sides[dst].add(dst_side)
        _reserve_path(blocked, path)
        routed.append((path, color))

    return routed


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, fill: int | None = None) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if fill is not None:
        grid[r0 + 1 : r1, c0 + 1 : c1] = fill


def _slot_bboxes(
    height: int,
    width: int,
    frag_h: int,
    frag_w: int,
    count: int,
    gap: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    total_w = count * frag_w + (count - 1) * gap
    left = int(rng.integers(1, width - total_w))
    bottom_margin = int(rng.integers(*BOTTOM_MARGIN_RANGE))
    row0 = height - bottom_margin - frag_h
    return [
        (row0, left + idx * (frag_w + gap), row0 + frag_h - 1, left + idx * (frag_w + gap) + frag_w - 1)
        for idx in range(count)
    ]


def _int_range(bounds: tuple[int, int], rng: np.random.Generator) -> int:
    return int(rng.integers(bounds[0], bounds[1]))


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = _int_range(FRAGMENT_COUNT_RANGE, rng)
    inner_h = _int_range(INNER_HEIGHT_RANGE, rng)
    inner_w = _int_range(INNER_WIDTH_RANGE, rng)
    frag_h = inner_h + 2
    frag_w = inner_w + 2
    gap = _int_range(SLOT_GAP_RANGE, rng)

    slot_span_w = count * frag_w + (count - 1) * gap
    width = slot_span_w + _int_range(WIDTH_SLACK_RANGE, rng)
    top_slack = _int_range(TOP_SLACK_RANGE, rng)
    height = top_slack + frag_h + 3
    if width < slot_span_w + 3:
        return None

    slot_bboxes = _slot_bboxes(height, width, frag_h, frag_w, count, gap, rng)
    fragment_bboxes = _place_fragment_bboxes((height, width), count, frag_h, frag_w, slot_bboxes, rng)
    if fragment_bboxes is None:
        return None

    start_idx = min(range(count), key=lambda idx: (fragment_bboxes[idx][1], fragment_bboxes[idx][0]))
    spatial = sorted(range(count), key=lambda idx: (fragment_bboxes[idx][1], fragment_bboxes[idx][0]))
    rest = [idx for idx in range(count) if idx != start_idx]
    for _ in range(20):
        rng.shuffle(rest)
        order = [start_idx, *rest]
        if order != spatial:
            break
    else:
        return None

    edge_colors = [int(rng.choice(LINE_COLORS)) for _ in range(count - 1)]
    if count >= 3 and len(set(edge_colors)) == 1:
        edge_colors[-1] = BLUE if edge_colors[-1] == RED else RED

    patterns: list[np.ndarray] = []
    seen_patterns: set[bytes] = set()
    for _ in range(count):
        for _ in range(120):
            pattern = _random_pattern(inner_h, inner_w, rng)
            sig = pattern.tobytes()
            if sig in seen_patterns:
                continue
            patterns.append(pattern)
            seen_patterns.add(sig)
            break
        else:
            return None

    fragment_patches = [_frame_patch(pattern) for pattern in patterns]

    routed = _route_edges((height, width), fragment_bboxes, slot_bboxes, order, edge_colors, start_idx, rng)
    if routed is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for bbox, patch in zip(fragment_bboxes, fragment_patches):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    for bbox in slot_bboxes:
        _draw_frame(grid, bbox, MAGENTA, fill=BG)

    for path, color in routed:
        for r, c in path:
            grid[r, c] = color

    green_r, green_c = _ports_for_bbox(fragment_bboxes[start_idx])["left"]
    grid[green_r, green_c] = GREEN

    fragments = [{"bbox": bbox, "patch": patch.copy()} for bbox, patch in zip(fragment_bboxes, fragment_patches)]
    slots = [{"bbox": bbox, "patch": None} for bbox in slot_bboxes]
    incoming = [None] + edge_colors
    expected = _assemble_output(fragments, slots, order, incoming)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
