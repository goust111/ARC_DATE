import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _is_frame_component(cells, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    comp = set(cells)
    if comp != expected:
        return False
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return not np.any(interior == GREY)


def _extract_blocks(grid: np.ndarray, k: int):
    mask = (grid != BLACK) & (grid != GREY)
    comps = _find_components(mask)
    blocks = []
    for cells in comps:
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 + 1 > k or c1 - c0 + 1 > k:
            continue
        colors = {int(grid[r, c]) for r, c in cells}
        if len(colors) != 1:
            continue
        sprite = np.zeros((k, k), dtype=int)
        color = next(iter(colors))
        for r, c in cells:
            sprite[r - r0, c - c0] = 1
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "sprite": sprite,
                "top": r0,
            }
        )
    return blocks


def _trace_neighbor(grid: np.ndarray, block, direction, block_index_by_cell):
    r0, c0, r1, c1 = block["bbox"]
    dr, dc = direction
    candidates = []
    if dr == -1:
        for c in range(c0, c1 + 1):
            if r0 - 1 >= 0 and grid[r0 - 1, c] == GREY:
                candidates.append((r0 - 1, c))
    elif dr == 1:
        for c in range(c0, c1 + 1):
            if r1 + 1 < grid.shape[0] and grid[r1 + 1, c] == GREY:
                candidates.append((r1 + 1, c))
    elif dc == -1:
        for r in range(r0, r1 + 1):
            if c0 - 1 >= 0 and grid[r, c0 - 1] == GREY:
                candidates.append((r, c0 - 1))
    else:
        for r in range(r0, r1 + 1):
            if c1 + 1 < grid.shape[1] and grid[r, c1 + 1] == GREY:
                candidates.append((r, c1 + 1))
    if not candidates:
        return None
    sr, sc = candidates[0]
    cr, cc = sr, sc
    h, w = grid.shape
    while 0 <= cr < h and 0 <= cc < w and grid[cr, cc] == GREY:
        cr += dr
        cc += dc
    if not (0 <= cr < h and 0 <= cc < w):
        return None
    return block_index_by_cell.get((cr, cc))


def _build_graph(grid: np.ndarray, blocks):
    block_index_by_cell = {}
    for i, block in enumerate(blocks):
        r0, c0, r1, c1 = block["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                block_index_by_cell[(r, c)] = i

    adj = [set() for _ in blocks]
    for i, block in enumerate(blocks):
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            j = _trace_neighbor(grid, block, direction, block_index_by_cell)
            if j is None or j == i:
                continue
            adj[i].add(j)
            adj[j].add(i)
    return adj


def _group_indices(adj):
    seen = set()
    groups = []
    for i in range(len(adj)):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(comp)
    return groups


def _overlay_group(blocks, group, adj, k):
    chosen = min(group, key=lambda idx: (-len(adj[idx]), blocks[idx]["top"]))
    color = blocks[chosen]["color"]
    sprite = np.zeros((k, k), dtype=int)
    for idx in group:
        sprite |= blocks[idx]["sprite"]
    return np.where(sprite, color, BLACK), color, int(sprite.sum())


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_cells = _find_components(g == GREY)
    frames = []
    for cells in frame_cells:
        if _is_frame_component(cells, g):
            r0, c0, r1, c1 = _bbox(cells)
            frames.append((r0, c0, r1, c1))
    assert frames, "no output frames found"
    frames.sort()
    inner_h = frames[0][2] - frames[0][0] - 1
    inner_w = frames[0][3] - frames[0][1] - 1
    assert inner_h == inner_w
    k = inner_h
    assert all((r1 - r0 - 1 == k and c1 - c0 - 1 == k) for r0, c0, r1, c1 in frames)

    blocks = _extract_blocks(g, k)
    assert blocks, "no pattern blocks found"
    adj = _build_graph(g, blocks)
    groups = _group_indices(adj)
    assert len(groups) == len(frames)

    overlays = []
    for group in groups:
        sprite, _color, count = _overlay_group(blocks, group, adj, k)
        top = min(blocks[idx]["top"] for idx in group)
        overlays.append((len(group), count, top, sprite))
    overlays.sort(key=lambda item: (-item[0], -item[1], item[2]))

    r0 = min(f[0] for f in frames)
    c0 = min(f[1] for f in frames)
    r1 = max(f[2] for f in frames)
    c1 = max(f[3] for f in frames)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for fr0, fc0, fr1, fc1 in frames:
        rr0 = fr0 - r0
        cc0 = fc0 - c0
        out[rr0 : rr0 + (fr1 - fr0 + 1), cc0 : cc0 + (fc1 - fc0 + 1)] = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    for (fr0, fc0, fr1, fc1), (_size, _count, _top, sprite) in zip(frames, overlays):
        rr0 = fr0 - r0 + 1
        cc0 = fc0 - c0 + 1
        out[rr0 : rr0 + k, cc0 : cc0 + k] = sprite
    return out


def _random_connected_shape(k: int, rng: random.Random):
    target = rng.randint(2, max(2, k * k - 1))
    cells = {(rng.randrange(k), rng.randrange(k))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < k and 0 <= nc < k]
        rng.shuffle(nbrs)
        cells.add(nbrs[0])
    sprite = np.zeros((k, k), dtype=int)
    for r, c in cells:
        sprite[r, c] = 1
    return sprite


def _paint_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, color: int):
    k = sprite.shape[0]
    for r in range(k):
        for c in range(k):
            if sprite[r, c]:
                grid[top + r, left + c] = color


def _draw_line(grid: np.ndarray, a, b):
    ar, ac = a
    br, bc = b
    if ar == br:
        for c in range(min(ac, bc), max(ac, bc) + 1):
            grid[ar, c] = GREY
    else:
        for r in range(min(ar, br), max(ar, br) + 1):
            grid[r, ac] = GREY


def _layout_group(group_type: str, k: int, gap: int):
    if group_type == "singleton":
        blocks = [(0, 0)]
        edges = []
    elif group_type == "chain2":
        blocks = [(0, 0), (k + gap, 0)]
        edges = [(0, 1)]
    elif group_type == "chain3":
        blocks = [(0, 0), (k + gap, 0), (2 * (k + gap), 0)]
        edges = [(0, 1), (1, 2)]
    elif group_type == "chain4":
        blocks = [(0, 0), (k + gap, 0), (2 * (k + gap), 0), (3 * (k + gap), 0)]
        edges = [(0, 1), (1, 2), (2, 3)]
    else:
        center = (k + gap, k + gap)
        dirs = {
            "up": (0, k + gap),
            "down": (2 * (k + gap), k + gap),
            "left": (k + gap, 0),
            "right": (k + gap, 2 * (k + gap)),
        }
        arm_count = int(group_type[-1])
        chosen_dirs = ["up", "down", "left", "right"][:]
        blocks = [center] + [dirs[d] for d in chosen_dirs[:arm_count]]
        edges = [(0, i) for i in range(1, len(blocks))]
    h = max(r for r, _ in blocks) + k
    w = max(c for _, c in blocks) + k
    return blocks, edges, h, w


def _edge_points(pos_a, pos_b, k):
    ar, ac = pos_a
    br, bc = pos_b
    if ar == br:
        if ac < bc:
            return (ar + k // 2, ac + k), (br + k // 2, bc - 1)
        return (ar + k // 2, ac - 1), (br + k // 2, bc + k)
    if ar < br:
        return (ar + k, ac + k // 2), (br - 1, bc + k // 2)
    return (ar - 1, ac + k // 2), (br + k, bc + k // 2)


def _gen_attempt(rng: random.Random):
    group_type_choices = ["singleton", "chain2", "chain3", "chain4", "star2", "star3", "star4"]
    num_groups = rng.randint(2, 4)
    k = rng.randint(3, 4)
    groups = []
    total_blocks = 0
    tie_present = False
    used_colors = PALETTE[:]
    rng.shuffle(used_colors)

    for _ in range(num_groups):
        group_type = rng.choice(group_type_choices)
        blocks, edges, h, w = _layout_group(group_type, k, rng.randint(1, 3))
        n = len(blocks)
        total_blocks += n
        tie_present |= group_type == "chain4"
        groups.append(
            {
                "type": group_type,
                "blocks": blocks,
                "edges": edges,
                "h": h,
                "w": w,
            }
        )
    if total_blocks > len(PALETTE):
        return None

    frame_gap = rng.randint(1, 3)
    frame_w = k + 2
    frame_h = k + 2
    left_pad = 1
    between_group_gap = 2
    top_pad = 1
    band_heights = [g["h"] for g in groups]
    frames_total_h = num_groups * frame_h + (num_groups - 1) * frame_gap
    scene_h = sum(band_heights) + between_group_gap * (num_groups - 1)
    h = max(frames_total_h, scene_h) + 2 * top_pad
    left_w = max(g["w"] for g in groups) + left_pad + 2
    w = left_w + 2 + frame_w + 1
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)

    frame_x = w - frame_w - 1
    frame_y = top_pad
    frames = []
    for _ in range(num_groups):
        r0 = frame_y
        c0 = frame_x
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        frames.append((r0, c0, r1, c1))
        frame_y = r1 + 1 + frame_gap

    cur_y = top_pad
    sort_keys = []
    for group in groups:
        base_x = 1
        base_y = cur_y
        sprites = []
        degrees = [0] * len(group["blocks"])
        for a, b in group["edges"]:
            degrees[a] += 1
            degrees[b] += 1
        for i, (br, bc) in enumerate(group["blocks"]):
            if not used_colors:
                return None
            color = used_colors.pop()
            sprite = _random_connected_shape(k, rng)
            sprites.append({"pos": (base_y + br, base_x + bc), "color": color, "sprite": sprite, "deg": degrees[i]})

        overlay = np.zeros((k, k), dtype=int)
        winner = min(range(len(sprites)), key=lambda i: (-sprites[i]["deg"], sprites[i]["pos"][0]))
        for sprite in sprites:
            overlay |= sprite["sprite"]
        overlay_count = int(overlay.sum())
        sort_keys.append((len(sprites), overlay_count))

        for sprite in sprites:
            _paint_sprite(grid, sprite["sprite"], sprite["pos"][0], sprite["pos"][1], sprite["color"])
        for a, b in group["edges"]:
            p1, p2 = _edge_points(sprites[a]["pos"], sprites[b]["pos"], k)
            _draw_line(grid, p1, p2)
        cur_y += group["h"] + between_group_gap

    if len(set(sort_keys)) != len(sort_keys):
        return None

    # Keep the output side isolated from the left scene.
    if np.any(grid[:, frame_x - 1] != BLACK):
        return None

    # Reject accidental grey structures that look like extra frames.
    frame_like = []
    for cells in _find_components(grid == GREY):
        if _is_frame_component(cells, grid):
            frame_like.append(_bbox(cells))
    if sorted(frame_like) != sorted(frames):
        return None

    # Ensure block extraction and group recovery match the intended structure.
    recovered_blocks = _extract_blocks(grid, k)
    if len(recovered_blocks) != total_blocks:
        return None
    recovered_adj = _build_graph(grid, recovered_blocks)
    if len(_group_indices(recovered_adj)) != num_groups:
        return None
    if not tie_present and rng.random() < 0.5:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
