import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in {BG, GREY})
MAX_GENERATION_ATTEMPTS = 300
BASE_CANVAS = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, skip_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in skip_colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(len(coords)),
                }
            )
    return comps


def _find_hint_color(components: list[dict]) -> int:
    large_colors = {int(comp["color"]) for comp in components if int(comp["area"]) > 1}
    hint_candidates = [comp for comp in components if int(comp["area"]) == 1 and int(comp["color"]) not in large_colors]
    assert len(hint_candidates) == 1, "expected exactly one singleton color hint"
    return int(hint_candidates[0]["color"])


def _find_port(sprite: dict, grid: np.ndarray) -> dict:
    h, w = grid.shape
    hits = []
    for r, c in sprite["coords"]:
        for name, dr, dc in (("N", -1, 0), ("S", 1, 0), ("W", 0, -1), ("E", 0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                hits.append({"dir": name, "pixel": (int(r), int(c)), "outside": (nr, nc)})
    assert len(hits) == 1, f"expected exactly one grey port, got {hits}"
    return hits[0]


def _match_horizontal(grid: np.ndarray, src: dict, dst: dict, bg: int) -> tuple[int, int, int, int] | None:
    sr0, sc0, sr1, sc1 = src["bbox"]
    dr0, dc0, dr1, dc1 = dst["bbox"]
    if (sr0, sr1) != (dr0, dr1):
        return None
    if src["mask"].shape != dst["mask"].shape:
        return None
    if not (sc1 < dc0):
        return None
    if src["port"]["dir"] != "E" or dst["port"]["dir"] != "W":
        return None
    port_row = src["port"]["pixel"][0]
    if dst["port"]["pixel"][0] != port_row:
        return None

    w = src["mask"].shape[1]
    row_segment = grid[port_row, sc1 + 1 : dc0]
    if row_segment.size < w + 2:
        return None

    left_len = 0
    while left_len < row_segment.size and int(row_segment[left_len]) == GREY:
        left_len += 1
    if left_len == 0:
        return None
    bg_len = 0
    while left_len + bg_len < row_segment.size and int(row_segment[left_len + bg_len]) == bg:
        bg_len += 1
    if bg_len != w:
        return None
    if left_len + bg_len >= row_segment.size:
        return None
    if not np.all(row_segment[left_len + bg_len :] == GREY):
        return None

    mc0 = sc1 + 1 + left_len
    mc1 = mc0 + w - 1
    if not np.all(grid[sr0 : sr1 + 1, mc0 : mc1 + 1] == bg):
        return None
    return sr0, mc0, sr1, mc1


def _match_vertical(grid: np.ndarray, src: dict, dst: dict, bg: int) -> tuple[int, int, int, int] | None:
    sr0, sc0, sr1, sc1 = src["bbox"]
    dr0, dc0, dr1, dc1 = dst["bbox"]
    if (sc0, sc1) != (dc0, dc1):
        return None
    if src["mask"].shape != dst["mask"].shape:
        return None
    if not (sr1 < dr0):
        return None
    if src["port"]["dir"] != "S" or dst["port"]["dir"] != "N":
        return None
    port_col = src["port"]["pixel"][1]
    if dst["port"]["pixel"][1] != port_col:
        return None

    h = src["mask"].shape[0]
    col_segment = grid[sr1 + 1 : dr0, port_col]
    if col_segment.size < h + 2:
        return None

    top_len = 0
    while top_len < col_segment.size and int(col_segment[top_len]) == GREY:
        top_len += 1
    if top_len == 0:
        return None
    bg_len = 0
    while top_len + bg_len < col_segment.size and int(col_segment[top_len + bg_len]) == bg:
        bg_len += 1
    if bg_len != h:
        return None
    if top_len + bg_len >= col_segment.size:
        return None
    if not np.all(col_segment[top_len + bg_len :] == GREY):
        return None

    mr0 = sr1 + 1 + top_len
    mr1 = mr0 + h - 1
    if not np.all(grid[mr0 : mr1 + 1, sc0 : sc1 + 1] == bg):
        return None
    return mr0, sc0, mr1, sc1


def _find_pairs(grid: np.ndarray, sprites: list[dict], bg: int) -> list[dict]:
    for sprite in sprites:
        sprite["port"] = _find_port(sprite, grid)

    east = [sp for sp in sprites if sp["port"]["dir"] == "E"]
    west = [sp for sp in sprites if sp["port"]["dir"] == "W"]
    south = [sp for sp in sprites if sp["port"]["dir"] == "S"]
    north = [sp for sp in sprites if sp["port"]["dir"] == "N"]
    assert len(east) == len(west)
    assert len(south) == len(north)

    pairs: list[dict] = []
    used: set[int] = set()

    for src in sorted(east, key=lambda sp: sp["bbox"]):
        src_id = id(src)
        if src_id in used:
            continue
        matches = []
        for dst in west:
            if id(dst) in used:
                continue
            mid_bbox = _match_horizontal(grid, src, dst, bg)
            if mid_bbox is not None:
                matches.append((dst, mid_bbox))
        assert len(matches) == 1, f"horizontal pairing ambiguous for {src['bbox']}"
        dst, mid_bbox = matches[0]
        pairs.append({"axis": "h", "src": src, "dst": dst, "mid_bbox": mid_bbox})
        used.add(src_id)
        used.add(id(dst))

    for src in sorted(south, key=lambda sp: sp["bbox"]):
        src_id = id(src)
        if src_id in used:
            continue
        matches = []
        for dst in north:
            if id(dst) in used:
                continue
            mid_bbox = _match_vertical(grid, src, dst, bg)
            if mid_bbox is not None:
                matches.append((dst, mid_bbox))
        assert len(matches) == 1, f"vertical pairing ambiguous for {src['bbox']}"
        dst, mid_bbox = matches[0]
        pairs.append({"axis": "v", "src": src, "dst": dst, "mid_bbox": mid_bbox})
        used.add(src_id)
        used.add(id(dst))

    assert len(used) == len(sprites), "every sprite should belong to exactly one pair"
    return pairs


def _intermediate_mask(src_mask: np.ndarray, dst_mask: np.ndarray, axis: str) -> np.ndarray:
    src_only = np.argwhere(src_mask & ~dst_mask)
    dst_only = np.argwhere(dst_mask & ~src_mask)
    assert len(src_only) == len(dst_only) and len(src_only) > 0 and len(src_only) % 2 == 0

    key = (lambda rc: (int(rc[1]), int(rc[0]))) if axis == "h" else (lambda rc: (int(rc[0]), int(rc[1])))
    src_sorted = sorted((tuple(map(int, rc)) for rc in src_only), key=key)
    dst_sorted = sorted((tuple(map(int, rc)) for rc in dst_only), key=key)
    half = len(src_sorted) // 2

    out = src_mask & dst_mask
    for r, c in src_sorted[:half]:
        out[r, c] = True
    for r, c in dst_sorted[:half]:
        out[r, c] = True
    return out


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    components = _find_components(g, {bg, GREY})
    hint_color = _find_hint_color(components)
    sprites = [comp for comp in components if int(comp["area"]) > 1]
    assert sprites, "expected at least one sprite pair"

    pairs = _find_pairs(g, sprites, bg)
    out = g.copy()
    for pair in pairs:
        mid = _intermediate_mask(pair["src"]["mask"], pair["dst"]["mask"], pair["axis"])
        r0, c0, _r1, _c1 = pair["mid_bbox"]
        sub = out[r0 : r0 + mid.shape[0], c0 : c0 + mid.shape[1]]
        sub[mid] = hint_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a halfway morph sprite inside each path gap between aligned endpoint sprites.

    Input structure:
    - The background is black, and grey one-cell paths point from a source sprite toward a target sprite.
    - Each pair contains two same-size colored sprites with one grey port on the facing side, plus a same-size blank rectangle between two grey path segments.
    - A single isolated hint pixel provides the third color used for every inserted middle sprite.

    Rule:
    - Find every pair of sprites whose grey ports face each other across a horizontal or vertical gap and whose bounding boxes have the same size.
    - Treat the left/top sprite as the source and the right/bottom sprite as the target, then compare their normalized shape masks inside the shared bounding-box size.
    - Keep the pixels shared by both masks, keep the source-only half that lies nearer the source side, and add the target-only half that lies nearer the source side, where "nearer" is measured along the path axis.
    - Paint that intermediate mask in the blank middle rectangle using the hint color, leaving the original sprites, paths, and hint pixel unchanged.

    Key insight:
    - The grey path does not just connect two objects: it also marks an empty slot whose inserted sprite must be computed from a pixelwise source-vs-target shape difference ordered along the path direction.
    """
    return _solve_grid(grid)


def _make_core_mask(h: int, w: int, axis: str) -> tuple[np.ndarray, int]:
    mask = np.zeros((h, w), dtype=bool)
    if axis == "h":
        port = random.randint(1, h - 2) if h >= 3 else 0
        mask[port, :] = True
        for c in random.sample(range(w), k=random.randint(1, min(2, w))):
            mask[: port + 1, c] = True
        for c in random.sample(range(w), k=random.randint(1, min(2, w))):
            mask[port:h, c] = True
        for _ in range(random.randint(0, 2)):
            c = random.randrange(w)
            if random.random() < 0.5 and port > 0:
                r = random.randint(0, port - 1)
                mask[r : port + 1, c] = True
            elif port < h - 1:
                r = random.randint(port + 1, h - 1)
                mask[port : r + 1, c] = True
    else:
        port = random.randint(1, w - 2) if w >= 3 else 0
        mask[:, port] = True
        for r in random.sample(range(h), k=random.randint(1, min(2, h))):
            mask[r, : port + 1] = True
        for r in random.sample(range(h), k=random.randint(1, min(2, h))):
            mask[r, port:w] = True
        for _ in range(random.randint(0, 2)):
            r = random.randrange(h)
            if random.random() < 0.5 and port > 0:
                c = random.randint(0, port - 1)
                mask[r, c : port + 1] = True
            elif port < w - 1:
                c = random.randint(port + 1, w - 1)
                mask[r, port : c + 1] = True
    return mask, port


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                cells.append((r, c))
    return cells


def _pick_distinct_axis_cells(cells: list[tuple[int, int]], axis_index: int, need: int) -> list[tuple[int, int]] | None:
    buckets: dict[int, list[tuple[int, int]]] = {}
    for cell in cells:
        buckets.setdefault(cell[axis_index], []).append(cell)
    if len(buckets) < need:
        return None
    coords = random.sample(list(buckets), need)
    return [random.choice(buckets[key]) for key in coords]


def _build_sprite_masks(h: int, w: int, axis: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    axis_index = 1 if axis == "h" else 0
    size_options = [4, 2]
    if (axis == "h" and w < 4) or (axis == "v" and h < 4):
        size_options = [2]

    for _ in range(120):
        core, port = _make_core_mask(h, w, axis)
        frontier = _frontier_cells(core)
        if len(frontier) < 4:
            continue
        for need in random.sample(size_options, k=len(size_options)):
            for _ in range(40):
                a_only = _pick_distinct_axis_cells(frontier, axis_index, need)
                if a_only is None:
                    break
                a_set = set(a_only)
                remaining = [cell for cell in frontier if cell not in a_set]
                b_only = _pick_distinct_axis_cells(remaining, axis_index, need)
                if b_only is None:
                    continue
                src = core.copy()
                dst = core.copy()
                for r, c in a_only:
                    src[r, c] = True
                for r, c in b_only:
                    dst[r, c] = True
                if np.array_equal(src, dst):
                    continue
                return src, dst, port
    return None


def _build_group(color_a: int, color_b: int) -> np.ndarray | None:
    axis = random.choice(["h", "v"])
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    masks = _build_sprite_masks(h, w, axis)
    if masks is None:
        return None
    src_mask, dst_mask, port = masks

    if axis == "h":
        left_gap = random.randint(1, 4)
        right_gap = random.randint(1, 4)
        total_w = 3 * w + left_gap + right_gap
        grid = np.full((h, total_w), BG, dtype=int)
        src_slice = grid[:, :w]
        src_slice[src_mask] = color_a
        mid_c0 = w + left_gap
        dst_c0 = 2 * w + left_gap + right_gap
        dst_slice = grid[:, dst_c0 : dst_c0 + w]
        dst_slice[dst_mask] = color_b
        grid[port, w:mid_c0] = GREY
        grid[port, mid_c0 + w : dst_c0] = GREY
        return grid

    top_gap = random.randint(1, 4)
    bottom_gap = random.randint(1, 4)
    total_h = 3 * h + top_gap + bottom_gap
    grid = np.full((total_h, w), BG, dtype=int)
    top_slice = grid[:h, :]
    top_slice[src_mask] = color_a
    mid_r0 = h + top_gap
    dst_r0 = 2 * h + top_gap + bottom_gap
    dst_slice = grid[dst_r0 : dst_r0 + h, :]
    dst_slice[dst_mask] = color_b
    grid[h:mid_r0, port] = GREY
    grid[mid_r0 + h : dst_r0, port] = GREY
    return grid


def _place_patch(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    mask = patch != BG
    for _ in range(250):
        top = random.randint(0, BASE_CANVAS - h)
        left = random.randint(0, BASE_CANVAS - w)
        r0 = max(0, top - 1)
        r1 = min(BASE_CANVAS, top + h + 1)
        c0 = max(0, left - 1)
        c1 = min(BASE_CANVAS, left + w + 1)
        if np.any(blocked[r0:r1, c0:c1]):
            continue
        canvas[top : top + h, left : left + w][mask] = patch[mask]
        blocked[r0:r1, c0:c1] = True
        return True
    return False


def _place_hint(canvas: np.ndarray, blocked: np.ndarray, hint_color: int) -> bool:
    candidates = np.argwhere(~blocked)
    if len(candidates) == 0:
        return False
    order = list(range(len(candidates)))
    random.shuffle(order)
    for idx in order:
        r, c = map(int, candidates[idx])
        if int(canvas[r, c]) != BG:
            continue
        canvas[r, c] = hint_color
        r0 = max(0, r - 1)
        r1 = min(BASE_CANVAS, r + 2)
        c0 = max(0, c - 1)
        c1 = min(BASE_CANVAS, c + 2)
        blocked[r0:r1, c0:c1] = True
        return True
    return False


def _crop_canvas(canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    pad_top = random.randint(0, min(4, int(r0)))
    pad_left = random.randint(0, min(4, int(c0)))
    pad_bottom = random.randint(0, min(4, int(canvas.shape[0] - 1 - r1)))
    pad_right = random.randint(0, min(4, int(canvas.shape[1] - 1 - c1)))
    return canvas[r0 - pad_top : r1 + pad_bottom + 1, c0 - pad_left : c1 + pad_right + 1]


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice([1, 2, 2, 3])
    colors = random.sample(PALETTE, 2 * group_count + 1)
    hint_color = colors[-1]
    group_colors = [(colors[2 * i], colors[2 * i + 1]) for i in range(group_count)]

    patches = []
    for color_a, color_b in group_colors:
        patch = _build_group(color_a, color_b)
        if patch is None:
            return None
        patches.append(patch)

    canvas = np.full((BASE_CANVAS, BASE_CANVAS), BG, dtype=int)
    blocked = np.zeros((BASE_CANVAS, BASE_CANVAS), dtype=bool)

    random.shuffle(patches)
    for patch in patches:
        if not _place_patch(canvas, blocked, patch):
            return None
    if not _place_hint(canvas, blocked, hint_color):
        return None

    cropped = _crop_canvas(canvas)
    try:
        out = _solve_grid(cropped)
    except AssertionError:
        return None
    if np.array_equal(out, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
