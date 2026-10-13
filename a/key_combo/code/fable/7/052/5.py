import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
WIRE_COLORS = [1, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500


# Minimal helpers adapted from arc_utils's canonical (r, c), bbox, crop, and component APIs.
def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segments must be axis-aligned")


def _path_from_points(points: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    path: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        if not path:
            path.extend(seg)
        else:
            path.extend(seg[1:])
    if len(set(path)) != len(path):
        return None
    return path


def _connected(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if not mask[start] or not mask[goal]:
        return False
    h, w = mask.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def _parse_panel(panel: np.ndarray) -> dict:
    assert panel.ndim == 2
    h, w = panel.shape
    assert h >= 4 and w >= 4
    border_mask = np.zeros_like(panel, dtype=bool)
    border_mask[[0, -1], :] = True
    border_mask[:, [0, -1]] = True
    assert np.all(panel[border_mask] == GREY)
    assert np.count_nonzero(panel == GREY) == int(border_mask.sum())
    assert not np.any(panel[1:-1, 1:-1] == GREY)

    interior = panel[1:-1, 1:-1]
    reds = [tuple(map(int, p)) for p in np.argwhere(interior == RED)]
    assert len(reds) == 2

    mono_colors = []
    colors = sorted(set(map(int, np.unique(interior))) - {BLACK, RED})
    for color in colors:
        allowed = interior == color
        if not any(0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc]
                   for nr, nc in neighbors4(*reds[0])):
            continue
        if not any(0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc]
                   for nr, nc in neighbors4(*reds[1])):
            continue
        traversable = allowed.copy()
        traversable[reds[0]] = True
        traversable[reds[1]] = True
        if _connected(traversable, reds[0], reds[1]):
            mono_colors.append(color)

    nonblack = interior != BLACK
    mixed_chain = _connected(nonblack, reds[0], reds[1]) and not mono_colors

    return {
        "panel": panel,
        "interior": interior,
        "reds": reds,
        "mono_colors": mono_colors,
        "mixed_chain": mixed_chain,
        "area": int(panel.shape[0] * panel.shape[1]),
        "wire_color_count": len(colors),
        "ink_count": int(np.count_nonzero(interior != BLACK)),
    }


def _panel_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_comps = components_mask(grid == GREY)
    boxes = []
    for coords in grey_comps:
        box = bbox_coords(coords)
        panel = crop_bbox(grid, box)
        try:
            _parse_panel(panel)
        except AssertionError:
            continue
        boxes.append(box)
    boxes.sort()
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique hollow panel whose red terminals are joined by one monochrome wire.

    Input structure:
    - A black background contains several grey hollow rectangular panels.
    - Inside each panel are exactly two red terminal pixels plus scattered colored wire fragments and dots.
    - Some panels contain a non-black chain between the two terminals, but that chain may mix colors.

    Rule:
    - Treat a valid wire as a 4-connected set of cells of one single non-red color, with the two red terminals allowed only at the two ends.
    - Test every panel independently for monochrome conduction between its two red terminals.
    - Return the unchanged tight crop of the unique panel that conducts.

    Key insight:
    - The decision is based on single-color connectivity inside each frame, not on generic non-black reachability through mixed-color bait chains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert get_bg(g) == BLACK

    winners = []
    for box in _panel_boxes(g):
        info = _parse_panel(crop_bbox(g, box))
        if info["mono_colors"]:
            winners.append(box)

    assert len(winners) == 1, "expected exactly one monochrome-conducting panel"
    return crop_bbox(g, winners[0])


def _edge_index(length: int, rng: random.Random) -> int:
    if length >= 5:
        return rng.randint(1, length - 2)
    return rng.randrange(length)


def _side_point(side: str, ih: int, iw: int, rng: random.Random) -> tuple[int, int]:
    if side == "left":
        return _edge_index(ih, rng), 0
    if side == "right":
        return _edge_index(ih, rng), iw - 1
    if side == "top":
        return 0, _edge_index(iw, rng)
    if side == "bottom":
        return ih - 1, _edge_index(iw, rng)
    raise ValueError(side)


def _choose_terminals(ih: int, iw: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    side_pairs = [
        ("left", "right"),
        ("top", "bottom"),
        ("left", "bottom"),
        ("top", "right"),
        ("left", "top"),
        ("right", "bottom"),
    ]
    for _ in range(100):
        side_a, side_b = rng.choice(side_pairs)
        a = _side_point(side_a, ih, iw, rng)
        b = _side_point(side_b, ih, iw, rng)
        if a == b:
            continue
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= max(4, (ih + iw) // 3):
            return a, b
    raise RuntimeError("failed to place terminals")


def _build_path(start: tuple[int, int], end: tuple[int, int], ih: int, iw: int, rng: random.Random) -> list[tuple[int, int]]:
    candidates = []
    candidates.append([start, (start[0], end[1]), end])
    candidates.append([start, (end[0], start[1]), end])

    if iw >= 3:
        midc = rng.randint(1, iw - 2)
        candidates.append([start, (start[0], midc), (end[0], midc), end])
    if ih >= 3:
        midr = rng.randint(1, ih - 2)
        candidates.append([start, (midr, start[1]), (midr, end[1]), end])
    if ih >= 4 and iw >= 4:
        midc1 = rng.randint(1, iw - 2)
        midr = rng.randint(1, ih - 2)
        candidates.append([start, (start[0], midc1), (midr, midc1), (midr, end[1]), end])
        midr1 = rng.randint(1, ih - 2)
        midc = rng.randint(1, iw - 2)
        candidates.append([start, (midr1, start[1]), (midr1, midc), (end[0], midc), end])

    rng.shuffle(candidates)
    for points in candidates:
        path = _path_from_points(points)
        if path is None:
            continue
        if len(path) >= 6:
            return path
    direct = _path_from_points([start, (start[0], end[1]), end])
    if direct is None or len(direct) < 4:
        raise RuntimeError("failed to build path")
    return direct


def _paint_cells(panel: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        panel[r + 1, c + 1] = color


def _reserve_mask(panel: np.ndarray) -> np.ndarray:
    return panel != BLACK


def _add_isolated_blobs(panel: np.ndarray, rng: random.Random, min_blobs: int = 1, max_blobs: int = 4) -> None:
    ih, iw = panel.shape[0] - 2, panel.shape[1] - 2
    target = rng.randint(min_blobs, max_blobs)
    for _ in range(target):
        for _attempt in range(80):
            horizontal = rng.choice([False, True])
            length = rng.choice([1, 1, 2, 2, 3])
            if horizontal:
                r = rng.randrange(ih)
                c = rng.randrange(max(1, iw - length + 1))
                cells = [(r, c + k) for k in range(length)]
            else:
                r = rng.randrange(max(1, ih - length + 1))
                c = rng.randrange(iw)
                cells = [(r + k, c) for k in range(length)]
            if any(panel[r + 1, c + 1] != BLACK for r, c in cells):
                continue
            ok = True
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < ih and 0 <= nc < iw and (nr, nc) not in cells and panel[nr + 1, nc + 1] != BLACK:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            color = rng.choice(WIRE_COLORS)
            _paint_cells(panel, cells, color)
            break


def _make_panel(kind: str, rng: random.Random) -> np.ndarray | None:
    h = rng.randint(6, 10)
    w = rng.randint(6, 10)
    panel = np.full((h, w), BLACK, dtype=int)
    panel[[0, -1], :] = GREY
    panel[:, [0, -1]] = GREY

    ih, iw = h - 2, w - 2
    try:
        start, end = _choose_terminals(ih, iw, rng)
        path = _build_path(start, end, ih, iw, rng)
    except RuntimeError:
        return None

    panel[start[0] + 1, start[1] + 1] = RED
    panel[end[0] + 1, end[1] + 1] = RED
    wire_cells = path[1:-1]

    if kind == "winner":
        color = rng.choice(WIRE_COLORS)
        _paint_cells(panel, wire_cells, color)
        _add_isolated_blobs(panel, rng, 1, 4)
    elif kind == "mixed":
        chunk_count = 3 if len(wire_cells) >= 7 and rng.random() < 0.55 else 2
        cuts = sorted(rng.sample(range(1, len(wire_cells)), chunk_count - 1))
        colors = rng.sample(WIRE_COLORS, chunk_count)
        prev = 0
        for cut, color in zip(cuts + [len(wire_cells)], colors):
            _paint_cells(panel, wire_cells[prev:cut], color)
            prev = cut
        _add_isolated_blobs(panel, rng, 1, 4)
    elif kind == "broken":
        if len(wire_cells) < 4:
            return None
        left_len = rng.randint(1, max(1, len(wire_cells) // 3))
        right_len = rng.randint(1, max(1, len(wire_cells) // 3))
        if left_len + right_len >= len(wire_cells):
            right_len = max(1, len(wire_cells) - left_len - 1)
        left_cells = wire_cells[:left_len]
        right_cells = wire_cells[-right_len:]
        left_color = rng.choice(WIRE_COLORS)
        right_color = rng.choice(WIRE_COLORS)
        _paint_cells(panel, left_cells, left_color)
        _paint_cells(panel, right_cells, right_color)
        _add_isolated_blobs(panel, rng, 2, 5)
    else:
        raise ValueError(kind)

    if rng.random() < 0.5:
        panel = np.rot90(panel, rng.randrange(4)).copy()
    else:
        panel = np.fliplr(np.rot90(panel, rng.randrange(4))).copy()

    try:
        info = _parse_panel(panel)
    except AssertionError:
        return None

    if kind == "winner" and len(info["mono_colors"]) != 1:
        return None
    if kind == "mixed" and not info["mixed_chain"]:
        return None
    if kind == "broken" and (info["mixed_chain"] or info["mono_colors"]):
        return None

    return panel


def _pack_panels(panels: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    panels = panels[:]
    rng.shuffle(panels)
    gap = rng.randint(1, 3)
    pad = rng.randint(0, 2)
    limit = 30 - pad

    placements = []
    r = pad
    c = pad
    row_h = 0
    max_c = pad
    for panel in panels:
        ph, pw = panel.shape
        if c > pad and c + pw > limit:
            r += row_h + gap
            c = pad
            row_h = 0
        if c + pw > limit:
            return None
        if r + ph > limit:
            return None
        placements.append((panel, r, c))
        c += pw + gap
        row_h = max(row_h, ph)
        max_c = max(max_c, c - gap)

    h = r + row_h + pad
    w = max_c + pad
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for panel, pr, pc in placements:
        ph, pw = panel.shape
        grid[pr : pr + ph, pc : pc + pw] = panel
    return grid


def _semantic_ok(grid: np.ndarray, panel_count: int) -> bool:
    boxes = _panel_boxes(grid)
    if len(boxes) != panel_count:
        return False

    infos = [_parse_panel(crop_bbox(grid, box)) for box in boxes]
    winners = [i for i, info in enumerate(infos) if info["mono_colors"]]
    mixed = [i for i, info in enumerate(infos) if info["mixed_chain"]]
    if len(winners) != 1 or not mixed:
        return False

    winner = winners[0]
    winner_panel = infos[winner]["panel"]
    if any(i != winner and np.array_equal(info["panel"], winner_panel) for i, info in enumerate(infos)):
        return False

    areas = [info["area"] for info in infos]
    color_counts = [info["wire_color_count"] for info in infos]
    ink_counts = [info["ink_count"] for info in infos]
    if not any(i != winner and areas[i] >= areas[winner] for i in range(len(infos))):
        return False
    if not any(i != winner and color_counts[i] >= color_counts[winner] for i in range(len(infos))):
        return False
    if not any(i != winner and ink_counts[i] >= ink_counts[winner] for i in range(len(infos))):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return np.array_equal(out, winner_panel)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_count = rng.choice([3, 3, 4])
    kinds = ["winner", "mixed"]
    while len(kinds) < panel_count:
        kinds.append(rng.choice(["mixed", "broken"]))
    rng.shuffle(kinds)

    panels = []
    for kind in kinds:
        panel = None
        for _ in range(60):
            panel = _make_panel(kind, rng)
            if panel is not None:
                break
        if panel is None:
            return None
        panels.append(panel)

    grid = _pack_panels(panels, rng)
    if grid is None:
        return None
    if not _semantic_ok(grid, panel_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid monochrome-panel input")
