import random
from collections import Counter, deque

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
LAYER_COUNT_CHOICES = (3, 4, 5)
GRID_SIZE_RANGE = (18, 30)
FILTER_LENGTH_CHOICES = (2, 3)
KEY_COUNT_CHOICES = (1, 2, 3)
CORNERS = ("tl", "tr", "bl", "br")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox_from_coords(coords)})
    return comps


def _frame_bbox_if_any(comp: dict) -> tuple[int, int, int, int] | None:
    coords = set(comp["coords"])
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return comp["bbox"] if coords == expected else None


def _shrink_bbox(bbox: tuple[int, int, int, int], d: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + d, c0 + d, r1 - d, c1 - d


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _find_frame_chain(grid: np.ndarray, bg: int) -> list[dict]:
    frame_candidates: list[dict] = []
    for comp in _components_by_color(grid, bg):
        bbox = _frame_bbox_if_any(comp)
        if bbox is not None:
            frame_candidates.append({"color": int(comp["color"]), "bbox": bbox})
    assert frame_candidates, "expected at least one frame"

    by_bbox = {frame["bbox"]: frame for frame in frame_candidates}
    chains: list[list[dict]] = []
    for frame in frame_candidates:
        bbox = frame["bbox"]
        prev = _shrink_bbox(bbox, -3)
        if prev in by_bbox:
            continue
        chain = [frame]
        cur = bbox
        while True:
            nxt = _shrink_bbox(cur, 3)
            if nxt not in by_bbox:
                break
            chain.append(by_bbox[nxt])
            cur = nxt
        if len(chain) >= 3:
            chains.append(chain)
    assert chains, "expected a nested frame chain"
    chains.sort(key=lambda chain: (len(chain), (chain[0]["bbox"][2] - chain[0]["bbox"][0]) * (chain[0]["bbox"][3] - chain[0]["bbox"][1])), reverse=True)
    best = chains[0]
    assert len(best) in {3, 4, 5}
    return best


def _find_filter(grid: np.ndarray, bg: int, frame_mask: np.ndarray, outer_bbox: tuple[int, int, int, int]) -> tuple[set[int], np.ndarray]:
    h, w = grid.shape
    r0, c0, r1, c1 = outer_bbox
    candidates: list[tuple[int, int, int, list[int]]] = []
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) == bg or frame_mask[r, c]:
                c += 1
                continue
            start = c
            colors: list[int] = []
            while c < w and int(grid[r, c]) != bg and not frame_mask[r, c]:
                colors.append(int(grid[r, c]))
                c += 1
            end = c - 1
            length = end - start + 1
            if length not in {2, 3} or len(set(colors)) != length:
                continue
            if start > 0 and int(grid[r, start - 1]) != bg:
                continue
            if end + 1 < w and int(grid[r, end + 1]) != bg:
                continue
            ok = True
            for cc in range(start, end + 1):
                if (r > 0 and int(grid[r - 1, cc]) != bg) or (r + 1 < h and int(grid[r + 1, cc]) != bg):
                    ok = False
                    break
            if not ok:
                continue
            in_corner = (
                (r < r0 and end < c0)
                or (r < r0 and start > c1)
                or (r > r1 and end < c0)
                or (r > r1 and start > c1)
            )
            if not in_corner:
                continue
            corner_dist = min(r + start, r + (w - 1 - end), (h - 1 - r) + start, (h - 1 - r) + (w - 1 - end))
            candidates.append((corner_dist, r, start, colors))
    assert candidates, "expected a corner filter"
    candidates.sort(key=lambda item: item[0])
    _, r, start, colors = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    for offset in range(len(colors)):
        mask[r, start + offset] = True
    return set(colors), mask


def _keys_for_layer(key_mask: np.ndarray, bbox_outer: tuple[int, int, int, int], bbox_inner: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox_outer
    ir0, ic0, ir1, ic1 = bbox_inner
    mask = np.zeros_like(key_mask, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    return key_mask & mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames = _find_frame_chain(g, bg)
    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_mask[r0, c0 : c1 + 1] = True
        frame_mask[r1, c0 : c1 + 1] = True
        frame_mask[r0 : r1 + 1, c0] = True
        frame_mask[r0 : r1 + 1, c1] = True

    retain_colors, filter_mask = _find_filter(g, bg, frame_mask, frames[0]["bbox"])
    key_mask = (g != bg) & (~frame_mask) & (~filter_mask)

    out = np.full_like(g, bg)
    _paint_frame(out, frames[0]["bbox"], frames[0]["color"])
    for outer, inner in zip(frames, frames[1:]):
        annulus_keys = _keys_for_layer(key_mask, outer["bbox"], inner["bbox"])
        key_colors = {int(v) for v in g[annulus_keys]}
        if not key_colors or key_colors.issubset(retain_colors):
            _paint_frame(out, inner["bbox"], inner["color"])
    return out


def _filter_anchor(corner: str, h: int, w: int, length: int) -> tuple[int, int]:
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, w - length - 1
    if corner == "bl":
        return h - 2, 1
    return h - 2, w - length - 1


def _annulus_positions(bbox_outer: tuple[int, int, int, int], bbox_inner: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox_outer
    ir0, ic0, ir1, ic1 = bbox_inner
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                continue
            cells.append((r, c))
    return cells


def _pick_non_adjacent_positions(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    order = cells[:]
    random.shuffle(order)
    chosen: list[tuple[int, int]] = []
    for cell in order:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _compute_expected_keep(frame_colors: list[int], retain: set[int], branch_specs: list[tuple[str, list[int]]]) -> list[bool]:
    keep = [True]
    for kind, colors in branch_specs:
        if kind == "none":
            keep.append(True)
        else:
            keep.append(set(colors).issubset(retain))
    assert len(keep) == len(frame_colors)
    return keep


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    bg = random.randint(0, 9)

    layer_count = random.choice(LAYER_COUNT_CHOICES)
    inner_min = 3
    outer_h_min = inner_min + 6 * (layer_count - 1)
    outer_w_min = inner_min + 6 * (layer_count - 1)
    if h < outer_h_min + 4 or w < outer_w_min + 6:
        return None

    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    frame_colors = palette[:layer_count]

    max_outer_h = h - 4
    max_outer_w = w - 4
    outer_h = random.randint(outer_h_min, max_outer_h)
    outer_w = random.randint(outer_w_min, max_outer_w)
    max_r0 = h - outer_h - 2
    max_c0 = w - outer_w - 2
    if max_r0 < 2 or max_c0 < 2:
        return None
    r0 = random.randint(2, max_r0)
    c0 = random.randint(2, max_c0)

    bboxes = []
    for i in range(layer_count):
        bboxes.append((r0 + 3 * i, c0 + 3 * i, r0 + outer_h - 1 - 3 * i, c0 + outer_w - 1 - 3 * i))

    retain_len_options = [k for k in FILTER_LENGTH_CHOICES if k < layer_count]
    if not retain_len_options:
        return None
    retain_len = random.choice(retain_len_options)
    retain_colors = set(random.sample(frame_colors, retain_len))

    branch_specs: list[tuple[str, list[int]]] = []
    placed_keys: list[tuple[int, int, int]] = []
    for i in range(1, layer_count):
        admissible = [frame_colors[j] for j in range(layer_count) if j not in {i - 1, i}]
        keep_choices = [c for c in admissible if c in retain_colors]
        remove_choices = [c for c in admissible if c not in retain_colors]
        options = ["none"]
        if keep_choices:
            options.append("keep")
        if remove_choices:
            options.append("remove")
        branch = random.choice(options)
        annulus = _annulus_positions(bboxes[i - 1], bboxes[i])
        if branch == "none":
            branch_specs.append(("none", []))
            continue
        count = random.choice(KEY_COUNT_CHOICES)
        cells = _pick_non_adjacent_positions(annulus, count)
        if cells is None:
            return None
        colors: list[int] = []
        if branch == "keep":
            colors = [random.choice(keep_choices) for _ in range(count)]
        else:
            colors.append(random.choice(remove_choices))
            pool = admissible[:]
            for _ in range(count - 1):
                colors.append(random.choice(pool))
            random.shuffle(colors)
        for (rr, cc), color in zip(cells, colors):
            placed_keys.append((rr, cc, color))
        branch_specs.append((branch, colors))

    grid = np.full((h, w), bg, dtype=int)
    for bbox, color in zip(bboxes, frame_colors):
        _paint_frame(grid, bbox, color)
    for rr, cc, color in placed_keys:
        grid[rr, cc] = color

    corner = random.choice(CORNERS)
    filter_start = _filter_anchor(corner, h, w, retain_len)
    fr, fc = filter_start
    if corner in {"tl", "tr"} and fr >= bboxes[0][0]:
        return None
    if corner in {"bl", "br"} and fr <= bboxes[0][2]:
        return None
    if corner in {"tl", "bl"} and fc + retain_len - 1 >= bboxes[0][1]:
        return None
    if corner in {"tr", "br"} and fc <= bboxes[0][3]:
        return None
    for offset, color in enumerate(random.sample(list(retain_colors), retain_len)):
        grid[fr, fc + offset] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected_keep = _compute_expected_keep(frame_colors, retain_colors, branch_specs)
    expected = np.full_like(grid, bg)
    for bbox, color, keep in zip(bboxes, frame_colors, expected_keep):
        if keep:
            _paint_frame(expected, bbox, color)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
