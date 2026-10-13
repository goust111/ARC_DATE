from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(arr != bg)
    assert pts.size, "expected at least one non-background pixel"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False
    mask = np.zeros((h, w), dtype=bool)
    pts = comp["coords"]
    mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _find_legend(grid: np.ndarray, bg: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int, list[int]]] = []

    for r in range(h):
        c = 0
        while c < w:
            if int(g[r, c]) == bg:
                c += 1
                continue
            c0 = c
            while c < w and int(g[r, c]) != bg:
                c += 1
            c1 = c - 1
            segment = [int(x) for x in g[r, c0 : c1 + 1]]
            if not 3 <= len(segment) <= 5:
                continue
            if len(set(segment)) != len(segment):
                continue
            if r > 0 and not np.all(g[r - 1, c0 : c1 + 1] == bg):
                continue
            if r + 1 < h and not np.all(g[r + 1, c0 : c1 + 1] == bg):
                continue
            corner_score = min(
                r + c0,
                r + (w - 1 - c1),
                (h - 1 - r) + c0,
                (h - 1 - r) + (w - 1 - c1),
            )
            candidates.append((corner_score, r, c0, c1, segment))

    assert candidates, "failed to find a horizontal color-ring legend"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][4]


def _find_nested_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = [comp for comp in _components_by_color(grid, bg) if _is_frame_component(comp)]
    assert 3 <= len(frames) <= 5, "expected 3 to 5 rectangular frame layers"
    frames.sort(key=lambda comp: ((comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1), comp["bbox"]))
    for inner, outer in zip(frames, frames[1:]):
        ir0, ic0, ir1, ic1 = inner["bbox"]
        or0, oc0, or1, oc1 = outer["bbox"]
        assert (or0, oc0, or1, oc1) == (ir0 - 2, ic0 - 2, ir1 + 2, ic1 + 2), "frame layers must be concentric with one-cell gaps"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Transform the inner pattern by comparing nested frame colors against a corner color ring."""
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend = _find_legend(g, bg)
    color_to_idx = {color: idx for idx, color in enumerate(legend)}

    frames = _find_nested_frames(g, bg)
    for frame in frames:
        assert int(frame["color"]) in color_to_idx, "every frame color must appear in the legend"

    r0, c0, r1, c1 = frames[0]["bbox"]
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    pattern = _crop_non_bg(inner, bg)

    out = pattern.copy()
    n = len(legend)
    for inner_frame, outer_frame in zip(frames, frames[1:]):
        inner_idx = color_to_idx[int(inner_frame["color"])]
        outer_idx = color_to_idx[int(outer_frame["color"])]
        if outer_idx == (inner_idx + 1) % n:
            out = np.rot90(out, -1)
        elif outer_idx == (inner_idx - 1) % n:
            out = np.fliplr(out)

    return _crop_non_bg(out, bg)


# --------------- generator ---------------

BG = 0
NON_BG_COLORS = tuple(range(1, 10))
FRAME_COUNTS = (3, 4, 5)
RING_LENGTHS = (4, 5)
BRANCHES = ("succ", "pred", "skip")
LEGEND_CORNERS = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_variants(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    sigs: set[tuple[tuple[int, int], bytes]] = set()
    base = _crop_mask(mask)
    for k in range(4):
        rotated = np.rot90(base, k)
        for flipped in (rotated, np.fliplr(rotated)):
            cropped = _crop_mask(flipped)
            sigs.add((tuple(map(int, cropped.shape)), cropped.astype(np.uint8).tobytes()))
    return sigs


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        add_frontier(*cell)
    return _crop_mask(mask)


def _generate_pattern_mask() -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = random.randint(4, min(h * w - 2, 8))
        mask = _random_connected_mask(h, w, area)
        cropped = _crop_mask(mask)
        if int(cropped.sum()) < 4:
            continue
        if len(_mask_variants(cropped)) != 8:
            continue
        return cropped
    return None


def _balanced_branches(count: int) -> list[str]:
    order = list(BRANCHES)
    random.shuffle(order)
    branches = [order[i % len(order)] for i in range(count)]
    random.shuffle(branches)
    return branches


def _relation_candidates(prev_idx: int, branch: str, ring_len: int, used: set[int]) -> list[int]:
    if branch == "succ":
        choices = [(prev_idx + 1) % ring_len]
    elif branch == "pred":
        choices = [(prev_idx - 1) % ring_len]
    else:
        adjacent = {prev_idx, (prev_idx + 1) % ring_len, (prev_idx - 1) % ring_len}
        choices = [idx for idx in range(ring_len) if idx not in adjacent]
    choices = [idx for idx in choices if idx not in used]
    random.shuffle(choices)
    return choices


def _sample_ring_path(layer_count: int, ring_len: int) -> tuple[list[int], list[str]] | None:
    for _ in range(120):
        branches = _balanced_branches(layer_count - 1)

        def backtrack(path: list[int]) -> list[int] | None:
            if len(path) == layer_count:
                return path
            branch = branches[len(path) - 1]
            for nxt in _relation_candidates(path[-1], branch, ring_len, set(path)):
                res = backtrack(path + [nxt])
                if res is not None:
                    return res
            return None

        starts = list(range(ring_len))
        random.shuffle(starts)
        for start in starts:
            path = backtrack([start])
            if path is not None:
                return path, branches
    return None


def _render_pattern(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _apply_branches(pattern: np.ndarray, branches: list[str]) -> np.ndarray:
    out = pattern.copy()
    for branch in branches:
        if branch == "succ":
            out = np.rot90(out, -1)
        elif branch == "pred":
            out = np.fliplr(out)
    return _crop_non_bg(out, BG)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _place_legend(grid: np.ndarray, colors: list[int], corner: str) -> None:
    h, w = grid.shape
    n = len(colors)
    if corner == "tl":
        r, c0 = 0, 0
    elif corner == "tr":
        r, c0 = 0, w - n
    elif corner == "bl":
        r, c0 = h - 1, 0
    else:
        r, c0 = h - 1, w - n
    grid[r, c0 : c0 + n] = np.asarray(colors, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.choice(FRAME_COUNTS)
    ring_len = random.choice([n for n in RING_LENGTHS if n >= layer_count])

    path_info = _sample_ring_path(layer_count, ring_len)
    if path_info is None:
        return None
    frame_positions, branches = path_info

    palette = random.sample(NON_BG_COLORS, ring_len + 1)
    ring_colors = palette[:ring_len]
    pattern_color = palette[-1]
    frame_colors = [ring_colors[idx] for idx in frame_positions]

    pattern_mask = _generate_pattern_mask()
    if pattern_mask is None:
        return None
    pattern = _render_pattern(pattern_mask, pattern_color, BG)
    expected = _apply_branches(pattern, branches)

    ph, pw = pattern.shape
    inner_h = ph + random.randint(2, 5)
    inner_w = pw + random.randint(2, 5)
    outer_h = inner_h + 2 + 4 * (layer_count - 1)
    outer_w = inner_w + 2 + 4 * (layer_count - 1)

    total_h = outer_h + random.randint(8, 14)
    total_w = outer_w + random.randint(8, 14)
    corner = random.choice(LEGEND_CORNERS)

    outer_r0 = (total_h - outer_h) // 2 + random.randint(-1, 1)
    outer_c0 = (total_w - outer_w) // 2 + random.randint(-1, 1)
    outer_r0 = min(max(outer_r0, 2), total_h - outer_h - 2)
    outer_c0 = min(max(outer_c0, 2), total_w - outer_w - 2)

    grid = np.full((total_h, total_w), BG, dtype=int)
    _place_legend(grid, ring_colors, corner)

    inner_r0 = outer_r0 + 2 * (layer_count - 1)
    inner_c0 = outer_c0 + 2 * (layer_count - 1)
    for depth, color in enumerate(frame_colors):
        inset = 2 * (layer_count - 1 - depth)
        bbox = (
            outer_r0 + inset,
            outer_c0 + inset,
            outer_r0 + outer_h - 1 - inset,
            outer_c0 + outer_w - 1 - inset,
        )
        _draw_frame(grid, bbox, color)

    max_pr = inner_h - ph
    max_pc = inner_w - pw
    pr = random.randint(1, max(1, max_pr - 1)) if max_pr >= 2 else max_pr // 2
    pc = random.randint(1, max(1, max_pc - 1)) if max_pc >= 2 else max_pc // 2
    inner = grid[inner_r0 + 1 : inner_r0 + 1 + inner_h, inner_c0 + 1 : inner_c0 + 1 + inner_w]
    inner[pr : pr + ph, pc : pc + pw][pattern != BG] = pattern_color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
