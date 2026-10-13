from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


TYPES = ("H", "V", "R")
ICON_MASKS = {
    "H": np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    "V": np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "R": np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _component_from(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w and bool(mask[sr, sc])):
        return []
    seen = {start}
    q = deque([start])
    pts: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        pts.append((int(r), int(c)))
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return pts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray | None:
    pts = np.argwhere(arr != bg)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _has_h_sym(arr: np.ndarray) -> bool:
    return bool(np.array_equal(arr, np.fliplr(arr)))


def _has_v_sym(arr: np.ndarray) -> bool:
    return bool(np.array_equal(arr, np.flipud(arr)))


def _has_r_sym(arr: np.ndarray) -> bool:
    return bool(arr.shape[0] == arr.shape[1] and np.array_equal(arr, np.rot90(arr)))


def _symmetry_set(arr: np.ndarray) -> set[str]:
    out: set[str] = set()
    if _has_h_sym(arr):
        out.add("H")
    if _has_v_sym(arr):
        out.add("V")
    if _has_r_sym(arr):
        out.add("R")
    return out


def _classify_icon(icon_mask: np.ndarray) -> str:
    for typ, templ in ICON_MASKS.items():
        if np.array_equal(icon_mask, templ):
            return typ
    raise ValueError("unknown legend icon")


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[str, int], list[str], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    legend_pts: list[tuple[int, int]] | None = None
    for corner in corners:
        if int(g[corner]) == bg:
            continue
        pts = _component_from(g != bg, corner)
        if not pts:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if len(pts) == hh * ww and (hh, ww) in {(9, 5), (3, 15)}:
            legend_pts = pts
            break
    if legend_pts is None:
        raise ValueError("legend not found")

    r0, c0, r1, c1 = _bbox(legend_pts)
    panel = g[r0 : r1 + 1, c0 : c1 + 1]
    panel_bg = _most_common_color(panel)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    items: list[np.ndarray] = []
    if panel.shape == (9, 5):
        items = [panel[i * 3 : (i + 1) * 3, :] for i in range(3)]
    elif panel.shape == (3, 15):
        items = [panel[:, i * 5 : (i + 1) * 5] for i in range(3)]
    else:
        raise ValueError("unexpected legend shape")

    mapping: dict[str, int] = {}
    order: list[str] = []
    for item in items:
        icon_mask = item[:, :3] != panel_bg
        typ = _classify_icon(icon_mask)
        label_col = _most_common_color(item[:, 4])
        mapping[typ] = int(label_col)
        order.append(typ)
    return mapping, order, legend_mask


def _is_frame_component(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 6 or ww < 6:
        return False
    mask = np.zeros((hh, ww), dtype=bool)
    for r, c in comp:
        mask[r - r0, c - c0] = True
    want = np.zeros((hh, ww), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return bool(np.array_equal(mask, want))


def _find_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    frames: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        mask = (g == color) & (~legend_mask)
        for comp in _components(mask):
            if _is_frame_component(comp):
                frames.append(_bbox(comp))
    frames = sorted(set(frames))
    if not frames:
        raise ValueError("no frames found")
    return frames


def _seed_pattern(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    inner = np.array(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    crop = _crop_non_bg(inner, bg)
    if crop is None:
        raise ValueError("frame has no seed")
    return crop


def _chosen_type(symmetries: set[str], order: list[str]) -> str:
    for typ in order:
        if typ in symmetries:
            return typ
    raise ValueError("seed lacks supported symmetry")


def _frame_distance(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return int(max(dr, dc))


def _blocked_mask(shape: tuple[int, int], legend_mask: np.ndarray, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    blocked = np.array(legend_mask, copy=True)
    for r0, c0, r1, c1 in frames:
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    return blocked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame's exterior territory with the color mapped from its seed pattern's symmetry type.

    Input structure:
    - bg = most common color. The grid contains several same-colored hollow rectangular frames, each enclosing a small non-background seed pattern.
    - A legend panel sits flush in one corner of the grid. It contains three ordered mini-icons for horizontal mirror symmetry, vertical mirror symmetry, and 90-degree rotational symmetry, each paired with a fill color.
    - The area outside every frame is blank background; pixels inside frames but not part of the seed remain background.

    Rule:
    - Read the ordered legend to map each symmetry type to its output color, and use the legend order itself as the tie-break priority when a seed satisfies multiple symmetry types.
    - For every frame, crop the enclosed seed pattern, test whether it is left-right symmetric, top-bottom symmetric, and/or 90-degree rotation invariant, then choose the earliest matching legend type.
    - Recolor every background pixel outside the frames and outside the legend with the chosen color of the nearest frame, breaking equal distances by the frame whose top-left corner is higher, then lefter.

    Key insight:
    - The task combines two layers: infer each seed's geometric symmetry class, then use the corner legend to convert that class into a territory fill color for the frame's exterior Voronoi-like region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping, order, legend_mask = _parse_legend(g, bg)
    frames = _find_frames(g, bg, legend_mask)

    frame_fill: dict[tuple[int, int, int, int], int] = {}
    for bbox in frames:
        seed = _seed_pattern(g, bbox, bg)
        typ = _chosen_type(_symmetry_set(seed), order)
        frame_fill[bbox] = int(mapping[typ])

    blocked = _blocked_mask(g.shape, legend_mask, frames)
    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) != bg or bool(blocked[r, c]):
                continue
            best_bbox = min(frames, key=lambda box: (_frame_distance(r, c, box), box[0], box[1]))
            out[r, c] = frame_fill[best_bbox]
    return out


def _orbit_h(h: int, w: int, r: int, c: int) -> set[tuple[int, int]]:
    return {(r, c), (r, w - 1 - c)}


def _orbit_v(h: int, w: int, r: int, c: int) -> set[tuple[int, int]]:
    return {(r, c), (h - 1 - r, c)}


def _orbit_hv(h: int, w: int, r: int, c: int) -> set[tuple[int, int]]:
    return {
        (r, c),
        (r, w - 1 - c),
        (h - 1 - r, c),
        (h - 1 - r, w - 1 - c),
    }


def _orbit_r(n: int, r: int, c: int) -> set[tuple[int, int]]:
    return {
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
    }


def _orbit_hvr(n: int, r: int, c: int) -> set[tuple[int, int]]:
    return {
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (c, r),
        (n - 1 - c, n - 1 - r),
    }


def _random_orbit_color(rng: random.Random, colors: list[int]) -> int:
    if rng.random() < 0.35:
        return 0
    return int(rng.choice(colors))


def _pattern_from_orbits(case: str, rng: random.Random, colors: list[int]) -> np.ndarray:
    if case in {"R", "HVR"}:
        h = w = rng.choice([3, 4])
    elif case == "HV":
        options = [(hh, ww) for hh in range(2, 5) for ww in range(2, 5) if hh != ww]
        h, w = rng.choice(options)
    else:
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)

    arr = np.full((h, w), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            if int(arr[r, c]) != -1:
                continue
            if case == "H":
                orbit = _orbit_h(h, w, r, c)
            elif case == "V":
                orbit = _orbit_v(h, w, r, c)
            elif case == "HV":
                orbit = _orbit_hv(h, w, r, c)
            elif case == "R":
                orbit = _orbit_r(h, r, c)
            else:
                orbit = _orbit_hvr(h, r, c)
            color = _random_orbit_color(rng, colors)
            for rr, cc in orbit:
                arr[rr, cc] = color
    return arr


def _generate_pattern(case: str, rng: random.Random, colors: list[int]) -> np.ndarray:
    targets = {
        "H": {"H"},
        "V": {"V"},
        "R": {"R"},
        "HV": {"H", "V"},
        "HVR": {"H", "V", "R"},
    }
    for _ in range(500):
        raw = _pattern_from_orbits(case, rng, colors)
        crop = _crop_non_bg(raw, 0)
        if crop is None:
            continue
        if crop.shape[0] < 2 or crop.shape[1] < 2 or crop.shape[0] > 4 or crop.shape[1] > 4:
            continue
        if int(np.count_nonzero(crop)) < 2:
            continue
        if _symmetry_set(crop) == targets[case]:
            return crop
    raise RuntimeError(f"failed to build {case} seed")


def _legend_item(panel_color: int, icon_color: int, fill_color: int, typ: str) -> np.ndarray:
    item = np.full((3, 5), int(panel_color), dtype=int)
    item[:, 4] = int(fill_color)
    item[:, :3][ICON_MASKS[typ]] = int(icon_color)
    return item


def _legend_panel(order: list[str], mapping: dict[str, int], orientation: str, panel_color: int, icon_color: int) -> np.ndarray:
    if orientation == "vertical":
        panel = np.full((9, 5), int(panel_color), dtype=int)
        for i, typ in enumerate(order):
            panel[i * 3 : (i + 1) * 3, :] = _legend_item(panel_color, icon_color, mapping[typ], typ)
        return panel
    panel = np.full((3, 15), int(panel_color), dtype=int)
    for i, typ in enumerate(order):
        panel[:, i * 5 : (i + 1) * 5] = _legend_item(panel_color, icon_color, mapping[typ], typ)
    return panel


def _paste(arr: np.ndarray, block: np.ndarray, top: int, left: int) -> None:
    h, w = block.shape
    arr[top : top + h, left : left + w] = block


def _place_legend(arr: np.ndarray, panel: np.ndarray, corner: str) -> np.ndarray:
    h, w = arr.shape
    ph, pw = panel.shape
    mask = np.zeros_like(arr, dtype=bool)
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, w - pw
    elif corner == "BL":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw
    _paste(arr, panel, r0, c0)
    mask[r0 : r0 + ph, c0 : c0 + pw] = True
    return mask


def _frame_cases(rng: random.Random) -> list[str]:
    count = rng.choices([2, 3, 4], weights=[1, 2, 3])[0]
    if count == 2:
        cases = rng.sample(list(TYPES), 2)
    elif count == 3:
        cases = list(TYPES)
        rng.shuffle(cases)
    else:
        extra = rng.choices(["HV", "HVR", "H", "V", "R"], weights=[2, 2, 1, 1, 1])[0]
        cases = ["H", "V", "R", extra]
        rng.shuffle(cases)
    return cases


def _reserve_free(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(reserved.shape[0], r0 + h + gap)
    cc1 = min(reserved.shape[1], c0 + w + gap)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _sample_positions(
    shape: tuple[int, int],
    legend_mask: np.ndarray,
    sizes: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    reserved = np.array(legend_mask, copy=True)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: list[tuple[int, int] | None] = [None] * len(sizes)
    for idx in order:
        fh, fw = sizes[idx]
        candidates = []
        for r0 in range(1, h - fh):
            for c0 in range(1, w - fw):
                if _reserve_free(reserved, r0, c0, fh, fw, gap=2):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placed[idx] = (int(r0), int(c0))
        reserved[r0 : r0 + fh, c0 : c0 + fw] = True
    return [p for p in placed if p is not None]


def _draw_frame_with_seed(
    grid: np.ndarray,
    top: int,
    left: int,
    frame_color: int,
    seed: np.ndarray,
    pads: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    top_pad, bottom_pad, left_pad, right_pad = pads
    sh, sw = seed.shape
    h = sh + top_pad + bottom_pad + 2
    w = sw + left_pad + right_pad + 2
    r1 = top + h - 1
    c1 = left + w - 1
    grid[top : r1 + 1, left] = int(frame_color)
    grid[top : r1 + 1, c1] = int(frame_color)
    grid[top, left : c1 + 1] = int(frame_color)
    grid[r1, left : c1 + 1] = int(frame_color)
    sr = top + 1 + top_pad
    sc = left + 1 + left_pad
    for r in range(sh):
        for c in range(sw):
            if int(seed[r, c]) != 0:
                grid[sr + r, sc + c] = int(seed[r, c])
    return (top, left, r1, c1)


def _build_sample(rng: random.Random) -> np.ndarray | None:
    bg = 0
    role_colors = rng.sample(list(range(1, 10)), 9)
    frame_color, panel_color, icon_color, fill_h, fill_v, fill_r, seed_a, seed_b, seed_c = role_colors
    seed_colors = [seed_a, seed_b, seed_c]
    type_to_fill = {"H": fill_h, "V": fill_v, "R": fill_r}

    orientation = rng.choice(["vertical", "horizontal"])
    corner = rng.choice(["TL", "TR", "BL", "BR"])
    order = rng.sample(list(TYPES), 3)
    panel = _legend_panel(order, type_to_fill, orientation, panel_color, icon_color)

    grid_h = rng.randint(max(18, panel.shape[0] + 6), 30)
    grid_w = rng.randint(max(18, panel.shape[1] + 6), 30)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    legend_mask = _place_legend(grid, panel, corner)

    cases = _frame_cases(rng)
    seeds = [_generate_pattern(case, rng, seed_colors) for case in cases]
    pads = [(rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 2)) for _ in seeds]
    sizes = [
        (seed.shape[0] + pad[0] + pad[1] + 2, seed.shape[1] + pad[2] + pad[3] + 2)
        for seed, pad in zip(seeds, pads, strict=True)
    ]
    positions = _sample_positions(grid.shape, legend_mask, sizes, rng)
    if positions is None:
        return None

    frame_bboxes = []
    for (top, left), seed, pad in zip(positions, seeds, pads, strict=True):
        frame_bboxes.append(_draw_frame_with_seed(grid, top, left, frame_color, seed, pad))

    try:
        parsed_mapping, parsed_order, parsed_legend = _parse_legend(grid, bg)
        if parsed_mapping != type_to_fill or parsed_order != order or not np.array_equal(parsed_legend, legend_mask):
            return None
        parsed_frames = _find_frames(grid, bg, legend_mask)
        if sorted(parsed_frames) != sorted(frame_bboxes):
            return None
        blocked = _blocked_mask(grid.shape, legend_mask, parsed_frames)
        out = main(grid)
        if np.array_equal(out, grid):
            return None
        if not np.array_equal(out[blocked], grid[blocked]):
            return None
        if bool(np.any((out == bg) & (~blocked))):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _build_sample(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
