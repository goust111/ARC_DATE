from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _legend_candidate(
    grid: np.ndarray, bg: int, row: int, start_col: int, step: int
) -> dict | None:
    h, w = grid.shape
    cols: list[int] = []
    c = int(start_col)
    while 0 <= c < w and int(grid[row, c]) != bg:
        cols.append(c)
        c += int(step)
    if len(cols) < 2:
        return None
    if h > 1:
        inward_row = 1 if row == 0 else h - 2
        if any(int(grid[inward_row, cc]) != bg for cc in cols):
            return None
    positions = [(int(row), int(cc)) for cc in cols]
    colors = [int(grid[r, c0]) for r, c0 in positions]
    if step < 0:
        positions.reverse()
        colors.reverse()
    return {"positions": positions, "colors": colors}


def _find_legend_strip(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    for row, start_col, step in [
        (0, 0, 1),
        (0, w - 1, -1),
        (h - 1, 0, 1),
        (h - 1, w - 1, -1),
    ]:
        candidate = _legend_candidate(grid, bg, row, start_col, step)
        if candidate is not None:
            candidates.append(candidate)
    assert candidates, "missing corner legend strip"
    candidates.sort(key=lambda item: len(item["positions"]), reverse=True)
    best = candidates[0]
    return list(best["colors"]), set(best["positions"])


def _most_common_excluding(region: np.ndarray, excluded: set[int]) -> int:
    counts = Counter(int(v) for v in region.flat if int(v) not in excluded)
    assert counts
    return int(counts.most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay several marked sprite windows into the unique marked empty window using the legend strip as the layer order.

    Input structure:
    - The most common color is the outer background, and one corner contains a 1-cell-tall horizontal legend strip.
    - Several separated solid rectangular windows appear on the background; exactly one window contains only the shared marker pixel, while every other window contains the same marker color plus a sprite.
    - Each sprite window is filled by a window color, and its dominant sprite color appears somewhere on the legend strip.

    Rule:
    - Identify the unique marker color shared by all windows and the unique target window that has no sprite pixels beyond its fill and marker.
    - For every sprite window, translate each non-fill, non-marker pixel so the source marker lands on the target marker, clipping any translated pixels that fall outside the target window.
    - Start from a blank target window filled with the target fill color, and apply the sprite windows in the left-to-right order of their legend colors; later legend entries overwrite earlier ones on overlaps.

    Key insight:
    - Marker alignment determines where every sprite lands, the empty marked window determines the output canvas, and only the legend strip determines which overlapping sprite stays visible on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend_colors, legend_positions = _find_legend_strip(g, bg)
    legend_index = {int(color): idx for idx, color in enumerate(legend_colors)}

    mask = g != bg
    for r, c in legend_positions:
        mask[r, c] = False

    windows = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if coords.shape[0] != h * w or h < 3 or w < 3:
            continue
        windows.append(
            {
                "bbox": (r0, c0, r1, c1),
                "region": g[r0 : r1 + 1, c0 : c1 + 1],
            }
        )
    assert len(windows) >= 3

    regions = [item["region"] for item in windows]
    all_colors = sorted({int(v) for region in regions for v in region.flat})
    marker_candidates = []
    for color in all_colors:
        per_region = [int(np.count_nonzero(region == color)) for region in regions]
        if sum(per_region) == len(regions) and all(count == 1 for count in per_region):
            marker_candidates.append(int(color))
    if len(marker_candidates) != 1:
        non_legend = [color for color in marker_candidates if color not in legend_index]
        assert len(non_legend) == 1
        marker = int(non_legend[0])
    else:
        marker = int(marker_candidates[0])

    parsed = []
    for item in windows:
        region = item["region"]
        marker_coords = np.argwhere(region == marker)
        assert marker_coords.shape == (1, 2)
        marker_rc = tuple(map(int, marker_coords[0]))
        fill = _most_common_excluding(region, {marker})
        sprite_mask = (region != fill) & (region != marker)
        parsed.append(
            {
                "region": region,
                "fill": int(fill),
                "marker": marker_rc,
                "sprite_mask": sprite_mask,
            }
        )

    targets = [item for item in parsed if not bool(item["sprite_mask"].any())]
    sources = [item for item in parsed if bool(item["sprite_mask"].any())]
    assert len(targets) == 1 and sources

    target = targets[0]
    dst_h, dst_w = target["region"].shape
    dst_marker = target["marker"]
    out = np.full((dst_h, dst_w), int(target["fill"]), dtype=int)

    ordered_sources = []
    for item in sources:
        sprite_values = [int(v) for v in item["region"][item["sprite_mask"]]]
        color_counts = Counter(sprite_values)
        best_count = max(color_counts.values())
        dominant = [color for color, count in color_counts.items() if count == best_count]
        dominant_legend = [color for color in dominant if color in legend_index]
        if len(dominant_legend) == 1:
            main_color = int(dominant_legend[0])
        else:
            assert len(dominant) == 1
            main_color = int(dominant[0])
        assert main_color in legend_index
        ordered_sources.append((legend_index[main_color], item))

    ordered_sources.sort(key=lambda pair: pair[0])
    for _order, item in ordered_sources:
        sr, sc = item["marker"]
        dr = int(dst_marker[0] - sr)
        dc = int(dst_marker[1] - sc)
        region = item["region"]
        for r, c in np.argwhere(item["sprite_mask"]):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < dst_h and 0 <= cc < dst_w:
                out[rr, cc] = int(region[r, c])

    return out


MAX_ATTEMPTS = 300


def _sample_target_marker(h: int, w: int) -> tuple[int, int]:
    rows = list(range(1, h - 1)) or list(range(h))
    cols = list(range(1, w - 1)) or list(range(w))
    return int(random.choice(rows)), int(random.choice(cols))


def _sample_source_marker(h: int, w: int, edge_bias: bool) -> tuple[int, int]:
    if edge_bias and random.random() < 0.8:
        side = random.choice(["top", "bottom", "left", "right", "corner"])
        if side == "top":
            return 0, int(random.randrange(w))
        if side == "bottom":
            return h - 1, int(random.randrange(w))
        if side == "left":
            return int(random.randrange(h)), 0
        if side == "right":
            return int(random.randrange(h)), w - 1
        return int(random.choice([0, h - 1])), int(random.choice([0, w - 1]))
    return int(random.randrange(h)), int(random.randrange(w))


def _shortest_path(
    h: int, w: int, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    q = deque([start])
    parents = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in blocked or nxt in parents:
                continue
            parents[nxt] = (r, c)
            if nxt == goal:
                path = [goal]
                cur = goal
                while parents[cur] is not None:
                    cur = parents[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(nxt)
    return None


def _grow_blob(
    h: int,
    w: int,
    count: int,
    marker: tuple[int, int],
    required: set[tuple[int, int]],
    crop_cells: set[tuple[int, int]],
    want_crop: bool,
) -> set[tuple[int, int]] | None:
    blocked = {marker}
    if required:
        cells = set(required)
    else:
        pool = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
        cells = {random.choice(pool)}

    if any(cell in blocked for cell in cells):
        return None

    if want_crop and crop_cells:
        crop_goal = random.choice(list(crop_cells))
        anchor = random.choice(list(cells))
        path = _shortest_path(h, w, anchor, crop_goal, blocked)
        if path is None:
            return None
        cells.update(path)

    count = max(int(count), len(cells))
    while len(cells) < count:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                nxt = (int(nr), int(nc))
                if 0 <= nr < h and 0 <= nc < w and nxt not in blocked and nxt not in cells:
                    frontier.append(nxt)
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return cells


def _sample_shared_cells(dst_marker: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    dr0, dc0 = dst_marker
    neighbors = [
        (dr0 + dr, dc0 + dc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= dr0 + dr < h and 0 <= dc0 + dc < w
    ]
    if not neighbors:
        fallback = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) != dst_marker
        ]
        return {random.choice(fallback)}
    anchor = random.choice(neighbors)
    shared = {anchor}
    if random.random() < 0.5:
        second = [
            (anchor[0] + dr, anchor[1] + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= anchor[0] + dr < h
            and 0 <= anchor[1] + dc < w
            and (anchor[0] + dr, anchor[1] + dc) != dst_marker
        ]
        if second:
            shared.add(random.choice(second))
    return shared


def _place_rectangles(
    canvas_shape: tuple[int, int],
    sizes: list[tuple[int, int]],
    reserved: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    occ = reserved.copy()
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    for idx in order:
        rh, rw = sizes[idx]
        candidates = []
        for r0 in range(1, h - rh):
            for c0 in range(1, w - rw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + rh + 1)
                cc1 = min(w, c0 + rw + 1)
                if not bool(occ[rr0:rr1, cc0:cc1].any()):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (int(r0), int(c0))
        occ[r0 : r0 + rh, c0 : c0 + rw] = True
    return [pos for pos in positions if pos is not None]


def _legend_positions(shape: tuple[int, int], corner: str, n: int) -> list[tuple[int, int]]:
    h, w = shape
    if corner == "tl":
        return [(0, c) for c in range(n)]
    if corner == "tr":
        return [(0, w - n + c) for c in range(n)]
    if corner == "bl":
        return [(h - 1, c) for c in range(n)]
    return [(h - 1, w - n + c) for c in range(n)]


def _build_source_window(
    color: int,
    fill: int,
    marker_color: int,
    window_shape: tuple[int, int],
    src_marker: tuple[int, int],
    dst_marker: tuple[int, int],
    target_shape: tuple[int, int],
    shared_cells: set[tuple[int, int]],
    overlap_participant: bool,
    want_crop: bool,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]] | None:
    wh, ww = window_shape
    th, tw = target_shape
    sr, sc = src_marker
    required = set()
    if overlap_participant:
        for tr, tc in shared_cells:
            lr = int(tr - dst_marker[0] + sr)
            lc = int(tc - dst_marker[1] + sc)
            if not (0 <= lr < wh and 0 <= lc < ww) or (lr, lc) == src_marker:
                return None
            required.add((lr, lc))

    crop_cells = set()
    for r in range(wh):
        for c in range(ww):
            if (r, c) == src_marker:
                continue
            tr = int(r - sr + dst_marker[0])
            tc = int(c - sc + dst_marker[1])
            if not (0 <= tr < th and 0 <= tc < tw):
                crop_cells.add((r, c))

    cell_count = random.randint(5, 9)
    blob = _grow_blob(wh, ww, cell_count, src_marker, required, crop_cells, want_crop)
    if blob is None:
        return None

    translated = {(int(r - sr + dst_marker[0]), int(c - sc + dst_marker[1])) for r, c in blob}
    visible = {(r, c) for r, c in translated if 0 <= r < th and 0 <= c < tw}
    if len(visible) < 3:
        return None
    if want_crop and not (translated - visible):
        return None

    window = np.full((wh, ww), int(fill), dtype=int)
    window[sr, sc] = int(marker_color)
    for r, c in blob:
        window[r, c] = int(color)
    return window, translated, visible


def _compose_expected(
    target_shape: tuple[int, int],
    fill: int,
    legend_order: list[int],
    visible_by_color: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    out = np.full(target_shape, int(fill), dtype=int)
    for color in legend_order:
        for r, c in visible_by_color[int(color)]:
            out[r, c] = int(color)
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_sprites = random.randint(2, 4)
    palette = random.sample(list(range(10)), n_sprites + 3)
    bg = int(palette[0])
    fill = int(palette[1])
    marker_color = int(palette[2])
    sprite_colors = [int(color) for color in palette[3:]]
    legend_order = sprite_colors[:]
    random.shuffle(legend_order)

    target_h = random.randint(7, 11)
    target_w = random.randint(7, 11)
    dst_marker = _sample_target_marker(target_h, target_w)
    target_window = np.full((target_h, target_w), int(fill), dtype=int)
    target_window[dst_marker] = int(marker_color)

    shared_cells = _sample_shared_cells(dst_marker, (target_h, target_w))
    overlap_participants = set(random.sample(range(n_sprites), random.randint(2, min(3, n_sprites))))
    crop_indices = set()
    if random.random() < 0.65:
        crop_indices.add(random.randrange(n_sprites))
        if n_sprites > 2 and random.random() < 0.25:
            crop_indices.add(random.randrange(n_sprites))

    source_windows = []
    translated_by_color: dict[int, set[tuple[int, int]]] = {}
    visible_by_color: dict[int, set[tuple[int, int]]] = {}
    for idx, color in enumerate(sprite_colors):
        built = None
        for _ in range(80):
            wh = random.randint(6, 11)
            ww = random.randint(6, 11)
            src_marker = _sample_source_marker(wh, ww, idx in crop_indices)
            built = _build_source_window(
                color=color,
                fill=fill,
                marker_color=marker_color,
                window_shape=(wh, ww),
                src_marker=src_marker,
                dst_marker=dst_marker,
                target_shape=(target_h, target_w),
                shared_cells=shared_cells,
                overlap_participant=idx in overlap_participants,
                want_crop=idx in crop_indices,
            )
            if built is not None:
                break
        if built is None:
            return None
        window, translated, visible = built
        source_windows.append(window)
        translated_by_color[int(color)] = translated
        visible_by_color[int(color)] = visible

    overlap_cells = set()
    colors = sprite_colors[:]
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            overlap_cells |= visible_by_color[colors[i]] & visible_by_color[colors[j]]
    if not overlap_cells:
        return None
    if len(overlap_cells) > 5:
        return None
    if crop_indices and not any(translated_by_color[sprite_colors[i]] - visible_by_color[sprite_colors[i]] for i in crop_indices):
        return None

    expected = _compose_expected((target_h, target_w), fill, legend_order, visible_by_color)
    if np.count_nonzero(expected != fill) < 8:
        return None
    for color in sprite_colors:
        if int(np.count_nonzero(expected == color)) < 2:
            return None

    sizes = [target_window.shape, *[window.shape for window in source_windows]]
    total_area = sum(h * w for h, w in sizes)
    min_side = max(18, int(np.ceil(np.sqrt(total_area * 2.1))))
    canvas_h = random.randint(min(min_side, 30), 30)
    canvas_w = random.randint(min(min_side, 30), 30)
    if n_sprites > canvas_w:
        return None

    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_cells = _legend_positions((canvas_h, canvas_w), corner, n_sprites)
    reserved = np.zeros((canvas_h, canvas_w), dtype=bool)
    for r, c in legend_cells:
        reserved[r, c] = True

    positions = _place_rectangles((canvas_h, canvas_w), sizes, reserved)
    if positions is None:
        return None

    grid = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    for (r, c), color in zip(legend_cells, legend_order):
        grid[r, c] = int(color)

    all_windows = [target_window, *source_windows]
    for (r0, c0), window in zip(positions, all_windows):
        h, w = window.shape
        grid[r0 : r0 + h, c0 : c0 + w] = window

    if _get_bg(grid) != bg:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
