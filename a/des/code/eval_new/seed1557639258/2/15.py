from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

LEGEND_THICKNESS = 7
FRAME_SIZE = 5
INNER_SIZE = 3
ENTRY_GAP = 1
GEN_RETRIES = 400
PLACE_RETRIES = 250
AVAILABLE_COLORS = tuple(range(10))

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

BASE_SHAPES = [
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
]


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_slab(grid: np.ndarray, side: str) -> np.ndarray | None:
    h, w = grid.shape
    if side == "top" and h > LEGEND_THICKNESS:
        return grid[:LEGEND_THICKNESS, :]
    if side == "bottom" and h > LEGEND_THICKNESS:
        return grid[h - LEGEND_THICKNESS :, :]
    if side == "left" and w > LEGEND_THICKNESS:
        return grid[:, :LEGEND_THICKNESS]
    if side == "right" and w > LEGEND_THICKNESS:
        return grid[:, w - LEGEND_THICKNESS :]
    return None


def _crop_main(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid[LEGEND_THICKNESS:, :]
    if side == "bottom":
        return grid[:-LEGEND_THICKNESS, :]
    if side == "left":
        return grid[:, LEGEND_THICKNESS:]
    if side == "right":
        return grid[:, :-LEGEND_THICKNESS]
    raise AssertionError("invalid legend side")


def _legend_candidate(grid: np.ndarray, side: str):
    slab = _extract_slab(grid, side)
    if slab is None:
        return None

    if side in {"top", "bottom"}:
        if not (
            np.all(slab[0, :] == slab[0, 0])
            and np.all(slab[-1, :] == slab[0, 0])
        ):
            return None
        pad_color = int(slab[0, 0])
    else:
        if not (
            np.all(slab[:, 0] == slab[0, 0])
            and np.all(slab[:, -1] == slab[0, 0])
        ):
            return None
        pad_color = int(slab[0, 0])

    best = None
    for color in map(int, np.unique(slab)):
        if color == pad_color:
            continue
        rings: list[tuple[int, int, int, int]] = []
        for comp in _components_mask(slab == color, connectivity=4):
            if len(comp) != 16:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != (FRAME_SIZE, FRAME_SIZE):
                continue
            if side in {"top", "bottom"} and (r0 != 1 or r1 != 5):
                continue
            if side in {"left", "right"} and (c0 != 1 or c1 != 5):
                continue
            patch = slab[r0 : r1 + 1, c0 : c1 + 1]
            border = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            border[1:-1, 1:-1] = False
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[1:-1, 1:-1] == color):
                continue
            rings.append((r0, c0, r1, c1))

        if len(rings) >= 2:
            score = (len(rings), -color)
            if best is None or score > best[0]:
                best = (score, pad_color, color, rings)

    if best is None:
        return None

    _, pad_color, frame_color, rings = best
    return {
        "side": side,
        "slab": slab,
        "pad_color": int(pad_color),
        "frame_color": int(frame_color),
        "rings": sorted(rings, key=lambda box: (box[1], box[0])),
    }


def _parse_legend(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[tuple[int, int], ...]]]:
    candidates = []
    for side in ("top", "bottom", "left", "right"):
        cand = _legend_candidate(grid, side)
        if cand is not None:
            candidates.append(cand)
    assert candidates, "expected one edge legend band"

    candidates.sort(key=lambda item: len(item["rings"]), reverse=True)
    chosen = candidates[0]
    slab = chosen["slab"]
    pad_color = chosen["pad_color"]
    frame_color = chosen["frame_color"]

    rules: dict[int, tuple[tuple[int, int], ...]] = {}
    for r0, c0, _r1, _c1 in chosen["rings"]:
        inner = slab[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        crystal = int(inner[1, 1])
        assert crystal not in (pad_color, frame_color), "template center must carry the crystal color"

        offsets = []
        for rr in range(INNER_SIZE):
            for cc in range(INNER_SIZE):
                value = int(inner[rr, cc])
                assert value in (pad_color, crystal), "template interior must only use legend bg and crystal color"
                if value == crystal and (rr, cc) != (1, 1):
                    offsets.append((rr - 1, cc - 1))
        assert offsets, "every template must mark at least one growth direction"
        assert crystal not in rules, "each crystal color must have exactly one legend entry"
        rules[crystal] = tuple(sorted(offsets))

    assert len(rules) >= 2, "legend must encode at least two crystal colors"
    return _crop_main(grid, chosen["side"]), rules


def _collect_proposals(grid: np.ndarray, rules: dict[int, tuple[tuple[int, int], ...]]):
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    h, w = g.shape
    proposals: dict[tuple[int, int], set[int]] = defaultdict(set)

    for color, offsets in rules.items():
        for r, c in np.argwhere(g == color):
            r = int(r)
            c = int(c)
            boundary = False
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == bg:
                    boundary = True
                    break
            if not boundary:
                continue

            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == bg:
                    proposals[(nr, nc)].add(int(color))

    return bg, proposals


def _grow_once(grid: np.ndarray, rules: dict[int, tuple[tuple[int, int], ...]]) -> tuple[np.ndarray, dict[tuple[int, int], set[int]]]:
    g = np.asarray(grid, dtype=int)
    bg, proposals = _collect_proposals(g, rules)
    out = g.copy()
    for (r, c), colors in proposals.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        else:
            out[r, c] = bg
    return out, proposals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the edge legend and grow each colored seed crystal by one template-guided step.

    Input structure:
    - The main canvas has a dominant background color and several small 4-connected monochrome seed crystals in different colors.
    - One grid edge holds a 7-cell-thick legend band. Inside that band, each crystal color has a framed 5x5 cell whose inner 3x3 patch marks the allowed growth directions around a colored center.
    - Legend cells use a neutral padding color and a separate frame color; only the inner 3x3 crystal-colored marks encode growth.

    Rule:
    - Detect the legend band, read each 3x3 template as a color -> direction set, and crop the full legend band away.
    - On the cropped main canvas, every crystal boundary pixel with at least one orthogonal background neighbor attempts to grow one same-colored pixel in each marked template direction, but only into background cells.
    - All crystals grow simultaneously for exactly one step. If different colors propose the same target cell, that cell stays background.

    Key insight:
    - The framed 3x3 legend patches are anisotropic growth kernels: once the legend is removed, the task is a single synchronous cellular-growth step with visible color-specific direction masks and conflict cancellation.
    """
    g = np.asarray(grid, dtype=int)
    main_grid, rules = _parse_legend(g)
    out, _ = _grow_once(main_grid, rules)
    return out


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells_list = list(cells)
    r0 = min(r for r, _ in cells_list)
    c0 = min(c for _, c in cells_list)
    return sorted((r - r0, c - c0) for r, c in cells_list)


def _rotate_cw(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h = max(r for r, _ in cells) + 1
    return _normalize([(c, h - 1 - r) for r, c in cells])


def _flip_h(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    w = max(c for _, c in cells) + 1
    return _normalize([(r, w - 1 - c) for r, c in cells])


def _random_shape() -> list[tuple[int, int]]:
    cells = _normalize(random.choice(BASE_SHAPES))
    for _ in range(random.randint(0, 3)):
        cells = _rotate_cw(cells)
    if random.random() < 0.5:
        cells = _flip_h(cells)
    return cells


def _shape_dims(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _random_rule() -> tuple[tuple[int, int], ...]:
    mode = random.choice(["orth", "diag", "mixed"])
    if mode == "orth":
        size = random.randint(1, len(ORTH_DIRS))
        return tuple(sorted(random.sample(ORTH_DIRS, size)))
    if mode == "diag":
        size = random.randint(1, len(DIAG_DIRS))
        return tuple(sorted(random.sample(DIAG_DIRS, size)))

    size = random.randint(2, 5)
    offsets = set(random.sample(ORTH_DIRS, random.randint(1, min(2, size - 1))))
    remaining = size - len(offsets)
    offsets.update(random.sample(DIAG_DIRS, 1))
    remaining = size - len(offsets)
    pool = [d for d in DIR8 if d not in offsets]
    if remaining > 0:
        offsets.update(random.sample(pool, remaining))
    return tuple(sorted(offsets))


def _fit_dims(side: str) -> tuple[int, int]:
    if side in {"top", "bottom"}:
        main_h = random.randint(12, 22)
        main_w = random.randint(18, 30)
    else:
        main_h = random.randint(18, 30)
        main_w = random.randint(12, 22)
    return main_h, main_w


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], radius: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                blocked[rr, cc] = True


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    color: int,
    shape: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    sh, sw = _shape_dims(shape)
    row_lo = 2
    row_hi = h - sh - 2
    col_lo = 2
    col_hi = w - sw - 2
    if row_lo > row_hi or col_lo > col_hi:
        return False

    for _ in range(PLACE_RETRIES):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(blocked[r, c] or int(grid[r, c]) != bg for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        _mark_blocked(blocked, cells, radius=2)
        return True
    return False


def _find_conflict_pair(colors: list[int], rules: dict[int, tuple[tuple[int, int], ...]]):
    options = []
    for i, c1 in enumerate(colors):
        for c2 in colors[i + 1 :]:
            for off1 in rules[c1]:
                for off2 in rules[c2]:
                    if off1 != off2:
                        options.append((c1, c2, off1, off2))
    if not options:
        return None
    return random.choice(options)


def _place_conflict_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    colors: tuple[int, int],
    offsets: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    h, w = grid.shape
    c1, c2 = colors
    off1, off2 = offsets
    targets = [
        (r, c)
        for r in range(3, h - 3)
        for c in range(3, w - 3)
    ]
    random.shuffle(targets)

    for tr, tc in targets:
        p1 = (tr - off1[0], tc - off1[1])
        p2 = (tr - off2[0], tc - off2[1])
        cells = [p1, p2]
        if len({p1, p2, (tr, tc)}) != 3:
            continue
        if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in cells):
            continue
        if any(blocked[r, c] or int(grid[r, c]) != bg for r, c in cells):
            continue
        if blocked[tr, tc] or int(grid[tr, tc]) != bg:
            continue
        grid[p1] = c1
        grid[p2] = c2
        _mark_blocked(blocked, [p1, p2, (tr, tc)], radius=3)
        return True
    return False


def _render_legend(
    main_grid: np.ndarray,
    side: str,
    pad_color: int,
    frame_color: int,
    rules: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    h, w = main_grid.shape
    if side == "top":
        out = np.full((h + LEGEND_THICKNESS, w), pad_color, dtype=int)
        out[LEGEND_THICKNESS:, :] = main_grid
        slab = out[:LEGEND_THICKNESS, :]
        axis_len = w
        vertical = False
    elif side == "bottom":
        out = np.full((h + LEGEND_THICKNESS, w), pad_color, dtype=int)
        out[:h, :] = main_grid
        slab = out[h:, :]
        axis_len = w
        vertical = False
    elif side == "left":
        out = np.full((h, w + LEGEND_THICKNESS), pad_color, dtype=int)
        out[:, LEGEND_THICKNESS:] = main_grid
        slab = out[:, :LEGEND_THICKNESS]
        axis_len = h
        vertical = True
    else:
        out = np.full((h, w + LEGEND_THICKNESS), pad_color, dtype=int)
        out[:, :w] = main_grid
        slab = out[:, w:]
        axis_len = h
        vertical = True

    colors = list(rules.keys())
    random.shuffle(colors)
    span = len(colors) * FRAME_SIZE + (len(colors) - 1) * ENTRY_GAP
    start = random.randint(1, axis_len - span - 1)

    for idx, color in enumerate(colors):
        anchor = start + idx * (FRAME_SIZE + ENTRY_GAP)
        if vertical:
            r0, c0 = anchor, 1
        else:
            r0, c0 = 1, anchor
        slab[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = pad_color
        slab[r0 : r0 + FRAME_SIZE, c0] = frame_color
        slab[r0 : r0 + FRAME_SIZE, c0 + FRAME_SIZE - 1] = frame_color
        slab[r0, c0 : c0 + FRAME_SIZE] = frame_color
        slab[r0 + FRAME_SIZE - 1, c0 : c0 + FRAME_SIZE] = frame_color
        inner = slab[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        inner[1, 1] = color
        for dr, dc in rules[color]:
            inner[1 + dr, 1 + dc] = color

    return out


def _new_pixels_by_color(inp: np.ndarray, out: np.ndarray, bg: int, colors: list[int]) -> dict[int, int]:
    result = {}
    for color in colors:
        result[color] = int(np.sum((inp == bg) & (out == color)))
    return result


def _build_sample() -> np.ndarray | None:
    side = random.choice(["top", "bottom", "left", "right"])
    main_h, main_w = _fit_dims(side)
    axis_len = main_w if side in {"top", "bottom"} else main_h
    max_colors = min(4, (axis_len - 1) // 6)
    if max_colors < 2:
        return None
    n_colors = random.randint(2, max_colors)

    palette = list(AVAILABLE_COLORS)
    random.shuffle(palette)
    bg, pad_color, frame_color = palette[:3]
    crystal_colors = palette[3 : 3 + n_colors]
    if len(crystal_colors) < n_colors:
        return None

    rules = {color: _random_rule() for color in crystal_colors}
    force_conflict = bool(random.getrandbits(1))
    if force_conflict:
        pair = _find_conflict_pair(crystal_colors, rules)
        if pair is None:
            c1, c2 = crystal_colors[:2]
            rules[c1] = tuple(sorted(set(rules[c1]) | {(0, 1)}))
            rules[c2] = tuple(sorted(set(rules[c2]) | {(0, -1)}))
        else:
            c1, c2, off1, off2 = pair
            if off1 == off2:
                rules[c1] = tuple(sorted(set(rules[c1]) | {(0, 1)}))
                rules[c2] = tuple(sorted(set(rules[c2]) | {(0, -1)}))

    main_grid = np.full((main_h, main_w), bg, dtype=int)
    blocked = np.zeros((main_h, main_w), dtype=bool)

    counts = {color: random.randint(1, 3) for color in crystal_colors}

    if force_conflict:
        conflict_pair = _find_conflict_pair(crystal_colors, rules)
        if conflict_pair is None:
            return None
        c1, c2, off1, off2 = conflict_pair
        counts[c1] = max(counts[c1], 2)
        counts[c2] = max(counts[c2], 2)
        if not _place_conflict_pair(main_grid, blocked, bg, (c1, c2), (off1, off2)):
            return None
        counts[c1] -= 1
        counts[c2] -= 1

    tasks: list[tuple[int, list[tuple[int, int]]]] = []
    for color in crystal_colors:
        for _ in range(counts[color]):
            tasks.append((color, _random_shape()))
    random.shuffle(tasks)

    for color, shape in tasks:
        if not _place_shape(main_grid, blocked, bg, color, shape):
            return None

    grown, proposals = _grow_once(main_grid, rules)
    bg_out = _majority_color(main_grid)
    new_by_color = _new_pixels_by_color(main_grid, grown, bg_out, crystal_colors)
    conflict_count = sum(1 for colors in proposals.values() if len(colors) > 1)

    if force_conflict and conflict_count == 0:
        return None
    if not force_conflict and conflict_count != 0:
        return None
    if any(count == 0 for count in new_by_color.values()):
        return None
    if np.array_equal(main_grid, grown):
        return None

    full_input = _render_legend(main_grid, side, pad_color, frame_color, rules)
    if not np.array_equal(main(full_input), grown):
        return None
    return full_input


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _build_sample()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid crystal-growth sample")
