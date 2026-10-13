from __future__ import annotations

import random

import numpy as np


DIRS4: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 3
PALETTE = tuple(range(10))

GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (2, 4)
VERTICAL_HALF_WIDTH_RANGE = (2, 4)
HORIZONTAL_HALF_HEIGHT_RANGE = (2, 4)
OBJECT_HEIGHT_RANGE = (4, 8)
OBJECT_WIDTH_RANGE = (4, 8)
EXTRA_COUNT_RANGE = (1, 3)
MAX_OBJECT_TRIES = 120
MAX_GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _parse_legend(row: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) == bg:
            c += 1
            continue
        start = c
        while c < width and int(row[c]) != bg:
            c += 1
        run = [int(v) for v in row[start:c]]
        assert run and len(run) % 2 == 0, "legend runs must contain adjacent color pairs"
        for i in range(0, len(run), 2):
            left = int(run[i])
            right = int(run[i + 1])
            assert left != bg and right != bg
            mapping[left] = right
    assert mapping, "expected at least one legend pair"
    return mapping


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append({"color": color, "r0": r0, "c0": c0, "mask": mask, "coords": arr})
    return comps


def _axis_overlap(mask: np.ndarray, kind: str, axis_twice: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    negative: list[tuple[int, int]] = []
    positive: list[tuple[int, int]] = []

    if kind == "vertical":
        for r in range(h):
            for c in range(w):
                c2 = axis_twice - c
                if c < c2 and 0 <= c2 < w and bool(mask[r, c]) and bool(mask[r, c2]):
                    negative.append((r, c))
                    positive.append((r, int(c2)))
    else:
        for r in range(h):
            r2 = axis_twice - r
            if not (0 <= r2 < h) or not (r < r2):
                continue
            rr2 = int(r2)
            for c in range(w):
                if bool(mask[r, c]) and bool(mask[rr2, c]):
                    negative.append((r, c))
                    positive.append((rr2, c))

    return len(negative), negative, positive


def _best_axis(mask: np.ndarray) -> tuple[dict[str, object], bool]:
    h, w = map(int, mask.shape)
    best_score = -1
    second_score = -1
    best: dict[str, object] | None = None

    for axis_twice in range(1, 2 * w - 2):
        score, neg, pos = _axis_overlap(mask, "vertical", axis_twice)
        if score > best_score:
            second_score = best_score
            best_score = score
            best = {
                "kind": "vertical",
                "axis_twice": axis_twice,
                "score": score,
                "negative": neg,
                "positive": pos,
            }
        elif score == best_score:
            second_score = best_score
        elif score > second_score:
            second_score = score

    for axis_twice in range(1, 2 * h - 2):
        score, neg, pos = _axis_overlap(mask, "horizontal", axis_twice)
        if score > best_score:
            second_score = best_score
            best_score = score
            best = {
                "kind": "horizontal",
                "axis_twice": axis_twice,
                "score": score,
                "negative": neg,
                "positive": pos,
            }
        elif score == best_score:
            second_score = best_score
        elif score > second_score:
            second_score = score

    assert best is not None and int(best["score"]) > 0, "object must have a usable symmetry axis"
    return best, int(best["score"]) > second_score


def _prefer_positive_side(kind: str, axis_abs: float, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    if kind == "vertical":
        return axis_abs < (w - 1) / 2.0
    return axis_abs < (h - 1) / 2.0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each object onto its own best mirror axis, keep only the overlapping cells, and recolor them via the bottom legend.

    Input structure:
    - The dominant color is the green background.
    - Above the last row are several separated 4-connected monochrome foreground objects.
    - The bottom row is a legend made of adjacent two-cell pairs: the left color names an object color and the right color gives that object's output color.

    Rule:
    - Parse the bottom-row legend into object-color -> output-color mappings.
    - For each foreground object above the legend, test every horizontal and vertical reflection axis and choose the one with the most mirrored pixel pairs.
    - Fold the object across that axis, keep only pixels that have counterparts on both sides, and place the kept pixels on the side of the axis closer to the whole-grid center.
    - Recolor those kept pixels with the legend's output color, erase all other object pixels to background, and leave the legend row unchanged.

    Key insight:
    - The transformation is not symmetry completion; it is symmetry intersection plus a separate color remapping specified by the bottom legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = _background_color(g)
    legend_map = _parse_legend(g[-1], bg)
    scene = g[:-1]
    out = np.full(g.shape, bg, dtype=int)
    out[-1] = g[-1]

    for comp in _find_components(scene, bg):
        color = int(comp["color"])
        assert color in legend_map, "every object color must appear in the legend"
        r0 = int(comp["r0"])
        c0 = int(comp["c0"])
        mask = np.asarray(comp["mask"], dtype=bool)

        best, _ = _best_axis(mask)
        kind = str(best["kind"])
        axis_twice = int(best["axis_twice"])
        negative = list(best["negative"])
        positive = list(best["positive"])

        if kind == "vertical":
            axis_abs = c0 + axis_twice / 2.0
        else:
            axis_abs = r0 + axis_twice / 2.0

        kept = positive if _prefer_positive_side(kind, axis_abs, g.shape) else negative
        out_color = int(legend_map[color])
        for rr, cc in kept:
            out[r0 + rr, c0 + cc] = out_color

    return out


def _grow_connected(shape: tuple[int, int], target: int, seeds: list[tuple[int, int]]) -> np.ndarray | None:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    seed = random.choice(seeds)
    mask[seed] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while int(mask.sum()) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        add_frontier(*cell)
    return mask


def _extra_candidates(mask: np.ndarray, orientation: str, span: int, side: str) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if orientation == "vertical":
                if side == "left" and not (0 <= c < span):
                    continue
                if side == "right" and not (span <= c < 2 * span):
                    continue
            else:
                if side == "top" and not (0 <= r < span):
                    continue
                if side == "bottom" and not (span <= r < 2 * span):
                    continue
            if any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in DIRS4):
                candidates.append((r, c))
    return candidates


def _make_object_spec(orientation: str) -> dict[str, object] | None:
    for _ in range(MAX_OBJECT_TRIES):
        if orientation == "vertical":
            h = random.randint(*OBJECT_HEIGHT_RANGE)
            span = random.randint(*VERTICAL_HALF_WIDTH_RANGE)
            max_area = h * span - 1
            min_area = max(3, span + 1)
            if min_area > max_area:
                continue
            target = random.randint(min_area, max_area)
            core = _grow_connected((h, span), target, [(r, 0) for r in range(h)])
            if core is None or not np.any(core[:, 0]):
                continue

            mask = np.zeros((h, 2 * span), dtype=bool)
            for r_raw, c_raw in np.argwhere(core):
                r = int(r_raw)
                c = int(c_raw)
                mask[r, span + c] = True
                mask[r, span - 1 - c] = True
            extra_side = random.choice(("left", "right"))
            orientation_side = extra_side
        else:
            span = random.randint(*HORIZONTAL_HALF_HEIGHT_RANGE)
            w = random.randint(*OBJECT_WIDTH_RANGE)
            max_area = span * w - 1
            min_area = max(3, span + 1)
            if min_area > max_area:
                continue
            target = random.randint(min_area, max_area)
            core = _grow_connected((span, w), target, [(0, c) for c in range(w)])
            if core is None or not np.any(core[0, :]):
                continue

            mask = np.zeros((2 * span, w), dtype=bool)
            for r_raw, c_raw in np.argwhere(core):
                r = int(r_raw)
                c = int(c_raw)
                mask[span + r, c] = True
                mask[span - 1 - r, c] = True
            extra_side = random.choice(("top", "bottom"))
            orientation_side = extra_side

        extras = random.randint(*EXTRA_COUNT_RANGE)
        ok = True
        for _ in range(extras):
            candidates = _extra_candidates(mask, orientation, span, orientation_side)
            if not candidates:
                ok = False
                break
            mask[random.choice(candidates)] = True
        if not ok:
            continue

        best, unique = _best_axis(mask)
        if not unique:
            continue
        if str(best["kind"]) != orientation:
            continue
        if int(best["axis_twice"]) != 2 * span - 1:
            continue
        if int(best["score"]) != int(core.sum()):
            continue

        return {
            "orientation": orientation,
            "span": span,
            "mask": mask,
            "core": core,
        }
    return None


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _axis_abs(spec: dict[str, object], r0: int, c0: int) -> float:
    if str(spec["orientation"]) == "vertical":
        return c0 + int(spec["span"]) - 0.5
    return r0 + int(spec["span"]) - 0.5


def _place_objects(specs: list[dict[str, object]], h: int, w: int) -> list[tuple[int, int]] | None:
    scene_h = h - 2
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    occupied = np.zeros((scene_h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: int(np.asarray(specs[i]["mask"], dtype=bool).sum()), reverse=True)

    for idx in order:
        mask = np.asarray(specs[idx]["mask"], dtype=bool)
        oh, ow = map(int, mask.shape)
        if oh > scene_h or ow > w:
            return None
        dilated = _dilate(occupied)
        candidates: list[tuple[int, int]] = []
        for r0 in range(scene_h - oh + 1):
            for c0 in range(w - ow + 1):
                if np.any(dilated[r0 : r0 + oh, c0 : c0 + ow] & mask):
                    continue
                axis_abs = _axis_abs(specs[idx], r0, c0)
                if str(specs[idx]["orientation"]) == "vertical":
                    if axis_abs == (w - 1) / 2.0:
                        continue
                else:
                    if axis_abs == (h - 1) / 2.0:
                        continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + oh, c0 : c0 + ow] |= mask

    assert all(pos is not None for pos in placements)
    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _legend_row(width: int, pairs: list[tuple[int, int]], bg: int) -> np.ndarray | None:
    pair_count = len(pairs)
    base = 2 * pair_count + max(0, pair_count - 1)
    if base > width:
        return None
    extras = width - base
    gaps = [0] * (pair_count + 1)
    for _ in range(extras):
        gaps[random.randrange(pair_count + 1)] += 1

    row = np.full(width, bg, dtype=int)
    cursor = gaps[0]
    for i, (src, dst) in enumerate(pairs):
        row[cursor] = src
        row[cursor + 1] = dst
        cursor += 2
        if i < pair_count - 1:
            cursor += 1 + gaps[i + 1]
    return row


def _paint_object(grid: np.ndarray, spec: dict[str, object], r0: int, c0: int, color: int) -> None:
    mask = np.asarray(spec["mask"], dtype=bool)
    region = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    region[mask] = int(color)


def _paint_expected_overlap(grid: np.ndarray, spec: dict[str, object], r0: int, c0: int, color: int) -> None:
    orientation = str(spec["orientation"])
    span = int(spec["span"])
    core = np.asarray(spec["core"], dtype=bool)
    axis_abs = _axis_abs(spec, r0, c0)
    positive = _prefer_positive_side(orientation, axis_abs, grid.shape)

    for rr_raw, cc_raw in np.argwhere(core):
        rr = int(rr_raw)
        cc = int(cc_raw)
        if orientation == "vertical":
            col = c0 + span + cc if positive else c0 + span - 1 - cc
            row = r0 + rr
        else:
            row = r0 + span + rr if positive else r0 + span - 1 - rr
            col = c0 + cc
        grid[row, col] = int(color)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    orientations = [random.choice(("vertical", "horizontal")) for _ in range(count)]
    specs: list[dict[str, object]] = []
    for orientation in orientations:
        spec = _make_object_spec(orientation)
        if spec is None:
            return None
        specs.append(spec)

    total_area = sum(int(np.asarray(spec["mask"], dtype=bool).sum()) for spec in specs)
    max_h = max(int(np.asarray(spec["mask"], dtype=bool).shape[0]) for spec in specs)
    max_w = max(int(np.asarray(spec["mask"], dtype=bool).shape[1]) for spec in specs)
    min_h = max(GRID_H_RANGE[0], max_h + 3)
    min_w = max(GRID_W_RANGE[0], max(max_w, 3 * count - 1))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h_choices = [h for h in range(min_h, GRID_H_RANGE[1] + 1)]
    w_choices = [w for w in range(min_w, GRID_W_RANGE[1] + 1)]
    random.shuffle(h_choices)
    random.shuffle(w_choices)

    placements: list[tuple[int, int]] | None = None
    h = w = 0
    for hh in h_choices:
        for ww in w_choices:
            if (hh - 2) * ww < total_area * 3:
                continue
            placements = _place_objects(specs, hh, ww)
            if placements is not None:
                h, w = hh, ww
                break
        if placements is not None:
            break
    if placements is None:
        return None

    available = [c for c in PALETTE if c != BG_COLOR]
    chosen = random.sample(available, 2 * count)
    object_colors = chosen[:count]
    output_colors = chosen[count:]
    legend_pairs = list(zip(object_colors, output_colors))
    random.shuffle(legend_pairs)
    legend = _legend_row(w, legend_pairs, BG_COLOR)
    if legend is None:
        return None

    inp = np.full((h, w), BG_COLOR, dtype=int)
    inp[-1] = legend
    expected = np.full((h, w), BG_COLOR, dtype=int)
    expected[-1] = legend

    for spec, (r0, c0), object_color, output_color in zip(specs, placements, object_colors, output_colors):
        _paint_object(inp, spec, r0, c0, object_color)
        _paint_expected_overlap(expected, spec, r0, c0, output_color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-overlap ARC input")
