from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
EDGE_COLORS = (RED, GREEN, YELLOW)
STEP_TO_CW_TURNS = {2: 0, 3: 1, 4: 2, 5: 3}

_GEN_COUNTER = 0
_STEP_PRESETS = [
    (2, 3, 4),
    (3, 4, 5),
    (2, 4, 5),
    (2, 3, 5),
    (2, 5),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "mask must be non-empty"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_mask_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, (-turns) % 4).copy()


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_contains_point(box: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    r, c = point
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _stamp_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                dst[top + r, left + c] = color


def _find_template(grid: np.ndarray) -> np.ndarray:
    comps = _components4(grid == GREY)
    assert len(comps) == 1, "expected exactly one grey template component"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    return crop == GREY


def _decode_steps(grid: np.ndarray) -> dict[int, int]:
    left_col = grid[:, 0]
    color_rows: dict[int, list[int]] = {}
    for r, value in enumerate(left_col):
        color = int(value)
        if color == BG:
            continue
        color_rows.setdefault(color, []).append(r)
    assert color_rows, "expected left-edge color pairs"
    steps: dict[int, int] = {}
    for color, rows in color_rows.items():
        assert len(rows) == 2, f"left-edge color {color} must appear exactly twice"
        rows = sorted(rows)
        step = rows[1] - rows[0]
        assert step in STEP_TO_CW_TURNS, f"unsupported encoded step {step}"
        steps[int(color)] = int(step)
    return steps


def _sorted_markers(grid: np.ndarray, colors: set[int]) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(1, w):
            color = int(grid[r, c])
            if color in colors:
                markers.append((r, c, color))
    return markers


def _apply_rule(grid: np.ndarray, template_mask: np.ndarray, steps: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    marker_colors = set(steps)
    markers = _sorted_markers(out, marker_colors)
    for r, c, color in markers:
        turns = STEP_TO_CW_TURNS[steps[color]]
        rotated = _rotate_mask_cw(template_mask, turns)
        _stamp_mask(out, rotated, r, c, color)

    out[:, 0] = grid[:, 0]
    for r, c, color in markers:
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the grey template by the left-edge gap code and stamp recolored copies at matching markers.

    Input structure:
    - The background is black, with a single grey irregular template sprite in the right half.
    - On the left edge, each active color appears exactly twice; the row gap of that pair encodes a rotation.
    - Elsewhere, single-pixel markers use those same colors to indicate where stamped copies should be placed.

    Rule:
    - For each color on the left edge, compute the gap between its two edge pixels.
    - Map gaps 2/3/4/5 to clockwise rotations 0/90/180/270 degrees respectively.
    - Find every non-edge marker of that color, rotate the grey template accordingly, and stamp it with the marker as the rotated bbox's top-left corner.
    - Recolor stamped pixels to the marker color, keep marker pixels and left-edge pixels unchanged, and resolve stamp overlaps by row-major marker order so later stamps overwrite earlier ones.

    Key insight:
    - The left-edge spacing is not a fill period but a hidden rotation code; color matches that code to each marker's placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    template_mask = _find_template(g)
    steps = _decode_steps(g)
    return _apply_rule(g, template_mask, steps)


def _all_rotations_unique(mask: np.ndarray) -> bool:
    rots: list[np.ndarray] = []
    for k in range(4):
        rot = _trim_mask(np.rot90(mask, k))
        for prev in rots:
            if rot.shape == prev.shape and np.array_equal(rot, prev):
                return False
        rots.append(rot)
    return True


def _random_connected_template(rng: random.Random) -> np.ndarray:
    for _ in range(400):
        canvas_h = rng.randint(2, 4)
        canvas_w = rng.randint(3, 5)
        max_area = min(8, canvas_h * canvas_w - 1)
        if max_area < 4:
            continue
        target_area = rng.randint(4, max_area)
        seed = (rng.randrange(canvas_h), rng.randrange(canvas_w))
        cells = {seed}
        frontier = {nb for nb in _neighbors4(*seed) if 0 <= nb[0] < canvas_h and 0 <= nb[1] < canvas_w}
        while len(cells) < target_area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(*cell):
                if 0 <= nb[0] < canvas_h and 0 <= nb[1] < canvas_w and nb not in cells:
                    frontier.add(nb)
        if len(cells) != target_area:
            continue

        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) < 4 or int(mask.sum()) > 8:
            continue
        if np.all(mask):
            continue
        if len(_components4(mask)) != 1:
            continue
        if not _all_rotations_unique(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric template")


def _choose_steps_and_colors(rng: random.Random) -> dict[int, int]:
    global _GEN_COUNTER
    preset = _STEP_PRESETS[_GEN_COUNTER % len(_STEP_PRESETS)]
    _GEN_COUNTER += 1
    if len(preset) == 2:
        colors = rng.sample(list(EDGE_COLORS), 2)
    else:
        colors = rng.sample(list(EDGE_COLORS), 3)
    rng.shuffle(colors)
    return {color: step for color, step in zip(colors, preset)}


def _choose_marker_counts(colors: list[int], rng: random.Random) -> dict[int, int]:
    if len(colors) == 2:
        options = [(1, 1), (2, 1), (2, 2)]
    else:
        options = [(1, 1, 1), (2, 1, 1), (2, 2, 1)]
    counts = list(rng.choice(options))
    rng.shuffle(counts)
    return {color: count for color, count in zip(colors, counts)}


def _choose_edge_pairs(height: int, steps: dict[int, int], rng: random.Random) -> dict[int, tuple[int, int]]:
    items = list(steps.items())
    for _ in range(300):
        rng.shuffle(items)
        used_rows: set[int] = set()
        pairs: dict[int, tuple[int, int]] = {}
        ok = True
        for color, step in items:
            candidates = [r for r in range(height - step) if r not in used_rows and r + step not in used_rows]
            if not candidates:
                ok = False
                break
            r0 = rng.choice(candidates)
            used_rows.add(r0)
            used_rows.add(r0 + step)
            pairs[color] = (r0, r0 + step)
        if ok:
            return pairs
    raise RuntimeError("failed to place left-edge pairs")


def _anchor_candidates(
    h: int,
    w: int,
    stamp_shape: tuple[int, int],
    template_box: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
    points: list[tuple[int, int]],
    allow_overlap_index: int | None = None,
    require_overlap_index: int | None = None,
) -> list[tuple[int, int]]:
    sh, sw = stamp_shape
    max_c = template_box[1] - sw - 1
    out: list[tuple[int, int]] = []
    if max_c < 2:
        return out
    for r in range(0, h - sh + 1):
        for c in range(2, max_c + 1):
            box = (r, c, r + sh - 1, c + sw - 1)
            if _box_overlap(box, template_box):
                continue

            bad = False
            for idx, other in enumerate(boxes):
                if require_overlap_index is not None and idx == require_overlap_index:
                    if not _box_overlap(box, other):
                        bad = True
                        break
                    if _box_contains_point(other, (r, c)):
                        bad = True
                        break
                elif allow_overlap_index is not None and idx == allow_overlap_index:
                    if not _box_overlap(box, other):
                        bad = True
                        break
                    if _box_contains_point(other, (r, c)):
                        bad = True
                        break
                else:
                    if _box_overlap(box, other):
                        bad = True
                        break
                if _box_contains_point(box, points[idx]):
                    bad = True
                    break
            if bad:
                continue
            out.append((r, c))
    return out


def _manual_expected(
    inp: np.ndarray,
    template_mask: np.ndarray,
    steps: dict[int, int],
) -> np.ndarray:
    return _apply_rule(inp, template_mask, steps)


def _attempt_overlap_reposition(
    specs: list[dict[str, object]],
    points: list[tuple[int, int]],
    boxes: list[tuple[int, int, int, int]],
    template_box: tuple[int, int, int, int],
    h: int,
    w: int,
    rng: random.Random,
) -> bool:
    for i in range(len(specs)):
        for j in range(len(specs)):
            if i == j:
                continue
            if specs[i]["color"] == specs[j]["color"]:
                continue
            candidates = _anchor_candidates(
                h,
                w,
                tuple(specs[j]["mask"].shape),
                template_box,
                boxes,
                points,
                allow_overlap_index=i,
                require_overlap_index=i,
            )
            candidates = [cand for cand in candidates if cand != points[j]]
            if not candidates:
                continue
            r, c = rng.choice(candidates)
            sh, sw = tuple(specs[j]["mask"].shape)
            points[j] = (r, c)
            boxes[j] = (r, c, r + sh - 1, c + sw - 1)
            return True
    return False


def _generate_one(rng: random.Random) -> np.ndarray:
    steps = _choose_steps_and_colors(rng)
    colors = list(steps)
    marker_counts = _choose_marker_counts(colors, rng)
    template_mask = _random_connected_template(rng)

    stamp_masks = {color: _rotate_mask_cw(template_mask, STEP_TO_CW_TURNS[step]) for color, step in steps.items()}

    for _ in range(300):
        height = rng.randint(13, 18)
        width = rng.randint(13, 18)
        th, tw = template_mask.shape
        tc_min = max(width // 2, 7)
        tc_max = width - tw
        if tc_min > tc_max:
            continue
        tr = rng.randint(0, height - th)
        tc = rng.randint(tc_min, tc_max)
        template_box = (tr, tc, tr + th - 1, tc + tw - 1)

        specs: list[dict[str, object]] = []
        for color in colors:
            for _ in range(marker_counts[color]):
                specs.append({"color": color, "mask": stamp_masks[color]})
        rng.shuffle(specs)

        points: list[tuple[int, int]] = []
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for spec in specs:
            candidates = _anchor_candidates(height, width, tuple(spec["mask"].shape), template_box, boxes, points)
            if not candidates:
                ok = False
                break
            r, c = rng.choice(candidates)
            sh, sw = tuple(spec["mask"].shape)
            points.append((r, c))
            boxes.append((r, c, r + sh - 1, c + sw - 1))
        if not ok:
            continue

        overlap_mode = rng.random() < 0.3 or (_GEN_COUNTER % 4 == 0)
        if overlap_mode:
            _attempt_overlap_reposition(specs, points, boxes, template_box, height, width, rng)

        edge_pairs = _choose_edge_pairs(height, steps, rng)

        grid = np.zeros((height, width), dtype=int)
        grid[tr : tr + th, tc : tc + tw][template_mask] = GREY

        for color, (r0, r1) in edge_pairs.items():
            grid[r0, 0] = color
            grid[r1, 0] = color

        for spec, (r, c) in zip(specs, points):
            grid[r, c] = int(spec["color"])

        expected = _manual_expected(grid, template_mask, steps)
        solved = main(grid)
        if not np.array_equal(expected, solved):
            continue
        if np.array_equal(grid, solved):
            continue
        if len(_components4(grid == GREY)) != 1:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    rng = random.Random()
    return _generate_one(rng)
