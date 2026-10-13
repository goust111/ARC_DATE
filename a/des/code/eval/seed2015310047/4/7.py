from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
ORANGE = 7
PATTERN_COLORS = (1, 2, 3, 5, 6, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_ATTEMPTS = 400
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
PAIR_COUNT_CHOICES = (1, 2, 3)
INNER_H_RANGE = (3, 7)
INNER_W_RANGE = (3, 7)
PAIR_GAP_RANGE = (2, 6)
EXTRA_EMPTY_PROB = 0.4
PLACEMENT_RETRIES = 120
COMPONENT_SIZE_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _marker_side(local_r: int, local_c: int, h: int, w: int) -> str:
    hits = []
    if local_r == 0:
        hits.append("top")
    if local_r == h - 1:
        hits.append("bottom")
    if local_c == 0:
        hits.append("left")
    if local_c == w - 1:
        hits.append("right")
    assert len(hits) == 1, "yellow marker must lie on exactly one side"
    return hits[0]


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    orange_comps = _components(g == ORANGE)
    assert orange_comps, "expected at least one orange frame"

    all_yellows = {tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)}
    seen_yellows: set[tuple[int, int]] = set()
    frames: list[dict] = []

    for comp in orange_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame must be at least 3x3"

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        orange_mask = sub == ORANGE
        yellow_mask = sub == YELLOW

        assert np.all(~orange_mask | border), "orange pixels must stay on the rectangle border"
        assert int(yellow_mask.sum()) == 1, "each frame must have one yellow marker"
        assert np.array_equal(orange_mask | yellow_mask, border), "border must be orange except one yellow marker"

        local_y = np.argwhere(yellow_mask)
        local_r, local_c = map(int, local_y[0])
        side = _marker_side(local_r, local_c, h, w)
        yellow = (r0 + local_r, c0 + local_c)
        assert yellow not in seen_yellows, "yellow markers must belong to one frame each"
        seen_yellows.add(yellow)

        inner = sub[1:-1, 1:-1].copy()
        assert not np.any(inner == ORANGE), "orange should only appear on borders"
        assert not np.any(inner == YELLOW), "yellow should only appear on borders"
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "yellow": (int(yellow[0]), int(yellow[1])),
                "side": side,
                "inner": inner,
                "empty": bool(np.all(inner == BG)),
            }
        )

    assert seen_yellows == all_yellows, "every yellow pixel must mark exactly one frame"
    return frames


def _frame_distance(source: dict, target: dict) -> int | None:
    sr0, sc0, sr1, sc1 = source["bbox"]
    tr0, tc0, tr1, tc1 = target["bbox"]
    yr, yc = source["yellow"]
    side = source["side"]

    if side == "right":
        if tc0 > sc1 and tr0 < yr < tr1:
            return int(tc0 - sc1)
        return None
    if side == "left":
        if tc1 < sc0 and tr0 < yr < tr1:
            return int(sc0 - tc1)
        return None
    if side == "bottom":
        if tr0 > sr1 and tc0 < yc < tc1:
            return int(tr0 - sr1)
        return None
    if side == "top":
        if tr1 < sr0 and tc0 < yc < tc1:
            return int(sr0 - tr1)
        return None
    raise AssertionError(f"unknown side: {side}")


def _find_target(source: dict, empty_frames: list[dict]) -> dict:
    candidates: list[tuple[int, dict]] = []
    for frame in empty_frames:
        dist = _frame_distance(source, frame)
        if dist is not None:
            candidates.append((dist, frame))
    assert candidates, "each non-empty frame needs an empty target in its marked direction"
    best_dist = min(dist for dist, _ in candidates)
    best = [frame for dist, frame in candidates if dist == best_dist]
    assert len(best) == 1, "target frame must be uniquely nearest"
    return best[0]


def _mirrored(inner: np.ndarray, side: str) -> np.ndarray:
    if side in {"left", "right"}:
        return np.fliplr(inner)
    if side in {"top", "bottom"}:
        return np.flipud(inner)
    raise AssertionError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each interior pattern into the nearest empty frame found in the yellow marker direction.

    The copied pattern keeps its colors and shape, while the full source interior is mirrored
    across the axis implied by the exit direction: horizontal transfer flips left-right and
    vertical transfer flips up-down.
    """
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    sources = [frame for frame in frames if not frame["empty"]]
    empties = [frame for frame in frames if frame["empty"]]
    assert sources and empties, "expected both source and empty frames"

    out = g.copy()
    used_targets: set[tuple[int, int, int, int]] = set()

    for source in sources:
        target = _find_target(source, empties)
        target_bbox = target["bbox"]
        assert target_bbox not in used_targets, "a target frame should receive at most one copied pattern"
        used_targets.add(target_bbox)

        mirrored = _mirrored(source["inner"], source["side"])
        assert mirrored.shape == target["inner"].shape, "source and target interiors must have the same shape"
        tr0, tc0, tr1, tc1 = target_bbox
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = mirrored

    return out


def _rectangles_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                stack.append((int(nr), int(nc)))
    return len(seen) == int(mask.sum())


def _sample_motif(inner_h: int, inner_w: int, side: str, color: int) -> np.ndarray:
    max_h = min(3, inner_h)
    max_w = min(3, inner_w)
    for _ in range(200):
        mh, mw = random.choice([shape for shape in COMPONENT_SIZE_CHOICES if shape[0] <= max_h and shape[1] <= max_w])
        cell_count = random.randint(2, mh * mw - 1)
        flat = [0] * (mh * mw)
        for idx in random.sample(range(mh * mw), cell_count):
            flat[idx] = 1
        mask = np.array(flat, dtype=bool).reshape(mh, mw)
        if not _connected(mask):
            continue
        if side in {"left", "right"} and np.array_equal(mask, np.fliplr(mask)):
            continue
        if side in {"top", "bottom"} and np.array_equal(mask, np.flipud(mask)):
            continue

        inner = np.full((inner_h, inner_w), BG, dtype=int)
        top = random.randint(0, inner_h - mh)
        left = random.randint(0, inner_w - mw)
        inner[top : top + mh, left : left + mw][mask] = color
        if np.array_equal(inner, _mirrored(inner, side)):
            continue
        return inner
    raise RuntimeError("failed to sample an asymmetric motif")


def _marker_on_side(side: str, outer_h: int, outer_w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, outer_w - 2)
    if side == "bottom":
        return outer_h - 1, random.randint(1, outer_w - 2)
    if side == "left":
        return random.randint(1, outer_h - 2), 0
    if side == "right":
        return random.randint(1, outer_h - 2), outer_w - 1
    raise AssertionError(f"unknown side: {side}")


def _sample_marker(outer_h: int, outer_w: int) -> tuple[str, tuple[int, int]]:
    side = random.choice(SIDES)
    return side, _marker_on_side(side, outer_h, outer_w)


def _frame_patch(inner: np.ndarray, marker_side: str, marker_pos: tuple[int, int]) -> np.ndarray:
    outer_h = inner.shape[0] + 2
    outer_w = inner.shape[1] + 2
    patch = np.full((outer_h, outer_w), BG, dtype=int)
    patch[0, :] = ORANGE
    patch[-1, :] = ORANGE
    patch[:, 0] = ORANGE
    patch[:, -1] = ORANGE
    mr, mc = marker_pos
    assert _marker_side(mr, mc, outer_h, outer_w) == marker_side
    patch[mr, mc] = YELLOW
    patch[1:-1, 1:-1] = inner
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _line_boxes(
    direction: str,
    outer_h: int,
    outer_w: int,
    gap1: int,
    gap2: int | None,
    top: int,
    left: int,
) -> list[tuple[int, int, int, int]]:
    if direction == "right":
        boxes = [(top, left, top + outer_h - 1, left + outer_w - 1)]
        next_left = left + outer_w + gap1
        boxes.append((top, next_left, top + outer_h - 1, next_left + outer_w - 1))
        if gap2 is not None:
            far_left = next_left + outer_w + gap2
            boxes.append((top, far_left, top + outer_h - 1, far_left + outer_w - 1))
        return boxes
    if direction == "left":
        boxes = [(top, left, top + outer_h - 1, left + outer_w - 1)]
        next_left = left + outer_w + gap1
        boxes.append((top, next_left, top + outer_h - 1, next_left + outer_w - 1))
        if gap2 is not None:
            far_left = next_left + outer_w + gap2
            boxes.append((top, far_left, top + outer_h - 1, far_left + outer_w - 1))
        return list(reversed(boxes))
    if direction == "bottom":
        boxes = [(top, left, top + outer_h - 1, left + outer_w - 1)]
        next_top = top + outer_h + gap1
        boxes.append((next_top, left, next_top + outer_h - 1, left + outer_w - 1))
        if gap2 is not None:
            far_top = next_top + outer_h + gap2
            boxes.append((far_top, left, far_top + outer_h - 1, left + outer_w - 1))
        return boxes
    if direction == "top":
        boxes = [(top, left, top + outer_h - 1, left + outer_w - 1)]
        next_top = top + outer_h + gap1
        boxes.append((next_top, left, next_top + outer_h - 1, left + outer_w - 1))
        if gap2 is not None:
            far_top = next_top + outer_h + gap2
            boxes.append((far_top, left, far_top + outer_h - 1, left + outer_w - 1))
        return list(reversed(boxes))
    raise AssertionError(f"unknown direction: {direction}")


def _place_line(
    h: int,
    w: int,
    direction: str,
    outer_h: int,
    outer_w: int,
    existing: list[tuple[int, int, int, int]],
    add_far_empty: bool,
) -> list[tuple[int, int, int, int]] | None:
    gap1 = random.randint(*PAIR_GAP_RANGE)
    gap2 = random.randint(*PAIR_GAP_RANGE) if add_far_empty else None
    if direction in {"left", "right"}:
        total_h = outer_h
        total_w = 2 * outer_w + gap1 + (outer_w + gap2 if gap2 is not None else 0)
    else:
        total_h = 2 * outer_h + gap1 + (outer_h + gap2 if gap2 is not None else 0)
        total_w = outer_w
    if total_h > h or total_w > w:
        return None

    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(0, h - total_h)
        left = random.randint(0, w - total_w)
        boxes = _line_boxes(direction, outer_h, outer_w, gap1, gap2, top, left)
        if any(_rectangles_too_close(box, other, pad=1) for box in boxes for other in existing):
            continue
        return boxes
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    used_boxes: list[tuple[int, int, int, int]] = []

    if pair_count > len(PATTERN_COLORS):
        return None
    colors = random.sample(PATTERN_COLORS, pair_count)

    for color in colors:
        direction = random.choice(SIDES)
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        add_far_empty = random.random() < EXTRA_EMPTY_PROB

        boxes = _place_line(h, w, direction, outer_h, outer_w, used_boxes, add_far_empty)
        if boxes is None:
            return None

        source_inner = _sample_motif(inner_h, inner_w, direction, color)
        empty_inner = np.full((inner_h, inner_w), BG, dtype=int)
        source_marker = _marker_on_side(direction, outer_h, outer_w)
        tgt_side, tgt_marker = _sample_marker(outer_h, outer_w)

        patches = [
            _frame_patch(source_inner, direction, source_marker),
            _frame_patch(empty_inner, tgt_side, tgt_marker),
        ]
        if add_far_empty:
            far_side, far_marker = _sample_marker(outer_h, outer_w)
            patches.append(_frame_patch(empty_inner, far_side, far_marker))

        for box, patch in zip(boxes, patches):
            r0, c0, r1, c1 = box
            _paste(grid, patch, r0, c0)
            _paste(expected, patch, r0, c0)
            used_boxes.append(box)

        tr0, tc0, tr1, tc1 = boxes[1]
        expected[tr0 + 1 : tr1, tc0 + 1 : tc1] = _mirrored(source_inner, direction)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
