from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = (GREY, MAROON)
FRAME_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

PATTERN_SIZE_MIN = 2
PATTERN_SIZE_MAX = 5
INNER_PADDING_MIN = 0
INNER_PADDING_MAX = 2
MIN_LAYERS = 2
MAX_LAYERS = 4
CANVAS_MAX = 36
GEN_RETRIES = 500


def _dominant_color(g: np.ndarray) -> int:
    values = np.asarray(g, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return bool(outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3])


def _frame_from_component(coords: list[tuple[int, int]], color: int) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(mask, ring):
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _find_frame_chain(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for comp in _components(g == int(color)):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                candidates.append(frame)
    assert candidates, "expected at least one rectangular frame"

    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for frame in candidates:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        groups.setdefault((int(r0 + r1), int(c0 + c1)), []).append(frame)

    best_key: tuple[int, int, int] | None = None
    best_group: list[dict[str, object]] | None = None
    for group in groups.values():
        ordered = sorted(group, key=lambda item: _bbox_area(item["bbox"]))  # type: ignore[index]
        if not all(_contains(ordered[i + 1]["bbox"], ordered[i]["bbox"]) for i in range(len(ordered) - 1)):  # type: ignore[index]
            continue
        key = (
            len(ordered),
            sum(_bbox_area(item["bbox"]) for item in ordered),  # type: ignore[index]
            _bbox_area(ordered[-1]["bbox"]),  # type: ignore[index]
        )
        if best_key is None or key > best_key:
            best_key = key
            best_group = ordered

    assert best_group is not None, "expected one nested frame chain"
    return best_group


def _extract_pattern(g: np.ndarray, bg: int, inner_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = inner_bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    mask = interior != int(bg)
    assert bool(mask.any()), "innermost frame must contain a pattern"
    pr0, pc0, pr1, pc1 = _bbox_from_mask(mask)
    return interior[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _center_start(start: int, span: int, item: int) -> int:
    assert item <= span
    return int(start + (span - item) // 2)


def _gap_slots(
    inner_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = inner_bbox
    or0, oc0, or1, oc1 = outer_bbox
    ph, pw = map(int, shape)

    top_h = int(ir0 - or0 - 1)
    bottom_h = int(or1 - ir1 - 1)
    left_w = int(ic0 - oc0 - 1)
    right_w = int(oc1 - ic1 - 1)
    outer_inner_h = int(or1 - or0 - 1)
    outer_inner_w = int(oc1 - oc0 - 1)

    assert top_h >= ph and bottom_h >= ph and left_w >= pw and right_w >= pw

    top_r = _center_start(or0 + 1, top_h, ph)
    top_c = _center_start(oc0 + 1, outer_inner_w, pw)
    bottom_r = _center_start(ir1 + 1, bottom_h, ph)
    bottom_c = top_c
    left_r = _center_start(or0 + 1, outer_inner_h, ph)
    left_c = _center_start(oc0 + 1, left_w, pw)
    right_r = left_r
    right_c = _center_start(ic1 + 1, right_w, pw)

    return [
        (int(top_r), int(top_c)),
        (int(bottom_r), int(bottom_c)),
        (int(left_r), int(left_c)),
        (int(right_r), int(right_c)),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the innermost pattern outward through a nested frame hierarchy.

    The gap between two adjacent frames is interpreted as four side-interval regions
    (top, bottom, left, right). On each outward step, the current pattern is flipped
    and copied into the centered slot of each side interval.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    frames = _find_frame_chain(g, bg)

    out = g.copy()
    if len(frames) < 2:
        return out

    pattern = _extract_pattern(g, bg, frames[0]["bbox"])  # type: ignore[index]
    current = pattern.copy()

    for depth in range(len(frames) - 1):
        current = np.fliplr(current) if depth % 2 == 0 else np.flipud(current)
        inner_bbox = frames[depth]["bbox"]  # type: ignore[index]
        outer_bbox = frames[depth + 1]["bbox"]  # type: ignore[index]
        for r, c in _gap_slots(inner_bbox, outer_bbox, current.shape):
            h, w = current.shape
            out[r : r + h, c : c + w] = current

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _pattern_has_frame_component(pattern: np.ndarray) -> bool:
    for color in PATTERN_COLORS:
        for comp in _components(pattern == int(color)):
            if _frame_from_component(comp, color) is not None:
                return True
    return False


def _random_pattern(height: int, width: int) -> np.ndarray:
    for _ in range(200):
        pattern = np.random.choice(np.array(PATTERN_COLORS, dtype=int), size=(height, width))
        if len(set(map(int, pattern.ravel()))) != 2:
            continue
        if np.array_equal(pattern, np.fliplr(pattern)):
            continue
        if np.array_equal(pattern, np.flipud(pattern)):
            continue
        if np.array_equal(pattern, np.rot90(pattern, 2)):
            continue
        if _pattern_has_frame_component(pattern):
            continue
        return pattern.astype(int)
    raise RuntimeError("failed to generate a usable asymmetric pattern")


def _choose_margins(struct_len: int) -> tuple[int, int]:
    max_extra = CANVAS_MAX - struct_len
    assert max_extra >= 2
    extra = random.randint(2, max_extra)
    before = extra // 2
    wiggle = min(2, before, extra - before)
    if wiggle:
        before += random.randint(-wiggle, wiggle)
    before = max(1, min(extra - 1, before))
    after = extra - before
    return int(before), int(after)


def _build_candidate() -> np.ndarray | None:
    ph = random.randint(PATTERN_SIZE_MIN, PATTERN_SIZE_MAX)
    pw = random.randint(PATTERN_SIZE_MIN, PATTERN_SIZE_MAX)
    pad_r = random.randint(INNER_PADDING_MIN, INNER_PADDING_MAX)
    pad_c = random.randint(INNER_PADDING_MIN, INNER_PADDING_MAX)
    inner_h = ph + 2 * pad_r
    inner_w = pw + 2 * pad_c

    valid_layers = []
    for layers in range(MIN_LAYERS, MAX_LAYERS + 1):
        struct_h = inner_h + 2 + 2 * (layers - 1) * (ph + 1)
        struct_w = inner_w + 2 + 2 * (layers - 1) * (pw + 1)
        if struct_h <= CANVAS_MAX - 2 and struct_w <= CANVAS_MAX - 2:
            valid_layers.append((layers, struct_h, struct_w))
    if not valid_layers:
        return None

    layers, struct_h, struct_w = random.choice(valid_layers)
    top, bottom = _choose_margins(struct_h)
    left, right = _choose_margins(struct_w)
    grid = np.full((struct_h + top + bottom, struct_w + left + right), BG, dtype=int)

    pattern = _random_pattern(ph, pw)
    frame_colors = random.sample(FRAME_COLOR_POOL, layers)

    inner_r0 = top + (layers - 1) * (ph + 1)
    inner_c0 = left + (layers - 1) * (pw + 1)
    inner_r1 = inner_r0 + inner_h + 1
    inner_c1 = inner_c0 + inner_w + 1

    bboxes = []
    r0, c0, r1, c1 = inner_r0, inner_c0, inner_r1, inner_c1
    for _ in range(layers):
        bboxes.append((int(r0), int(c0), int(r1), int(c1)))
        r0 -= ph + 1
        c0 -= pw + 1
        r1 += ph + 1
        c1 += pw + 1

    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(grid, bbox, color)

    pattern_r0 = inner_r0 + 1 + pad_r
    pattern_c0 = inner_c0 + 1 + pad_c
    grid[pattern_r0 : pattern_r0 + ph, pattern_c0 : pattern_c0 + pw] = pattern

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-frame input")
