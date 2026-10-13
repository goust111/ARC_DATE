from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_DIRS: dict[str, tuple[int, int]] = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
_HEAD_SHAPES: dict[str, set[tuple[int, int]]] = {
    "right": {(0, 0), (1, 0), (1, 1), (2, 0)},
    "left": {(0, 1), (1, 0), (1, 1), (2, 1)},
    "up": {(0, 1), (1, 0), (1, 1), (1, 2)},
    "down": {(0, 0), (0, 1), (0, 2), (1, 1)},
}

_GRID_MIN = 18
_GRID_MAX = 29
_PAIR_COUNT_CHOICES = (2, 3, 4)
_RECT_H_RANGE = (5, 9)
_RECT_W_RANGE = (5, 9)
_SHAFT_RANGE = (2, 4)
_STEP_RANGE = (3, 7)
_GEN_ATTEMPTS = 400
_PLACE_ATTEMPTS = 160
_COLORS = tuple(range(1, 10))


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_list(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _expected_arrow_rel(direction: str, shaft_len: int) -> set[tuple[int, int]]:
    if direction == "right":
        width = shaft_len + 2
        return {(1, c) for c in range(width)} | {(0, width - 2), (2, width - 2)}
    if direction == "left":
        width = shaft_len + 2
        return {(1, c) for c in range(width)} | {(0, 1), (2, 1)}
    if direction == "up":
        height = shaft_len + 2
        return {(r, 1) for r in range(height)} | {(1, 0), (1, 2)}
    height = shaft_len + 2
    return {(r, 1) for r in range(height)} | {(height - 2, 0), (height - 2, 2)}


def _head_shift(direction: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if direction == "right":
        return 0, w - 2
    if direction == "left":
        return 0, 0
    if direction == "up":
        return 0, 0
    return h - 2, 0


def _detect_arrow(comp: dict[str, object]) -> dict[str, object] | None:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
    coords = comp["coords"]  # type: ignore[assignment]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}

    for direction in ("right", "left", "up", "down"):
        if direction in {"right", "left"}:
            if h != 3 or not (4 <= w <= 6):
                continue
            shaft_len = w - 2
        else:
            if w != 3 or not (4 <= h <= 6):
                continue
            shaft_len = h - 2
        if rel != _expected_arrow_rel(direction, shaft_len):
            continue

        sr, sc = _head_shift(direction, (h, w))
        head = {(int(r0 + sr + rr), int(c0 + sc + cc)) for rr, cc in _HEAD_SHAPES[direction]}
        return {
            "color": int(comp["color"]),
            "direction": direction,
            "coords": {(int(r), int(c)) for r, c in coords},
            "head": head,
        }
    return None


def _notch_patterns(height: int, width: int) -> list[tuple[str, set[tuple[int, int]]]]:
    patterns: list[tuple[str, set[tuple[int, int]]]] = []

    for top in range(height - 2):
        patterns.append(
            (
                "right",
                {(top, 0), (top + 1, 0), (top + 1, 1), (top + 2, 0)},
            )
        )
        patterns.append(
            (
                "left",
                {(top, width - 1), (top + 1, width - 2), (top + 1, width - 1), (top + 2, width - 1)},
            )
        )
    for left in range(width - 2):
        patterns.append(
            (
                "down",
                {(0, left), (0, left + 1), (0, left + 2), (1, left + 1)},
            )
        )
        patterns.append(
            (
                "up",
                {(height - 2, left + 1), (height - 1, left), (height - 1, left + 1), (height - 1, left + 2)},
            )
        )
    return patterns


def _detect_target(comp: dict[str, object]) -> dict[str, object] | None:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
    coords = comp["coords"]  # type: ignore[assignment]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    present = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = {(rr, cc) for rr in range(h) for cc in range(w)} - present
    if len(missing) != 4:
        return None

    matches = [(direction, pattern) for direction, pattern in _notch_patterns(h, w) if pattern == missing]
    if len(matches) != 1:
        return None

    direction, pattern = matches[0]
    notch = {(int(r0 + rr), int(c0 + cc)) for rr, cc in pattern}
    return {
        "color": int(comp["color"]),
        "direction": direction,
        "notch": notch,
    }


def _translation_steps(
    head: set[tuple[int, int]],
    notch: set[tuple[int, int]],
    direction: str,
) -> int | None:
    dr, dc = _DIRS[direction]
    head_proj = [dr * r + dc * c for r, c in head]
    notch_proj = [dr * r + dc * c for r, c in notch]
    steps = int(min(notch_proj) - min(head_proj))
    if steps <= 0:
        return None
    translated = {(r + steps * dr, c + steps * dc) for r, c in head}
    return steps if translated == notch else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each arrowhead into the matching same-color rectangle notch, erasing the original arrow shaft.

    Input structure:
    - The background is black.
    - Each foreground color appears in one small arrow object and one larger same-color rectangle-like block.
    - Every arrow is a 4-connected monochrome component made of a triangular arrowhead plus a straight shaft, while every target block is a solid rectangle with exactly one triangular notch cut out of one edge.

    Rule:
    - Detect the direction of each arrow from its arrowhead orientation and discard the shaft.
    - Find the same-color notched rectangle whose missing triangle is exactly the arrowhead translated forward along that direction.
    - Fill that notch with the arrowhead shape and clear the arrow's original pixels to black.

    Key insight:
    - The arrow carries two pieces of information at once: its orientation gives the translation direction, and its head shape is the exact patch that repairs the matching rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    out = g.copy()

    arrows: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    for comp in _component_list(g, bg):
        arrow = _detect_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
            continue
        target = _detect_target(comp)
        if target is not None:
            targets.append(target)

    assert arrows, "expected at least one arrow"
    used_targets: set[int] = set()

    for arrow in arrows:
        candidates: list[tuple[int, int]] = []
        for idx, target in enumerate(targets):
            if idx in used_targets:
                continue
            if int(target["color"]) != int(arrow["color"]) or target["direction"] != arrow["direction"]:
                continue
            steps = _translation_steps(
                arrow["head"],  # type: ignore[arg-type]
                target["notch"],  # type: ignore[arg-type]
                str(arrow["direction"]),
            )
            if steps is not None:
                candidates.append((steps, idx))

        assert len(candidates) == 1, "each arrow must have exactly one forward same-color notch"
        _, idx = candidates[0]
        used_targets.add(idx)

        for r, c in arrow["coords"]:  # type: ignore[assignment]
            out[int(r), int(c)] = bg
        for r, c in targets[idx]["notch"]:  # type: ignore[index]
            out[int(r), int(c)] = int(arrow["color"])

    return out


def _make_arrow_patch(color: int, direction: str, shaft_len: int) -> np.ndarray:
    if direction == "right":
        patch = np.zeros((3, shaft_len + 2), dtype=int)
        patch[1, :] = color
        patch[0, shaft_len] = color
        patch[2, shaft_len] = color
        return patch
    if direction == "left":
        patch = np.zeros((3, shaft_len + 2), dtype=int)
        patch[1, :] = color
        patch[0, 1] = color
        patch[2, 1] = color
        return patch
    if direction == "up":
        patch = np.zeros((shaft_len + 2, 3), dtype=int)
        patch[:, 1] = color
        patch[1, 0] = color
        patch[1, 2] = color
        return patch
    patch = np.zeros((shaft_len + 2, 3), dtype=int)
    patch[:, 1] = color
    patch[shaft_len, 0] = color
    patch[shaft_len, 2] = color
    return patch


def _notch_cells(direction: str, rect_h: int, rect_w: int, offset: int) -> set[tuple[int, int]]:
    if direction == "right":
        return {(offset, 0), (offset + 1, 0), (offset + 1, 1), (offset + 2, 0)}
    if direction == "left":
        return {(offset, rect_w - 1), (offset + 1, rect_w - 2), (offset + 1, rect_w - 1), (offset + 2, rect_w - 1)}
    if direction == "up":
        return {(rect_h - 2, offset + 1), (rect_h - 1, offset), (rect_h - 1, offset + 1), (rect_h - 1, offset + 2)}
    return {(0, offset), (0, offset + 1), (0, offset + 2), (1, offset + 1)}


def _make_pair_patches(
    color: int,
    direction: str,
    rect_h: int,
    rect_w: int,
    shaft_len: int,
    steps: int,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrow = _make_arrow_patch(color, direction, shaft_len)
    target_in = np.full((rect_h, rect_w), color, dtype=int)
    for rr, cc in _notch_cells(direction, rect_h, rect_w, offset):
        target_in[rr, cc] = _BG
    target_out = np.full((rect_h, rect_w), color, dtype=int)

    if direction == "right":
        scene_h = rect_h
        scene_w = arrow.shape[1] - 2 + steps + rect_w
        arrow_r, arrow_c = offset, 0
        target_r, target_c = 0, arrow.shape[1] - 2 + steps
    elif direction == "left":
        scene_h = rect_h
        scene_w = rect_w - 2 + steps + arrow.shape[1]
        target_r, target_c = 0, 0
        arrow_r, arrow_c = offset, rect_w - 2 + steps
    elif direction == "up":
        scene_h = rect_h - 2 + steps + arrow.shape[0]
        scene_w = rect_w
        target_r, target_c = 0, 0
        arrow_r, arrow_c = rect_h - 2 + steps, offset
    else:
        scene_h = arrow.shape[0] - 2 + steps + rect_h
        scene_w = rect_w
        arrow_r, arrow_c = 0, offset
        target_r, target_c = arrow.shape[0] - 2 + steps, 0

    inp = np.zeros((scene_h, scene_w), dtype=int)
    out = np.zeros((scene_h, scene_w), dtype=int)
    inp[arrow_r : arrow_r + arrow.shape[0], arrow_c : arrow_c + arrow.shape[1]][arrow != 0] = color
    inp[target_r : target_r + rect_h, target_c : target_c + rect_w][target_in != 0] = target_in[target_in != 0]
    out[target_r : target_r + rect_h, target_c : target_c + rect_w] = target_out
    return inp, out


def _pair_spec(color: int) -> tuple[np.ndarray, np.ndarray]:
    direction = random.choice(tuple(_DIRS))
    rect_h = random.randint(*_RECT_H_RANGE)
    rect_w = random.randint(*_RECT_W_RANGE)
    shaft_len = random.randint(*_SHAFT_RANGE)
    steps = random.randint(*_STEP_RANGE)
    if direction in {"right", "left"}:
        offset = random.randint(1, rect_h - 4)
    else:
        offset = random.randint(1, rect_w - 4)
    return _make_pair_patches(color, direction, rect_h, rect_w, shaft_len, steps, offset)


def _candidate_positions(blocked: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = patch.shape
    coords = np.argwhere(patch != 0)
    positions: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if any(blocked[r0 + int(rr), c0 + int(cc)] for rr, cc in coords):
                continue
            positions.append((r0, c0))
    return positions


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for rr, cc in np.argwhere(patch != 0):
        gr = r0 + int(rr)
        gc = c0 + int(cc)
        for nr in range(max(0, gr - 1), min(h, gr + 2)):
            for nc in range(max(0, gc - 1), min(w, gc + 2)):
                blocked[nr, nc] = True


def _build_sample() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(_GRID_MIN, _GRID_MAX)
    grid_w = random.randint(_GRID_MIN, _GRID_MAX)
    pair_count = random.choice(_PAIR_COUNT_CHOICES)
    colors = random.sample(_COLORS, pair_count)
    patches = [_pair_spec(color) for color in colors]
    max_h = max(inp.shape[0] for inp, _ in patches)
    max_w = max(inp.shape[1] for inp, _ in patches)
    if max_h > grid_h or max_w > grid_w:
        return None

    order = list(range(pair_count))
    random.shuffle(order)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        inp_patch, _ = patches[idx]
        positions = _candidate_positions(blocked, inp_patch)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        placements[idx] = (r0, c0)
        _mark_blocked(blocked, inp_patch, r0, c0)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    out = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (inp_patch, out_patch) in enumerate(patches):
        r0, c0 = placements[idx]
        ph, pw = inp_patch.shape
        inp_region = inp[r0 : r0 + ph, c0 : c0 + pw]
        out_region = out[r0 : r0 + ph, c0 : c0 + pw]
        inp_region[inp_patch != 0] = inp_patch[inp_patch != 0]
        out_region[out_patch != 0] = out_patch[out_patch != 0]

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        sample = _build_sample()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid arrow-notch ARC sample")
