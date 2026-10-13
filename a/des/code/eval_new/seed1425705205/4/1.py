from __future__ import annotations

from collections import deque
import random

import numpy as np


_SIDES = ("top", "right", "bottom", "left")
_ROTATE_LEFT = {"top": "left", "left": "bottom", "bottom": "right", "right": "top"}
_FRAME_SIZE_CHOICES = [
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 4),
    (4, 5),
    (5, 5),
]
_MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _rotate_side(side: str, turns_ccw: int) -> str:
    out = side
    for _ in range(int(turns_ccw) % 4):
        out = _ROTATE_LEFT[out]
    return out


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    target[mask] = patch[mask]


def _detect_frame(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    colors = {int(grid[int(r), int(c)]) for r, c in coords}
    if len(colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(local & ~border):
        return None

    missing = border & ~local
    if int(missing.sum()) != 1:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(patch[1:-1, 1:-1] != bg):
        return None

    mr, mc = map(int, np.argwhere(missing)[0])
    if (mr, mc) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        return None
    if mr == 0:
        gap_side = "top"
    elif mr == h - 1:
        gap_side = "bottom"
    elif mc == 0:
        gap_side = "left"
    else:
        gap_side = "right"

    return {
        "kind": "frame",
        "bbox": (r0, c0, r1, c1),
        "color": int(next(iter(colors))),
        "gap_side": gap_side,
        "inner_shape": (h - 2, w - 2),
        "coords": coords,
    }


def _detect_pattern(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    non_bg = patch != bg
    h, w = patch.shape
    candidates: list[tuple[str, np.ndarray, int]] = []

    if h >= 2 and np.count_nonzero(non_bg[0, :]) == 1 and np.all(non_bg[1:, :]):
        candidates.append(("top", patch[1:, :].copy(), int(np.flatnonzero(non_bg[0, :])[0])))
    if h >= 2 and np.count_nonzero(non_bg[-1, :]) == 1 and np.all(non_bg[:-1, :]):
        candidates.append(("bottom", patch[:-1, :].copy(), int(np.flatnonzero(non_bg[-1, :])[0])))
    if w >= 2 and np.count_nonzero(non_bg[:, 0]) == 1 and np.all(non_bg[:, 1:]):
        candidates.append(("left", patch[:, 1:].copy(), int(np.flatnonzero(non_bg[:, 0])[0])))
    if w >= 2 and np.count_nonzero(non_bg[:, -1]) == 1 and np.all(non_bg[:, :-1]):
        candidates.append(("right", patch[:, :-1].copy(), int(np.flatnonzero(non_bg[:, -1])[0])))

    if len(candidates) != 1:
        return None

    marker_side, payload, marker_idx = candidates[0]
    if payload.shape[0] < 1 or payload.shape[1] < 1:
        return None

    if marker_side == "top":
        marker_color = int(patch[0, marker_idx])
    elif marker_side == "bottom":
        marker_color = int(patch[-1, marker_idx])
    elif marker_side == "left":
        marker_color = int(patch[marker_idx, 0])
    else:
        marker_color = int(patch[marker_idx, -1])

    if np.any(payload == bg):
        return None

    return {
        "kind": "pattern",
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "payload": payload,
        "marker_side": marker_side,
        "marker_color": marker_color,
    }


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _most_common_color(grid)
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in _components(grid != bg):
        frame = _detect_frame(grid, comp, bg)
        if frame is not None:
            frames.append(frame)
            continue
        pattern = _detect_pattern(grid, comp, bg)
        if pattern is not None:
            patterns.append(pattern)
            continue
        raise AssertionError("unexpected foreground object")

    frames.sort(key=lambda obj: obj["bbox"])
    patterns.sort(key=lambda obj: obj["bbox"])
    assert frames and patterns and len(frames) == len(patterns)
    return bg, frames, patterns


def _assign_patterns(frames: list[dict], patterns: list[dict]) -> dict[int, tuple[int, int]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for fi, frame in enumerate(frames):
        opts: list[tuple[int, int]] = []
        for pi, pattern in enumerate(patterns):
            if int(pattern["marker_color"]) != int(frame["color"]):
                continue
            for turns in range(4):
                if _rotate_side(pattern["marker_side"], turns) != frame["gap_side"]:
                    continue
                rotated = np.rot90(pattern["payload"], turns)
                if rotated.shape == frame["inner_shape"]:
                    opts.append((pi, turns))
        assert opts, "every frame must have at least one compatible pattern"
        candidates[fi] = opts

    order = sorted(range(len(frames)), key=lambda idx: (len(candidates[idx]), frames[idx]["bbox"]))
    used: set[int] = set()
    assignment: dict[int, tuple[int, int]] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        fi = order[pos]
        for pi, turns in candidates[fi]:
            if pi in used:
                continue
            used.add(pi)
            assignment[fi] = (pi, turns)
            if dfs(pos + 1):
                return True
            used.remove(pi)
            del assignment[fi]
        return False

    assert dfs(0), "pattern-frame matching must be solvable"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each detached color block to the frame it fits, rotate it by the gap direction, and fill the frame interior.

    Input structure:
    - bg is the most common color.
    - Several monochrome hollow rectangular frames appear on the grid; each has a one-cell gap on exactly one side and a background-colored interior.
    - Separate compact rectangular color patterns also appear elsewhere; each pattern is fully filled and has one extra marker cell attached on one side.
    - The marker cell has the same color as the border of the frame that pattern belongs to.

    Rule:
    - Identify each frame by its one-pixel-thick rectangular border with a single missing border cell, and identify each detached pattern by removing its single attached marker cell.
    - Pair patterns with frames by exact interior size, allowing 90-degree rotations when the pattern rectangle is transposed.
    - Rotate each matched pattern until its marker side aligns with the side of the frame gap, then copy the rotated pattern into the frame interior.
    - Remove every detached pattern and marker from its original location; keep the frame borders and the frame gaps unchanged.

    Key insight:
    - The task requires two linked inferences: size decides which pattern belongs to which frame, while the frame gap decides how that pattern must be rotated before insertion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, frames, patterns = _parse_objects(g)
    assignment = _assign_patterns(frames, patterns)

    out = g.copy()
    for pattern in patterns:
        for r, c in pattern["coords"]:
            out[int(r), int(c)] = bg

    for fi, frame in enumerate(frames):
        pi, turns = assignment[fi]
        payload = np.rot90(patterns[pi]["payload"], turns)
        r0, c0, r1, c1 = frame["bbox"]
        assert payload.shape == (r1 - r0 - 1, c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = payload

    return out


def _make_payload(h: int, w: int, colors: list[int]) -> np.ndarray:
    for _ in range(120):
        palette_size = random.randint(2, min(4, len(colors)))
        palette = random.sample(colors, palette_size)
        arr = np.array([[random.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, arr.ravel()))) < 2:
            continue
        if np.array_equal(arr, np.rot90(arr, 2)):
            continue
        if h == w and any(np.array_equal(arr, np.rot90(arr, k)) for k in (1, 3)):
            continue
        return arr
    raise RuntimeError("failed to build asymmetric payload")


def _make_frame_patch(inner_h: int, inner_w: int, color: int, gap_side: str, bg: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if gap_side == "top":
        patch[0, random.randint(1, inner_w)] = bg
    elif gap_side == "bottom":
        patch[-1, random.randint(1, inner_w)] = bg
    elif gap_side == "left":
        patch[random.randint(1, inner_h), 0] = bg
    else:
        patch[random.randint(1, inner_h), -1] = bg
    return patch


def _make_pattern_patch(payload: np.ndarray, marker_side: str, marker_color: int, bg: int) -> np.ndarray:
    h, w = payload.shape
    if marker_side == "top":
        patch = np.full((h + 1, w), bg, dtype=int)
        patch[1:, :] = payload
        patch[0, random.randrange(w)] = marker_color
        return patch
    if marker_side == "bottom":
        patch = np.full((h + 1, w), bg, dtype=int)
        patch[:-1, :] = payload
        patch[-1, random.randrange(w)] = marker_color
        return patch
    if marker_side == "left":
        patch = np.full((h, w + 1), bg, dtype=int)
        patch[:, 1:] = payload
        patch[random.randrange(h), 0] = marker_color
        return patch
    patch = np.full((h, w + 1), bg, dtype=int)
    patch[:, :-1] = payload
    patch[random.randrange(h), -1] = marker_color
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    oh, ow = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(oh, r0 + h + pad)
    cc1 = min(ow, c0 + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    oh, ow = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(oh, r0 + h + pad)
    cc1 = min(ow, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_patch(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    oh, ow = occupied.shape
    spots = [(r, c) for r in range(oh - h + 1) for c in range(ow - w + 1)]
    random.shuffle(spots)
    for r0, c0 in spots:
        if _can_place(occupied, r0, c0, h, w):
            _mark_occupied(occupied, r0, c0, h, w)
            return r0, c0
    return None


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    bg = 0
    n_objects = random.randint(2, 4)

    all_sizes = _FRAME_SIZE_CHOICES.copy()
    first_size = random.choice([size for size in all_sizes if size[0] != size[1]])
    chosen_sizes = [first_size]
    remaining = [size for size in all_sizes if tuple(sorted(size)) != tuple(sorted(first_size))]
    while len(chosen_sizes) < n_objects:
        choice = random.choice(remaining)
        chosen_sizes.append(choice)
        remaining = [size for size in remaining if tuple(sorted(size)) != tuple(sorted(choice))]

    frame_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    payload_colors = [c for c in range(1, 10) if c != frame_color]

    specs: list[dict] = []
    for base_h, base_w in chosen_sizes:
        if base_h != base_w and random.random() < 0.5:
            inner_h, inner_w = base_w, base_h
        else:
            inner_h, inner_w = base_h, base_w
        marker_side = random.choice(_SIDES)
        turns = random.randrange(4)
        gap_side = _rotate_side(marker_side, turns)
        payload_h, payload_w = (inner_h, inner_w) if turns % 2 == 0 else (inner_w, inner_h)
        payload = _make_payload(payload_h, payload_w, payload_colors)
        frame_patch = _make_frame_patch(inner_h, inner_w, frame_color, gap_side, bg)
        pattern_patch = _make_pattern_patch(payload, marker_side, frame_color, bg)
        specs.append(
            {
                "inner_shape": (inner_h, inner_w),
                "gap_side": gap_side,
                "turns": turns,
                "payload": payload,
                "frame_patch": frame_patch,
                "pattern_patch": pattern_patch,
            }
        )

    max_h = max(max(spec["frame_patch"].shape[0], spec["pattern_patch"].shape[0]) for spec in specs)
    max_w = max(max(spec["frame_patch"].shape[1], spec["pattern_patch"].shape[1]) for spec in specs)
    total_bbox_area = sum(int(np.prod(spec["frame_patch"].shape) + np.prod(spec["pattern_patch"].shape)) for spec in specs)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_bbox_area * 2.0))))
    h = random.randint(max(18, min_side), 30)
    w = random.randint(max(18, min_side), 30)

    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int, str, int]] = []
    objects: list[tuple[int, str, np.ndarray]] = []
    for idx, spec in enumerate(specs):
        objects.append((idx, "frame", spec["frame_patch"]))
        objects.append((idx, "pattern", spec["pattern_patch"]))
    random.shuffle(objects)
    objects.sort(key=lambda item: int(np.prod(item[2].shape)), reverse=True)

    for idx, kind, patch in objects:
        loc = _place_patch(occupied, patch.shape[0], patch.shape[1])
        if loc is None:
            raise RuntimeError("placement failed")
        placements.append((idx, kind, loc[0], loc[1]))

    inp = np.full((h, w), bg, dtype=int)
    expected = inp.copy()
    frame_positions: dict[int, tuple[int, int]] = {}
    pattern_positions: dict[int, tuple[int, int]] = {}

    for idx, kind, r0, c0 in placements:
        patch = specs[idx][f"{kind}_patch"]
        _paste_non_bg(inp, patch, r0, c0, bg)
        if kind == "frame":
            frame_positions[idx] = (r0, c0)
            _paste_non_bg(expected, patch, r0, c0, bg)
        else:
            pattern_positions[idx] = (r0, c0)

    for idx, spec in enumerate(specs):
        fr0, fc0 = frame_positions[idx]
        inner_h, inner_w = spec["inner_shape"]
        rotated = np.rot90(spec["payload"], spec["turns"])
        assert rotated.shape == (inner_h, inner_w)
        expected[fr0 + 1 : fr0 + 1 + inner_h, fc0 + 1 : fc0 + 1 + inner_w] = rotated
        assert idx in pattern_positions

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected = _build_example()
        except RuntimeError:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
