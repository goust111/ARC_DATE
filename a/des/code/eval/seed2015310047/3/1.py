from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BLACK = 0
FRAME_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "flip_d",
    "flip_a",
)
PRESERVING_TRANSFORMS = ("id", "rot180", "flip_h", "flip_v")
RULE_CHOICES = [
    rule
    for rule in itertools.permutations(TRANSFORMS[1:], 3)
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "flip_h":
        return np.fliplr(arr)
    if name == "flip_v":
        return np.flipud(arr)
    if name == "flip_d":
        return arr.T.copy()
    if name == "flip_a":
        return np.rot90(arr.T, 2)
    raise ValueError(f"unknown transform: {name}")


def _inverse_transform(name: str) -> str:
    if name == "rot90":
        return "rot270"
    if name == "rot270":
        return "rot90"
    return name


def _allowed_transforms(shape: tuple[int, int]) -> tuple[str, ...]:
    return TRANSFORMS if shape[0] == shape[1] else PRESERVING_TRANSFORMS


def _quadrant_slices(shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    h, w = shape
    cr = h // 2
    cc = w // 2
    return [
        (slice(1, cr), slice(1, cc)),
        (slice(1, cr), slice(cc + 1, w - 1)),
        (slice(cr + 1, h - 1), slice(1, cc)),
        (slice(cr + 1, h - 1), slice(cc + 1, w - 1)),
    ]


def _extract_quadrants(frame: np.ndarray) -> list[dict[str, object]]:
    quads: list[dict[str, object]] = []
    for rows, cols in _quadrant_slices(frame.shape):
        sub = frame[rows, cols]
        mask = sub != BLACK
        colors = [int(c) for c in np.unique(sub[mask]) if int(c) != BLACK]
        assert len(colors) <= 1, "each quadrant sprite should use at most one color"
        quads.append(
            {
                "rows": rows,
                "cols": cols,
                "mask": mask,
                "color": colors[0] if colors else BLACK,
            }
        )
    return quads


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    for color in np.unique(g):
        if int(color) == BLACK:
            continue
        mask = g == int(color)
        for cells in _components(mask):
            r0, c0, r1, c1 = _bbox(cells)
            local = g[r0 : r1 + 1, c0 : c1 + 1]
            local_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
            h, w = local.shape
            if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
                continue
            cr = h // 2
            cc = w // 2
            expected = np.zeros((h, w), dtype=bool)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            expected[cr, :] = True
            expected[:, cc] = True
            if np.array_equal(local_mask, expected):
                frames.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "color": int(color),
                        "grid": local.copy(),
                        "quads": _extract_quadrants(local),
                    }
                )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames, "no valid quadrant frame found"

    examples: list[dict[str, object]] = []
    target: dict[str, object] | None = None
    for frame in frames:
        filled = [i for i, q in enumerate(frame["quads"]) if bool(np.any(q["mask"]))]
        if len(filled) == 4:
            examples.append(frame)
        elif len(filled) == 1:
            assert target is None, "expected a unique target frame"
            target = frame
    assert examples, "no fully filled example frame found"
    assert target is not None, "no incomplete target frame found"

    qshape = target["quads"][0]["mask"].shape
    allowed = _allowed_transforms(qshape)
    inferred: dict[int, set[str]] = {1: set(allowed), 2: set(allowed), 3: set(allowed)}
    for frame in examples:
        base = frame["quads"][0]["mask"]
        assert bool(np.any(base)), "example TL quadrant must be non-empty"
        for idx in (1, 2, 3):
            goal = frame["quads"][idx]["mask"]
            cands = {name for name in allowed if np.array_equal(_apply_transform(base, name), goal)}
            inferred[idx] &= cands
            assert inferred[idx], "examples do not support a consistent transform rule"

    rule = {0: "id"}
    for idx in (1, 2, 3):
        assert len(inferred[idx]) == 1, "transform rule is ambiguous"
        rule[idx] = next(iter(inferred[idx]))

    target_quads = target["quads"]
    seed_idx = next(i for i, q in enumerate(target_quads) if bool(np.any(q["mask"])))
    seed_mask = target_quads[seed_idx]["mask"]
    seed_color = int(target_quads[seed_idx]["color"])
    base_mask = _apply_transform(seed_mask, _inverse_transform(rule[seed_idx]))

    out = target["grid"].copy()
    for q in target_quads:
        out[q["rows"], q["cols"]] = BLACK
    for idx, q in enumerate(target_quads):
        filled = _apply_transform(base_mask, rule[idx])
        sub = out[q["rows"], q["cols"]]
        sub[filled] = seed_color
    return out


def _make_frame(side: int) -> np.ndarray:
    size = 2 * side + 3
    frame = np.zeros((size, size), dtype=int)
    center = side + 1
    frame[0, :] = FRAME_COLOR
    frame[-1, :] = FRAME_COLOR
    frame[:, 0] = FRAME_COLOR
    frame[:, -1] = FRAME_COLOR
    frame[center, :] = FRAME_COLOR
    frame[:, center] = FRAME_COLOR
    return frame


def _unique_transform_signature(mask: np.ndarray) -> bool:
    seen = set()
    for name in TRANSFORMS:
        arr = _apply_transform(mask, name)
        seen.add(arr.tobytes())
    return len(seen) == len(TRANSFORMS)


def _random_connected_mask(rng: np.random.Generator, side: int) -> np.ndarray | None:
    total = side * side
    target = int(rng.integers(max(4, side), max(5, total - 1)))
    start = (int(rng.integers(side)), int(rng.integers(side)))
    cells = {start}
    frontier = {start}
    while len(cells) < target:
        candidates = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    candidates.add((nr, nc))
        if not candidates:
            return None
        pick = list(candidates)[int(rng.integers(len(candidates)))]
        cells.add(pick)
        frontier.add(pick)
        if rng.random() < 0.3:
            frontier = {pick}
    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if np.all(mask) or np.sum(mask) < 4:
        return None
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    if not _unique_transform_signature(mask):
        return None
    return mask


def _sample_mask(rng: np.random.Generator, side: int) -> np.ndarray:
    for _ in range(200):
        mask = _random_connected_mask(rng, side)
        if mask is not None:
            return mask
    raise RuntimeError("failed to sample a non-degenerate sprite mask")


def _build_example_frame(
    side: int,
    full_rule: tuple[str, str, str, str],
    base_mask: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    frame = _make_frame(side)
    for idx, (rows, cols) in enumerate(_quadrant_slices(frame.shape)):
        sub = frame[rows, cols]
        sub[_apply_transform(base_mask, full_rule[idx])] = int(colors[idx])
    return frame


def _build_target_frame(
    side: int,
    full_rule: tuple[str, str, str, str],
    base_mask: np.ndarray,
    seed_idx: int,
    seed_color: int,
) -> np.ndarray:
    frame = _make_frame(side)
    rows, cols = _quadrant_slices(frame.shape)[seed_idx]
    sub = frame[rows, cols]
    sub[_apply_transform(base_mask, full_rule[seed_idx])] = int(seed_color)
    return frame


def _place_frames(rng: np.random.Generator, frames: list[np.ndarray]) -> np.ndarray:
    shuffled = list(frames)
    rng.shuffle(shuffled)
    fh, fw = shuffled[0].shape
    n = len(shuffled)
    rows = int(rng.integers(1, min(3, n) + 1))
    cols = int(np.ceil(n / rows))
    slot_h = fh + int(rng.integers(2, 6))
    slot_w = fw + int(rng.integers(2, 6))
    h = rows * slot_h + int(rng.integers(1, 5))
    w = cols * slot_w + int(rng.integers(1, 5))
    grid = np.zeros((h, w), dtype=int)

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = rng.choice(len(slots), size=n, replace=False)
    for frame, slot_idx in zip(shuffled, chosen):
        sr, sc = slots[int(slot_idx)]
        top0 = sr * slot_h
        left0 = sc * slot_w
        max_dr = slot_h - fh
        max_dc = slot_w - fw
        top = top0 + int(rng.integers(max_dr + 1))
        left = left0 + int(rng.integers(max_dc + 1))
        grid[top : top + fh, left : left + fw] = frame
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    side = int(rng.integers(4, 7))
    rule = RULE_CHOICES[int(rng.integers(len(RULE_CHOICES)))]
    full_rule = ("id",) + rule

    example_count = int(rng.integers(2, 5))
    frames: list[np.ndarray] = []
    for _ in range(example_count):
        base_mask = _sample_mask(rng, side)
        colors = rng.choice(SPRITE_COLORS, size=4, replace=False)
        frames.append(_build_example_frame(side, full_rule, base_mask, colors))

    target_mask = _sample_mask(rng, side)
    seed_idx = int(rng.integers(4))
    seed_color = int(rng.choice(SPRITE_COLORS))
    frames.append(_build_target_frame(side, full_rule, target_mask, seed_idx, seed_color))

    grid = _place_frames(rng, frames)
    solved = main(grid)
    assert solved.shape == frames[0].shape
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if isinstance(grid, np.ndarray):
            return grid
    raise RuntimeError("failed to generate a valid input")
