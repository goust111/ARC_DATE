from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6

FRAME_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA)
SIDE_TO_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}
SIDES = tuple(SIDE_TO_TURNS)

LAYER_RANGE = (2, 8)
PAD_RANGE = (2, 5)
INNER_MARGIN = 1
CANVAS_MARGIN_RANGE = (1, 5)
MAX_STACK_SIZE = 22
MAX_GENERATION_ATTEMPTS = 400


PATTERN_LIBRARY = [
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=bool,
    ),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_frame(coords: np.ndarray) -> dict:
    coords = np.asarray(coords, dtype=int)
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 3 and w >= 3, "frame bbox must be at least 3x3"

    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert not np.any(patch & ~border), "frame pixels must lie on the border"
    missing = np.argwhere(border & ~patch)
    assert len(missing) == 1, "each frame must have exactly one gap"
    mr, mc = map(int, missing[0])
    if mr == 0 and 0 < mc < w - 1:
        side = "up"
    elif mc == w - 1 and 0 < mr < h - 1:
        side = "right"
    elif mr == h - 1 and 0 < mc < w - 1:
        side = "down"
    elif mc == 0 and 0 < mr < h - 1:
        side = "left"
    else:
        raise AssertionError("gap must be a non-corner border cell")

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": coords,
        "side": side,
        "bbox_area": h * w,
    }


def _strictly_contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 + 1 < ir0 and oc0 + 1 < ic0 and ir1 < or1 - 1 and ic1 < oc1 - 1


def _extract_frames(g: np.ndarray) -> list[dict]:
    frames = [_parse_frame(coords) for coords in _components(g == GREEN)]
    assert frames, "at least one green frame is required"
    frames.sort(key=lambda frame: (-frame["bbox_area"], frame["bbox"][0], frame["bbox"][1]))
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            assert _strictly_contains(frames[i], frames[j]), "frames must form one strict nesting chain"
    return frames


def _rotate_coords(coords: np.ndarray, center2: tuple[int, int], turns: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    turns %= 4
    if turns == 0:
        return coords.copy()

    cr2, cc2 = map(int, center2)
    rotated: list[tuple[int, int]] = []
    for r, c in coords:
        dr2 = 2 * int(r) - cr2
        dc2 = 2 * int(c) - cc2
        if turns == 1:
            nr2, nc2 = dc2, -dr2
        elif turns == 2:
            nr2, nc2 = -dr2, -dc2
        else:
            nr2, nc2 = -dc2, dr2
        rr2 = cr2 + nr2
        cc22 = cc2 + nc2
        assert rr2 % 2 == 0 and cc22 % 2 == 0, "rotation center must map pixels back to the lattice"
        rotated.append((rr2 // 2, cc22 // 2))
    return np.array(rotated, dtype=int)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[gap[0], gap[1]] = BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested gap-frames by depth and rotate the inner yellow pattern by the
    cumulative gap directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, GREEN, YELLOW}), "unexpected input colors"

    frames = _extract_frames(g)
    yellow = np.argwhere(g == YELLOW)
    assert len(yellow) > 0, "expected a yellow marker pattern"

    inner = frames[-1]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    inside_inner = (
        (yellow[:, 0] > ir0)
        & (yellow[:, 0] < ir1)
        & (yellow[:, 1] > ic0)
        & (yellow[:, 1] < ic1)
    )
    assert bool(np.all(inside_inner)), "yellow pattern must lie inside the innermost frame"

    center2 = (ir0 + ir1, ic0 + ic1)
    assert center2[0] % 2 == center2[1] % 2, "inner frame center must support quarter-turn rotation"
    turns = sum(SIDE_TO_TURNS[frame["side"]] for frame in frames) % 4

    out = np.full_like(g, BLACK)
    for depth, frame in enumerate(frames):
        out[frame["coords"][:, 0], frame["coords"][:, 1]] = FRAME_COLORS[depth % len(FRAME_COLORS)]

    rotated = _rotate_coords(yellow, center2, turns)
    assert bool(
        np.all(
            (rotated[:, 0] > ir0)
            & (rotated[:, 0] < ir1)
            & (rotated[:, 1] > ic0)
            & (rotated[:, 1] < ic1)
        )
    ), "rotated pattern must remain inside the innermost frame"
    out[rotated[:, 0], rotated[:, 1]] = YELLOW
    return out


def _canonical_gap(bbox: tuple[int, int, int, int], side: str, index: int) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    if side == "up":
        return (r0, c0 + index)
    if side == "right":
        return (r0 + index, c1)
    if side == "down":
        return (r1, c0 + index)
    if side == "left":
        return (r0 + index, c0)
    raise ValueError(side)


def _frame_coords_from_bbox(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    coords = {(r0, c) for c in range(c0, c1 + 1)}
    coords |= {(r1, c) for c in range(c0, c1 + 1)}
    coords |= {(r, c0) for r in range(r0, r1 + 1)}
    coords |= {(r, c1) for r in range(r0, r1 + 1)}
    coords.discard((int(gap[0]), int(gap[1])))
    return np.array(sorted(coords), dtype=int)


def _pattern_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for shape in PATTERN_LIBRARY:
        turns = []
        for k in range(4):
            turns.append(np.rot90(shape, -k))
        if len({variant.tobytes() for variant in turns}) == 4:
            variants.append(shape)
    assert variants, "pattern library must contain asymmetric shapes"
    return variants


PATTERNS = _pattern_variants()


def _odd_choices(lo: int, hi: int) -> list[int]:
    lo = int(lo)
    hi = int(hi)
    return [value for value in range(lo, hi + 1) if value % 2 == 1]


def _expected_output(
    shape: tuple[int, int],
    frames: list[dict],
    pattern_coords: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for depth, frame in enumerate(frames):
        gap = frame["gap"]
        _draw_frame(out, frame["bbox"], FRAME_COLORS[depth % len(FRAME_COLORS)], gap)

    inner = frames[-1]
    turns = sum(SIDE_TO_TURNS[frame["side"]] for frame in frames) % 4
    center2 = (inner["bbox"][0] + inner["bbox"][2], inner["bbox"][1] + inner["bbox"][3])
    rotated = _rotate_coords(pattern_coords, center2, turns)
    out[rotated[:, 0], rotated[:, 1]] = YELLOW
    return out


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    layers = random.randint(*LAYER_RANGE)
    pad_specs = [
        {
            "top": random.randint(*PAD_RANGE),
            "bottom": random.randint(*PAD_RANGE),
            "left": random.randint(*PAD_RANGE),
            "right": random.randint(*PAD_RANGE),
        }
        for _ in range(layers - 1)
    ]
    extra_h = sum(spec["top"] + spec["bottom"] for spec in pad_specs)
    extra_w = sum(spec["left"] + spec["right"] for spec in pad_specs)

    base_shape = random.choice(PATTERNS)
    patch_h, patch_w = map(int, base_shape.shape)
    min_inner_h = patch_h + 2 * (INNER_MARGIN + 1)
    min_inner_w = patch_w + 2 * (INNER_MARGIN + 1)
    inner_h_choices = _odd_choices(min_inner_h, MAX_STACK_SIZE - extra_h)
    inner_w_choices = _odd_choices(min_inner_w, MAX_STACK_SIZE - extra_w)
    if not inner_h_choices or not inner_w_choices:
        return None

    inner_h = random.choice(inner_h_choices)
    inner_w = random.choice(inner_w_choices)
    bboxes_inner_to_outer = [(0, 0, inner_h - 1, inner_w - 1)]
    for spec in pad_specs:
        r0, c0, r1, c1 = bboxes_inner_to_outer[-1]
        bboxes_inner_to_outer.append(
            (
                r0 - spec["top"],
                c0 - spec["left"],
                r1 + spec["bottom"],
                c1 + spec["right"],
            )
        )

    bboxes = list(reversed(bboxes_inner_to_outer))
    min_r = min(bbox[0] for bbox in bboxes)
    min_c = min(bbox[1] for bbox in bboxes)
    max_r = max(bbox[2] for bbox in bboxes)
    max_c = max(bbox[3] for bbox in bboxes)
    stack_h = int(max_r - min_r + 1)
    stack_w = int(max_c - min_c + 1)

    margin_top = random.randint(*CANVAS_MARGIN_RANGE)
    margin_bottom = random.randint(*CANVAS_MARGIN_RANGE)
    margin_left = random.randint(*CANVAS_MARGIN_RANGE)
    margin_right = random.randint(*CANVAS_MARGIN_RANGE)
    shift_r = margin_top - min_r
    shift_c = margin_left - min_c
    h = stack_h + margin_top + margin_bottom
    w = stack_w + margin_left + margin_right

    frames: list[dict] = []
    for bbox in bboxes:
        r0, c0, r1, c1 = bbox
        shifted = (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        side = random.choice(SIDES)
        if side in {"up", "down"}:
            gap_index = random.randint(1, shifted[3] - shifted[1] - 1)
        else:
            gap_index = random.randint(1, shifted[2] - shifted[0] - 1)
        gap = _canonical_gap(shifted, side, gap_index)
        frames.append(
            {
                "bbox": shifted,
                "side": side,
                "gap": gap,
                "coords": _frame_coords_from_bbox(shifted, gap),
            }
        )

    inp = np.full((h, w), BLACK, dtype=int)
    for frame in frames:
        _draw_frame(inp, frame["bbox"], GREEN, frame["gap"])

    inner = frames[-1]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    center_r = (ir0 + ir1) // 2
    center_c = (ic0 + ic1) // 2
    pattern = np.rot90(base_shape, -random.randint(0, 3))
    top = center_r - patch_h // 2
    left = center_c - patch_w // 2
    pattern_coords = np.argwhere(pattern)
    pattern_coords = pattern_coords + np.array([top, left], dtype=int)

    if not np.all(
        (pattern_coords[:, 0] > ir0 + INNER_MARGIN)
        & (pattern_coords[:, 0] < ir1 - INNER_MARGIN)
        & (pattern_coords[:, 1] > ic0 + INNER_MARGIN)
        & (pattern_coords[:, 1] < ic1 - INNER_MARGIN)
    ):
        return None
    inp[pattern_coords[:, 0], pattern_coords[:, 1]] = YELLOW

    expected = _expected_output(inp.shape, frames, pattern_coords)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid nested-frame rotation input")
