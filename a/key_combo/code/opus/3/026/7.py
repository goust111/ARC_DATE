from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg


BG = 0
NONZERO_COLORS = tuple(range(1, 10))
GRID_SIZE_RANGE = (15, 22)
INNER_SIZE_CHOICES = (3, 4)
MIN_FRAME_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
SPRITE_ATTEMPTS = 80
HIDDEN_CYCLE_WEIGHTS = ((3, 1), (4, 2), (5, 2))


def _is_hollow_square_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    side = int(r1 - r0 + 1)
    if side < 5 or side != int(c1 - c0 + 1):
        return False
    if int(component["area"]) != 4 * (side - 1):
        return False
    coords = {(int(r), int(c)) for r, c in component["coords"]}
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return coords == border


def _find_frames(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    frames = [
        component
        for component in find_components(grid, bg=bg, connectivity=4, monochromatic=True)
        if _is_hollow_square_frame(component)
    ]
    return sorted(frames, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))


def _split_frames(frames: list[dict]) -> tuple[dict, dict, dict]:
    by_color: dict[int, list[dict]] = {}
    for frame in frames:
        by_color.setdefault(int(frame["color"]), []).append(frame)

    pair_groups = [group for group in by_color.values() if len(group) == 2]
    test_groups = [group for group in by_color.values() if len(group) == 1]
    assert len(pair_groups) == 1 and len(test_groups) == 1, "expected exactly one training pair"

    train_pair = sorted(pair_groups[0], key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    return train_pair[0], train_pair[1], test_groups[0][0]


def _inner_sprite(grid: np.ndarray, frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    return crop_bbox(grid, (r0 + 1, c0 + 1, r1 - 1, c1 - 1))


def _collect_observed_mapping(src: np.ndarray, dst: np.ndarray) -> dict[int, int]:
    assert src.shape == dst.shape, "training pair interiors must match in size"
    src_mask = src != BG
    dst_mask = dst != BG
    assert np.array_equal(src_mask, dst_mask), "training pair must preserve occupied cells"

    mapping: dict[int, int] = {}
    for r, c in np.argwhere(src_mask):
        color_in = int(src[r, c])
        color_out = int(dst[r, c])
        prev = mapping.setdefault(color_in, color_out)
        assert prev == color_out, "a source color must map consistently"
        assert color_in != color_out, "shift step must be non-zero"
    assert len(mapping) >= 2, "training pair must expose at least two color mappings"
    return mapping


def _candidate_shift_maps(colors: list[int], observed: dict[int, int]) -> list[dict[int, int]]:
    assert len(colors) >= 3, "cycle must contain at least three colors"
    anchor = colors[0]
    dedup: dict[tuple[int, ...], dict[int, int]] = {}
    for perm in itertools.permutations(colors[1:]):
        cycle = (anchor, *perm)
        for step in range(1, len(colors)):
            shift = {cycle[i]: cycle[(i + step) % len(colors)] for i in range(len(colors))}
            if all(shift[src] == dst for src, dst in observed.items()):
                signature = tuple(shift[color] for color in colors)
                dedup[signature] = shift
    return list(dedup.values())


def _infer_test_shift(colors: list[int], observed: dict[int, int], test_colors: list[int]) -> dict[int, int]:
    candidates = _candidate_shift_maps(colors, observed)
    assert candidates, "no cycle explains the observed training recoloring"

    full_signatures = {tuple(candidate[color] for color in colors) for candidate in candidates}
    if len(full_signatures) == 1:
        only = candidates[0]
        return {color: only[color] for color in test_colors}

    test_signatures = {tuple(candidate[color] for color in test_colors) for candidate in candidates}
    assert len(test_signatures) == 1, "test recoloring must be uniquely determined"
    signature = next(iter(test_signatures))
    return {color: mapped for color, mapped in zip(test_colors, signature)}


def _apply_shift(sprite: np.ndarray, shift: dict[int, int]) -> np.ndarray:
    out = sprite.copy()
    for color, mapped in shift.items():
        out[sprite == int(color)] = int(mapped)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer a cyclic color-shift rule from a framed training pair and recolor the test sprite.

    Input structure
    - The grid has a black background and exactly three non-overlapping hollow square frames.
    - Two frames share the same border color and form the training pair; the remaining frame is the test case.
    - The training pair contains the same interior occupied-cell pattern in both frames, while colors are shifted along an unknown cycle.

    Rule
    - Detect the two same-border frames, ordering them by leftmost top-left position to get training input then training output.
    - Compare their interior sprites cell by cell to collect the observed non-black color mapping.
    - Over the union of colors seen inside the training pair and the test sprite, find the shortest cyclic color order and a non-zero step whose shift matches every observed mapping.
    - Apply that same color shift to every non-black pixel in the test frame interior and return only the recolored test interior.

    Key insight
    - Nothing moves geometrically; the analogy is purely in color space, so the task is to infer a hidden cyclic palette and shared shift amount from one framed example.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert len(frames) == 3, "expected exactly three hollow square frames"
    train_in_frame, train_out_frame, test_frame = _split_frames(frames)

    train_in = _inner_sprite(g, train_in_frame)
    train_out = _inner_sprite(g, train_out_frame)
    test_in = _inner_sprite(g, test_frame)

    observed = _collect_observed_mapping(train_in, train_out)
    palette = sorted(
        (
            set(int(color) for color in np.unique(train_in))
            | set(int(color) for color in np.unique(train_out))
            | set(int(color) for color in np.unique(test_in))
        )
        - {BG}
    )
    test_colors = sorted(set(int(color) for color in np.unique(test_in)) - {BG})
    shift = _infer_test_shift(palette, observed, test_colors)
    return _apply_shift(test_in, shift)


def _weighted_cycle_length(rng: random.Random) -> int:
    pool: list[int] = []
    for length, weight in HIDDEN_CYCLE_WEIGHTS:
        pool.extend([length] * weight)
    return rng.choice(pool)


def _hidden_shift(cycle: list[int], step: int) -> dict[int, int]:
    return {cycle[i]: cycle[(i + step) % len(cycle)] for i in range(len(cycle))}


def _random_connected_mask(size: int, area: int, rng: random.Random) -> np.ndarray:
    cells = {(rng.randrange(size), rng.randrange(size))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(rng.choice(tuple(frontier)))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite(size: int, palette: list[int], rng: random.Random) -> np.ndarray:
    min_area = max(len(palette) + 1, size + 1)
    max_area = size * size - 1
    assert min_area <= max_area

    for _ in range(SPRITE_ATTEMPTS):
        area = rng.randint(min_area, max_area)
        mask = _random_connected_mask(size, area, rng)
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        rng.shuffle(coords)

        sprite = np.zeros((size, size), dtype=int)
        for color, (r, c) in zip(palette, coords):
            sprite[r, c] = int(color)
        for r, c in coords[len(palette) :]:
            sprite[r, c] = int(rng.choice(palette))
        if len(set(int(x) for x in sprite[sprite != BG])) == len(palette):
            return sprite
    raise RuntimeError("failed to build a valid sprite")


def _boxes_conflict(box_a: tuple[int, int, int], box_b: tuple[int, int, int], gap: int = MIN_FRAME_GAP) -> bool:
    ra, ca, sa = box_a
    rb, cb, sb = box_b
    a_bottom = ra + sa - 1
    a_right = ca + sa - 1
    b_bottom = rb + sb - 1
    b_right = cb + sb - 1
    return not (
        a_bottom + gap < rb
        or b_bottom + gap < ra
        or a_right + gap < cb
        or b_right + gap < ca
    )


def _sample_positions(
    height: int,
    width: int,
    pair_side: int,
    test_side: int,
    rng: random.Random,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None:
    vertical_pair = rng.random() < 0.25
    if vertical_pair:
        for _ in range(PLACEMENT_ATTEMPTS):
            c = rng.randint(0, width - pair_side)
            r_top = rng.randint(0, height - pair_side)
            r_bottom = rng.randint(0, height - pair_side)
            box_a = (min(r_top, r_bottom), c, pair_side)
            box_b = (max(r_top, r_bottom), c, pair_side)
            if _boxes_conflict(box_a, box_b):
                continue
            for _ in range(PLACEMENT_ATTEMPTS):
                test_box = (
                    rng.randint(0, height - test_side),
                    rng.randint(0, width - test_side),
                    test_side,
                )
                if not _boxes_conflict(test_box, box_a) and not _boxes_conflict(test_box, box_b):
                    return box_a, box_b, test_box
        return None

    boxes: list[tuple[int, int, int]] = []
    for side in (pair_side, pair_side, test_side):
        placed = None
        for _ in range(PLACEMENT_ATTEMPTS):
            candidate = (
                rng.randint(0, height - side),
                rng.randint(0, width - side),
                side,
            )
            if all(not _boxes_conflict(candidate, other) for other in boxes):
                placed = candidate
                break
        if placed is None:
            return None
        boxes.append(placed)
    return boxes[0], boxes[1], boxes[2]


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int], border_color: int, inner: np.ndarray) -> None:
    r0, c0, side = box
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner


def _test_palette(cycle: list[int], source_count: int, shift: dict[int, int], rng: random.Random) -> list[int]:
    source_colors = cycle[:source_count]
    target_only = [shift[color] for color in source_colors if shift[color] not in source_colors]
    unseen_sources = [color for color in cycle if color not in source_colors]

    base: list[int] = []
    for color in target_only + unseen_sources:
        if color not in base:
            base.append(color)

    extras = [color for color in source_colors if color not in base]
    rng.shuffle(extras)
    target_size = rng.randint(max(2, len(base)), min(3, len(cycle)))
    return base + extras[: target_size - len(base)]


def _semantic_validator(
    grid: np.ndarray,
    hidden_shift: dict[int, int],
    source_colors: list[int],
    expected_palette: set[int],
) -> bool:
    try:
        frames = _find_frames(grid)
        if len(frames) != 3:
            return False
        train_in_frame, train_out_frame, test_frame = _split_frames(frames)
        train_in = _inner_sprite(grid, train_in_frame)
        train_out = _inner_sprite(grid, train_out_frame)
        test_in = _inner_sprite(grid, test_frame)
        observed = _collect_observed_mapping(train_in, train_out)
    except Exception:
        return False

    if set(observed) != set(source_colors):
        return False

    visible_palette = set(int(color) for color in np.unique(train_in))
    visible_palette |= set(int(color) for color in np.unique(train_out))
    visible_palette |= set(int(color) for color in np.unique(test_in))
    visible_palette.discard(BG)
    if visible_palette != expected_palette:
        return False

    test_colors = sorted(set(int(color) for color in np.unique(test_in)) - {BG})
    if not any(color not in observed for color in test_colors):
        return False

    palette = sorted(visible_palette)
    candidates = _candidate_shift_maps(palette, observed)
    if not candidates:
        return False
    test_signatures = {tuple(candidate[color] for color in test_colors) for candidate in candidates}
    if len(test_signatures) != 1:
        return False

    expected = _apply_shift(test_in, {color: hidden_shift[color] for color in test_colors})
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    cycle_length = _weighted_cycle_length(rng)
    cycle = rng.sample(list(NONZERO_COLORS), cycle_length)
    step = rng.randint(1, cycle_length - 1)
    source_count = 2 if cycle_length <= 4 else 3
    shift = _hidden_shift(cycle, step)

    remaining_colors = [color for color in NONZERO_COLORS if color not in cycle]
    train_border, test_border = rng.sample(remaining_colors, 2)

    train_inner_size = rng.choice(INNER_SIZE_CHOICES)
    test_inner_size = rng.choice(INNER_SIZE_CHOICES)
    train_side = train_inner_size + 2
    test_side = test_inner_size + 2

    source_colors = cycle[:source_count]
    train_in = _make_sprite(train_inner_size, source_colors, rng)
    train_out = _apply_shift(train_in, shift)

    test_colors = _test_palette(cycle, source_count, shift, rng)
    test_in = _make_sprite(test_inner_size, test_colors, rng)

    for _ in range(PLACEMENT_ATTEMPTS):
        height = rng.randint(*GRID_SIZE_RANGE)
        width = rng.randint(*GRID_SIZE_RANGE)
        if height < max(train_side, test_side) or width < max(train_side, test_side):
            continue
        boxes = _sample_positions(height, width, train_side, test_side, rng)
        if boxes is None:
            continue

        box_a, box_b, test_box = boxes
        grid = np.zeros((height, width), dtype=int)

        train_boxes = sorted((box_a, box_b), key=lambda box: (box[1], box[0]))
        _draw_frame(grid, train_boxes[0], train_border, train_in)
        _draw_frame(grid, train_boxes[1], train_border, train_out)
        _draw_frame(grid, test_box, test_border, test_in)

        if _semantic_validator(grid, shift, source_colors, set(cycle)):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cyclic-color analogy puzzle")
