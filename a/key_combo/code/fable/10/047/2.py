from __future__ import annotations

import numpy as np


BACKGROUND = 0
MIN_FRAME_SIDE = 5
MAX_GENERATION_ATTEMPTS = 400


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_info(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "frame color must be present"

    r0, c0, r1, c1 = _bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3, "frame bbox must be at least 3x3"

    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = _border_mask(height, width)
    assert not np.any(local & ~border), "all colored cells must lie on the frame perimeter"

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "size": (height, width),
        "complete": bool(np.array_equal(local, border)),
        "mask": local,
    }


def _parse_scene(grid: np.ndarray) -> tuple[dict, dict[tuple[int, int], dict]]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != BACKGROUND)
    assert colors, "expected at least one foreground frame"

    frames = [_frame_info(grid, color) for color in colors]
    complete = [frame for frame in frames if frame["complete"]]
    assert len(complete) == 1, "expected exactly one complete source frame"

    sizes = [frame["size"] for frame in frames]
    assert len(set(sizes)) == len(sizes), "frame bbox sizes must be unique"

    source = complete[0]
    fragmented = {frame["size"]: frame for frame in frames if not frame["complete"]}
    return source, fragmented


def _propagation_sizes(
    source_size: tuple[int, int],
    fragmented: dict[tuple[int, int], dict],
) -> list[tuple[int, int]]:
    height, width = map(int, source_size)
    infected: list[tuple[int, int]] = []
    while True:
        next_size = (height - 2, width - 2)
        if next_size not in fragmented:
            break
        infected.append(next_size)
        height, width = next_size
    return infected


def _paint_frame_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete and recolor only the consecutive size-decreasing fragmented-frame chain below the unique source frame.

    Input structure:
    - A black background contains exactly one complete monochrome hollow rectangular frame (the source frame).
    - Every other non-black color forms a fragmented hollow rectangular frame: its colored cells lie on the perimeter
      of a unique bbox, but some perimeter cells are missing.
    - Frame bbox sizes are all distinct, and some fragmented frames follow a size chain where both height and width
      decrease by exactly two at each step.

    Rule:
    - Find the unique complete source frame and note its color and bbox size.
    - Repeatedly look for the fragmented frame whose bbox is exactly one ring smaller: `(h-2, w-2)` relative to the
      current chain tail. If it exists, complete its full perimeter and recolor that entire perimeter to the source color.
    - The newly completed frame becomes the next chain tail. If any size step is missing, propagation stops immediately,
      even if smaller fragmented frames elsewhere would match later sizes.

    Key insight:
    - The transformation depends on causal continuity along a strict `(h-2, w-2)` bbox-size chain, not on completing
      every fragmented frame or every smaller compatible-looking frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source, fragmented = _parse_scene(g)
    infected_sizes = _propagation_sizes(source["size"], fragmented)

    out = g.copy()
    source_color = int(source["color"])
    for size in infected_sizes:
        _paint_frame_border(out, fragmented[size]["bbox"], source_color)
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_boxes(
    sizes: list[tuple[int, int]],
    height: int,
    width: int,
    rng: np.random.Generator,
) -> dict[tuple[int, int], tuple[int, int, int, int]] | None:
    placed: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for size in sorted(sizes, key=lambda item: item[0] * item[1], reverse=True):
        h, w = size
        candidates = []
        for r0 in range(height - h + 1):
            for c0 in range(width - w + 1):
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(not _boxes_overlap(box, other, pad=1) for other in placed.values()):
                    candidates.append(box)
        if not candidates:
            return None
        placed[size] = candidates[int(rng.integers(len(candidates)))]
    return placed


def _fragment_patch(height: int, width: int, color: int, rng: np.random.Generator) -> np.ndarray:
    border = _border_mask(height, width)
    keep = border.copy()

    top_inner = [(0, c) for c in range(1, width - 1)]
    bottom_inner = [(height - 1, c) for c in range(1, width - 1)]
    left_inner = [(r, 0) for r in range(1, height - 1)]
    right_inner = [(r, width - 1) for r in range(1, height - 1)]
    side_cells = {
        "top": top_inner,
        "bottom": bottom_inner,
        "left": left_inner,
        "right": right_inner,
    }

    mandatory = {
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    }
    for cells in side_cells.values():
        mandatory.add(cells[int(rng.integers(len(cells)))])

    chosen_sides = rng.choice(list(side_cells.keys()), size=int(rng.integers(1, 5)), replace=False)
    for side in chosen_sides:
        removable = [coord for coord in side_cells[side] if coord not in mandatory]
        if not removable:
            continue
        segment_len = int(rng.integers(1, min(len(removable), max(2, len(side_cells[side]) // 2)) + 1))
        start = int(rng.integers(0, len(removable) - segment_len + 1))
        for coord in removable[start : start + segment_len]:
            keep[coord] = False

    extra_removable = [coord for coord in np.argwhere(border) if tuple(map(int, coord)) not in mandatory]
    rng.shuffle(extra_removable)
    extra_count = int(rng.integers(0, min(len(extra_removable), max(2, (height + width) // 5)) + 1))
    for rr, cc in extra_removable[:extra_count]:
        keep[int(rr), int(cc)] = False

    if np.array_equal(keep, border):
        rr, cc = extra_removable[0]
        keep[int(rr), int(cc)] = False

    patch = np.zeros((height, width), dtype=int)
    patch[keep] = int(color)
    return patch


def _size_candidates(
    source_size: tuple[int, int],
    used_sizes: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    source_h, source_w = source_size
    candidates: set[tuple[int, int]] = set()

    for level in range(1, 7):
        base_h = source_h - 2 * level
        base_w = source_w - 2 * level
        for delta_h, delta_w in (
            (0, 1),
            (1, 0),
            (1, 1),
            (0, -1),
            (-1, 0),
            (2, -1),
            (-1, 2),
        ):
            h = base_h + delta_h
            w = base_w + delta_w
            if h >= MIN_FRAME_SIDE and w >= MIN_FRAME_SIDE and (h, w) not in used_sizes:
                candidates.add((h, w))

    for delta_h, delta_w in ((1, 3), (3, 1), (2, 2), (1, 1)):
        h = source_h + delta_h
        w = source_w + delta_w
        if h <= 18 and w <= 18 and (h, w) not in used_sizes:
            candidates.add((h, w))

    return sorted(candidates)


def _build_expected_output(
    inp: np.ndarray,
    boxes_by_size: dict[tuple[int, int], tuple[int, int, int, int]],
    infected_sizes: list[tuple[int, int]],
    source_color: int,
) -> np.ndarray:
    out = inp.copy()
    for size in infected_sizes:
        _paint_frame_border(out, boxes_by_size[size], source_color)
    return out


def _validate_generated_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    source_size: tuple[int, int],
    infected_sizes: list[tuple[int, int]],
    bait_sizes: list[tuple[int, int]],
    mismatch_sizes: list[tuple[int, int]],
    boxes_by_size: dict[tuple[int, int], tuple[int, int, int, int]],
    source_color: int,
) -> bool:
    try:
        solved = main(inp)
        parsed_source, fragmented = _parse_scene(inp)
    except AssertionError:
        return False

    if int(parsed_source["color"]) != int(source_color) or parsed_source["size"] != source_size:
        return False
    if _propagation_sizes(source_size, fragmented) != infected_sizes:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    if not bait_sizes or not mismatch_sizes:
        return False

    gap_continuation = infected_sizes + bait_sizes
    shortcut = _build_expected_output(inp, boxes_by_size, gap_continuation, source_color)
    if np.array_equal(shortcut, expected):
        return False

    mismatch_shortcut = _build_expected_output(inp, boxes_by_size, infected_sizes + [mismatch_sizes[0]], source_color)
    if np.array_equal(mismatch_shortcut, expected):
        return False

    unchanged_fragment_count = 0
    for size, box in boxes_by_size.items():
        if size in infected_sizes:
            continue
        r0, c0, r1, c1 = box
        if np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], expected[r0 : r1 + 1, c0 : c1 + 1]):
            unchanged_fragment_count += 1
    if unchanged_fragment_count == 0:
        return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    source_h = int(rng.integers(11, 16))
    source_w = int(rng.integers(11, 18))
    max_level = min((source_h - MIN_FRAME_SIDE) // 2, (source_w - MIN_FRAME_SIDE) // 2)
    if max_level < 3:
        return None

    max_infected = min(3, max_level - 2)
    infected_len = int(rng.integers(1, max_infected + 1))
    gap_level = infected_len + 1

    infected_sizes = [(source_h - 2 * level, source_w - 2 * level) for level in range(1, infected_len + 1)]
    bait_candidates = [(source_h - 2 * level, source_w - 2 * level) for level in range(gap_level + 1, max_level + 1)]
    bait_count = int(rng.integers(1, min(2, len(bait_candidates)) + 1))
    bait_sizes = list(rng.choice(np.array(bait_candidates, dtype=int), size=bait_count, replace=False))
    bait_sizes = [tuple(map(int, size)) for size in bait_sizes]
    bait_sizes.sort()

    used_sizes = {(source_h, source_w), *infected_sizes, *bait_sizes}
    mismatch_candidates = _size_candidates((source_h, source_w), used_sizes)
    if len(mismatch_candidates) < 1:
        return None
    rng.shuffle(mismatch_candidates)
    mismatch_count = 1 + int(rng.random() < 0.55 and len(mismatch_candidates) >= 2)
    mismatch_sizes = mismatch_candidates[:mismatch_count]

    all_fragment_sizes = infected_sizes + bait_sizes + mismatch_sizes
    total_objects = 1 + len(all_fragment_sizes)
    if total_objects > 9:
        return None

    max_h = max(size[0] for size in [(source_h, source_w), *all_fragment_sizes])
    max_w = max(size[1] for size in [(source_h, source_w), *all_fragment_sizes])
    grid_h = int(rng.integers(max(max_h + 4, 22), 31))
    grid_w = int(rng.integers(max(max_w + 4, 22), 31))

    boxes_by_size = _place_boxes([(source_h, source_w), *all_fragment_sizes], grid_h, grid_w, rng)
    if boxes_by_size is None:
        return None

    colors = list(map(int, rng.choice(np.arange(1, 10), size=total_objects, replace=False)))
    source_color = colors[0]
    fragment_colors = colors[1:]

    inp = np.zeros((grid_h, grid_w), dtype=int)
    _paint_frame_border(inp, boxes_by_size[(source_h, source_w)], source_color)

    for size, color in zip(all_fragment_sizes, fragment_colors):
        patch = _fragment_patch(size[0], size[1], color, rng)
        r0, c0, r1, c1 = boxes_by_size[size]
        inp[r0 : r1 + 1, c0 : c1 + 1] = patch

    expected = _build_expected_output(inp, boxes_by_size, infected_sizes, source_color)
    if not _validate_generated_scene(
        inp,
        expected,
        (source_h, source_w),
        infected_sizes,
        bait_sizes,
        mismatch_sizes,
        boxes_by_size,
        source_color,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid size-decreasing infection-chain input")
