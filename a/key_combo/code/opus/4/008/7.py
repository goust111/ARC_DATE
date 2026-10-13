from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, paste


BLOCK_SIZE = 5
INNER_SIZE = 3
BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
GREY = int(Color.GREY)
ORANGE = int(Color.ORANGE)
RING_MASK = np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
RING_MASK[1:-1, 1:-1] = False


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=int))


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    rows = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            rows.append(r)
    mask = mask[rows, :]

    cols = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            cols.append(c)
    return mask[:, cols]


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(np.array(variant, copy=True))
    return variants


def _compressed_signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    compressed = _compress_runs(mask)
    return {_mask_key(variant) for variant in _d4_variants(compressed)}


def _raw_signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_key(variant) for variant in _d4_variants(mask)}


def _canonical_compressed(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_compressed_signature_set(mask))


def _canonical_raw_orbit(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_raw_signature_set(mask))


def _make_block(frame_color: int, pattern: np.ndarray) -> np.ndarray:
    patch = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    inner = patch[1:-1, 1:-1]
    inner[np.asarray(pattern, dtype=bool)] = BLUE
    return patch


def _extract_blocks(grid: np.ndarray, frame_color: int) -> list[dict]:
    blocks: list[dict] = []
    for coords in components_mask(grid == int(frame_color), connectivity=4):
        if int(coords.shape[0]) != 16:
            continue
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = map(int, bbox)
        if r1 - r0 + 1 != BLOCK_SIZE or c1 - c0 + 1 != BLOCK_SIZE:
            continue
        patch = crop_bbox(grid, bbox)
        if not np.all(patch[RING_MASK] == int(frame_color)):
            continue
        inner = patch[1:-1, 1:-1]
        if not np.all(np.isin(inner, [BG, BLUE])):
            continue
        blocks.append(
            {
                "bbox": bbox,
                "input_r": int(r0),
                "input_c": int(c0),
                "patch": patch,
                "pattern": inner == BLUE,
            }
        )
    return blocks


def _rect_from_top_left(r0: int, c0: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + BLOCK_SIZE - 1), int(c0 + BLOCK_SIZE - 1)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _ordered_group(blocks: list[dict], *, left_side: bool, wrong_sort: bool = False) -> list[dict]:
    reverse = not left_side
    if wrong_sort:
        reverse = not reverse
    return sorted(blocks, key=lambda block: int(block["input_c"]), reverse=reverse)


def _layout_candidates(
    candidates: list[dict],
    width: int,
    *,
    classify_fn=None,
    wrong_sort: bool = False,
) -> list[dict] | None:
    if classify_fn is None:
        classify_fn = lambda block: bool(block["match"])

    left_group = [block for block in candidates if bool(classify_fn(block))]
    right_group = [block for block in candidates if not bool(classify_fn(block))]
    placements: list[dict] = []

    for left_side, group in ((True, left_group), (False, right_group)):
        placed_side: list[dict] = []
        edge_col = 0 if left_side else int(width - BLOCK_SIZE)
        step = BLOCK_SIZE if left_side else -BLOCK_SIZE

        for block in _ordered_group(group, left_side=left_side, wrong_sort=wrong_sort):
            r0 = int(block["input_r"])
            c0 = int(edge_col)
            rect = _rect_from_top_left(r0, c0)
            while any(_rects_overlap(rect, placed["rect"]) for placed in placed_side):
                c0 += int(step)
                if c0 < 0 or c0 + BLOCK_SIZE > int(width):
                    return None
                rect = _rect_from_top_left(r0, c0)

            placed = {
                "block": block,
                "side": "left" if left_side else "right",
                "rect": rect,
                "r0": int(r0),
                "c0": int(c0),
            }
            placed_side.append(placed)
            placements.append(placed)

    return placements


def _render_output(grid: np.ndarray, candidates: list[dict], placements: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for block in candidates:
        r0 = int(block["input_r"])
        c0 = int(block["input_c"])
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = BG
    for placed in placements:
        paste(out, placed["block"]["patch"], int(placed["r0"]), int(placed["c0"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gray framed block to the left or right edge according to compressed shape-matching against the orange prototype.

    Input structure:
    - The background is black.
    - One orange 5x5 ring is the fixed prototype block.
    - Four to six gray 5x5 ring blocks are movable candidates.
    - Every block has a blue-on-black 3x3 inner pattern.

    Rule:
    - Crop each block's inner 3x3 pattern and run-compress it by deleting consecutive duplicate rows, then consecutive duplicate columns.
    - A gray block belongs to the left group iff its compressed pattern matches the prototype's compressed pattern under rotation or reflection; otherwise it belongs to the right group.
    - Clear the original gray blocks, keep the prototype unchanged, and redraw left-group blocks against column 0 and right-group blocks against the right edge.
    - Within each side, process blocks by original column distance to that edge; if a new block would overlap an already placed same-side block, shift it one full block-width inward and repeat until the overlap disappears.

    Key insight:
    - The decisive label is not the gray frame color but whether the inner 3x3 topology survives run-compression to the prototype up to dihedral symmetry, and that classification then drives the edge-stacking order.
    """
    g = np.asarray(grid, dtype=int)
    proto_blocks = _extract_blocks(g, ORANGE)
    cand_blocks = _extract_blocks(g, GREY)
    assert len(proto_blocks) == 1, "expected exactly one orange prototype block"
    assert len(cand_blocks) >= 1, "expected at least one gray candidate block"

    proto_pattern = np.asarray(proto_blocks[0]["pattern"], dtype=bool)
    proto_signatures = _compressed_signature_set(proto_pattern)
    for block in cand_blocks:
        compressed = _compress_runs(np.asarray(block["pattern"], dtype=bool))
        block["match"] = _mask_key(compressed) in proto_signatures

    placements = _layout_candidates(cand_blocks, g.shape[1])
    assert placements is not None, "candidate placement overflowed the grid width"
    return _render_output(g, cand_blocks, placements)


def _build_pattern_catalog() -> tuple[list[dict], dict, list[tuple[tuple[int, ...], ...]]]:
    entries: list[dict] = []
    classes: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)

    for bits in range(1 << (INNER_SIZE * INNER_SIZE)):
        mask = np.array(
            [(bits >> i) & 1 for i in range(INNER_SIZE * INNER_SIZE)],
            dtype=bool,
        ).reshape(INNER_SIZE, INNER_SIZE)
        area = int(mask.sum())
        if not 3 <= area <= 7:
            continue
        entry = {
            "mask": mask,
            "raw_key": _mask_key(mask),
            "canon": _canonical_compressed(mask),
            "raw_orbit": _canonical_raw_orbit(mask),
            "area": area,
        }
        entries.append(entry)
        classes[entry["canon"]].append(entry)

    good_classes: list[tuple[tuple[int, ...], ...]] = []
    for canon, members in classes.items():
        orbit_keys = {member["raw_orbit"] for member in members}
        if len(members) >= 4 and len(orbit_keys) >= 2:
            good_classes.append(canon)

    return entries, classes, good_classes


PATTERN_ENTRIES, PATTERN_CLASSES, GOOD_PROTOTYPE_CLASSES = _build_pattern_catalog()
COUNT_OPTIONS = [(4, 2), (4, 3), (5, 2), (5, 3), (6, 3)]


def _choose_patterns(n_match: int, n_nonmatch: int) -> tuple[dict, list[dict], list[dict]] | None:
    viable = [canon for canon in GOOD_PROTOTYPE_CLASSES if len(PATTERN_CLASSES[canon]) >= n_match + 1]
    if not viable:
        return None

    proto_canon = random.choice(viable)
    members = list(PATTERN_CLASSES[proto_canon])
    orbit_groups: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for member in members:
        orbit_groups[member["raw_orbit"]].append(member)

    proto_orbit = random.choice(list(orbit_groups))
    other_orbits = [orbit for orbit in orbit_groups if orbit != proto_orbit]
    if not other_orbits:
        return None

    prototype = random.choice(orbit_groups[proto_orbit])
    forced_match = random.choice(orbit_groups[random.choice(other_orbits)])

    used = {prototype["raw_key"], forced_match["raw_key"]}
    remaining_matches = [member for member in members if member["raw_key"] not in used]
    if len(remaining_matches) < n_match - 1:
        return None
    matches = [forced_match] + random.sample(remaining_matches, n_match - 1)

    pool = [
        member
        for member in PATTERN_ENTRIES
        if member["canon"] != proto_canon and member["raw_key"] not in used
    ]
    preferred = [member for member in pool if member["area"] == prototype["area"]]
    source = preferred if len(preferred) >= n_nonmatch else pool
    unique_pool = {member["raw_key"]: member for member in source}
    if len(unique_pool) < n_nonmatch:
        return None
    nonmatches = random.sample(list(unique_pool.values()), n_nonmatch)
    return prototype, matches, nonmatches


def _rows_overlap(r0: int, other_r0: int) -> bool:
    return not (r0 + BLOCK_SIZE - 1 < other_r0 or other_r0 + BLOCK_SIZE - 1 < r0)


def _pick_rows(candidates: list[dict], height: int) -> bool:
    max_r = int(height - BLOCK_SIZE)
    groups = {
        True: [idx for idx, block in enumerate(candidates) if bool(block["match"])],
        False: [idx for idx, block in enumerate(candidates) if not bool(block["match"])],
    }
    overlap_sides = [side for side, idxs in groups.items() if len(idxs) >= 2]
    if not overlap_sides:
        return False

    for block in candidates:
        block["input_r"] = random.randint(0, max_r)

    side = random.choice(overlap_sides)
    i0, i1 = random.sample(groups[side], 2)
    base_row = random.randint(0, max_r)
    offsets = [offset for offset in range(-4, 5) if 0 <= base_row + offset <= max_r]
    candidates[i0]["input_r"] = base_row
    candidates[i1]["input_r"] = base_row + random.choice(offsets)

    if len(groups[side]) >= 3 and random.random() < 0.5:
        remaining = [idx for idx in groups[side] if idx not in {i0, i1}]
        if remaining:
            k = random.choice(remaining)
            offsets = [offset for offset in range(-4, 5) if 0 <= base_row + offset <= max_r]
            candidates[k]["input_r"] = base_row + random.choice(offsets)
    return True


def _input_rect_has_gap(rect: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    return (
        rect[2] < other[0] - 1
        or other[2] < rect[0] - 1
        or rect[3] < other[1] - 1
        or other[3] < rect[1] - 1
    )


def _place_input_columns(
    width: int,
    prototype: dict,
    candidates: list[dict],
) -> bool:
    row_counts = Counter([int(prototype["input_r"])] + [int(block["input_r"]) for block in candidates])
    objects = [{"kind": "prototype", "block": prototype}] + [
        {"kind": "candidate", "block": block} for block in candidates
    ]
    random.shuffle(objects)
    objects.sort(
        key=lambda item: (
            int(row_counts[int(item["block"]["input_r"])]),
            1 if item["kind"] == "candidate" else 0,
        ),
        reverse=True,
    )

    placed_rects: list[tuple[int, int, int, int]] = []
    used_candidate_cols: set[int] = set()
    mid_lo = max(0, width // 3 - 1)
    mid_hi = min(width - BLOCK_SIZE, (2 * width) // 3)

    for item in objects:
        block = item["block"]
        r0 = int(block["input_r"])
        possible = []
        for c0 in range(width - BLOCK_SIZE + 1):
            rect = _rect_from_top_left(r0, c0)
            if all(_input_rect_has_gap(rect, other) for other in placed_rects):
                if item["kind"] == "candidate" and c0 in used_candidate_cols:
                    continue
                possible.append(c0)

        if item["kind"] == "prototype":
            preferred = [c0 for c0 in possible if mid_lo <= c0 <= mid_hi]
            if preferred:
                possible = preferred

        if not possible:
            return False

        c0 = random.choice(possible)
        block["input_c"] = int(c0)
        placed_rects.append(_rect_from_top_left(r0, c0))
        if item["kind"] == "candidate":
            used_candidate_cols.add(int(c0))
    return True


def _all_rects_non_overlapping(rects: list[tuple[int, int, int, int]]) -> bool:
    for i, rect in enumerate(rects):
        for other in rects[i + 1 :]:
            if _rects_overlap(rect, other):
                return False
    return True


def _build_scene(height: int, width: int, prototype: dict, candidates: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    paste(grid, prototype["patch"], int(prototype["input_r"]), int(prototype["input_c"]))
    for block in candidates:
        paste(grid, block["patch"], int(block["input_r"]), int(block["input_c"]))
    return grid


def _same_render(a: list[dict] | None, b: list[dict] | None, grid: np.ndarray, candidates: list[dict]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(_render_output(grid, candidates, a), _render_output(grid, candidates, b))


def _scene_is_semantically_valid(grid: np.ndarray, prototype: dict, candidates: list[dict]) -> bool:
    width = int(grid.shape[1])
    prototype_rect = _rect_from_top_left(int(prototype["input_r"]), int(prototype["input_c"]))

    match_cols = [int(block["input_c"]) for block in candidates if bool(block["match"])]
    non_cols = [int(block["input_c"]) for block in candidates if not bool(block["match"])]
    if not (len(match_cols) >= 2 and len(non_cols) >= 1):
        return False
    if not (min(match_cols) < max(non_cols) and min(non_cols) < max(match_cols)):
        return False

    proto_raw = _raw_signature_set(np.asarray(prototype["pattern"], dtype=bool))
    if not any(
        bool(block["match"]) and _mask_key(np.asarray(block["pattern"], dtype=bool)) not in proto_raw
        for block in candidates
    ):
        return False

    placements = _layout_candidates(candidates, width)
    if placements is None:
        return False

    all_output_rects = [placed["rect"] for placed in placements] + [prototype_rect]
    if not _all_rects_non_overlapping(all_output_rects):
        return False

    if not any(int(placed["c0"]) not in {0, width - BLOCK_SIZE} for placed in placements):
        return False

    wrong_sort = _layout_candidates(candidates, width, wrong_sort=True)
    if _same_render(placements, wrong_sort, grid, candidates):
        return False

    raw_match = _layout_candidates(
        candidates,
        width,
        classify_fn=lambda block: _mask_key(np.asarray(block["pattern"], dtype=bool)) in proto_raw,
    )
    if _same_render(placements, raw_match, grid, candidates):
        return False

    expected = _render_output(grid, candidates, placements)
    return np.array_equal(main(grid), expected)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(24, 30)
    n_candidates, n_match = random.choice(COUNT_OPTIONS)
    n_nonmatch = n_candidates - n_match

    chosen = _choose_patterns(n_match, n_nonmatch)
    if chosen is None:
        return None
    proto_entry, match_entries, nonmatch_entries = chosen

    prototype = {
        "pattern": np.array(proto_entry["mask"], copy=True),
        "patch": _make_block(ORANGE, proto_entry["mask"]),
        "input_r": random.randint(0, height - BLOCK_SIZE),
    }

    candidates: list[dict] = []
    for entry in match_entries:
        candidates.append(
            {
                "pattern": np.array(entry["mask"], copy=True),
                "patch": _make_block(GREY, entry["mask"]),
                "match": True,
            }
        )
    for entry in nonmatch_entries:
        candidates.append(
            {
                "pattern": np.array(entry["mask"], copy=True),
                "patch": _make_block(GREY, entry["mask"]),
                "match": False,
            }
        )
    random.shuffle(candidates)

    if not _pick_rows(candidates, height):
        return None
    if random.random() < 0.6:
        prototype["input_r"] = random.randint(0, height - BLOCK_SIZE)
    else:
        prototype["input_r"] = max(
            0,
            min(height - BLOCK_SIZE, int(np.median([block["input_r"] for block in candidates])) + random.randint(-6, 6)),
        )

    if not _place_input_columns(width, prototype, candidates):
        return None

    grid = _build_scene(height, width, prototype, candidates)
    if not _scene_is_semantically_valid(grid, prototype, candidates):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-classification edge-placement input")
