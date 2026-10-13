from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
YELLOW = 4
FRAME_SIZE = 5
INNER_SIZE = 3
SLOT_STRIDE = 6
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
RETRY_LIMIT = 500

_BORDER_MASK = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool)
_BORDER_MASK[1:-1, 1:-1] = False
_INNER_POSITIONS = [(r, c) for r in range(INNER_SIZE) for c in range(INNER_SIZE)]


def _frame_bbox_overlaps(a: dict, b: dict) -> bool:
    return not (
        a["r0"] + FRAME_SIZE - 1 < b["r0"]
        or b["r0"] + FRAME_SIZE - 1 < a["r0"]
        or a["c0"] + FRAME_SIZE - 1 < b["c0"]
        or b["c0"] + FRAME_SIZE - 1 < a["c0"]
    )


def _parse_frame(block: np.ndarray) -> tuple[int, np.ndarray] | None:
    border = block[_BORDER_MASK]
    color = int(border[0])
    if color == BG or not np.all(border == color):
        return None
    interior = block[1:-1, 1:-1]
    if not np.all((interior == BG) | (interior == color)):
        return None
    return color, (interior == color)


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            parsed = _parse_frame(g[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE])
            if parsed is None:
                continue
            color, inner = parsed
            frame = {
                "r0": r0,
                "c0": c0,
                "color": color,
                "inner": inner.copy(),
                "visible_count": int(inner.sum()),
            }
            if any(_frame_bbox_overlaps(frame, other) for other in frames):
                raise AssertionError("overlapping frame detections")
            frames.append(frame)
    frames.sort(key=lambda fr: (fr["r0"], fr["c0"]))
    return frames


def _group_summary(frames: list[dict]) -> list[dict]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)

    summary = []
    for color, group_frames in by_color.items():
        group_frames = sorted(group_frames, key=lambda fr: (fr["r0"], fr["c0"]))
        union = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        visible_counts = []
        total_visible = 0
        for frame in group_frames:
            union |= frame["inner"]
            visible = int(frame["inner"].sum())
            visible_counts.append(visible)
            total_visible += visible
        summary.append(
            {
                "color": color,
                "frames": group_frames,
                "union": union,
                "size": int(union.sum()),
                "max_visible": max(visible_counts),
                "total_visible": total_visible,
                "first": (group_frames[0]["r0"], group_frames[0]["c0"]),
            }
        )
    summary.sort(key=lambda info: info["first"])
    return summary


def _rank_groups(groups: list[dict], metric: str) -> list[dict]:
    return sorted(groups, key=lambda info: (-int(info[metric]), info["first"][0], info["first"][1]))


def _render_frame(out: np.ndarray, r0: int, c0: int, color: int, inner_mask: np.ndarray) -> None:
    patch = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=int)
    patch[_BORDER_MASK] = int(color)
    patch[1:-1, 1:-1][inner_mask] = int(color)
    out[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = patch


def _expected_from_groups(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    ranked = _rank_groups(groups, "size")
    out = np.zeros(shape, dtype=int)
    if len(ranked) == 1:
        survivors = {ranked[0]["color"]: YELLOW}
    else:
        survivors = {ranked[0]["color"]: YELLOW}
        for info in ranked[1:-1]:
            survivors[info["color"]] = info["color"]
    for info in groups:
        draw_color = survivors.get(info["color"])
        if draw_color is None:
            continue
        for frame in info["frames"]:
            _render_frame(out, frame["r0"], frame["c0"], draw_color, info["union"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-colored frame motifs first, then recolor or erase groups according to the completed motif-area ranking.

    Input structure:
    - The background is black, and the foreground consists of 4-6 separate 5x5 single-color square frames.
    - Each frame has a one-pixel border enclosing a 3x3 interior whose colored cells, if any, use the same color as the border.
    - Frames of the same color belong to one group and carry different partial subsets of a shared interior motif.

    Rule:
    - Detect every 5x5 monochrome frame and group frames by border color.
    - For each color group, take the union of all occupied interior positions across that group's frames and paint that completed motif into every frame of the group.
    - Rank the color groups by the completed single-frame motif size in descending order, breaking ties by the raster position of the group's first frame.
    - Repaint the highest-ranked group yellow, erase the lowest-ranked group to black, and keep all middle groups in their own colors with the completed motif.

    Key insight:
    - The decisive area comparison happens after motif completion: the union-filled motif size can reverse the apparent size order seen in the incomplete input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _detect_frames(g)
    assert 4 <= len(frames) <= 6, "expected 4-6 frames"
    groups = _group_summary(frames)
    assert 2 <= len(groups) <= 3, "expected 2-3 color groups"
    assert all(2 <= len(info["frames"]) <= 3 for info in groups), "each color must appear 2-3 times"
    return _expected_from_groups(g.shape, groups)


def _sample_used_slots(rows: int, cols: int, fill_count: int) -> list[tuple[int, int]] | None:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(80):
        used = random.sample(slots, fill_count)
        row_cover = {r for r, _ in used}
        col_cover = {c for _, c in used}
        if len(row_cover) == rows and len(col_cover) == cols:
            return sorted(used)
    return None


def _choose_group_sizes(fill_count: int) -> list[int]:
    if fill_count == 4:
        return [2, 2]
    if fill_count == 5:
        return random.choice(([2, 3], [3, 2]))
    return random.choice(([3, 3], [2, 2, 2]))


def _sample_pattern_size(frame_count: int, *, max_size: int = 8) -> tuple[int, int]:
    size = random.randint(2, max_size)
    min_visible = max(1, (size + frame_count - 1) // frame_count)
    max_visible = size - 1
    if min_visible > max_visible:
        raise ValueError("infeasible incomplete motif size")
    return size, random.randint(min_visible, max_visible)


def _make_subsets(full_positions: set[tuple[int, int]], frame_count: int, mode: str, max_visible: int) -> list[set[tuple[int, int]]] | None:
    full = sorted(full_positions)
    size = len(full)
    for _ in range(120):
        if mode == "one_full":
            subsets = [set(full_positions)]
            for _idx in range(frame_count - 1):
                target = random.randint(max(1, size - 3), size - 1)
                subsets.append(set(random.sample(full, target)))
        else:
            if max_visible >= size or size > frame_count * max_visible:
                return None
            anchor = set(random.sample(full, max_visible))
            missing = list(full_positions - anchor)
            random.shuffle(missing)
            buckets = [set() for _ in range(frame_count - 1)]
            for idx, pos in enumerate(missing):
                buckets[idx % len(buckets)].add(pos)
            subsets = [anchor]
            anchor_list = sorted(anchor)
            for bucket in buckets:
                min_target = max(len(bucket), max(1, max_visible - 2))
                target = random.randint(min_target, max_visible)
                subset = set(bucket)
                need = target - len(subset)
                if need > 0:
                    subset |= set(random.sample(anchor_list, need))
                if subset == full_positions:
                    removable = [pos for pos in subset if pos in anchor]
                    if removable:
                        subset.remove(random.choice(removable))
                if not subset:
                    subset.add(random.choice(full))
                subsets.append(subset)

        if set().union(*subsets) != full_positions:
            continue
        if mode == "all_incomplete" and any(subset == full_positions for subset in subsets):
            continue
        if len({frozenset(subset) for subset in subsets}) < 2:
            continue
        if max(len(subset) for subset in subsets) != max_visible and mode == "all_incomplete":
            continue
        if mode == "one_full" and max(len(subset) for subset in subsets) != size:
            continue
        return subsets
    return None


def _choose_group_specs(group_sizes: list[int]) -> list[dict] | None:
    group_count = len(group_sizes)
    for _ in range(160):
        overtaker, overtaken = random.sample(range(group_count), 2)
        specs: list[dict | None] = [None] * group_count

        small_size = random.randint(3, 6)
        large_size = random.randint(small_size + 1, min(8, small_size + 3))
        try:
            _size_a, visible_a = _sample_pattern_size(group_sizes[overtaker], max_size=large_size)
        except ValueError:
            continue
        if _size_a != large_size:
            visible_a = max(1, min(visible_a, large_size - 1))
        min_visible_a = max(1, (large_size + group_sizes[overtaker] - 1) // group_sizes[overtaker])
        max_visible_a = min(large_size - 1, small_size - 1)
        if min_visible_a > max_visible_a:
            continue
        visible_a = random.randint(min_visible_a, max_visible_a)
        specs[overtaker] = {
            "post_size": large_size,
            "mode": "all_incomplete",
            "max_visible": visible_a,
        }
        specs[overtaken] = {
            "post_size": small_size,
            "mode": "one_full",
            "max_visible": small_size,
        }

        if group_count == 3:
            extra = next(idx for idx in range(group_count) if idx not in {overtaker, overtaken})
            tie_branch = random.random() < 0.45
            post_size = small_size if tie_branch else random.randint(2, small_size)
            if random.random() < 0.65 and post_size >= 3:
                max_visible = random.randint(max(1, (post_size + group_sizes[extra] - 1) // group_sizes[extra]), post_size - 1)
                mode = "all_incomplete"
            else:
                max_visible = post_size
                mode = "one_full"
            specs[extra] = {
                "post_size": post_size,
                "mode": mode,
                "max_visible": max_visible,
            }

        if any(spec is None for spec in specs):
            continue
        return specs  # type: ignore[return-value]
    return None


def _build_group_subsets(spec: dict, frame_count: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    full_positions = set(random.sample(_INNER_POSITIONS, spec["post_size"]))
    subsets = _make_subsets(full_positions, frame_count, spec["mode"], spec["max_visible"])
    if subsets is None:
        return None
    full_mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
    for r, c in full_positions:
        full_mask[r, c] = True
    subset_masks = []
    for subset in sorted(subsets, key=lambda cells: (-len(cells), sorted(cells))):
        mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        for r, c in subset:
            mask[r, c] = True
        subset_masks.append(mask)
    return full_mask, subset_masks


def _render_input_instance(
    rows: int,
    cols: int,
    row_pad: tuple[int, int],
    col_pad: tuple[int, int],
    assignments: list[dict],
) -> np.ndarray:
    top, bottom = row_pad
    left, right = col_pad
    height = top + rows * FRAME_SIZE + (rows - 1) + bottom
    width = left + cols * FRAME_SIZE + (cols - 1) + right
    grid = np.zeros((height, width), dtype=int)
    for item in assignments:
        slot_r, slot_c = item["slot"]
        r0 = top + slot_r * SLOT_STRIDE
        c0 = left + slot_c * SLOT_STRIDE
        _render_frame(grid, r0, c0, item["color"], item["inner"])
    return grid


def _build_expected_from_assignment(shape: tuple[int, int], assignments: list[dict]) -> np.ndarray:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in assignments:
        grouped[item["color"]].append(
            {
                "r0": item["r0"],
                "c0": item["c0"],
                "color": item["color"],
                "inner": item["inner"].copy(),
                "visible_count": int(item["inner"].sum()),
            }
        )
    groups = _group_summary([frame for frames in grouped.values() for frame in frames])
    return _expected_from_groups(shape, groups)


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, fill_count: int) -> bool:
    try:
        frames = _detect_frames(inp)
    except AssertionError:
        return False
    if len(frames) != fill_count:
        return False
    groups = _group_summary(frames)
    if not (2 <= len(groups) <= 3):
        return False
    if any(not (2 <= len(info["frames"]) <= 3) for info in groups):
        return False
    if not any(
        any(not np.array_equal(frame["inner"], info["union"]) for frame in info["frames"])
        for info in groups
    ):
        return False

    post_order = [info["color"] for info in _rank_groups(groups, "size")]
    pre_order = [info["color"] for info in _rank_groups(groups, "max_visible")]
    if post_order == pre_order:
        return False
    if not any(
        a["max_visible"] < b["max_visible"] and a["size"] > b["size"]
        for a in groups
        for b in groups
        if a["color"] != b["color"]
    ):
        return False
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        rows, cols = random.choice(((2, 3), (3, 2)))
        fill_count = random.choices([4, 5, 6], weights=[2, 3, 5], k=1)[0]
        used_slots = _sample_used_slots(rows, cols, fill_count)
        if used_slots is None:
            continue

        group_sizes = _choose_group_sizes(fill_count)
        specs = _choose_group_specs(group_sizes)
        if specs is None:
            continue
        colors = random.sample(INPUT_COLORS, len(group_sizes))

        shuffled_slots = used_slots[:]
        random.shuffle(shuffled_slots)
        slot_groups = []
        cursor = 0
        for size in group_sizes:
            slot_groups.append(sorted(shuffled_slots[cursor : cursor + size]))
            cursor += size

        assignments = []
        top, bottom = random.randint(0, 3), random.randint(0, 3)
        left, right = random.randint(0, 3), random.randint(0, 3)
        for color, slots, spec in zip(colors, slot_groups, specs):
            built = _build_group_subsets(spec, len(slots))
            if built is None:
                break
            _full_mask, subset_masks = built
            if len(subset_masks) != len(slots):
                break
            for slot, subset_mask in zip(slots, subset_masks):
                r0 = top + slot[0] * SLOT_STRIDE
                c0 = left + slot[1] * SLOT_STRIDE
                assignments.append(
                    {
                        "slot": slot,
                        "r0": r0,
                        "c0": c0,
                        "color": color,
                        "inner": subset_mask,
                    }
                )
        else:
            inp = _render_input_instance(rows, cols, (top, bottom), (left, right), assignments)
            expected = _build_expected_from_assignment(inp.shape, assignments)
            if _validate_semantics(inp, expected, fill_count):
                return inp
            continue
    raise RuntimeError("failed to generate a valid input")
