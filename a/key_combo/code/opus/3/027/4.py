import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FRAME_SIDES = (3, 4)
FRAME_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp_coords = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": comp_coords,
                    "bbox": _bbox_coords(comp_coords),
                    "area": int(comp_coords.shape[0]),
                }
            )
    return components


def _border_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"

    frames: list[dict] = []
    side_seen: set[int] = set()
    for component in _find_components(g, bg):
        r0, c0, r1, c1 = component["bbox"]
        side = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert side == width, "every foreground component must have a square bbox"
        assert side in FRAME_SIDES, "frame side must be 3 or 4"
        side_seen.add(side)

        patch = _crop_bbox(g, component["bbox"])
        color = int(component["color"])
        border = _border_mask(side)
        assert np.all(patch[border] == color), "frame border must be complete"
        assert np.all((patch == bg) | (patch == color)), "frame patch must only contain bg or its own color"

        inner_rel = frozenset(
            (rr, cc)
            for rr in range(1, side - 1)
            for cc in range(1, side - 1)
            if int(patch[rr, cc]) == color
        )
        frames.append(
            {
                "color": color,
                "bbox": component["bbox"],
                "side": side,
                "inner_rel": inner_rel,
            }
        )

    assert frames, "expected at least one frame"
    assert len(side_seen) == 1, "all frames must have the same side length"
    return side_seen.pop(), frames


def _group_frames(frames: list[dict]) -> list[dict]:
    by_color: dict[int, list[dict]] = {}
    for frame in frames:
        by_color.setdefault(int(frame["color"]), []).append(frame)

    groups: list[dict] = []
    for color, items in by_color.items():
        full_inner = set()
        for frame in items:
            full_inner |= set(frame["inner_rel"])
        first_col = min(int(frame["bbox"][1]) for frame in items)
        groups.append(
            {
                "color": int(color),
                "frames": items,
                "count": len(items),
                "first_col": first_col,
                "full_inner": frozenset(full_inner),
            }
        )

    tie_cols = {}
    for group in groups:
        tie_cols.setdefault(int(group["count"]), []).append(int(group["first_col"]))
    for cols in tie_cols.values():
        assert len(cols) == len(set(cols)), "equal-sized groups need distinct leftmost columns"

    groups.sort(key=lambda group: (-int(group["count"]), int(group["first_col"])))
    return groups


def _build_output(groups: list[dict]) -> np.ndarray:
    layer_colors: list[int] = []
    for group in groups:
        layer_colors.extend([int(group["color"])] * int(group["count"]))

    total_layers = len(layer_colors)
    side = 2 * total_layers - 1
    out = np.full((side, side), BACKGROUND, dtype=int)
    for layer, color in enumerate(layer_colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete grouped square-frame motifs, then turn the per-color frame counts into concentric layers.

    Input structure:
    - Black is the background.
    - Several equal-sized monochrome square frames of side 3 or 4 are separated by background gaps.
    - Frames sharing a border color belong to one group; inside each frame, same-color motif pixels show only a subset of that group's full interior pattern.

    Rule:
    - Detect every square frame and group frames by their border color.
    - Within each color group, union the interior relative offsets and use that union as the completed motif for every frame in the group.
    - Count how many frames each color group contains, and order the groups by decreasing frame count, breaking ties by the group's leftmost input column.
    - Let S be the total number of frames. Build a compact square with exactly S visible concentric layers, so its side is 2*S-1; from outside to inside, repeat each group's color for as many consecutive layers as that group has frames.

    Key insight:
    - The same-color frames are incomplete views of one shared motif, and those color-group sizes become the legend that determines the outside-to-inside layer thicknesses.
    """
    _side, frames = _extract_frames(grid)
    groups = _group_frames(frames)

    completed_groups = []
    for group in groups:
        full_inner = set(group["full_inner"])
        completed_groups.append({**group, "completed_inner": frozenset(full_inner)})

    return _build_output(completed_groups)


VALID_COUNT_TUPLES = {
    2: [
        (4, 2),
        (3, 3),
        (4, 3),
        (4, 4),
    ],
    3: [
        (2, 2, 2),
        (3, 2, 2),
        (3, 3, 2),
        (4, 2, 2),
        (4, 3, 2),
        (4, 4, 2),
        (4, 3, 3),
        (3, 3, 3),
    ],
}


def _sample_counts(color_count: int) -> list[int]:
    options = VALID_COUNT_TUPLES[color_count]
    want_tie = random.random() < 0.55
    filtered = [counts for counts in options if (len(set(counts)) < len(counts)) == want_tie]
    if not filtered:
        filtered = options
    return list(random.choice(filtered))


def _sample_full_motif(frame_side: int) -> frozenset[tuple[int, int]]:
    inner_cells = [(r, c) for r in range(1, frame_side - 1) for c in range(1, frame_side - 1)]
    if frame_side == 3:
        return frozenset(inner_cells)
    motif_size = random.choice((1, 2, 2, 3, 4))
    chosen = random.sample(inner_cells, k=motif_size)
    return frozenset(chosen)


def _sample_subsets(full_motif: frozenset[tuple[int, int]], count: int) -> list[frozenset[tuple[int, int]]]:
    motif_cells = list(full_motif)
    if len(motif_cells) == 1:
        subsets = [frozenset()] * (count - 1) + [full_motif]
        random.shuffle(subsets)
        return subsets

    for _ in range(200):
        subsets = [set() for _ in range(count)]
        for cell in motif_cells:
            owners = [i for i in range(count) if random.random() < 0.6]
            if not owners:
                owners = [random.randrange(count)]
            for idx in owners:
                subsets[idx].add(cell)

        frozen = [frozenset(subset) for subset in subsets]
        union = frozenset().union(*frozen)
        unique = len(set(frozen))
        has_proper_subset = any(subset != full_motif for subset in frozen)
        if union == full_motif and unique >= 2 and has_proper_subset:
            return frozen

    raise RuntimeError("failed to sample non-degenerate motif subsets")


def _choose_slot_grid(total_frames: int, color_count: int, frame_side: int) -> tuple[int, int]:
    candidates = []
    for rows in (2, 3):
        cols = max(color_count, (total_frames + rows - 1) // rows)
        if rows * cols < total_frames:
            continue
        max_h = rows * frame_side + (rows - 1) * 2 + 6
        max_w = cols * frame_side + (cols - 1) * 2 + 6
        if max_h <= 30 and max_w <= 30:
            candidates.append((rows, cols))
    assert candidates, "expected at least one feasible slot grid"
    return random.choice(candidates)


def _sample_axis_positions(count: int, span: int) -> tuple[list[int], list[int], int, int]:
    gaps = [random.choice((1, 1, 2)) for _ in range(count - 1)]
    lead = random.choice((1, 1, 2, 3))
    tail = random.choice((1, 1, 2, 3))
    positions = [lead]
    for gap in gaps:
        positions.append(positions[-1] + span + gap)
    size = positions[-1] + span + tail
    return positions, gaps, lead, size


def _assign_anchor_columns(counts: list[int], row_count: int, col_count: int) -> dict[int, int] | None:
    order = sorted(range(len(counts)), key=lambda idx: (-counts[idx], idx))
    assignment: dict[int, int] = {}

    def backtrack(i: int, used: set[int]) -> bool:
        if i == len(order):
            return True
        group_idx = order[i]
        feasible = [
            col
            for col in range(col_count)
            if col not in used and row_count * (col_count - col) >= counts[group_idx]
        ]
        random.shuffle(feasible)
        for col in feasible:
            assignment[group_idx] = col
            used.add(col)
            if backtrack(i + 1, used):
                return True
            used.remove(col)
            del assignment[group_idx]
        return False

    if backtrack(0, set()):
        return assignment
    return None


def _assign_slots(counts: list[int], row_count: int, col_count: int, anchor_cols: dict[int, int]) -> dict[int, list[tuple[int, int]]] | None:
    available = {(r, c) for r in range(row_count) for c in range(col_count)}
    slot_map: dict[int, list[tuple[int, int]]] = {}
    for group_idx in sorted(range(len(counts)), key=lambda idx: anchor_cols[idx], reverse=True):
        anchor_col = anchor_cols[group_idx]
        anchor_choices = [slot for slot in available if slot[1] == anchor_col]
        if not anchor_choices:
            return None
        anchor_slot = random.choice(anchor_choices)
        available.remove(anchor_slot)

        allowed = [slot for slot in available if slot[1] >= anchor_col]
        need = counts[group_idx] - 1
        if len(allowed) < need:
            return None
        chosen = random.sample(allowed, k=need)
        for slot in chosen:
            available.remove(slot)
        group_slots = [anchor_slot] + chosen
        random.shuffle(group_slots)
        slot_map[group_idx] = group_slots
    return slot_map


def _paint_frame(grid: np.ndarray, top: int, left: int, side: int, color: int, inner_rel: frozenset[tuple[int, int]]) -> None:
    grid[top, left : left + side] = color
    grid[top + side - 1, left : left + side] = color
    grid[top : top + side, left] = color
    grid[top : top + side, left + side - 1] = color
    for dr, dc in inner_rel:
        grid[top + dr, left + dc] = color


def _spec_output(groups: list[dict]) -> np.ndarray:
    ordered = sorted(groups, key=lambda group: (-int(group["count"]), int(group["first_col"])))
    return _build_output(ordered)


def _validate_generated(grid: np.ndarray, spec_groups: list[dict], frame_side: int) -> bool:
    parsed_side, frames = _extract_frames(grid)
    if parsed_side != frame_side:
        return False

    parsed_groups = _group_frames(frames)
    expected_by_color = {int(group["color"]): group for group in spec_groups}
    if {int(group["color"]) for group in parsed_groups} != set(expected_by_color):
        return False

    for group in parsed_groups:
        color = int(group["color"])
        expected = expected_by_color[color]
        if int(group["count"]) != int(expected["count"]):
            return False
        if int(group["first_col"]) != int(expected["first_col"]):
            return False
        if frozenset(group["full_inner"]) != frozenset(expected["full_inner"]):
            return False

        subset_patterns = [frozenset(frame["inner_rel"]) for frame in group["frames"]]
        if len(set(subset_patterns)) < 2:
            return False
        if not any(pattern != frozenset(expected["full_inner"]) for pattern in subset_patterns):
            return False

    tied_groups = Counter(group["count"] for group in spec_groups)
    if not any(v > 1 for v in tied_groups.values()):
        count_order = [group["count"] for group in sorted(spec_groups, key=lambda g: g["first_col"])]
        if count_order == sorted(count_order, reverse=True):
            return False

    expected_output = _spec_output(spec_groups)
    actual_output = main(grid)
    return np.array_equal(actual_output, expected_output)


def _gen_attempt() -> np.ndarray | None:
    color_count = random.choice((2, 2, 3))
    counts = _sample_counts(color_count)
    random.shuffle(counts)
    total_frames = sum(counts)
    frame_side = random.choice(FRAME_SIDES)
    row_count, col_count = _choose_slot_grid(total_frames, color_count, frame_side)

    row_tops, _row_gaps, _top_margin, height = _sample_axis_positions(row_count, frame_side)
    col_tops, _col_gaps, _left_margin, width = _sample_axis_positions(col_count, frame_side)
    if height > 30 or width > 30:
        return None

    anchor_cols = _assign_anchor_columns(counts, row_count, col_count)
    if anchor_cols is None:
        return None

    slot_map = _assign_slots(counts, row_count, col_count, anchor_cols)
    if slot_map is None:
        return None

    colors = random.sample(FRAME_COLORS, k=color_count)
    spec_groups: list[dict] = []
    for group_idx, (color, count) in enumerate(zip(colors, counts)):
        full_inner = _sample_full_motif(frame_side)
        subsets = _sample_subsets(full_inner, count)
        slots = slot_map[group_idx]
        frames = []
        for slot, inner_rel in zip(slots, subsets):
            sr, sc = slot
            top = row_tops[sr]
            left = col_tops[sc]
            frames.append(
                {
                    "top": top,
                    "left": left,
                    "inner_rel": frozenset(inner_rel),
                }
            )
        spec_groups.append(
            {
                "color": color,
                "count": count,
                "first_col": min(frame["left"] for frame in frames),
                "full_inner": full_inner,
                "frames": frames,
            }
        )

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for group in spec_groups:
        color = int(group["color"])
        for frame in group["frames"]:
            _paint_frame(
                grid,
                int(frame["top"]),
                int(frame["left"]),
                frame_side,
                color,
                frozenset(frame["inner_rel"]),
            )

    if not _validate_generated(grid, spec_groups, frame_side):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
