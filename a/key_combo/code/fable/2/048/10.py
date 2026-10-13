from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
BLOCK_COLOR = Color.GREY
CORNER_TO_LOCAL = {
    0: (0, 0),   # top-left
    1: (0, -1),  # top-right
    2: (-1, 0),  # bottom-left
    3: (-1, -1),  # bottom-right
}

FRAME_COUNT_RANGE = (3, 5)
PAIR_ODD_SIDES = (3, 5)
PAIR_EVEN_SIDES = (4,)
EXTRA_SIDE_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 500
MAX_EXTRA_SPEC_ATTEMPTS = 120


def _corner_point(height: int, width: int, corner_idx: int) -> tuple[int, int]:
    row, col = CORNER_TO_LOCAL[int(corner_idx)]
    return (row if row >= 0 else height + row, col if col >= 0 else width + col)


def _phase_mask(inner_h: int, inner_w: int, corner_idx: int) -> np.ndarray:
    sr, sc = _corner_point(inner_h, inner_w, corner_idx)
    rr, cc = np.indices((inner_h, inner_w))
    return ((rr - sr) + (cc - sc)) % 2 == 0


def _corner_phase_class(inner_h: int, inner_w: int, corner_idx: int) -> int:
    sr, sc = _corner_point(inner_h, inner_w, corner_idx)
    return int((sr + sc) % 2)


def _dilate8(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(int(radius)):
        padded = np.pad(out, 1, constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        h, w = out.shape
        for dr in range(3):
            for dc in range(3):
                grown |= padded[dr : dr + h, dc : dc + w]
        out = grown
    return out


def _pick_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    pool = list(cells)
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    used = set(chosen)
    for cell in pool:
        if cell in used:
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    raise ValueError("not enough cells to sample")


def _make_spec(
    inner_h: int,
    inner_w: int,
    corner_idx: int,
    active_blocks: int,
    inactive_blocks: int,
) -> dict:
    phase = _phase_mask(inner_h, inner_w, corner_idx)
    active_cells = [tuple(map(int, coord)) for coord in np.argwhere(phase)]
    inactive_cells = [tuple(map(int, coord)) for coord in np.argwhere(~phase)]
    chosen_active = _pick_cells(active_cells, active_blocks)
    chosen_inactive = _pick_cells(inactive_cells, inactive_blocks)

    blocked_mask = np.zeros((inner_h, inner_w), dtype=bool)
    for rr, cc in chosen_active + chosen_inactive:
        blocked_mask[rr, cc] = True

    return {
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "corner_idx": int(corner_idx),
        "blocked_local": tuple(sorted(chosen_active + chosen_inactive)),
        "blocked_mask": blocked_mask,
        "active_blocks": int(active_blocks),
        "inactive_blocks": int(inactive_blocks),
    }


def _spec_capacity(spec: dict, *, phase_mode: str = "marker", block_mode: str = "phase") -> int:
    corner_idx = spec["corner_idx"] if phase_mode == "marker" else 0
    phase = _phase_mask(spec["inner_h"], spec["inner_w"], corner_idx)
    if block_mode == "none":
        blocked = 0
    elif block_mode == "all":
        blocked = int(spec["blocked_mask"].sum())
    else:
        blocked = int(np.count_nonzero(phase & spec["blocked_mask"]))
    return int(np.count_nonzero(phase) - blocked)


def _frame_patch(spec: dict, body_color: int, marker_color: int) -> np.ndarray:
    patch = np.full((spec["inner_h"] + 2, spec["inner_w"] + 2), BG, dtype=int)
    patch[0, :] = body_color
    patch[-1, :] = body_color
    patch[:, 0] = body_color
    patch[:, -1] = body_color

    mr, mc = _corner_point(patch.shape[0], patch.shape[1], spec["corner_idx"])
    patch[mr, mc] = marker_color

    for rr, cc in spec["blocked_local"]:
        patch[rr + 1, cc + 1] = BLOCK_COLOR
    return patch


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    components = find_components(
        grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"
    )
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        color = int(comp["color"])
        mask = crop_bbox(grid == color, comp["bbox"])
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if np.any(mask & ~border):
            continue
        missing = border & ~mask
        if int(missing.sum()) != 1:
            continue

        mr, mc = map(int, np.argwhere(missing)[0])
        if (mr, mc) not in {
            (0, 0),
            (0, w - 1),
            (h - 1, 0),
            (h - 1, w - 1),
        }:
            continue

        marker_color = int(grid[r0 + mr, c0 + mc])
        if marker_color in (bg, color):
            continue

        corner_idx = {
            (0, 0): 0,
            (0, w - 1): 1,
            (h - 1, 0): 2,
            (h - 1, w - 1): 3,
        }[(mr, mc)]
        blocked_mask = (grid[r0 + 1 : r1, c0 + 1 : c1] != bg)

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "body_color": color,
                "marker_color": marker_color,
                "corner_idx": corner_idx,
                "inner_h": h - 2,
                "inner_w": w - 2,
                "blocked_mask": blocked_mask,
            }
        )

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _frame_capacity(frame: dict, *, phase_mode: str = "marker", block_mode: str = "phase") -> int:
    corner_idx = frame["corner_idx"] if phase_mode == "marker" else 0
    phase = _phase_mask(frame["inner_h"], frame["inner_w"], corner_idx)
    if block_mode == "none":
        blocked = 0
    elif block_mode == "all":
        blocked = int(frame["blocked_mask"].sum())
    else:
        blocked = int(np.count_nonzero(phase & frame["blocked_mask"]))
    return int(np.count_nonzero(phase) - blocked)


def _bbox_union_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        assert not mask[r0 : r1 + 1, c0 : c1 + 1].any()
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _solve_grid(
    grid: np.ndarray,
    *,
    count_phase_mode: str,
    fill_phase_mode: str,
    block_count_mode: str,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frames = _detect_frames(g, bg)
    assert len(frames) >= 2, "expected multiple corner-marked frames"

    bbox_mask = _bbox_union_mask(g.shape, frames)
    outside = g[(~bbox_mask) & (g != bg)]
    outside_counts = Counter(map(int, outside.tolist()))
    assert len(outside_counts) == len(frames), "expected one scatter color per frame"
    count_to_color: dict[int, int] = {}
    for color, count in outside_counts.items():
        assert count not in count_to_color, "scatter counts must be unique"
        count_to_color[int(count)] = int(color)

    capacities: list[int] = []
    for frame in frames:
        capacity = _frame_capacity(
            frame, phase_mode=count_phase_mode, block_mode=block_count_mode
        )
        capacities.append(capacity)
    assert len(set(capacities)) == len(capacities), "frame capacities must be unique"
    assert set(capacities) == set(count_to_color), "scatter counts must match capacities"

    out = g.copy()
    out[~bbox_mask] = bg
    for frame, capacity in zip(frames, capacities):
        r0, c0, r1, c1 = frame["bbox"]
        corner_idx = frame["corner_idx"] if fill_phase_mode == "marker" else 0
        fill_mask = _phase_mask(frame["inner_h"], frame["inner_w"], corner_idx) & ~frame["blocked_mask"]
        out[r0 + 1 : r1, c0 + 1 : c1][fill_mask] = count_to_color[capacity]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marked rectangular cage with the outside color whose pixel count matches its corner-phased checkerboard capacity.

    Input structure:
    - The grid has several one-cell-thick hollow rectangles on a background; each frame is one body color except for one corner marker cell of a different color.
    - Inside every frame are some grey blocked cells, while outside the frames are isolated singleton scatter pixels in several other colors.
    - At least two frames share the same inner size but differ in corner marker position and blocker layout.

    Rule:
    - For each frame, start the checkerboard phase from the interior corner cell adjacent to the marked border corner.
    - Its effective capacity is the number of cells on that phase that are not occupied by grey blockers.
    - Erase all outside scatter pixels, then recolor the unblocked phase cells of the unique frame whose capacity equals each scatter color's pixel count; keep the frame border, marker corner, and blockers unchanged.

    Key insight:
    - The corner marker selects which checkerboard parity is usable, so the same-size cavity can hold a different number of cells once blocker overlap is counted on that phase only.
    """
    return _solve_grid(
        grid,
        count_phase_mode="marker",
        fill_phase_mode="marker",
        block_count_mode="phase",
    )


def _sample_pair_specs() -> list[dict]:
    if random.random() < 0.5:
        inner_h = random.choice(PAIR_ODD_SIDES)
        inner_w = random.choice(PAIR_EVEN_SIDES)
    else:
        inner_h = random.choice(PAIR_EVEN_SIDES)
        inner_w = random.choice(PAIR_ODD_SIDES)

    phase0_corners = [
        corner_idx
        for corner_idx in range(4)
        if _corner_phase_class(inner_h, inner_w, corner_idx) == 0
    ]
    phase1_corners = [
        corner_idx
        for corner_idx in range(4)
        if _corner_phase_class(inner_h, inner_w, corner_idx) == 1
    ]
    corner_a = random.choice(phase0_corners)
    corner_b = random.choice(phase1_corners)

    base_capacity = int(np.count_nonzero(_phase_mask(inner_h, inner_w, corner_a)))
    active_a = random.randint(1, min(2, base_capacity - 2))
    active_b = active_a + 1

    inactive_a = random.randint(0, 1)
    inactive_b = random.randint(0, 1)
    if inactive_a == inactive_b == 0:
        inactive_b = 1

    return [
        _make_spec(inner_h, inner_w, corner_a, active_a, inactive_a),
        _make_spec(inner_h, inner_w, corner_b, active_b, inactive_b),
    ]


def _sample_extra_spec(used_capacities: set[int], *, require_inactive_only: bool) -> dict | None:
    for _ in range(MAX_EXTRA_SPEC_ATTEMPTS):
        inner_h = random.randint(EXTRA_SIDE_RANGE[0], EXTRA_SIDE_RANGE[1])
        inner_w = random.randint(EXTRA_SIDE_RANGE[0], EXTRA_SIDE_RANGE[1])
        corner_idx = random.randint(0, 3)
        phase_cells = int(np.count_nonzero(_phase_mask(inner_h, inner_w, corner_idx)))
        inactive_cells = inner_h * inner_w - phase_cells
        if inactive_cells <= 0:
            continue

        if require_inactive_only:
            active_blocks = 0
            inactive_blocks = random.randint(1, min(2, inactive_cells))
        else:
            max_active = min(2, phase_cells - 2)
            if max_active < 0:
                continue
            active_blocks = random.randint(0, max_active)
            inactive_blocks = random.randint(0, min(2, inactive_cells))
            if active_blocks == inactive_blocks == 0:
                inactive_blocks = 1

        spec = _make_spec(inner_h, inner_w, corner_idx, active_blocks, inactive_blocks)
        capacity = _spec_capacity(spec)
        if 2 <= capacity <= 14 and capacity not in used_capacities:
            return spec
    return None


def _sample_specs() -> list[dict] | None:
    frame_count = random.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    specs = _sample_pair_specs()
    capacities = {_spec_capacity(spec) for spec in specs}
    if len(capacities) != 2:
        return None

    first_extra = _sample_extra_spec(capacities, require_inactive_only=True)
    if first_extra is None:
        return None
    specs.append(first_extra)
    capacities.add(_spec_capacity(first_extra))

    while len(specs) < frame_count:
        extra = _sample_extra_spec(capacities, require_inactive_only=False)
        if extra is None:
            return None
        specs.append(extra)
        capacities.add(_spec_capacity(extra))

    random.shuffle(specs)
    return specs


def _specs_have_required_pair(specs: list[dict]) -> bool:
    capacities = [_spec_capacity(spec) for spec in specs]
    for i, spec_a in enumerate(specs):
        for j in range(i + 1, len(specs)):
            spec_b = specs[j]
            if (spec_a["inner_h"], spec_a["inner_w"]) != (spec_b["inner_h"], spec_b["inner_w"]):
                continue
            if abs(capacities[i] - capacities[j]) != 1:
                continue
            if _corner_phase_class(spec_a["inner_h"], spec_a["inner_w"], spec_a["corner_idx"]) != _corner_phase_class(
                spec_b["inner_h"], spec_b["inner_w"], spec_b["corner_idx"]
            ):
                return True
    return False


def _choose_colors(frame_count: int) -> tuple[int, int, list[int]]:
    palette = [color for color in Color.NOT_BLACK if color != BLOCK_COLOR]
    body_color, marker_color = random.sample(palette, 2)
    fill_palette = [color for color in palette if color not in {body_color, marker_color}]
    fill_colors = random.sample(fill_palette, frame_count)
    return int(body_color), int(marker_color), [int(color) for color in fill_colors]


def _place_frames(
    shape: tuple[int, int],
    specs: list[dict],
    body_color: int,
    marker_color: int,
) -> tuple[np.ndarray, list[dict]] | None:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=int)
    records = [None] * len(specs)

    order = sorted(
        range(len(specs)),
        key=lambda idx: (specs[idx]["inner_h"] + 2) * (specs[idx]["inner_w"] + 2),
        reverse=True,
    )
    for idx in order:
        spec = specs[idx]
        patch = _frame_patch(spec, body_color, marker_color)
        rect = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                occupied, rect, transparent=0, border=1, padding=1, padding_connectivity=8
            )
        except ValueError:
            return None

        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=BG)
        records[idx] = {
            **spec,
            "bbox": (int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1)),
        }

    return grid, records  # type: ignore[return-value]


def _place_scatter_points(
    grid: np.ndarray,
    records: list[dict],
    fill_colors: list[int],
) -> np.ndarray | None:
    out = grid.copy()
    bbox_mask = np.zeros_like(out, dtype=bool)
    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    forbidden = _dilate8(bbox_mask, radius=1)
    items = list(zip(fill_colors, records))
    items.sort(key=lambda item: _spec_capacity(item[1]), reverse=True)
    for color, record in items:
        capacity = _spec_capacity(record)
        for _ in range(capacity):
            candidates = np.argwhere((out == BG) & ~forbidden)
            if candidates.size == 0:
                return None
            rr, cc = map(int, candidates[random.randrange(len(candidates))])
            out[rr, cc] = int(color)
            point = np.zeros_like(forbidden, dtype=bool)
            point[rr, cc] = True
            forbidden |= _dilate8(point, radius=1)
    return out


def _render_expected(grid: np.ndarray, records: list[dict], fill_colors: list[int]) -> np.ndarray:
    out = grid.copy()
    bbox_mask = np.zeros_like(out, dtype=bool)
    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    out[~bbox_mask] = BG

    for record, color in zip(records, fill_colors):
        r0, c0, r1, c1 = record["bbox"]
        fill_mask = _phase_mask(record["inner_h"], record["inner_w"], record["corner_idx"]) & ~record["blocked_mask"]
        out[r0 + 1 : r1, c0 + 1 : c1][fill_mask] = int(color)
    return out


def _variant_differs(grid: np.ndarray, expected: np.ndarray, **kwargs) -> bool:
    try:
        variant = _solve_grid(grid, **kwargs)
    except AssertionError:
        return True
    return not np.array_equal(variant, expected)


def _validate_instance(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if int(get_bg(grid)) != BG:
        return False
    if len(_detect_frames(grid, BG)) != len(specs):
        return False
    if not _specs_have_required_pair(specs):
        return False
    if not any(spec["active_blocks"] > 0 for spec in specs):
        return False
    if not any(spec["inactive_blocks"] > 0 for spec in specs):
        return False

    capacities = [_spec_capacity(spec) for spec in specs]
    if len(set(capacities)) != len(capacities):
        return False

    if not np.array_equal(main(grid), expected):
        return False

    if not _variant_differs(
        grid,
        expected,
        count_phase_mode="top_left",
        fill_phase_mode="top_left",
        block_count_mode="phase",
    ):
        return False
    if not _variant_differs(
        grid,
        expected,
        count_phase_mode="marker",
        fill_phase_mode="marker",
        block_count_mode="none",
    ):
        return False
    if not _variant_differs(
        grid,
        expected,
        count_phase_mode="marker",
        fill_phase_mode="marker",
        block_count_mode="all",
    ):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    total_scatter = sum(_spec_capacity(spec) for spec in specs)
    reserve_area = sum((spec["inner_h"] + 4) * (spec["inner_w"] + 4) for spec in specs)
    min_side = max(18, int(np.ceil(np.sqrt(total_scatter * 4 + reserve_area))))
    if min_side > 30:
        return None

    body_color, marker_color, fill_colors = _choose_colors(len(specs))
    for _ in range(40):
        height = random.randint(min_side, 30)
        width = random.randint(min_side, 30)
        placed = _place_frames((height, width), specs, body_color, marker_color)
        if placed is None:
            continue
        frame_grid, records = placed
        grid = _place_scatter_points(frame_grid, records, fill_colors)
        if grid is None:
            continue
        expected = _render_expected(grid, records, fill_colors)
        if _validate_instance(grid, expected, specs):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-phased checkerboard cage task")
