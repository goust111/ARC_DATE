from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, paste  # noqa: E402


BG = Color.BLACK
MARK = Color.RED
SEP = Color.GREEN
FILTER = Color.ORANGE
INPUT_COLORS = [Color.BLUE, Color.YELLOW, Color.GRAY, Color.PINK, Color.TEAL, Color.MAROON]

SLOT_ROWS = 2
SLOT_COLS = 4
MARGIN = 1
GAP = 1
MAX_ATTEMPTS = 400

BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _full_green_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r] == SEP))]


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    green_rows = _full_green_rows(grid)
    assert len(green_rows) == 1, "expected a unique full-width green separator row"
    sep = green_rows[0]
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    assert top.shape == bottom.shape and top.size > 0, "panels must be non-empty and equal-sized"
    return top.copy(), bottom.copy()


def _color_first_index(panel: np.ndarray, color: int) -> int:
    hits = np.argwhere(panel == int(color))
    assert hits.size > 0
    r, c = map(int, hits[0])
    return r * panel.shape[1] + c


def _rank_non_marker_colors(panel: np.ndarray) -> list[int]:
    colors = [int(c) for c in np.unique(panel) if int(c) not in {BG, MARK}]
    counts = {color: int(np.sum(panel == int(color))) for color in colors}
    first = {color: _color_first_index(panel, color) for color in colors}
    return sorted(colors, key=lambda color: (-counts[color], first[color]))


def _simplify_noisy_panel(panel: np.ndarray) -> np.ndarray:
    ranked = _rank_non_marker_colors(panel)
    assert len(ranked) >= 3, "noisy panel must contain at least three non-marker colors"
    out = panel.copy()
    mask = (out != BG) & (out != MARK)
    for color in ranked[2:]:
        out[(out == int(color)) & mask] = FILTER
    return out


def _overlay(quiet_panel: np.ndarray, noisy_panel_processed: np.ndarray) -> np.ndarray:
    out = quiet_panel.copy()
    only_noisy = (out == BG) & (noisy_panel_processed != BG)
    out[only_noisy] = noisy_panel_processed[only_noisy]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the noisier panel by color frequency, remove the separator, and overlay with quiet-panel collision priority.

    Input structure:
    - A black background grid is split into equal-height top and bottom panels by one full-width green separator row.
    - Each panel contains several disconnected foreground objects made of ordinary colors plus some red marker pixels embedded inside objects.
    - Local coordinates align between the two panels, so foreground cells in matching positions can collide after the separator is removed.

    Rule:
    - Count red marker pixels separately in the top and bottom panels; the panel with more markers is the noisy panel and the other is the quiet panel.
    - In the noisy panel, count only non-background, non-red colors; keep the two most frequent such colors unchanged and recolor every other such color to orange, while preserving red markers.
    - Delete the green separator row, then overlay the two panels by local coordinates: if both panels have foreground at a cell, keep the quiet-panel pixel; otherwise keep the lone foreground pixel, or black if both are background.

    Key insight:
    - Marker density drives two different decisions at once: the marker-heavier panel loses minor colors before overlay, while the marker-lighter panel wins every collision during overlay.
    """
    g = np.asarray(grid, dtype=int)
    top, bottom = _split_panels(g)
    top_markers = int(np.sum(top == MARK))
    bottom_markers = int(np.sum(bottom == MARK))
    assert top_markers != bottom_markers, "marker totals must distinguish noisy and quiet panels"

    if top_markers > bottom_markers:
        noisy, quiet = top, bottom
    else:
        noisy, quiet = bottom, top

    noisy_processed = _simplify_noisy_panel(noisy)
    return _overlay(quiet, noisy_processed)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


@lru_cache(maxsize=None)
def _shape_variants(slot_h: int, slot_w: int) -> tuple[np.ndarray, ...]:
    variants: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for base in BASE_SHAPES:
        for rot in range(4):
            rotated = np.rot90(base, rot)
            for flip in (False, True):
                mask = np.fliplr(rotated) if flip else rotated
                trimmed = _trim_mask(mask)
                h, w = map(int, trimmed.shape)
                area = int(trimmed.sum())
                if 3 <= area <= 8 and h <= slot_h and w <= slot_w:
                    key = ((h, w), tuple(int(x) for x in trimmed.flatten()))
                    variants[key] = trimmed.copy()
    return tuple(variants.values())


def _panel_geometry(slot_h: int, slot_w: int) -> tuple[int, int, list[tuple[int, int]]]:
    panel_h = 2 * MARGIN + SLOT_ROWS * slot_h + (SLOT_ROWS - 1) * GAP
    panel_w = 2 * MARGIN + SLOT_COLS * slot_w + (SLOT_COLS - 1) * GAP
    slots: list[tuple[int, int]] = []
    for sr in range(SLOT_ROWS):
        for sc in range(SLOT_COLS):
            r0 = MARGIN + sr * (slot_h + GAP)
            c0 = MARGIN + sc * (slot_w + GAP)
            slots.append((r0, c0))
    return panel_h, panel_w, slots


def _new_shape_spec(slot_h: int, slot_w: int) -> dict:
    mask = random.choice(_shape_variants(slot_h, slot_w)).copy()
    h, w = map(int, mask.shape)
    off_r = random.randint(0, slot_h - h)
    off_c = random.randint(0, slot_w - w)
    return {"mask": mask, "offset": (off_r, off_c)}


def _occupied_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _choose_marker_coords(mask: np.ndarray, marker_count: int) -> list[tuple[int, int]]:
    cells = _occupied_coords(mask)
    assert 0 <= marker_count < len(cells)
    if marker_count == 0:
        return []
    random.shuffle(cells)
    return sorted(cells[:marker_count])


def _render_panel(
    panel_shape: tuple[int, int],
    slots: list[tuple[int, int]],
    object_specs: list[dict],
) -> np.ndarray:
    panel = np.full(panel_shape, BG, dtype=int)
    for spec in object_specs:
        mask = spec["mask"]
        color = int(spec["color"])
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = color
        for r, c in spec["markers"]:
            patch[int(r), int(c)] = MARK
        slot_r, slot_c = slots[int(spec["slot"])]
        off_r, off_c = spec["offset"]
        paste(panel, patch, slot_r + int(off_r), slot_c + int(off_c), transparent=BG)
    return panel


def _sample_panel_colors(num_objects: int, distinct_colors: int) -> list[int]:
    palette = random.sample(INPUT_COLORS, distinct_colors)
    colors = palette.copy()
    while len(colors) < num_objects:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _distribute_markers(total: int, maxima: list[int]) -> list[int] | None:
    counts = [0] * len(maxima)
    available = [i for i, limit in enumerate(maxima) if limit > 0]
    while total > 0:
        choices = [i for i in available if counts[i] < maxima[i]]
        if not choices:
            return None
        idx = random.choice(choices)
        counts[idx] += 1
        total -= 1
    return counts


def _non_marker_counts(panel: np.ndarray) -> dict[int, int]:
    return {
        int(color): int(np.sum(panel == int(color)))
        for color in np.unique(panel)
        if int(color) not in {BG, MARK}
    }


def _build_panel_specs(
    slot_h: int,
    slot_w: int,
    chosen_slots: list[int],
    shared_shapes: dict[int, dict],
    force_different_from: dict[int, int] | None,
    marker_total: int,
) -> list[dict] | None:
    num_objects = len(chosen_slots)
    distinct_colors = random.randint(3, min(4, num_objects))
    object_colors = _sample_panel_colors(num_objects, distinct_colors)

    if force_different_from:
        for idx, slot in enumerate(chosen_slots):
            other_color = force_different_from.get(int(slot))
            if other_color is None or object_colors[idx] != other_color:
                continue
            replacements = [color for color in INPUT_COLORS if color != other_color]
            object_colors[idx] = random.choice(replacements)

    specs: list[dict] = []
    maxima: list[int] = []
    for idx, slot in enumerate(chosen_slots):
        shape_spec = shared_shapes.get(int(slot), _new_shape_spec(slot_h, slot_w))
        area = int(shape_spec["mask"].sum())
        max_markers = min(3, area - 1)
        specs.append(
            {
                "slot": int(slot),
                "mask": shape_spec["mask"].copy(),
                "offset": tuple(map(int, shape_spec["offset"])),
                "color": int(object_colors[idx]),
            }
        )
        maxima.append(int(max_markers))

    marker_counts = _distribute_markers(marker_total, maxima)
    if marker_counts is None:
        return None

    for spec, marker_count in zip(specs, marker_counts):
        spec["markers"] = _choose_marker_coords(spec["mask"], int(marker_count))
    return specs


def _marker_capacity(specs: list[dict]) -> int:
    return sum(min(3, int(spec["mask"].sum()) - 1) for spec in specs)


def _shape_map_for_slots(
    chosen_slots: list[int],
    slot_h: int,
    slot_w: int,
    preset: dict[int, dict] | None = None,
) -> dict[int, dict]:
    shape_map: dict[int, dict] = {}
    for slot in chosen_slots:
        if preset and int(slot) in preset:
            source = preset[int(slot)]
            shape_map[int(slot)] = {
                "mask": source["mask"].copy(),
                "offset": tuple(map(int, source["offset"])),
            }
        else:
            shape_map[int(slot)] = _new_shape_spec(slot_h, slot_w)
    return shape_map


def _panel_is_valid(panel: np.ndarray, expected_objects: int, require_clear_ranking: bool) -> bool:
    objects = find_components(panel, bg=BG, connectivity=4, monochromatic=False)
    if len(objects) != expected_objects:
        return False
    if any(not (3 <= int(obj["area"]) <= 8) for obj in objects):
        return False

    colors = [int(color) for color in np.unique(panel) if int(color) not in {BG, MARK}]
    if not (3 <= len(colors) <= 4):
        return False

    counts = sorted(_non_marker_counts(panel).values(), reverse=True)
    if len(counts) < 3 or counts[2] < 2:
        return False
    if require_clear_ranking and len(set(counts[:3])) < 3:
        return False
    return True


def _build_input(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    sep_row = np.full((1, top.shape[1]), SEP, dtype=int)
    return np.vstack([top, sep_row, bottom])


def _priority_variant(quiet: np.ndarray, noisy_processed: np.ndarray) -> np.ndarray:
    out = noisy_processed.copy()
    only_quiet = (out == BG) & (quiet != BG)
    out[only_quiet] = quiet[only_quiet]
    return out


def _semantic_checks(inp: np.ndarray, quiet_panel: np.ndarray, noisy_panel: np.ndarray) -> bool:
    out = main(inp)
    noisy_processed = _simplify_noisy_panel(noisy_panel)
    collisions = int(np.sum((quiet_panel != BG) & (noisy_panel != BG)))
    if collisions < 3:
        return False

    if not np.array_equal(out, _overlay(quiet_panel, noisy_processed)):
        return False

    if np.array_equal(out, _overlay(quiet_panel, noisy_panel)):
        return False
    if np.array_equal(out, _priority_variant(quiet_panel, noisy_processed)):
        return False

    visible_filter = (quiet_panel == BG) & (noisy_panel != noisy_processed) & (noisy_processed == FILTER)
    if not bool(np.any(visible_filter)):
        return False

    collision_difference = (
        (quiet_panel != BG)
        & (noisy_processed != BG)
        & (quiet_panel != noisy_processed)
    )
    if not bool(np.any(collision_difference)):
        return False

    quiet_marks = int(np.sum(quiet_panel == MARK))
    noisy_marks = int(np.sum(noisy_panel == MARK))
    if noisy_marks <= quiet_marks:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    slot_h = random.choice([3, 4])
    slot_w = random.choice([4, 5])
    panel_h, panel_w, slots = _panel_geometry(slot_h, slot_w)

    quiet_count = random.randint(3, 5)
    noisy_count = random.randint(3, 5)
    min_common = max(1, quiet_count + noisy_count - (SLOT_ROWS * SLOT_COLS))
    max_common = min(2, quiet_count, noisy_count)
    common_count = random.randint(min_common, max_common)

    all_slots = list(range(SLOT_ROWS * SLOT_COLS))
    common_slots = random.sample(all_slots, common_count)
    remaining = [slot for slot in all_slots if slot not in common_slots]
    quiet_unique_slots = random.sample(remaining, quiet_count - common_count)
    remaining_for_noisy = [slot for slot in remaining if slot not in quiet_unique_slots]
    noisy_unique_slots = random.sample(remaining_for_noisy, noisy_count - common_count)
    quiet_slots = common_slots + quiet_unique_slots
    noisy_slots = common_slots + noisy_unique_slots
    random.shuffle(quiet_slots)
    random.shuffle(noisy_slots)

    shared_shapes = {int(slot): _new_shape_spec(slot_h, slot_w) for slot in common_slots}
    quiet_shape_map = _shape_map_for_slots(quiet_slots, slot_h, slot_w, preset=shared_shapes)
    quiet_capacity = sum(min(3, int(spec["mask"].sum()) - 1) for spec in quiet_shape_map.values())
    if quiet_capacity < 1:
        return None
    quiet_marker_total = random.randint(1, min(4, quiet_capacity))

    quiet_specs = _build_panel_specs(
        slot_h,
        slot_w,
        quiet_slots,
        quiet_shape_map,
        force_different_from=None,
        marker_total=quiet_marker_total,
    )
    if quiet_specs is None:
        return None

    quiet_color_by_slot = {int(spec["slot"]): int(spec["color"]) for spec in quiet_specs}
    noisy_shapes = {int(slot): quiet_shape_map[int(slot)] for slot in common_slots}
    noisy_shape_map = _shape_map_for_slots(noisy_slots, slot_h, slot_w, preset=noisy_shapes)
    noisy_capacity = sum(min(3, int(spec["mask"].sum()) - 1) for spec in noisy_shape_map.values())
    if noisy_capacity < quiet_marker_total + 1:
        return None
    noisy_marker_total = random.randint(quiet_marker_total + 1, min(quiet_marker_total + 3, noisy_capacity))
    noisy_specs = _build_panel_specs(
        slot_h,
        slot_w,
        noisy_slots,
        noisy_shape_map,
        force_different_from=quiet_color_by_slot,
        marker_total=noisy_marker_total,
    )
    if noisy_specs is None:
        return None

    quiet_panel = _render_panel((panel_h, panel_w), slots, quiet_specs)
    noisy_panel = _render_panel((panel_h, panel_w), slots, noisy_specs)
    if not _panel_is_valid(quiet_panel, quiet_count, require_clear_ranking=False):
        return None
    if not _panel_is_valid(noisy_panel, noisy_count, require_clear_ranking=True):
        return None

    if random.random() < 0.5:
        top, bottom = quiet_panel, noisy_panel
    else:
        top, bottom = noisy_panel, quiet_panel

    inp = _build_input(top, bottom)
    if max(inp.shape) > 30:
        return None

    if top is noisy_panel:
        quiet_for_checks, noisy_for_checks = bottom, top
    else:
        quiet_for_checks, noisy_for_checks = top, bottom

    if not _semantic_checks(inp, quiet_for_checks, noisy_for_checks):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-overlay sample")
