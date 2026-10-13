from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if not _ARC_UTILS_DIR.exists():
    raise ImportError(f"arc_utils directory not found: {_ARC_UTILS_DIR}")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import (  # type: ignore
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    paste,
    random_free_location,
    random_sprite,
    scale_sprite,
)


BG = 0
SEPARATOR_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
TRANSFORMS = ("hflip", "vflip", "rot180", "rot90")
PANEL_COUNT_RANGE = (2, 4)
MATCHED_PER_PANEL = (1, 2)
DECOY_COUNT_RANGE = (0, 2)
GEN_RETRIES = 400


def _transform_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.fliplr(mask)
    if kind == "vflip":
        return np.flipud(mask)
    if kind == "rot180":
        return np.rot90(mask, 2)
    if kind == "rot90":
        return np.rot90(mask, 1)
    if kind == "rot270":
        return np.rot90(mask, 3)
    raise ValueError(f"unknown transform: {kind}")


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _scaled_binary(mask: np.ndarray, factor: int) -> np.ndarray:
    return scale_sprite(mask.astype(int), int(factor)).astype(bool)


def _match_scaled(patch_mask: np.ndarray, exemplar_mask: np.ndarray) -> int | None:
    ph, pw = map(int, patch_mask.shape)
    eh, ew = map(int, exemplar_mask.shape)
    if ph % eh != 0 or pw % ew != 0:
        return None
    factor_h = ph // eh
    factor_w = pw // ew
    if factor_h != factor_w or factor_h <= 0:
        return None
    factor = int(factor_h)
    return factor if np.array_equal(patch_mask, _scaled_binary(exemplar_mask, factor)) else None


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(1, w - 1):
        column = grid[:, c]
        vals = {int(v) for v in column}
        if bg in vals or len(vals) != 1:
            continue
        if not (grid[:, :c] != bg).any():
            continue
        if not (grid[:, c + 1 :] != bg).any():
            continue
        candidates.append(int(c))
    if not candidates:
        raise ValueError("separator column not found")
    return int(candidates[0])


def _extract_panels(left_region: np.ndarray, bg: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    comps = find_components(
        left_region,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    panels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for color, group in by_color.items():
        if len(group) != 2:
            continue
        group.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
        states: list[np.ndarray] = []
        for comp in group:
            patch = crop_bbox(left_region, comp["bbox"])
            states.append((patch == int(color)))
        if np.array_equal(states[0], states[1]):
            continue
        panels[int(color)] = (states[0], states[1])
    return panels


def _toggle_component(
    out: np.ndarray,
    source: np.ndarray,
    comp: dict,
    states: tuple[np.ndarray, np.ndarray],
) -> None:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(source, comp["bbox"])
    patch_mask = patch == color

    factor0 = _match_scaled(patch_mask, states[0])
    factor1 = _match_scaled(patch_mask, states[1])
    if (factor0 is None) == (factor1 is None):
        return

    if factor0 is not None:
        factor = int(factor0)
        target_state = states[1]
    else:
        factor = int(factor1)
        target_state = states[0]

    replacement_mask = _scaled_binary(target_state, factor)
    replacement = _render_mask(replacement_mask, color)
    if replacement.shape != patch.shape:
        raise ValueError("replacement shape must preserve object bbox")

    out[r0 : r1 + 1, c0 : c1 + 1] = int(get_bg(source))
    paste(out, replacement, r0, c0, transparent=BG)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Toggle each same-colored right-side object to the other demonstrated pose and drop the left code panels.

    Input structure:
    - A full-height vertical separator splits the grid into a narrow left column and a wider right region.
    - In the left column, each panel color appears exactly twice: two disconnected same-colored examples of one asymmetric shape in two poses.
    - In the right region, monochrome objects may share a panel color and match one of that panel's poses, possibly after integer scaling; some colors have no panel.

    Rule:
    - For each right-region object whose color has a left-panel pair, identify which of the two demonstrated poses it currently matches.
    - Replace that object in the same bounding box with the other demonstrated pose, keeping its color and integer scale.
    - Objects whose colors do not appear in the left column stay unchanged, and the output is only the processed right region.

    Key insight:
    - The left pair is a bidirectional switch rather than a left-to-right instruction: the object's current pose determines which example is "source" and which is the required "other" state.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = int(get_bg(grid))
    sep_col = _find_separator_col(grid, bg)
    left_region = grid[:, :sep_col]
    right_region = grid[:, sep_col + 1 :]
    panels = _extract_panels(left_region, bg)

    out = right_region.copy()
    comps = find_components(
        right_region,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        color = int(comp["color"])
        if color not in panels:
            continue
        _toggle_component(out, right_region, comp, panels[color])
    return out


def _mask_is_asymmetric(mask: np.ndarray) -> bool:
    checks = ("hflip", "vflip", "rot180")
    if mask.shape[0] == mask.shape[1]:
        checks += ("rot90", "rot270")
    return all(not np.array_equal(mask, _transform_mask(mask, kind)) for kind in checks)


def _sample_mask(kind: str) -> np.ndarray:
    for _ in range(200):
        if kind == "rot90":
            side = random.randint(3, 5)
            shape = ([side], [side])
        else:
            h = random.randint(3, 5)
            w = random.randint(3, 5)
            shape = ([h], [w])
        sprite = random_sprite(
            shape[0],
            shape[1],
            density=random.uniform(0.42, 0.68),
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=4,
            background=BG,
        )
        mask = sprite != BG
        if not is_contiguous(mask.astype(int), background=BG, connectivity=4):
            continue
        if not _mask_is_asymmetric(mask):
            continue
        transformed = _transform_mask(mask, kind)
        if transformed.shape != mask.shape:
            continue
        if np.array_equal(mask, transformed):
            continue
        return mask
    raise RuntimeError(f"failed to sample mask for {kind}")


def _sample_panel_specs(panel_colors: list[int]) -> list[dict]:
    panel_count = len(panel_colors)
    transform_kinds = random.sample(list(TRANSFORMS), k=min(panel_count, len(TRANSFORMS)))
    while len(transform_kinds) < panel_count:
        transform_kinds.append(random.choice(TRANSFORMS))
    random.shuffle(transform_kinds)

    specs: list[dict] = []
    for color, kind in zip(panel_colors, transform_kinds):
        base = _sample_mask(kind)
        other = _transform_mask(base, kind)
        states = (base, other) if random.random() < 0.5 else (other, base)
        specs.append({"color": int(color), "kind": kind, "states": states})
    return specs


def _allowed_scales(mask: np.ndarray) -> list[int]:
    scales = [1]
    if max(mask.shape) <= 5:
        scales.append(2)
    if max(mask.shape) <= 3:
        scales.append(3)
    return scales


def _sample_decoy_patch(color: int) -> np.ndarray:
    for _ in range(100):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        sprite = random_sprite(
            [h],
            [w],
            density=random.uniform(0.45, 0.72),
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=4,
            background=BG,
        )
        mask = sprite != BG
        if not _mask_is_asymmetric(mask):
            continue
        scale = random.choice(_allowed_scales(mask))
        return _render_mask(_scaled_binary(mask, scale), color)
    raise RuntimeError("failed to sample decoy")


def _sample_objects(panel_specs: list[dict], decoy_colors: list[int]) -> list[dict]:
    objects: list[dict] = []
    matched_records: list[dict] = []

    for spec in panel_specs:
        count = random.randint(*MATCHED_PER_PANEL)
        orientations = [random.randint(0, 1) for _ in range(count)]
        if count == 2 and random.random() < 0.7:
            orientations = [0, 1]
            random.shuffle(orientations)
        for state_index in orientations:
            mask = spec["states"][state_index]
            factor = random.choice(_allowed_scales(mask))
            patch = _render_mask(_scaled_binary(mask, factor), spec["color"])
            expected = _render_mask(_scaled_binary(spec["states"][1 - state_index], factor), spec["color"])
            record = {
                "color": int(spec["color"]),
                "patch": patch,
                "expected": expected,
                "matched": True,
                "state_index": int(state_index),
                "factor": int(factor),
            }
            objects.append(record)
            matched_records.append(record)

    if matched_records and len({int(obj["state_index"]) for obj in matched_records}) == 1:
        pivot = matched_records[0]
        pivot["state_index"] = 1 - int(pivot["state_index"])
        mask = next(spec["states"][pivot["state_index"]] for spec in panel_specs if int(spec["color"]) == int(pivot["color"]))
        factor = int(pivot["factor"])
        pivot["patch"] = _render_mask(_scaled_binary(mask, factor), int(pivot["color"]))
        pivot["expected"] = _render_mask(
            _scaled_binary(next(spec["states"][1 - pivot["state_index"]] for spec in panel_specs if int(spec["color"]) == int(pivot["color"])), factor),
            int(pivot["color"]),
        )

    for color in decoy_colors:
        patch = _sample_decoy_patch(int(color))
        objects.append(
            {
                "color": int(color),
                "patch": patch,
                "expected": patch.copy(),
                "matched": False,
                "state_index": None,
                "factor": None,
            }
        )

    random.shuffle(objects)
    return objects


def _place_panels(height: int, width: int, panel_specs: list[dict]) -> np.ndarray:
    canvas = np.full((height, width), BG, dtype=int)
    pair_gap = random.randint(1, 2)
    row_gap = random.randint(1, 2)
    panel_heights = [max(spec["states"][0].shape[0], spec["states"][1].shape[0]) for spec in panel_specs]
    stack_h = int(sum(panel_heights) + row_gap * (len(panel_specs) - 1))
    top = random.randint(0, max(0, height - stack_h))
    row = top
    for spec, panel_h in zip(panel_specs, panel_heights):
        left_mask, right_mask = spec["states"]
        pair_w = int(left_mask.shape[1] + pair_gap + right_mask.shape[1])
        col = (width - pair_w) // 2
        r_left = row + (panel_h - left_mask.shape[0]) // 2
        r_right = row + (panel_h - right_mask.shape[0]) // 2
        paste(canvas, _render_mask(left_mask, spec["color"]), r_left, col, transparent=BG)
        paste(
            canvas,
            _render_mask(right_mask, spec["color"]),
            r_right,
            col + left_mask.shape[1] + pair_gap,
            transparent=BG,
        )
        row += int(panel_h + row_gap)
    return canvas


def _place_objects(height: int, width: int, objects: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    ordered = sorted(
        objects,
        key=lambda obj: (int(obj["patch"].shape[0]) * int(obj["patch"].shape[1]), int(obj["matched"])),
        reverse=True,
    )
    for _ in range(30):
        inp = np.full((height, width), BG, dtype=int)
        out = np.full((height, width), BG, dtype=int)
        reserve = np.full((height, width), BG, dtype=int)
        placements: list[dict] = []
        try:
            for obj in ordered:
                patch = obj["patch"]
                expected = obj["expected"]
                footprint = np.ones(patch.shape, dtype=int)
                loc = random_free_location(
                    reserve,
                    footprint,
                    transparent=BG,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
                r0, c0 = map(int, loc)
                paste(reserve, footprint, r0, c0, transparent=BG)
                paste(inp, patch, r0, c0, transparent=BG)
                paste(out, expected, r0, c0, transparent=BG)
                placements.append(
                    {
                        "color": int(obj["color"]),
                        "bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1),
                        "matched": bool(obj["matched"]),
                    }
                )
        except ValueError:
            continue
        return inp, out, placements
    raise ValueError("failed to place objects")


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    decoy_count = random.randint(*DECOY_COUNT_RANGE)
    if panel_count + decoy_count > len(PALETTE):
        raise RuntimeError("palette exhausted")

    colors = random.sample(PALETTE, k=panel_count + decoy_count)
    panel_specs = _sample_panel_specs(colors[:panel_count])
    objects = _sample_objects(panel_specs, colors[panel_count:])

    left_width = max(
        int(spec["states"][0].shape[1] + spec["states"][1].shape[1] + 2 + 2)
        for spec in panel_specs
    )
    panel_stack_min_h = sum(
        max(spec["states"][0].shape[0], spec["states"][1].shape[0]) for spec in panel_specs
    ) + (len(panel_specs) - 1) + 2
    max_patch_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_patch_w = max(int(obj["patch"].shape[1]) for obj in objects)

    min_height = max(12, int(panel_stack_min_h), int(max_patch_h + 2))
    if min_height > 30:
        raise RuntimeError("height budget exceeded")
    height = random.randint(min_height, min(30, min_height + 8))

    right_width_min = max(12, int(max_patch_w + 2))
    right_width_max = 30 - left_width - 1
    if right_width_min > right_width_max:
        raise RuntimeError("width budget exceeded")
    right_width = random.randint(max(right_width_min, right_width_max - 4), right_width_max)
    total_width = int(left_width + 1 + right_width)

    left_region = _place_panels(height, left_width, panel_specs)
    right_input, right_output, placements = _place_objects(height, right_width, objects)
    separator = np.full((height, 1), SEPARATOR_COLOR, dtype=int)
    input_grid = np.hstack([left_region, separator, right_input])

    meta = {
        "panel_specs": panel_specs,
        "placements": placements,
        "matched_count": sum(1 for obj in objects if obj["matched"]),
        "decoy_count": sum(1 for obj in objects if not obj["matched"]),
    }
    return input_grid, right_output, meta


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected, meta = _build_case()
        except (RuntimeError, ValueError):
            continue
        out = main(grid)
        if out.shape != expected.shape:
            continue
        if not np.array_equal(out, expected):
            continue
        if meta["matched_count"] <= 0:
            continue
        if meta["decoy_count"] == 0 and random.random() < 0.35:
            continue
        return grid, expected, meta
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
