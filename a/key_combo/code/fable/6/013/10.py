from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BLACK = int(Color.BLACK)
MAX_GENERATION_ATTEMPTS = 800
PANEL_SIZE_RANGE = (7, 10)
GAP_RANGE = (1, 3)

CORNER_ORDER = ("TL", "TR", "BR", "BL")
ROTATE_CCW_BY_CORNER = {
    "TL": 0,
    "TR": 3,
    "BR": 2,
    "BL": 1,
}
UNROTATE_CCW_BY_CORNER = {
    "TL": 0,
    "TR": 1,
    "BR": 2,
    "BL": 3,
}


def _is_filled_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(mask[r0 : r1 + 1, c0 : c1 + 1].sum()) == (int(r1 - r0 + 1) * int(c1 - c0 + 1))


def _corner_from_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = map(int, shape)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    vertical = "T" if center_r < h / 2.0 else "B"
    horizontal = "L" if center_c < w / 2.0 else "R"
    return vertical + horizontal


def _extract_panels(grid: np.ndarray) -> tuple[list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(g[g.shape[0] // 2, g.shape[1] // 2])
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) == 4, "expected four corner panels"

    panels: list[dict[str, object]] = []
    sizes = set()
    for comp in components:
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, bbox)
        assert patch.shape[0] == patch.shape[1], "panel bbox must be square"
        sizes.add(tuple(map(int, patch.shape)))
        panels.append(
            {
                "bbox": bbox,
                "patch": patch,
                "corner": _corner_from_bbox(bbox, g.shape),
            }
        )

    assert len(sizes) == 1, "all panels must share the same size"
    return panels, bg


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    panels, bg = _extract_panels(np.asarray(grid, dtype=int))
    panel_size = int(panels[0]["patch"].shape[0])

    swatch_panel = None
    pattern_panels: list[dict[str, object]] = []
    for panel in panels:
        patch = np.asarray(panel["patch"], dtype=int)
        unique = np.unique(patch)
        if unique.size == 1 and int(unique[0]) != bg:
            swatch_panel = panel
        else:
            pattern_panels.append(panel)

    assert swatch_panel is not None and len(pattern_panels) == 3, "expected three damaged panels and one swatch"
    swatch_color = int(np.asarray(swatch_panel["patch"], dtype=int)[0, 0])

    stain_colors: list[int] = []
    pattern_colors: list[int] = []
    for panel in pattern_panels:
        patch = np.asarray(panel["patch"], dtype=int)
        colors = [int(c) for c in np.unique(patch) if int(c) != bg]
        assert len(colors) == 2, "each damaged panel must contain exactly mask and stain colors"
        rect_colors = [color for color in colors if _is_filled_rectangle(patch == int(color))]
        assert len(rect_colors) == 1, "expected exactly one rectangular stain color per panel"
        stain = int(rect_colors[0])
        pattern = next(color for color in colors if int(color) != stain)
        stain_colors.append(stain)
        pattern_colors.append(int(pattern))

    assert len(set(stain_colors)) == 1, "stain color must be consistent"
    assert len(set(pattern_colors)) == 1, "mask color must be consistent"
    pattern_color = int(pattern_colors[0])

    aligned_masks: list[np.ndarray] = []
    for panel in pattern_panels:
        patch = np.asarray(panel["patch"], dtype=int)
        visible = patch == int(pattern_color)
        aligned = np.rot90(visible, k=UNROTATE_CCW_BY_CORNER[str(panel["corner"])])
        aligned_masks.append(aligned)

    merged_mask = np.zeros((panel_size, panel_size), dtype=bool)
    for mask in aligned_masks:
        merged_mask |= mask

    assert merged_mask.any(), "merged mask cannot be empty"
    answer_mask = np.rot90(merged_mask, k=ROTATE_CCW_BY_CORNER[str(swatch_panel["corner"])])
    out = np.full((panel_size, panel_size), bg, dtype=int)
    out[answer_mask] = int(swatch_color)

    return out, {
        "bg": int(bg),
        "panel_size": panel_size,
        "pattern_color": int(pattern_color),
        "stain_color": int(stain_colors[0]),
        "swatch_color": int(swatch_color),
        "missing_corner": str(swatch_panel["corner"]),
        "merged_mask": merged_mask,
        "answer_mask": answer_mask,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden template by inverse-rotating the three damaged copies.

    The damaged panels share one irregular mask color and one small rectangular
    stain color. After aligning their visible mask fragments back to the top-left
    orientation, their union restores the full template. The answer is that
    template rotated into the swatch panel's missing corner and recolored with
    the swatch color.
    """
    return _solve_details(np.asarray(input_grid, dtype=int))[0]


def _path_between(start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    r, c = map(int, start)
    tr, tc = map(int, end)
    path = [(r, c)]
    while (r, c) != (tr, tc):
        options: list[tuple[int, int]] = []
        if r != tr:
            options.append((r + (1 if tr > r else -1), c))
        if c != tc:
            options.append((r, c + (1 if tc > c else -1)))
        r, c = options[int(rng.integers(0, len(options)))]
        path.append((r, c))
    return path


def _grow_connected_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    target_min = max(size + 5, int(0.35 * size * size))
    target_max = int(0.62 * size * size)

    for _ in range(MAX_GENERATION_ATTEMPTS):
        mask = np.zeros((size, size), dtype=bool)

        top = (0, int(rng.integers(1, size - 1)))
        bottom = (size - 1, int(rng.integers(1, size - 1)))
        left = (int(rng.integers(1, size - 1)), 0)
        right = (int(rng.integers(1, size - 1)), size - 1)
        hub = (int(rng.integers(1, size - 1)), int(rng.integers(1, size - 1)))

        for a, b in ((hub, top), (hub, bottom), (hub, left), (hub, right)):
            for r, c in _path_between(a, b, rng):
                mask[int(r), int(c)] = True

        target_area = int(rng.integers(target_min, target_max + 1))
        while int(mask.sum()) < target_area:
            cells = np.argwhere(mask)
            base_r, base_c = map(int, cells[int(rng.integers(0, len(cells)))])
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = base_r + dr
                nc = base_c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    neighbors.append((nr, nc))
            nr, nc = neighbors[int(rng.integers(0, len(neighbors)))]
            mask[int(nr), int(nc)] = True

        if np.all(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k=k)) for k in (1, 2, 3)):
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if _is_filled_rectangle(mask):
            continue
        return mask

    raise AssertionError("failed to generate an asymmetric border-touching mask")


def _rectangle_candidates(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = map(int, mask.shape)
    candidates: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in ((2, 2), (1, 2), (2, 1), (1, 1)):
        for r0 in range(h - rect_h + 1):
            for c0 in range(w - rect_w + 1):
                if np.all(mask[r0 : r0 + rect_h, c0 : c0 + rect_w]):
                    candidates.append((r0, c0, rect_h, rect_w))
    return candidates


def _choose_occlusions(mask: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    candidates = _rectangle_candidates(mask)
    rng.shuffle(candidates)
    chosen: list[tuple[int, int, int, int]] = []
    used = np.zeros_like(mask, dtype=bool)

    for rect in candidates:
        r0, c0, rect_h, rect_w = rect
        region = used[r0 : r0 + rect_h, c0 : c0 + rect_w]
        if region.any():
            continue
        chosen.append(rect)
        used[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        if len(chosen) == 3:
            return chosen

    raise AssertionError("failed to choose three disjoint rectangular occlusions")


def _rotate_rect(rect: tuple[int, int, int, int], size: int, corner: str) -> tuple[int, int, int, int]:
    r0, c0, rect_h, rect_w = map(int, rect)
    mask = np.zeros((size, size), dtype=bool)
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
    rotated = np.rot90(mask, k=ROTATE_CCW_BY_CORNER[corner])
    coords = np.argwhere(rotated)
    rr0, cc0 = coords.min(axis=0)
    rr1, cc1 = coords.max(axis=0)
    return int(rr0), int(cc0), int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)


def _render_pattern_panel(
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
    stain_color: int,
    corner: str,
) -> np.ndarray:
    size = int(mask.shape[0])
    patch = np.full((size, size), BLACK, dtype=int)
    rotated_mask = np.rot90(mask, k=ROTATE_CCW_BY_CORNER[corner])
    patch[rotated_mask] = int(color)

    rr0, cc0, rect_h, rect_w = _rotate_rect(rect, size, corner)
    patch[rr0 : rr0 + rect_h, cc0 : cc0 + rect_w] = int(stain_color)
    return patch


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    panel_size = int(rng.integers(PANEL_SIZE_RANGE[0], PANEL_SIZE_RANGE[1] + 1))
    gap = int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1))

    mask = _grow_connected_mask(panel_size, rng)
    occlusions = _choose_occlusions(mask, rng)

    colors = rng.choice(np.array(Color.NOT_BLACK, dtype=int), size=3, replace=False).tolist()
    pattern_color, stain_color, swatch_color = [int(v) for v in colors]

    missing_corner = CORNER_ORDER[int(rng.integers(0, len(CORNER_ORDER)))]
    occlusion_iter = iter(occlusions)

    grid = np.full((2 * panel_size + gap, 2 * panel_size + gap), BLACK, dtype=int)
    placements = {
        "TL": (0, 0),
        "TR": (0, panel_size + gap),
        "BR": (panel_size + gap, panel_size + gap),
        "BL": (panel_size + gap, 0),
    }

    for corner in CORNER_ORDER:
        r0, c0 = placements[corner]
        if corner == missing_corner:
            patch = np.full((panel_size, panel_size), int(swatch_color), dtype=int)
        else:
            patch = _render_pattern_panel(
                mask,
                next(occlusion_iter),
                int(pattern_color),
                int(stain_color),
                corner,
            )
        paste(grid, patch, int(r0), int(c0))

    answer_mask = np.rot90(mask, k=ROTATE_CCW_BY_CORNER[missing_corner])
    expected = np.full((panel_size, panel_size), BLACK, dtype=int)
    expected[answer_mask] = int(swatch_color)

    return grid, expected, {
        "panel_size": panel_size,
        "gap": gap,
        "pattern_color": int(pattern_color),
        "stain_color": int(stain_color),
        "swatch_color": int(swatch_color),
        "missing_corner": missing_corner,
        "mask": mask,
        "occlusions": occlusions,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid, expected, _meta = _generate_case(rng)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise AssertionError("failed to generate a self-consistent input")
