from __future__ import annotations

import random
from collections import Counter

import numpy as np


PANEL_NAMES = ("UL", "UR", "LL", "LR")
OVERLAY_ORDER = ("UL", "LR", "LL", "UR")
TEMPLATES = ("ur_ll_hollow", "ur_lr_hollow", "ll_lr_hollow")
BACKGROUND = 0
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    comps: list[tuple[int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        mask = grid == color
        seen = np.zeros_like(mask, dtype=bool)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append((color, np.asarray(cells, dtype=int)))
    return comps


def _rect_size(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    h, w = _rect_size(rect)
    return h * w


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _gap_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _rect_inside(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> bool:
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


def _hollow_rect(grid: np.ndarray, rect: tuple[int, int, int, int], bg: int) -> None:
    r0, c0, r1, c1 = rect
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(bg)


def _split_panels(grid: np.ndarray) -> dict[str, np.ndarray]:
    h, w = grid.shape
    panel_h = h // 2
    panel_w = w // 2
    return {
        "UL": grid[:panel_h, :panel_w],
        "UR": grid[:panel_h, panel_w:],
        "LL": grid[panel_h:, :panel_w],
        "LR": grid[panel_h:, panel_w:],
    }


def _compose_panels(panels: dict[str, np.ndarray], bg: int) -> tuple[np.ndarray, np.ndarray]:
    sample = next(iter(panels.values()))
    out = np.full(sample.shape, bg, dtype=int)
    coverage = np.zeros(sample.shape, dtype=int)
    for panel in panels.values():
        coverage += (panel != bg).astype(int)
    for name in OVERLAY_ORDER:
        panel = panels[name]
        mask = panel != bg
        out[mask] = panel[mask]
    return out, coverage


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transparently overlay the four quadrant panels, then hollow only the visible rectangles that hide multi-panel overlap.

    Input structure:
    - The grid has black background and conceptually splits into four equal panels: upper-left, upper-right, lower-left, and lower-right.
    - Each panel contains one or two solid axis-aligned rectangles, all drawn in that panel's single foreground color.
    - Corresponding local panel coordinates refer to the same output cell after overlay.

    Rule:
    - Split the input into the four equal panels and overlay them onto one panel-sized canvas in the fixed order upper-left, lower-right, lower-left, upper-right; later non-background pixels overwrite earlier ones.
    - Simultaneously count, for every output cell, how many panels contribute a non-background pixel at that local coordinate.
    - In the composite image, inspect each monochrome 4-connected component that fully fills its own bounding box, i.e. each visible solid rectangle.
    - If any cell of that visible rectangle has coverage count at least 2, keep its one-pixel border and set its strict interior to black; otherwise leave it solid.

    Key insight:
    - The same overlay yields two necessary signals at once: visible color comes from layer priority, but the decision to hollow comes from hidden multi-layer coverage that may be invisible in the final composite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "grid must split into four equal panels"

    bg = _get_bg(g)
    panels = _split_panels(g)
    composite, coverage = _compose_panels(panels, bg)

    out = composite.copy()
    for color, coords in _components_by_color(composite, bg):
        r0, c0, r1, c1 = _bbox(coords)
        block = composite[r0 : r1 + 1, c0 : c1 + 1]
        if coords.shape[0] != block.size or not np.all(block == color):
            continue
        if np.any(coverage[r0 : r1 + 1, c0 : c1 + 1] >= 2):
            _hollow_rect(out, (r0, c0, r1, c1), bg)
    return out


def _zone_bounds(panel_h: int, panel_w: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "A": (0, panel_h // 2 - 1, 0, panel_w // 2 - 1),
        "B": (panel_h // 2 + 1, panel_h - 1, 0, panel_w // 2 - 1),
        "C": (0, panel_h - 1, panel_w // 2 + 1, panel_w - 1),
    }


def _sample_rect_in_zone(
    zone: tuple[int, int, int, int],
    min_h: int,
    min_w: int,
) -> tuple[int, int, int, int] | None:
    zr0, zr1, zc0, zc1 = zone
    zone_h = zr1 - zr0 + 1
    zone_w = zc1 - zc0 + 1
    if zone_h < min_h or zone_w < min_w:
        return None
    height = random.randint(min_h, zone_h)
    width = random.randint(min_w, zone_w)
    r0 = random.randint(zr0, zr1 - height + 1)
    c0 = random.randint(zc0, zc1 - width + 1)
    return (r0, c0, r0 + height - 1, c0 + width - 1)


def _sample_support(outer: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = outer
    height, width = _rect_size(outer)
    inner_h = random.randint(3, height)
    inner_w = random.randint(3, width)
    sr0 = random.randint(r0, r1 - inner_h + 1)
    sc0 = random.randint(c0, c1 - inner_w + 1)
    return (sr0, sc0, sr0 + inner_h - 1, sc0 + inner_w - 1)


def _render_input(
    panel_h: int,
    panel_w: int,
    panel_colors: dict[str, int],
    panel_rects: dict[str, list[tuple[int, int, int, int]]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    panels: dict[str, np.ndarray] = {}
    for name in PANEL_NAMES:
        panel = np.full((panel_h, panel_w), BACKGROUND, dtype=int)
        for rect in panel_rects[name]:
            _draw_rect(panel, rect, panel_colors[name])
        panels[name] = panel

    full = np.full((panel_h * 2, panel_w * 2), BACKGROUND, dtype=int)
    full[:panel_h, :panel_w] = panels["UL"]
    full[:panel_h, panel_w:] = panels["UR"]
    full[panel_h:, :panel_w] = panels["LL"]
    full[panel_h:, panel_w:] = panels["LR"]
    return full, panels


def _independent_expected(scene: dict) -> np.ndarray:
    composite = scene["composite"].copy()
    for name, info in scene["visible_regions"].items():
        if info["hollow"]:
            _hollow_rect(composite, info["rect"], BACKGROUND)
    return composite


def _validate_scene(scene: dict) -> bool:
    panel_h = scene["panel_h"]
    panel_w = scene["panel_w"]
    panel_colors = scene["panel_colors"]
    panel_rects = scene["panel_rects"]
    panels = scene["panels"]
    composite = scene["composite"]
    coverage = scene["coverage"]
    visible_regions = scene["visible_regions"]
    inp = scene["input"]

    if inp.shape != (panel_h * 2, panel_w * 2):
        return False
    if max(inp.shape) > 30:
        return False

    for name in PANEL_NAMES:
        rects = panel_rects[name]
        if not (1 <= len(rects) <= 2):
            return False
        for rect in rects:
            h, w = _rect_size(rect)
            if h < 3 or w < 3:
                return False
        if len(rects) == 2 and not _gap_disjoint(rects[0], rects[1], gap=1):
            return False
        comps = _components_by_color(panels[name], BACKGROUND)
        if len(comps) != len(rects):
            return False
        for color, coords in comps:
            if color != panel_colors[name]:
                return False
            r0, c0, r1, c1 = _bbox(coords)
            if coords.shape[0] != _rect_area((r0, c0, r1, c1)):
                return False

    overlap_pairs = 0
    all_panel_rects: list[tuple[str, tuple[int, int, int, int]]] = []
    for name in PANEL_NAMES:
        all_panel_rects.extend((name, rect) for rect in panel_rects[name])
    for i, (name_a, rect_a) in enumerate(all_panel_rects):
        for name_b, rect_b in all_panel_rects[i + 1 :]:
            if name_a == name_b:
                continue
            area = _overlap_area(rect_a, rect_b)
            if area > 0:
                if area < 9:
                    return False
                overlap_pairs += 1
    if overlap_pairs < 2:
        return False

    actual_panels = _split_panels(inp)
    if any(not np.array_equal(actual_panels[name], panels[name]) for name in PANEL_NAMES):
        return False
    actual_composite, actual_coverage = _compose_panels(actual_panels, BACKGROUND)
    if not np.array_equal(actual_composite, composite):
        return False
    if not np.array_equal(actual_coverage, coverage):
        return False

    actual_visible = {
        (int(color), _bbox(coords))
        for color, coords in _components_by_color(composite, BACKGROUND)
    }
    intended_visible = {
        (panel_colors[info["owner"]], info["rect"])
        for info in visible_regions.values()
    }
    if actual_visible != intended_visible:
        return False

    hollow_count = 0
    solid_count = 0
    for info in visible_regions.values():
        rect = info["rect"]
        r0, c0, r1, c1 = rect
        if not np.all(composite[r0 : r1 + 1, c0 : c1 + 1] == panel_colors[info["owner"]]):
            return False
        has_overlap = bool(np.any(coverage[r0 : r1 + 1, c0 : c1 + 1] >= 2))
        if has_overlap != info["hollow"]:
            return False
        if info["hollow"]:
            hollow_count += 1
        else:
            solid_count += 1
    if hollow_count < 1 or solid_count < 1:
        return False

    expected = _independent_expected(scene)
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _attempt_scene() -> dict | None:
    panel_h = random.randint(9, 12)
    panel_w = random.randint(9, 12)
    zone = _zone_bounds(panel_h, panel_w)
    template = random.choice(TEMPLATES)
    colors = random.sample(list(range(1, 10)), 4)
    panel_colors = {name: color for name, color in zip(PANEL_NAMES, colors)}

    hollow_regions: dict[str, bool]
    if template == "ur_ll_hollow":
        hollow_regions = {"A": True, "B": True, "C": False}
    elif template == "ur_lr_hollow":
        hollow_regions = {"A": True, "B": False, "C": True}
    else:
        hollow_regions = {"A": False, "B": True, "C": True}

    visible_rects: dict[str, tuple[int, int, int, int]] = {}
    for region_name in ("A", "B", "C"):
        min_side = 4 if hollow_regions[region_name] else 3
        rect = _sample_rect_in_zone(zone[region_name], min_side, min_side)
        if rect is None:
            return None
        visible_rects[region_name] = rect

    if not _gap_disjoint(visible_rects["A"], visible_rects["C"], gap=1):
        return None
    if not _gap_disjoint(visible_rects["B"], visible_rects["C"], gap=1):
        return None

    panel_rects: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in PANEL_NAMES}
    visible_regions: dict[str, dict[str, object]] = {}

    if template == "ur_ll_hollow":
        panel_rects["UR"].append(visible_rects["A"])
        panel_rects["LL"].append(visible_rects["B"])
        panel_rects["LR"].append(visible_rects["C"])
        panel_rects["UL"].append(_sample_support(visible_rects["A"]))
        panel_rects["UL"].append(_sample_support(visible_rects["B"]))
        panel_rects["LR"].append(_sample_support(visible_rects["A"]))
        visible_regions = {
            "A": {"rect": visible_rects["A"], "owner": "UR", "hollow": True},
            "B": {"rect": visible_rects["B"], "owner": "LL", "hollow": True},
            "C": {"rect": visible_rects["C"], "owner": "LR", "hollow": False},
        }
    elif template == "ur_lr_hollow":
        panel_rects["UR"].append(visible_rects["A"])
        panel_rects["LL"].append(visible_rects["B"])
        panel_rects["LR"].append(visible_rects["C"])
        panel_rects["UL"].append(_sample_support(visible_rects["A"]))
        panel_rects["UL"].append(_sample_support(visible_rects["C"]))
        panel_rects["LL"].append(_sample_support(visible_rects["A"]))
        visible_regions = {
            "A": {"rect": visible_rects["A"], "owner": "UR", "hollow": True},
            "B": {"rect": visible_rects["B"], "owner": "LL", "hollow": False},
            "C": {"rect": visible_rects["C"], "owner": "LR", "hollow": True},
        }
    else:
        panel_rects["UR"].append(visible_rects["A"])
        panel_rects["LL"].append(visible_rects["B"])
        panel_rects["LR"].append(visible_rects["C"])
        panel_rects["UL"].append(_sample_support(visible_rects["B"]))
        panel_rects["UL"].append(_sample_support(visible_rects["C"]))
        panel_rects["LR"].append(_sample_support(visible_rects["B"]))
        visible_regions = {
            "A": {"rect": visible_rects["A"], "owner": "UR", "hollow": False},
            "B": {"rect": visible_rects["B"], "owner": "LL", "hollow": True},
            "C": {"rect": visible_rects["C"], "owner": "LR", "hollow": True},
        }

    for info in visible_regions.values():
        if info["hollow"]:
            h, w = _rect_size(info["rect"])
            if h < 4 or w < 4:
                return None

    for name in PANEL_NAMES:
        for rect in panel_rects[name]:
            h, w = _rect_size(rect)
            if h < 3 or w < 3:
                return None

    inp, panels = _render_input(panel_h, panel_w, panel_colors, panel_rects)
    composite, coverage = _compose_panels(panels, BACKGROUND)
    scene = {
        "template": template,
        "panel_h": panel_h,
        "panel_w": panel_w,
        "panel_colors": panel_colors,
        "panel_rects": panel_rects,
        "visible_regions": visible_regions,
        "input": inp,
        "panels": panels,
        "composite": composite,
        "coverage": coverage,
    }
    if not _validate_scene(scene):
        return None
    return scene


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        scene = _attempt_scene()
        if scene is not None:
            return scene["input"].copy()
    raise RuntimeError("failed to generate a valid transparent-overlay hollowing input")
