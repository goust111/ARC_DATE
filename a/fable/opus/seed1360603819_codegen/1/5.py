from __future__ import annotations

import random
import numpy as np


GREY = 5
BLACK = 0
DYNAMIC_POOL = (1, 2, 3, 4, 6)

EARLY_PANEL_COUNT = 4
TOTAL_PANEL_COUNT = EARLY_PANEL_COUNT + 1

PANEL_HEIGHT_RANGE = (6, 8)
PANEL_WIDTH_RANGE = (6, 8)
TEMPLATE_RETRIES = 80
SHAPE_RETRIES = 120
OVERLAP_RETRIES = 240
GEN_RETRIES = 400


def _candidate_segmentations(grid: np.ndarray) -> list[tuple[int, int, int, list[np.ndarray]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, list[np.ndarray]]] = []
    for panel_count in range(3, w + 1):
        remain = w - (panel_count - 1)
        if remain <= 0 or remain % panel_count != 0:
            continue
        panel_w = remain // panel_count
        if panel_w < 2:
            continue
        sep_cols = [panel_w + i * (panel_w + 1) for i in range(panel_count - 1)]
        sep_color = int(grid[0, sep_cols[0]])
        if any(not np.all(grid[:, c] == sep_color) for c in sep_cols):
            continue
        panels = [grid[:, i * (panel_w + 1) : i * (panel_w + 1) + panel_w] for i in range(panel_count)]
        if _is_valid_panel_sequence(panels, sep_color):
            candidates.append((panel_count, panel_w, sep_color, panels))
    return candidates


def _is_valid_panel_sequence(panels: list[np.ndarray], sep_color: int) -> bool:
    template = panels[-1]
    template_colors = set(map(int, np.unique(template)))
    if len(template_colors) != 2 or sep_color not in template_colors:
        return False

    dynamic_colors: set[int] = set()
    template_mask = np.isin(template, list(template_colors))
    if not np.all(template_mask):
        return False

    for panel in panels[:-1]:
        panel_colors = set(map(int, np.unique(panel)))
        dynamic_colors |= panel_colors - template_colors

        static_here = np.isin(panel, list(template_colors))
        if np.any(static_here & (panel != template)):
            return False

    if not dynamic_colors:
        return False

    for color in dynamic_colors:
        seen_masks = [panel == color for panel in panels[:-1] if np.any(panel == color)]
        if not seen_masks:
            return False
        base = seen_masks[0]
        for other in seen_masks[1:]:
            if not np.array_equal(base, other):
                return False

    return True


def _extract_color_info(panels: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int, int, np.ndarray]]]:
    template = panels[-1]
    early = panels[:-1]
    template_colors = set(map(int, np.unique(template)))
    dynamic_colors = sorted(set(map(int, np.unique(np.concatenate(early, axis=1)))) - template_colors)

    info: list[tuple[int, int, int, int, np.ndarray]] = []
    for color in dynamic_colors:
        seen_masks: list[np.ndarray] = []
        for panel in early:
            mask = panel == color
            if np.any(mask):
                seen_masks.append(mask)
        assert seen_masks
        last_mask = seen_masks[-1]
        coords = np.argwhere(last_mask)
        top = int(coords[:, 0].min())
        left = int(coords[:, 1].min())
        info.append((len(seen_masks), top, left, color, last_mask))
    return template, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order dynamic colors by cross-panel appearance frequency, then paint them onto the final template.

    Input structure:
    - A single row of equal-width panels separated by one grey column.
    - The last panel is a static template containing only black background and grey structure pixels.
    - Each earlier panel is the template plus some dynamic-color pixels.

    Rule:
    - Count in how many earlier panels each dynamic color appears.
    - Sort dynamic colors by count descending; break ties by the highest, then leftmost pixel position.
    - Starting from the last-panel template, paint each dynamic color's fixed shape in that order.
      Later colors overwrite earlier ones on overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _candidate_segmentations(g)
    assert len(candidates) == 1, "expected one valid equal-panel segmentation"
    _panel_count, _panel_w, _sep_color, panels = candidates[0]

    template, color_info = _extract_color_info(panels)
    color_info.sort(key=lambda item: (-item[0], item[1], item[2]))

    out = template.copy()
    for _count, _top, _left, color, mask in color_info:
        out[mask] = color
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_shape_cells() -> set[tuple[int, int]]:
    h = random.randint(1, 3)
    w = random.randint(1, 3)
    area = random.randint(1, min(h * w, 4))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return _random_shape_cells()
        cells.add(random.choice(frontier))
    return _normalize_cells(cells)


def _abs_cells(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in cells}


def _bbox_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _top_left(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return min(cells)


def _cells_to_mask(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_template(h: int, w: int) -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        template = np.full((h, w), BLACK, dtype=int)
        grey_cells: set[tuple[int, int]] = set()
        target = random.randint(4, min(h * w // 2, 10))
        while len(grey_cells) < target:
            kind = random.choice(("dot", "h", "v", "L"))
            r = random.randrange(h)
            c = random.randrange(w)
            if kind == "dot":
                grey_cells.add((r, c))
            elif kind == "h":
                length = random.randint(2, min(4, w))
                c0 = random.randint(0, w - length)
                for cc in range(c0, c0 + length):
                    grey_cells.add((r, cc))
            elif kind == "v":
                length = random.randint(2, min(4, h))
                r0 = random.randint(0, h - length)
                for rr in range(r0, r0 + length):
                    grey_cells.add((rr, c))
            else:
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w:
                        grey_cells.add((rr, cc))
        if not grey_cells:
            continue
        if any(all((rr, c) in grey_cells for rr in range(h)) for c in range(w)):
            continue
        for r, c in grey_cells:
            template[r, c] = GREY
        return template
    return None


def _place_nonoverlap_shape(
    panel_h: int,
    panel_w: int,
    occupied: set[tuple[int, int]],
    used_top_lefts: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        local = _random_shape_cells()
        sh, sw = _bbox_size(local)
        if sh > panel_h or sw > panel_w:
            continue
        top = random.randint(0, panel_h - sh)
        left = random.randint(0, panel_w - sw)
        cells = _abs_cells(local, top, left)
        tl = _top_left(cells)
        if tl in used_top_lefts or cells & occupied:
            continue
        return cells
    return None


def _place_overlap_pair(
    panel_h: int,
    panel_w: int,
    used_top_lefts: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(OVERLAP_RETRIES):
        local_a = _random_shape_cells()
        local_b = _random_shape_cells()
        sha, swa = _bbox_size(local_a)
        shb, swb = _bbox_size(local_b)
        if sha > panel_h or swa > panel_w or shb > panel_h or swb > panel_w:
            continue
        top_a = random.randint(0, panel_h - sha)
        left_a = random.randint(0, panel_w - swa)
        top_b = random.randint(0, panel_h - shb)
        left_b = random.randint(0, panel_w - swb)
        cells_a = _abs_cells(local_a, top_a, left_a)
        cells_b = _abs_cells(local_b, top_b, left_b)
        overlap = cells_a & cells_b
        if not overlap:
            continue
        if overlap == cells_a or overlap == cells_b:
            continue
        tl_a = _top_left(cells_a)
        tl_b = _top_left(cells_b)
        if tl_a == tl_b or tl_a in used_top_lefts or tl_b in used_top_lefts:
            continue
        return cells_a, cells_b
    return None


def _build_role_specs(mode: str, panel_h: int, panel_w: int) -> list[dict] | None:
    tie_modes = {"tie_no_overlap", "tie_overlap"}
    overlap_modes = {"distinct_overlap", "tie_overlap"}

    colors = random.sample(DYNAMIC_POOL, 4 if mode in tie_modes else 3)
    medium_phase = random.choice((0, 1))
    medium_sets = [{0, 2}, {1, 3}]
    medium_main = medium_sets[medium_phase]
    medium_other = medium_sets[1 - medium_phase]
    slow_main = {1} if medium_phase == 0 else {2}

    if mode == "distinct_no_overlap":
        roles = [
            {"name": "fast", "panels": {0, 1, 2, 3}, "color": colors[0]},
            {"name": "mid", "panels": medium_main, "color": colors[1]},
            {"name": "slow", "panels": slow_main, "color": colors[2]},
        ]
        overlap_pair: tuple[str, str] | None = None
    elif mode == "tie_no_overlap":
        roles = [
            {"name": "fast", "panels": {0, 1, 2, 3}, "color": colors[0]},
            {"name": "mid_a", "panels": medium_main, "color": colors[1]},
            {"name": "mid_b", "panels": medium_other, "color": colors[2]},
            {"name": "slow", "panels": random.choice(({1}, {2})), "color": colors[3]},
        ]
        overlap_pair = None
    elif mode == "distinct_overlap":
        roles = [
            {"name": "fast", "panels": {0, 1, 2, 3}, "color": colors[0]},
            {"name": "mid", "panels": medium_main, "color": colors[1]},
            {"name": "slow", "panels": slow_main, "color": colors[2]},
        ]
        overlap_pair = ("mid", "slow")
    else:
        roles = [
            {"name": "fast", "panels": {0, 1, 2, 3}, "color": colors[0]},
            {"name": "mid_a", "panels": medium_sets[0], "color": colors[1]},
            {"name": "mid_b", "panels": medium_sets[1], "color": colors[2]},
            {"name": "slow", "panels": random.choice(({1}, {2})), "color": colors[3]},
        ]
        overlap_pair = ("mid_a", "mid_b")

    used_top_lefts: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    by_name = {role["name"]: role for role in roles}

    if overlap_pair is not None:
        placed_pair = _place_overlap_pair(panel_h, panel_w, used_top_lefts)
        if placed_pair is None:
            return None
        cells_a, cells_b = placed_pair
        name_a, name_b = overlap_pair
        by_name[name_a]["cells"] = cells_a
        by_name[name_b]["cells"] = cells_b
        used_top_lefts.add(_top_left(cells_a))
        used_top_lefts.add(_top_left(cells_b))
        occupied |= cells_a | cells_b

    remaining = [role for role in roles if "cells" not in role]
    random.shuffle(remaining)
    for role in remaining:
        cells = _place_nonoverlap_shape(panel_h, panel_w, occupied, used_top_lefts)
        if cells is None:
            return None
        role["cells"] = cells
        used_top_lefts.add(_top_left(cells))
        occupied |= cells

    for role in roles:
        role["mask"] = _cells_to_mask(role["cells"], panel_h, panel_w)
        role["count"] = len(role["panels"])
        role["top_left"] = _top_left(role["cells"])

    counts = [role["count"] for role in roles]
    has_tie = len(counts) != len(set(counts))
    expect_tie = mode in tie_modes
    if has_tie != expect_tie:
        return None

    overlap_pairs = 0
    for i, a in enumerate(roles):
        for b in roles[i + 1 :]:
            if a["cells"] & b["cells"]:
                overlap_pairs += 1
    expect_overlap = mode in overlap_modes
    if (overlap_pairs > 0) != expect_overlap:
        return None
    if expect_overlap and overlap_pairs != 1:
        return None

    for count in set(counts):
        positions = [role["top_left"] for role in roles if role["count"] == count]
        if len(positions) != len(set(positions)):
            return None

    return roles


def _render_input(template: np.ndarray, roles: list[dict]) -> np.ndarray:
    panels = []
    for panel_idx in range(EARLY_PANEL_COUNT):
        panel = template.copy()
        for role in roles:
            if panel_idx in role["panels"]:
                panel[role["mask"]] = role["color"]
        panels.append(panel)
    panels.append(template.copy())

    h, w = template.shape
    full = np.full((h, TOTAL_PANEL_COUNT * w + (TOTAL_PANEL_COUNT - 1)), GREY, dtype=int)
    for idx, panel in enumerate(panels):
        c0 = idx * (w + 1)
        full[:, c0 : c0 + w] = panel
    return full


def _expected_output(template: np.ndarray, roles: list[dict]) -> np.ndarray:
    out = template.copy()
    ordered = sorted(roles, key=lambda role: (-role["count"], role["top_left"][0], role["top_left"][1]))
    for role in ordered:
        out[role["mask"]] = role["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    template = _random_template(panel_h, panel_w)
    if template is None:
        return None

    mode = random.choice(
        ("distinct_no_overlap", "tie_no_overlap", "distinct_overlap", "tie_overlap")
    )
    roles = _build_role_specs(mode, panel_h, panel_w)
    if roles is None:
        return None

    grid = _render_input(template, roles)
    expected = _expected_output(template, roles)

    if np.array_equal(grid[:, panel_w:: panel_w + 1], np.full((panel_h, TOTAL_PANEL_COUNT - 1), GREY, dtype=int)):
        pass
    else:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
