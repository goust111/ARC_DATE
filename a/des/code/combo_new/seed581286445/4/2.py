from __future__ import annotations

import random

import numpy as np


GEN_BG = 0
GEN_SEPARATOR = 5
GEN_TARGET_CHOICES = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected_coords(coords: tuple[tuple[int, int], ...]) -> bool:
    coord_set = set(coords)
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int]:
    row_lines = [
        r for r in range(grid.shape[0]) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_lines = [
        c for c in range(grid.shape[1]) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(row_lines) + len(col_lines) == 1, "expected exactly one full grey separator line"
    if row_lines:
        return "h", int(row_lines[0])
    return "v", int(col_lines[0])


def _split_panels(grid: np.ndarray, orient: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    if orient == "h":
        first = grid[:index, :]
        second = grid[index + 1 :, :]
    else:
        first = grid[:, :index]
        second = grid[:, index + 1 :]
    assert first.size > 0 and second.size > 0, "separator must divide the grid into two non-empty panels"
    return first.copy(), second.copy()


def _select_panels(first: np.ndarray, second: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    first_colors = set(map(int, np.unique(first))) - {bg}
    second_colors = set(map(int, np.unique(second))) - {bg}
    if len(first_colors) == 1 and len(second_colors) >= 2:
        return first, second
    if len(second_colors) == 1 and len(first_colors) >= 2:
        return second, first
    raise AssertionError("could not distinguish the monochrome target panel from the template panel")


def _extract_target(panel: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    colors = set(map(int, np.unique(panel))) - {bg}
    assert len(colors) == 1, "target panel must contain exactly one foreground color"
    target_color = next(iter(colors))
    mask = panel == target_color
    assert len(_components(mask)) == 1, "target object must be a single 4-connected component"
    return int(target_color), mask


def _extract_templates(panel: np.ndarray, bg: int) -> list[dict]:
    comps = _components(panel != bg)
    assert len(comps) >= 2, "template panel must contain several disconnected templates"
    templates = []
    for comp in comps:
        assert 2 <= len(comp) <= 5, "each template must have 2-5 pixels"
        r0, c0, r1, c1 = _bbox(comp)
        colors = [int(panel[r, c]) for r, c in comp]
        assert len(set(colors)) == len(colors), "template pixels must all have distinct colors"
        rel_coords = tuple(sorted((r - r0, c - c0) for r, c in comp))
        assert _is_connected_coords(rel_coords), "template must stay 4-connected after normalization"
        color_map = {rc: int(panel[r0 + rc[0], c0 + rc[1]]) for rc in rel_coords}
        templates.append(
            {
                "shape": rel_coords,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "area": len(comp),
                "colors": color_map,
            }
        )
    return templates


def _find_template_match(target_mask: np.ndarray, template: dict) -> tuple[int, int]:
    h, w = target_mask.shape
    th = int(template["h"])
    tw = int(template["w"])
    shape = tuple(template["shape"])
    area = int(template["area"])
    matches: list[tuple[int, int]] = []
    for r0 in range(h - th + 1):
        for c0 in range(w - tw + 1):
            window = target_mask[r0 : r0 + th, c0 : c0 + tw]
            if int(window.sum()) != area:
                continue
            if all(bool(window[dr, dc]) for dr, dc in shape):
                matches.append((r0, c0))
    assert len(matches) == 1, "each template must have exactly one translation-only match in the target"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Locate each small multicolor template inside a larger monochrome shape and transfer the template colors onto the matched cells.

    Input structure:
    - The background is black.
    - A single-cell grey separator row or column splits the grid into two panels.
    - One panel contains one large 4-connected monochrome target object on the background.
    - The other panel contains several disconnected 2-5-pixel templates whose pixels use distinct non-background colors.

    Rule:
    - Identify which side is the monochrome target panel and which side holds the small templates.
    - For each template, ignore its colors and keep only its occupied relative shape inside its tight bounding box.
    - Search the target panel for the unique translation of that exact occupied pattern inside a same-size window.
    - Recolor the matched target cells pixel-by-pixel with the template's colors in the corresponding relative positions.
    - Leave every unmatched target pixel unchanged and output only the target panel.

    Key insight:
    - The template shape is the lookup key, while the template colors are the payload copied onto the uniquely matched target subregion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    orient, sep_index = _find_separator(g, bg)
    first, second = _split_panels(g, orient, sep_index)
    target_panel, template_panel = _select_panels(first, second, bg)
    target_color, target_mask = _extract_target(target_panel, bg)

    out = target_panel.copy()
    painted: dict[tuple[int, int], int] = {}
    for template in _extract_templates(template_panel, bg):
        r0, c0 = _find_template_match(target_mask, template)
        for (dr, dc), color in template["colors"].items():
            rr, cc = r0 + dr, c0 + dc
            if (rr, cc) in painted:
                assert painted[(rr, cc)] == color, "overlapping templates must agree on color"
            painted[(rr, cc)] = int(color)
            out[rr, cc] = int(color)

    out[target_mask & (out == bg)] = target_color
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    cells = [seed]
    frontier = {nbr for nbr in _neighbors4(*seed) if 0 <= nbr[0] < height and 0 <= nbr[1] < width}
    while len(cells) < area and frontier:
        choices = list(frontier)
        weights = []
        for r, c in choices:
            degree = sum(1 for nr, nc in _neighbors4(r, c) if 0 <= nr < height and 0 <= nc < width and mask[nr, nc])
            weights.append(max(1, 4 - degree))
        r, c = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove((r, c))
        if mask[r, c]:
            continue
        mask[r, c] = True
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _enumerate_unique_templates(mask: np.ndarray) -> list[dict]:
    height, width = mask.shape
    occurrences: dict[tuple[tuple[tuple[int, int], ...], int, int], list[dict]] = {}
    for h in range(1, 6):
        for w in range(1, 6):
            if h > height or w > width:
                continue
            for r0 in range(height - h + 1):
                for c0 in range(width - w + 1):
                    window = mask[r0 : r0 + h, c0 : c0 + w]
                    area = int(window.sum())
                    if area < 2 or area > 5:
                        continue
                    rel_coords = tuple((int(r), int(c)) for r, c in np.argwhere(window))
                    rows = {r for r, _ in rel_coords}
                    cols = {c for _, c in rel_coords}
                    if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
                        continue
                    if not _is_connected_coords(rel_coords):
                        continue
                    abs_cells = tuple(sorted((r0 + r, c0 + c) for r, c in rel_coords))
                    key = (rel_coords, h, w)
                    occurrences.setdefault(key, []).append(
                        {
                            "shape": rel_coords,
                            "h": h,
                            "w": w,
                            "area": area,
                            "top_left": (r0, c0),
                            "cells": abs_cells,
                        }
                    )
    return [items[0] for items in occurrences.values() if len(items) == 1]


def _pick_disjoint_templates(candidates: list[dict], count: int) -> list[dict] | None:
    if len(candidates) < count:
        return None
    for _ in range(80):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[dict] = []
        used: set[tuple[int, int]] = set()
        for cand in order:
            if any(cell in used for cell in cand["cells"]):
                continue
            chosen.append(cand)
            used.update(cand["cells"])
            if len(chosen) == count:
                return chosen
    return None


def _make_template_patch(candidate: dict, bg: int, palette: list[int]) -> dict:
    patch = np.full((candidate["h"], candidate["w"]), bg, dtype=int)
    colors = random.sample(palette, int(candidate["area"]))
    random.shuffle(colors)
    color_map: dict[tuple[int, int], int] = {}
    for color, (r, c) in zip(colors, candidate["shape"]):
        patch[r, c] = int(color)
        color_map[(r, c)] = int(color)
    return {
        "grid": patch,
        "coords": tuple(candidate["shape"]),
        "colors": color_map,
        "shape": tuple(candidate["shape"]),
        "h": int(candidate["h"]),
        "w": int(candidate["w"]),
        "area": int(candidate["area"]),
        "cells": tuple(candidate["cells"]),
    }


def _halo8(cells: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                halo.add((rr, cc))
    return halo


def _place_templates(panel_shape: tuple[int, int], patches: list[dict], bg: int) -> np.ndarray | None:
    height, width = panel_shape
    order = sorted(range(len(patches)), key=lambda i: (-int(patches[i]["area"]), random.random()))
    for _ in range(60):
        panel = np.full((height, width), bg, dtype=int)
        blocked: set[tuple[int, int]] = set()
        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch["grid"].shape
            placements: list[tuple[int, int, list[tuple[int, int]]]] = []
            for r0 in range(height - ph + 1):
                for c0 in range(width - pw + 1):
                    cells = [(r0 + dr, c0 + dc) for dr, dc in patch["coords"]]
                    if any(cell in blocked for cell in cells):
                        continue
                    placements.append((r0, c0, cells))
            if not placements:
                ok = False
                break
            r0, c0, cells = random.choice(placements)
            for dr in range(ph):
                for dc in range(pw):
                    color = int(patch["grid"][dr, dc])
                    if color != bg:
                        panel[r0 + dr, c0 + dc] = color
            blocked.update(_halo8(cells, height, width))
        if ok:
            return panel
    return None


def _build_target_panel(height: int, width: int, target_color: int, template_count: int) -> tuple[np.ndarray, list[dict]] | None:
    area_low = max(20, (height * width) // 5)
    area_high = min(60, (height * width) // 2)
    if area_low > area_high:
        return None
    for _ in range(150):
        area = random.randint(area_low, area_high)
        mask = _random_connected_mask(height, width, area)
        unique = _enumerate_unique_templates(mask)
        chosen = _pick_disjoint_templates(unique, template_count)
        if chosen is None:
            continue
        panel = np.full((height, width), GEN_BG, dtype=int)
        panel[mask] = target_color
        return panel, chosen
    return None


def _compose_input(
    orient: str,
    target_first: bool,
    target_panel: np.ndarray,
    template_panel: np.ndarray,
    separator_color: int,
    bg: int,
) -> np.ndarray:
    if orient == "v":
        height = target_panel.shape[0]
        width = target_panel.shape[1] + 1 + template_panel.shape[1]
        grid = np.full((height, width), bg, dtype=int)
        if target_first:
            grid[:, : target_panel.shape[1]] = target_panel
            grid[:, target_panel.shape[1]] = separator_color
            grid[:, target_panel.shape[1] + 1 :] = template_panel
        else:
            grid[:, : template_panel.shape[1]] = template_panel
            grid[:, template_panel.shape[1]] = separator_color
            grid[:, template_panel.shape[1] + 1 :] = target_panel
        return grid

    height = target_panel.shape[0] + 1 + template_panel.shape[0]
    width = target_panel.shape[1]
    grid = np.full((height, width), bg, dtype=int)
    if target_first:
        grid[: target_panel.shape[0], :] = target_panel
        grid[target_panel.shape[0], :] = separator_color
        grid[target_panel.shape[0] + 1 :, :] = template_panel
    else:
        grid[: template_panel.shape[0], :] = template_panel
        grid[template_panel.shape[0], :] = separator_color
        grid[template_panel.shape[0] + 1 :, :] = target_panel
    return grid


def _apply_templates(target_panel: np.ndarray, patches: list[dict]) -> np.ndarray:
    out = target_panel.copy()
    bg = _most_common_color(target_panel)
    target_colors = set(map(int, np.unique(target_panel))) - {bg}
    assert len(target_colors) == 1, "target panel must stay monochrome before recoloring"
    target_mask = target_panel == next(iter(target_colors))
    for patch in patches:
        r0, c0 = _find_template_match(target_mask, patch)
        for (dr, dc), color in patch["colors"].items():
            out[r0 + dr, c0 + dc] = int(color)
    return out


def _generate_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    orient = random.choice(("v", "h"))
    target_first = bool(random.getrandbits(1))
    template_count = random.randint(2, 4)
    target_color = random.choice(GEN_TARGET_CHOICES)
    palette = [c for c in range(10) if c not in (GEN_BG, GEN_SEPARATOR, target_color)]

    if orient == "v":
        shared = random.randint(13, 18)
        target_width = random.randint(10, 16)
        max_template_width = min(12, 29 - target_width)
        if max_template_width < 7:
            return None
        target_data = _build_target_panel(shared, target_width, target_color, template_count)
        if target_data is None:
            return None
        target_panel, chosen = target_data
        patches = [_make_template_patch(cand, GEN_BG, palette) for cand in chosen]
        min_template_width = max(7, max(int(p["w"]) for p in patches) + 1)
        if min_template_width > max_template_width:
            return None
        template_width = random.randint(min_template_width, max_template_width)
        template_panel = _place_templates((shared, template_width), patches, GEN_BG)
        if template_panel is None:
            return None
    else:
        shared = random.randint(13, 18)
        target_height = random.randint(10, 16)
        max_template_height = min(12, 29 - target_height)
        if max_template_height < 7:
            return None
        target_data = _build_target_panel(target_height, shared, target_color, template_count)
        if target_data is None:
            return None
        target_panel, chosen = target_data
        patches = [_make_template_patch(cand, GEN_BG, palette) for cand in chosen]
        min_template_height = max(7, max(int(p["h"]) for p in patches) + 1)
        if min_template_height > max_template_height:
            return None
        template_height = random.randint(min_template_height, max_template_height)
        template_panel = _place_templates((template_height, shared), patches, GEN_BG)
        if template_panel is None:
            return None

    inp = _compose_input(orient, target_first, target_panel, template_panel, GEN_SEPARATOR, GEN_BG)
    expected = _apply_templates(target_panel, patches)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid template-recolor ARC input")
