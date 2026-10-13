import random
from collections import Counter

import numpy as np


GREY = 5
ALL_COLORS = tuple(range(10))
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRY_LIMIT = 500
GEN_GRID_MIN = 16
GEN_GRID_MAX = 24
GEN_TEMPLATE_COUNT_RANGE = (2, 4)
GEN_TEMPLATE_EXTRA_ANCHORS = (1, 3)
GEN_NOISE_COUNT_RANGE = (1, 3)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR8:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _make_template_spec(color: int) -> dict | None:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        handle = (random.randint(1, h - 2), random.randint(1, w - 2))
        coords = {
            handle,
            (handle[0] - 1, handle[1]),
            (handle[0] + 1, handle[1]),
            (handle[0], handle[1] - 1),
            (handle[0], handle[1] + 1),
        }
        target = random.randint(6, min(9, h * w - 1))
        while len(coords) < target:
            frontier = set()
            for r, c in coords:
                for dr, dc in DIR8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(tuple(frontier)))

        arr = np.array(sorted(coords), dtype=int)
        r0, c0, r1, c1 = _bbox(arr)
        arr[:, 0] -= r0
        arr[:, 1] -= c0
        hr = handle[0] - r0
        hc = handle[1] - c0
        hh = int(arr[:, 0].max()) + 1
        ww = int(arr[:, 1].max()) + 1
        if hh < 3 or ww < 3:
            continue
        if hr in (0, hh - 1) or hc in (0, ww - 1):
            continue

        mask = np.zeros((hh, ww), dtype=bool)
        mask[arr[:, 0], arr[:, 1]] = True
        if mask.all():
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue

        cells_bbox = []
        cells_rel = []
        occupied = set()
        for lr, lc in arr:
            lr = int(lr)
            lc = int(lc)
            val = GREY if (lr, lc) == (hr, hc) else color
            cells_bbox.append((lr, lc, val))
            cells_rel.append((lr - hr, lc - hc, val))
            occupied.add((lr, lc))

        first_local = min(occupied)
        rel_first = [(lr - first_local[0], lc - first_local[1], val) for lr, lc, val in cells_bbox]
        bbox_center = ((hh - 1) / 2.0, (ww - 1) / 2.0)
        handle_is_bbox_center = hr == bbox_center[0] and hc == bbox_center[1]
        return {
            "color": color,
            "bbox_h": hh,
            "bbox_w": ww,
            "handle_local": (hr, hc),
            "handle_is_bbox_center": handle_is_bbox_center,
            "cells_bbox": cells_bbox,
            "cells_rel": cells_rel,
            "cells_rel_first": rel_first,
            "cell_count": len(cells_bbox),
        }
    return None


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_component(
    grid: np.ndarray,
    blocked: np.ndarray,
    template: dict,
    bg: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = []
    for r0 in range(h - template["bbox_h"] + 1):
        for c0 in range(w - template["bbox_w"] + 1):
            coords = []
            ok = True
            for lr, lc, val in template["cells_bbox"]:
                rr = r0 + lr
                cc = c0 + lc
                if blocked[rr, cc]:
                    ok = False
                    break
                coords.append((rr, cc, val))
            if ok:
                candidates.append((r0, c0, coords))
    if not candidates:
        return None

    r0, c0, coords = random.choice(candidates)
    occupied = []
    for rr, cc, val in coords:
        grid[rr, cc] = val
        occupied.append((rr, cc))
    _mark_blocked(blocked, occupied)
    return int(r0), int(c0)


def _stamp_cells(
    template: dict,
    anchor: tuple[int, int],
    shape: tuple[int, int],
    *,
    mode: str = "handle",
) -> list[tuple[int, int, int]]:
    if mode == "handle":
        rel = template["cells_rel"]
    elif mode == "top_left":
        rel = template["cells_bbox"]
    elif mode == "first_cell":
        rel = template["cells_rel_first"]
    else:
        raise ValueError(mode)

    ar, ac = anchor
    stamped = []
    for dr, dc, val in rel:
        rr = ar + dr
        cc = ac + dc
        if _in_bounds(shape, rr, cc):
            stamped.append((rr, cc, val))
    return stamped


def _anchor_candidates(
    shape: tuple[int, int],
    blocked_input: np.ndarray,
    used_output: np.ndarray,
    forbidden_output: np.ndarray,
    template: dict,
) -> list[dict]:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if blocked_input[r, c]:
                continue
            cells = _stamp_cells(template, (r, c), shape, mode="handle")
            if len(cells) < 3:
                continue
            footprint = {(rr, cc) for rr, cc, _ in cells}
            if any(used_output[rr, cc] for rr, cc in footprint):
                continue
            if any(forbidden_output[rr, cc] for rr, cc in footprint):
                continue
            candidates.append(
                {
                    "anchor": (r, c),
                    "cells": cells,
                    "clipped": len(cells) < template["cell_count"],
                }
            )
    return candidates


def _build_output(
    shape: tuple[int, int],
    bg: int,
    templates_by_color: dict[int, dict],
    anchors: list[dict],
    *,
    mode: str = "handle",
    noise: list[dict] | None = None,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for rec in anchors:
        template = templates_by_color[rec["color"]]
        for rr, cc, val in _stamp_cells(template, rec["anchor"], shape, mode=mode):
            out[rr, cc] = val
    if noise is not None:
        for rec in noise:
            out[rec["anchor"][0], rec["anchor"][1]] = rec["color"]
    return out


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict], list[np.ndarray], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    templates: dict[int, dict] = {}
    template_components = []
    singletons = []

    for comp in _components(g != bg):
        values = g[comp[:, 0], comp[:, 1]]
        if len(comp) == 1:
            r, c = map(int, comp[0])
            singletons.append({"color": int(values[0]), "anchor": (r, c)})
            continue

        grey_idx = np.flatnonzero(values == GREY)
        other = sorted({int(v) for v in values if int(v) != GREY})
        assert len(grey_idx) == 1
        assert len(other) == 1
        color = other[0]
        assert color not in templates

        handle_r, handle_c = map(int, comp[int(grey_idx[0])])
        cells_rel = []
        cells_bbox = []
        r0, c0, _r1, _c1 = _bbox(comp)
        for rr, cc in comp:
            rr = int(rr)
            cc = int(cc)
            val = int(g[rr, cc])
            cells_rel.append((rr - handle_r, cc - handle_c, val))
            cells_bbox.append((rr - r0, cc - c0, val))

        templates[color] = {
            "color": color,
            "cells_rel": cells_rel,
            "cells_bbox": cells_bbox,
            "cell_count": len(cells_rel),
        }
        template_components.append(comp)

    assert templates
    return bg, templates, template_components, singletons


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each handled template cluster, then restamp its full shape onto matching isolated pixels.

    Input structure:
    - The grid has one dominant background color plus several disconnected foreground components.
    - Some multi-pixel components are monochrome template clusters with exactly one embedded gray
      handle pixel.
    - Additional foreground components are isolated single pixels; some share a color with a
      template cluster and some do not.

    Rule:
    - Treat every non-background connected component with more than one cell as a template cluster,
      record its full shape including the gray handle, and erase that cluster.
    - For each isolated single pixel whose color matches a recorded template, translate that entire
      template so the template's gray handle lands exactly on the singleton cell, clipping any
      out-of-bounds cells.
    - Erase isolated pixels whose color has no template; after stamping, each matched anchor cell
      becomes gray because the handle lands on it.

    Key insight:
    - Color chooses which template to copy, but the placement anchor is not the shape center or a
      bbox corner: it is the visible gray handle embedded inside the template cluster.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, templates, template_components, singletons = _parse_scene(g)
    out = g.copy()

    for comp in template_components:
        out[comp[:, 0], comp[:, 1]] = bg
    for rec in singletons:
        r, c = rec["anchor"]
        out[r, c] = bg

    for rec in singletons:
        template = templates.get(rec["color"])
        if template is None:
            continue
        for rr, cc, val in _stamp_cells(template, rec["anchor"], g.shape, mode="handle"):
            out[rr, cc] = val

    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    bg = random.choice([c for c in ALL_COLORS if c != GREY])

    palette = [c for c in ALL_COLORS if c not in (bg, GREY)]
    n_templates = random.randint(*GEN_TEMPLATE_COUNT_RANGE)
    if len(palette) < n_templates + 1:
        return None

    random.shuffle(palette)
    template_colors = palette[:n_templates]
    noise_palette = palette[n_templates:]
    if not noise_palette:
        return None

    templates = []
    for color in template_colors:
        spec = _make_template_spec(color)
        if spec is None:
            return None
        templates.append(spec)

    if all(spec["handle_is_bbox_center"] for spec in templates):
        return None

    grid = np.full((h, w), bg, dtype=int)
    blocked_input = np.zeros((h, w), dtype=bool)
    forbidden_output = np.zeros((h, w), dtype=bool)
    placements = {}

    for spec in sorted(templates, key=lambda item: item["bbox_h"] * item["bbox_w"], reverse=True):
        pos = _place_component(grid, blocked_input, spec, bg)
        if pos is None:
            return None
        placements[spec["color"]] = pos
        for lr, lc, _ in spec["cells_bbox"]:
            forbidden_output[pos[0] + lr, pos[1] + lc] = True

    templates_by_color = {spec["color"]: spec for spec in templates}

    anchor_colors = list(template_colors)
    extra_count = random.randint(*GEN_TEMPLATE_EXTRA_ANCHORS)
    repeated_color = random.choice(template_colors)
    for _ in range(extra_count):
        anchor_colors.append(repeated_color if random.random() < 0.6 else random.choice(template_colors))
    random.shuffle(anchor_colors)

    want_clipped = random.random() < 0.75
    used_output = np.zeros((h, w), dtype=bool)
    anchors = []
    clipped_count = 0
    unclipped_count = 0

    for color in anchor_colors:
        spec = templates_by_color[color]
        candidates = _anchor_candidates((h, w), blocked_input, used_output, forbidden_output, spec)
        if not candidates:
            return None
        clipped = [cand for cand in candidates if cand["clipped"]]
        unclipped = [cand for cand in candidates if not cand["clipped"]]

        pick_pool = candidates
        if want_clipped and clipped_count == 0 and clipped:
            pick_pool = clipped
        elif unclipped_count == 0 and unclipped:
            pick_pool = unclipped
        elif want_clipped and clipped_count == 0 and not clipped:
            return None
        elif unclipped_count == 0 and not unclipped:
            return None

        picked = random.choice(pick_pool)
        ar, ac = picked["anchor"]
        grid[ar, ac] = color
        _mark_blocked(blocked_input, [(ar, ac)])
        for rr, cc, _ in picked["cells"]:
            used_output[rr, cc] = True

        anchors.append({"color": color, "anchor": (ar, ac), "clipped": picked["clipped"]})
        clipped_count += int(picked["clipped"])
        unclipped_count += int(not picked["clipped"])

    noise_count = random.randint(*GEN_NOISE_COUNT_RANGE)
    noise = []
    for _ in range(noise_count):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not blocked_input[r, c] and not used_output[r, c]
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        color = random.choice(noise_palette)
        grid[r, c] = color
        _mark_blocked(blocked_input, [(r, c)])
        noise.append({"color": color, "anchor": (int(r), int(c))})

    if clipped_count == 0 and want_clipped:
        return None
    if unclipped_count == 0:
        return None

    expected = _build_output((h, w), bg, templates_by_color, anchors, mode="handle")
    if np.count_nonzero(expected != bg) < 6:
        return None
    if np.array_equal(expected, _build_output((h, w), bg, templates_by_color, anchors, mode="top_left")):
        return None
    if np.array_equal(expected, _build_output((h, w), bg, templates_by_color, anchors, mode="first_cell")):
        return None
    if np.array_equal(expected, _build_output((h, w), bg, templates_by_color, anchors, mode="handle", noise=noise)):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    parsed_bg, parsed_templates, _parsed_template_components, parsed_singletons = _parse_scene(grid)
    if parsed_bg != bg:
        return None
    if set(parsed_templates) != set(template_colors):
        return None
    if len(parsed_singletons) != len(anchors) + len(noise):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid handled-template stamping input")
