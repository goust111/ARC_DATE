from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
MAROON = 9
TEMPLATE_SIZE = 5
INNER_SIZE = 3

BAND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
FRAGMENT_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 500
SCENE_RETRIES = 300
PLACE_RETRIES = 250

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _majority_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_coords(points: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    rel = tuple(sorted((r - r0, c - c0) for r, c in points))
    return rel, (r1 - r0 + 1, c1 - c0 + 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _placements_in_board(
    rel: tuple[tuple[int, int], ...], shape: tuple[int, int], board_size: int = INNER_SIZE
) -> list[tuple[tuple[int, int], ...]]:
    sh, sw = shape
    out: list[tuple[tuple[int, int], ...]] = []
    for r0 in range(board_size - sh + 1):
        for c0 in range(board_size - sw + 1):
            out.append(tuple(sorted((r0 + r, c0 + c) for r, c in rel)))
    return out


def _unique_cover_solution(
    rel_a: tuple[tuple[int, int], ...],
    shape_a: tuple[int, int],
    rel_b: tuple[tuple[int, int], ...],
    shape_b: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] | None:
    if len(rel_a) + len(rel_b) != INNER_SIZE * INNER_SIZE:
        return None

    solutions: list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
    for placed_a in _placements_in_board(rel_a, shape_a):
        set_a = set(placed_a)
        for placed_b in _placements_in_board(rel_b, shape_b):
            set_b = set(placed_b)
            if set_a & set_b:
                continue
            if len(set_a | set_b) == INNER_SIZE * INNER_SIZE:
                solutions.append((placed_a, placed_b))
                if len(solutions) > 1:
                    return None
    return solutions[0] if len(solutions) == 1 else None


def _find_templates(grid: np.ndarray, bg: int) -> tuple[list[dict[str, int]], np.ndarray]:
    h, w = grid.shape
    mask = np.zeros_like(grid, dtype=bool)
    templates: list[dict[str, int]] = []
    for r0 in range(h - TEMPLATE_SIZE + 1):
        for c0 in range(w - TEMPLATE_SIZE + 1):
            patch = grid[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE]
            if np.any(patch[1:-1, 1:-1] != bg):
                continue
            band = int(patch[0, TEMPLATE_SIZE // 2])
            if band in (bg, FRAME):
                continue

            ok = True
            for rr in range(TEMPLATE_SIZE):
                for cc in range(TEMPLATE_SIZE):
                    if rr not in (0, TEMPLATE_SIZE - 1) and cc not in (0, TEMPLATE_SIZE - 1):
                        continue
                    val = int(patch[rr, cc])
                    if rr == 0 and cc == TEMPLATE_SIZE // 2:
                        if val in (bg, FRAME):
                            ok = False
                            break
                    elif val != FRAME:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            bbox = (r0, c0, r0 + TEMPLATE_SIZE - 1, c0 + TEMPLATE_SIZE - 1)
            if any(not _boxes_separated(bbox, prev["bbox"], 0) for prev in templates):
                continue

            templates.append({"r": r0, "c": c0, "band": band, "bbox": bbox})
            mask[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE] = True

    templates.sort(key=lambda item: (item["r"], item["c"]))
    return templates, mask


def _extract_fragments(grid: np.ndarray, blocked: np.ndarray, bg: int) -> list[dict[str, object]]:
    fragments: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        comps = _components((grid == color) & (~blocked))
        for pts in comps:
            rel, shape = _crop_coords(tuple(pts))
            fragments.append(
                {
                    "color": color,
                    "coords": tuple(sorted(pts)),
                    "rel": rel,
                    "shape": shape,
                    "area": len(pts),
                    "bbox": _bbox(pts),
                }
            )
    fragments.sort(key=lambda item: (item["bbox"], item["color"], item["area"]))
    return fragments


def _unique_pairing(
    fragments: list[dict[str, object]],
) -> list[tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] | None:
    n = len(fragments)
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    edge_solution: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = {}

    for i in range(n):
        for j in range(i + 1, n):
            if int(fragments[i]["area"]) == int(fragments[j]["area"]):
                continue
            sol = _unique_cover_solution(
                fragments[i]["rel"],
                fragments[i]["shape"],
                fragments[j]["rel"],
                fragments[j]["shape"],
            )
            if sol is None:
                continue
            edge_solution[(i, j)] = sol
            neighbors[i].append(j)
            neighbors[j].append(i)

    solutions: list[list[tuple[int, int]]] = []
    unmatched = set(range(n))

    def backtrack(chosen: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if not unmatched:
            solutions.append(sorted(chosen))
            return

        idx = min(unmatched, key=lambda k: sum(1 for v in neighbors[k] if v in unmatched))
        opts = [v for v in neighbors[idx] if v in unmatched]
        if not opts:
            return

        unmatched.remove(idx)
        for other in opts:
            unmatched.remove(other)
            chosen.append((min(idx, other), max(idx, other)))
            backtrack(chosen)
            chosen.pop()
            unmatched.add(other)
            if len(solutions) > 1:
                break
        unmatched.add(idx)

    backtrack([])
    if len(solutions) != 1:
        return None

    result: list[tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
    for i, j in solutions[0]:
        placed_i, placed_j = edge_solution[(i, j)]
        result.append((i, j, placed_i, placed_j))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered fragments by shape complement and insert them into marked frame templates.

    Input structure:
    - The background is black.
    - Several identical 5x5 templates appear as one-pixel-thick grey frames with a single colored band pixel at the top center.
    - Each template interior is a blank 3x3 black region.
    - Elsewhere are monochrome 4-connected fragments of size 2-6, with two fragments per template.

    Rule:
    - Match fragments into pairs whose untranslated shapes can be shifted inside one 3x3 region to cover it exactly, with no overlap.
    - In each pair, the larger fragment's input color identifies the destination template because it matches that template's top band color.
    - Erase the scattered fragments from their original locations.
    - Fill the matched template interior with the unique 3x3 placement: the larger fragment keeps the band color, while the smaller fragment is recolored to maroon.

    Key insight:
    - Colors do not reveal the pairing directly; the hidden first step is to solve a shape-complement matching problem in 3x3 space, then use the larger fragment's color as the template label.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    templates, template_mask = _find_templates(g, bg)
    assert templates, "expected at least one 5x5 template"

    fragments = _extract_fragments(g, template_mask, bg)
    assert len(fragments) == 2 * len(templates), "fragment count must be twice the template count"

    pairing = _unique_pairing(fragments)
    assert pairing is not None, "fragments must admit a unique complement pairing"

    by_band: dict[int, dict[str, int]] = {}
    for template in templates:
        band = int(template["band"])
        assert band not in by_band, "template band colors must be unique"
        by_band[band] = template

    out = g.copy()
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = bg
    for template in templates:
        r0 = int(template["r"]) + 1
        c0 = int(template["c"]) + 1
        out[r0 : r0 + INNER_SIZE, c0 : c0 + INNER_SIZE] = bg

    used_bands: set[int] = set()
    for i, j, placed_i, placed_j in pairing:
        frag_i = fragments[i]
        frag_j = fragments[j]
        if int(frag_i["area"]) > int(frag_j["area"]):
            large = frag_i
            small = frag_j
            large_cells, small_cells = placed_i, placed_j
        else:
            large = frag_j
            small = frag_i
            large_cells, small_cells = placed_j, placed_i

        band = int(large["color"])
        assert band in by_band, "larger fragment color must match a template band"
        assert band not in used_bands, "each template must receive exactly one pair"
        used_bands.add(band)
        template = by_band[band]
        base_r = int(template["r"]) + 1
        base_c = int(template["c"]) + 1

        for dr, dc in large_cells:
            out[base_r + dr, base_c + dc] = band
        for dr, dc in small_cells:
            out[base_r + dr, base_c + dc] = MAROON

        assert int(large["area"]) > int(small["area"]), "generator guarantees distinct fragment sizes"

    assert len(used_bands) == len(templates), "every template must be filled exactly once"
    return out


def _build_pair_library() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    all_cells = [(r, c) for r in range(INNER_SIZE) for c in range(INNER_SIZE)]
    for bits in range(1, 1 << (INNER_SIZE * INNER_SIZE)):
        big_cells = [(r, c) for idx, (r, c) in enumerate(all_cells) if (bits >> idx) & 1]
        big_set = set(big_cells)
        small_cells = [cell for cell in all_cells if cell not in big_set]
        if len(big_cells) <= len(small_cells):
            continue
        if not (2 <= len(big_cells) <= 6 and 2 <= len(small_cells) <= 6):
            continue
        big_mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        small_mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        for r, c in big_cells:
            big_mask[r, c] = True
        for r, c in small_cells:
            small_mask[r, c] = True
        if len(_components(big_mask)) != 1 or len(_components(small_mask)) != 1:
            continue

        big_rel, big_shape = _crop_coords(tuple(big_cells))
        small_rel, small_shape = _crop_coords(tuple(small_cells))
        if _unique_cover_solution(big_rel, big_shape, small_rel, small_shape) is None:
            continue
        entries.append(
            {
                "big_rel": big_rel,
                "big_shape": big_shape,
                "big_area": len(big_cells),
                "small_rel": small_rel,
                "small_shape": small_shape,
                "small_area": len(small_cells),
            }
        )
    return entries


PAIR_LIBRARY = _build_pair_library()
PAIR_LIBRARY_BY_SPLIT = {
    (5, 4): [entry for entry in PAIR_LIBRARY if entry["big_area"] == 5],
    (6, 3): [entry for entry in PAIR_LIBRARY if entry["big_area"] == 6],
}


def _sample_fragments(
    n_templates: int,
) -> tuple[
    list[dict[str, object]],
    list[tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]],
] | None:
    for _ in range(SCENE_RETRIES):
        chosen: list[dict[str, object]] = []
        used_shapes: set[tuple[tuple[int, int], ...]] = set()
        for _pair_idx in range(n_templates):
            split = (5, 4) if random.random() < 0.5 else (6, 3)
            pool = [
                entry
                for entry in PAIR_LIBRARY_BY_SPLIT[split]
                if entry["big_rel"] not in used_shapes and entry["small_rel"] not in used_shapes
            ]
            if not pool:
                break
            entry = random.choice(pool)
            used_shapes.add(entry["big_rel"])
            used_shapes.add(entry["small_rel"])
            chosen.append(entry)
        if len(chosen) != n_templates:
            continue

        fragments: list[dict[str, object]] = []
        for entry in chosen:
            fragments.append(
                {
                    "rel": entry["big_rel"],
                    "shape": entry["big_shape"],
                    "area": entry["big_area"],
                    "color": None,
                    "coords": (),
                    "bbox": (0, 0, 0, 0),
                }
            )
            fragments.append(
                {
                    "rel": entry["small_rel"],
                    "shape": entry["small_shape"],
                    "area": entry["small_area"],
                    "color": None,
                    "coords": (),
                    "bbox": (0, 0, 0, 0),
                }
            )
        random.shuffle(fragments)
        pairing = _unique_pairing(fragments)
        if pairing is not None and len(pairing) == n_templates:
            return fragments, pairing
    return None


def _place_boxes(
    height: int, width: int, sizes: list[tuple[int, int]], gap: int = 1, margin: int = 1
) -> list[tuple[int, int, int, int]] | None:
    order = list(range(len(sizes)))
    random.shuffle(order)
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    existing: list[tuple[int, int, int, int]] = []

    for idx in order:
        box_h, box_w = sizes[idx]
        max_r0 = height - box_h - margin
        max_c0 = width - box_w - margin
        if max_r0 < margin or max_c0 < margin:
            return None
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(margin, max_r0)
            c0 = random.randint(margin, max_c0)
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_boxes_separated(box, other, gap) for other in existing):
                placed[idx] = box
                existing.append(box)
                break
        else:
            return None

    return [box for box in placed if box is not None]


def _draw_template(grid: np.ndarray, r0: int, c0: int, band: int) -> None:
    grid[r0, c0 : c0 + TEMPLATE_SIZE] = FRAME
    grid[r0 + TEMPLATE_SIZE - 1, c0 : c0 + TEMPLATE_SIZE] = FRAME
    grid[r0 : r0 + TEMPLATE_SIZE, c0] = FRAME
    grid[r0 : r0 + TEMPLATE_SIZE, c0 + TEMPLATE_SIZE - 1] = FRAME
    grid[r0, c0 + TEMPLATE_SIZE // 2] = band


def _render_expected_output(
    input_grid: np.ndarray,
    fragments: list[dict[str, object]],
    template_records: list[dict[str, object]],
) -> np.ndarray:
    out = input_grid.copy()
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = BG
    for record in template_records:
        r0 = int(record["r"])
        c0 = int(record["c"])
        out[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE] = BG
        for dr, dc in record["large_cells"]:
            out[r0 + 1 + dr, c0 + 1 + dc] = int(record["band"])
        for dr, dc in record["small_cells"]:
            out[r0 + 1 + dr, c0 + 1 + dc] = MAROON
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_templates = random.randint(2, 4)
        sampled = _sample_fragments(n_templates)
        if sampled is None:
            continue
        fragments, pairing = sampled

        large_colors = random.sample(BAND_COLOR_POOL, n_templates)
        small_pool = [color for color in FRAGMENT_COLOR_POOL if color not in large_colors]
        small_colors = random.sample(small_pool, n_templates)

        template_payloads: list[dict[str, object]] = []
        for pair, band, small_color in zip(pairing, large_colors, small_colors):
            i, j, placed_i, placed_j = pair
            if int(fragments[i]["area"]) > int(fragments[j]["area"]):
                large_idx, small_idx = i, j
                large_cells, small_cells = placed_i, placed_j
            else:
                large_idx, small_idx = j, i
                large_cells, small_cells = placed_j, placed_i
            fragments[large_idx]["color"] = band
            fragments[small_idx]["color"] = small_color
            template_payloads.append(
                {
                    "band": band,
                    "large_cells": large_cells,
                    "small_cells": small_cells,
                }
            )

        random.shuffle(template_payloads)

        min_side = max(18, 8 + 4 * n_templates)
        height = random.randint(min_side, 30)
        width = random.randint(min_side, 30)

        sizes = [(TEMPLATE_SIZE, TEMPLATE_SIZE)] * n_templates + [fragment["shape"] for fragment in fragments]
        boxes = _place_boxes(height, width, sizes, gap=1, margin=1)
        if boxes is None:
            continue

        grid = np.full((height, width), BG, dtype=int)

        template_records: list[dict[str, object]] = []
        for box, payload in zip(boxes[:n_templates], template_payloads):
            r0, c0, _, _ = box
            _draw_template(grid, r0, c0, int(payload["band"]))
            template_records.append(
                {
                    "r": r0,
                    "c": c0,
                    "band": int(payload["band"]),
                    "large_cells": payload["large_cells"],
                    "small_cells": payload["small_cells"],
                }
            )

        for fragment, box in zip(fragments, boxes[n_templates:]):
            r0, c0, _, _ = box
            coords = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in fragment["rel"]))
            fragment["coords"] = coords
            fragment["bbox"] = box
            for r, c in coords:
                grid[r, c] = int(fragment["color"])

        expected = _render_expected_output(grid, fragments, template_records)
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid complement-template ARC sample")
