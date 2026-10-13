from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TEMPLATE_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _transform(coords, reflect: bool = False, rot_cw: int = 0) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    if reflect:
        pts = [(r, -c) for r, c in pts]
    for _ in range(int(rot_cw) % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize(pts)


def _canonical(coords) -> tuple[tuple[int, int], ...]:
    return min(_transform(coords, reflect, rot) for reflect in (False, True) for rot in range(4))


def _bbox(coords) -> tuple[int, int, int, int]:
    pts = np.asarray(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_hw(coords) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _shift(coords, dr: int, dc: int) -> np.ndarray:
    pts = np.asarray(coords, dtype=int).copy()
    pts[:, 0] += int(dr)
    pts[:, 1] += int(dc)
    return pts


def _component_mask(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "shape": _normalize(coords),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _find_divider_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected one full-height grey divider column"
    return int(cols[0])


def _best_alignment(query_shape, template_shape) -> tuple[bool, int] | None:
    pure = [rot for rot in range(4) if _transform(query_shape, False, rot) == template_shape]
    if pure:
        return False, int(min(pure))
    mirrored = [rot for rot in range(4) if _transform(query_shape, True, rot) == template_shape]
    if mirrored:
        return True, int(min(mirrored))
    return None


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    divider = _find_divider_col(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]

    templates = _monochrome_components(left, set(TEMPLATE_COLORS))
    assert 3 <= len(templates) <= 4, "expected 3-4 legend templates"
    template_colors = {t["color"] for t in templates}
    assert len(template_colors) == len(templates), "template colors must be distinct"
    assert len({_canonical(t["shape"]) for t in templates}) == len(templates), "template shapes must be dihedrally distinct"

    queries = _monochrome_components(right, {RED})
    assert len(queries) == len(templates), "expected a one-to-one template/query count"
    for query in queries:
        query["coords"] = _shift(query["coords"], 0, divider + 1)
        r0, c0, r1, c1 = query["bbox"]
        query["bbox"] = (int(r0), int(c0 + divider + 1), int(r1), int(c1 + divider + 1))

    matches = []
    used_templates = set()
    for query in queries:
        qshape = query["shape"]
        options = []
        for idx, template in enumerate(templates):
            align = _best_alignment(qshape, template["shape"])
            if align is not None:
                options.append((idx, align[0], align[1]))
        assert len(options) == 1, "each query must match exactly one template"
        idx, reflect, rot = options[0]
        assert idx not in used_templates, "each template must be matched exactly once"
        used_templates.add(idx)
        t = templates[idx]
        qr0, qc0, qr1, qc1 = query["bbox"]
        h = int(qr1 - qr0 + 1)
        w = int(qc1 - qc0 + 1)
        input_c0 = int(qc0)
        near_c0 = divider + 1
        far_c0 = g.shape[1] - w
        dest_c0 = near_c0 if rot % 2 == 0 else far_c0
        matches.append(
            {
                "query": query,
                "template": t,
                "template_index": idx,
                "reflect": reflect,
                "rot": rot,
                "parity": rot % 2,
                "height": h,
                "width": w,
                "input_c0": input_c0,
                "dest_c0": dest_c0,
                "shift_dc": dest_c0 - input_c0,
            }
        )

    assert len(used_templates) == len(templates)
    return {"divider": divider, "templates": templates, "matches": matches}


def _render_from_matches(grid: np.ndarray, matches: list[dict], *, recolor: bool, slide: bool, force_parity: int | None = None) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    divider = _find_divider_col(np.asarray(grid, dtype=int))
    width_total = grid.shape[1]
    for m in matches:
        coords = m["query"]["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for m in matches:
        color = m["template"]["color"] if recolor else RED
        coords = m["query"]["coords"]
        if slide:
            parity = m["parity"] if force_parity is None else int(force_parity)
            target_c0 = divider + 1 if parity == 0 else width_total - m["width"]
            dc = int(target_c0 - m["input_c0"])
            shifted = _shift(coords, 0, dc)
        else:
            shifted = coords
        out[shifted[:, 0], shifted[:, 1]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Match each red query shape to the legend, recolor it, then slide it by the parity of the required rotation.

    Input structure:
    - The grid has a black background and one full-height grey vertical divider splitting a left legend half from a right query half.
    - The left half contains 3-4 small 4-connected colored template sprites, all with distinct colors and pairwise distinct shapes up to rotation/reflection.
    - The right half contains the same number of red 4-connected query sprites whose shapes each match exactly one template under rotation/reflection.

    Rule:
    - For each red query, compare its shape with all legend templates under the 8 rigid transforms and identify the unique matching template.
    - Recolor the query to the matched template color.
    - Read off the minimum clockwise rotation steps needed to align the query to that template, preferring a pure-rotation explanation over any reflected explanation when both exist.
    - If that step count is even, slide the recolored sprite horizontally toward the divider until its leftmost occupied column is immediately to the right of the divider; if odd, slide it horizontally to the far right edge.
    - Clear the original query position; keep the legend and divider unchanged.

    Key insight:
    - The same shape match yields two independent outputs: template identity determines recoloring, while the matched rotation parity determines whether the sprite moves inward or outward.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    out = g.copy()
    for m in info["matches"]:
        coords = m["query"]["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for m in info["matches"]:
        shifted = _shift(m["query"]["coords"], 0, m["shift_dc"])
        out[shifted[:, 0], shifted[:, 1]] = int(m["template"]["color"])
    return out


def _grow_polyominoes(max_size: int = 6) -> list[tuple[tuple[int, int], ...]]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, max_size + 1):
        out = set()
        for shape in by_size[size - 1]:
            cells = set(shape)
            for r, c in shape:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        grown = _normalize(cells | {nxt})
                        h, w = _shape_hw(grown)
                        if h <= 3 and w <= 3:
                            out.add(grown)
        by_size[size] = out
    catalog = []
    seen = set()
    for size in range(3, max_size + 1):
        for shape in by_size[size]:
            key = _canonical(shape)
            if key in seen:
                continue
            seen.add(key)
            catalog.append(key)
    return catalog


def _query_options(template_shape) -> list[dict]:
    options = []
    seen = set()
    for reflect in (False, True):
        for rot in range(4):
            qshape = _transform(template_shape, reflect, rot)
            if qshape in seen:
                continue
            seen.add(qshape)
            best = _best_alignment(qshape, template_shape)
            assert best is not None
            options.append(
                {
                    "shape": qshape,
                    "reflect": best[0],
                    "rot": best[1],
                    "parity": best[1] % 2,
                }
            )
    return options


SHAPE_CATALOG = []
for _shape in _grow_polyominoes():
    _opts = _query_options(_shape)
    parities = {opt["parity"] for opt in _opts}
    if parities == {0, 1}:
        SHAPE_CATALOG.append({"canonical": _shape, "options": _opts})


def _bbox_gap_ok(box, placed, gap: int = 1) -> bool:
    r0, c0, r1, c1 = map(int, box)
    for er0, ec0, er1, ec1 in placed:
        if r1 < er0:
            dr = er0 - r1 - 1
        elif er1 < r0:
            dr = r0 - er1 - 1
        else:
            dr = 0
        if c1 < ec0:
            dc = ec0 - c1 - 1
        elif ec1 < c0:
            dc = c0 - ec1 - 1
        else:
            dc = 0
        if max(dr, dc) < gap:
            return False
    return True


def _draw_shape(grid: np.ndarray, shape, color: int, top: int, left: int) -> tuple[int, int, int, int]:
    coords = np.asarray(shape, dtype=int)
    grid[top + coords[:, 0], left + coords[:, 1]] = int(color)
    h, w = _shape_hw(shape)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _place_anywhere(grid: np.ndarray, shape, color: int, row_range: range, col_range: range, placed: list[tuple[int, int, int, int]], gap: int = 1):
    h, w = _shape_hw(shape)
    candidates = []
    for top in row_range:
        for left in col_range:
            if top + h > grid.shape[0] or left + w > grid.shape[1]:
                continue
            box = (top, left, top + h - 1, left + w - 1)
            if _bbox_gap_ok(box, placed, gap):
                candidates.append((top, left, box))
    if not candidates:
        return None
    top, left, box = random.choice(candidates)
    _draw_shape(grid, shape, color, top, left)
    placed.append(box)
    return {"top": int(top), "left": int(left), "box": box}


def _pick_templates(count: int):
    entries = random.sample(SHAPE_CATALOG, count)
    templates = []
    for entry in entries:
        orientations = sorted({_transform(entry["canonical"], reflect, rot) for reflect in (False, True) for rot in range(4)})
        shape = random.choice(orientations)
        opts = _query_options(shape)
        assert {opt["parity"] for opt in opts} == {0, 1}
        templates.append({"shape": shape, "options": opts})
    return templates


def _assign_query_options(templates: list[dict]) -> list[dict] | None:
    count = len(templates)
    parity_targets = [0, 1]
    while len(parity_targets) < count:
        parity_targets.append(random.randint(0, 1))
    random.shuffle(parity_targets)
    chosen = []
    for template, parity in zip(templates, parity_targets):
        candidates = [opt for opt in template["options"] if opt["parity"] == parity]
        if not candidates:
            return None
        chosen.append(random.choice(candidates))
    if not any(opt["reflect"] for opt in chosen):
        for i, template in enumerate(templates):
            replacements = [
                opt for opt in template["options"] if opt["parity"] == chosen[i]["parity"] and opt["reflect"]
            ]
            if replacements:
                chosen[i] = random.choice(replacements)
                break
    return chosen


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        info = _analyze_input(inp)
    except AssertionError:
        return False
    matches = info["matches"]
    if {m["parity"] for m in matches} != {0, 1}:
        return False
    if not all(m["shift_dc"] != 0 for m in matches):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    recolor_only = _render_from_matches(inp, matches, recolor=True, slide=False)
    slide_only = _render_from_matches(inp, matches, recolor=False, slide=True)
    all_near = _render_from_matches(inp, matches, recolor=True, slide=True, force_parity=0)
    all_far = _render_from_matches(inp, matches, recolor=True, slide=True, force_parity=1)
    if np.array_equal(recolor_only, expected):
        return False
    if np.array_equal(slide_only, expected):
        return False
    if np.array_equal(all_near, expected):
        return False
    if np.array_equal(all_far, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(13, 18)
    W = random.randint(16, 22)
    divider = random.randint(6, min(9, W - 7))
    template_count = random.randint(3, 4)
    templates = _pick_templates(template_count)
    query_choices = _assign_query_options(templates)
    if query_choices is None:
        return None

    legend = np.full((H, divider), BG, dtype=int)
    legend_boxes: list[tuple[int, int, int, int]] = []
    colors = random.sample(TEMPLATE_COLORS, template_count)
    for template, color in zip(templates, colors):
        placed = _place_anywhere(
            legend,
            template["shape"],
            color,
            range(0, H),
            range(0, divider),
            legend_boxes,
            gap=1,
        )
        if placed is None:
            return None
        template["color"] = int(color)

    out = np.full((H, W), BG, dtype=int)
    out[:, divider] = GREY
    out[:, :divider] = legend

    query_boxes: list[tuple[int, int, int, int]] = []
    query_layouts = []
    order = list(range(template_count))
    random.shuffle(order)
    for idx in order:
        template = templates[idx]
        choice = query_choices[idx]
        qshape = choice["shape"]
        h, w = _shape_hw(qshape)
        dest_left = divider + 1 if choice["parity"] == 0 else W - w
        placed = _place_anywhere(
            out,
            qshape,
            template["color"],
            range(0, H),
            range(dest_left, dest_left + 1),
            query_boxes,
            gap=1,
        )
        if placed is None:
            return None
        query_layouts.append(
            {
                "idx": idx,
                "shape": qshape,
                "color": template["color"],
                "parity": choice["parity"],
                "reflect": choice["reflect"],
                "rot": choice["rot"],
                "top": placed["top"],
                "out_left": placed["left"],
                "width": w,
            }
        )

    inp = np.full((H, W), BG, dtype=int)
    inp[:, divider] = GREY
    inp[:, :divider] = legend
    used_input_boxes: list[tuple[int, int, int, int]] = []
    for layout in query_layouts:
        shape = layout["shape"]
        h, w = _shape_hw(shape)
        if layout["parity"] == 0:
            cols = list(range(layout["out_left"] + 1, W - w + 1))
        else:
            cols = list(range(divider + 1, layout["out_left"]))
        random.shuffle(cols)
        placed = None
        for left in cols:
            box = (layout["top"], left, layout["top"] + h - 1, left + w - 1)
            if _bbox_gap_ok(box, used_input_boxes, gap=1):
                _draw_shape(inp, shape, RED, layout["top"], left)
                used_input_boxes.append(box)
                placed = left
                break
        if placed is None:
            return None

    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
