import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
DIVIDER = 9
BODY_COLORS = [1, 2, 3, 4, 6, 7]

SYMMETRIES = [(False, k) for k in range(4)] + [(True, k) for k in range(4)]
NONTRIVIAL_SYMMETRIES = [sym for sym in SYMMETRIES if sym != (False, 0)]
GEN_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[tuple[int, int]]:
    steps4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    steps8 = steps4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    steps = steps8 if connectivity == 8 else steps4
    out = []
    for dr, dc in steps:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 8) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if seen[nr, nc] or int(grid[nr, nc]) == int(bg):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            colors = {int(grid[rr, cc]) for rr, cc in cells}
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "colors": colors,
                }
            )
    return comps


def _crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _apply_symmetry(arr: np.ndarray, flip_ud: bool, rot_k: int) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, rot_k)


def _pair_templates(inputs: list[dict], outputs: list[dict]) -> list[tuple[dict, dict]]:
    remaining = outputs[:]
    pairs = []
    inputs_sorted = sorted(inputs, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    for inp in inputs_sorted:
        ir0, ic0, ir1, ic1 = inp["bbox"]
        icy = ir0 + ir1
        best_idx = None
        best_score = None
        for idx, out in enumerate(remaining):
            or0, oc0, or1, oc1 = out["bbox"]
            if oc0 <= ic1:
                continue
            overlap = min(ir1, or1) - max(ir0, or0) + 1
            row_gap = 0 if overlap > 0 else min(abs(or0 - ir1), abs(ir0 - or1))
            score = (row_gap, oc0 - ic1, abs((or0 + or1) - icy), abs(oc1 - ic1))
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            raise AssertionError("could not pair template input with output")
        pairs.append((inp, remaining.pop(best_idx)))
    return pairs


def _infer_symmetry(inp_patch: np.ndarray, out_patch: np.ndarray, bg: int) -> tuple[bool, int]:
    inp_mask = inp_patch != int(bg)
    out_mask = out_patch != int(bg)
    matches = []
    for sym in SYMMETRIES:
        if np.array_equal(_apply_symmetry(inp_mask, sym[0], sym[1]), out_mask):
            matches.append(sym)
    if len(matches) != 1:
        raise AssertionError("template transform must be uniquely determined")
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer numbered template transforms from the upper panel and apply them to matching marked objects below.

    Input structure:
    - The background is the dominant color, and a single full-width colored divider row splits the grid into an upper template zone and a lower work zone.
    - In the upper zone, each template pair has a left grey input shape containing N azure marker pixels and a right grey output shape showing the transformed silhouette.
    - In the lower zone, each colored foreground object is a connected component containing some azure marker pixels inside it.

    Rule:
    - Count the azure pixels inside each lower object to get its template number N.
    - In the template zone, pair each azure-marked input template with the nearest grey-only shape to its right, then infer the unique geometric symmetry that maps the full input silhouette to that output silhouette.
    - Apply the same symmetry to the entire lower object with count N, preserving its colors, and place the transformed object in the lower zone so its bounding-box center stays fixed.
    - Remove the whole template zone and divider from the output, keeping only the transformed lower zone.

    Key insight:
    Azure-count labels do not directly encode the action; they select an example pair whose before/after silhouettes reveal the symmetry to reuse on the matching work object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != int(bg)]
    if len(divider_rows) != 1:
        raise AssertionError("expected exactly one colored divider row")
    divider_r = int(divider_rows[0])

    top = g[:divider_r, :]
    bottom = g[divider_r + 1 :, :]
    if top.size == 0 or bottom.size == 0:
        raise AssertionError("both template and work zones must exist")

    top_comps = _find_components(top, bg=bg, connectivity=8)
    template_inputs = [comp for comp in top_comps if int(np.sum(_crop(top, comp["bbox"]) == AZURE)) > 0]
    template_outputs = [comp for comp in top_comps if comp["colors"] == {GREY}]
    if not template_inputs or len(template_inputs) != len(template_outputs):
        raise AssertionError("template zone must contain matched input/output pairs")

    mapping: dict[int, tuple[bool, int]] = {}
    for inp_comp, out_comp in _pair_templates(template_inputs, template_outputs):
        count = int(np.sum(_crop(top, inp_comp["bbox"]) == AZURE))
        if count in mapping:
            raise AssertionError("template counts must be unique")
        mapping[count] = _infer_symmetry(_crop(top, inp_comp["bbox"]), _crop(top, out_comp["bbox"]), bg)

    out = np.full_like(bottom, int(bg))
    work_comps = _find_components(bottom, bg=bg, connectivity=8)
    for comp in work_comps:
        patch = _crop(bottom, comp["bbox"])
        count = int(np.sum(patch == AZURE))
        if count not in mapping:
            raise AssertionError("every work object must match a template count")
        sym = mapping[count]
        moved = _apply_symmetry(patch, sym[0], sym[1])

        r0, c0, _r1, _c1 = comp["bbox"]
        h1, w1 = patch.shape
        h2, w2 = moved.shape
        top_num = 2 * int(r0) + int(h1) - int(h2)
        left_num = 2 * int(c0) + int(w1) - int(w2)
        if top_num % 2 != 0 or left_num % 2 != 0:
            raise AssertionError("center-preserving transform landed off-grid")
        nr0 = top_num // 2
        nc0 = left_num // 2
        if nr0 < 0 or nc0 < 0 or nr0 + h2 > bottom.shape[0] or nc0 + w2 > bottom.shape[1]:
            raise AssertionError("transformed object left the work zone")

        mask = moved != int(bg)
        target = out[nr0 : nr0 + h2, nc0 : nc0 + w2]
        if np.any((target != int(bg)) & mask):
            raise AssertionError("transformed work objects must not overlap")
        target[mask] = moved[mask]

    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in pts[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, h, w, 4):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if bool(mask[r, c])]


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = tuple((int(r), int(c)) for r, c in np.argwhere(mask))
    return mask.shape, coords


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _all_symmetries_unique(mask: np.ndarray) -> bool:
    sigs = {_shape_signature(_apply_symmetry(mask, sym[0], sym[1])) for sym in SYMMETRIES}
    return len(sigs) == len(SYMMETRIES)


def _generate_shape(min_markers: int, require_unique: bool, same_parity: bool) -> np.ndarray:
    dims = []
    for h in range(3, 6):
        for w in range(3, 6):
            if (h - 2) * (w - 2) < min_markers:
                continue
            if same_parity and (h % 2) != (w % 2):
                continue
            dims.append((h, w))
    if not dims:
        raise AssertionError("no valid dimensions for requested marker count")

    for _ in range(400):
        h, w = random.choice(dims)
        mask = np.ones((h, w), dtype=bool)
        boundary = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        random.shuffle(boundary)
        target_removals = random.randint(1, min(4, len(boundary) - 1))
        removed = 0
        for r, c in boundary:
            if removed >= target_removals:
                break
            if not random.random() < 0.7:
                continue
            cand = mask.copy()
            cand[r, c] = False
            if len(_interior_cells(cand)) < min_markers:
                continue
            if not _is_connected(cand):
                continue
            mask = cand
            removed += 1
        if removed == 0:
            continue
        mask = _tight_mask(mask)
        if same_parity and (mask.shape[0] % 2) != (mask.shape[1] % 2):
            continue
        if len(_interior_cells(mask)) < min_markers:
            continue
        if require_unique and not _all_symmetries_unique(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a valid asymmetric shape")


def _render_shape(mask: np.ndarray, color: int, markers: list[tuple[int, int]] | None = None) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    for r, c in markers or []:
        patch[int(r), int(c)] = AZURE
    return patch


def _make_template_pair(count: int, sym: tuple[bool, int]) -> dict:
    mask = _generate_shape(min_markers=count, require_unique=True, same_parity=(sym[1] % 2 == 1))
    markers = random.sample(_interior_cells(mask), count)
    inp = _render_shape(mask, GREY, markers)
    out = _render_shape(_apply_symmetry(mask, sym[0], sym[1]), GREY)
    pair_h = max(inp.shape[0], out.shape[0])
    pair_w = inp.shape[1] + 1 + out.shape[1]
    pair = np.full((pair_h, pair_w), BG, dtype=int)
    inp_r = (pair_h - inp.shape[0]) // 2
    out_r = (pair_h - out.shape[0]) // 2
    pair[inp_r : inp_r + inp.shape[0], : inp.shape[1]] = inp
    start_c = inp.shape[1] + 1
    pair[out_r : out_r + out.shape[0], start_c : start_c + out.shape[1]] = out
    return {"count": count, "sym": sym, "pair": pair}


def _make_work_object(count: int, sym: tuple[bool, int], color: int) -> dict:
    mask = _generate_shape(min_markers=count, require_unique=False, same_parity=(sym[1] % 2 == 1))
    markers = random.sample(_interior_cells(mask), count)
    inp = _render_shape(mask, color, markers)
    out = _apply_symmetry(inp, sym[0], sym[1])
    return {"count": count, "input": inp, "output": out}


def _plan_template_layout(pairs: list[np.ndarray]) -> dict:
    order = pairs[:]
    random.shuffle(order)
    preferred_rows = 2 if len(order) >= 3 else random.choice([1, 2])
    for row_count in [preferred_rows, 1 if preferred_rows == 2 else 2]:
        rows = [[] for _ in range(row_count)]
        for idx, pair in enumerate(order):
            rows[idx % row_count].append(pair)
        row_gaps = []
        row_heights = []
        row_widths = []
        for row in rows:
            gaps = [random.randint(2, 4) for _ in range(max(0, len(row) - 1))]
            row_gaps.append(gaps)
            row_heights.append(max(arr.shape[0] for arr in row))
            row_widths.append(sum(arr.shape[1] for arr in row) + sum(gaps))
        row_gap = random.randint(1, 2)
        required_w = max(row_widths) + 2
        required_h = sum(row_heights) + row_gap * (row_count - 1) + 2
        if required_w <= 30 and required_h <= 18:
            return {
                "rows": rows,
                "row_gaps": row_gaps,
                "row_heights": row_heights,
                "row_gap": row_gap,
                "required_w": required_w,
                "required_h": required_h,
            }
    raise RuntimeError("failed to plan template layout")


def _render_template_zone(layout: dict, width: int) -> np.ndarray:
    top = np.full((layout["required_h"], width), BG, dtype=int)
    y = 1
    for row, gaps, row_h in zip(layout["rows"], layout["row_gaps"], layout["row_heights"]):
        row_w = sum(arr.shape[1] for arr in row) + sum(gaps)
        x = random.randint(1, width - row_w - 1)
        for idx, arr in enumerate(row):
            dy = random.randint(0, row_h - arr.shape[0])
            top[y + dy : y + dy + arr.shape[0], x : x + arr.shape[1]] = arr
            x += arr.shape[1]
            if idx < len(gaps):
                x += gaps[idx]
        y += row_h + layout["row_gap"]
    return top


def _can_place_coords(coords: np.ndarray, r0: int, c0: int, occ: np.ndarray, pad: int) -> bool:
    h, w = occ.shape
    for rr, cc in coords:
        r = int(r0 + rr)
        c = int(c0 + cc)
        if r < 0 or c < 0 or r >= h or c >= w:
            return False
        r1 = max(0, r - pad)
        r2 = min(h, r + pad + 1)
        c1 = max(0, c - pad)
        c2 = min(w, c + pad + 1)
        if np.any(occ[r1:r2, c1:c2]):
            return False
    return True


def _mark_coords(coords: np.ndarray, r0: int, c0: int, occ: np.ndarray) -> None:
    for rr, cc in coords:
        occ[int(r0 + rr), int(c0 + cc)] = True


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    if np.any((target != BG) & mask):
        raise AssertionError("patch placement overlapped")
    target[mask] = patch[mask]


def _place_work_objects(objects: list[dict], height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    occ_in = np.zeros((height, width), dtype=bool)
    occ_out = np.zeros((height, width), dtype=bool)

    order = sorted(range(len(objects)), key=lambda idx: max(int((objects[idx]["input"] != BG).sum()), int((objects[idx]["output"] != BG).sum())), reverse=True)
    placements: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        obj = objects[idx]
        in_patch = obj["input"]
        out_patch = obj["output"]
        h1, w1 = in_patch.shape
        h2, w2 = out_patch.shape
        in_coords = np.argwhere(in_patch != BG)
        out_coords = np.argwhere(out_patch != BG)

        candidates = [(r, c) for r in range(height - h1 + 1) for c in range(width - w1 + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            top_num = 2 * int(r0) + int(h1) - int(h2)
            left_num = 2 * int(c0) + int(w1) - int(w2)
            if top_num % 2 != 0 or left_num % 2 != 0:
                continue
            rr = top_num // 2
            cc = left_num // 2
            if rr < 0 or cc < 0 or rr + h2 > height or cc + w2 > width:
                continue
            if not _can_place_coords(in_coords, r0, c0, occ_in, pad=1):
                continue
            if not _can_place_coords(out_coords, rr, cc, occ_out, pad=1):
                continue
            placements[idx] = (r0, c0, rr, cc)
            _mark_coords(in_coords, r0, c0, occ_in)
            _mark_coords(out_coords, rr, cc, occ_out)
            placed = True
            break
        if not placed:
            return None

    for idx, (r0, c0, rr, cc) in placements.items():
        _place_patch(inp, objects[idx]["input"], r0, c0)
        _place_patch(out, objects[idx]["output"], rr, cc)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    template_count = random.randint(2, 4)
    syms = random.sample(NONTRIVIAL_SYMMETRIES, template_count)
    templates = [_make_template_pair(idx + 1, sym) for idx, sym in enumerate(syms)]
    template_layout = _plan_template_layout([tpl["pair"] for tpl in templates])

    extra_count = random.randint(0, min(2, len(BODY_COLORS) - template_count))
    object_counts = list(range(1, template_count + 1))
    if extra_count:
        object_counts.extend(random.choices(object_counts, k=extra_count))
    total_objects = len(object_counts)
    colors = random.sample(BODY_COLORS, total_objects)
    sym_by_count = {tpl["count"]: tpl["sym"] for tpl in templates}
    objects = [_make_work_object(count, sym_by_count[count], color) for count, color in zip(object_counts, colors)]

    columns = 3 if total_objects >= 5 else random.choice([2, 3]) if total_objects >= 3 else 2
    rows_needed = (total_objects + columns - 1) // columns
    bottom_min_w = columns * 7 + 2
    bottom_min_h = rows_needed * 7 + 2

    full_min_w = max(template_layout["required_w"], bottom_min_w)
    max_bottom_h = 30 - template_layout["required_h"] - 1
    if full_min_w > 30 or bottom_min_h > max_bottom_h:
        return None

    for _ in range(60):
        width = random.randint(full_min_w, 30)
        bottom_h = random.randint(bottom_min_h, max_bottom_h)
        placed = _place_work_objects(objects, bottom_h, width)
        if placed is None:
            continue
        bottom_in, bottom_out = placed
        top = _render_template_zone(template_layout, width)
        full = np.full((top.shape[0] + 1 + bottom_h, width), BG, dtype=int)
        full[: top.shape[0], :] = top
        full[top.shape[0], :] = DIVIDER
        full[top.shape[0] + 1 :, :] = bottom_in
        if not np.array_equal(main(full), bottom_out):
            continue
        return full
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 5")
