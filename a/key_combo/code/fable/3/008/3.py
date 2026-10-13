from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

TEMPLATE_SIZE = 3
TEMPLATE_GAP = 1
TEMPLATE_STRIDE = TEMPLATE_SIZE + TEMPLATE_GAP
UNKNOWN = -1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            q: deque[Coord] = deque([(int(sr), int(sc))])
            seen[sr, sc] = True
            cells: list[Coord] = []
            while q:
                r, c = q.popleft()
                cells.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            out.append(np.array(cells, dtype=int))
    return out


def _component_boxes_by_color(g: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, g.flat)) - {int(bg)}):
        for coords in _components(g == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_solid_box(g: np.ndarray, box: BBox, color: int) -> bool:
    r0, c0, r1, c1 = map(int, box)
    return bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _solid_3x3_components(g: np.ndarray, bg: int) -> list[dict]:
    out = []
    for comp in _component_boxes_by_color(g, bg):
        box = comp["bbox"]
        if (
            _bbox_h(box) == TEMPLATE_SIZE
            and _bbox_w(box) == TEMPLATE_SIZE
            and int(comp["area"]) == TEMPLATE_SIZE * TEMPLATE_SIZE
            and _is_solid_box(g, box, int(comp["color"]))
        ):
            out.append(comp)
    return out


def _find_template_lattice(g: np.ndarray, bg: int) -> tuple[list[int], list[int], dict[tuple[int, int], int], BBox]:
    comps = _solid_3x3_components(g, bg)
    assert comps, "expected visible 3x3 template components"

    groups: dict[tuple[int, int], list[dict]] = {}
    for comp in comps:
        r0, c0, _r1, _c1 = comp["bbox"]
        groups.setdefault((int(r0) % TEMPLATE_STRIDE, int(c0) % TEMPLATE_STRIDE), []).append(comp)

    candidates = []
    for group in groups.values():
        if len(group) < 4:
            continue
        rows = sorted({int(comp["bbox"][0]) for comp in group})
        cols = sorted({int(comp["bbox"][1]) for comp in group})
        if len(rows) < 3 or len(cols) < 3:
            continue
        if any(rows[i + 1] - rows[i] != TEMPLATE_STRIDE for i in range(len(rows) - 1)):
            continue
        if any(cols[i + 1] - cols[i] != TEMPLATE_STRIDE for i in range(len(cols) - 1)):
            continue

        row_i = {r: i for i, r in enumerate(rows)}
        col_i = {c: j for j, c in enumerate(cols)}
        colors: dict[tuple[int, int], int] = {}
        ok = True
        for comp in group:
            r0, c0, _r1, _c1 = comp["bbox"]
            key = (row_i[int(r0)], col_i[int(c0)])
            if key in colors:
                ok = False
                break
            colors[key] = int(comp["color"])
        if not ok:
            continue
        if len(colors) == len(rows) * len(cols):
            continue

        box = (
            int(rows[0]),
            int(cols[0]),
            int(rows[-1] + TEMPLATE_SIZE - 1),
            int(cols[-1] + TEMPLATE_SIZE - 1),
        )
        candidates.append((len(colors), rows, cols, colors, box))

    assert candidates, "expected one incomplete regular template lattice"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_count = candidates[0][0]
    best = [item for item in candidates if item[0] == best_count]
    assert len(best) == 1, "template lattice must be unique"
    _count, rows, cols, colors, box = best[0]
    return rows, cols, colors, box


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _find_control_bbox(g: np.ndarray, bg: int, rows: int, cols: int, lattice_box: BBox) -> BBox:
    candidates: list[BBox] = []
    for coords in _components(g != int(bg)):
        box = _bbox_coords(coords)
        if _bbox_h(box) != int(rows) or _bbox_w(box) != int(cols):
            continue
        if _boxes_overlap(box, lattice_box):
            continue
        r0, c0, r1, c1 = map(int, box)
        if not bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] != int(bg))):
            continue
        if int(coords.shape[0]) != int(rows) * int(cols):
            continue
        candidates.append(box)

    assert len(candidates) == 1, "expected one dense control matrix matching the lattice dimensions"
    return candidates[0]


def _find_occluder_color(g: np.ndarray, control_box: BBox, templates: np.ndarray, bg: int) -> int:
    r0, c0, r1, c1 = map(int, control_box)
    in_control = np.zeros(g.shape, dtype=bool)
    in_control[r0 : r1 + 1, c0 : c1 + 1] = True
    control = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    candidates = []
    for color in sorted(set(map(int, g[r0 : r1 + 1, c0 : c1 + 1].flat)) - {int(bg)}):
        color_mask = g == int(color)
        if not bool(np.all(in_control[color_mask])):
            continue
        try:
            recovered = _recover_control(templates, control, int(color))
        except Exception:
            continue
        if int(color) in set(map(int, recovered.flat)):
            continue
        candidates.append(int(color))
    assert len(candidates) == 1, "expected one matrix-only occluder color"
    return int(candidates[0])


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    row_anchors, col_anchors, visible_templates, lattice_box = _find_template_lattice(g, bg)
    rows = len(row_anchors)
    cols = len(col_anchors)
    control_box = _find_control_bbox(g, bg, rows, cols, lattice_box)

    templates = np.full((rows, cols), UNKNOWN, dtype=int)
    for (i, j), color in visible_templates.items():
        templates[int(i), int(j)] = int(color)

    occluder = _find_occluder_color(g, control_box, templates, bg)

    r0, c0, r1, c1 = map(int, control_box)
    control = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "bg": int(bg),
        "rows": int(rows),
        "cols": int(cols),
        "templates": templates,
        "control": control,
        "occluder": int(occluder),
        "control_box": control_box,
        "lattice_box": lattice_box,
    }


def _recover_control(templates: np.ndarray, control: np.ndarray, occluder: int) -> np.ndarray:
    rows, cols = control.shape
    out = control.copy()
    unknown = out == int(occluder)
    assert bool(unknown.any()), "expected at least one occluded control-matrix cell"

    for i in range(rows):
        for j in range(cols):
            if not bool(unknown[i, j]) and int(templates[i, j]) != UNKNOWN:
                assert int(out[i, j]) == int(templates[i, j]), "visible matrix and template evidence disagree"

    changed = True
    while bool(unknown.any()) and changed:
        changed = False
        for i in range(rows):
            for j in range(cols):
                if not bool(unknown[i, j]):
                    continue
                evidence: list[int] = []
                if int(templates[i, j]) != UNKNOWN:
                    evidence.append(int(templates[i, j]))
                mirror_j = int(cols - 1 - j)
                if not bool(unknown[i, mirror_j]):
                    evidence.append(int(out[i, mirror_j]))
                if not evidence:
                    continue
                assert all(int(x) == int(evidence[0]) for x in evidence), "template and mirror evidence disagree"
                out[i, j] = int(evidence[0])
                unknown[i, j] = False
                changed = True

    assert not bool(unknown.any()), "occluded cells must be recoverable from template or mirror evidence"
    for i in range(rows):
        for j in range(cols):
            assert int(out[i, j]) == int(out[i, cols - 1 - j]), "completed control matrix must be left-right symmetric"
            if int(templates[i, j]) != UNKNOWN:
                assert int(out[i, j]) == int(templates[i, j]), "completed matrix must match visible templates"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the occluded color legend by combining lattice-position evidence with left-right mirror evidence.

    Input structure:
    - The background contains an incomplete regular lattice of same-shape solid 3x3 colored templates, spaced by one background row and column.
    - Elsewhere there is a dense control matrix with the same row and column count as the lattice; its true colors are left-right symmetric.
    - Some control cells are covered by a unique matrix-only occluder color, while some corresponding template slots in the large lattice are missing.

    Rule:
    - Each control cell's true color equals the color of the template at the same lattice row and column.
    - For every occluded cell, use its same-position visible template when that template exists; if that template slot is missing, use the cell's left-right mirror mate in the same control-matrix row.
    - Iterate these two evidence sources until all occluded cells are filled, then return only the completed control matrix.

    Key insight:
    - The small matrix is the answer being repaired, not an instruction to repaint the lattice; template-position evidence and mirror symmetry have complementary gaps.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _recover_control(info["templates"], info["control"], info["occluder"])


# ---------------- generator ----------------

PALETTE = list(range(10))
GEN_DIMS = [(3, 4), (4, 4), (4, 5), (5, 4), (5, 5)]
GEN_ATTEMPTS = 500


def _paste_solid_template(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE] = int(color)


def _layout(rows: int, cols: int, rng: random.Random) -> tuple[int, int, int, int, int, int]:
    lattice_h = rows * TEMPLATE_STRIDE - TEMPLATE_GAP
    lattice_w = cols * TEMPLATE_STRIDE - TEMPLATE_GAP
    gap = rng.randint(2, 4)
    margin = rng.randint(1, 3)
    side = rng.choice(["right", "left", "below", "above"])

    if side in ("right", "left"):
        min_h = margin * 2 + max(lattice_h, rows)
        min_w = margin * 2 + lattice_w + gap + cols
    else:
        min_h = margin * 2 + lattice_h + gap + rows
        min_w = margin * 2 + max(lattice_w, cols)

    if min_h > 30 or min_w > 30:
        raise ValueError("layout too large")
    h = min_h + rng.randint(0, 30 - min_h)
    w = min_w + rng.randint(0, 30 - min_w)

    if side == "right":
        lat_r = rng.randint(margin, h - margin - lattice_h)
        lat_c = rng.randint(margin, w - margin - lattice_w - gap - cols)
        ctl_r = rng.randint(margin, h - margin - rows)
        ctl_c = rng.randint(lat_c + lattice_w + gap, w - margin - cols)
    elif side == "left":
        ctl_r = rng.randint(margin, h - margin - rows)
        ctl_c = rng.randint(margin, w - margin - cols - gap - lattice_w)
        lat_r = rng.randint(margin, h - margin - lattice_h)
        lat_c = rng.randint(ctl_c + cols + gap, w - margin - lattice_w)
    elif side == "below":
        lat_r = rng.randint(margin, h - margin - lattice_h - gap - rows)
        lat_c = rng.randint(margin, w - margin - lattice_w)
        ctl_r = rng.randint(lat_r + lattice_h + gap, h - margin - rows)
        ctl_c = rng.randint(margin, w - margin - cols)
    else:
        ctl_r = rng.randint(margin, h - margin - rows - gap - lattice_h)
        ctl_c = rng.randint(margin, w - margin - cols)
        lat_r = rng.randint(ctl_r + rows + gap, h - margin - lattice_h)
        lat_c = rng.randint(margin, w - margin - lattice_w)

    return h, w, lat_r, lat_c, ctl_r, ctl_c


def _make_control(rows: int, cols: int, colors: list[int], rng: random.Random) -> np.ndarray:
    half = (cols + 1) // 2
    control = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        row_choices = rng.sample(colors, k=min(len(colors), max(2, half)))
        for j in range(half):
            color = row_choices[j % len(row_choices)]
            if rng.random() < 0.35:
                color = rng.choice(colors)
            control[i, j] = int(color)
            control[i, cols - 1 - j] = int(color)

    for color in colors:
        if int(color) not in set(map(int, control.flat)):
            i = rng.randrange(rows)
            j = rng.randrange(half)
            control[i, j] = int(color)
            control[i, cols - 1 - j] = int(color)
    return control


def _occlusion_and_missing(rows: int, cols: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    occ = np.zeros((rows, cols), dtype=bool)
    present = np.ones((rows, cols), dtype=bool)

    pair_cols = [(j, cols - 1 - j) for j in range(cols // 2)]
    left_a, right_a = rng.choice(pair_cols)
    row_a = rng.randrange(rows)
    for j in range(left_a, right_a + 1):
        occ[row_a, j] = True
    present[row_a, left_a] = False
    present[row_a, right_a] = True
    if cols % 2 == 1 and left_a <= cols // 2 <= right_a:
        present[row_a, cols // 2] = True

    row_b_choices = [r for r in range(rows) if abs(r - row_a) == 1]
    if not row_b_choices:
        return occ, present
    row_b = rng.choice(row_b_choices)
    occ[row_b, left_a] = True
    present[row_b, left_a] = False
    present[row_b, right_a] = True

    if rng.random() < 0.55:
        center_cols = [cols // 2] if cols % 2 == 1 else []
        extra_options = [(row_a, j) for j in center_cols]
        extra_options += [(row_b, j) for j in range(max(0, left_a - 1), min(cols, left_a + 2))]
        rng.shuffle(extra_options)
        for i, j in extra_options[: rng.randint(0, min(2, len(extra_options)))]:
            occ[i, j] = True
            if j == cols // 2 and cols % 2 == 1:
                present[i, j] = True

    for _ in range(rng.randint(1, 4)):
        i = rng.randrange(rows)
        j = rng.randrange(cols)
        if occ[i, j]:
            continue
        mirror_j = cols - 1 - j
        if occ[i, mirror_j]:
            continue
        present[i, j] = False

    return occ, present


def _no_full_3x3_same_color(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            if bool(np.all(patch == int(patch[0, 0]))):
                return False
    return True


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _iterative_depth_needed(templates: np.ndarray, observed: np.ndarray, occluder: int) -> bool:
    unknown = observed == int(occluder)
    first_wave = np.zeros_like(unknown, dtype=bool)
    rows, cols = unknown.shape
    for i in range(rows):
        for j in range(cols):
            first_wave[i, j] = bool(unknown[i, j] and int(templates[i, j]) != UNKNOWN)
    for i in range(rows):
        for j in range(cols):
            if not bool(unknown[i, j]) or int(templates[i, j]) != UNKNOWN:
                continue
            mirror_j = cols - 1 - j
            if bool(first_wave[i, mirror_j]):
                return True
    return False


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, occ_mask: np.ndarray, present: np.ndarray, occluder: int) -> bool:
    try:
        info = _analyze(inp)
        out = _recover_control(info["templates"], info["control"], info["occluder"])
    except Exception:
        return False

    if int(info["occluder"]) != int(occluder):
        return False
    if not np.array_equal(out, expected):
        return False
    if not np.array_equal(info["templates"] != UNKNOWN, present):
        return False

    templates = info["templates"]
    observed = info["control"]
    rows, cols = observed.shape
    unknown = observed == int(occluder)

    if not np.array_equal(unknown, occ_mask):
        return False
    if not _connected(unknown):
        return False

    has_template_direct = bool(np.any(unknown & (templates != UNKNOWN)))
    has_mirror_direct = False
    has_paired_occlusion = False
    has_missing_template_occluded = bool(np.any(unknown & (templates == UNKNOWN)))
    for i in range(rows):
        for j in range(cols):
            if not bool(unknown[i, j]):
                continue
            mirror_j = cols - 1 - j
            if bool(unknown[i, mirror_j]) and mirror_j != j:
                has_paired_occlusion = True
            if int(templates[i, j]) == UNKNOWN and not bool(unknown[i, mirror_j]):
                has_mirror_direct = True

    if not (has_template_direct and has_mirror_direct and has_paired_occlusion and has_missing_template_occluded):
        return False
    if not _iterative_depth_needed(templates, observed, int(occluder)):
        return False
    if set(map(int, expected.flat)) - {int(x) for x in templates.flat if int(x) != UNKNOWN}:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(GEN_DIMS)
    bg = rng.choice(PALETTE)
    occluder = rng.choice([c for c in PALETTE if c != bg])
    usable = [c for c in PALETTE if c not in (bg, occluder)]
    color_count = rng.randint(3, min(6, len(usable)))
    colors = rng.sample(usable, color_count)

    control = _make_control(rows, cols, colors, rng)
    occ_mask, present = _occlusion_and_missing(rows, cols, rng)

    if not _connected(occ_mask):
        return None
    if not any(not bool(present[i, j]) for i in range(rows) for j in range(cols)):
        return None
    if any(not bool(present[i].any()) for i in range(rows)):
        return None
    if any(not bool(present[:, j].any()) for j in range(cols)):
        return None

    visible_template_colors = {int(control[i, j]) for i in range(rows) for j in range(cols) if bool(present[i, j])}
    if set(map(int, control.flat)) - visible_template_colors:
        return None

    observed = control.copy()
    observed[occ_mask] = int(occluder)
    if not _no_full_3x3_same_color(observed):
        return None

    templates = np.where(present, control, UNKNOWN)
    if not _iterative_depth_needed(templates, observed, int(occluder)):
        return None

    try:
        h, w, lat_r, lat_c, ctl_r, ctl_c = _layout(rows, cols, rng)
    except ValueError:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if not bool(present[i, j]):
                continue
            rr = lat_r + i * TEMPLATE_STRIDE
            cc = lat_c + j * TEMPLATE_STRIDE
            _paste_solid_template(grid, rr, cc, int(control[i, j]))

    grid[ctl_r : ctl_r + rows, ctl_c : ctl_c + cols] = observed
    if _get_bg(grid) != int(bg):
        return None

    if not _semantic_checks(grid, control, occ_mask, present, int(occluder)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
