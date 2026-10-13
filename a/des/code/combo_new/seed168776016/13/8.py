from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "id": int(next_id),
                    "color": int(color),
                    "coords": [(int(rr), int(cc)) for rr, cc in coords],
                    "coord_set": {(int(rr), int(cc)) for rr, cc in coords},
                    "area": int(len(coords)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
            next_id += 1

    return comps


def _is_ink(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return int(comp["area"]) == 4 and int(r1 - r0 + 1) == 2 and int(c1 - c0 + 1) == 2


def _components_adjacent(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIRS4:
            if (int(r + dr), int(c + dc)) in b:
                return True
    return False


def _unique_contact_pixel(stamp: dict, ink: dict) -> tuple[int, int]:
    contacts: list[tuple[int, int]] = []
    ink_cells = ink["coord_set"]
    for r, c in stamp["coords"]:
        for dr, dc in DIRS4:
            if (int(r + dr), int(c + dc)) in ink_cells:
                contacts.append((int(r), int(c)))
                break
    assert len(contacts) == 1, "expected a unique stamp pixel adjacent to the ink pad"
    return contacts[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each irregular source shape at its matching anchor using the adjacent 2x2 pad's color.

    Input structure:
    - Background is black.
    - Foreground components are single-color 4-connected objects of three roles:
      irregular stamps, solid 2x2 ink pads, and singleton anchors.
    - Every stamp touches exactly one ink pad by 4-neighbor adjacency, and every
      ink-pad color appears once more as a singleton anchor elsewhere in the grid.

    Rule:
    - Classify all non-black monochrome components by shape and size.
    - Pair each stamp with the unique adjacent ink pad, then find the anchor with
      that ink pad's color.
    - Translate the stamp so its unique pixel touching the ink pad lands on the
      anchor, and redraw the whole stamp in the ink-pad color.
    - Remove the original stamps and ink pads; keep the anchors.

    Key insight:
    - The solution requires chaining local adjacency (stamp to ink pad) with
      remote color matching (ink pad to anchor), then recoloring the moved shape
      with the ink-pad color rather than the original stamp color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    comps = _components4(g)
    anchors: list[dict] = []
    inks: list[dict] = []
    stamps: list[dict] = []

    for comp in comps:
        area = int(comp["area"])
        if area == 1:
            anchors.append(comp)
        elif _is_ink(comp):
            inks.append(comp)
        else:
            assert area >= 4, "non-anchor components must be stamps or ink pads"
            stamps.append(comp)

    assert stamps and inks and anchors, "expected at least one stamp, ink pad, and anchor"
    assert len(stamps) == len(inks) == len(anchors), "role counts must match"

    stamp_colors = {int(comp["color"]) for comp in stamps}
    assert all(int(ink["color"]) not in stamp_colors for ink in inks), "ink colors must differ from stamp colors"

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for anchor in anchors:
        color = int(anchor["color"])
        assert color not in anchors_by_color, "expected a unique anchor for each ink color"
        anchors_by_color[color] = tuple(map(int, anchor["coords"][0]))

    inks_by_color: dict[int, dict] = {}
    for ink in inks:
        color = int(ink["color"])
        assert color not in inks_by_color, "expected a unique ink pad for each color"
        inks_by_color[color] = ink

    assert set(anchors_by_color) == set(inks_by_color), "anchor colors must match ink-pad colors"

    paired_ink_ids: set[int] = set()
    pairs: list[tuple[dict, dict, tuple[int, int], tuple[int, int]]] = []
    for stamp in stamps:
        adjacent_inks = [ink for ink in inks if _components_adjacent(stamp["coord_set"], ink["coord_set"])]
        assert len(adjacent_inks) == 1, "each stamp must touch exactly one ink pad"
        ink = adjacent_inks[0]
        paired_ink_ids.add(int(ink["id"]))
        contact = _unique_contact_pixel(stamp, ink)
        anchor = anchors_by_color[int(ink["color"])]
        pairs.append((stamp, ink, contact, anchor))

    assert len(paired_ink_ids) == len(inks), "each ink pad must pair with exactly one stamp"

    out = np.zeros((h, w), dtype=int)
    for color, (r, c) in anchors_by_color.items():
        out[int(r), int(c)] = int(color)

    for stamp, ink, contact, anchor in pairs:
        dr = int(anchor[0] - contact[0])
        dc = int(anchor[1] - contact[1])
        color = int(ink["color"])
        for r, c in stamp["coords"]:
            rr, cc = int(r + dr), int(c + dc)
            assert 0 <= rr < h and 0 <= cc < w, "translated stamp must stay within bounds"
            current = int(out[rr, cc])
            assert current in (BG, color), "translated stamps must not overlap mismatched anchors or other stamps"
            out[rr, cc] = color

    return out


def _transform_point_square(point: tuple[int, int], size: int, rot: int, flip_lr: bool) -> tuple[int, int]:
    r, c = map(int, point)
    for _ in range(int(rot % 4)):
        r, c = int(c), int(size - 1 - r)
    if flip_lr:
        c = int(size - 1 - c)
    return int(r), int(c)


def _transform_cells_square(
    cells: set[tuple[int, int]], size: int, rot: int, flip_lr: bool
) -> set[tuple[int, int]]:
    return {_transform_point_square(cell, size, rot, flip_lr) for cell in cells}


def _make_pair_template() -> dict | None:
    for _ in range(200):
        canvas = random.randint(7, 9)
        ink_r = canvas // 2 - 1
        ink_c = canvas // 2 - 1
        ink = {
            (int(ink_r), int(ink_c)),
            (int(ink_r), int(ink_c + 1)),
            (int(ink_r + 1), int(ink_c)),
            (int(ink_r + 1), int(ink_c + 1)),
        }
        contact_options = {
            (int(ink_r - 1), int(ink_c)),
            (int(ink_r - 1), int(ink_c + 1)),
            (int(ink_r + 2), int(ink_c)),
            (int(ink_r + 2), int(ink_c + 1)),
            (int(ink_r), int(ink_c - 1)),
            (int(ink_r + 1), int(ink_c - 1)),
            (int(ink_r), int(ink_c + 2)),
            (int(ink_r + 1), int(ink_c + 2)),
        }
        contact = random.choice(tuple(contact_options))
        forbidden = set(contact_options)
        forbidden.remove(contact)

        target_area = random.randint(4, 8)
        stamp = {contact}
        while len(stamp) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in stamp:
                for dr, dc in DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < canvas and 0 <= nc < canvas):
                        continue
                    if (nr, nc) in stamp or (nr, nc) in ink or (nr, nc) in forbidden:
                        continue
                    frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            stamp.add(random.choice(tuple(frontier)))
        if len(stamp) != target_area:
            continue

        rot = random.randrange(4)
        flip_lr = bool(random.randrange(2))
        stamp = _transform_cells_square(stamp, canvas, rot, flip_lr)
        ink = _transform_cells_square(ink, canvas, rot, flip_lr)
        contact = _transform_point_square(contact, canvas, rot, flip_lr)

        stamp_r0, stamp_c0, stamp_r1, stamp_c1 = _bbox(stamp)
        stamp_h = int(stamp_r1 - stamp_r0 + 1)
        stamp_w = int(stamp_c1 - stamp_c0 + 1)
        if stamp_h < 2 or stamp_w < 2:
            continue
        if len(stamp) == stamp_h * stamp_w:
            continue

        union = stamp | ink
        union_r0, union_c0, union_r1, union_c1 = _bbox(union)
        stamp_rel = [(int(r - union_r0), int(c - union_c0)) for r, c in stamp]
        ink_rel = [(int(r - union_r0), int(c - union_c0)) for r, c in ink]
        contact_rel = (int(contact[0] - union_r0), int(contact[1] - union_c0))
        out_rel = [(int(r - contact_rel[0]), int(c - contact_rel[1])) for r, c in stamp_rel]

        out_rows = [int(r) for r, _ in out_rel]
        out_cols = [int(c) for _, c in out_rel]
        return {
            "stamp_rel": stamp_rel,
            "ink_rel": ink_rel,
            "pair_shape": (int(union_r1 - union_r0 + 1), int(union_c1 - union_c0 + 1)),
            "pair_cells": [(int(r), int(c)) for r, c in sorted(stamp_rel + ink_rel)],
            "out_rel": [(int(r), int(c)) for r, c in sorted(out_rel)],
            "out_span": (
                int(max(out_rows) - min(out_rows) + 1),
                int(max(out_cols) - min(out_cols) + 1),
            ),
        }

    return None


def _mark_buffer(blocked: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS8 + [(0, 0)]:
            nr, nc = int(rr + dr), int(cc + dc)
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_output_anchor(template: dict, blocked_input: np.ndarray, blocked_output: np.ndarray) -> tuple[int, int] | None:
    h, w = blocked_input.shape
    min_dr = min(int(r) for r, _ in template["out_rel"])
    max_dr = max(int(r) for r, _ in template["out_rel"])
    min_dc = min(int(c) for _, c in template["out_rel"])
    max_dc = max(int(c) for _, c in template["out_rel"])

    row_lo = int(-min_dr)
    row_hi = int(h - 1 - max_dr)
    col_lo = int(-min_dc)
    col_hi = int(w - 1 - max_dc)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    candidates = [(int(r), int(c)) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(candidates)
    for ar, ac in candidates:
        if blocked_input[ar, ac]:
            continue
        out_cells = [(int(ar + dr), int(ac + dc)) for dr, dc in template["out_rel"]]
        if any(blocked_output[r, c] for r, c in out_cells):
            continue
        _mark_buffer(blocked_input, {(int(ar), int(ac))})
        _mark_buffer(blocked_output, out_cells)
        return int(ar), int(ac)
    return None


def _place_source_pair(template: dict, blocked_input: np.ndarray) -> tuple[int, int] | None:
    h, w = blocked_input.shape
    pair_h, pair_w = map(int, template["pair_shape"])
    candidates = [(int(r), int(c)) for r in range(h - pair_h + 1) for c in range(w - pair_w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        cells = [(int(r0 + r), int(c0 + c)) for r, c in template["pair_cells"]]
        if any(blocked_input[r, c] for r, c in cells):
            continue
        _mark_buffer(blocked_input, cells)
        return int(r0), int(c0)
    return None


def _build_expected(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        ar, ac = map(int, spec["anchor"])
        out[ar, ac] = int(spec["ink_color"])
    for spec in specs:
        ar, ac = map(int, spec["anchor"])
        color = int(spec["ink_color"])
        for dr, dc in spec["out_rel"]:
            out[int(ar + dr), int(ac + dc)] = color
    return out


def _build_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        ar, ac = map(int, spec["anchor"])
        grid[ar, ac] = int(spec["ink_color"])
    for spec in specs:
        sr0, sc0 = map(int, spec["source_origin"])
        for r, c in spec["stamp_rel"]:
            grid[int(sr0 + r), int(sc0 + c)] = int(spec["stamp_color"])
        for r, c in spec["ink_rel"]:
            grid[int(sr0 + r), int(sc0 + c)] = int(spec["ink_color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    n_pairs = random.randint(2, 4)

    available = list(range(1, 10))
    random.shuffle(available)
    needed = 2 * n_pairs
    if needed > len(available):
        return None

    stamp_colors = available[:n_pairs]
    ink_colors = available[n_pairs:needed]

    specs: list[dict] = []
    for stamp_color, ink_color in zip(stamp_colors, ink_colors):
        template = _make_pair_template()
        if template is None:
            return None
        template["stamp_color"] = int(stamp_color)
        template["ink_color"] = int(ink_color)
        specs.append(template)

    blocked_input = np.zeros((h, w), dtype=bool)
    blocked_output = np.zeros((h, w), dtype=bool)

    anchor_order = sorted(
        range(len(specs)),
        key=lambda i: int(specs[i]["out_span"][0]) * int(specs[i]["out_span"][1]),
        reverse=True,
    )
    for idx in anchor_order:
        anchor = _place_output_anchor(specs[idx], blocked_input, blocked_output)
        if anchor is None:
            return None
        specs[idx]["anchor"] = (int(anchor[0]), int(anchor[1]))

    source_order = sorted(
        range(len(specs)),
        key=lambda i: int(specs[i]["pair_shape"][0]) * int(specs[i]["pair_shape"][1]),
        reverse=True,
    )
    for idx in source_order:
        origin = _place_source_pair(specs[idx], blocked_input)
        if origin is None:
            return None
        specs[idx]["source_origin"] = (int(origin[0]), int(origin[1]))

    inp = _build_input((h, w), specs)
    expected = _build_expected((h, w), specs)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid stamp-ink ARC input")
