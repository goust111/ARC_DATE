import random

import numpy as np


BG = 0
RED = 2
AZURE = 8
PIECE_COLORS = (1, 3, 4, 5, 6, 7, 9)
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_DELTA = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
MAX_GEN_ATTEMPTS = 300

SHAPE_LIBRARY = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in SIDE_TO_DELTA.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, r0, c0, r1, c1


def _rotate_side(side: str, k_ccw: int) -> str:
    idx = SIDES.index(side)
    return SIDES[(idx - k_ccw) % 4]


def _marker_side(marker: tuple[int, int], coords_set: set[tuple[int, int]]) -> str:
    r, c = marker
    hits: list[str] = []
    if (r + 1, c) in coords_set:
        hits.append("top")
    if (r, c - 1) in coords_set:
        hits.append("right")
    if (r - 1, c) in coords_set:
        hits.append("bottom")
    if (r, c + 1) in coords_set:
        hits.append("left")
    assert len(hits) == 1
    return hits[0]


def _marker_candidates(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    dr, dc = SIDE_TO_DELTA[side]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in coords:
        mr = r + dr
        mc = c + dc
        if (mr, mc) in coords or (mr, mc) in seen:
            continue
        neighbor_count = 0
        for ddr, ddc in SIDE_TO_DELTA.values():
            if (mr + ddr, mc + ddc) in coords:
                neighbor_count += 1
        if neighbor_count == 1:
            seen.add((mr, mc))
            out.append((mr, mc))
    return out


def _template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid == AZURE)
    assert comps
    template = max(comps, key=lambda comp: comp.shape[0])
    return _bbox(template)


def _extract_slots(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[dict]:
    tr0, tc0, tr1, tc1 = template_bbox
    sub = grid[tr0 : tr1 + 1, tc0 : tc1 + 1]
    h, w = sub.shape
    slots: list[dict] = []
    for comp in _components(sub == BG):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == w - 1)
        )
        if touches_edge:
            continue
        coords_set = {tuple(map(int, rc)) for rc in comp}
        marker_cells: set[tuple[int, int]] = set()
        for rr, cc in coords_set:
            for dr, dc in SIDE_TO_DELTA.values():
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and int(sub[nr, nc]) == RED:
                    marker_cells.add((nr, nc))
        assert len(marker_cells) == 1
        marker_local = next(iter(marker_cells))
        side = _marker_side(marker_local, coords_set)
        mask, sr0, sc0, _, _ = _mask_from_coords(comp)
        slots.append(
            {
                "mask": mask,
                "side": side,
                "top": tr0 + sr0,
                "left": tc0 + sc0,
                "marker": (tr0 + marker_local[0], tc0 + marker_local[1]),
            }
        )
    assert slots
    return slots


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    fragments: list[dict] = []
    mask = (grid != BG) & (grid != AZURE)
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        if colors == {RED}:
            continue
        assert RED in colors
        piece_coords = np.array(
            [(int(r), int(c)) for r, c in comp if int(grid[r, c]) != RED],
            dtype=int,
        )
        marker_cells = [(int(r), int(c)) for r, c in comp if int(grid[r, c]) == RED]
        assert piece_coords.size > 0
        assert len(marker_cells) == 1
        piece_colors = {int(grid[r, c]) for r, c in piece_coords}
        assert len(piece_colors) == 1
        marker = marker_cells[0]
        side = _marker_side(marker, {tuple(map(int, rc)) for rc in piece_coords})
        mask0, _, _, _, _ = _mask_from_coords(piece_coords)
        fragments.append(
            {
                "mask": mask0,
                "side": side,
                "color": next(iter(piece_colors)),
            }
        )
    assert fragments
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered colored fragments into the matching template holes using side markers.

    The input contains one large azure template with several black holes. Each hole has one
    red side marker on the surrounding template border. Outside the template, each colored
    fragment has the same shape as one hole up to rotation, plus one red side marker. Match
    each fragment to the hole whose shape it can become after rotation, but choose the exact
    rotation by aligning the fragment marker side with the hole marker side. In the output,
    fill every hole with its fragment color and remove all red markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_bbox = _template_bbox(g)
    slots = _extract_slots(g, template_bbox)
    fragments = _extract_fragments(g)
    assert len(slots) == len(fragments)

    out = np.full_like(g, BG)
    out[g == AZURE] = AZURE
    for slot in slots:
        mr, mc = slot["marker"]
        out[mr, mc] = AZURE

    unused_slots = set(range(len(slots)))
    for fragment in fragments:
        matches: list[int] = []
        for slot_idx in unused_slots:
            slot = slots[slot_idx]
            for k in range(4):
                if not np.array_equal(np.rot90(fragment["mask"], k), slot["mask"]):
                    continue
                if _rotate_side(fragment["side"], k) != slot["side"]:
                    continue
                matches.append(slot_idx)
                break
        assert len(matches) == 1
        slot_idx = matches[0]
        unused_slots.remove(slot_idx)
        slot = slots[slot_idx]
        sh, sw = slot["mask"].shape
        patch = out[slot["top"] : slot["top"] + sh, slot["left"] : slot["left"] + sw]
        patch[slot["mask"]] = fragment["color"]

    assert not unused_slots
    return out


def _make_fragment_component(piece_mask: np.ndarray, piece_side: str) -> dict | None:
    candidates = _marker_candidates(piece_mask, piece_side)
    if not candidates:
        return None
    marker = random.choice(candidates)
    piece_coords = np.argwhere(piece_mask).astype(int)
    all_rows = [int(v) for v in piece_coords[:, 0]] + [marker[0]]
    all_cols = [int(v) for v in piece_coords[:, 1]] + [marker[1]]
    r0 = min(all_rows)
    c0 = min(all_cols)
    r1 = max(all_rows)
    c1 = max(all_cols)
    rel_piece = piece_coords - np.array([r0, c0], dtype=int)
    rel_marker = (marker[0] - r0, marker[1] - c0)
    return {
        "piece_coords": rel_piece,
        "marker": rel_marker,
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
    }


def _layout_template(specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    count = len(specs)
    order = list(range(count))
    random.shuffle(order)

    row_count = 1 if count == 1 else random.choice((1, 2))
    if row_count == 1:
        rows = [order]
    else:
        cut = random.randint(1, count - 1)
        rows = [order[:cut], order[cut:]]

    row_defs = []
    for row in rows:
        widths = [specs[idx]["slot_mask"].shape[1] for idx in row]
        heights = [specs[idx]["slot_mask"].shape[0] for idx in row]
        gaps = [random.randint(2, 4) for _ in range(max(0, len(row) - 1))]
        row_defs.append(
            {
                "indices": row,
                "width": sum(widths) + sum(gaps),
                "height": max(heights),
                "gaps": gaps,
            }
        )

    pad_top = random.randint(2, 4)
    pad_bottom = random.randint(2, 4)
    pad_left = random.randint(2, 4)
    pad_right = random.randint(2, 4)
    row_gap = random.randint(2, 4) if len(row_defs) == 2 else 0
    extra_w = random.randint(0, 4)
    extra_h = random.randint(0, 2)

    template_w = pad_left + max(row["width"] for row in row_defs) + pad_right + extra_w
    template_h = pad_top + sum(row["height"] for row in row_defs) + row_gap + pad_bottom + extra_h
    template = np.full((template_h, template_w), AZURE, dtype=int)
    placements: list[dict | None] = [None] * count

    current_top = pad_top
    for row in row_defs:
        slack = template_w - pad_left - pad_right - row["width"]
        start_left = pad_left + random.randint(0, slack)
        cursor = start_left
        for pos, idx in enumerate(row["indices"]):
            mask = specs[idx]["slot_mask"]
            hh, ww = mask.shape
            top = current_top + random.randint(0, row["height"] - hh)
            left = cursor
            marker_candidates = [
                (top + mr, left + mc)
                for mr, mc in _marker_candidates(mask, specs[idx]["slot_side"])
                if 0 <= top + mr < template_h and 0 <= left + mc < template_w
            ]
            marker_candidates = [(r, c) for r, c in marker_candidates if int(template[r, c]) == AZURE]
            if not marker_candidates:
                return None
            marker = random.choice(marker_candidates)
            window = template[top : top + hh, left : left + ww]
            if np.any(window[mask] != AZURE):
                return None
            window[mask] = BG
            template[marker] = RED
            placements[idx] = {"top": top, "left": left, "marker": marker}
            cursor += ww
            if pos < len(row["gaps"]):
                cursor += row["gaps"][pos]
        current_top += row["height"] + row_gap

    assert all(place is not None for place in placements)
    return template, placements  # type: ignore[return-value]


def _place_fragments(grid: np.ndarray, template_bbox: tuple[int, int, int, int], specs: list[dict]) -> bool:
    h, w = grid.shape
    tr0, tc0, tr1, tc1 = template_bbox
    reserved = np.zeros_like(grid, dtype=bool)
    reserved[max(0, tr0 - 1) : min(h, tr1 + 2), max(0, tc0 - 1) : min(w, tc1 + 2)] = True

    components = []
    for spec in specs:
        comp = _make_fragment_component(spec["piece_mask"], spec["piece_side"])
        if comp is None:
            return False
        components.append(comp)

    order = sorted(
        range(len(specs)),
        key=lambda idx: components[idx]["height"] * components[idx]["width"],
        reverse=True,
    )

    for idx in order:
        comp = components[idx]
        ch = comp["height"]
        cw = comp["width"]
        if h - ch <= 1 or w - cw <= 1:
            return False
        positions = [(r, c) for r in range(1, h - ch) for c in range(1, w - cw)]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            rs = top - 1
            re = top + ch + 1
            cs = left - 1
            ce = left + cw + 1
            if np.any(reserved[rs:re, cs:ce]):
                continue
            for pr, pc in comp["piece_coords"]:
                grid[top + int(pr), left + int(pc)] = specs[idx]["color"]
            mr, mc = comp["marker"]
            grid[top + mr, left + mc] = RED
            reserved[rs:re, cs:ce] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_generated(grid: np.ndarray, expected_count: int) -> bool:
    try:
        template_bbox = _template_bbox(grid)
        slots = _extract_slots(grid, template_bbox)
        fragments = _extract_fragments(grid)
        if len(slots) != expected_count or len(fragments) != expected_count:
            return False
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape and not np.any(out == RED) and np.any(out != grid)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    colors = random.sample(PIECE_COLORS, count)
    bases = [mask.copy() for mask in random.sample(SHAPE_LIBRARY, count)]
    specs: list[dict] = []
    for color, base in zip(colors, bases):
        slot_mask = np.rot90(base, random.randrange(4))
        slot_sides = [side for side in SIDES if _marker_candidates(slot_mask, side)]
        if not slot_sides:
            return None
        slot_side = random.choice(slot_sides)
        needed_rotation = random.randrange(4)
        piece_mask = np.rot90(slot_mask, -needed_rotation)
        piece_side = _rotate_side(slot_side, -needed_rotation)
        if not _marker_candidates(piece_mask, piece_side):
            return None
        specs.append(
            {
                "color": color,
                "slot_mask": slot_mask,
                "slot_side": slot_side,
                "piece_mask": piece_mask,
                "piece_side": piece_side,
            }
        )

    laid_out = _layout_template(specs)
    if laid_out is None:
        return None
    template, placements = laid_out
    template_h, template_w = template.shape

    grid_h = template_h + random.randint(12, 18)
    grid_w = template_w + random.randint(12, 18)
    if grid_h - template_h <= 6 or grid_w - template_w <= 6:
        return None

    top = random.randint(3, grid_h - template_h - 3)
    left = random.randint(3, grid_w - template_w - 3)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[top : top + template_h, left : left + template_w] = template

    template_bbox = (top, left, top + template_h - 1, left + template_w - 1)
    if not _place_fragments(grid, template_bbox, specs):
        return None

    if not _validate_generated(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
