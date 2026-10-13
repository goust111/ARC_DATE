import numpy as np


BG = 0
MARKER = 2
TEMPLATE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return np.asarray(mask[rows][:, cols], dtype=bool)


def _mask_sig(mask: np.ndarray) -> tuple:
    m = np.asarray(mask, dtype=np.uint8)
    return m.shape, tuple(map(int, m.ravel()))


def _rot_sigs(mask: np.ndarray) -> set[tuple]:
    return {_mask_sig(np.rot90(mask, k)) for k in range(4)}


def _canon_rot(mask: np.ndarray) -> tuple:
    return min(_rot_sigs(mask))


def _canon_all(mask: np.ndarray) -> tuple:
    flipped = np.fliplr(mask)
    return min(_rot_sigs(mask) | _rot_sigs(flipped))


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_cells = np.argwhere(g == TEMPLATE)
    assert len(template_cells) > 0, "missing template"
    tr0, tc0, tr1, tc1 = _bbox(template_cells)
    template_bbox = (tr0, tc0, tr1, tc1)
    template_patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert np.all((template_patch == BG) | (template_patch == TEMPLATE)), "template patch must use only bg and template color"

    holes = []
    for coords in _components(template_patch == BG):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 0] == template_patch.shape[0] - 1)
            or np.any(coords[:, 1] == template_patch.shape[1] - 1)
        ):
            continue
        mask, (r0, c0, r1, c1) = _mask_from_coords(coords)
        holes.append(
            {
                "mask": mask,
                "bbox": (tr0 + r0, tc0 + c0, tr0 + r1, tc0 + c1),
            }
        )
    assert holes, "template must contain holes"

    piece_mask = (g != BG) & (g != TEMPLATE)
    pieces = []
    for coords in _components(piece_mask):
        mask, bbox = _mask_from_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        vals = patch[mask]
        colors = sorted({int(v) for v in vals if int(v) != MARKER})
        assert len(colors) == 1, "each piece must have one body color"
        pieces.append(
            {
                "mask": mask,
                "bbox": bbox,
                "color": colors[0],
                "marker_count": int(np.count_nonzero(vals == MARKER)),
            }
        )
    assert pieces, "missing pieces"
    return g, template_bbox, holes, pieces


def _candidate_holes(piece_mask: np.ndarray, marker_count: int, holes: list[dict]) -> list[int]:
    result = set()
    for k in range(4):
        rotated = np.rot90(piece_mask, k)
        if not any(_same_mask(rotated, hole["mask"]) for hole in holes):
            continue
        final_mask = np.fliplr(rotated) if marker_count % 2 == 1 else rotated
        for idx, hole in enumerate(holes):
            if _same_mask(final_mask, hole["mask"]):
                result.add(idx)
    return sorted(result)


def _solve_assignment(pieces: list[dict], holes: list[dict]) -> list[int]:
    candidates = [_candidate_holes(piece["mask"], piece["marker_count"], holes) for piece in pieces]
    assert all(candidates), "every piece must match at least one hole after parity transform"

    order = sorted(range(len(pieces)), key=lambda i: len(candidates[i]))
    assign = [-1] * len(pieces)
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for hole_idx in candidates[idx]:
            if hole_idx in used:
                continue
            used.add(hole_idx)
            assign[idx] = hole_idx
            if dfs(pos + 1):
                return True
            used.remove(hole_idx)
            assign[idx] = -1
        return False

    ok = dfs(0)
    assert ok and all(v >= 0 for v in assign), "piece-hole mapping must be uniquely solvable"
    return assign


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered fragments to holes in the central template by shape.
    A fragment first rotates to align with a hole; odd numbers of red markers
    then trigger an additional horizontal flip before placement.
    Red markers disappear and the filled hole becomes the fragment's body color.
    """
    g, template_bbox, holes, pieces = _extract_scene(grid)
    assert len(pieces) == len(holes), "piece count must equal hole count"

    out = np.full_like(g, BG)
    out[g == TEMPLATE] = TEMPLATE

    assignment = _solve_assignment(pieces, holes)
    for piece, hole_idx in zip(pieces, assignment):
        hole = holes[hole_idx]
        r0, c0, r1, c1 = hole["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[hole["mask"]] = int(piece["color"])

    return out


def _random_shape(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(200):
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        max_area = min(h * w - 1, 8)
        min_area = max(4, (h + w) // 2 + 1)
        if min_area > max_area:
            continue
        area = int(rng.integers(min_area, max_area + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim(mask)
        area = int(mask.sum())
        mh, mw = mask.shape
        if area < 4 or area > 8 or mh < 2 or mw < 3:
            continue
        if area == mh * mw:
            continue
        if len(_rot_sigs(mask)) < 4:
            continue
        if any(_same_mask(np.rot90(mask, k), np.fliplr(mask)) for k in range(4)):
            continue
        return mask
    return None


def _make_pair_layout(hole_a: np.ndarray, hole_b: np.ndarray, gap: int):
    ha, wa = hole_a.shape
    hb, wb = hole_b.shape
    row_h = max(ha, hb)
    row_w = wa + gap + wb
    return {
        "row_h": row_h,
        "row_w": row_w,
        "a_pos": ((row_h - ha) // 2, 0),
        "b_pos": ((row_h - hb) // 2, wa + gap),
    }


def _stamp_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, value: int) -> None:
    h, w = mask.shape
    canvas[top : top + h, left : left + w][mask] = value


def _dilate_bbox(top: int, left: int, h: int, w: int, pad: int, H: int, W: int) -> tuple[int, int, int, int]:
    return (
        max(0, top - pad),
        max(0, left - pad),
        min(H - 1, top + h - 1 + pad),
        min(W - 1, left + w - 1 + pad),
    )


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_piece_in_zone(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    zone: tuple[int, int, int, int],
    blocked: list[tuple[int, int, int, int]],
) -> bool:
    h, w = patch.shape
    r0, c0, r1, c1 = zone
    if r1 - r0 + 1 < h or c1 - c0 + 1 < w:
        return False
    rows = list(range(r0, r1 - h + 2))
    cols = list(range(c0, c1 - w + 2))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for top in rows:
        for left in cols:
            rect = (top, left, top + h - 1, left + w - 1)
            padded = _dilate_bbox(top, left, h, w, 1, grid.shape[0], grid.shape[1])
            if any(_rects_overlap(padded, other) for other in blocked):
                continue
            if np.any(grid[top : top + h, left : left + w] != BG):
                continue
            mask = patch != BG
            grid[top : top + h, left : left + w][mask] = patch[mask]
            blocked.append(rect)
            return True
    return False


def _build_piece_patch(
    rng: np.random.Generator,
    mask: np.ndarray,
    color: int,
    marker_count: int,
) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    coords = np.argwhere(mask)
    if marker_count > 0:
        chosen = rng.choice(len(coords), size=marker_count, replace=False)
        for idx in np.atleast_1d(chosen):
            r, c = coords[int(idx)]
            patch[r, c] = MARKER
    return patch


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        g, _bbox_unused, holes, pieces = _extract_scene(grid)
        if len(holes) != len(pieces):
            return False
        assignment = _solve_assignment(pieces, holes)
        if len(set(assignment)) != len(holes):
            return False
        out = main(g)
        if out.shape != g.shape:
            return False
        if np.any((out != BG) & (out != TEMPLATE) & (out == MARKER)):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = RNG
    pair_count = int(rng.choice([1, 2], p=[0.45, 0.55]))

    families = []
    used_families = set()
    while len(families) < pair_count:
        shape = _random_shape(rng)
        if shape is None:
            return None
        key = _canon_all(shape)
        if key in used_families:
            continue
        used_families.add(key)
        families.append(shape)

    holes_by_pair = []
    pieces = []
    pair_parities = []
    if pair_count == 1:
        pair_parities = [int(rng.integers(2))]
    else:
        pair_parities = [0, 1]
        rng.shuffle(pair_parities)

    available_colors = [c for c in range(1, 10) if c not in (MARKER, TEMPLATE)]
    rng.shuffle(available_colors)
    if len(available_colors) < pair_count * 2:
        return None

    color_idx = 0
    max_piece_h = 0
    max_piece_w = 0
    for pair_idx, base in enumerate(families):
        hole_a = np.rot90(base, int(rng.integers(4)))
        hole_b = np.rot90(np.fliplr(base), int(rng.integers(4)))
        holes_by_pair.append((hole_a, hole_b))

        for anchor in (hole_a, hole_b):
            piece_mask = np.rot90(anchor, int(rng.integers(4)))
            area = int(piece_mask.sum())
            odd = bool(pair_parities[pair_idx])
            allowed = [k for k in range(area) if k % 2 == int(odd) and k <= area - 1]
            if not allowed:
                return None
            marker_count = int(rng.choice(allowed))
            piece = {
                "mask": piece_mask,
                "color": int(available_colors[color_idx]),
                "marker_count": marker_count,
            }
            pieces.append(piece)
            color_idx += 1
            max_piece_h = max(max_piece_h, piece_mask.shape[0])
            max_piece_w = max(max_piece_w, piece_mask.shape[1])

    row_layouts = []
    for hole_a, hole_b in holes_by_pair:
        row_layouts.append(_make_pair_layout(hole_a, hole_b, int(rng.integers(2, 5))))

    row_gap = int(rng.integers(2, 5))
    layout_h = row_layouts[0]["row_h"]
    layout_w = row_layouts[0]["row_w"]
    if pair_count == 2:
        layout_h += row_gap + row_layouts[1]["row_h"]
        layout_w = max(layout_w, row_layouts[1]["row_w"])

    pad_top = int(rng.integers(1, 3))
    pad_bottom = int(rng.integers(1, 3))
    pad_left = int(rng.integers(1, 3))
    pad_right = int(rng.integers(1, 3))
    template_h = layout_h + pad_top + pad_bottom
    template_w = layout_w + pad_left + pad_right

    grid_h_min = max(template_h + 8, 18)
    grid_w_min = max(template_w + 8, 18)
    grid_h_max = min(30, template_h + 14)
    grid_w_max = min(30, template_w + 14)
    if grid_h_min > grid_h_max or grid_w_min > grid_w_max:
        return None
    grid_h = int(rng.integers(grid_h_min, grid_h_max + 1))
    grid_w = int(rng.integers(grid_w_min, grid_w_max + 1))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    tpl_r0 = (grid_h - template_h) // 2
    tpl_c0 = (grid_w - template_w) // 2
    tpl_r1 = tpl_r0 + template_h - 1
    tpl_c1 = tpl_c0 + template_w - 1
    grid[tpl_r0 : tpl_r1 + 1, tpl_c0 : tpl_c1 + 1] = TEMPLATE

    hole_masks = []
    current_row = pad_top
    piece_anchor_masks = []
    piece_order = 0
    for row_idx, ((hole_a, hole_b), row_info) in enumerate(zip(holes_by_pair, row_layouts)):
        x_offset = pad_left
        if layout_w > row_info["row_w"]:
            x_offset += int(rng.integers(0, layout_w - row_info["row_w"] + 1))

        for local_mask, (rr, cc) in ((hole_a, row_info["a_pos"]), (hole_b, row_info["b_pos"])):
            top = tpl_r0 + current_row + rr
            left = tpl_c0 + x_offset + cc
            _stamp_mask(grid, local_mask, top, left, BG)
            hole_masks.append(local_mask)
            piece_anchor_masks.append(local_mask)
            piece_order += 1

        current_row += row_info["row_h"]
        if row_idx + 1 < pair_count:
            current_row += row_gap

    zones = {
        "top": (0, 0, tpl_r0 - 2, grid_w - 1),
        "bottom": (tpl_r1 + 2, 0, grid_h - 1, grid_w - 1),
        "left": (0, 0, grid_h - 1, tpl_c0 - 2),
        "right": (0, tpl_c1 + 2, grid_h - 1, grid_w - 1),
    }

    if pair_count == 1:
        zone_names = list(rng.choice(["top", "bottom", "left", "right"], size=2, replace=False))
    else:
        zone_names = ["top", "bottom", "left", "right"]
        rng.shuffle(zone_names)

    blocked = [(tpl_r0, tpl_c0, tpl_r1, tpl_c1)]
    for piece, zone_name in zip(pieces, zone_names):
        patch = _build_piece_patch(rng, piece["mask"], piece["color"], piece["marker_count"])
        if not _place_piece_in_zone(rng, grid, patch, zones[zone_name], blocked):
            return None

    odd_total = sum(piece["marker_count"] % 2 for piece in pieces)
    even_total = len(pieces) - odd_total
    if pair_count == 2 and (odd_total == 0 or even_total == 0):
        return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
