import numpy as np


GEN_BG = 0
GEN_TEMPLATE = 5
GEN_COLORS = (1, 2, 3)
GEN_GRID_H_RANGE = (18, 29)
GEN_GRID_W_RANGE = (18, 29)
GEN_TEMPLATE_COUNT_CHOICES = (2, 3)
GEN_PADDING = 1
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([[r0, c0]], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = _tight_crop_mask(mask)
    return int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    keys = []
    for k in range(4):
        rot = _tight_crop_mask(np.rot90(mask, k))
        keys.append(_shape_key(rot))
        keys.append(_shape_key(np.fliplr(rot)))
    return min(keys)


def _build_shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for h in range(2, 5):
        for w in range(2, 5):
            cell_count = h * w
            for bits in range(1 << cell_count):
                area = int(bits.bit_count())
                if area < 3 or area > 6:
                    continue
                flat = np.array([(bits >> i) & 1 for i in range(cell_count)], dtype=bool)
                mask = flat.reshape(h, w)
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if len(_components(mask)) != 1:
                    continue
                rot_keys = {_shape_key(np.rot90(mask, k)) for k in range(4)}
                if len(rot_keys) != 4:
                    continue
                canon = _canonical_shape_key(mask)
                if canon in seen:
                    continue
                seen.add(canon)
                shapes.append(mask.copy())
    shapes.sort(key=lambda m: (int(m.sum()), int(m.shape[0]), int(m.shape[1]), _shape_key(m)))
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey template hole with the matching same-color piece after rotating that piece clockwise.

    Input structure:
    - Background is black.
    - Grey rectangular templates contain one black irregular hole each.
    - Directly above each template sit 1-3 same-color dots.
    - Elsewhere, one same-color piece per template is scattered on the canvas.

    Rule:
    - Match piece to template by dot color.
    - Let N be the number of dots above the template.
    - Rotate the piece clockwise by N quarter turns.
    - Move it into the hole, erase its original location, keep everything else unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    templates: list[dict] = []
    template_mask = np.zeros((h, w), dtype=bool)
    marker_mask = np.zeros((h, w), dtype=bool)

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components(g == color):
            comp_mask, (r0, c0, r1, c1) = _tight_mask_from_coords(coords)
            if comp_mask.shape[0] < 3 or comp_mask.shape[1] < 3:
                continue
            if not (
                bool(comp_mask[0, :].all())
                and bool(comp_mask[-1, :].all())
                and bool(comp_mask[:, 0].all())
                and bool(comp_mask[:, -1].all())
            ):
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((patch != color) & (patch != bg)):
                continue
            hole_mask = patch == bg
            if not hole_mask.any() or len(_components(hole_mask)) != 1:
                continue
            hole_tight = _tight_crop_mask(hole_mask)
            if not (3 <= int(hole_tight.sum()) <= 6):
                continue
            marker_row = r0 - 1
            if marker_row < 0:
                continue
            marker_cols = [c for c in range(c0, c1 + 1) if int(g[marker_row, c]) not in (bg, color)]
            if not marker_cols:
                continue
            marker_colors = {int(g[marker_row, c]) for c in marker_cols}
            if len(marker_colors) != 1:
                continue
            marker_color = marker_colors.pop()
            if not all(int(g[r0, c]) == color for c in marker_cols):
                continue
            if max(marker_cols) - min(marker_cols) + 1 != len(marker_cols):
                continue
            if not (1 <= len(marker_cols) <= 3):
                continue

            templates.append(
                {
                    "template_color": int(color),
                    "piece_color": int(marker_color),
                    "turns": int(len(marker_cols)),
                    "hole_shape": hole_tight,
                    "hole_coords": np.argwhere(hole_mask) + np.array([[r0, c0]], dtype=int),
                }
            )
            template_mask[coords[:, 0], coords[:, 1]] = True
            for c in marker_cols:
                marker_mask[marker_row, c] = True

    assert templates, "expected at least one template"
    piece_colors = [int(t["piece_color"]) for t in templates]
    assert len(piece_colors) == len(set(piece_colors)), "template colors must match pieces unambiguously"

    movable = (g != bg) & (~template_mask) & (~marker_mask)
    pieces: dict[int, dict] = {}
    for coords in _components(movable):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each piece must be monochrome"
        color = colors.pop()
        mask, bbox = _tight_mask_from_coords(coords)
        assert color not in pieces, "each piece color must be unique"
        pieces[int(color)] = {"mask": mask, "bbox": bbox, "coords": coords}

    out = g.copy()
    for template in templates:
        color = int(template["piece_color"])
        assert color in pieces, "missing piece for template"
        turns = int(template["turns"]) % 4
        rotated = np.rot90(pieces[color]["mask"], -turns)
        assert np.array_equal(rotated, template["hole_shape"]), "piece rotation must match the hole"
        coords = pieces[color]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
        hole_coords = template["hole_coords"]
        out[hole_coords[:, 0], hole_coords[:, 1]] = color

    return out


def _make_template_patch(
    hole_mask: np.ndarray, piece_color: int, turns: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    hole_h, hole_w = map(int, hole_mask.shape)
    top = int(rng.integers(1, 3))
    bottom = int(rng.integers(1, 3))
    left = int(rng.integers(1, 3))
    right = int(rng.integers(1, 3))
    tpl_h = hole_h + top + bottom
    tpl_w = max(hole_w + left + right, turns + 2)

    patch = np.full((tpl_h + 1, tpl_w), GEN_BG, dtype=int)
    patch[1:, :] = GEN_TEMPLATE
    patch[1 + top : 1 + top + hole_h, left : left + hole_w][hole_mask] = GEN_BG

    dot_start = int(rng.integers(0, tpl_w - turns + 1))
    patch[0, dot_start : dot_start + turns] = int(piece_color)

    piece_mask = np.rot90(hole_mask, turns)
    piece = np.full(piece_mask.shape, GEN_BG, dtype=int)
    piece[piece_mask] = int(piece_color)
    return patch, piece


def _rect_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _choose_position(
    occupied: np.ndarray, obj_h: int, obj_w: int, pad: int, rng: np.random.Generator
) -> tuple[int, int] | None:
    choices: list[tuple[int, int]] = []
    h, w = occupied.shape
    for r0 in range(h - obj_h + 1):
        for c0 in range(w - obj_w + 1):
            if _rect_is_free(occupied, int(r0), int(c0), int(obj_h), int(obj_w), int(pad)):
                choices.append((int(r0), int(c0)))
    if not choices:
        return None
    return choices[int(rng.integers(0, len(choices)))]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    mask = patch != GEN_BG
    view = grid[r0 : r0 + ph, c0 : c0 + pw]
    view[mask] = patch[mask]


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(GEN_GRID_H_RANGE[0], GEN_GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GEN_GRID_W_RANGE[0], GEN_GRID_W_RANGE[1] + 1))
    n_templates = int(rng.choice(np.array(GEN_TEMPLATE_COUNT_CHOICES)))

    colors = [int(x) for x in rng.choice(np.array(GEN_COLORS), size=n_templates, replace=False)]
    turns = [int(x) for x in rng.choice(np.array([1, 2, 3]), size=n_templates, replace=False)]
    shape_ids = rng.choice(len(SHAPE_LIBRARY), size=n_templates, replace=False)

    grid = np.full((h, w), GEN_BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    templates: list[dict] = []
    pieces: list[dict] = []

    for color, turn, shape_id in zip(colors, turns, shape_ids):
        hole = SHAPE_LIBRARY[int(shape_id)].copy()
        hole = np.rot90(hole, int(rng.integers(0, 4)))
        template_patch, piece_patch = _make_template_patch(hole, int(color), int(turn), rng)

        tpl_pos = _choose_position(occupied, template_patch.shape[0], template_patch.shape[1], GEN_PADDING, rng)
        if tpl_pos is None:
            return None
        tpl_r0, tpl_c0 = tpl_pos
        _paste_patch(grid, template_patch, tpl_r0, tpl_c0)
        _mark_rect(occupied, tpl_r0, tpl_c0, template_patch.shape[0], template_patch.shape[1])

        piece_pos = _choose_position(occupied, piece_patch.shape[0], piece_patch.shape[1], GEN_PADDING, rng)
        if piece_pos is None:
            return None
        piece_r0, piece_c0 = piece_pos
        _paste_patch(grid, piece_patch, piece_r0, piece_c0)
        _mark_rect(occupied, piece_r0, piece_c0, piece_patch.shape[0], piece_patch.shape[1])

        hole_rel = np.argwhere(template_patch == GEN_BG)
        hole_rel = hole_rel[(hole_rel[:, 0] > 0)]
        hole_global = hole_rel + np.array([[tpl_r0, tpl_c0]], dtype=int)
        piece_global = np.argwhere(piece_patch != GEN_BG) + np.array([[piece_r0, piece_c0]], dtype=int)

        templates.append({"color": int(color), "hole": hole_global})
        pieces.append({"color": int(color), "coords": piece_global})

    expected = grid.copy()
    for piece in pieces:
        coords = piece["coords"]
        expected[coords[:, 0], coords[:, 1]] = GEN_BG
    for template in templates:
        coords = template["hole"]
        expected[coords[:, 0], coords[:, 1]] = int(template["color"])

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
