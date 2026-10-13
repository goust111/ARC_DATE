from __future__ import annotations

import random

import numpy as np


MARKER_COLOR = 2
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _rotate_anchor(shape: tuple[int, int], anchor: tuple[int, int], k: int) -> tuple[int, int]:
    probe = np.zeros(shape, dtype=bool)
    probe[int(anchor[0]), int(anchor[1])] = True
    rotated = np.rot90(probe, k)
    point = np.argwhere(rotated)
    assert len(point) == 1
    return int(point[0, 0]), int(point[0, 1])


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _rotation_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _rotation_class_size(mask: np.ndarray) -> int:
    return len({_mask_key(np.rot90(mask, k)) for k in range(4)})


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                    cells.append((r, c))
                    break
    assert cells
    return cells


def _find_template(grid: np.ndarray, bg: int) -> dict:
    best: dict | None = None
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, MARKER_COLOR):
            continue
        for coords in _components(grid == color):
            area = int(coords.shape[0])
            if best is None or area > best["area"]:
                best = {
                    "color": color,
                    "coords": coords,
                    "area": area,
                    "bbox": _bbox(coords),
                }
    assert best is not None, "expected a monochrome template object"
    return best


def _extract_holes(grid: np.ndarray, template: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = template["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for coords in _components(sub != int(template["color"])):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 0] == sub.shape[0] - 1)
            or np.any(coords[:, 1] == sub.shape[1] - 1)
        ):
            continue
        values = np.array([int(sub[rr, cc]) for rr, cc in coords], dtype=int)
        assert set(values.tolist()) <= {bg, MARKER_COLOR}
        marker_coords = coords[values == MARKER_COLOR]
        assert len(marker_coords) == 1, "each hole must have one marker"
        mask, (hr0, hc0) = _tight_mask(coords)
        anchor = (
            int(marker_coords[0, 0] - hr0),
            int(marker_coords[0, 1] - hc0),
        )
        holes.append(
            {
                "mask": mask,
                "anchor": anchor,
                "top_left": (r0 + hr0, c0 + hc0),
            }
        )
    assert holes, "expected at least one template hole"
    return holes


def _extract_pieces(grid: np.ndarray, template_color: int, bg: int) -> list[dict]:
    piece_mask = (grid != bg) & (grid != template_color)
    pieces: list[dict] = []
    for coords in _components(piece_mask):
        values = np.array([int(grid[rr, cc]) for rr, cc in coords], dtype=int)
        colors = set(values.tolist())
        if colors == {MARKER_COLOR}:
            assert len(coords) == 1, "standalone markers should only appear in holes"
            continue
        assert MARKER_COLOR in colors, "each fragment must carry a red marker"
        fill_colors = colors - {MARKER_COLOR}
        assert len(fill_colors) == 1, "each fragment must be monochrome apart from the marker"
        fill_color = next(iter(fill_colors))
        marker_coords = coords[values == MARKER_COLOR]
        assert len(marker_coords) == 1, "each fragment must have one marker"
        mask, (pr0, pc0) = _tight_mask(coords)
        anchor = (
            int(marker_coords[0, 0] - pr0),
            int(marker_coords[0, 1] - pc0),
        )
        pieces.append(
            {
                "mask": mask,
                "anchor": anchor,
                "color": int(fill_color),
                "coords": coords,
            }
        )
    assert pieces, "expected at least one fragment"
    return pieces


def _match_piece_to_hole(piece: dict, hole: dict) -> list[int]:
    options: list[int] = []
    for k in range(4):
        rotated_mask = np.rot90(piece["mask"], k)
        if rotated_mask.shape != hole["mask"].shape:
            continue
        if not np.array_equal(rotated_mask, hole["mask"]):
            continue
        if _rotate_anchor(piece["mask"].shape, piece["anchor"], k) != hole["anchor"]:
            continue
        options.append(k)
    return options


def _assign_matches(pieces: list[dict], holes: list[dict]) -> dict[int, tuple[int, int]]:
    assert len(pieces) == len(holes), "piece count must equal hole count"
    choices: dict[int, list[tuple[int, int]]] = {}
    for i, piece in enumerate(pieces):
        options: list[tuple[int, int]] = []
        for j, hole in enumerate(holes):
            for k in _match_piece_to_hole(piece, hole):
                options.append((j, k))
        assert options, "every piece must match at least one hole"
        choices[i] = options

    order = sorted(range(len(pieces)), key=lambda idx: len(choices[idx]))
    used_holes: set[int] = set()
    assignment: dict[int, tuple[int, int]] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        piece_idx = order[pos]
        for hole_idx, rotation in choices[piece_idx]:
            if hole_idx in used_holes:
                continue
            used_holes.add(hole_idx)
            assignment[piece_idx] = (hole_idx, rotation)
            if dfs(pos + 1):
                return True
            used_holes.remove(hole_idx)
            del assignment[piece_idx]
        return False

    assert dfs(0), "piece-hole matching must be solvable"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill template holes with scattered fragments after rotating them to align both shape and red anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    template = _find_template(g, bg)
    holes = _extract_holes(g, template, bg)
    pieces = _extract_pieces(g, int(template["color"]), bg)
    matches = _assign_matches(pieces, holes)

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg

    for piece_idx, (hole_idx, _rotation) in matches.items():
        piece = pieces[piece_idx]
        hole = holes[hole_idx]
        r0, c0 = hole["top_left"]
        region = out[r0 : r0 + hole["mask"].shape[0], c0 : c0 + hole["mask"].shape[1]]
        region[hole["mask"]] = int(piece["color"])

    return out


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _shape(["11", "11"]),
    _shape(["1111"]),
    _shape(["111", "010"]),
    _shape(["110", "011"]),
    _shape(["10", "10", "11"]),
    _shape(["11", "11", "10"]),
    _shape(["010", "111", "010"]),
    _shape(["101", "111"]),
    _shape(["110", "011", "001"]),
    _shape(["100", "111", "001"]),
    _shape(["011", "110", "010"]),
    _shape(["111", "001", "001"]),
]


def _mark_with_padding(reserve: np.ndarray, mask: np.ndarray, r0: int, c0: int, pad: int) -> None:
    h, w = reserve.shape
    for dr, dc in np.argwhere(mask):
        rr, cc = int(r0 + dr), int(c0 + dc)
        reserve[max(0, rr - pad) : min(h, rr + pad + 1), max(0, cc - pad) : min(w, cc + pad + 1)] = True


def _can_place_mask(reserve: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = reserve.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    cells = np.argwhere(mask)
    for dr, dc in cells:
        if reserve[int(r0 + dr), int(c0 + dc)]:
            return False
    return True


def _place_holes(template_h: int, template_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(masks)), key=lambda idx: int(masks[idx].sum()), reverse=True)
    for _ in range(50):
        reserve = np.zeros((template_h, template_w), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(masks)
        ok = True
        for idx in order:
            mask = masks[idx]
            mh, mw = mask.shape
            candidates = [
                (r0, c0)
                for r0 in range(1, template_h - mh)
                for c0 in range(1, template_w - mw)
                if _can_place_mask(reserve, mask, r0, c0)
            ]
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            placements[idx] = (r0, c0)
            _mark_with_padding(reserve, mask, r0, c0, pad=1)
        if ok:
            return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]
    return None


def _place_pieces(
    grid_shape: tuple[int, int],
    template_rect: tuple[int, int, int, int],
    masks: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    tr0, tc0, th, tw = template_rect
    order = sorted(range(len(masks)), key=lambda idx: int(masks[idx].sum()), reverse=True)
    for _ in range(80):
        reserve = np.zeros((h, w), dtype=bool)
        reserve[max(0, tr0 - 1) : min(h, tr0 + th + 1), max(0, tc0 - 1) : min(w, tc0 + tw + 1)] = True
        placements: list[tuple[int, int] | None] = [None] * len(masks)
        ok = True
        for idx in order:
            mask = masks[idx]
            mh, mw = mask.shape
            candidates = [
                (r0, c0)
                for r0 in range(h - mh + 1)
                for c0 in range(w - mw + 1)
                if _can_place_mask(reserve, mask, r0, c0)
            ]
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            placements[idx] = (r0, c0)
            _mark_with_padding(reserve, mask, r0, c0, pad=1)
        if ok:
            return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]
    return None


def _sample_shapes(n_shapes: int) -> list[np.ndarray]:
    indices = random.sample(range(len(SHAPE_LIBRARY)), n_shapes)
    if random.random() < 0.5:
        ambiguous = [idx for idx, mask in enumerate(SHAPE_LIBRARY) if _rotation_class_size(mask) < 4]
        if ambiguous and all(_rotation_class_size(SHAPE_LIBRARY[idx]) == 4 for idx in indices):
            indices[random.randrange(n_shapes)] = random.choice(ambiguous)
    seen: set[tuple[str, ...]] = set()
    chosen: list[np.ndarray] = []
    for idx in indices:
        mask = SHAPE_LIBRARY[idx]
        sig = _rotation_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        chosen.append(mask.copy())
    if len(chosen) != n_shapes:
        return []
    return chosen


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    n_shapes = random.randint(2, 4)
    base_masks = _sample_shapes(n_shapes)
    if len(base_masks) != n_shapes:
        return None

    palette = list(range(10))
    palette.remove(MARKER_COLOR)
    random.shuffle(palette)
    bg = palette[0]
    template_color = palette[1]
    piece_colors = palette[2 : 2 + n_shapes]
    if len(piece_colors) != n_shapes:
        return None

    hole_masks: list[np.ndarray] = []
    hole_anchors: list[tuple[int, int]] = []
    solve_rotations: list[int] = []
    max_h = 0
    max_w = 0
    for base in base_masks:
        hole_mask = np.rot90(base, random.randrange(4))
        hole_masks.append(hole_mask)
        hole_anchors.append(random.choice(_boundary_cells(hole_mask)))
        solve_rotations.append(random.randrange(4))
        max_h = max(max_h, hole_mask.shape[0])
        max_w = max(max_w, hole_mask.shape[1])

    template_h_lo = max(9, max_h + 4, n_shapes * 3 + 2)
    template_w_lo = max(9, max_w + 4, n_shapes * 3 + 2)
    template_h_hi = min(15, h - 6)
    template_w_hi = min(15, w - 6)
    if template_h_lo > template_h_hi or template_w_lo > template_w_hi:
        return None

    template_h = random.randint(template_h_lo, template_h_hi)
    template_w = random.randint(template_w_lo, template_w_hi)
    hole_positions = _place_holes(template_h, template_w, hole_masks)
    if hole_positions is None:
        return None

    tr0 = random.randint(2, h - template_h - 2)
    tc0 = random.randint(2, w - template_w - 2)
    output = np.full((h, w), bg, dtype=int)
    output[tr0 : tr0 + template_h, tc0 : tc0 + template_w] = template_color

    for mask, (hr0, hc0), color in zip(hole_masks, hole_positions, piece_colors):
        region = output[tr0 + hr0 : tr0 + hr0 + mask.shape[0], tc0 + hc0 : tc0 + hc0 + mask.shape[1]]
        region[mask] = int(color)

    input_grid = output.copy()
    for mask, anchor, (hr0, hc0) in zip(hole_masks, hole_anchors, hole_positions):
        region = input_grid[tr0 + hr0 : tr0 + hr0 + mask.shape[0], tc0 + hc0 : tc0 + hc0 + mask.shape[1]]
        region[mask] = bg
        region[int(anchor[0]), int(anchor[1])] = MARKER_COLOR

    piece_masks: list[np.ndarray] = []
    piece_anchors: list[tuple[int, int]] = []
    for hole_mask, hole_anchor, solve_k in zip(hole_masks, hole_anchors, solve_rotations):
        piece_k = (-solve_k) % 4
        piece_masks.append(np.rot90(hole_mask, piece_k))
        piece_anchors.append(_rotate_anchor(hole_mask.shape, hole_anchor, piece_k))

    piece_positions = _place_pieces((h, w), (tr0, tc0, template_h, template_w), piece_masks)
    if piece_positions is None:
        return None

    for mask, anchor, (pr0, pc0), color in zip(piece_masks, piece_anchors, piece_positions, piece_colors):
        region = input_grid[pr0 : pr0 + mask.shape[0], pc0 : pc0 + mask.shape[1]]
        region[mask] = int(color)
        region[int(anchor[0]), int(anchor[1])] = MARKER_COLOR

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, out = attempt
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
