from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
BORDER_COLORS = [2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_BUILD_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rot = [(c, h - 1 - r) for r, c in coords]
    return _normalize(rot)


def _all_rotations(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    cur = _normalize(coords)
    out: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(4):
        norm, h, w = cur
        if norm not in seen:
            seen.add(norm)
            out.append((norm, h, w))
        cur = _rot90(norm, h, w)
    return out


def _dilate(coords: set[tuple[int, int]], diagonal: bool) -> set[tuple[int, int]]:
    dirs = DIR8 if diagonal else DIR4
    out = set(coords)
    for r, c in coords:
        for dr, dc in dirs:
            out.add((int(r + dr), int(c + dc)))
    return out


def _shift(coords: set[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in coords}


def _l_mask(v_len: int, h_len: int, thickness: int, rotation: int) -> np.ndarray:
    base = np.zeros((v_len, h_len), dtype=bool)
    base[:, :thickness] = True
    base[v_len - thickness :, :] = True
    return np.rot90(base, rotation)


def _hole_border_and_margin(mask: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    hole = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    border = _dilate(hole, diagonal=True) - hole
    margin = _dilate(hole | border, diagonal=True)
    return hole, border, margin


def _fit_positions(shape: tuple[int, int], rel_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    min_r = min(r for r, _ in rel_cells)
    max_r = max(r for r, _ in rel_cells)
    min_c = min(c for _, c in rel_cells)
    max_c = max(c for _, c in rel_cells)
    return [
        (r0, c0)
        for r0 in range(-min_r, h - max_r)
        for c0 in range(-min_c, w - max_c)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each scattered L-shaped piece into the matching colored hole and repaint it with the hole-border color.

    Input structure:
    - Background is black.
    - A single large central template is the largest non-black component; it is mostly blue but contains black L-shaped holes, each wrapped by a one-pixel colored border.
    - Outside the template are separate L-shaped piece components whose body color is uniform and which contain exactly one single-cell colored marker.
    - Each marker color appears on exactly one hole border.

    Rule:
    - Find every enclosed black hole and read its border color from the surrounding colored ring.
    - For each outside piece, use its marker color to choose the matching hole, and require that some rotation of the piece's occupied shape matches the hole shape.
    - Clear each scattered piece from its original location, then fill the matched hole with the border color so the entire placed piece becomes monochrome.
    - Leave the blue template area and all colored borders unchanged.

    Key insight:
    - The marker color only solves the pairing; the actual placement still depends on recognizing that the piece and hole share the same L-shape up to rotation, after which the piece is fully recolored to the border color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    non_black_components = _components(g != BLACK)
    assert non_black_components, "expected at least one non-black component"
    template_index = max(range(len(non_black_components)), key=lambda i: len(non_black_components[i]))

    holes_by_color: dict[int, dict[str, object]] = {}
    for coords in _components(g == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        border_colors = {
            int(g[nr, nc])
            for r, c in coords
            for dr, dc in DIR4
            for nr, nc in [(r + dr, c + dc)]
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != BLACK
        }
        assert len(border_colors) == 1, "each hole must have one border color"
        border_color = border_colors.pop()
        norm, _, _ = _normalize(coords)
        holes_by_color[border_color] = {"coords": coords, "norm": norm}

    assert holes_by_color, "expected enclosed holes inside the template"

    piece_colors: set[int] = set()
    for i, coords in enumerate(non_black_components):
        if i == template_index:
            continue
        counts: dict[int, int] = {}
        for r, c in coords:
            val = int(g[r, c])
            counts[val] = counts.get(val, 0) + 1
        assert len(counts) == 2, "each scattered piece must contain a body color plus one marker"
        marker_candidates = [color for color, count in counts.items() if count == 1]
        assert len(marker_candidates) == 1, "piece marker must be unique"
        marker_color = int(marker_candidates[0])
        assert marker_color in holes_by_color, "piece marker color must match a hole border"
        piece_colors.add(marker_color)

        piece_norm, _, _ = _normalize(coords)
        hole_norm = holes_by_color[marker_color]["norm"]
        assert any(rot_norm == hole_norm for rot_norm, _, _ in _all_rotations(piece_norm)), "piece shape must match its hole up to rotation"

        for r, c in coords:
            out[r, c] = BLACK
        for r, c in holes_by_color[marker_color]["coords"]:
            out[r, c] = marker_color

    assert piece_colors == set(holes_by_color), "holes and pieces must match one-to-one"
    return out


def _random_hole_spec(color: int) -> dict[str, object]:
    thickness = random.randint(1, 2)
    v_len = random.randint(thickness + 2, thickness + 5)
    h_len = random.randint(thickness + 2, thickness + 5)
    rotation = random.randrange(4)
    mask = _l_mask(v_len, h_len, thickness, rotation)
    hole, border, margin = _hole_border_and_margin(mask)
    return {
        "color": int(color),
        "mask": mask,
        "hole_rel": hole,
        "border_rel": border,
        "margin_rel": margin,
    }


def _place_holes(template_shape: tuple[int, int], specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []
    order = sorted(specs, key=lambda spec: len(spec["margin_rel"]), reverse=True)
    for spec in order:
        positions = _fit_positions(template_shape, spec["margin_rel"])
        random.shuffle(positions)
        chosen = None
        for r0, c0 in positions:
            margin_abs = _shift(spec["margin_rel"], r0, c0)
            if margin_abs & blocked:
                continue
            chosen = {
                "color": spec["color"],
                "hole_abs": _shift(spec["hole_rel"], r0, c0),
                "border_abs": _shift(spec["border_rel"], r0, c0),
                "margin_abs": margin_abs,
            }
            blocked |= margin_abs
            placed.append(chosen)
            break
        if chosen is None:
            return None
    return placed


def _piece_variants(hole_abs: set[tuple[int, int]], color: int) -> list[dict[str, object]]:
    hole_norm, _, _ = _normalize(hole_abs)
    variants: list[dict[str, object]] = []
    for rot_norm, _, _ in _all_rotations(hole_norm):
        cells = set(rot_norm)
        safe = _dilate(cells, diagonal=True)
        variants.append({"color": int(color), "cells_rel": cells, "safe_rel": safe})
    return variants


def _template_forbidden(full_shape: tuple[int, int], top: int, left: int, th: int, tw: int) -> set[tuple[int, int]]:
    full_h, full_w = full_shape
    forbidden: set[tuple[int, int]] = set()
    for r in range(max(0, top - 1), min(full_h, top + th + 1)):
        for c in range(max(0, left - 1), min(full_w, left + tw + 1)):
            forbidden.add((int(r), int(c)))
    return forbidden


def _place_pieces(full_shape: tuple[int, int], forbidden: set[tuple[int, int]], hole_specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    blocked = set(forbidden)
    pieces: list[dict[str, object]] = []
    hole_specs = sorted(hole_specs, key=lambda spec: len(spec["hole_abs"]), reverse=True)
    for spec in hole_specs:
        color = int(spec["color"])
        variants = _piece_variants(spec["hole_abs"], color)
        delta = random.randrange(4)
        variants = variants[delta:] + variants[:delta]
        chosen = None
        for variant in variants:
            positions = _fit_positions(full_shape, variant["safe_rel"])
            random.shuffle(positions)
            for r0, c0 in positions:
                safe_abs = _shift(variant["safe_rel"], r0, c0)
                if safe_abs & blocked:
                    continue
                cells_abs = _shift(variant["cells_rel"], r0, c0)
                marker = random.choice(sorted(cells_abs))
                chosen = {
                    "color": color,
                    "cells_abs": cells_abs,
                    "safe_abs": safe_abs,
                    "marker": marker,
                }
                blocked |= safe_abs
                pieces.append(chosen)
                break
            if chosen is not None:
                break
        if chosen is None:
            return None
    return pieces


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    full_h = random.randint(22, 30)
    full_w = random.randint(22, 30)
    hole_count = random.randint(2, 4)

    colors = BORDER_COLORS.copy()
    random.shuffle(colors)
    specs = [_random_hole_spec(color) for color in colors[:hole_count]]

    min_template_h = max(
        10,
        max(max(r for r, _ in spec["margin_rel"]) - min(r for r, _ in spec["margin_rel"]) + 1 for spec in specs),
    )
    min_template_w = max(
        10,
        max(max(c for _, c in spec["margin_rel"]) - min(c for _, c in spec["margin_rel"]) + 1 for spec in specs),
    )
    max_template_h = min(18, full_h - 6)
    max_template_w = min(18, full_w - 6)
    if min_template_h > max_template_h or min_template_w > max_template_w:
        return None

    template_h = random.randint(min_template_h, max_template_h)
    template_w = random.randint(min_template_w, max_template_w)
    holes = _place_holes((template_h, template_w), specs)
    if holes is None:
        return None

    top_center = (full_h - template_h) // 2
    left_center = (full_w - template_w) // 2
    top = min(max(3, top_center + random.randint(-1, 1)), full_h - template_h - 3)
    left = min(max(3, left_center + random.randint(-1, 1)), full_w - template_w - 3)

    inp = np.full((full_h, full_w), BLACK, dtype=int)
    inp[top : top + template_h, left : left + template_w] = BLUE

    expected = inp.copy()
    for spec in holes:
        color = int(spec["color"])
        for r, c in spec["border_abs"]:
            inp[top + r, left + c] = color
            expected[top + r, left + c] = color
        for r, c in spec["hole_abs"]:
            inp[top + r, left + c] = BLACK
            expected[top + r, left + c] = color

    pieces = _place_pieces((full_h, full_w), _template_forbidden((full_h, full_w), top, left, template_h, template_w), holes)
    if pieces is None:
        return None

    for piece in pieces:
        marker = piece["marker"]
        color = int(piece["color"])
        for r, c in piece["cells_abs"]:
            inp[r, c] = GREY
        inp[marker] = color
        for r, c in piece["cells_abs"]:
            expected[r, c] = BLACK

    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_BUILD_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ARC instance")
