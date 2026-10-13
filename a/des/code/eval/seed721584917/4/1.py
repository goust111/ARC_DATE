import random
from typing import Iterable

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 400


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_coords(coords: Iterable[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.asarray(list(coords), dtype=int)
    pts = pts - pts.min(axis=0)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    return tuple((int(r), int(c)) for r, c in pts)


def _dihedral_variants(coords: Iterable[tuple[int, int]] | np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    pts = np.asarray(list(coords), dtype=int)
    r = pts[:, 0]
    c = pts[:, 1]
    variants = [
        np.stack([r, c], axis=1),
        np.stack([r, -c], axis=1),
        np.stack([-r, c], axis=1),
        np.stack([-r, -c], axis=1),
        np.stack([c, r], axis=1),
        np.stack([c, -r], axis=1),
        np.stack([-c, r], axis=1),
        np.stack([-c, -r], axis=1),
    ]
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for variant in variants:
        norm = _normalize_coords(variant)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _canonical_shape(coords: Iterable[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_dihedral_variants(coords))


def _coords_to_mask(coords: Iterable[tuple[int, int]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(list(coords), dtype=int)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    comps = _components_mask(grid != BG)
    by_col: dict[int, list[tuple[int, int]]] = {}
    for comp in comps:
        if len(comp) != 1:
            continue
        r, c = map(int, comp[0])
        by_col.setdefault(c, []).append((r, int(grid[r, c])))

    candidates = []
    for col, items in by_col.items():
        items = sorted(items)
        rows = [r for r, _ in items]
        colors = [color for _, color in items]
        if len(items) < 3:
            continue
        if len(set(colors)) != len(colors):
            continue
        if any(b - a < 2 for a, b in zip(rows, rows[1:])):
            continue
        candidates.append((len(items), col, items))

    assert candidates, "legend column not found"
    best_len = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_len]
    assert len(best) == 1, "legend must be unique"

    _, col, items = best[0]
    mask = np.zeros_like(grid, dtype=bool)
    for r, _ in items:
        mask[r, col] = True
    return [color for _, color in items], mask


def _extract_rings(grid: np.ndarray, legend_colors: list[int], legend_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    used_mask = legend_mask.copy()
    rings: list[dict] = []
    previous_bbox = None
    for layer, color in enumerate(legend_colors, start=1):
        comps = _components_mask((grid == color) & (~legend_mask))
        assert comps, f"missing ring color {color}"
        comp = max(comps, key=len)
        r0, c0, r1, c1 = _bbox(comp)
        border = _rect_border_coords(r0, c0, r1, c1)
        border_set = set(border)
        comp_set = {tuple(map(int, xy)) for xy in comp}
        assert comp_set <= border_set, "ring pixels must lie on a rectangle border"
        gap_abs = sorted(border_set - comp_set)
        if gap_abs:
            gap_rel = [(r - r0, c - c0) for r, c in gap_abs]
            gap_mask = _coords_to_mask(gap_rel)
            gap_parts = _components_mask(gap_mask)
            assert len(gap_parts) == 1, "each ring gap must be connected"
            gap_sig = _canonical_shape(gap_parts[0])
        else:
            gap_sig = None

        if previous_bbox is not None:
            pr0, pc0, pr1, pc1 = previous_bbox
            assert r0 > pr0 and c0 > pc0 and r1 < pr1 and c1 < pc1
        previous_bbox = (r0, c0, r1, c1)

        used_mask[comp[:, 0], comp[:, 1]] = True
        rings.append(
            {
                "layer": layer,
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "gap_abs": gap_abs,
                "gap_sig": gap_sig,
            }
        )
    return rings, used_mask


def _extract_pieces(grid: np.ndarray, used_mask: np.ndarray) -> list[dict]:
    comps = _components_mask((grid != BG) & (~used_mask))
    pieces: list[dict] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "pieces must be monochrome"
        color = next(iter(colors))
        r0, c0, _, _ = _bbox(comp)
        rel = [(int(r - r0), int(c - c0)) for r, c in comp]
        pieces.append({"color": color, "sig": _canonical_shape(rel)})
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_colors, legend_mask = _find_legend(g)
    rings, used_mask = _extract_rings(g, legend_colors, legend_mask)
    pieces = _extract_pieces(g, used_mask)

    gaps = [ring for ring in rings if ring["gap_sig"] is not None]
    gap_by_sig: dict[tuple[tuple[int, int], ...], dict] = {}
    for ring in gaps:
        sig = ring["gap_sig"]
        assert sig not in gap_by_sig, "gap shapes must be unique"
        gap_by_sig[sig] = ring

    piece_sigs = [piece["sig"] for piece in pieces]
    assert len(piece_sigs) == len(set(piece_sigs)), "piece shapes must be unique"
    assert len(pieces) == len(gaps), "piece count must match gap count"

    outer_r0, outer_c0, outer_r1, outer_c1 = rings[0]["bbox"]
    out = np.full((outer_r1 - outer_r0 + 1, outer_c1 - outer_c0 + 1), BG, dtype=int)

    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            out[r - outer_r0, c - outer_c0] = ring["color"]

    layer_of_color = {color: idx for idx, color in enumerate(legend_colors, start=1)}
    for piece in pieces:
        ring = gap_by_sig.get(piece["sig"])
        assert ring is not None, "every piece must match exactly one gap"
        layer = layer_of_color.get(piece["color"])
        fill = piece["color"] if layer is not None and layer % 2 == 1 else ring["color"]
        for r, c in ring["gap_abs"]:
            out[r - outer_r0, c - outer_c0] = fill

    return out


SHAPE_SPECS = [
    ("line", 2),
    ("line", 3),
    ("line", 4),
    ("line", 5),
    ("corner", 2, 2),
    ("corner", 2, 3),
    ("corner", 2, 4),
    ("corner", 3, 3),
]


def _spec_signature(spec: tuple) -> tuple:
    if spec[0] == "line":
        return spec
    return ("corner", min(spec[1], spec[2]), max(spec[1], spec[2]))


def _spec_fits(spec: tuple, h: int, w: int) -> bool:
    if spec[0] == "line":
        length = spec[1]
        return length < h or length < w
    _, a, b = spec
    return (a < w and b < h) or (b < w and a < h)


def _choose_gap_specs(dimensions: list[tuple[int, int]], rng: random.Random) -> list[tuple] | None:
    order = sorted(range(len(dimensions)), key=lambda i: (min(dimensions[i]), max(dimensions[i])))
    fit_options = {
        idx: [spec for spec in SHAPE_SPECS if _spec_fits(spec, *dimensions[idx])]
        for idx in range(len(dimensions))
    }

    chosen: dict[int, tuple] = {}
    used: set[tuple] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        options = fit_options[idx][:]
        rng.shuffle(options)
        options.sort(key=lambda spec: (0 if spec[0] == "line" else 1, rng.random()))
        for spec in options:
            sig = _spec_signature(spec)
            if sig in used:
                continue
            used.add(sig)
            chosen[idx] = spec
            if backtrack(pos + 1):
                return True
            used.remove(sig)
            chosen.pop(idx, None)
        return False

    if not backtrack(0):
        return None
    return [chosen[i] for i in range(len(dimensions))]


def _instantiate_gap(bbox: tuple[int, int, int, int], spec: tuple, rng: random.Random) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if spec[0] == "line":
        length = spec[1]
        options: list[list[tuple[int, int]]] = []
        if length < w:
            start = rng.randint(0, w - length)
            options.append([(r0, c0 + start + i) for i in range(length)])
            start = rng.randint(0, w - length)
            options.append([(r1, c0 + start + i) for i in range(length)])
        if length < h:
            start = rng.randint(0, h - length)
            options.append([(r0 + start + i, c0) for i in range(length)])
            start = rng.randint(0, h - length)
            options.append([(r0 + start + i, c1) for i in range(length)])
        assert options
        return rng.choice(options)

    _, a, b = spec
    options: list[list[tuple[int, int]]] = []
    for horiz, vert in {(a, b), (b, a)}:
        if horiz >= w or vert >= h:
            continue
        options.append([(r0, c0 + i) for i in range(horiz)] + [(r0 + i, c0) for i in range(1, vert)])
        options.append([(r0, c1 - i) for i in range(horiz)] + [(r0 + i, c1) for i in range(1, vert)])
        options.append([(r1, c0 + i) for i in range(horiz)] + [(r1 - i, c0) for i in range(1, vert)])
        options.append([(r1, c1 - i) for i in range(horiz)] + [(r1 - i, c1) for i in range(1, vert)])
    assert options
    return rng.choice(options)


def _build_frame(n_rings: int, outer_h: int, outer_w: int, ring_colors: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    frame = np.full((outer_h, outer_w), BG, dtype=int)
    bboxes = []
    for idx, color in enumerate(ring_colors):
        r0 = 2 * idx
        c0 = 2 * idx
        r1 = outer_h - 1 - 2 * idx
        c1 = outer_w - 1 - 2 * idx
        bboxes.append((r0, c0, r1, c1))
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            frame[r, c] = color
    return frame, bboxes


def _choose_piece_colors(ring_colors: list[int], rng: random.Random) -> list[int] | None:
    odd_colors = [color for idx, color in enumerate(ring_colors, start=1) if idx % 2 == 1]
    even_colors = [color for idx, color in enumerate(ring_colors, start=1) if idx % 2 == 0]
    distractors = [color for color in COLOR_POOL if color not in ring_colors]

    roles = ["keep", "recolor"] + [rng.choice(["keep", "recolor"]) for _ in range(len(ring_colors) - 2)]
    rng.shuffle(roles)

    piece_colors: list[int] = []
    for target_color, role in zip(ring_colors, roles, strict=False):
        if role == "keep":
            options = [color for color in odd_colors if color != target_color]
            if not options:
                return None
            piece_colors.append(rng.choice(options))
            continue

        even_options = [color for color in even_colors if color != target_color]
        other_options = [color for color in distractors if color != target_color]
        pools = []
        if even_options:
            pools.append(rng.choice(even_options))
        if other_options:
            pools.append(rng.choice(other_options))
        if not pools:
            return None
        piece_colors.append(rng.choice(pools))

    assert any(color in odd_colors for color in piece_colors)
    assert any(color not in odd_colors for color in piece_colors)
    return piece_colors


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_bbox(
    occupied: np.ndarray,
    obj_h: int,
    obj_w: int,
    rng: random.Random,
    tries: int = 400,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    if obj_h > h or obj_w > w:
        return None
    for _ in range(tries):
        r0 = rng.randint(0, h - obj_h)
        c0 = rng.randint(0, w - obj_w)
        if occupied[r0 : r0 + obj_h, c0 : c0 + obj_w].any():
            continue
        return r0, c0
    return None


def _random_piece_mask(gap_coords: list[tuple[int, int]], rng: random.Random) -> np.ndarray:
    rel = _normalize_coords(gap_coords)
    oriented = rng.choice(_dihedral_variants(rel))
    return _coords_to_mask(oriented)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_rings = rng.choice([3, 4])
    min_side = 4 * n_rings + 1
    outer_h = rng.randint(min_side, min_side + 2 + (1 if n_rings == 3 else 0))
    outer_w = rng.randint(min_side, min_side + 2 + (1 if n_rings == 3 else 0))
    ring_colors = rng.sample(COLOR_POOL, n_rings)

    solved_frame, ring_bboxes = _build_frame(n_rings, outer_h, outer_w, ring_colors)
    specs = _choose_gap_specs([(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in ring_bboxes], rng)
    if specs is None:
        return None

    gap_signatures: set[tuple[tuple[int, int], ...]] = set()
    gaps: list[list[tuple[int, int]]] = []
    input_frame = solved_frame.copy()
    for bbox, spec in zip(ring_bboxes, specs, strict=False):
        gap = _instantiate_gap(bbox, spec, rng)
        rel = _normalize_coords(gap)
        sig = _canonical_shape(rel)
        if sig in gap_signatures:
            return None
        gap_signatures.add(sig)
        gaps.append(gap)
        for r, c in gap:
            input_frame[r, c] = BG

    piece_colors = _choose_piece_colors(ring_colors, rng)
    if piece_colors is None:
        return None

    legend_h = 2 * n_rings - 1
    grid_h = outer_h + rng.randint(7, 10)
    grid_w = outer_w + rng.randint(7, 10)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    frame_pos = _place_bbox(occupied, outer_h, outer_w, rng)
    if frame_pos is None:
        return None
    fr0, fc0 = frame_pos
    grid[fr0 : fr0 + outer_h, fc0 : fc0 + outer_w] = input_frame
    _reserve_bbox(occupied, (fr0, fc0, fr0 + outer_h - 1, fc0 + outer_w - 1), pad=1)

    legend_pos = _place_bbox(occupied, legend_h, 1, rng)
    if legend_pos is None:
        return None
    lr0, lc0 = legend_pos
    for idx, color in enumerate(ring_colors):
        grid[lr0 + 2 * idx, lc0] = color
    _reserve_bbox(occupied, (lr0, lc0, lr0 + legend_h - 1, lc0), pad=1)

    for gap, piece_color in zip(gaps, piece_colors, strict=False):
        piece_mask = _random_piece_mask(gap, rng)
        ph, pw = piece_mask.shape
        pos = _place_bbox(occupied, ph, pw, rng)
        if pos is None:
            return None
        pr0, pc0 = pos
        patch = grid[pr0 : pr0 + ph, pc0 : pc0 + pw]
        patch[piece_mask] = piece_color
        _reserve_bbox(occupied, (pr0, pc0, pr0 + ph - 1, pc0 + pw - 1), pad=1)

    expected = solved_frame.copy()
    legend_layers = {color: idx for idx, color in enumerate(ring_colors, start=1)}
    for gap, target_color, piece_color in zip(gaps, ring_colors, piece_colors, strict=False):
        fill = piece_color if piece_color in legend_layers and legend_layers[piece_color] % 2 == 1 else target_color
        for r, c in gap:
            expected[r, c] = fill

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
