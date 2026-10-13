from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

BG_COLOR = 0
GENERATION_ATTEMPTS = 400
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
TEMPLATE_SIDE_MIN = 10
TEMPLATE_SIDE_MAX = 17
PIECE_COUNT_RANGE = (2, 4)
SHAPE_AREA_MIN = 4
SHAPE_AREA_MAX = 7
MAX_SHAPE_SIDE = 5
REPEAT_AREA_PROBABILITY = 0.55
ROTATION_DELTAS = (1, 2, 3)
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _coords_to_mask(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> np.ndarray:
    norm, h, w = _normalize_coords(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[int(r), int(c)] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    m = np.asarray(mask, dtype=bool)
    coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(m)))
    return coords, int(m.shape[0]), int(m.shape[1])


def _rotation_signatures(mask: np.ndarray) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    cur = np.asarray(mask, dtype=bool)
    out = []
    for _ in range(4):
        out.append(_mask_signature(cur))
        cur = np.rot90(cur)
    return out


def _rotation_canonical_key(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    return min(_rotation_signatures(mask))


def _border_mode_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    counts = Counter(map(int, border.tolist()))
    return int(counts.most_common(1)[0][0])


def _colored_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(g)))):
        if int(color) == int(bg):
            continue
        for coords in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": tuple(sorted((int(r), int(c)) for r, c in coords)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _find_template_and_pieces(grid: np.ndarray, bg: int) -> tuple[dict, list[dict]]:
    comps = _colored_components(grid, bg)
    assert comps, "expected at least one non-background component"
    comps.sort(key=lambda comp: (int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
    template = comps[-1]
    pieces = comps[:-1]
    assert pieces, "expected external fragments in addition to the template"
    return template, pieces


def _template_holes(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = map(int, bbox)
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for comp in _components(crop == int(bg)):
        if any(
            rr == 0 or rr == crop.shape[0] - 1 or cc == 0 or cc == crop.shape[1] - 1
            for rr, cc in comp
        ):
            continue
        mask = _coords_to_mask(comp)
        holes.append(
            {
                "coords": tuple(sorted((int(r0 + rr), int(c0 + cc)) for rr, cc in comp)),
                "signature": _mask_signature(mask),
                "area": int(len(comp)),
            }
        )
    assert holes, "expected at least one internal hole"
    return holes


def _piece_info(pieces: list[dict]) -> list[dict]:
    out = []
    for piece in pieces:
        mask = _coords_to_mask(piece["coords"])
        out.append(
            {
                "color": int(piece["color"]),
                "coords": piece["coords"],
                "area": int(piece["area"]),
                "rotations": set(_rotation_signatures(mask)),
            }
        )
    return out


def _match_holes_to_pieces(holes: list[dict], pieces: list[dict]) -> dict[int, int]:
    assert len(holes) == len(pieces), "hole/piece count mismatch"
    compat: dict[int, list[int]] = {}
    for hi, hole in enumerate(holes):
        options = [
            pi
            for pi, piece in enumerate(pieces)
            if int(piece["area"]) == int(hole["area"]) and hole["signature"] in piece["rotations"]
        ]
        assert options, "every hole must have at least one compatible piece"
        compat[hi] = options

    order = sorted(range(len(holes)), key=lambda idx: (len(compat[idx]), int(holes[idx]["area"])))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def search(depth: int) -> bool:
        if depth == len(order):
            return True
        hi = order[depth]
        choices = [pi for pi in compat[hi] if pi not in used]
        for pi in choices:
            used.add(int(pi))
            assignment[int(hi)] = int(pi)
            if search(depth + 1):
                return True
            assignment.pop(int(hi), None)
            used.remove(int(pi))
        return False

    assert search(0), "expected a unique cover by rotation-compatible pieces"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered fragments into the only matching holes inside the gray template.

    Input structure:
    - The background is black.
    - One large gray 4-connected rectangle occupies most of the scene; its bbox is a filled rectangle except for several internal black holes.
    - Outside that rectangle are several smaller monochrome 4-connected fragments, each with a distinct color.

    Rule:
    - Identify the gray template as the largest non-background component and extract the black hole shapes inside its bbox.
    - For each external fragment, compare its shape under the four quarter-turn rotations against the hole shapes.
    - Fill each hole with the color of the uniquely matching fragment, and erase the fragment from its original position back to black.

    Key insight:
    - Area narrows the candidates, but the decisive signal is exact shape matching under rotation only; the fragments are puzzle pieces for the template's internal voids.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_mode_color(g)
    template, raw_pieces = _find_template_and_pieces(g, bg)
    holes = _template_holes(g, bg, template["bbox"])
    pieces = _piece_info(raw_pieces)
    matches = _match_holes_to_pieces(holes, pieces)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)
    for hi, pi in matches.items():
        color = int(pieces[int(pi)]["color"])
        for r, c in holes[int(hi)]["coords"]:
            out[int(r), int(c)] = int(color)
    return out


def _mark_cells_with_halo(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = map(int, forbidden.shape)
    for r, c in cells:
        for dr, dc in DIR8:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                forbidden[rr, cc] = True
        forbidden[int(r), int(c)] = True


def _mark_rect_with_halo(forbidden: np.ndarray, r0: int, c0: int, h0: int, w0: int) -> None:
    h, w = map(int, forbidden.shape)
    for rr in range(int(r0) - 1, int(r0 + h0) + 1):
        for cc in range(int(c0) - 1, int(c0 + w0) + 1):
            if 0 <= rr < h and 0 <= cc < w:
                forbidden[rr, cc] = True


def _cells_from_mask(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(np.asarray(mask, dtype=bool))]


def _candidate_cells(mask: np.ndarray, forbidden: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, forbidden.shape)
    mh, mw = map(int, mask.shape)
    candidates: list[list[tuple[int, int]]] = []
    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            cells = _cells_from_mask(mask, r0, c0)
            if all(not bool(forbidden[r, c]) for r, c in cells):
                candidates.append(cells)
    return candidates


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < int(area):
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt not in cells:
                        frontier.append(nxt)
            choices = list(dict.fromkeys(frontier))
            weighted = []
            for cell in choices:
                touches = sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in DIR4)
                weighted.extend([cell] * max(1, touches))
            cells.add(random.choice(weighted))

        norm, hh, ww = _normalize_coords(tuple(cells))
        if hh < 2 or ww < 2:
            continue
        if hh > MAX_SHAPE_SIDE or ww > MAX_SHAPE_SIDE:
            continue
        if int(area) == int(hh * ww):
            continue

        mask = np.zeros((hh, ww), dtype=bool)
        for r, c in norm:
            mask[int(r), int(c)] = True

        if len(set(_rotation_signatures(mask))) != 4:
            continue
        if int(mask.sum()) * 3 < int(mask.size) * 2:
            continue
        return mask
    return None


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    used: set[tuple[tuple[tuple[int, int], ...], int, int]] = set()
    chosen_areas: list[int] = []

    for i in range(int(count)):
        preferred: list[int] = []
        if chosen_areas and random.random() < REPEAT_AREA_PROBABILITY:
            preferred.append(int(random.choice(chosen_areas)))
        remaining = list(range(SHAPE_AREA_MIN, SHAPE_AREA_MAX + 1))
        random.shuffle(remaining)
        for area in remaining:
            if int(area) not in preferred:
                preferred.append(int(area))

        found = None
        for area in preferred:
            for _ in range(SHAPE_RETRIES):
                shape = _random_polyomino(int(area))
                if shape is None:
                    continue
                key = _rotation_canonical_key(shape)
                if key in used:
                    continue
                used.add(key)
                chosen_areas.append(int(area))
                found = shape
                break
            if found is not None:
                break
        if found is None:
            return None
        shapes.append(found)
    return shapes


def _place_holes(masks: list[np.ndarray], rect_h: int, rect_w: int) -> list[list[tuple[int, int]]] | None:
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    forbidden = np.zeros((rect_h, rect_w), dtype=bool)
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True
    placements: list[list[tuple[int, int]] | None] = [None] * len(masks)

    for idx in order:
        candidates = _candidate_cells(masks[idx], forbidden)
        if not candidates:
            return None
        cells = random.choice(candidates)
        placements[idx] = cells
        _mark_cells_with_halo(forbidden, cells)

    return [cells for cells in placements if cells is not None]


def _place_pieces(
    masks: list[np.ndarray],
    grid_h: int,
    grid_w: int,
    template_bbox: tuple[int, int, int, int],
) -> list[list[tuple[int, int]]] | None:
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True

    tr0, tc0, tr1, tc1 = map(int, template_bbox)
    _mark_rect_with_halo(forbidden, tr0, tc0, tr1 - tr0 + 1, tc1 - tc0 + 1)
    placements: list[list[tuple[int, int]] | None] = [None] * len(masks)

    for idx in order:
        candidates = _candidate_cells(masks[idx], forbidden)
        if not candidates:
            return None
        cells = random.choice(candidates)
        placements[idx] = cells
        _mark_cells_with_halo(forbidden, cells)

    return [cells for cells in placements if cells is not None]


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid_w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    piece_count = random.randint(*PIECE_COUNT_RANGE)
    base_shapes = _sample_shapes(piece_count)
    if base_shapes is None:
        return None

    template_color = random.randint(1, 9)
    piece_colors = random.sample([c for c in range(1, 10) if c != template_color], piece_count)

    hole_masks: list[np.ndarray] = []
    piece_masks: list[np.ndarray] = []
    for shape in base_shapes:
        hole_rot = random.randrange(4)
        hole_mask = np.rot90(shape, hole_rot)
        delta = random.choice(ROTATION_DELTAS)
        piece_mask = np.rot90(hole_mask, delta)
        hole_masks.append(hole_mask)
        piece_masks.append(piece_mask)

    min_template_h = max(
        TEMPLATE_SIDE_MIN,
        max(int(mask.shape[0]) for mask in hole_masks) + 4,
    )
    min_template_w = max(
        TEMPLATE_SIDE_MIN,
        max(int(mask.shape[1]) for mask in hole_masks) + 4,
    )
    max_template_h = min(TEMPLATE_SIDE_MAX, grid_h - 4)
    max_template_w = min(TEMPLATE_SIDE_MAX, grid_w - 4)
    if min_template_h > max_template_h or min_template_w > max_template_w:
        return None

    for _ in range(PLACEMENT_RETRIES):
        template_h = random.randint(min_template_h, max_template_h)
        template_w = random.randint(min_template_w, max_template_w)
        local_holes = _place_holes(hole_masks, template_h, template_w)
        if local_holes is None:
            continue

        max_r0 = grid_h - template_h - 2
        max_c0 = grid_w - template_w - 2
        if max_r0 < 1 or max_c0 < 1:
            continue
        template_r0 = random.randint(1, max_r0)
        template_c0 = random.randint(1, max_c0)
        template_bbox = (
            int(template_r0),
            int(template_c0),
            int(template_r0 + template_h - 1),
            int(template_c0 + template_w - 1),
        )

        piece_positions = _place_pieces(piece_masks, grid_h, grid_w, template_bbox)
        if piece_positions is None:
            continue

        inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
        out = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

        tr0, tc0, tr1, tc1 = template_bbox
        inp[tr0 : tr1 + 1, tc0 : tc1 + 1] = template_color
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = template_color

        for idx, local_cells in enumerate(local_holes):
            abs_cells = [(int(tr0 + r), int(tc0 + c)) for r, c in local_cells]
            for r, c in abs_cells:
                inp[r, c] = BG_COLOR
                out[r, c] = int(piece_colors[idx])

        for idx, cells in enumerate(piece_positions):
            for r, c in cells:
                inp[r, c] = int(piece_colors[idx])

        if np.array_equal(main(inp), out):
            return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate a valid rotated-fragment puzzle")
