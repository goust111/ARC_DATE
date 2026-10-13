import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREY = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = _neighbors(connectivity)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (int(mask.shape[0]), int(mask.shape[1])), mask.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate marked outside fragments and fit them into the matching holes of the central template.

    Input structure:
    - The background is black.
    - A large grey template occupies the middle of the grid and contains several black holes.
    - Outside the template are small monochrome colored fragments.
    - Each fragment is 8-connected to 1, 2, or 3 adjacent red marker pixels.

    Rule:
    - Find each fragment and count its adjacent red marker pixels.
    - Rotate the fragment clockwise by that many quarter-turns.
    - Match the rotated fragment to one template hole of the same cropped shape and place its color there.
    - Return the tight crop of the template after all holes are filled; red markers do not appear in the output.

    Key insight:
    The red marker count is a rotation code, so the outside fragment must be turned first before it can be matched to the correct hole in the grey template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_components = _components(g == GREY, connectivity=4)
    assert grey_components, "expected a grey template"
    grey_components.sort(key=lambda comp: len(comp), reverse=True)
    template_coords = grey_components[0]
    tr0, tc0, tr1, tc1 = _bbox(template_coords)
    crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    holes = []
    for comp in _components(crop == BG, connectivity=4):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == crop.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == crop.shape[1] - 1):
            continue
        hole_mask, hole_bbox = _mask_from_coords(comp)
        holes.append({"mask": hole_mask, "bbox": hole_bbox, "key": _mask_key(hole_mask)})
    assert holes, "expected at least one template hole"

    pieces = []
    outside_mask = (g != BG) & (g != GREY)
    for comp in _components(outside_mask, connectivity=8):
        vals = {int(g[r, c]) for r, c in comp}
        assert RED in vals, "every outside component must contain red markers"
        body_coords = np.array([(int(r), int(c)) for r, c in comp if int(g[r, c]) != RED], dtype=int)
        assert len(body_coords) > 0, "every fragment needs a non-red body"
        body_colors = {int(g[r, c]) for r, c in body_coords}
        assert len(body_colors) == 1, "each fragment body must be monochrome"
        body_color = next(iter(body_colors))
        turns = sum(1 for r, c in comp if int(g[r, c]) == RED)
        assert turns in (1, 2, 3), "red marker count must be 1, 2, or 3"
        body_mask, _ = _mask_from_coords(body_coords)
        rotated = np.rot90(body_mask, (-turns) % 4)
        pieces.append({"color": body_color, "key": _mask_key(rotated)})

    assert len(pieces) == len(holes), "fragment count must match hole count"
    holes_by_key: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, hole in enumerate(holes):
        holes_by_key.setdefault(hole["key"], []).append(idx)

    used_holes: set[int] = set()
    for piece in pieces:
        candidates = [idx for idx in holes_by_key.get(piece["key"], []) if idx not in used_holes]
        assert candidates, "every rotated fragment must match a unique hole"
        hole = holes[candidates[0]]
        used_holes.add(candidates[0])
        hr0, hc0, hr1, hc1 = hole["bbox"]
        patch = crop[hr0 : hr1 + 1, hc0 : hc1 + 1]
        patch[hole["mask"]] = int(piece["color"])

    return crop


def _choose_hole_specs(rng: random.Random, num_holes: int) -> list[dict[str, int]]:
    transposed_pairs = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    a, b = rng.choice(transposed_pairs)
    odd_turn = rng.choice([1, 3])
    other_odd = 3 if odd_turn == 1 else 1
    if rng.random() < 0.5:
        specs = [{"h": a, "w": b, "turns": 2}, {"h": b, "w": a, "turns": odd_turn}]
    else:
        specs = [{"h": a, "w": b, "turns": odd_turn}, {"h": b, "w": a, "turns": 2}]

    used_oriented = {(spec["h"], spec["w"]) for spec in specs}
    used_unoriented = {tuple(sorted((spec["h"], spec["w"]))) for spec in specs}

    while len(specs) < num_holes:
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        if (h, w) in used_oriented or tuple(sorted((h, w))) in used_unoriented:
            continue
        turns = other_odd if len(specs) == 2 else rng.choice([1, 2, 3])
        specs.append({"h": h, "w": w, "turns": turns})
        used_oriented.add((h, w))
        used_unoriented.add(tuple(sorted((h, w))))

    rng.shuffle(specs)
    return specs


def _rectangles_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_holes(
    rng: random.Random, template_h: int, template_w: int, specs: list[dict[str, int]]
) -> list[dict[str, int]] | None:
    placed: list[dict[str, int]] = []
    for spec in sorted(specs, key=lambda item: item["h"] * item["w"], reverse=True):
        h = spec["h"]
        w = spec["w"]
        success = False
        for _ in range(200):
            r0 = rng.randint(1, template_h - h - 1)
            c0 = rng.randint(1, template_w - w - 1)
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_rectangles_separated(rect, other["rect"], gap=1) for other in placed):
                placed.append({**spec, "r0": r0, "c0": c0, "rect": rect})
                success = True
                break
        if not success:
            return None
    return placed


def _marker_cells(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    options: list[list[tuple[int, int]]] = []
    for start in range(-1, w - count + 2):
        options.append([(-1, start + offset) for offset in range(count)])
        options.append([(h, start + offset) for offset in range(count)])
    for start in range(-1, h - count + 2):
        options.append([(start + offset, -1) for offset in range(count)])
        options.append([(start + offset, w) for offset in range(count)])
    return rng.choice(options)


def _normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = [(r - min_r, c - min_c) for r, c in pts]
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return norm, height, width


def _can_place(blocked: np.ndarray, cells: list[tuple[int, int]], r0: int, c0: int) -> bool:
    h, w = blocked.shape
    for rr, cc in cells:
        r = r0 + rr
        c = c0 + cc
        if blocked[r, c]:
            return False
        r_lo = max(0, r - 1)
        r_hi = min(h - 1, r + 1)
        c_lo = max(0, c - 1)
        c_hi = min(w - 1, c + 1)
        if np.any(blocked[r_lo : r_hi + 1, c_lo : c_hi + 1]):
            return False
    return True


def _central_start(total: int, span: int, rng: random.Random) -> int:
    center = (total - span) // 2
    lo = max(2, center - 2)
    hi = min(total - span - 3, center + 2)
    return rng.randint(lo, hi)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_holes = rng.choice([3, 4])
    specs = _choose_hole_specs(rng, num_holes)

    min_template_h = max(max(spec["h"] for spec in specs) + 4, 11 + num_holes - 3)
    min_template_w = max(max(spec["w"] for spec in specs) + 4, 11 + num_holes - 3)
    template_h = rng.randint(min_template_h, min(18, min_template_h + 5))
    template_w = rng.randint(min_template_w, min(18, min_template_w + 5))

    placed_holes = _place_holes(rng, template_h, template_w, specs)
    if placed_holes is None:
        return None

    grid_h = rng.randint(max(template_h + 8, 22), 30)
    grid_w = rng.randint(max(template_w + 8, 22), 30)
    tr0 = _central_start(grid_h, template_h, rng)
    tc0 = _central_start(grid_w, template_w, rng)

    colors = rng.sample(PALETTE, num_holes)
    for spec, color in zip(placed_holes, colors):
        spec["color"] = color

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[tr0 : tr0 + template_h, tc0 : tc0 + template_w] = GREY
    expected = np.full((template_h, template_w), GREY, dtype=int)

    for spec in placed_holes:
        r0 = spec["r0"]
        c0 = spec["c0"]
        h = spec["h"]
        w = spec["w"]
        color = spec["color"]
        grid[tr0 + r0 : tr0 + r0 + h, tc0 + c0 : tc0 + c0 + w] = BG
        expected[r0 : r0 + h, c0 : c0 + w] = color

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    blocked[tr0 : tr0 + template_h, tc0 : tc0 + template_w] = True

    for spec in rng.sample(placed_holes, len(placed_holes)):
        hole_mask = np.ones((spec["h"], spec["w"]), dtype=bool)
        body_mask = np.rot90(hole_mask, spec["turns"])
        bh, bw = map(int, body_mask.shape)
        body_cells = [(r, c) for r in range(bh) for c in range(bw) if body_mask[r, c]]
        marker_cells = _marker_cells(bh, bw, spec["turns"], rng)
        all_cells, comp_h, comp_w = _normalize_cells(body_cells + marker_cells)
        body_set = set(body_cells)
        marker_set = set(marker_cells)
        mapping = {}
        for orig, norm in zip(body_cells + marker_cells, all_cells):
            if orig in body_set:
                mapping[norm] = spec["color"]
            elif orig in marker_set:
                mapping[norm] = RED

        placed = False
        for _ in range(400):
            rr = rng.randint(0, grid_h - comp_h)
            cc = rng.randint(0, grid_w - comp_w)
            if not _can_place(blocked, all_cells, rr, cc):
                continue
            for nr, nc in all_cells:
                grid[rr + nr, cc + nc] = mapping[(nr, nc)]
                blocked[rr + nr, cc + nc] = True
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
