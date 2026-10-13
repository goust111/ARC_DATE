import random
from typing import Any

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotate(arr: np.ndarray, steps_cw: int) -> np.ndarray:
    return np.rot90(arr, (-steps_cw) % 4).copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    trimmed = _trim_mask(mask)
    return trimmed.shape, trimmed.astype(np.uint8).tobytes()


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {_mask_key(_rotate(mask, k)) for k in range(4)}


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_rotation_keys(mask))


def _find_colored_components(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    comps: list[dict[str, Any]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "mask": _coords_to_mask(coords),
                    "area": int(len(coords)),
                }
            )
    return comps


def _quadrant_steps(piece_bbox: tuple[int, int, int, int], template_bbox: tuple[int, int, int, int]) -> int:
    pr0, pc0, pr1, pc1 = piece_bbox
    tr0, tc0, tr1, tc1 = template_bbox
    piece_r = (pr0 + pr1) / 2.0
    piece_c = (pc0 + pc1) / 2.0
    center_r = (tr0 + tr1) / 2.0
    center_c = (tc0 + tc1) / 2.0

    if piece_r < center_r and piece_c < center_c:
        return 0
    if piece_r < center_r and piece_c > center_c:
        return 1
    if piece_r > center_r and piece_c > center_c:
        return 2
    if piece_r > center_r and piece_c < center_c:
        return 3
    raise AssertionError("piece must lie strictly inside one quadrant")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered colored pieces to the template holes, then rotate each piece
    according to its quadrant relative to the template center before inserting it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_colored_components(g, BG)
    assert comps, "grid must contain non-background objects"

    template = max(comps, key=lambda comp: comp["area"])
    tr0, tc0, tr1, tc1 = template["bbox"]
    template_crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    hole_mask = template_crop == BG
    holes = []
    for coords in _components(hole_mask):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == hole_mask.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == hole_mask.shape[1] - 1)
        ):
            continue
        hr0, hc0, hr1, hc1 = _bbox(coords)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        mask[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append({"bbox": (hr0, hc0, hr1, hc1), "mask": mask})
    assert holes, "template must contain holes"

    hole_by_key: dict[tuple[tuple[int, int], bytes], dict[str, Any]] = {}
    for hole in holes:
        key = _mask_key(hole["mask"])
        assert key not in hole_by_key, "hole shapes must be unique after rotation is applied"
        hole_by_key[key] = hole

    used_holes: set[tuple[tuple[int, int], bytes]] = set()
    pieces = [comp for comp in comps if comp is not template]
    assert len(pieces) == len(holes), "piece count must equal hole count"

    for piece in pieces:
        steps = _quadrant_steps(piece["bbox"], template["bbox"])
        rotated_mask = _rotate(piece["mask"], steps)
        key = _mask_key(rotated_mask)
        assert key in hole_by_key, "every rotated piece must fit one hole"
        assert key not in used_holes, "piece-hole pairing must be one-to-one"
        used_holes.add(key)

        hole = hole_by_key[key]
        hr0, hc0, hr1, hc1 = hole["bbox"]
        patch = template_crop[hr0 : hr1 + 1, hc0 : hc1 + 1]
        patch[rotated_mask] = int(piece["color"])
        template_crop[hr0 : hr1 + 1, hc0 : hc1 + 1] = patch

    assert len(used_holes) == len(holes)
    return template_crop


def _polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand not in cells:
                    frontier.add(cand)
        cells.add(random.choice(sorted(frontier)))
    coords = np.array(sorted(cells), dtype=int)
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _random_shape() -> np.ndarray | None:
    area = random.randint(4, 6)
    mask = _polyomino(area)
    h, w = mask.shape
    if h < 2 or w < 2 or h > 4 or w > 4:
        return None
    if len(_rotation_keys(mask)) != 4:
        return None
    return mask


def _quadrant_plan(indices: list[int], masks: list[np.ndarray]) -> dict[str, Any]:
    order = indices[:]
    random.shuffle(order)

    if len(order) == 1:
        idx = order[0]
        h, w = masks[idx].shape
        return {"order": order, "positions": [(0, 0)], "zone_h": h, "zone_w": w}

    assert len(order) == 2
    first, second = order
    h1, w1 = masks[first].shape
    h2, w2 = masks[second].shape

    if random.random() < 0.5:
        zone_h = h1 + 1 + h2
        zone_w = max(w1, w2)
        pos1 = (0, random.randint(0, zone_w - w1))
        pos2 = (h1 + 1, random.randint(0, zone_w - w2))
    else:
        zone_h = max(h1, h2)
        zone_w = w1 + 1 + w2
        pos1 = (random.randint(0, zone_h - h1), 0)
        pos2 = (random.randint(0, zone_h - h2), w1 + 1)

    return {"order": order, "positions": [pos1, pos2], "zone_h": zone_h, "zone_w": zone_w}


def _build_template(slot_masks: list[np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n = len(slot_masks)
    rows = 2
    cols = 3 if n == 5 or (n == 4 and random.random() < 0.5) else 2
    assert rows * cols >= n

    max_h = max(int(mask.shape[0]) for mask in slot_masks)
    max_w = max(int(mask.shape[1]) for mask in slot_masks)
    cell_h = max(4, max_h + random.randint(0, 1))
    cell_w = max(4, max_w + random.randint(0, 1))
    gap = 1

    th = 2 + rows * cell_h + (rows - 1) * gap
    tw = 2 + cols * cell_w + (cols - 1) * gap
    template = np.full((th, tw), TEMPLATE_COLOR, dtype=int)

    cell_ids = list(range(rows * cols))
    random.shuffle(cell_ids)
    chosen = cell_ids[:n]

    holes: list[dict[str, Any]] = []
    for mask, cell_id in zip(slot_masks, chosen):
        cell_r, cell_c = divmod(cell_id, cols)
        base_r = 1 + cell_r * (cell_h + gap)
        base_c = 1 + cell_c * (cell_w + gap)
        h, w = mask.shape
        off_r = random.randint(0, cell_h - h)
        off_c = random.randint(0, cell_w - w)
        r = base_r + off_r
        c = base_c + off_c
        patch = template[r : r + h, c : c + w]
        patch[mask] = BG
        template[r : r + h, c : c + w] = patch
        holes.append({"bbox": (r, c, r + h - 1, c + w - 1), "mask": mask})

    return template, holes


def _gen_attempt() -> np.ndarray | None:
    n_pieces = random.choice([4, 5])
    quadrants = [0, 1, 2, 3]
    if n_pieces == 5:
        quadrants.append(random.randint(0, 3))
    random.shuffle(quadrants)

    pieces: list[dict[str, Any]] = []
    seen_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for q in quadrants:
        shape = None
        for _ in range(100):
            cand = _random_shape()
            if cand is None:
                continue
            canon = _canonical_rotation_key(cand)
            if canon in seen_shapes:
                continue
            seen_shapes.add(canon)
            shape = cand
            break
        if shape is None:
            return None
        observed = _rotate(shape, random.randint(0, 3))
        slot = _rotate(observed, q)
        pieces.append({"quadrant": q, "observed": observed, "slot": slot})

    colors = random.sample(PIECE_COLORS, n_pieces)
    random.shuffle(colors)
    for piece, color in zip(pieces, colors):
        piece["color"] = color

    slot_masks = [piece["slot"] for piece in pieces]
    template, holes = _build_template(slot_masks)

    quadrant_indices = {0: [], 1: [], 2: [], 3: []}
    observed_masks = [piece["observed"] for piece in pieces]
    for idx, piece in enumerate(pieces):
        quadrant_indices[int(piece["quadrant"])].append(idx)

    plans = {q: _quadrant_plan(indices, observed_masks) for q, indices in quadrant_indices.items()}

    extra_v = random.randint(2, 4)
    extra_h = random.randint(2, 4)
    top_margin = max(plans[0]["zone_h"], plans[1]["zone_h"]) + extra_v
    bottom_margin = max(plans[3]["zone_h"], plans[2]["zone_h"]) + extra_v
    left_margin = max(plans[0]["zone_w"], plans[3]["zone_w"]) + extra_h
    right_margin = max(plans[1]["zone_w"], plans[2]["zone_w"]) + extra_h

    th, tw = template.shape
    grid_h = top_margin + th + bottom_margin
    grid_w = left_margin + tw + right_margin
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[top_margin : top_margin + th, left_margin : left_margin + tw] = template

    origins = {
        0: (random.randint(0, top_margin - plans[0]["zone_h"]), random.randint(0, left_margin - plans[0]["zone_w"])),
        1: (
            random.randint(0, top_margin - plans[1]["zone_h"]),
            left_margin + tw + random.randint(0, right_margin - plans[1]["zone_w"]),
        ),
        2: (
            top_margin + th + random.randint(0, bottom_margin - plans[2]["zone_h"]),
            left_margin + tw + random.randint(0, right_margin - plans[2]["zone_w"]),
        ),
        3: (
            top_margin + th + random.randint(0, bottom_margin - plans[3]["zone_h"]),
            random.randint(0, left_margin - plans[3]["zone_w"]),
        ),
    }

    expected = template.copy()
    for piece, hole in zip(pieces, holes):
        hr0, hc0, hr1, hc1 = hole["bbox"]
        patch = expected[hr0 : hr1 + 1, hc0 : hc1 + 1]
        patch[hole["mask"]] = int(piece["color"])
        expected[hr0 : hr1 + 1, hc0 : hc1 + 1] = patch

    for q, plan in plans.items():
        base_r, base_c = origins[q]
        for idx, (off_r, off_c) in zip(plan["order"], plan["positions"]):
            piece = pieces[idx]
            mask = piece["observed"]
            h, w = mask.shape
            r = base_r + off_r
            c = base_c + off_c
            patch = grid[r : r + h, c : c + w]
            patch[mask] = int(piece["color"])
            grid[r : r + h, c : c + w] = patch

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
