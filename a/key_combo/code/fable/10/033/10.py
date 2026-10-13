from __future__ import annotations

from collections import Counter
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))

MIN_SLOT_H = 4
MAX_SLOT_H = 7
MIN_SLOT_W = 4
MAX_SLOT_W = 7
MIN_CANVAS = 18
MAX_CANVAS = 30
MIN_TOTAL_PIECES = 4
MAX_TOTAL_PIECES = 6
PLACEMENT_PADDING = 1
MAX_GEN_ATTEMPTS = 500
MAX_SUB_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    target = int(color)
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) != target:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == target:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _make_piece_from_mask(mask: np.ndarray, color: int, bg: int) -> dict:
    local = np.asarray(mask, dtype=bool)
    assert local.ndim == 2 and local.any()
    cropped, _ = _crop_mask(local)
    coords = np.argwhere(cropped)
    patch = np.full(cropped.shape, int(bg), dtype=int)
    patch[cropped] = int(color)
    return {
        "color": int(color),
        "mask": cropped,
        "patch": patch,
        "local_coords": coords,
        "area": int(cropped.sum()),
        "shape": tuple(map(int, cropped.shape)),
    }


def _make_piece_from_component(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    piece = _make_piece_from_mask(mask, color=color, bg=bg)
    piece["bbox"] = (r0, c0, r1, c1)
    piece["abs_coords"] = coords
    return piece


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _slot_and_pieces(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    slot_candidates = []
    for coords in _find_components_of_color(g, 0):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1):
            slot_candidates.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "shape": (r1 - r0 + 1, c1 - c0 + 1),
                    "area": int(coords.shape[0]),
                }
            )
    assert len(slot_candidates) == 1, "expected exactly one solid black rectangular slot"
    slot = slot_candidates[0]

    pieces = []
    for color in sorted(set(map(int, g.flatten())) - {bg, 0}):
        comps = _find_components_of_color(g, color)
        assert len(comps) == 1, "each visible piece color must form one component"
        pieces.append(_make_piece_from_component(g, comps[0], color=color, bg=bg))
    assert len(pieces) >= 2, "expected at least two colored pieces"
    return bg, slot, pieces


def _piece_placements(piece: dict, slot_h: int, slot_w: int) -> list[tuple[int, int, np.ndarray]]:
    ph, pw = piece["shape"]
    if ph > slot_h or pw > slot_w:
        return []
    placements = []
    for r0 in range(slot_h - ph + 1):
        for c0 in range(slot_w - pw + 1):
            occ = np.zeros((slot_h, slot_w), dtype=bool)
            shifted = piece["local_coords"] + np.array([r0, c0], dtype=int)
            occ[shifted[:, 0], shifted[:, 1]] = True
            placements.append((int(r0), int(c0), occ))
    return placements


def _enumerate_pair_packings(slot_h: int, slot_w: int, piece_a: dict, piece_b: dict) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if piece_a["area"] + piece_b["area"] != slot_h * slot_w:
        return []
    pa = _piece_placements(piece_a, slot_h, slot_w)
    pb = _piece_placements(piece_b, slot_h, slot_w)
    sols = []
    for ar, ac, occ_a in pa:
        for br, bc, occ_b in pb:
            if np.any(occ_a & occ_b):
                continue
            sols.append(((int(ar), int(ac)), (int(br), int(bc))))
    return sols


def _enumerate_input_solutions(grid: np.ndarray) -> tuple[int, dict, list[dict], list[dict]]:
    bg, slot, pieces = _slot_and_pieces(grid)
    slot_h, slot_w = slot["shape"]
    solutions = []
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            packings = _enumerate_pair_packings(slot_h, slot_w, pieces[i], pieces[j])
            for pos_a, pos_b in packings:
                solutions.append(
                    {
                        "pair": (i, j),
                        "positions": (pos_a, pos_b),
                    }
                )
    return bg, slot, pieces, solutions


def _has_area_decoy(slot_shape: tuple[int, int], pieces: list[dict], solution_pair: tuple[int, int]) -> bool:
    slot_h, slot_w = map(int, slot_shape)
    slot_area = slot_h * slot_w
    wanted = tuple(sorted(map(int, solution_pair)))
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if (i, j) == wanted:
                continue
            if pieces[i]["area"] + pieces[j]["area"] != slot_area:
                continue
            if not _enumerate_pair_packings(slot_h, slot_w, pieces[i], pieces[j]):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the unique complementary pair of pieces into the black slot and clear their original locations.

    Input structure:
    - A dominant background color contains one solid black rectangular slot and several separated monochrome pieces of distinct colors.
    - Every colored piece is a single 4-connected component that may be translated but not rotated or flipped.
    - Exactly one pair of pieces can fit inside the slot without overlap and fill all slot cells.

    Rule:
    - Find the unique feasible pair and its unique translation-only packing inside the slot.
    - Remove those two pieces from their original positions.
    - Replace the black slot by the same two pieces drawn in the discovered packed positions, while leaving all other cells unchanged.

    Key insight:
    - The answer comes from a global search over piece pairs and in-slot translations; local cues such as color, distance, or area alone are insufficient because decoys may match the slot area but still fail to pack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, slot, pieces, solutions = _enumerate_input_solutions(g)
    assert len(solutions) == 1, "expected a unique feasible pair and placement"

    solution = solutions[0]
    (i, j) = solution["pair"]
    (pos_a, pos_b) = solution["positions"]
    slot_r0, slot_c0, slot_r1, slot_c1 = slot["bbox"]

    out = g.copy()
    for idx in (i, j):
        coords = pieces[idx]["abs_coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    out[slot_r0 : slot_r1 + 1, slot_c0 : slot_c1 + 1] = int(bg)
    _paint_patch(out, pieces[i]["patch"], slot_r0 + pos_a[0], slot_c0 + pos_a[1], transparent=bg)
    _paint_patch(out, pieces[j]["patch"], slot_r0 + pos_b[0], slot_c0 + pos_b[1], transparent=bg)
    return out


def _random_connected_mask_exact(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    area = int(area)
    feasible = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1) if h * w >= area]
    if not feasible:
        return None
    h, w = random.choice(feasible)
    mask = np.zeros((h, w), dtype=bool)
    sr = random.randrange(h)
    sc = random.randrange(w)
    mask[sr, sc] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True
    cropped, _ = _crop_mask(mask)
    return cropped


def _random_partition(slot_h: int, slot_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    total = slot_h * slot_w
    min_a = max(3, total // 3)
    max_a = min(total - 3, (2 * total) // 3)
    if min_a > max_a:
        return None
    target_a = random.randint(min_a, max_a)
    labels = np.zeros((slot_h, slot_w), dtype=int)
    seed_a = (random.randrange(slot_h), random.randrange(slot_w))
    seed_b = seed_a
    while seed_b == seed_a:
        seed_b = (random.randrange(slot_h), random.randrange(slot_w))
    labels[seed_a] = 1
    labels[seed_b] = 2

    while np.any(labels == 0):
        frontier_a = set()
        frontier_b = set()
        for r, c in np.argwhere(labels == 1):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < slot_h and 0 <= nc < slot_w and labels[nr, nc] == 0:
                    frontier_a.add((nr, nc))
        for r, c in np.argwhere(labels == 2):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < slot_h and 0 <= nc < slot_w and labels[nr, nc] == 0:
                    frontier_b.add((nr, nc))

        if not frontier_a and not frontier_b:
            return None
        if not frontier_a:
            choice = 2
        elif not frontier_b:
            choice = 1
        else:
            need_a = max(0, target_a - int(np.count_nonzero(labels == 1)))
            need_b = max(0, (total - target_a) - int(np.count_nonzero(labels == 2)))
            if need_a == 0 and need_b == 0:
                choice = random.choice([1, 2])
            elif need_a == 0:
                choice = 2
            elif need_b == 0:
                choice = 1
            else:
                choice = 1 if random.random() < need_a / (need_a + need_b) else 2

        frontier = frontier_a if choice == 1 else frontier_b
        nr, nc = random.choice(list(frontier))
        labels[nr, nc] = choice

    a_mask = labels == 1
    b_mask = labels == 2
    if not (_connected(a_mask) and _connected(b_mask)):
        return None
    return a_mask, b_mask


def _unique_real_pair(slot_h: int, slot_w: int, color_a: int, color_b: int, bg: int) -> tuple[dict, dict, tuple[int, int], tuple[int, int]] | None:
    for _ in range(MAX_SUB_ATTEMPTS):
        parts = _random_partition(slot_h, slot_w)
        if parts is None:
            continue
        a_full, b_full = parts
        a_cropped, a_pos = _crop_mask(a_full)
        b_cropped, b_pos = _crop_mask(b_full)
        if min(int(a_cropped.sum()), int(b_cropped.sum())) < 3:
            continue
        piece_a = _make_piece_from_mask(a_cropped, color=color_a, bg=bg)
        piece_b = _make_piece_from_mask(b_cropped, color=color_b, bg=bg)
        packings = _enumerate_pair_packings(slot_h, slot_w, piece_a, piece_b)
        if len(packings) != 1:
            continue
        if packings[0] != (a_pos, b_pos):
            continue
        return piece_a, piece_b, a_pos, b_pos
    return None


def _decoy_pair(slot_h: int, slot_w: int, slot_area: int, color_a: int, color_b: int, bg: int) -> tuple[dict, dict] | None:
    for _ in range(MAX_SUB_ATTEMPTS):
        area_a = random.randint(2, slot_area - 2)
        area_b = slot_area - area_a
        mask_a = _random_connected_mask_exact(area_a, slot_h, slot_w)
        mask_b = _random_connected_mask_exact(area_b, slot_h, slot_w)
        if mask_a is None or mask_b is None:
            continue
        piece_a = _make_piece_from_mask(mask_a, color=color_a, bg=bg)
        piece_b = _make_piece_from_mask(mask_b, color=color_b, bg=bg)
        if _enumerate_pair_packings(slot_h, slot_w, piece_a, piece_b):
            continue
        return piece_a, piece_b
    return None


def _extra_piece(slot_h: int, slot_w: int, slot_area: int, color: int, bg: int) -> dict | None:
    for _ in range(MAX_SUB_ATTEMPTS):
        area = random.randint(2, max(2, slot_area - 1))
        max_h = min(slot_h + 2, max(2, slot_area))
        max_w = min(slot_w + 2, max(2, slot_area))
        mask = _random_connected_mask_exact(area, max_h=max_h, max_w=max_w)
        if mask is None:
            continue
        return _make_piece_from_mask(mask, color=color, bg=bg)
    return None


def _place_objects(canvas_h: int, canvas_w: int, bg: int, slot_shape: tuple[int, int], pieces: list[dict]) -> tuple[np.ndarray, dict] | None:
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: dict[str, tuple[int, int]] = {}

    objects = [("slot", np.zeros(slot_shape, dtype=int))]
    for idx, piece in enumerate(pieces):
        objects.append((f"piece_{idx}", piece["patch"]))
    objects.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    for name, patch in objects:
        h, w = patch.shape
        if h > canvas_h or w > canvas_w:
            return None
        candidates = []
        for _ in range(250):
            r0 = random.randint(0, canvas_h - h)
            c0 = random.randint(0, canvas_w - w)
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(canvas_h, r0 + h + PLACEMENT_PADDING)
            cc1 = min(canvas_w, c0 + w + PLACEMENT_PADDING)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
            if len(candidates) >= 12:
                break
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[name] = (int(r0), int(c0))
        if name == "slot":
            canvas[r0 : r0 + h, c0 : c0 + w] = 0
        else:
            _paint_patch(canvas, patch, r0, c0, transparent=bg)
        rr0 = max(0, r0 - PLACEMENT_PADDING)
        cc0 = max(0, c0 - PLACEMENT_PADDING)
        rr1 = min(canvas_h, r0 + h + PLACEMENT_PADDING)
        cc1 = min(canvas_w, c0 + w + PLACEMENT_PADDING)
        blocked[rr0:rr1, cc0:cc1] = True
    return canvas, placements


def _render_expected(inp: np.ndarray, bg: int, slot_bbox: tuple[int, int, int, int], piece_a: dict, piece_b: dict, src_a: tuple[int, int], src_b: tuple[int, int], dst_a: tuple[int, int], dst_b: tuple[int, int]) -> np.ndarray:
    out = inp.copy()
    ar, ac = src_a
    br, bc = src_b
    ah, aw = piece_a["shape"]
    bh, bw = piece_b["shape"]
    out[ar : ar + ah, ac : ac + aw][piece_a["mask"]] = int(bg)
    out[br : br + bh, bc : bc + bw][piece_b["mask"]] = int(bg)
    r0, c0, r1, c1 = slot_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    _paint_patch(out, piece_a["patch"], r0 + dst_a[0], c0 + dst_a[1], transparent=bg)
    _paint_patch(out, piece_b["patch"], r0 + dst_b[0], c0 + dst_b[1], transparent=bg)
    return out


def _generate_case() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    available_colors = [c for c in PALETTE if c != bg]
    total_pieces = random.randint(MIN_TOTAL_PIECES, MAX_TOTAL_PIECES)
    if len(available_colors) < total_pieces:
        return None
    colors = random.sample(available_colors, total_pieces)

    slot_h = random.randint(MIN_SLOT_H, MAX_SLOT_H)
    slot_w = random.randint(MIN_SLOT_W, MAX_SLOT_W)
    slot_area = slot_h * slot_w

    real = _unique_real_pair(slot_h, slot_w, colors[0], colors[1], bg)
    if real is None:
        return None
    real_a, real_b, dst_a, dst_b = real

    decoys = _decoy_pair(slot_h, slot_w, slot_area, colors[2], colors[3], bg)
    if decoys is None:
        return None
    all_pieces = [real_a, real_b, decoys[0], decoys[1]]

    for color in colors[4:]:
        extra = _extra_piece(slot_h, slot_w, slot_area, color=color, bg=bg)
        if extra is None:
            return None
        all_pieces.append(extra)

    min_dim = max(
        MIN_CANVAS,
        slot_h + 4,
        slot_w + 4,
        max(piece["shape"][0] for piece in all_pieces) + 4,
        max(piece["shape"][1] for piece in all_pieces) + 4,
    )
    if min_dim > MAX_CANVAS:
        return None

    for _ in range(MAX_SUB_ATTEMPTS):
        canvas_h = random.randint(min_dim, MAX_CANVAS)
        canvas_w = random.randint(min_dim, MAX_CANVAS)
        placed = _place_objects(canvas_h, canvas_w, bg, (slot_h, slot_w), all_pieces)
        if placed is None:
            continue
        inp, placements = placed
        slot_r0, slot_c0 = placements["slot"]
        slot_bbox = (slot_r0, slot_c0, slot_r0 + slot_h - 1, slot_c0 + slot_w - 1)
        src_a = placements["piece_0"]
        src_b = placements["piece_1"]
        expected = _render_expected(inp, bg, slot_bbox, real_a, real_b, src_a, src_b, dst_a, dst_b)

        try:
            solved = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue

        bg2, slot2, pieces2, solutions = _enumerate_input_solutions(inp)
        if bg2 != bg or slot2["bbox"] != slot_bbox or len(solutions) != 1:
            continue
        sol = solutions[0]
        pair_colors = {pieces2[idx]["color"] for idx in sol["pair"]}
        if pair_colors != {real_a["color"], real_b["color"]}:
            continue
        if real_a["area"] + real_b["area"] != slot_area:
            continue
        if not _has_area_decoy(slot2["shape"], pieces2, sol["pair"]):
            continue

        counts = Counter(map(int, inp.flatten()))
        if counts[int(bg)] <= max(counts[c] for c in counts if c != int(bg)):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-pack instance")
