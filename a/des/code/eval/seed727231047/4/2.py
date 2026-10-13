from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
GEN_TEMPLATE_COLOR = 9
GEN_RETRIES = 500
GEN_PIECE_COUNT = (2, 5)
GEN_TEMPLATE_SIDE = (11, 18)
GEN_GRID_MARGIN = (4, 8)
GEN_SHAPE_BBOX = (2, 4)
GEN_SHAPE_AREA = (3, 6)
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    out, _ = _tight_mask_from_coords(coords)
    return out


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = (rotated.shape, rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rotated.copy())
    return variants


def _rotation_key(mask: np.ndarray) -> tuple:
    keys = []
    for rotated in _rotation_variants(mask):
        keys.append((rotated.shape[0], rotated.shape[1], rotated.tobytes()))
    return min(keys)


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        coords = np.argwhere(grid == int(color))
        if len(coords) == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        allowed = (sub == int(bg)) | (sub == int(color))
        if not bool(np.all(allowed)):
            continue
        if not bool(np.all(sub[0, :] == int(color))):
            continue
        if not bool(np.all(sub[-1, :] == int(color))):
            continue
        if not bool(np.all(sub[:, 0] == int(color))):
            continue
        if not bool(np.all(sub[:, -1] == int(color))):
            continue
        if not bool(np.any(sub == int(bg))):
            continue
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        color_area = int(len(coords))
        candidate = (bbox_area, color_area, (r0, c0, r1, c1))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None, "expected one rectangular template with internal holes"
    bbox = best[2]
    r0, c0, r1, c1 = bbox
    return int(grid[r0, c0]), bbox


def _extract_slots(template_crop: np.ndarray, bg: int) -> list[dict]:
    slots = []
    for slot_id, coords in enumerate(_components(template_crop == int(bg))):
        if any(
            int(r) == 0
            or int(c) == 0
            or int(r) == template_crop.shape[0] - 1
            or int(c) == template_crop.shape[1] - 1
            for r, c in coords
        ):
            continue
        mask, bbox = _tight_mask_from_coords(coords)
        slots.append(
            {
                "id": int(slot_id),
                "coords": coords.copy(),
                "mask": mask,
                "bbox": bbox,
                "key": _rotation_key(mask),
            }
        )
    slots.sort(key=lambda slot: (-int(slot["mask"].sum()), int(slot["bbox"][0]), int(slot["bbox"][1])))
    for idx, slot in enumerate(slots):
        slot["id"] = int(idx)
    return slots


def _extract_pieces(grid: np.ndarray, bg: int, template_color: int) -> list[dict]:
    piece_mask = (grid != int(bg)) & (grid != int(template_color))
    pieces = []
    for piece_id, coords in enumerate(_components(piece_mask)):
        colors = {int(grid[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each loose piece must be monochrome"
        mask, bbox = _tight_mask_from_coords(coords)
        pieces.append(
            {
                "id": int(piece_id),
                "coords": coords.copy(),
                "mask": mask,
                "bbox": bbox,
                "color": int(next(iter(colors))),
                "key": _rotation_key(mask),
            }
        )
    pieces.sort(key=lambda piece: (-int(piece["mask"].sum()), int(piece["bbox"][0]), int(piece["bbox"][1])))
    for idx, piece in enumerate(pieces):
        piece["id"] = int(idx)
    return pieces


def _match_slots(slots: list[dict], pieces: list[dict]) -> dict[int, int]:
    assert len(slots) == len(pieces), "piece count must equal slot count"
    slot_candidates: dict[int, list[int]] = {}
    for slot in slots:
        candidates = [int(piece["id"]) for piece in pieces if piece["key"] == slot["key"]]
        assert candidates, "every slot needs at least one matching piece"
        slot_candidates[int(slot["id"])] = candidates

    order = sorted((int(slot["id"]) for slot in slots), key=lambda sid: len(slot_candidates[sid]))
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        sid = order[i]
        for pid in slot_candidates[sid]:
            if pid in used:
                continue
            used.add(pid)
            assignment[sid] = pid
            if backtrack(i + 1):
                return True
            used.remove(pid)
            del assignment[sid]
        return False

    assert backtrack(0), "slot-piece matching must be one-to-one"
    return assignment


def _slot_neighbors(template_crop: np.ndarray, owner: np.ndarray) -> list[set[int]]:
    count = int(owner.max()) + 1 if owner.size else 0
    neighbors = [set() for _ in range(count)]
    h, w = owner.shape
    template_color = int(template_crop[0, 0])
    for r in range(h):
        for c in range(w):
            sid = int(owner[r, c])
            if sid < 0:
                continue
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(owner[nr, nc])
                    if other >= 0 and other != sid:
                        neighbors[sid].add(other)
                nr2, nc2 = r + 2 * dr, c + 2 * dc
                if 0 <= nr2 < h and 0 <= nc2 < w and int(template_crop[nr, nc]) == template_color:
                    other = int(owner[nr2, nc2])
                    if other >= 0 and other != sid:
                        neighbors[sid].add(other)
    return neighbors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate loose monochrome pieces into the matching template holes, then recolor
    neighboring filled slots by swapping their original piece colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    template_color, (r0, c0, r1, c1) = _find_template(g, bg)
    template_crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    slots = _extract_slots(template_crop, bg)
    pieces = _extract_pieces(g, bg, template_color)
    assignment = _match_slots(slots, pieces)
    piece_by_id = {int(piece["id"]): piece for piece in pieces}

    owner = np.full(template_crop.shape, -1, dtype=int)
    original_colors: list[int] = [0] * len(slots)
    out = template_crop.copy()
    for slot in slots:
        sid = int(slot["id"])
        pid = assignment[sid]
        color = int(piece_by_id[pid]["color"])
        original_colors[sid] = color
        coords = slot["coords"]
        out[coords[:, 0], coords[:, 1]] = color
        owner[coords[:, 0], coords[:, 1]] = sid

    neighbors = _slot_neighbors(template_crop, owner)
    final_colors = original_colors[:]
    for sid, adjacent in enumerate(neighbors):
        if not adjacent:
            continue
        incoming = [original_colors[other] for other in adjacent]
        if all(color == incoming[0] for color in incoming):
            final_colors[sid] = incoming[0]

    for slot in slots:
        sid = int(slot["id"])
        coords = slot["coords"]
        out[coords[:, 0], coords[:, 1]] = int(final_colors[sid])
    return out


def _grow_shape(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((int(nr), int(nc)))

    add_frontier(*start)
    while int(mask.sum()) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        rr, cc = cell
        mask[rr, cc] = True
        add_frontier(rr, cc)

    tight = _tight_mask(mask)
    if tight.shape[0] > GEN_SHAPE_BBOX[1] or tight.shape[1] > GEN_SHAPE_BBOX[1]:
        return None
    if tight.shape[0] < GEN_SHAPE_BBOX[0] and tight.shape[1] < GEN_SHAPE_BBOX[0]:
        return None
    return tight


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    keys: set[tuple] = set()
    for _ in range(count):
        for _attempt in range(100):
            height = random.randint(*GEN_SHAPE_BBOX)
            width = random.randint(*GEN_SHAPE_BBOX)
            max_area = min(GEN_SHAPE_AREA[1], height * width)
            min_area = min(GEN_SHAPE_AREA[0], max_area)
            area = random.randint(min_area, max_area)
            mask = _grow_shape(height, width, area)
            if mask is None:
                continue
            key = _rotation_key(mask)
            if key in keys:
                continue
            keys.add(key)
            shapes.append(mask)
            break
        else:
            return None
    return shapes


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _normalize_coord_groups(groups: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[int, int]]:
    all_coords = np.concatenate(groups, axis=0)
    min_r = int(all_coords[:, 0].min())
    min_c = int(all_coords[:, 1].min())
    shifted = [coords - np.array([min_r, min_c], dtype=int) for coords in groups]
    all_shifted = np.concatenate(shifted, axis=0)
    height = int(all_shifted[:, 0].max()) + 1
    width = int(all_shifted[:, 1].max()) + 1
    return shifted, (height, width)


def _build_pair_group(mask_a: np.ndarray, mask_b: np.ndarray) -> list[np.ndarray] | None:
    coords_a = _coords_from_mask(mask_a)
    coords_b = _coords_from_mask(mask_b)
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    candidates: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for dr in range(-hb - 1, ha + 2):
        for dc in range(-wb - 1, wa + 2):
            shifted_b = coords_b + np.array([dr, dc], dtype=int)
            set_a = {tuple(map(int, xy)) for xy in coords_a}
            set_b = {tuple(map(int, xy)) for xy in shifted_b}
            if set_a & set_b:
                continue
            direct_touch = False
            near_wall = False
            for ar, ac in set_a:
                for br, bc in set_b:
                    dist = abs(ar - br) + abs(ac - bc)
                    if dist == 1:
                        direct_touch = True
                        break
                    if (ar == br and abs(ac - bc) == 2) or (ac == bc and abs(ar - br) == 2):
                        near_wall = True
                if direct_touch:
                    break
            if direct_touch or not near_wall:
                continue
            norm_groups, shape = _normalize_coord_groups([coords_a, shifted_b])
            height, width = shape
            area = height * width
            candidates.append((area, height, width, len(candidates), norm_groups[0], norm_groups[1]))
    if not candidates:
        return None
    best_area = min(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_area]
    _area, _height, _width, _idx, group_a, group_b = random.choice(best)
    return [group_a, group_b]


def _mark_forbidden(forbidden: np.ndarray, coords: np.ndarray) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_groups(
    template_h: int, template_w: int, groups: list[list[np.ndarray]]
) -> list[list[np.ndarray]] | None:
    forbidden = np.zeros((template_h, template_w), dtype=bool)
    placed: list[list[np.ndarray]] = []
    random_order = list(range(len(groups)))
    random.shuffle(random_order)
    temp: dict[int, list[np.ndarray]] = {}
    for group_idx in random_order:
        group = groups[group_idx]
        local_groups, shape = _normalize_coord_groups(group)
        gh, gw = shape
        candidates = [
            (top, left)
            for top in range(1, template_h - gh)
            for left in range(1, template_w - gw)
        ]
        random.shuffle(candidates)
        found = None
        for top, left in candidates:
            ok = True
            for coords in local_groups:
                shifted = coords + np.array([top, left], dtype=int)
                for r, c in shifted:
                    if forbidden[int(r), int(c)]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                found = (top, left)
                break
        if found is None:
            return None
        top, left = found
        shifted_group = []
        for coords in local_groups:
            shifted = coords + np.array([top, left], dtype=int)
            shifted_group.append(shifted)
        for coords in shifted_group:
            _mark_forbidden(forbidden, coords)
        temp[group_idx] = shifted_group
    for group_idx in range(len(groups)):
        placed.append(temp[group_idx])
    return placed


def _slot_mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _compose_template(
    template_h: int,
    template_w: int,
    slot_groups: list[list[np.ndarray]],
    slot_colors: list[int],
    template_color: int,
    pair_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    template = np.full((template_h, template_w), int(template_color), dtype=int)
    expected = template.copy()
    slot_index = 0
    for group in slot_groups:
        for coords in group:
            template[coords[:, 0], coords[:, 1]] = BG
            slot_index += 1
    slot_index = 0
    for group in slot_groups:
        for coords in group:
            out_color = slot_colors[pair_map.get(slot_index, slot_index)]
            expected[coords[:, 0], coords[:, 1]] = int(out_color)
            slot_index += 1
    return template, expected


def _blocked_with_padding(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_piece_patches(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    piece_patches: list[np.ndarray],
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    tr0, tc0, tr1, tc1 = template_bbox
    for r in range(max(0, tr0 - 1), min(h, tr1 + 2)):
        for c in range(max(0, tc0 - 1), min(w, tc1 + 2)):
            blocked[r, c] = True

    order = list(range(len(piece_patches)))
    random.shuffle(order)
    for piece_idx in order:
        patch = piece_patches[piece_idx]
        coords = np.argwhere(patch != BG)
        ph, pw = patch.shape
        candidates = [
            (top, left)
            for top in range(h - ph + 1)
            for left in range(w - pw + 1)
        ]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            shifted = coords + np.array([top, left], dtype=int)
            if any(blocked[int(r), int(c)] for r, c in shifted):
                continue
            grid[top : top + ph, left : left + pw][patch != BG] = patch[patch != BG]
            _blocked_with_padding(blocked, shifted)
            placed = True
            break
        if not placed:
            return False
    return True


def _make_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    slot_count = random.randint(*GEN_PIECE_COUNT)
    shapes = _sample_shapes(slot_count)
    if shapes is None:
        return None

    colors = random.sample(GEN_COLORS, slot_count)
    hole_masks = [random.choice(_rotation_variants(shape)).copy() for shape in shapes]
    piece_masks = [random.choice(_rotation_variants(shape)).copy() for shape in shapes]

    max_pairs = slot_count // 2
    pair_count = random.randint(0, max_pairs)
    shuffled_indices = list(range(slot_count))
    random.shuffle(shuffled_indices)
    group_specs: list[dict] = []
    paired_indices: set[int] = set()

    for pair_idx in range(pair_count):
        a = shuffled_indices[2 * pair_idx]
        b = shuffled_indices[2 * pair_idx + 1]
        group = _build_pair_group(hole_masks[a], hole_masks[b])
        if group is None:
            return None
        group_specs.append({"slot_indices": [a, b], "coords_list": group})
        paired_indices.add(a)
        paired_indices.add(b)

    for idx in range(slot_count):
        if idx in paired_indices:
            continue
        group_specs.append({"slot_indices": [idx], "coords_list": [_coords_from_mask(hole_masks[idx])]})

    template_h = random.randint(*GEN_TEMPLATE_SIDE)
    template_w = random.randint(*GEN_TEMPLATE_SIDE)
    slot_groups = _place_groups(template_h, template_w, [spec["coords_list"] for spec in group_specs])
    if slot_groups is None:
        return None

    pair_map: dict[int, int] = {}
    slot_order: list[int] = []
    global_slot = 0
    for spec in group_specs:
        indices = [int(idx) for idx in spec["slot_indices"]]
        if len(indices) == 2:
            pair_map[global_slot] = global_slot + 1
            pair_map[global_slot + 1] = global_slot
        slot_order.extend(indices)
        global_slot += len(indices)

    slot_colors = [colors[idx] for idx in slot_order]
    piece_patches = [np.where(piece_masks[idx], colors[idx], BG).astype(int) for idx in slot_order]

    template, expected = _compose_template(
        template_h=template_h,
        template_w=template_w,
        slot_groups=slot_groups,
        slot_colors=slot_colors,
        template_color=GEN_TEMPLATE_COLOR,
        pair_map=pair_map,
    )

    margin_h = random.randint(*GEN_GRID_MARGIN)
    margin_w = random.randint(*GEN_GRID_MARGIN)
    grid_h = template_h + margin_h
    grid_w = template_w + margin_w
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    top = random.randint(2, grid_h - template_h - 2)
    left = random.randint(2, grid_w - template_w - 2)
    grid[top : top + template_h, left : left + template_w] = template

    if not _place_piece_patches(grid, (top, left, top + template_h - 1, left + template_w - 1), piece_patches):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        attempt = _make_attempt()
        if attempt is not None:
            grid, _expected = attempt
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
