from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 5
NOISE = 4
PAYLOAD_COLORS = (1, 2, 3, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale2(tile: np.ndarray) -> np.ndarray:
    return np.kron(tile, np.ones((2, 2), dtype=int))


def _component_infos(grid: np.ndarray, mask: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for coords in _components(mask):
        box = _bbox(coords)
        infos.append(
            {
                "coords": coords,
                "coord_set": {tuple(map(int, rc)) for rc in coords},
                "bbox": box,
                "area": int(len(coords)),
                "crop": _crop(grid, box),
            }
        )
    return infos


def _clean_noise(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    cleaned = np.array(grid, dtype=int, copy=True)
    object_infos = _component_infos(cleaned, (cleaned != BG) & (cleaned != NOISE))
    touch_records: list[dict] = []

    for yellow_coords in _components(cleaned == NOISE):
        best_idx = None
        best_contacts: set[tuple[int, int]] = set()
        best_count = -1
        for idx, obj in enumerate(object_infos):
            contacts: set[tuple[int, int]] = set()
            for r, c in yellow_coords:
                for dr, dc in NEIGHBORS4:
                    nb = (int(r + dr), int(c + dc))
                    if nb in obj["coord_set"]:
                        contacts.add(nb)
            if len(contacts) > best_count:
                best_count = len(contacts)
                best_contacts = contacts
                best_idx = idx

        if best_count >= 2:
            colors = [int(cleaned[r, c]) for r, c in best_contacts]
            color_counts = Counter(colors).most_common()
            assert len(color_counts) == 1 or color_counts[0][1] > color_counts[1][1]
            fill_color = int(color_counts[0][0])
            for r, c in yellow_coords:
                cleaned[r, c] = fill_color
            touch_records.append(
                {
                    "coords": yellow_coords,
                    "decision": "merge",
                    "object_index": int(best_idx),
                    "contact_count": int(best_count),
                    "fill_color": fill_color,
                }
            )
        else:
            for r, c in yellow_coords:
                cleaned[r, c] = BG
            touch_records.append(
                {
                    "coords": yellow_coords,
                    "decision": "delete",
                    "object_index": None,
                    "contact_count": int(best_count),
                    "fill_color": None,
                }
            )

    return cleaned, touch_records


def _analyze(grid: np.ndarray) -> dict:
    cleaned, noise_actions = _clean_noise(grid)
    infos = _component_infos(cleaned, cleaned != BG)
    assert len(infos) >= 4, "expected one template plus 3-4 large objects"

    infos.sort(key=lambda info: (info["area"], info["bbox"][0], info["bbox"][1]))
    template = infos[0]
    large_objects = infos[1:]
    assert 3 <= len(large_objects) <= 4
    assert template["area"] < min(obj["area"] for obj in large_objects)

    template_crop = template["crop"]
    variants = {k: np.rot90(_scale2(template_crop), k) for k in range(4)}
    signatures = {(arr.shape, arr.tobytes()) for arr in variants.values()}
    assert len(signatures) == 4, "template rotations must be uniquely identifiable"

    matches: list[dict] = []
    for obj in large_objects:
        matched = [k for k, variant in variants.items() if obj["crop"].shape == variant.shape and np.array_equal(obj["crop"], variant)]
        assert len(matched) == 1, "every large object must match exactly one rotation"
        matches.append({"rotation": int(matched[0]), "bbox": obj["bbox"], "crop": obj["crop"]})

    seed_matches = [m for m in matches if m["rotation"] == 0]
    assert len(seed_matches) == 1, "there must be exactly one unrotated large object"

    return {
        "cleaned": cleaned,
        "noise_actions": noise_actions,
        "template": template_crop,
        "variants": variants,
        "matches": matches,
        "seed": seed_matches[0]["crop"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean yellow noise, template-match scaled rotated copies, and crop the unrotated seed object.

    Input structure:
    - The grid has grey background.
    - A small multicolor template sits near the upper-left corner.
    - Elsewhere are 3-4 larger objects that are 2x scaled rotated copies of the template, plus yellow noise clusters attached to their edges and a few isolated yellow noise clusters.

    Rule:
    - For each yellow 4-connected cluster, count its 4-neighbor contact pixels with each non-grey non-yellow object.
    - If the best contact count is at least 2, recolor the whole yellow cluster with the dominant contacted object color; otherwise erase it to grey.
    - Crop every cleaned large object, compare it against the 2x-scaled template under the four rotations, and select the unique object that matches rotation 0.
    - Output exactly that object's bounding-box crop.

    Key insight:
    - Noise cleaning is required before exact template matching, and the match orientation reveals which large copy is the seed to crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _analyze(g)["seed"].copy()


def _rotation_signatures(tile: np.ndarray) -> int:
    return len({(np.rot90(tile, k).shape, np.rot90(tile, k).tobytes()) for k in range(4)})


def _make_template() -> np.ndarray | None:
    h = random.randint(3, 4)
    w = random.randint(3, 4)
    n_colors = random.choice((2, 3))
    colors = random.sample(PAYLOAD_COLORS, n_colors)
    tile = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            tile[r, c] = random.choice(colors)

    for color in colors:
        if not np.any(tile == color):
            return None
    if _rotation_signatures(tile) != 4:
        return None
    return tile


def _free_positions(occupied: np.ndarray, shape: tuple[int, int], padding: int, *, r_min: int = 0, c_min: int = 0) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = shape
    positions: list[tuple[int, int]] = []
    for r in range(r_min, h - ph + 1):
        for c in range(c_min, w - pw + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(h, r + ph + padding)
            cc1 = min(w, c + pw + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    return positions


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: tuple[int, int]) -> None:
    r, c = top
    h, w = patch.shape
    grid[r : r + h, c : c + w] = patch


def _boundary_blocks(patch: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = patch.shape
    blocks: list[list[tuple[int, int]]] = []
    for r0 in range(0, h, 2):
        for c0 in range(0, w, 2):
            if r0 in (0, h - 2) or c0 in (0, w - 2):
                blocks.append([(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)])
    return blocks


def _connected_subset_from_block(block: list[tuple[int, int]], size: int) -> list[tuple[int, int]] | None:
    for start in random.sample(block, len(block)):
        q = deque([start])
        seen = {start}
        ordered: list[tuple[int, int]] = []
        while q and len(ordered) < size:
            cell = q.popleft()
            ordered.append(cell)
            for dr, dc in NEIGHBORS4:
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt in block and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if len(ordered) == size:
            return ordered
    return None


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = set()
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen.add(start)
    target = {tuple(map(int, rc)) for rc in coords}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in target and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(target)


def _choose_noise_clusters(patch: np.ndarray) -> list[list[tuple[int, int]]]:
    total_noise = random.randint(2, 5)
    if total_noise == 5 or (total_noise >= 4 and random.random() < 0.45):
        first = random.randint(1, total_noise - 1)
        sizes = [first, total_noise - first]
    else:
        sizes = [total_noise]

    used: set[tuple[int, int]] = set()
    clusters: list[list[tuple[int, int]]] = []
    blocks = _boundary_blocks(patch)

    for size in sizes:
        placed = False
        block_order = blocks[:]
        random.shuffle(block_order)
        for block in block_order:
            if size > 4:
                continue
            free_block = [cell for cell in block if cell not in used]
            if len(free_block) < size:
                continue
            cluster = _connected_subset_from_block(free_block, size)
            if cluster is None:
                continue
            if any(any((r + dr, c + dc) in used for dr, dc in NEIGHBORS4) for r, c in cluster):
                continue
            remain = np.ones_like(patch, dtype=bool)
            for r, c in cluster:
                remain[r, c] = False
            if not _mask_connected(remain):
                continue
            used.update(cluster)
            clusters.append(cluster)
            placed = True
            break
        if not placed:
            return []
    return clusters


def _place_isolated_noise(grid: np.ndarray, occupied: np.ndarray, count: int) -> bool:
    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
    ]
    h, w = grid.shape
    for _ in range(count):
        placed = False
        shape_order = shapes[:]
        random.shuffle(shape_order)
        for cells in shape_order:
            sh = 1 + max(r for r, _ in cells)
            sw = 1 + max(c for _, c in cells)
            candidates: list[tuple[int, int]] = []
            for r0 in range(h - sh + 1):
                for c0 in range(w - sw + 1):
                    ok = True
                    for dr, dc in cells:
                        rr, cc = r0 + dr, c0 + dc
                        if occupied[rr, cc]:
                            ok = False
                            break
                        for nr in range(max(0, rr - 1), min(h, rr + 2)):
                            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                                if occupied[nr, nc]:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if ok:
                        candidates.append((r0, c0))
            if not candidates:
                continue
            r0, c0 = random.choice(candidates)
            for dr, dc in cells:
                rr, cc = r0 + dr, c0 + dc
                grid[rr, cc] = NOISE
                occupied[rr, cc] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    template = _make_template()
    if template is None:
        return None

    object_rotations = [0]
    other_rots = [1, 2, 3]
    random.shuffle(other_rots)
    object_rotations.extend(other_rots[: random.choice((2, 3))])
    random.shuffle(object_rotations)
    assert object_rotations.count(0) == 1

    big_patches = [np.rot90(_scale2(template), k) for k in object_rotations]
    max_h = max(p.shape[0] for p in big_patches)
    max_w = max(p.shape[1] for p in big_patches)
    min_side = max(20, 6 + max_h * 2, 6 + max_w * 2)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    tr = random.randint(0, 1)
    tc = random.randint(0, 1)
    _paint_patch(grid, template, (tr, tc))
    occupied[max(0, tr - 1) : min(h, tr + template.shape[0] + 1), max(0, tc - 1) : min(w, tc + template.shape[1] + 1)] = True

    placed_pairs: list[tuple[np.ndarray, tuple[int, int]]] = []
    for patch in big_patches:
        positions = _free_positions(occupied, patch.shape, padding=1, r_min=template.shape[0] + 2, c_min=0)
        if not positions:
            positions = _free_positions(occupied, patch.shape, padding=1, r_min=0, c_min=template.shape[1] + 2)
        if not positions:
            return None
        top = random.choice(positions)
        placed_pairs.append((patch, top))
        _paint_patch(grid, patch, top)
        r, c = top
        ph, pw = patch.shape
        occupied[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)] = True

    for patch, top in placed_pairs:
        clusters = _choose_noise_clusters(patch)
        if not clusters:
            return None
        r0, c0 = top
        for cluster in clusters:
            for dr, dc in cluster:
                grid[r0 + dr, c0 + dc] = NOISE

    occ_for_isolated = grid != BG
    if not _place_isolated_noise(grid, occ_for_isolated, random.randint(1, 2)):
        return None

    try:
        analysis = _analyze(grid)
    except AssertionError:
        return None

    if len(analysis["noise_actions"]) < len(big_patches) + 1:
        return None
    if not any(action["decision"] == "merge" for action in analysis["noise_actions"]):
        return None
    if not any(action["decision"] == "delete" for action in analysis["noise_actions"]):
        return None
    if not np.array_equal(analysis["seed"], _scale2(template)):
        return None
    if max(grid.shape) > 30 or max(analysis["seed"].shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
