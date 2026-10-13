from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
YELLOW = 4
MARKER_POOL = (1, 2, 3, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _has_internal_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool((bg & ~seen).any())


def _fill_enclosed(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.where(mask, GREY, BLACK)
    external = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not external[r, c]:
                external[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not external[r, c]:
                external[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not external[nr, nc]:
                external[nr, nc] = True
                q.append((nr, nc))
    out[(~mask) & (~external)] = YELLOW
    return out


def _marker_exposed_dir(obj_mask: np.ndarray, r: int, c: int) -> tuple[int, int]:
    exposed: list[tuple[int, int]] = []
    h, w = obj_mask.shape
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or not obj_mask[nr, nc]:
            exposed.append((dr, dc))
    assert len(exposed) == 1, "marker must have exactly one exposed side"
    return exposed[0]


def _object_records(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for obj_id, coords in enumerate(_components(grid != BLACK)):
        box = _bbox(coords)
        sprite = _crop(grid, box)
        mask = sprite != BLACK
        markers: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for rr, cc in np.argwhere((sprite != BLACK) & (sprite != GREY)):
            r, c = int(rr), int(cc)
            color = int(sprite[r, c])
            markers[color] = ((r, c), _marker_exposed_dir(mask, r, c))
        objects.append(
            {
                "id": obj_id,
                "bbox": box,
                "sprite": sprite,
                "mask": mask,
                "markers": markers,
            }
        )
    return objects


def _assemble_grey(grid: np.ndarray, ignore_colors: set[int] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    ignore = set() if ignore_colors is None else {int(x) for x in ignore_colors}
    objects = _object_records(g)
    assert 3 <= len(objects) <= 4

    occurrences: dict[int, list[dict]] = {}
    color_order: list[int] = []
    seen_colors: set[int] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BLACK, GREY):
                continue
            if color not in seen_colors:
                color_order.append(color)
                seen_colors.add(color)

    for obj in objects:
        for color, (local_pos, direction) in obj["markers"].items():
            lr, lc = local_pos
            abs_pos = (obj["bbox"][0] + lr, obj["bbox"][1] + lc)
            occurrences.setdefault(color, []).append(
                {"obj": obj["id"], "local": local_pos, "dir": direction, "abs": abs_pos}
            )

    assert all(len(v) == 2 for v in occurrences.values())
    assert all(1 <= len(obj["markers"]) <= 2 for obj in objects)

    offsets: dict[int, tuple[int, int]] = {
        obj["id"]: (obj["bbox"][0], obj["bbox"][1]) for obj in objects
    }
    cluster_of = {obj["id"]: obj["id"] for obj in objects}
    clusters = {obj["id"]: {obj["id"]} for obj in objects}

    for color in color_order:
        if color in ignore:
            continue
        a, b = sorted(occurrences[color], key=lambda x: (x["abs"][0], x["abs"][1]))
        assert b["dir"] == (-a["dir"][0], -a["dir"][1])
        a_obj, b_obj = int(a["obj"]), int(b["obj"])
        a_off = offsets[a_obj]
        b_off = offsets[b_obj]
        a_pos = (a_off[0] + a["local"][0], a_off[1] + a["local"][1])
        b_pos = (b_off[0] + b["local"][0], b_off[1] + b["local"][1])
        target_b = (a_pos[0] + a["dir"][0], a_pos[1] + a["dir"][1])

        ca = cluster_of[a_obj]
        cb = cluster_of[b_obj]
        if ca == cb:
            assert b_pos == target_b
            continue

        dr = target_b[0] - b_pos[0]
        dc = target_b[1] - b_pos[1]
        moved = list(clusters[cb])
        for obj_id in moved:
            ro, co = offsets[obj_id]
            offsets[obj_id] = (ro + dr, co + dc)
            cluster_of[obj_id] = ca
        clusters[ca].update(moved)
        del clusters[cb]

    assembled = np.zeros_like(g)
    for obj in objects:
        r0, c0 = offsets[obj["id"]]
        for rr, cc in np.argwhere(obj["mask"]):
            tr = r0 + int(rr)
            tc = c0 + int(cc)
            assert 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1]
            assembled[tr, tc] = GREY
    return assembled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock equal-colored marker-bearing gray objects, then fill the newly enclosed black hole(s) yellow.

    Input structure:
    - The grid is black background with 3-4 disconnected gray objects.
    - Each object is a single 4-connected component and carries 1-2 single-cell colored markers.
    - Every marker color appears exactly twice globally, and each marker lies on an object boundary with exactly one exposed side.

    Rule:
    - Read marker colors by first occurrence in scan order, top-to-bottom then left-to-right.
    - For each color pair, keep the cluster containing the earlier marker fixed and translate the other cluster so the two markers become 4-neighbors with opposite exposed sides facing each other.
    - After all dockings, convert every marker cell to gray, take the union of all placed object pixels, and keep the original canvas size.
    - Fill every black 4-connected region that is no longer connected to the outside of the grid with yellow.

    Key insight:
    - The colored markers tell both which pieces must dock and which direction the translation must use; only after the docking translations does a new enclosed hole emerge to be filled.
    """
    assembled = _assemble_grey(grid)
    return _fill_enclosed(assembled == GREY)


def _frame_mask(hole_h: int, hole_w: int) -> np.ndarray:
    h = hole_h + 6
    w = hole_w + 6
    mask = np.ones((h, w), dtype=bool)
    mask[3 : 3 + hole_h, 3 : 3 + hole_w] = False
    return mask


def _slit_cells(kind: str, pos: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if kind == "T":
        return [(r, pos) for r in range(3)]
    if kind == "B":
        return [(h - 1 - r, pos) for r in range(3)]
    if kind == "L":
        return [(pos, c) for c in range(3)]
    return [(pos, w - 1 - c) for c in range(3)]


def _marker_pair(kind: str, pos: int, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    if kind == "T":
        return (1, pos - 1), (1, pos + 1)
    if kind == "B":
        return (h - 2, pos - 1), (h - 2, pos + 1)
    if kind == "L":
        return (pos - 1, 1), (pos + 1, 1)
    return (pos - 1, w - 2), (pos + 1, w - 2)


def _hole_cells(hole_h: int, hole_w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(3, 3 + hole_h) for c in range(3, 3 + hole_w)}


def _render_components(shape: tuple[int, int], components: list[set[tuple[int, int]]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for comp in components:
        for r, c in comp:
            out[r, c] = GREY
    return out


def _remove_cell_if_valid(
    components: list[set[tuple[int, int]]],
    comp_index: int,
    cell: tuple[int, int],
) -> bool:
    comp = components[comp_index]
    if cell not in comp:
        return False
    new_comp = set(comp)
    new_comp.remove(cell)
    if not new_comp:
        return False

    shape_h = max(r for cells in components for r, _ in cells) + 1
    shape_w = max(c for cells in components for _, c in cells) + 1
    comp_mask = np.zeros((shape_h, shape_w), dtype=bool)
    for r, c in new_comp:
        comp_mask[r, c] = True
    if not _mask_connected(comp_mask):
        return False

    if _has_internal_hole(comp_mask):
        return False

    components[comp_index] = new_comp
    return True


def _trim_component(
    components: list[set[tuple[int, int]]],
    comp_index: int,
    target_size: int,
    protected: set[tuple[int, int]],
    hole: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    comp = components[comp_index]
    while len(comp) > target_size:
        candidates: list[tuple[int, tuple[int, int]]] = []
        for cell in comp:
            if cell in protected:
                continue
            r, c = cell
            if any(abs(r - hr) + abs(c - hc) == 1 for hr, hc in hole):
                continue
            deg = sum((r + dr, c + dc) in comp for dr, dc in DIR4)
            if deg > 2:
                continue
            score = deg + rng.randint(0, 2)
            candidates.append((score, cell))
        candidates.sort()
        removed = False
        for _score, cell in candidates:
            if _remove_cell_if_valid(components, comp_index, cell):
                comp = components[comp_index]
                removed = True
                break
        if not removed:
            return False
    return True


def _optional_trim(
    components: list[set[tuple[int, int]]],
    protected: set[tuple[int, int]],
    hole: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    for idx in range(len(components)):
        extra = rng.randint(0, 2)
        for _ in range(extra):
            if len(components[idx]) <= 8:
                break
            comp = components[idx]
            candidates = []
            for cell in comp:
                if cell in protected:
                    continue
                r, c = cell
                if any(abs(r - hr) + abs(c - hc) == 1 for hr, hc in hole):
                    continue
                deg = sum((r + dr, c + dc) in comp for dr, dc in DIR4)
                if deg > 2:
                    continue
                candidates.append(cell)
            rng.shuffle(candidates)
            for cell in candidates:
                if _remove_cell_if_valid(components, idx, cell):
                    break
    return True


def _build_local_layout(rng: random.Random) -> np.ndarray | None:
    piece_count = 3 if rng.random() < 0.4 else 4
    if piece_count == 4:
        hole_h, hole_w = rng.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
    else:
        hole_h, hole_w = 2, 2

    base = _frame_mask(hole_h, hole_w)
    shape = base.shape
    hole = _hole_cells(hole_h, hole_w)

    slit_candidates: list[tuple[str, int]] = []
    for c in range(3, 3 + hole_w):
        slit_candidates.append(("T", c))
        slit_candidates.append(("B", c))
    for r in range(3, 3 + hole_h):
        slit_candidates.append(("L", r))
        slit_candidates.append(("R", r))

    combos = []
    k = piece_count
    items = list(slit_candidates)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pass
    import itertools

    for slits in itertools.combinations(items, k):
        mask = base.copy()
        for kind, pos in slits:
            for r, c in _slit_cells(kind, pos, shape):
                mask[r, c] = False
        comps = [set(map(tuple, comp)) for comp in _components(mask)]
        if len(comps) != piece_count:
            continue
        sizes = sorted(len(comp) for comp in comps)
        if sizes[0] < 8:
            continue
        if piece_count == 4 and sizes[-1] > 15:
            continue
        combos.append((slits, comps))
    if not combos:
        return None

    slits, components = rng.choice(combos)
    components = [set(comp) for comp in components]

    marker_face_cells = {cell for slit in slits for cell in _marker_pair(slit[0], slit[1], shape)}
    protected = set(marker_face_cells)
    for kind, pos in slits:
        for cell in _slit_cells(kind, pos, shape):
            for dr, dc in DIR4:
                protected.add((cell[0] + dr, cell[1] + dc))

    if piece_count == 3:
        largest = max(range(len(components)), key=lambda i: len(components[i]))
        target = rng.randint(14, 15)
        if not _trim_component(components, largest, target, protected, hole, rng):
            return None
    else:
        _optional_trim(components, protected, hole, rng)

    sizes = sorted(len(comp) for comp in components)
    if not all(8 <= s <= 15 for s in sizes):
        return None

    for comp in components:
        obj_mask = np.zeros(shape, dtype=bool)
        for r, c in comp:
            obj_mask[r, c] = True
        box = _bbox(np.array(list(comp), dtype=int))
        if _has_internal_hole(_crop(obj_mask.astype(int), box).astype(bool)):
            return None

    marker_slits = list(slits)
    omitted = rng.randrange(len(marker_slits))
    active_slits = [slit for i, slit in enumerate(marker_slits) if i != omitted]
    colors = list(rng.sample(MARKER_POOL, len(active_slits)))

    grid = np.zeros(shape, dtype=int)
    for comp in components:
        for r, c in comp:
            grid[r, c] = GREY

    for color, (kind, pos) in zip(colors, active_slits):
        a, b = _marker_pair(kind, pos, shape)
        owner_a = owner_b = None
        for idx, comp in enumerate(components):
            if a in comp:
                owner_a = idx
            if b in comp:
                owner_b = idx
        if owner_a is None or owner_b is None or owner_a == owner_b:
            return None
        grid[a] = color
        grid[b] = color

    if rng.random() < 0.5:
        grid = np.fliplr(grid)
    if rng.random() < 0.5:
        grid = np.flipud(grid)
    grid = np.rot90(grid, k=rng.randrange(4))
    return grid


def _bbox_gap_ok(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _scatter_sprites(local: np.ndarray, rng: random.Random) -> np.ndarray | None:
    sprites = []
    for comp in _components(local != BLACK):
        box = _bbox(comp)
        sprites.append(_crop(local, box))
    rng.shuffle(sprites)

    h = rng.randint(19, 25)
    w = rng.randint(19, 25)
    canvas = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    for sprite in sprites:
        sh, sw = sprite.shape
        positions = [
            (r, c)
            for r in range(1, h - sh)
            for c in range(1, w - sw)
        ]
        rng.shuffle(positions)
        placed = False
        for r, c in positions[:300]:
            box = (r, c, r + sh - 1, c + sw - 1)
            if any(not _bbox_gap_ok(box, other, 2) for other in placed_boxes):
                continue
            patch_mask = sprite != BLACK
            if np.any((canvas[r : r + sh, c : c + sw] != BLACK) & patch_mask):
                continue
            canvas[r : r + sh, c : c + sw][patch_mask] = sprite[patch_mask]
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return canvas


def _marker_color_order(grid: np.ndarray) -> list[int]:
    order = []
    seen = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or color in seen:
                continue
            seen.add(color)
            order.append(color)
    return order


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    objects = _object_records(inp)
    if not 3 <= len(objects) <= 4:
        return False
    color_counts: dict[int, int] = {}
    for obj in objects:
        area = int(obj["mask"].sum())
        if not 8 <= area <= 15:
            return False
        if not 1 <= len(obj["markers"]) <= 2:
            return False
        if _has_internal_hole(obj["mask"]):
            return False
        for color in obj["markers"]:
            color_counts[color] = color_counts.get(color, 0) + 1
    if not color_counts or any(v != 2 for v in color_counts.values()):
        return False

    shortcut = _fill_enclosed(np.where(inp == BLACK, BLACK, GREY) == GREY)
    if np.array_equal(shortcut, out):
        return False

    if not np.any(out == YELLOW):
        return False
    if np.any((out != BLACK) & (out != GREY) & (out != YELLOW)):
        return False

    order = _marker_color_order(inp)
    if len(order) != len(color_counts):
        return False
    for color in order:
        try:
            weakened = _fill_enclosed(_assemble_grey(inp, {color}) == GREY)
        except Exception:
            continue
        if np.array_equal(weakened, out):
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(600):
        local = _build_local_layout(rng)
        if local is None:
            continue
        inp = _scatter_sprites(local, rng)
        if inp is None:
            continue
        try:
            out = main(inp)
        except Exception:
            continue
        if _validate_semantics(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
