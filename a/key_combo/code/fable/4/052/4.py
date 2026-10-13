from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Rect = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[Coord]) -> Rect:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _detect_structure(grid: np.ndarray) -> tuple[int, int, list[dict], list[list[int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    candidates = []

    for corridor in colors:
        room_mask = (g != bg) & (g != corridor)
        room_comps = _components(room_mask)
        if len(room_comps) < 2:
            continue

        rooms = []
        room_shapes = set()
        room_bases = set()
        ok = True

        for comp in room_comps:
            r0, c0, r1, c1 = _bbox(comp)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if patch.size != len(comp):
                ok = False
                break
            if np.any(patch == bg) or np.any(patch == corridor):
                ok = False
                break

            counts = Counter(map(int, patch.ravel()))
            if len(counts) < 2:
                ok = False
                break
            ranked = counts.most_common()
            base = int(ranked[0][0])
            if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
                ok = False
                break

            rooms.append({"bbox": (r0, c0, r1, c1), "base": base, "shape": patch.shape})
            room_shapes.add(patch.shape)
            room_bases.add(base)

        if not ok or len(room_shapes) != 1 or len(room_bases) != 1:
            continue

        room_index = np.full(g.shape, -1, dtype=int)
        for idx, room in enumerate(rooms):
            r0, c0, r1, c1 = room["bbox"]
            room_index[r0 : r1 + 1, c0 : c1 + 1] = idx

        corridor_comps = _components(g == corridor)
        if not corridor_comps:
            continue

        touched_by_corridor: list[list[int]] = []
        corridor_area = 0
        for comp in corridor_comps:
            r0, c0, r1, c1 = _bbox(comp)
            if r0 != r1 and c0 != c1:
                ok = False
                break
            if (r1 - r0 + 1) * (c1 - c0 + 1) != len(comp):
                ok = False
                break
            touched = set()
            for r, c in comp:
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(g.shape, nr, nc) and room_index[nr, nc] != -1:
                        touched.add(int(room_index[nr, nc]))
            if len(touched) < 2:
                ok = False
                break
            touched_by_corridor.append(sorted(touched))
            corridor_area += len(comp)

        if not ok:
            continue

        score = (
            len(rooms),
            len({idx for group in touched_by_corridor for idx in group}),
            corridor_area,
        )
        candidates.append((score, corridor, bg, rooms, touched_by_corridor))

    assert candidates, "failed to identify rooms and corridor color"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, corridor, bg, rooms, touched_by_corridor = candidates[0]
    return int(bg), int(corridor), rooms, touched_by_corridor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge marker layers only among rooms connected through corridor components.

    Input structure:
    - A black background contains several same-sized filled rectangular rooms with a shared base color.
    - Each room contains one or more non-base marker pixels, and some room pairs are linked by a dedicated 1-cell-wide straight corridor color.
    - Corridors touch room walls from outside, while isolated rooms have no corridor touching them.

    Rule:
    - Detect the repeated room rectangles and the dedicated corridor color.
    - Use corridor connectivity to partition rooms into connected groups, then take the union of marker pixels by room-relative coordinates within each group.
    - Paint that merged marker layer back into every room of the same corridor-connected group, while preserving isolated rooms and leaving corridors unchanged.

    Key insight:
    - The repeated room rectangles are overlay layers, but the overlay scope is restricted to each corridor-connected component rather than to all rooms globally.
    """
    g = np.asarray(grid, dtype=int)
    bg, _corridor, rooms, touched_by_corridor = _detect_structure(g)
    del bg

    dsu = _DSU(len(rooms))
    for touched in touched_by_corridor:
        root = touched[0]
        for other in touched[1:]:
            dsu.union(root, other)

    groups: dict[int, list[int]] = {}
    for idx in range(len(rooms)):
        groups.setdefault(dsu.find(idx), []).append(idx)

    base = int(rooms[0]["base"])
    h, w = rooms[0]["shape"]
    out = g.copy()

    for group in groups.values():
        merged = np.full((h, w), base, dtype=int)
        for room_idx in group:
            r0, c0, r1, c1 = rooms[room_idx]["bbox"]
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            for rr, cc in np.argwhere(patch != base):
                rr = int(rr)
                cc = int(cc)
                value = int(patch[rr, cc])
                assert int(merged[rr, cc]) in {base, value}
                merged[rr, cc] = value
        for room_idx in group:
            r0, c0, r1, c1 = rooms[room_idx]["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = merged

    return out


def _interior_cells(h: int, w: int) -> list[Coord]:
    rows = range(1, h - 1) if h >= 3 else range(h)
    cols = range(1, w - 1) if w >= 3 else range(w)
    cells = [(r, c) for r in rows for c in cols]
    return cells if cells else [(r, c) for r in range(h) for c in range(w)]


def _make_connected_room_patches(count: int, h: int, w: int, base: int, marker_colors: list[int]) -> list[np.ndarray]:
    cells = _interior_cells(h, w)
    if len(cells) < count + 1:
        cells = [(r, c) for r in range(h) for c in range(w)]
    max_union = min(len(cells), max(count + 1, 6))
    union_size = random.randint(count + 1, max_union)
    chosen = random.sample(cells, union_size)
    colors = [random.choice(marker_colors) for _ in range(union_size)]

    owned_by = [set() for _ in range(count)]
    order = list(range(union_size))
    random.shuffle(order)
    for room_idx, pos_idx in enumerate(order[:count]):
        owned_by[room_idx].add(pos_idx)
    for pos_idx in order[count:]:
        owner = random.randrange(count)
        owned_by[owner].add(pos_idx)
        if count >= 3 and random.random() < 0.3:
            extra_choices = [i for i in range(count) if i != owner]
            owned_by[random.choice(extra_choices)].add(pos_idx)

    patches = [np.full((h, w), base, dtype=int) for _ in range(count)]
    for room_idx, indices in enumerate(owned_by):
        for pos_idx in indices:
            rr, cc = chosen[pos_idx]
            patches[room_idx][rr, cc] = colors[pos_idx]

    union_patch = np.full((h, w), base, dtype=int)
    for pos_idx, (rr, cc) in enumerate(chosen):
        union_patch[rr, cc] = colors[pos_idx]

    assert any(not np.array_equal(patch, union_patch) for patch in patches)
    assert all(np.any(patch != base) for patch in patches)
    return patches


def _make_isolated_patch(h: int, w: int, base: int, marker_colors: list[int]) -> np.ndarray:
    patch = np.full((h, w), base, dtype=int)
    cells = _interior_cells(h, w)
    count = random.randint(1, min(3, len(cells)))
    for rr, cc in random.sample(cells, count):
        patch[rr, cc] = random.choice(marker_colors)
    return patch


def _rect_clear(rect: Rect, others: list[Rect], margin: int = 1) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (r1 + margin < or0 or or1 + margin < r0 or c1 + margin < oc0 or oc1 + margin < c0):
            return False
    return True


def _room_id_at(rooms: list[Rect], r: int, c: int) -> int:
    for idx, (r0, c0, r1, c1) in enumerate(rooms):
        if r0 <= r <= r1 and c0 <= c <= c1:
            return idx
    return -1


def _build_group_layout(count: int, h: int, w: int) -> tuple[list[Rect], list[list[Coord]]]:
    if count == 1:
        return [(0, 0, h - 1, w - 1)], []

    for _ in range(300):
        rooms: list[Rect] = [(0, 0, h - 1, w - 1)]
        corridors: list[list[Coord]] = []
        used_corridor = set()
        ok = True

        for _room_idx in range(1, count):
            placed = False
            parent_order = list(range(len(rooms)))
            random.shuffle(parent_order)
            for parent_idx in parent_order:
                pr0, pc0, pr1, pc1 = rooms[parent_idx]
                options = []
                row_deltas = list(range(-(h - 2), h - 1))
                col_deltas = list(range(-(w - 2), w - 1))
                for gap in random.sample(list(range(2, 7)), 5):
                    random.shuffle(row_deltas)
                    for delta in row_deltas:
                        options.append(("right", gap, delta))
                        options.append(("left", gap, delta))
                    random.shuffle(col_deltas)
                    for delta in col_deltas:
                        options.append(("down", gap, delta))
                        options.append(("up", gap, delta))
                random.shuffle(options)

                for direction, gap, delta in options:
                    if direction == "right":
                        rr0 = pr0 + delta
                        cc0 = pc1 + gap + 1
                        rect = (rr0, cc0, rr0 + h - 1, cc0 + w - 1)
                        overlap = range(max(pr0, rect[0]), min(pr1, rect[2]) + 1)
                        preferred = [r for r in overlap if pr0 < r < pr1 and rect[0] < r < rect[2]]
                        rows = preferred if preferred else list(overlap)
                        if not rows:
                            continue
                        row = random.choice(rows)
                        path = [(row, c) for c in range(pc1 + 1, cc0)]
                    elif direction == "left":
                        rr0 = pr0 + delta
                        cc1 = pc0 - gap - 1
                        rect = (rr0, cc1 - w + 1, rr0 + h - 1, cc1)
                        overlap = range(max(pr0, rect[0]), min(pr1, rect[2]) + 1)
                        preferred = [r for r in overlap if pr0 < r < pr1 and rect[0] < r < rect[2]]
                        rows = preferred if preferred else list(overlap)
                        if not rows:
                            continue
                        row = random.choice(rows)
                        path = [(row, c) for c in range(rect[3] + 1, pc0)]
                    elif direction == "down":
                        rr0 = pr1 + gap + 1
                        cc0 = pc0 + delta
                        rect = (rr0, cc0, rr0 + h - 1, cc0 + w - 1)
                        overlap = range(max(pc0, rect[1]), min(pc1, rect[3]) + 1)
                        preferred = [c for c in overlap if pc0 < c < pc1 and rect[1] < c < rect[3]]
                        cols = preferred if preferred else list(overlap)
                        if not cols:
                            continue
                        col = random.choice(cols)
                        path = [(r, col) for r in range(pr1 + 1, rr0)]
                    else:
                        rr1 = pr0 - gap - 1
                        cc0 = pc0 + delta
                        rect = (rr1 - h + 1, cc0, rr1, cc0 + w - 1)
                        overlap = range(max(pc0, rect[1]), min(pc1, rect[3]) + 1)
                        preferred = [c for c in overlap if pc0 < c < pc1 and rect[1] < c < rect[3]]
                        cols = preferred if preferred else list(overlap)
                        if not cols:
                            continue
                        col = random.choice(cols)
                        path = [(r, col) for r in range(rect[2] + 1, pr0)]

                    if not path:
                        continue
                    if not _rect_clear(rect, rooms, margin=1):
                        continue

                    path_set = set(path)
                    if path_set & used_corridor:
                        continue

                    collision = False
                    for r, c in path:
                        if _room_id_at(rooms, r, c) != -1:
                            collision = True
                            break
                        for nr, nc in _neighbors4(r, c):
                            room_id = _room_id_at(rooms, nr, nc)
                            if room_id != -1 and room_id != parent_idx:
                                collision = True
                                break
                            if (nr, nc) in used_corridor:
                                collision = True
                                break
                        if collision:
                            break
                    if collision:
                        continue

                    rooms.append(rect)
                    corridors.append(path)
                    used_corridor.update(path)
                    placed = True
                    break

                if placed:
                    break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        all_coords = []
        for r0, c0, r1, c1 in rooms:
            all_coords.extend([(r0, c0), (r1, c1)])
        for path in corridors:
            all_coords.extend(path)
        min_r, min_c, max_r, max_c = _bbox(all_coords)
        if max_r - min_r + 1 <= 18 and max_c - min_c + 1 <= 18:
            return rooms, corridors

    raise RuntimeError("failed to build room group")


def _sprite_from_group(
    room_patches: list[np.ndarray],
    corridors: list[list[Coord]],
    rooms: list[Rect],
    corridor_color: int,
) -> np.ndarray:
    all_coords = []
    for r0, c0, r1, c1 in rooms:
        all_coords.extend([(r0, c0), (r1, c1)])
    for path in corridors:
        all_coords.extend(path)
    min_r, min_c, max_r, max_c = _bbox(all_coords)

    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for patch, (r0, c0, r1, c1) in zip(room_patches, rooms):
        sr0 = r0 - min_r
        sc0 = c0 - min_c
        sprite[sr0 : sr0 + patch.shape[0], sc0 : sc0 + patch.shape[1]] = patch
    for path in corridors:
        for r, c in path:
            sprite[r - min_r, c - min_c] = corridor_color
    return sprite


def _can_place(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int, padding: int) -> bool:
    h, w = canvas.shape
    sh, sw = sprite.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    cells = np.argwhere(sprite != 0)
    for rr, cc in cells:
        r = top + int(rr)
        c = left + int(cc)
        if canvas[r, c] != 0:
            return False
        for nr in range(max(0, r - padding), min(h, r + padding + 1)):
            for nc in range(max(0, c - padding), min(w, c + padding + 1)):
                if canvas[nr, nc] != 0:
                    return False
    return True


def _place_sprite(canvas: np.ndarray, sprite: np.ndarray, border: int = 1, padding: int = 2) -> bool:
    h, w = canvas.shape
    sh, sw = sprite.shape
    rows = list(range(border, h - sh - border + 1))
    cols = list(range(border, w - sw - border + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for top in rows:
        for left in cols:
            if _can_place(canvas, sprite, top, left, padding):
                mask = sprite != 0
                canvas[top : top + sh, left : left + sw][mask] = sprite[mask]
                return True
    return False


def _choose_group_sizes(room_count: int) -> list[int]:
    mode = random.choice(["all", "one_isolate", "two_groups", "pair_plus_isolates"])
    if room_count == 3:
        mode = random.choice(["all", "one_isolate"])

    if mode == "all":
        return [room_count]
    if mode == "one_isolate":
        sizes = [room_count - 1, 1]
    elif mode == "two_groups":
        first = random.randint(2, room_count - 2)
        second = room_count - first
        if second == 1:
            sizes = [first - 1, 2, 1]
        else:
            sizes = [first, second]
    else:
        sizes = [2] + [1] * (room_count - 2)

    random.shuffle(sizes)
    return sizes


def _gen_attempt() -> np.ndarray | None:
    room_h = random.randint(4, 6)
    room_w = random.randint(4, 7)
    room_count = random.randint(3, 6)
    group_sizes = _choose_group_sizes(room_count)

    palette = list(range(1, 10))
    random.shuffle(palette)
    base = palette.pop()
    corridor_color = palette.pop()
    marker_pool = palette[: random.randint(2, 4)]

    sprites = []
    has_isolated = any(size == 1 for size in group_sizes)
    for size in group_sizes:
        if size == 1:
            patch = _make_isolated_patch(room_h, room_w, base, marker_pool)
            sprites.append(patch.copy())
            continue

        try:
            rooms, corridors = _build_group_layout(size, room_h, room_w)
        except RuntimeError:
            return None
        patches = _make_connected_room_patches(size, room_h, room_w, base, marker_pool)
        sprites.append(_sprite_from_group(patches, corridors, rooms, corridor_color))

    min_h = max(max(sprite.shape[0] for sprite in sprites) + 2, 16)
    min_w = max(max(sprite.shape[1] for sprite in sprites) + 2, 16)
    if has_isolated:
        min_h = max(min_h, 18)
        min_w = max(min_w, 18)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    random.shuffle(sprites)

    for sprite in sprites:
        if not _place_sprite(grid, sprite):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-room overlay input")
