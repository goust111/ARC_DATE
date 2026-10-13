from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ROOM_COUNTS = [2, 3, 4]
ROOM_LAYOUTS = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}
ROOM_SIZE_MIN = 4
ROOM_SIZE_MAX = 6
GRID_PAD_MIN = 9
GRID_PAD_MAX = 13
GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 240
PLACE_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if box is None:
        box = _bbox(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _unique_rotations(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    seen = set()
    out: list[tuple[int, np.ndarray]] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((k, rot))
    return out


def _mask_signature(mask: np.ndarray) -> bytes:
    return mask.tobytes()


def _ordered_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted([mask_a, mask_b], key=_mask_signature)
    return ordered[0], ordered[1]


def _valid_placements(container: np.ndarray, piece: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    ch, cw = map(int, container.shape)
    ph, pw = map(int, piece.shape)
    if ph > ch or pw > cw:
        return []
    placements = []
    for r0 in range(ch - ph + 1):
        for c0 in range(cw - pw + 1):
            sub = container[r0 : r0 + ph, c0 : c0 + pw]
            if not bool(np.all(sub[piece])):
                continue
            placed = np.zeros_like(container, dtype=bool)
            placed[r0 : r0 + ph, c0 : c0 + pw] |= piece
            placements.append((int(r0), int(c0), placed))
    return placements


def _detect_guide_and_cavities(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    bg_comps = _components(grid == int(bg))
    candidates: dict[int, list[dict]] = defaultdict(list)
    for coords in bg_comps:
        if _touches_border(coords, grid.shape):
            continue
        neigh = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    val = int(grid[nr, nc])
                    if val != int(bg):
                        neigh.add(val)
        if len(neigh) != 1:
            continue
        color = int(next(iter(neigh)))
        box = _bbox(coords)
        candidates[color].append({"bbox": box, "mask": _mask_from_coords(coords, box)})

    assert candidates, "failed to identify enclosed cavities"
    guide = max(
        candidates,
        key=lambda color: (len(candidates[color]), int(np.sum(grid == int(color)))),
    )
    cavities = candidates[int(guide)]
    cavities.sort(
        key=lambda item: (
            item["mask"].shape[0] * item["mask"].shape[1],
            item["bbox"][0],
            item["bbox"][1],
        )
    )
    return int(guide), cavities


def _find_fragments(grid: np.ndarray, bg: int, guide: int) -> list[dict]:
    mask = (grid != int(bg)) & (grid != int(guide))
    frags = []
    for coords in _components(mask):
        box = _bbox(coords)
        frag_mask = _mask_from_coords(coords, box)
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "fragment must be monochrome"
        frags.append(
            {
                "bbox": box,
                "mask": frag_mask,
                "color": int(next(iter(colors))),
                "size": int(frag_mask.sum()),
            }
        )
    frags.sort(key=lambda item: (-item["size"], item["bbox"][0], item["bbox"][1]))
    return frags


def _pair_fragments(fragments: list[dict]) -> list[tuple[dict, dict]]:
    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for frag in fragments:
        groups[_rotation_key(frag["mask"])].append(frag)
    pairs = []
    for group in groups.values():
        assert len(group) == 2, "each fragment shape must appear exactly twice"
        group.sort(key=lambda frag: (frag["bbox"][0], frag["bbox"][1], frag["color"]))
        pairs.append((group[0], group[1]))
    pairs.sort(key=lambda pair: -int(pair[0]["mask"].sum()))
    return pairs


def _match_pair_to_cavity(pair: tuple[dict, dict], cavity: dict) -> dict | None:
    cavity_mask = cavity["mask"]
    rotations_a = _unique_rotations(pair[0]["mask"])
    rotations_b = _unique_rotations(pair[1]["mask"])
    placements_cache: dict[tuple[int, int, bytes], list[tuple[int, int, np.ndarray]]] = {}

    def get_placements(mask: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        key = (mask.shape[0], mask.shape[1], mask.tobytes())
        if key not in placements_cache:
            placements_cache[key] = _valid_placements(cavity_mask, mask)
        return placements_cache[key]

    for k_a, rot_a in rotations_a:
        for top_a, left_a, placed_a in get_placements(rot_a):
            for k_b, rot_b in rotations_b:
                for top_b, left_b, placed_b in get_placements(rot_b):
                    if np.array_equal(placed_a | placed_b, cavity_mask):
                        return {
                            "k_a": int(k_a),
                            "top_a": int(top_a),
                            "left_a": int(left_a),
                            "placed_a": placed_a,
                            "k_b": int(k_b),
                            "top_b": int(top_b),
                            "left_b": int(left_b),
                            "placed_b": placed_b,
                        }
    return None


def _solve_assignment(pairs: list[tuple[dict, dict]], cavities: list[dict]) -> dict[int, tuple[int, dict]]:
    matches: dict[int, list[tuple[int, dict]]] = {}
    for pi, pair in enumerate(pairs):
        opts = []
        for ci, cavity in enumerate(cavities):
            sol = _match_pair_to_cavity(pair, cavity)
            if sol is not None:
                opts.append((int(ci), sol))
        matches[pi] = opts

    order = sorted(matches, key=lambda pi: len(matches[pi]))
    assignment: dict[int, tuple[int, dict]] = {}
    used = set()

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        pi = order[idx]
        for ci, sol in matches[pi]:
            if ci in used:
                continue
            assignment[pi] = (ci, sol)
            used.add(ci)
            if dfs(idx + 1):
                return True
            used.remove(ci)
            assignment.pop(pi, None)
        return False

    assert dfs(0), "failed to assign fragment pairs to cavities"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered same-shape fragments and overlay each pair into the matching guided cavity.

    Input structure:
    - The background is the most common color.
    - One guide-colored 1-pixel outline encloses several background cavities, and
      interior guide segments split that outline into distinct cavity shapes.
    - Outside the outline sit monochrome 4-connected fragments; every fragment
      shape appears exactly twice, with the two copies using different colors.

    Rule:
    - Detect the external fragments and pair the two copies of each shape up to
      rotation only.
    - For each pair, find the unique cavity that can be filled by rotating and
      translating the two fragments so their union matches the cavity exactly.
    - Keep the guide structure unchanged, erase the outside fragments, and paint
      each cavity with the matched pair; cells covered by both fragments become
      the guide color, while non-overlapping cells keep the fragment colors.

    Key insight:
    The matching is by shared shape between fragments, but the destination is
    determined by the union of the paired shapes after rotation, not by either
    fragment alone; the overlap then turns into guide color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    guide, cavities = _detect_guide_and_cavities(g, bg)
    fragments = _find_fragments(g, bg, guide)
    pairs = _pair_fragments(fragments)
    assignment = _solve_assignment(pairs, cavities)

    out = np.full_like(g, int(bg))
    out[g == int(guide)] = int(guide)

    for pi, pair in enumerate(pairs):
        ci, sol = assignment[pi]
        cavity = cavities[ci]
        r0, c0, r1, c1 = cavity["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        placed_a, placed_b = _ordered_masks(sol["placed_a"], sol["placed_b"])
        overlap = placed_a & placed_b
        only_a = placed_a & ~placed_b
        only_b = placed_b & ~placed_a
        patch[only_a] = int(pair[0]["color"])
        patch[only_b] = int(pair[1]["color"])
        patch[overlap] = int(guide)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _has_hole(mask: np.ndarray) -> bool:
    for coords in _components(~mask):
        if not _touches_border(coords, mask.shape):
            return True
    return False


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _generate_cover_shape(h: int, w: int) -> np.ndarray | None:
    area = h * w
    for _ in range(SHAPE_ATTEMPTS):
        mask = np.ones((h, w), dtype=bool)
        target = random.randint(max(1, area // 5), max(2, area // 3))
        removed = 0
        steps = 0
        while removed < target and steps < area * 30:
            steps += 1
            candidates = []
            for r in range(h):
                for c in range(w):
                    if not mask[r, c]:
                        continue
                    pr, pc = h - 1 - r, w - 1 - c
                    if (r, c) == (pr, pc):
                        continue
                    if not mask[pr, pc]:
                        continue
                    candidates.append((r, c))
            if not candidates:
                break
            r, c = random.choice(candidates)
            mask[r, c] = False
            if (
                not _touches_all_sides(mask)
                or not _is_connected(mask)
                or _has_hole(mask)
            ):
                mask[r, c] = True
                continue
            removed += 1
        if removed == 0:
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _sample_room_layout() -> tuple[int, int, list[int], list[int]] | None:
    room_count = random.choice(ROOM_COUNTS)
    rows, cols = random.choice(ROOM_LAYOUTS[int(room_count)])
    for _ in range(60):
        if rows == 1:
            row_sizes = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)]
        else:
            row_sizes = random.sample(range(ROOM_SIZE_MIN, ROOM_SIZE_MAX + 1), rows)
        if cols == 1:
            col_sizes = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)]
        else:
            col_sizes = random.sample(range(ROOM_SIZE_MIN, ROOM_SIZE_MAX + 1), cols)

        dims = [(int(rh), int(cw)) for rh in row_sizes for cw in col_sizes]
        canon = [tuple(sorted((h, w))) for h, w in dims]
        if len(set(canon)) != len(canon):
            continue

        container_h = sum(row_sizes) + rows + 1
        container_w = sum(col_sizes) + cols + 1
        if container_h > 20 or container_w > 20:
            continue
        return int(rows), int(cols), [int(x) for x in row_sizes], [int(x) for x in col_sizes]
    return None


def _place_patch(
    grid: np.ndarray, occupied: np.ndarray, patch_mask: np.ndarray, color: int
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    ph, pw = patch_mask.shape
    blocked = _dilate8(occupied)
    row_choices = list(range(h - ph + 1))
    col_choices = list(range(w - pw + 1))
    if not row_choices or not col_choices:
        return None

    for _ in range(PLACE_ATTEMPTS):
        r0 = random.choice(row_choices)
        c0 = random.choice(col_choices)
        sub = blocked[r0 : r0 + ph, c0 : c0 + pw]
        if bool((sub & patch_mask).any()):
            continue
        coords = np.argwhere(patch_mask)
        for rr, cc in coords:
            grid[r0 + int(rr), c0 + int(cc)] = int(color)
            occupied[r0 + int(rr), c0 + int(cc)] = True
        return int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    layout = _sample_room_layout()
    if layout is None:
        return None
    rows, cols, row_sizes, col_sizes = layout
    container_h = sum(row_sizes) + rows + 1
    container_w = sum(col_sizes) + cols + 1

    grid_h = min(30, container_h + random.randint(GRID_PAD_MIN, GRID_PAD_MAX))
    grid_w = min(30, container_w + random.randint(GRID_PAD_MIN, GRID_PAD_MAX))
    if grid_h <= container_h + 2 or grid_w <= container_w + 2:
        return None

    top = random.randint(1, grid_h - container_h - 1)
    left = random.randint(1, grid_w - container_w - 1)

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    guide = random.choice(palette)
    palette.remove(guide)

    room_total = rows * cols
    if len(palette) < 2 * room_total:
        return None
    frag_colors = random.sample(palette, 2 * room_total)

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    inp[top : top + container_h, left : left + container_w] = int(guide)
    container_mask = np.zeros((grid_h, grid_w), dtype=bool)
    container_mask[top : top + container_h, left : left + container_w] = True

    rooms = []
    rr = top + 1
    color_idx = 0
    for rh in row_sizes:
        cc = left + 1
        for cw in col_sizes:
            room_box = (rr, cc, rr + rh - 1, cc + cw - 1)
            inp[rr : rr + rh, cc : cc + cw] = int(bg)
            shape = _generate_cover_shape(rh, cw)
            if shape is None:
                return None
            color_a = frag_colors[color_idx]
            color_b = frag_colors[color_idx + 1]
            color_idx += 2
            rooms.append(
                {
                    "bbox": room_box,
                    "shape": shape,
                    "color_a": int(color_a),
                    "color_b": int(color_b),
                }
            )
            cc += cw + 1
        rr += rh + 1

    guide_only = inp.copy()
    occupied = container_mask.copy()

    fragments = []
    for idx, room in enumerate(rooms):
        shape = room["shape"]
        fragments.append(
            {
                "room_idx": int(idx),
                "mask": np.rot90(shape, random.randrange(4)),
                "color": int(room["color_a"]),
            }
        )
        fragments.append(
            {
                "room_idx": int(idx),
                "mask": np.rot90(shape, random.randrange(4)),
                "color": int(room["color_b"]),
            }
        )

    random.shuffle(fragments)
    fragments.sort(key=lambda item: -int(item["mask"].sum()))
    for frag in fragments:
        placed_box = _place_patch(inp, occupied, frag["mask"], frag["color"])
        if placed_box is None:
            return None
        frag["bbox"] = placed_box

    expected = guide_only.copy()
    for idx, room in enumerate(rooms):
        room_frags = [frag for frag in fragments if frag["room_idx"] == idx]
        room_frags.sort(key=lambda frag: (frag["bbox"][0], frag["bbox"][1], frag["color"]))
        colors = [int(room_frags[0]["color"]), int(room_frags[1]["color"])]

        r0, c0, r1, c1 = room["bbox"]
        shape = room["shape"]
        cover_a, cover_b = _ordered_masks(shape, np.rot90(shape, 2))
        patch = expected[r0 : r1 + 1, c0 : c1 + 1]
        only_a = cover_a & ~cover_b
        only_b = cover_b & ~cover_a
        overlap = cover_a & cover_b
        patch[only_a] = colors[0]
        patch[only_b] = colors[1]
        patch[overlap] = int(guide)

    if _most_common_color(inp) != int(bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC input")
