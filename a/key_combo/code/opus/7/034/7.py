from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _uniform_square_info(grid: np.ndarray, coords: np.ndarray):
    r0, c0, r1, c1 = bbox_coords(coords)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    if side_r != side_c or side_r < 2:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(patch[0, 0])
    if color == BG or not np.all(patch == color) or patch.size != len(coords):
        return None
    return {
        "r0": int(r0),
        "c0": int(c0),
        "size": int(side_r),
        "color": color,
        "coords": coords,
    }


def _scene_components(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    comps = components_mask(g != BG)
    assert comps, "expected a minimap and at least one block"
    infos = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        infos.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "crop": crop_bbox(g, (r0, c0, r1, c1)),
            }
        )
    infos.sort(key=lambda info: (info["bbox"][0], info["bbox"][1], info["bbox"][2] - info["bbox"][0], info["bbox"][3] - info["bbox"][1]))
    minimap = infos[0]
    block_infos = []
    for info in infos[1:]:
        block = _uniform_square_info(g, info["coords"])
        if block is not None:
            block_infos.append(block)
    assert block_infos, "expected square blocks outside the minimap"
    return minimap, block_infos


def _layout_candidates(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    minimap_info, block_infos = _scene_components(g)
    minimap = np.asarray(minimap_info["crop"], dtype=int)
    mh, mw = minimap.shape

    size_counts = Counter(block["size"] for block in block_infos)
    block_size = max(size_counts, key=size_counts.get)
    block_infos = [block for block in block_infos if block["size"] == block_size]

    candidates = []
    for step in range(block_size + 1, block_size + 4):
        max_base_r = h - ((mh - 1) * step + block_size)
        max_base_c = w - ((mw - 1) * step + block_size)
        if max_base_r < 0 or max_base_c < 0:
            continue
        for base_r in range(max_base_r + 1):
            for base_c in range(max_base_c + 1):
                seen_positions: set[tuple[int, int]] = set()
                direct_votes = 0
                ok = True
                for block in block_infos:
                    dr = block["r0"] - base_r
                    dc = block["c0"] - base_c
                    if dr % step != 0 or dc % step != 0:
                        ok = False
                        break
                    i = dr // step
                    j = dc // step
                    if not (0 <= i < mh and 0 <= j < mw):
                        ok = False
                        break
                    if (i, j) in seen_positions:
                        ok = False
                        break
                    map_color = int(minimap[i, j])
                    sym_color = int(minimap[mh - 1 - i, mw - 1 - j])
                    if block["color"] == map_color:
                        direct_votes += 1
                    elif block["color"] != sym_color:
                        ok = False
                        break
                    seen_positions.add((i, j))
                if ok and direct_votes >= 2:
                    candidates.append(
                        {
                            "direct_votes": direct_votes,
                            "step": int(step),
                            "base_r": int(base_r),
                            "base_c": int(base_c),
                            "block_size": int(block_size),
                            "minimap": minimap,
                            "blocks": block_infos,
                        }
                    )
    return candidates


def _solve_layout(grid: np.ndarray):
    candidates = _layout_candidates(grid)
    assert candidates, "could not infer the block lattice"
    best_votes = max(candidate["direct_votes"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["direct_votes"] == best_votes]
    assert len(best) == 1, "block lattice alignment must be uniquely determined"
    return best[0]


def _rot_partner(shape: tuple[int, int], pos: tuple[int, int]) -> tuple[int, int]:
    mh, mw = shape
    i, j = pos
    return mh - 1 - i, mw - 1 - j


def _place_block(grid: np.ndarray, top: tuple[int, int], size: int, color: int) -> None:
    r0, c0 = top
    grid[r0 : r0 + size, c0 : c0 + size] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a block layout from a top-left minimap while preserving blocks justified by 180-degree rotational symmetry.

    Input structure:
    - A black background contains one compact multicolor minimap in the top-left corner.
    - Elsewhere, the foreground consists of separated monochromatic square blocks of one shared size.
    - The blocks lie on a regular lattice whose step and offset are only partially revealed by the input.

    Rule:
    - Extract the top-left minimap and infer the block lattice from the already placed blocks.
    - For each minimap cell, project its color onto the corresponding block position in the main lattice.
    - Also compare that projected color with the color of the 180-degree partner minimap cell; when they differ, an already present block keeps its current color, otherwise an empty position takes the minimap color.

    Key insight:
    - The output is the minimap projection completed onto the whole block lattice, except that pre-existing conflict blocks are preserved because they are supported by the global 180-degree rotational counterpart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    layout = _solve_layout(g)
    minimap = layout["minimap"]
    block_size = layout["block_size"]
    step = layout["step"]
    base_r = layout["base_r"]
    base_c = layout["base_c"]
    mh, mw = minimap.shape

    existing: dict[tuple[int, int], int] = {}
    for block in layout["blocks"]:
        i = (block["r0"] - base_r) // step
        j = (block["c0"] - base_c) // step
        existing[(int(i), int(j))] = int(block["color"])

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            map_color = int(minimap[i, j])
            partner = _rot_partner((mh, mw), (i, j))
            sym_color = int(minimap[partner])
            if (i, j) in existing:
                color = existing[(i, j)]
            elif map_color == sym_color:
                color = map_color
            else:
                color = map_color
            _place_block(out, (base_r + i * step, base_c + j * step), block_size, color)
    return out


def _random_minimap() -> np.ndarray:
    shapes = ((4, 4), (4, 5), (5, 4))
    mh, mw = shapes[np.random.randint(0, len(shapes))]
    n_colors = int(np.random.randint(3, 6))
    palette = np.random.choice(PALETTE, size=n_colors, replace=False)
    while True:
        minimap = np.random.choice(palette, size=(mh, mw), replace=True).astype(int)
        if not np.array_equal(minimap, np.rot90(minimap, 2)):
            return minimap


def _all_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    mh, mw = shape
    return [(i, j) for i in range(mh) for j in range(mw)]


def _conflict_positions(minimap: np.ndarray) -> list[tuple[int, int]]:
    mh, mw = minimap.shape
    out = []
    for i in range(mh):
        for j in range(mw):
            ii, jj = _rot_partner((mh, mw), (i, j))
            if minimap[i, j] != minimap[ii, jj]:
                out.append((i, j))
    return out


def _choose_occupied_positions(minimap: np.ndarray) -> dict[tuple[int, int], str]:
    mh, mw = minimap.shape
    positions = _all_positions((mh, mw))
    far_positions = [pos for pos in positions if pos[0] >= mh // 2 or pos[1] >= mw // 2]
    conflict_positions = _conflict_positions(minimap)

    labels: dict[tuple[int, int], str] = {}

    conflict_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for pos in conflict_positions:
        partner = _rot_partner(minimap.shape, pos)
        pair = tuple(sorted((pos, partner)))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            conflict_pairs.append(pair)

    np.random.shuffle(positions)
    np.random.shuffle(far_positions)
    np.random.shuffle(conflict_pairs)

    must_pair = conflict_pairs[0]
    labels[must_pair[np.random.randint(0, 2)]] = "sym"

    conflict_target = int(np.random.randint(1, min(4, len(conflict_pairs) + 1)))
    for pair in conflict_pairs[1:]:
        if sum(kind == "sym" for kind in labels.values()) >= conflict_target:
            break
        chosen = pair[np.random.randint(0, 2)]
        if chosen not in labels and _rot_partner(minimap.shape, chosen) not in labels:
            labels[chosen] = "sym"

    direct_count = int(np.random.randint(3, 6))
    direct_pool = far_positions[:1] + [pos for pos in positions if pos not in far_positions[:1]]
    for pos in direct_pool:
        if pos in labels or _rot_partner(minimap.shape, pos) in labels:
            continue
        labels[pos] = "direct"
        if sum(kind == "direct" for kind in labels.values()) >= direct_count:
            break

    extras = [pos for pos in positions if pos not in labels]
    np.random.shuffle(extras)
    extra_direct = int(np.random.randint(0, 3))
    for pos in extras[:extra_direct]:
        labels[pos] = "direct"

    assert any(kind == "sym" for kind in labels.values())
    assert any(pos[0] >= mh // 2 or pos[1] >= mw // 2 for pos in labels)
    return labels


def _build_case():
    h = int(np.random.choice(np.arange(20, 29, 2)))
    w = int(np.random.choice(np.arange(20, 29, 2)))
    block_size = int(np.random.choice([2, 3]))
    gap = int(np.random.choice([1, 2]))
    step = block_size + gap
    minimap = _random_minimap()
    mh, mw = minimap.shape

    min_base_r = mh + 2
    min_base_c = mw + 3
    max_base_r = h - ((mh - 1) * step + block_size)
    max_base_c = w - ((mw - 1) * step + block_size)
    if max_base_r < min_base_r or max_base_c < min_base_c:
        return None

    base_r = int(np.random.randint(min_base_r, max_base_r + 1))
    base_c = int(np.random.randint(min_base_c, max_base_c + 1))
    occupied = _choose_occupied_positions(minimap)

    grid = np.zeros((h, w), dtype=int)
    grid[1 : 1 + mh, 1 : 1 + mw] = minimap

    for (i, j), kind in occupied.items():
        rr = base_r + i * step
        cc = base_c + j * step
        if kind == "direct":
            color = int(minimap[i, j])
        else:
            ii, jj = _rot_partner((mh, mw), (i, j))
            color = int(minimap[ii, jj])
        _place_block(grid, (rr, cc), block_size, color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is None:
            continue
        try:
            layout = _solve_layout(grid)
            minimap = layout["minimap"]
            blocks = layout["blocks"]
            seen_sym = False
            for block in blocks:
                i = (block["r0"] - layout["base_r"]) // layout["step"]
                j = (block["c0"] - layout["base_c"]) // layout["step"]
                partner = _rot_partner(minimap.shape, (i, j))
                if block["color"] == int(minimap[partner]) and block["color"] != int(minimap[i, j]):
                    seen_sym = True
                    break
            if not seen_sym:
                continue
            out = main(grid)
            if out.shape != grid.shape or max(out.shape) > 30:
                continue
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 7")
