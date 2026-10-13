from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
OUTLINE = 3
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
ROT_K_CCW = {"right": 0, "down": 3, "left": 2, "up": 1}
AREA_POOL = (3, 4, 5, 6)
GEN_ATTEMPTS = 500

SHAPE_POOLS: dict[int, list[set[tuple[int, int]]]] = {
    3: [
        {(0, 0), (1, 0), (1, 1)},
    ],
    4: [
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2)},
        {(0, 1), (0, 2), (1, 0), (1, 1)},
    ],
    5: [
        {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
        {(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)},
    ],
    6: [
        {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 2), (2, 3)},
        {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)},
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = DIRS4 if connectivity == 4 else DIRS8
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in nbrs:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _crop_tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=mask.dtype)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    tight = _crop_tight(mask)
    out = np.zeros(tight.shape, dtype=int)
    out[tight] = int(color)
    return out


def _nonzero_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != 0)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_patch_for_direction(patch: np.ndarray, direction: str) -> np.ndarray:
    return _nonzero_patch(np.rot90(patch, ROT_K_CCW[direction]))


def _transform_cells(cells: set[tuple[int, int]], rot_k: int, flip_h: bool) -> set[tuple[int, int]]:
    pts = set(cells)
    if flip_h:
        max_c = max(c for _r, c in pts)
        pts = {(r, max_c - c) for r, c in pts}
    for _ in range(rot_k):
        max_r = max(r for r, _c in pts)
        pts = {(c, max_r - r) for r, c in pts}
    min_r = min(r for r, _c in pts)
    min_c = min(c for _r, c in pts)
    return {(r - min_r, c - min_c) for r, c in pts}


def _shape_to_patch(cells: set[tuple[int, int]], color: int = 1) -> np.ndarray:
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in cells:
        patch[r, c] = int(color)
    return patch


def _random_body_patch(area: int, color: int) -> np.ndarray:
    base = random.choice(SHAPE_POOLS[int(area)])
    seen = set()
    variants: list[np.ndarray] = []
    for rot_k in range(4):
        for flip_h in (False, True):
            cells = _transform_cells(base, rot_k, flip_h)
            patch = _shape_to_patch(cells, color=color)
            key = (patch.shape, patch.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(patch)
    return random.choice(variants)


def _edge_positions(body_patch: np.ndarray, direction: str) -> list[int]:
    mask = body_patch != 0
    if direction == "right":
        return [int(r) for r in np.where(mask[:, -1])[0]]
    if direction == "left":
        return [int(r) for r in np.where(mask[:, 0])[0]]
    if direction == "down":
        return [int(c) for c in np.where(mask[-1, :])[0]]
    return [int(c) for c in np.where(mask[0, :])[0]]


def _attach_stem(body_patch: np.ndarray, direction: str, stem_len: int, color: int) -> tuple[np.ndarray, int]:
    bh, bw = body_patch.shape
    if direction in ("right", "left"):
        patch = np.zeros((bh, bw + stem_len), dtype=int)
        if direction == "right":
            patch[:, :bw] = body_patch
            anchor = random.choice(_edge_positions(body_patch, direction))
            patch[anchor, bw : bw + stem_len] = int(color)
        else:
            patch[:, stem_len:] = body_patch
            anchor = random.choice(_edge_positions(body_patch, direction))
            patch[anchor, :stem_len] = int(color)
    else:
        patch = np.zeros((bh + stem_len, bw), dtype=int)
        if direction == "down":
            patch[:bh, :] = body_patch
            anchor = random.choice(_edge_positions(body_patch, direction))
            patch[bh : bh + stem_len, anchor] = int(color)
        else:
            patch[stem_len:, :] = body_patch
            anchor = random.choice(_edge_positions(body_patch, direction))
            patch[:stem_len, anchor] = int(color)
    return patch, anchor


def _outline_patch_from_interior(interior_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner = np.zeros((interior_patch.shape[0] + 2, interior_patch.shape[1] + 2), dtype=bool)
    inner[1:-1, 1:-1] = interior_patch != 0
    ring = np.zeros_like(inner)
    for r, c in np.argwhere(inner):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < inner.shape[0] and 0 <= nc < inner.shape[1] and not inner[nr, nc]:
                ring[nr, nc] = True
    patch = np.zeros(inner.shape, dtype=int)
    patch[ring] = OUTLINE
    return patch, inner


def _cavity_mask_for_outline(grid: np.ndarray, outline_coords: np.ndarray, bg: int) -> np.ndarray:
    out = np.zeros(grid.shape, dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(outline_coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    wall = sub == OUTLINE
    reachable = np.zeros_like(wall, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    h, w = wall.shape
    border_cells = [(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)] + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)]
    for rr, cc in border_cells:
        if not wall[rr, cc] and not reachable[rr, cc]:
            reachable[rr, cc] = True
            q.append((rr, cc))
    while q:
        rr, cc = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))
    inside = (~wall) & (~reachable) & (sub == bg)
    out[r0 : r1 + 1, c0 : c1 + 1] = inside
    return out


def _candidate_split(mask: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if direction == "right":
        max_len = width - 1
        for stem_len in range(max_len, 1, -1):
            cols = range(c1 - stem_len + 1, c1 + 1)
            for rr in range(r0, r1 + 1):
                if not all(mask[rr, cc] for cc in cols):
                    continue
                if not all(int(mask[:, cc].sum()) == 1 for cc in cols):
                    continue
                if c1 - stem_len < 0 or not mask[rr, c1 - stem_len]:
                    continue
                stem = np.zeros_like(mask)
                for cc in cols:
                    stem[rr, cc] = True
                body = mask & ~stem
                if _valid_body(body) and _bbox_coords(np.argwhere(body))[3] == c1 - stem_len:
                    return body, stem
    if direction == "left":
        max_len = width - 1
        for stem_len in range(max_len, 1, -1):
            cols = range(c0, c0 + stem_len)
            for rr in range(r0, r1 + 1):
                if not all(mask[rr, cc] for cc in cols):
                    continue
                if not all(int(mask[:, cc].sum()) == 1 for cc in cols):
                    continue
                if c0 + stem_len >= mask.shape[1] or not mask[rr, c0 + stem_len]:
                    continue
                stem = np.zeros_like(mask)
                for cc in cols:
                    stem[rr, cc] = True
                body = mask & ~stem
                if _valid_body(body) and _bbox_coords(np.argwhere(body))[1] == c0 + stem_len:
                    return body, stem
    if direction == "down":
        max_len = height - 1
        for stem_len in range(max_len, 1, -1):
            rows = range(r1 - stem_len + 1, r1 + 1)
            for cc in range(c0, c1 + 1):
                if not all(mask[rr, cc] for rr in rows):
                    continue
                if not all(int(mask[rr, :].sum()) == 1 for rr in rows):
                    continue
                if r1 - stem_len < 0 or not mask[r1 - stem_len, cc]:
                    continue
                stem = np.zeros_like(mask)
                for rr in rows:
                    stem[rr, cc] = True
                body = mask & ~stem
                if _valid_body(body) and _bbox_coords(np.argwhere(body))[2] == r1 - stem_len:
                    return body, stem
    if direction == "up":
        max_len = height - 1
        for stem_len in range(max_len, 1, -1):
            rows = range(r0, r0 + stem_len)
            for cc in range(c0, c1 + 1):
                if not all(mask[rr, cc] for rr in rows):
                    continue
                if not all(int(mask[rr, :].sum()) == 1 for rr in rows):
                    continue
                if r0 + stem_len >= mask.shape[0] or not mask[r0 + stem_len, cc]:
                    continue
                stem = np.zeros_like(mask)
                for rr in rows:
                    stem[rr, cc] = True
                body = mask & ~stem
                if _valid_body(body) and _bbox_coords(np.argwhere(body))[0] == r0 + stem_len:
                    return body, stem
    return None


def _valid_body(body: np.ndarray) -> bool:
    area = int(body.sum())
    if area < 3 or area > 6:
        return False
    comps = _components(body, connectivity=4)
    if len(comps) != 1:
        return False
    tight = _crop_tight(body)
    if tight.shape[0] == 1 or tight.shape[1] == 1:
        return False
    if int(tight.sum()) == tight.shape[0] * tight.shape[1]:
        return False
    return True


def _try_split_body_and_stem(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str] | None:
    candidates = []
    for direction in ("right", "down", "left", "up"):
        split = _candidate_split(mask, direction)
        if split is None:
            continue
        body, stem = split
        candidates.append((direction, body, stem, int(stem.sum())))
    if len(candidates) != 1:
        return None
    direction, body, stem, _ = candidates[0]
    return body, stem, direction


def _split_body_and_stem(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    result = _try_split_body_and_stem(mask)
    assert result is not None, "expected a unique stem direction"
    return result


def _extract_outline_info(grid: np.ndarray, bg: int) -> list[dict]:
    infos = []
    for coords in _components(grid == OUTLINE, connectivity=8):
        cavity = _cavity_mask_for_outline(grid, coords, bg)
        cavity_coords = np.argwhere(cavity)
        assert cavity_coords.size > 0, "outline must enclose a cavity"
        infos.append(
            {
                "outline_coords": coords,
                "cavity_mask": cavity,
                "cavity_coords": cavity_coords,
                "cavity_area": int(cavity_coords.shape[0]),
                "cavity_bbox": _bbox_coords(cavity_coords),
            }
        )
    areas = [info["cavity_area"] for info in infos]
    assert len(areas) == len(set(areas)), "cavity areas must be distinct"
    return infos


def _extract_sprite_info(grid: np.ndarray, bg: int) -> list[dict]:
    infos = []
    colors = sorted(set(map(int, grid.flat)) - {int(bg), OUTLINE})
    for color in colors:
        comps = _components(grid == color, connectivity=4)
        for coords in comps:
            mask = _mask_from_coords(coords, grid.shape)
            body, stem, direction = _split_body_and_stem(mask)
            infos.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "body_mask": body,
                    "stem_mask": stem,
                    "direction": direction,
                    "body_area": int(body.sum()),
                    "body_patch": _patch_from_mask(body, color),
                }
            )
    areas = [info["body_area"] for info in infos]
    assert 2 <= len(infos) <= 3, "expected 2-3 sprites"
    assert len(areas) == len(set(areas)), "sprite body areas must be distinct"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each stem-marked sprite body into the unique green cavity whose area matches the body size, rotating by the stem direction.

    Input structure:
    - A black-background grid contains 2-3 small monochrome sprite-plus-stem components in distinct non-green colors.
    - Each such component is 4-connected and consists of an irregular body plus a straight one-cell-wide stem extending from one side of the body's bbox.
    - The grid also contains 2-3 green outline components, each an 8-connected closed ring surrounding a black cavity with a distinct area.

    Rule:
    - For every non-green sprite component, separate the irregular body from its straight stem.
    - Count only the body pixels, find the unique green cavity whose interior black area equals that count, and decode the stem direction as a rotation: right->0, down->clockwise 90, left->180, up->counterclockwise 90.
    - Erase the sprite and stem from their source location, rotate the body by that angle, and place it at the top-left anchor of the matched cavity bbox while keeping the green outline unchanged.

    Key insight:
    - The same stem carries two coupled signals: it marks which pixels must be excluded from the area count, and its pointing direction encodes the rotation needed before the body is inserted into the matched cavity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    outlines = _extract_outline_info(g, bg)
    sprites = _extract_sprite_info(g, bg)
    assert len(outlines) == len(sprites), "outline and sprite counts must match"
    assert sorted(info["cavity_area"] for info in outlines) == sorted(info["body_area"] for info in sprites)

    outline_by_area = {info["cavity_area"]: info for info in outlines}
    out = g.copy()

    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = int(bg)

    for sprite in sprites:
        target = outline_by_area[sprite["body_area"]]
        rotated = _rotate_patch_for_direction(sprite["body_patch"], sprite["direction"])
        cavity_bbox = target["cavity_bbox"]
        base_r, base_c = cavity_bbox[0], cavity_bbox[1]
        rot_coords = np.argwhere(rotated != 0)
        assert int(rot_coords.shape[0]) == sprite["body_area"]
        for rr, cc in rot_coords:
            gr = base_r + int(rr)
            gc = base_c + int(cc)
            assert target["cavity_mask"][gr, gc], "rotated body must stay inside the matched cavity"
        filled_mask = np.zeros_like(target["cavity_mask"])
        for rr, cc in rot_coords:
            gr = base_r + int(rr)
            gc = base_c + int(cc)
            filled_mask[gr, gc] = True
            out[gr, gc] = sprite["color"]
        assert int(filled_mask.sum()) == target["cavity_area"], "area-equal placement must fill the entire cavity"

    return out


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    cells = np.argwhere(patch != 0)
    for rr, cc in cells:
        gr = r0 + int(rr)
        gc = c0 + int(cc)
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                    return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    cells = np.argwhere(patch != 0)
    for rr, cc in cells:
        grid[r0 + int(rr), c0 + int(cc)] = int(patch[int(rr), int(cc)])


def _random_location(occupied: np.ndarray, patch: np.ndarray, padding: int) -> tuple[int, int]:
    h, w = patch.shape
    candidates = []
    for r0 in range(occupied.shape[0] - h + 1):
        for c0 in range(occupied.shape[1] - w + 1):
            if _can_place(occupied, patch, r0, c0, padding):
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _sample_areas_and_stems(count: int) -> tuple[list[int], list[int]]:
    for _ in range(200):
        areas = random.sample(AREA_POOL, count)
        stems = [random.randint(2, 4) for _ in range(count)]
        if all((area + stem) not in areas for area, stem in zip(areas, stems)):
            return areas, stems
    raise RuntimeError("failed to sample non-degenerate area/stem pairs")


def _instance_output(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        for rr, cc in np.argwhere(item["sprite_patch"] != 0):
            out[item["sprite_pos"][0] + int(rr), item["sprite_pos"][1] + int(cc)] = BG
        for rr, cc in np.argwhere(item["final_patch"] != 0):
            out[item["outline_pos"][0] + 1 + int(rr), item["outline_pos"][1] + 1 + int(cc)] = int(item["color"])
    return out


def _centroid_from_coords(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        produced = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(produced, out):
        return False

    try:
        bg = _get_bg(inp)
        outlines = _extract_outline_info(inp, bg)
        sprites = _extract_sprite_info(inp, bg)
    except AssertionError:
        return False
    if len(outlines) != len(sprites):
        return False
    if sorted(info["cavity_area"] for info in outlines) != sorted(info["body_area"] for info in sprites):
        return False

    if not any(
        info["direction"] != "right"
        and _rotate_patch_for_direction(info["body_patch"], info["direction"]).tobytes() != info["body_patch"].tobytes()
        for info in sprites
    ):
        return False

    body_areas = {info["body_area"] for info in sprites}
    if any(int(info["coords"].shape[0]) in body_areas for info in sprites):
        return False

    if len(sprites) >= 2:
        cavity_centers = {info["cavity_area"]: _centroid_from_coords(info["cavity_coords"]) for info in outlines}
        mismatch_found = False
        for sprite in sprites:
            s_center = _centroid_from_coords(sprite["coords"])
            dists = {
                area: abs(s_center[0] - center[0]) + abs(s_center[1] - center[1])
                for area, center in cavity_centers.items()
            }
            nearest_area = min(dists, key=dists.get)
            if nearest_area != sprite["body_area"]:
                mismatch_found = True
                break
        if not mismatch_found:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([2, 3])
    areas, stem_lengths = _sample_areas_and_stems(count)
    colors = random.sample([1, 2, 4, 5, 6, 7, 8, 9], count)
    directions = random.sample(["right", "down", "left", "up"], count)
    if all(direction == "right" for direction in directions):
        return None

    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    placements: list[dict] = []
    try:
        for area, color, direction, stem_len in zip(areas, colors, directions, stem_lengths):
            body_patch = _random_body_patch(area, color)
            final_patch = _rotate_patch_for_direction(body_patch, direction)
            outline_patch, inner_mask = _outline_patch_from_interior(final_patch)
            sprite_patch, _ = _attach_stem(body_patch, direction, stem_len, color)

            parsed = _try_split_body_and_stem(sprite_patch != 0)
            if parsed is None or parsed[2] != direction:
                return None

            outline_pos = _random_location(occupied, outline_patch, padding=2)
            _place_patch(grid, outline_patch, *outline_pos)
            outline_mask = np.zeros_like(occupied)
            outline_mask[
                outline_pos[0] : outline_pos[0] + outline_patch.shape[0],
                outline_pos[1] : outline_pos[1] + outline_patch.shape[1],
            ] = outline_patch != 0
            occupied |= outline_mask

            sprite_pos = _random_location(occupied, sprite_patch, padding=2)
            _place_patch(grid, sprite_patch, *sprite_pos)
            sprite_mask = np.zeros_like(occupied)
            sprite_mask[
                sprite_pos[0] : sprite_pos[0] + sprite_patch.shape[0],
                sprite_pos[1] : sprite_pos[1] + sprite_patch.shape[1],
            ] = sprite_patch != 0
            occupied |= sprite_mask

            placements.append(
                {
                    "area": area,
                    "color": color,
                    "direction": direction,
                    "stem_len": stem_len,
                    "body_patch": body_patch,
                    "final_patch": final_patch,
                    "outline_patch": outline_patch,
                    "inner_mask": inner_mask,
                    "sprite_patch": sprite_patch,
                    "outline_pos": outline_pos,
                    "sprite_pos": sprite_pos,
                }
            )
    except ValueError:
        return None

    out = _instance_output(grid, placements)
    if not _semantic_checks(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
