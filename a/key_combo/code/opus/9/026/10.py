from __future__ import annotations
from collections import Counter, deque

import numpy as np


BG = 0
ANCHOR_COLOR = 8
MOBILE_COLORS = (2, 3, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in DIR8]


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[Coord]) -> tuple[Coord, ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _translate(coords: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in coords}


def _core_pixels(coords: set[Coord]) -> set[Coord]:
    if not coords:
        return set()
    r0, c0, r1, c1 = _bbox(coords)
    core: set[Coord] = set()
    for r in range(r0, r1):
        for c in range(c0, c1):
            block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if block <= coords:
                core |= block
    return core


def _component_masks(grid: np.ndarray, color: int) -> list[set[Coord]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[Coord]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        comp: set[Coord] = set()
        while q:
            cr, cc = q.popleft()
            comp.add((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _is_connected(coords: set[Coord]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nb in _neighbors4(*cur):
            if nb in coords and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(coords)


def _is_simple_path(coords: set[Coord]) -> tuple[bool, list[Coord]]:
    if not coords or not _is_connected(coords):
        return False, []
    if len(coords) == 1:
        return True, list(coords)
    deg = {cell: sum(nb in coords for nb in _neighbors4(*cell)) for cell in coords}
    ends = [cell for cell, d in deg.items() if d == 1]
    if len(ends) != 2:
        return False, []
    if any(d not in (1, 2) for d in deg.values()):
        return False, []
    return True, ends


def _decompose_mobile_component(coords: set[Coord]) -> dict:
    core = _core_pixels(coords)
    candidates: list[dict] = []
    if not core:
        raise ValueError("component has no 2x2 body core")

    degrees = {cell: sum(nb in coords for nb in _neighbors4(*cell)) for cell in coords}
    leaves = [cell for cell, deg in degrees.items() if deg == 1]
    for leaf in leaves:
        prev = None
        cur = leaf
        chain = [leaf]
        while cur not in core:
            nxt = [nb for nb in _neighbors4(*cur) if nb in coords and nb != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
        if cur not in core:
            continue

        path_cells = set(chain[:-1])
        if len(path_cells) < 2:
            continue
        body = coords - path_cells
        if not (4 <= len(body) <= 7):
            continue
        if not core <= body:
            continue
        if not _is_connected(body):
            continue

        extras = body - core
        if any(not any(nb in core for nb in _neighbors4(*cell)) for cell in extras):
            continue

        body_attach = chain[-1]
        near_body = chain[-2]
        if {nb for nb in _neighbors4(*near_body) if nb in body} != {body_attach}:
            continue
        if any(any(nb in body for nb in _neighbors4(*cell)) for cell in chain[:-2]):
            continue

        candidates.append(
            {
                "body": body,
                "path": path_cells,
                "body_attach": body_attach,
                "path_start": near_body,
                "endpoint": leaf,
                "norm": _normalize(body),
            }
        )

    if len(candidates) != 1:
        raise ValueError(f"expected exactly one sprite/path decomposition, found {len(candidates)}")
    return candidates[0]


def _paint(grid: np.ndarray, coords: set[Coord], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored sprite to its path endpoint, keep only anchor-shaped results, then align them to the anchor's top row.

    Input structure:
    - The background is black.
    - One cyan 4-connected sprite is the anchor and has no path.
    - Three or four other colors each form one connected sprite-plus-path component: a 4-7 pixel body with a same-colored 1-pixel-wide 4-connected trail attached at exactly one body edge pixel.

    Rule:
    - For every non-anchor color, separate the component into the sprite body and its attached simple path.
    - Move the whole body by the vector from the attachment body pixel to the path endpoint, then erase the original body and the full path.
    - Compare the moved body's normalized mask with the cyan anchor's normalized mask.
    - If the masks match, keep that sprite but vertically shift it so its bounding-box top row equals the anchor top row while preserving its moved columns; otherwise erase it.
    - Keep the cyan anchor unchanged.

    Key insight:
    - The endpoint of each attached trail determines a translated candidate sprite position, but only candidates sharing the anchor's normalized mask survive to the final common top-row alignment.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = np.full_like(g, bg)

    anchor_comps = _component_masks(g, ANCHOR_COLOR)
    if len(anchor_comps) != 1:
        raise ValueError("expected exactly one cyan anchor component")
    anchor = anchor_comps[0]
    anchor_norm = _normalize(anchor)
    anchor_top = _bbox(anchor)[0]
    _paint(out, anchor, ANCHOR_COLOR)

    for color in sorted(set(int(v) for v in np.unique(g)) - {bg, ANCHOR_COLOR}):
        comps = _component_masks(g, color)
        if len(comps) != 1:
            raise ValueError(f"expected one component for color {color}")
        decomp = _decompose_mobile_component(comps[0])
        ar, ac = decomp["body_attach"]
        er, ec = decomp["endpoint"]
        moved = _translate(decomp["body"], er - ar, ec - ac)
        if _normalize(moved) != anchor_norm:
            continue
        moved_top = _bbox(moved)[0]
        aligned = _translate(moved, anchor_top - moved_top, 0)
        if any(not _in_bounds(g.shape, cell) for cell in aligned):
            raise ValueError("aligned sprite left grid bounds")
        if any(out[r, c] != bg for r, c in aligned):
            raise ValueError("aligned sprites overlap")
        _paint(out, aligned, color)

    return out


def _mask_dimensions(mask: set[Coord]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(mask)
    return r1 - r0 + 1, c1 - c0 + 1


def _relative_mask(mask: set[Coord]) -> set[Coord]:
    return set(_normalize(mask))


def _random_connected_mask(area: int) -> set[Coord]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = {nb for cell in cells for nb in _neighbors4(*cell) if nb not in cells}
        choice = sorted(frontier)[int(np.random.randint(len(frontier)))]
        cells.add(choice)
    norm = _relative_mask(cells)
    h, w = _mask_dimensions(norm)
    if h > 4 or w > 4:
        return _random_connected_mask(area)
    return norm


DELTA_POOL: tuple[tuple[int, int], ...] = ((-3, 2), (-2, 3), (2, 3), (3, 2), (-3, -2), (-2, -3), (2, -3), (3, -2))
_ATTACHMENT_CACHE: dict[tuple[Coord, ...], list[tuple[int, int, int, int, bool]]] = {}


def _random_transform(mask: set[Coord]) -> set[Coord]:
    arr = np.array(sorted(mask), dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return {tuple(map(int, cell)) for cell in arr}


def _core_family_mask(extra_count: int) -> set[Coord]:
    core = {(0, 0), (0, 1), (1, 0), (1, 1)}
    extras = [(-1, 0), (-1, 1), (0, -1), (1, -1), (0, 2), (1, 2), (2, 0), (2, 1)]
    chosen = np.random.choice(len(extras), size=extra_count, replace=False)
    mask = set(core)
    for idx in chosen:
        mask.add(extras[int(idx)])
    norm = _relative_mask(mask)
    return _random_transform(norm)


def _attachment_specs(mask: set[Coord]) -> list[tuple[int, int, int, int, bool]]:
    key = _normalize(mask)
    if key in _ATTACHMENT_CACHE:
        return _ATTACHMENT_CACHE[key]

    specs: list[tuple[int, int, int, int, bool]] = []
    shifted_mask = _translate(mask, 10, 10)
    core = _core_pixels(mask)
    for ar, ac in sorted(core):
        for dr, dc in DELTA_POOL:
            attach0 = (10 + ar, 10 + ac)
            end0 = (attach0[0] + dr, attach0[1] + dc)
            for horizontal_first in (True, False):
                path = _manhattan_path(attach0, end0, horizontal_first)
                if not _valid_path(shifted_mask, attach0, path, set(), (30, 30)):
                    continue
                specs.append((ar, ac, dr, dc, horizontal_first))
    _ATTACHMENT_CACHE[key] = specs
    return specs


def _random_anchor_mask() -> set[Coord]:
    for _ in range(100):
        mask = _core_family_mask(int(np.random.randint(1, 4)))
        h, w = _mask_dimensions(mask)
        if len(mask) != h * w:
            return mask
    raise RuntimeError("failed to create anchor mask")


def _random_distractor_mask(anchor_norm: tuple[Coord, ...]) -> set[Coord]:
    for _ in range(100):
        mask = _core_family_mask(int(np.random.randint(1, 4)))
        if _normalize(mask) != anchor_norm:
            return mask
    raise RuntimeError("failed to create distractor mask")


def _placeable(mask: set[Coord], top: int, left: int, blocked_halo: set[Coord], shape: tuple[int, int]) -> bool:
    placed = _translate(mask, top, left)
    if any(not _in_bounds(shape, cell) for cell in placed):
        return False
    return all(cell not in blocked_halo for cell in placed)


def _halo(coords: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out = set(coords)
    for r, c in coords:
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, (nr, nc)):
                out.add((nr, nc))
    return out


def _boundary_pixels(mask: set[Coord]) -> list[Coord]:
    return sorted(cell for cell in mask if any(nb not in mask for nb in _neighbors4(*cell)))


def _manhattan_path(start: Coord, end: Coord, horizontal_first: bool) -> list[Coord]:
    r, c = start
    er, ec = end
    path: list[Coord] = []
    if horizontal_first:
        step_c = 0 if ec == c else (1 if ec > c else -1)
        while c != ec:
            c += step_c
            path.append((r, c))
        step_r = 0 if er == r else (1 if er > r else -1)
        while r != er:
            r += step_r
            path.append((r, c))
    else:
        step_r = 0 if er == r else (1 if er > r else -1)
        while r != er:
            r += step_r
            path.append((r, c))
        step_c = 0 if ec == c else (1 if ec > c else -1)
        while c != ec:
            c += step_c
            path.append((r, c))
    return path


def _valid_path(
    body: set[Coord],
    attachment: Coord,
    path: list[Coord],
    blocked_halo: set[Coord],
    shape: tuple[int, int],
) -> bool:
    if len(path) < 2:
        return False
    if len(set(path)) != len(path):
        return False
    if any(not _in_bounds(shape, cell) for cell in path):
        return False
    if any(cell in blocked_halo for cell in path):
        return False
    body_set = set(body)
    path_set = set(path)
    if body_set & path_set:
        return False

    for i, cell in enumerate(path):
        r, c = cell
        body4 = {nb for nb in _neighbors4(r, c) if nb in body_set}
        if i == 0:
            if body4 != {attachment}:
                return False
        else:
            if body4:
                return False

    for i, cell in enumerate(path):
        prev = path[i - 1] if i > 0 else attachment
        nxt = path[i + 1] if i + 1 < len(path) else None
        path4 = {nb for nb in _neighbors4(*cell) if nb in path_set}
        expected = {prev}
        if nxt is not None:
            expected.add(nxt)
        if path4 != expected & path_set:
            return False
    return True


def _place_objects(
    shape: tuple[int, int],
    anchor_mask: set[Coord],
    anchor_top: int,
    anchor_left: int,
    objects: list[dict],
) -> bool:
    anchor_abs = _translate(anchor_mask, anchor_top, anchor_left)
    input_halo = _halo(anchor_abs, shape)
    moved_halo = _halo(anchor_abs, shape)
    final_halo = _halo(anchor_abs, shape)
    used_lefts: set[int] = set()

    order = sorted(range(len(objects)), key=lambda i: (not objects[i]["keep"], np.random.rand()))
    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        h, w = _mask_dimensions(mask)
        candidates = []
        specs = _attachment_specs(mask)
        if not specs:
            return False
        perm = np.random.permutation(len(specs))
        for spec_i in perm:
            ar, ac, dr, dc, horizontal_first = specs[int(spec_i)]
            top_lo = max(0, -dr)
            top_hi = min(shape[0] - h, shape[0] - h - dr)
            left_lo = max(0, -dc)
            left_hi = min(shape[1] - w, shape[1] - w - dc)
            if top_lo > top_hi or left_lo > left_hi:
                continue
            for top in range(top_lo, top_hi + 1):
                for left in range(left_lo, left_hi + 1):
                    body = _translate(mask, top, left)
                    if any(cell in input_halo for cell in body):
                        continue

                    moved = _translate(body, dr, dc)
                    moved_top, moved_left, _mr1, _mc1 = _bbox(moved)
                    if moved_left in used_lefts:
                        continue
                    if any(cell in moved_halo for cell in moved):
                        continue
                    if obj["keep"] and moved_top == anchor_top:
                        continue

                    if obj["keep"]:
                        aligned = _translate(moved, anchor_top - moved_top, 0)
                        if any(not _in_bounds(shape, cell) for cell in aligned):
                            continue
                        if any(cell in final_halo for cell in aligned):
                            continue

                    attach = (top + ar, left + ac)
                    end = (attach[0] + dr, attach[1] + dc)
                    if obj["keep"] and end[1] == attach[1]:
                        continue
                    path = _manhattan_path(attach, end, horizontal_first)
                    if _valid_path(body, attach, path, input_halo, shape):
                        candidates.append((top, left, dr, dc, attach, end, path))
            if candidates:
                break
        if not candidates:
            return False

        top, left, dr, dc, attach, end, path = candidates[int(np.random.randint(len(candidates)))]
        obj["input_top"] = top
        obj["input_left"] = left
        obj["attach"] = attach
        obj["endpoint"] = end
        obj["path"] = path
        obj["delta"] = (dr, dc)
        obj["moved_top"] = top + dr
        obj["moved_left"] = left + dc

        body = _translate(mask, top, left)
        moved = _translate(body, dr, dc)
        input_halo |= _halo(body | set(path), shape)
        moved_halo |= _halo(moved, shape)
        used_lefts.add(obj["moved_left"])
        if obj["keep"]:
            aligned = _translate(moved, anchor_top - _bbox(moved)[0], 0)
            final_halo |= _halo(aligned, shape)
    return True


def _render_input(shape: tuple[int, int], anchor_mask: set[Coord], anchor_top: int, anchor_left: int, objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _paint(grid, _translate(anchor_mask, anchor_top, anchor_left), ANCHOR_COLOR)
    for obj in objects:
        body = _translate(obj["mask"], obj["input_top"], obj["input_left"])
        _paint(grid, body | set(obj["path"]), obj["color"])
    return grid


def _validate_case(grid: np.ndarray, anchor_mask: set[Coord], objects: list[dict]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    anchor = _component_masks(grid, ANCHOR_COLOR)
    if len(anchor) != 1 or _normalize(anchor[0]) != _normalize(anchor_mask):
        return False
    anchor_top = _bbox(anchor[0])[0]

    kept = 0
    removed = 0
    bent = 0
    for obj in objects:
        comps = _component_masks(grid, obj["color"])
        if len(comps) != 1:
            return False
        try:
            decomp = _decompose_mobile_component(comps[0])
        except Exception:
            return False
        if decomp["norm"] != _normalize(obj["mask"]):
            return False
        if decomp["body_attach"] != obj["attach"]:
            return False
        if decomp["endpoint"] != obj["endpoint"]:
            return False
        moved = _translate(decomp["body"], obj["delta"][0], obj["delta"][1])
        if _bbox(moved)[1] != obj["moved_left"]:
            return False
        if obj["keep"]:
            kept += 1
            if obj["delta"][1] == 0:
                return False
            if _bbox(moved)[0] == anchor_top:
                return False
            aligned = _translate(moved, anchor_top - _bbox(moved)[0], 0)
            if any(out[r, c] != obj["color"] for r, c in aligned):
                return False
        else:
            removed += 1
            if np.any(out == obj["color"]):
                return False
        bent += int(obj["delta"][0] != 0 and obj["delta"][1] != 0)

    if kept not in (2, 3) or removed not in (1, 2):
        return False
    if bent < 1:
        return False
    if np.count_nonzero(out == ANCHOR_COLOR) != len(anchor[0]):
        return False
    return True


def _attempt() -> np.ndarray | None:
    shape = (int(np.random.randint(18, 24)), int(np.random.randint(20, 27)))
    anchor_mask = _random_anchor_mask()
    ah, aw = _mask_dimensions(anchor_mask)
    anchor_top = int(np.random.randint(1, shape[0] - ah - 1))
    anchor_left = int(np.random.randint(1, shape[1] - aw - 1))

    total = 3 if np.random.rand() < 0.5 else 4
    same_count = 2 if total == 3 else int(np.random.choice([2, 3]))
    distract_count = total - same_count
    colors = [int(c) for c in np.random.choice(MOBILE_COLORS, size=total, replace=False)]
    np.random.shuffle(colors)

    anchor_norm = _normalize(anchor_mask)
    objects: list[dict] = []
    for i in range(total):
        keep = i < same_count
        mask = set(anchor_mask) if keep else _random_distractor_mask(anchor_norm)
        objects.append({"color": colors[i], "mask": mask, "keep": keep})
    np.random.shuffle(objects)

    if not _place_objects(shape, anchor_mask, anchor_top, anchor_left, objects):
        return None

    grid = _render_input(shape, anchor_mask, anchor_top, anchor_left, objects)
    if not _validate_case(grid, anchor_mask, objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
