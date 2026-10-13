from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
SIDE_DELTA = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
DIR_ORDER = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 500
CHAIN_LEN_RANGE = (3, 5)
SPRITE_SIZE_RANGE = (3, 5)
CANVAS_SIZE_RANGE = (20, 29)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, 4):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _port_side(r: int, c: int, h: int, w: int) -> str:
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "port must lie on exactly one non-corner bbox edge"
    return sides[0]


def _parse_sprite(grid: np.ndarray, coords: np.ndarray, sprite_id: int) -> dict:
    bbox = _bbox_from_coords(coords)
    r0, c0, _r1, _c1 = bbox
    patch = _crop_bbox(grid, bbox)
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    values = [int(grid[r, c]) for r, c in coords]
    counts = Counter(values)
    body_color, body_count = counts.most_common(1)[0]
    assert sum(v == body_count for v in counts.values()) == 1, "body color must be uniquely dominant"
    assert body_count >= 3, "body must occupy more than the ports"

    ports = []
    filled = patch.copy()
    for color, count in sorted(counts.items()):
        if color == body_color:
            continue
        assert count == 1, "each port must be a single pixel"
        idx = next(i for i, value in enumerate(values) if int(value) == color)
        rr, cc = map(int, rel[idx])
        side = _port_side(rr, cc, *patch.shape)
        filled[rr, cc] = int(body_color)
        ports.append({"color": int(color), "rel": (rr, cc), "side": side})

    assert len(ports) in (1, 2), "each sprite must have one or two ports"
    return {
        "id": int(sprite_id),
        "body_color": int(body_color),
        "bbox": bbox,
        "mask": mask,
        "patch": patch,
        "filled": filled,
        "ports": ports,
    }


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _component_coords(g != BG, connectivity=8)
    assert len(comps) >= 3, "expected at least three sprites"
    sprites = [_parse_sprite(g, coords, idx) for idx, coords in enumerate(comps)]
    body_colors = [sprite["body_color"] for sprite in sprites]
    assert len(body_colors) == len(set(body_colors)), "sprite body colors must be distinct"
    return sprites


def _chain_structure(sprites: list[dict]) -> tuple[list[int], dict[int, list[tuple[int, dict, dict]]]]:
    by_color = {sprite["body_color"]: sprite for sprite in sprites}
    assert len(by_color) == len(sprites), "body colors must be unique"

    links: dict[int, list[tuple[int, dict, dict]]] = {sprite["id"]: [] for sprite in sprites}
    undirected_edges = set()
    for sprite in sprites:
        for port in sprite["ports"]:
            assert port["color"] in by_color, "port must name an existing body color"
            target = by_color[port["color"]]
            assert target["id"] != sprite["id"], "port cannot target its own sprite"
            reciprocal = [p for p in target["ports"] if p["color"] == sprite["body_color"]]
            assert len(reciprocal) == 1, "connections must be mutual and unique"
            in_port = reciprocal[0]
            assert OPPOSITE[port["side"]] == in_port["side"], "mutual ports must lie on opposite sides"
            links[sprite["id"]].append((target["id"], port, in_port))
            undirected_edges.add(tuple(sorted((sprite["id"], target["id"]))))

    degrees = {sprite_id: len(items) for sprite_id, items in links.items()}
    endpoints = [sprite_id for sprite_id, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "expected exactly two chain endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "degrees must define a chain"
    assert len(undirected_edges) == len(sprites) - 1, "expected a single path over the sprites"

    order = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        options = [item for item in links[cur] if item[0] != prev]
        if not options:
            break
        assert len(options) == 1, "chain traversal must be unambiguous"
        nxt = options[0][0]
        order.append(nxt)
        prev, cur = cur, nxt
    assert len(order) == len(sprites), "chain must visit every sprite exactly once"
    return order, links


def _assemble_sprites(sprites: list[dict]) -> np.ndarray:
    order, links = _chain_structure(sprites)
    by_id = {sprite["id"]: sprite for sprite in sprites}

    placements = {order[0]: (0, 0)}
    for cur_id, nxt_id in zip(order, order[1:]):
        cur_links = [item for item in links[cur_id] if item[0] == nxt_id]
        assert len(cur_links) == 1
        _target, out_port, in_port = cur_links[0]
        cur_top_r, cur_top_c = placements[cur_id]
        dr, dc = SIDE_DELTA[out_port["side"]]
        next_top_r = cur_top_r + out_port["rel"][0] + dr - in_port["rel"][0]
        next_top_c = cur_top_c + out_port["rel"][1] + dc - in_port["rel"][1]
        placements[nxt_id] = (int(next_top_r), int(next_top_c))

    filled_boxes = []
    for sprite_id in order:
        top_r, top_c = placements[sprite_id]
        h, w = by_id[sprite_id]["filled"].shape
        filled_boxes.append((top_r, top_c, top_r + h - 1, top_c + w - 1))
    min_r = min(box[0] for box in filled_boxes)
    min_c = min(box[1] for box in filled_boxes)
    max_r = max(box[2] for box in filled_boxes)
    max_c = max(box[3] for box in filled_boxes)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    occupied = set()
    for sprite_id in order:
        sprite = by_id[sprite_id]
        top_r, top_c = placements[sprite_id]
        for rr, cc in np.argwhere(sprite["mask"]):
            ar = int(top_r + rr - min_r)
            ac = int(top_c + cc - min_c)
            assert (ar, ac) not in occupied, "assembled sprites must not overlap"
            occupied.add((ar, ac))
            out[ar, ac] = int(sprite["filled"][rr, cc])

    assert max(out.shape) <= 30, "assembled output must stay within ARC size bounds"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble a scattered chain of monochrome sprites by matching color-named edge ports.

    Input structure:
    - A black background contains several separated multicolor components, each representing one sprite.
    - Every sprite has a unique body color plus one or two single-pixel ports embedded on non-corner bbox edges.
    - A port's color equals the body color of the neighbor sprite it must connect to; exactly two sprites have one port and the rest have two.

    Rule:
    - Treat each component as one sprite, infer its body color as the dominant non-black color, and treat the remaining edge pixels as ports.
    - Starting from either one-port endpoint, follow port colors to the uniquely named neighboring sprites.
    - Translate each next sprite without rotating it so the two mutual ports lie on opposite edges and become 4-neighbor adjacent, then recolor every port pixel to its own sprite's body color.
    - After the whole chain is assembled, return the minimal bounding-box crop of the result.

    Key insight:
    - The off-color edge pixels are not noise; they are neighbor identity labels, and the edge they occupy fixes the only valid seam direction.
    """
    sprites = _parse_sprites(np.asarray(grid, dtype=int))
    return _assemble_sprites(sprites)


def _random_mask(h: int, w: int, forced: set[tuple[int, int]]) -> np.ndarray:
    area = h * w
    low = max(len(forced) + 2, int(np.ceil(area * 0.50)))
    high = max(low, int(np.floor(area * 0.82)))
    target = random.randint(low, high)

    mask = np.ones((h, w), dtype=bool)
    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forced]
    random.shuffle(cells)
    for r, c in cells:
        if int(mask.sum()) <= target:
            break
        mask[r, c] = False
        if not _is_connected(mask):
            mask[r, c] = True
    assert _is_connected(mask)
    assert forced.issubset({tuple(map(int, rc)) for rc in np.argwhere(mask)})
    return mask


def _side_slots(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    raise ValueError(side)


def _choose_directions(length: int) -> list[str]:
    first = random.choice(DIR_ORDER)
    directions = [first]
    for _ in range(length - 2):
        prev = directions[-1]
        turns = [side for side in DIR_ORDER if side not in (prev, OPPOSITE[prev])]
        if random.random() < 0.60:
            directions.append(random.choice(turns))
        else:
            directions.append(prev)
    return directions


def _make_sprite(body_color: int, port_specs: list[tuple[str, int]], sprite_id: int) -> dict:
    for _ in range(80):
        h = random.randint(*SPRITE_SIZE_RANGE)
        w = random.randint(*SPRITE_SIZE_RANGE)

        port_positions: dict[str, tuple[int, int]] = {}
        forced = set()
        for side, _neighbor_color in port_specs:
            slots = _side_slots(side, h, w)
            assert slots, "sprite dimensions must support non-corner ports"
            choices = [slot for slot in slots if slot not in forced]
            port_rc = random.choice(choices)
            port_positions[side] = port_rc
            forced.add(port_rc)

        mask = _random_mask(h, w, forced)
        coords = np.argwhere(mask)
        crop_box = _bbox_from_coords(coords)
        r0, c0, r1, c1 = crop_box
        cropped_mask = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
        if min(cropped_mask.shape) < 3:
            continue

        ports = []
        ok = True
        for side, neighbor_color in port_specs:
            rr, cc = port_positions[side]
            adj = (rr - r0, cc - c0)
            try:
                actual_side = _port_side(adj[0], adj[1], *cropped_mask.shape)
            except AssertionError:
                ok = False
                break
            if actual_side != side:
                ok = False
                break
            ports.append({"color": int(neighbor_color), "rel": adj, "side": side})
        if not ok:
            continue

        patch = np.full(cropped_mask.shape, BG, dtype=int)
        patch[cropped_mask] = int(body_color)
        filled = patch.copy()
        input_patch = patch.copy()
        for port in ports:
            rr, cc = port["rel"]
            input_patch[rr, cc] = int(port["color"])

        return {
            "id": int(sprite_id),
            "body_color": int(body_color),
            "bbox": (0, 0, patch.shape[0] - 1, patch.shape[1] - 1),
            "mask": cropped_mask,
            "patch": input_patch,
            "filled": filled,
            "ports": ports,
        }
    raise RuntimeError("failed to generate a sprite with stable non-corner ports")


def _row_major_chain(body_grid: np.ndarray) -> list[int]:
    comps = _component_coords(body_grid != BG, connectivity=8)
    items = []
    for coords in comps:
        r0, c0, _r1, _c1 = _bbox_from_coords(coords)
        color_counts = Counter(int(body_grid[r, c]) for r, c in coords)
        color = color_counts.most_common(1)[0][0]
        items.append((r0, c0, color))
    items.sort()
    return [color for _r0, _c0, color in items]


def _place_scattered(patches: list[np.ndarray]) -> np.ndarray | None:
    height = random.randint(*CANVAS_SIZE_RANGE)
    width = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((height, width), BG, dtype=int)
    rects: list[tuple[int, int, int, int]] = []

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        random.shuffle(positions)
        chosen = None
        for r, c in positions:
            rect = (r, c, r + ph - 1, c + pw - 1)
            ok = True
            for rr0, cc0, rr1, cc1 in rects:
                if not (rect[2] + 1 < rr0 or rr1 + 1 < rect[0] or rect[3] + 1 < cc0 or cc1 + 1 < rect[1]):
                    ok = False
                    break
            if ok:
                chosen = (r, c, rect)
                break
        if chosen is None:
            return None
        r, c, rect = chosen
        mask = patch != BG
        canvas[r : r + ph, c : c + pw][mask] = patch[mask]
        rects.append(rect)
    return canvas


def _has_turn_or_offset(sprites: list[dict]) -> bool:
    order, links = _chain_structure(sprites)
    directions = []
    for cur_id, nxt_id in zip(order, order[1:]):
        cur_link = [item for item in links[cur_id] if item[0] == nxt_id][0]
        _target, out_port, in_port = cur_link
        directions.append(out_port["side"])
        if out_port["side"] in ("left", "right") and out_port["rel"][0] != in_port["rel"][0]:
            return True
        if out_port["side"] in ("top", "bottom") and out_port["rel"][1] != in_port["rel"][1]:
            return True
    return any(a != b for a, b in zip(directions, directions[1:]))


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(*CHAIN_LEN_RANGE)
    colors = random.sample(COLOR_POOL, count)
    directions = _choose_directions(count)

    sprites = []
    for idx in range(count):
        port_specs = []
        if idx > 0:
            port_specs.append((OPPOSITE[directions[idx - 1]], colors[idx - 1]))
        if idx < count - 1:
            port_specs.append((directions[idx], colors[idx + 1]))
        sprites.append(_make_sprite(colors[idx], port_specs, idx))

    if not _has_turn_or_offset(sprites):
        return None

    try:
        expected = _assemble_sprites(sprites)
    except AssertionError:
        return None
    if max(expected.shape) > 30:
        return None

    scattered = _place_scattered([sprite["patch"] for sprite in sprites])
    if scattered is None:
        return None

    try:
        parsed = _parse_sprites(scattered)
        solved = _assemble_sprites(parsed)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    chain_colors = colors
    row_major = _row_major_chain(np.where(scattered != BG, scattered, 0))
    if row_major == chain_colors or row_major == list(reversed(chain_colors)):
        return None

    if np.array_equal(scattered, expected):
        return None

    return scattered, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _build_instance()
        if sample is not None:
            grid, _expected = sample
            return grid
    raise RuntimeError("failed to generate a valid port-chain puzzle")
