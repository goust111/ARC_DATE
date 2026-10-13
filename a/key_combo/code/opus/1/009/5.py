from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BACKGROUND = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 400


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(coords: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return sorted((int(r - r0), int(c - c0)) for r, c in coords)


def _component_from_seed(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    q = deque([start])
    seen[start] = True
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((int(r), int(c)))
        for dr, dc in DIRS4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not bool(seen[rr, cc]):
                seen[rr, cc] = True
                q.append((rr, cc))
    return comp


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and not bool(seen[r, c]):
                comps.append(_component_from_seed(mask, (r, c), seen))
    return comps


def _body_connected(body_coords: list[tuple[int, int]]) -> bool:
    if not body_coords:
        return False
    r0, c0, r1, c1 = _bbox(body_coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in body_coords:
        mask[int(r - r0), int(c - c0)] = True
    return len(_components(mask)) == 1


def _attachment_direction(body_set: set[tuple[int, int]], connector: tuple[int, int]) -> tuple[int, int]:
    touching = []
    for dr, dc in DIRS4:
        rr, cc = int(connector[0] - dr), int(connector[1] - dc)
        if (rr, cc) in body_set:
            touching.append((int(dr), int(dc)))
    if len(touching) != 1:
        raise AssertionError("connector must attach to the body at exactly one side")
    return touching[0]


def _infer_direction(body_coords: list[tuple[int, int]], connector: tuple[int, int]) -> tuple[int, int]:
    center = np.mean(np.asarray(body_coords, dtype=float), axis=0)
    dr = float(connector[0]) - float(center[0])
    dc = float(connector[1]) - float(center[1])
    if abs(dr) > abs(dc):
        return (_sign(dr), 0)
    if abs(dc) > abs(dr):
        return (0, _sign(dc))
    return _attachment_direction(set(body_coords), connector)


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    comps = _components(g != BACKGROUND)
    sprites: list[dict] = []
    for comp in comps:
        values = np.array([int(g[r, c]) for r, c in comp], dtype=int)
        colors, counts = np.unique(values, return_counts=True)
        if len(colors) != 2:
            raise AssertionError("each sprite must contain one body color and one connector color")
        singleton = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
        if len(singleton) != 1:
            raise AssertionError("each sprite must have exactly one singleton connector pixel")

        connector_color = int(singleton[0])
        body_color = int([int(color) for color in colors if int(color) != connector_color][0])
        connector = next((int(r), int(c)) for r, c in comp if int(g[r, c]) == connector_color)
        body_coords = [(int(r), int(c)) for r, c in comp if int(g[r, c]) == body_color]
        if len(body_coords) < 3 or not _body_connected(body_coords):
            raise AssertionError("body must be a 4-connected shape of area at least 3")

        sprites.append(
            {
                "body_color": body_color,
                "connector_color": connector_color,
                "connector": connector,
                "body_coords": body_coords,
                "body_set": set(body_coords),
                "full_coords": [(int(r), int(c)) for r, c in comp],
                "full_set": set((int(r), int(c)) for r, c in comp),
                "area": int(len(body_coords)),
                "direction": _infer_direction(body_coords, connector),
            }
        )

    if not (4 <= len(sprites) <= 6):
        raise AssertionError("expected 4 to 6 sprites")
    body_colors = [int(sprite["body_color"]) for sprite in sprites]
    if len(set(body_colors)) != len(body_colors):
        raise AssertionError("all sprite body colors must be distinct")
    return sprites


def _pair_sprites(sprites: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        grouped[int(sprite["connector_color"])].append(sprite)

    if not (2 <= len(grouped) <= 3):
        raise AssertionError("expected 2 to 3 connector-color pairs")

    pairs: list[dict] = []
    for connector_color, group in sorted(grouped.items()):
        if len(group) != 2:
            raise AssertionError("each connector color must belong to exactly two sprites")
        first, second = group
        if int(first["area"]) == int(second["area"]):
            raise AssertionError("paired sprites must have different body areas")
        anchor, mover = (first, second) if int(first["area"]) > int(second["area"]) else (second, first)
        pairs.append(
            {
                "connector_color": int(connector_color),
                "anchor": anchor,
                "mover": mover,
            }
        )
    return pairs


def _find_translation(grid_shape: tuple[int, int], mover: dict, anchor: dict, occupied: set[tuple[int, int]]) -> int:
    height, width = map(int, grid_shape)
    dr, dc = map(int, mover["direction"])
    blocked = set(occupied) - set(mover["full_set"])

    for steps in range(1, max(height, width) + 1):
        moved_full = _translate(list(mover["full_coords"]), steps * dr, steps * dc)
        if not all(0 <= r < height and 0 <= c < width for r, c in moved_full):
            break
        if set(moved_full) & blocked:
            break

        moved_connector = (int(mover["connector"][0] + steps * dr), int(mover["connector"][1] + steps * dc))
        if _manhattan(moved_connector, anchor["connector"]) == 1:
            return int(steps)

    raise AssertionError("mover could not dock at the matching anchor connector")


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    sprites = _parse_sprites(g)
    pairs = _pair_sprites(sprites)
    occupied = {(int(r), int(c)) for r, c in np.argwhere(g != BACKGROUND)}

    placements: list[dict] = []
    for pair in pairs:
        mover = pair["mover"]
        anchor = pair["anchor"]
        steps = _find_translation(g.shape, mover, anchor, occupied)
        dr, dc = map(int, mover["direction"])
        moved_body = _translate(list(mover["body_coords"]), steps * dr, steps * dc)
        moved_connector = (int(mover["connector"][0] + steps * dr), int(mover["connector"][1] + steps * dc))
        placements.append(
            {
                "connector_color": int(pair["connector_color"]),
                "anchor": anchor,
                "mover": mover,
                "steps": int(steps),
                "moved_body": moved_body,
                "moved_connector": moved_connector,
            }
        )

    out = g.copy()
    for placement in placements:
        for r, c in placement["mover"]["full_coords"]:
            out[int(r), int(c)] = BACKGROUND

    for placement in placements:
        anchor = placement["anchor"]
        out[int(anchor["connector"][0]), int(anchor["connector"][1])] = int(anchor["body_color"])
        for r, c in placement["moved_body"]:
            out[int(r), int(c)] = int(placement["mover"]["body_color"])
        out[int(placement["moved_connector"][0]), int(placement["moved_connector"][1])] = int(anchor["body_color"])

    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair sprites by connector color, slide the smaller one straight into docking contact, and repaint both connectors to the larger sprite's color.

    Input structure:
    - The grid is a black background with 4 to 6 disjoint 4-connected multicolor sprites.
    - Each sprite has a monochrome body of area 3 to 8 in one unique color plus exactly one differently colored connector pixel on its boundary.
    - Each connector color appears on exactly two sprites, defining 2 to 3 pairs.

    Rule:
    - Pair sprites that share the same connector color.
    - In each pair, compare body areas excluding the connector pixel; the larger sprite is the stationary anchor and the smaller sprite is the mover.
    - Infer the mover's slide direction from the connector's cardinal side relative to the mover body centroid, then translate the whole mover one cell at a time along that straight line until the two matching connectors become 4-neighbors.
    - Recolor both pair connectors to the anchor body color, keep the anchor fixed, erase the mover from its original location, and leave the moved body in its original body color.

    Key insight:
    - The shared connector color only decides who pairs with whom; the actual moving role comes from geometric area comparison, and the slide stops on connector adjacency rather than on ordinary shape collision.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _grow_body(area: int) -> list[tuple[int, int]] | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                rr, cc = int(r + dr), int(c + dc)
                if abs(rr) <= 3 and abs(cc) <= 3 and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    body = _normalize(list(cells))
    rows = {int(r) for r, _c in body}
    cols = {int(c) for _r, c in body}
    if len(rows) < 2 or len(cols) < 2:
        return None
    if _bbox_area(body) == len(body):
        return None
    return body


def _connector_candidates(body: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    body_set = set(body)
    center = np.mean(np.asarray(body, dtype=float), axis=0)
    dr, dc = map(int, direction)
    ranked: list[tuple[tuple[float, float], tuple[int, int]]] = []

    for br, bc in body:
        rr, cc = int(br + dr), int(bc + dc)
        if (rr, cc) in body_set:
            continue
        touching = sum((int(rr + adr), int(cc + adc)) in body_set for adr, adc in DIRS4)
        if touching != 1:
            continue
        vr = float(rr) - float(center[0])
        vc = float(cc) - float(center[1])
        if dr != 0:
            if _sign(vr) != dr or abs(vr) <= abs(vc):
                continue
            ranked.append(((abs(vc), -(vr * dr)), (rr, cc)))
        else:
            if _sign(vc) != dc or abs(vc) <= abs(vr):
                continue
            ranked.append(((abs(vr), -(vc * dc)), (rr, cc)))

    ranked.sort(key=lambda item: item[0])
    return [coord for _key, coord in ranked]


def _make_sprite(area: int, direction: tuple[int, int]) -> dict | None:
    for _ in range(200):
        body = _grow_body(area)
        if body is None:
            continue
        candidates = _connector_candidates(body, direction)
        if not candidates:
            continue
        connector = candidates[0]
        full = sorted(body + [connector])
        return {
            "body_coords": body,
            "connector": connector,
            "full_coords": full,
            "area": int(area),
            "bbox_area": int(_bbox_area(body)),
            "direction": (int(direction[0]), int(direction[1])),
        }
    return None


def _shift_groups(groups: list[list[tuple[int, int]]], margin: int = 1) -> list[list[tuple[int, int]]]:
    min_r = min(int(r) for group in groups for r, _c in group)
    min_c = min(int(c) for group in groups for _r, c in group)
    dr = int(margin - min_r)
    dc = int(margin - min_c)
    return [_translate(group, dr, dc) for group in groups]


def _expand_halo(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    halo = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((int(r + dr), int(c + dc)))
    return sorted(halo)


def _build_pair(
    direction: tuple[int, int],
    mover_color: int,
    anchor_color: int,
    connector_color: int,
    require_close_gap: bool,
    require_bbox_counterexample: bool,
) -> dict | None:
    dr, dc = map(int, direction)
    for _ in range(250):
        mover_area = random.randint(3, 6)
        if require_close_gap:
            diff = random.choice((1, 1, 2))
            anchor_area = mover_area + diff
            if anchor_area > 8:
                continue
        else:
            anchor_area = random.randint(mover_area + 1, 8)

        mover = _make_sprite(mover_area, direction)
        anchor = _make_sprite(anchor_area, (-dr, -dc))
        if mover is None or anchor is None:
            continue
        if require_bbox_counterexample and int(mover["bbox_area"]) < int(anchor["bbox_area"]):
            continue

        separation = random.randint(4, 8)
        mover_target = (0, 0)
        anchor_target = (int(dr * separation), int(dc * separation))

        mover_shifted = _translate(list(mover["full_coords"]), mover_target[0] - int(mover["connector"][0]), mover_target[1] - int(mover["connector"][1]))
        mover_body_shifted = _translate(list(mover["body_coords"]), mover_target[0] - int(mover["connector"][0]), mover_target[1] - int(mover["connector"][1]))
        anchor_shifted = _translate(list(anchor["full_coords"]), anchor_target[0] - int(anchor["connector"][0]), anchor_target[1] - int(anchor["connector"][1]))
        anchor_body_shifted = _translate(list(anchor["body_coords"]), anchor_target[0] - int(anchor["connector"][0]), anchor_target[1] - int(anchor["connector"][1]))
        anchor_connector_shifted = anchor_target

        steps = separation - 1
        moved_final = _translate(mover_shifted, steps * dr, steps * dc)
        moved_final_body = _translate(mover_body_shifted, steps * dr, steps * dc)
        moved_final_connector = (int(steps * dr), int(steps * dc))
        if _manhattan(moved_final_connector, anchor_connector_shifted) != 1:
            continue
        if set(moved_final) & set(anchor_shifted):
            continue
        if any(set(_translate(mover_shifted, t * dr, t * dc)) & set(anchor_shifted) for t in range(steps)):
            continue

        sweep = set(anchor_shifted)
        for t in range(steps + 1):
            sweep.update(_translate(mover_shifted, t * dr, t * dc))

        groups = _shift_groups(
            [
                mover_shifted,
                mover_body_shifted,
                [mover_target],
                anchor_shifted,
                anchor_body_shifted,
                [anchor_connector_shifted],
                moved_final_body,
                [moved_final_connector],
                sorted(sweep),
            ],
            margin=1,
        )
        (
            mover_shifted,
            mover_body_shifted,
            mover_connector_shifted,
            anchor_shifted,
            anchor_body_shifted,
            anchor_connector_shifted_list,
            moved_final_body,
            moved_final_connector_list,
            sweep_list,
        ) = groups
        anchor_connector_shifted = anchor_connector_shifted_list[0]
        mover_connector_shifted = mover_connector_shifted[0]
        moved_final_connector = moved_final_connector_list[0]

        return {
            "direction": (dr, dc),
            "connector_color": int(connector_color),
            "mover_color": int(mover_color),
            "anchor_color": int(anchor_color),
            "mover_area": int(mover_area),
            "anchor_area": int(anchor_area),
            "mover_bbox_area": int(mover["bbox_area"]),
            "anchor_bbox_area": int(anchor["bbox_area"]),
            "steps": int(steps),
            "mover_full": mover_shifted,
            "mover_body": mover_body_shifted,
            "mover_connector": mover_connector_shifted,
            "anchor_full": anchor_shifted,
            "anchor_body": anchor_body_shifted,
            "anchor_connector": anchor_connector_shifted,
            "moved_body": moved_final_body,
            "moved_connector": moved_final_connector,
            "sweep": sweep_list,
            "halo": _expand_halo(sweep_list),
        }
    return None


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _globalize_pair(pair: dict, offset: tuple[int, int]) -> dict:
    dr, dc = map(int, offset)
    coord_keys = {
        "mover_full",
        "mover_body",
        "mover_connector",
        "anchor_full",
        "anchor_body",
        "anchor_connector",
        "moved_body",
        "moved_connector",
        "sweep",
        "halo",
    }
    result = {}
    for key, value in pair.items():
        if key in coord_keys and isinstance(value, list):
            result[key] = _translate(value, dr, dc)
        elif key in coord_keys and isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, int) for x in value):
            result[key] = (int(value[0] + dr), int(value[1] + dc))
        else:
            result[key] = value
    return result


def _place_pairs(pair_templates: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    pair_count = len(pair_templates)
    base = 16 if pair_count == 2 else 20
    height = random.randint(base, min(30, base + 5))
    width = random.randint(base, min(30, base + 5))
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    for template in sorted(pair_templates, key=lambda pair: len(pair["halo"]), reverse=True):
        halo = template["halo"]
        max_r = max(int(r) for r, _c in halo)
        max_c = max(int(c) for _r, c in halo)
        candidates = []
        for r0 in range(height - max_r):
            for c0 in range(width - max_c):
                if all(not bool(reserved[int(r + r0), int(c + c0)]) for r, c in halo):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None

        offset = random.choice(candidates)
        pair = _globalize_pair(template, offset)
        placed.append(pair)
        for r, c in pair["halo"]:
            reserved[int(r), int(c)] = True

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for pair in placed:
        _paint_coords(grid, pair["mover_body"], int(pair["mover_color"]))
        grid[int(pair["mover_connector"][0]), int(pair["mover_connector"][1])] = int(pair["connector_color"])
        _paint_coords(grid, pair["anchor_body"], int(pair["anchor_color"]))
        grid[int(pair["anchor_connector"][0]), int(pair["anchor_connector"][1])] = int(pair["connector_color"])
    return grid, placed


def _expected_from_metadata(shape: tuple[int, int], placed_pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for pair in placed_pairs:
        _paint_coords(out, pair["anchor_body"], int(pair["anchor_color"]))
        out[int(pair["anchor_connector"][0]), int(pair["anchor_connector"][1])] = int(pair["anchor_color"])
        _paint_coords(out, pair["moved_body"], int(pair["mover_color"]))
        out[int(pair["moved_connector"][0]), int(pair["moved_connector"][1])] = int(pair["anchor_color"])
    return out


def _nearest_partner_is_not_always_true(placed_pairs: list[dict]) -> bool:
    sprites = []
    for idx, pair in enumerate(placed_pairs):
        sprites.append((idx, "mover", pair["mover_connector"]))
        sprites.append((idx, "anchor", pair["anchor_connector"]))
    for idx, kind, connector in sprites:
        others = []
        for other_idx, other_kind, other_connector in sprites:
            if other_idx == idx and other_kind == kind:
                continue
            others.append((_manhattan(connector, other_connector), other_idx))
        if others and min(others)[1] != idx:
            return True
    return False


def _validate_generated_case(grid: np.ndarray, placed_pairs: list[dict]) -> None:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        raise AssertionError("generated grid must stay within ARC bounds")

    if len(placed_pairs) not in (2, 3):
        raise AssertionError("generator must build 2 or 3 pairs")
    directions = {tuple(map(int, pair["direction"])) for pair in placed_pairs}
    axes = {0 if int(direction[0]) != 0 else 1 for direction in directions}
    if axes != {0, 1}:
        raise AssertionError("each generated sample must include both horizontal and vertical motion")
    if not any(int(pair["anchor_area"]) - int(pair["mover_area"]) <= 2 for pair in placed_pairs):
        raise AssertionError("at least one pair should require close area comparison")
    if not any(int(pair["mover_bbox_area"]) >= int(pair["anchor_bbox_area"]) for pair in placed_pairs):
        raise AssertionError("at least one pair should defeat bbox-size shortcuts")

    connector_colors = [int(pair["connector_color"]) for pair in placed_pairs]
    if len(set(connector_colors)) != len(connector_colors):
        raise AssertionError("connector colors must be unique across pairs")
    body_colors = [int(pair["mover_color"]) for pair in placed_pairs] + [int(pair["anchor_color"]) for pair in placed_pairs]
    if len(set(body_colors)) != len(body_colors):
        raise AssertionError("body colors must be globally unique")

    parsed = _parse_sprites(g)
    paired = _pair_sprites(parsed)
    if len(paired) != len(placed_pairs):
        raise AssertionError("parser and generator disagree on pair count")

    expected = _expected_from_metadata(g.shape, placed_pairs)
    actual = main(g)
    if not np.array_equal(actual, expected):
        raise AssertionError("metadata oracle and solver output disagree")

    if len(placed_pairs) == 3 and not _nearest_partner_is_not_always_true(placed_pairs):
        raise AssertionError("three-pair cases should include at least one misleading nearer non-partner")

    for pair in placed_pairs:
        for r, c in pair["mover_full"]:
            if int(actual[r, c]) != BACKGROUND:
                raise AssertionError("mover origin must be erased")
        if int(actual[int(pair["anchor_connector"][0]), int(pair["anchor_connector"][1])]) != int(pair["anchor_color"]):
            raise AssertionError("anchor connector must be repainted to anchor color")
        if int(actual[int(pair["moved_connector"][0]), int(pair["moved_connector"][1])]) != int(pair["anchor_color"]):
            raise AssertionError("moved connector must be repainted to anchor color")


def _sample_directions(pair_count: int) -> list[tuple[int, int]]:
    horizontal = random.choice(((0, 1), (0, -1)))
    vertical = random.choice(((1, 0), (-1, 0)))
    if pair_count == 2:
        return [horizontal, vertical]
    remaining = [direction for direction in ((0, 1), (0, -1), (1, 0), (-1, 0)) if direction not in {horizontal, vertical}]
    third = random.choice(remaining)
    directions = [horizontal, vertical, third]
    random.shuffle(directions)
    return directions


def _gen_attempt() -> tuple[np.ndarray, list[dict]] | None:
    pair_count = 3 if random.random() < 0.7 else 2
    colors = list(ALL_COLORS)
    random.shuffle(colors)
    body_colors = colors[: 2 * pair_count]
    connector_colors = colors[2 * pair_count : 3 * pair_count]
    directions = _sample_directions(pair_count)

    pair_templates: list[dict] = []
    for index in range(pair_count):
        pair = _build_pair(
            direction=directions[index],
            mover_color=int(body_colors[2 * index]),
            anchor_color=int(body_colors[2 * index + 1]),
            connector_color=int(connector_colors[index]),
            require_close_gap=index == 0,
            require_bbox_counterexample=index == 0,
        )
        if pair is None:
            return None
        pair_templates.append(pair)

    placed = _place_pairs(pair_templates)
    if placed is None:
        return None
    grid, placed_pairs = placed
    try:
        _validate_generated_case(grid, placed_pairs)
    except Exception:
        return None
    return grid, placed_pairs


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        generated = _gen_attempt()
        if generated is None:
            continue
        grid, _placed_pairs = generated
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
