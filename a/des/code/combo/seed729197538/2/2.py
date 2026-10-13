import random
from typing import Iterable

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {step: idx for idx, step in enumerate(DIRS)}
RIGHT_DIR = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rotate_vec_cw(dr: int, dc: int, quarter_turns: int) -> tuple[int, int]:
    quarter_turns %= 4
    for _ in range(quarter_turns):
        dr, dc = dc, -dr
    return dr, dc


def _rotate_cells(cells: Iterable[tuple[int, int]], quarter_turns: int) -> tuple[tuple[int, int], ...]:
    return tuple(_rotate_vec_cw(dr, dc, quarter_turns) for dr, dc in cells)


def main(grid: np.ndarray) -> np.ndarray:
    """Teleport each gray object to its paired portal and rotate it by portal-orientation difference."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    bg = _get_bg(g)

    non_bg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert non_bg_colors, "expected foreground colors"

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    multi_cell_colors: list[int] = []
    for color in non_bg_colors:
        comps = _components(g == color)
        comps_by_color[color] = comps
        if any(len(comp) > 1 for comp in comps):
            multi_cell_colors.append(color)

    if len(multi_cell_colors) == 1:
        object_color = multi_cell_colors[0]
    else:
        non_pair_colors = [color for color in non_bg_colors if int(np.count_nonzero(g == color)) != 2]
        assert len(non_pair_colors) == 1, "could not isolate the gray object color"
        object_color = non_pair_colors[0]

    object_components = comps_by_color[object_color]
    assert object_components, "expected at least one gray object"

    object_index = np.full((height, width), -1, dtype=int)
    for obj_id, comp in enumerate(object_components):
        for r, c in comp:
            object_index[r, c] = obj_id

    portal_records_by_color: dict[int, list[dict[str, object]]] = {}
    portal_by_object: list[tuple[int, tuple[int, int], int] | None] = [None] * len(object_components)

    for color in non_bg_colors:
        if color == object_color:
            continue
        portal_positions = [tuple(map(int, pos)) for pos in np.argwhere(g == color)]
        assert len(portal_positions) == 2, "each portal color must appear exactly twice"

        records: list[dict[str, object]] = []
        for pr, pc in portal_positions:
            touches: list[tuple[int, int]] = []
            for dr, dc in DIRS:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < height and 0 <= nc < width and object_index[nr, nc] >= 0:
                    obj_id = int(object_index[nr, nc])
                    entry_dir = DIR_TO_INDEX[(-dr, -dc)]
                    touches.append((obj_id, entry_dir))
            assert len(touches) == 1, "each portal must touch exactly one object cell on one side"

            obj_id, entry_dir = touches[0]
            assert portal_by_object[obj_id] is None, "each object must touch exactly one portal"
            portal_by_object[obj_id] = (color, (pr, pc), entry_dir)
            records.append({"obj_id": obj_id, "pos": (pr, pc), "dir": entry_dir})

        portal_records_by_color[color] = records

    assert all(record is not None for record in portal_by_object), "every object needs a portal"

    out = np.full_like(g, bg)
    for obj_id, comp in enumerate(object_components):
        color, src_portal, src_dir = portal_by_object[obj_id]  # type: ignore[misc]
        pair = portal_records_by_color[color]
        dst = pair[1] if int(pair[0]["obj_id"]) == obj_id else pair[0]
        dst_portal = dst["pos"]
        dst_dir = int(dst["dir"])
        turns = (dst_dir - src_dir) % 4
        spr, spc = src_portal
        dpr, dpc = dst_portal

        for r, c in comp:
            rel_r, rel_c = int(r) - spr, int(c) - spc
            rot_r, rot_c = _rotate_vec_cw(rel_r, rel_c, turns)
            rr, cc = dpr + rot_r, dpc + rot_c
            assert 0 <= rr < height and 0 <= cc < width, "teleported object left the canvas"
            assert out[rr, cc] in (bg, object_color), "teleported objects should not overlap"
            out[rr, cc] = object_color

    return out


GEN_BG = 0
GEN_OBJECT = 5
GEN_PORTAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
GEN_PAIR_RANGE = (1, 4)
GEN_AREA_RANGE = (2, 7)
GEN_SPAN = 2
GEN_DEPTH = 4

CANONICAL_DOMAIN = tuple(
    (dr, dc)
    for dr in range(-GEN_SPAN, GEN_SPAN + 1)
    for dc in range(-GEN_DEPTH, 0)
)
CANONICAL_DOMAIN_SET = set(CANONICAL_DOMAIN)
RIGHT_SLOT = CANONICAL_DOMAIN


def _neighbors_in_domain(cell: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = cell
    result: list[tuple[int, int]] = []
    for sdr, sdc in DIRS:
        nxt = (dr + sdr, dc + sdc)
        if nxt in CANONICAL_DOMAIN_SET:
            result.append(nxt)
    return result


DOMAIN_NEIGHBORS = {cell: _neighbors_in_domain(cell) for cell in CANONICAL_DOMAIN}
SLOT_RELS = {dir_idx: _rotate_cells(RIGHT_SLOT, (dir_idx - RIGHT_DIR) % 4) for dir_idx in range(4)}

BUFFER_RELS: dict[int, tuple[tuple[int, int], ...]] = {}
for dir_idx, slot in SLOT_RELS.items():
    base = set(slot)
    base.add((0, 0))
    buf = {
        (dr + edr, dc + edc)
        for dr, dc in base
        for edr in (-1, 0, 1)
        for edc in (-1, 0, 1)
    }
    BUFFER_RELS[dir_idx] = tuple(buf)


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    target_area = rng.randint(*GEN_AREA_RANGE)
    cells = {(0, -1)}
    frontier = set(DOMAIN_NEIGHBORS[(0, -1)])
    while frontier and len(cells) < target_area:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in DOMAIN_NEIGHBORS[cell]:
            if nbr not in cells:
                frontier.add(nbr)
    if len(cells) < 2:
        return None
    return tuple(sorted(cells))


def _translate_cells(cells: Iterable[tuple[int, int]], anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = anchor
    return tuple((ar + dr, ac + dc) for dr, dc in cells)


def _buffer_cells(height: int, width: int, anchor: tuple[int, int], dir_idx: int) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {
        (ar + dr, ac + dc)
        for dr, dc in BUFFER_RELS[dir_idx]
        if 0 <= ar + dr < height and 0 <= ac + dc < width
    }


def _fits_slot(height: int, width: int, anchor: tuple[int, int], dir_idx: int, reserved: set[tuple[int, int]]) -> bool:
    ar, ac = anchor
    if not (0 <= ar < height and 0 <= ac < width):
        return False
    for dr, dc in SLOT_RELS[dir_idx]:
        rr, cc = ar + dr, ac + dc
        if not (0 <= rr < height and 0 <= cc < width):
            return False
    return not (_buffer_cells(height, width, anchor, dir_idx) & reserved)


def _place_endpoints(height: int, width: int, endpoint_count: int, rng: random.Random) -> list[dict[str, object]] | None:
    endpoints: list[dict[str, object]] = []
    reserved: set[tuple[int, int]] = set()
    for _ in range(endpoint_count):
        candidates: list[tuple[int, int, int]] = []
        for dir_idx in rng.sample(range(4), 4):
            for r in range(height):
                for c in range(width):
                    if _fits_slot(height, width, (r, c), dir_idx, reserved):
                        candidates.append((dir_idx, r, c))
        if not candidates:
            return None

        dir_idx, pr, pc = rng.choice(candidates)
        canonical = _random_shape(rng)
        if canonical is None:
            return None
        rel_cells = _rotate_cells(canonical, (dir_idx - RIGHT_DIR) % 4)
        abs_cells = _translate_cells(rel_cells, (pr, pc))

        endpoints.append(
            {
                "dir": dir_idx,
                "portal": (pr, pc),
                "rel_cells": rel_cells,
                "abs_cells": abs_cells,
            }
        )
        reserved.update(_buffer_cells(height, width, (pr, pc), dir_idx))

    return endpoints


def _pair_endpoints(endpoint_count: int, rng: random.Random) -> list[tuple[int, int]]:
    indices = list(range(endpoint_count))
    rng.shuffle(indices)
    return [(indices[i], indices[i + 1]) for i in range(0, endpoint_count, 2)]


def _build_input(height: int, width: int, endpoints: list[dict[str, object]], pairs: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    inp = np.full((height, width), GEN_BG, dtype=int)
    for endpoint in endpoints:
        for r, c in endpoint["abs_cells"]:
            inp[r, c] = GEN_OBJECT
    for color, (a, b) in zip(colors, pairs):
        ar, ac = endpoints[a]["portal"]
        br, bc = endpoints[b]["portal"]
        inp[ar, ac] = color
        inp[br, bc] = color
    return inp


def _build_expected(height: int, width: int, endpoints: list[dict[str, object]], pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((height, width), GEN_BG, dtype=int)
    partner: dict[int, int] = {}
    for a, b in pairs:
        partner[a] = b
        partner[b] = a

    for idx, endpoint in enumerate(endpoints):
        src_portal = endpoint["portal"]
        src_dir = int(endpoint["dir"])
        dst_endpoint = endpoints[partner[idx]]
        dst_portal = dst_endpoint["portal"]
        dst_dir = int(dst_endpoint["dir"])
        turns = (dst_dir - src_dir) % 4

        spr, spc = src_portal
        dpr, dpc = dst_portal
        for r, c in endpoint["abs_cells"]:
            rel_r, rel_c = int(r) - spr, int(c) - spc
            rot_r, rot_c = _rotate_vec_cw(rel_r, rel_c, turns)
            rr, cc = dpr + rot_r, dpc + rot_c
            if not (0 <= rr < height and 0 <= cc < width):
                raise AssertionError("generated teleport left the canvas")
            if out[rr, cc] not in (GEN_BG, GEN_OBJECT):
                raise AssertionError("generated teleports overlapped")
            out[rr, cc] = GEN_OBJECT
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    pair_count = rng.randint(*GEN_PAIR_RANGE)
    endpoint_count = 2 * pair_count

    base_side = 14 + 3 * pair_count
    height = rng.randint(base_side, base_side + 6)
    width = rng.randint(base_side, base_side + 6)

    endpoints = _place_endpoints(height, width, endpoint_count, rng)
    if endpoints is None:
        return None

    pairs = _pair_endpoints(endpoint_count, rng)
    portal_colors = rng.sample(list(GEN_PORTAL_COLORS), pair_count)
    inp = _build_input(height, width, endpoints, pairs, portal_colors)
    expected = _build_expected(height, width, endpoints, pairs)

    if np.count_nonzero(inp == GEN_OBJECT) <= 2:
        return None
    if _get_bg(inp) != GEN_BG:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport puzzle input")
