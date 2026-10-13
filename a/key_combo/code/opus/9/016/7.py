from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_VEC: dict[str, Coord] = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
MARKER_COLORS = {1, 2, 3, 4}
WIRE = 5
TEMPLATE = 8
BG = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_from_mask(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _degree_in_component(cell: Coord, comp: set[Coord]) -> int:
    r, c = cell
    return sum((nr, nc) in comp for nr, nc in _neighbors4(r, c))


def _rot_from_east(offset: Coord, direction: str) -> Coord:
    dr, dc = map(int, offset)
    if direction == "E":
        return dr, dc
    if direction == "S":
        return dc, -dr
    if direction == "W":
        return -dr, -dc
    if direction == "N":
        return -dc, dr
    raise ValueError(direction)


def _template_offsets(cells: set[Coord], anchor_row: int) -> list[Coord]:
    return sorted((int(r - anchor_row), int(c + 2)) for r, c in cells)


def _parse_networks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    teal_comps = _components_from_mask(g == TEMPLATE)
    teal_owner: dict[Coord, int] = {}
    for idx, comp in enumerate(teal_comps):
        for cell in comp:
            teal_owner[cell] = idx

    wire_comps = _components_from_mask(g == WIRE)
    networks: list[dict] = []
    used_templates: set[int] = set()

    for comp in wire_comps:
        endpoints = sorted(cell for cell in comp if _degree_in_component(cell, comp) == 1)
        assert len(endpoints) == 2, "every wire network must be a single path with two endpoints"

        endpoint_info = []
        for ep in endpoints:
            markers = []
            for nr, nc in _neighbors4(*ep):
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) in MARKER_COLORS:
                    markers.append((nr, nc))
            assert len(markers) == 1, "each endpoint must have exactly one adjacent color marker"
            marker = markers[0]
            adjacent_templates = {
                teal_owner[(nr, nc)]
                for nr, nc in _neighbors4(*marker)
                if (nr, nc) in teal_owner
            }
            assert len(adjacent_templates) <= 1, "marker must touch at most one template"
            endpoint_info.append(
                {
                    "endpoint": ep,
                    "marker": marker,
                    "marker_color": int(g[marker]),
                    "template_ids": adjacent_templates,
                }
            )

        source_infos = [info for info in endpoint_info if info["template_ids"]]
        assert len(source_infos) == 1, "exactly one endpoint must be the source"
        src = source_infos[0]
        dst = endpoint_info[0] if endpoint_info[1] is src else endpoint_info[1]
        template_id = next(iter(src["template_ids"]))
        assert template_id not in used_templates, "source templates must pair with exactly one wire"
        used_templates.add(template_id)
        template_cells = teal_comps[template_id]
        offsets = sorted((r - src["endpoint"][0], c - src["endpoint"][1]) for r, c in template_cells)

        networks.append(
            {
                "wire": frozenset(comp),
                "source_endpoint": tuple(src["endpoint"]),
                "source_marker": tuple(src["marker"]),
                "template_cells": frozenset(template_cells),
                "offsets": offsets,
                "remote_endpoint": tuple(dst["endpoint"]),
                "remote_marker": tuple(dst["marker"]),
                "remote_color": int(dst["marker_color"]),
            }
        )

    assert len(used_templates) == len(teal_comps), "every template must belong to one source endpoint"
    networks.sort(key=lambda item: tuple(sorted(item["wire"]))[0])
    return networks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit each cyan template across its grey wire and stamp it at the far endpoint using the far-end marker color.

    Input structure:
    - The black background contains 2-3 independent grey single-pixel 4-connected wire paths, each with two endpoints.
    - Every endpoint has one adjacent color marker pixel chosen from blue/red/green/yellow.
    - Exactly one endpoint of each wire is a source: its marker touches a connected cyan template placed beyond the endpoint.

    Rule:
    - For each grey wire, identify its two endpoints and find the source endpoint by the adjacent cyan template.
    - Record the cyan template's cell offsets relative to the source endpoint, then translate that same offset set to the opposite endpoint.
    - Stamp the translated template in the far-end marker color, but only onto black cells; do not overwrite grey wire cells or existing colored markers.
    - Erase the source template and the source marker, keep the far-end marker, and leave all grey wires unchanged.

    Key insight:
    - The wire determines where the template moves, while the far endpoint's local marker determines only the new color; the template itself is copied by pure translation from endpoint to endpoint.
    """
    g = np.asarray(grid, dtype=int)
    networks = _parse_networks(g)
    out = g.copy()

    for net in networks:
        sr, sc = net["source_marker"]
        out[sr, sc] = BG
        for r, c in net["template_cells"]:
            out[r, c] = BG

    for net in networks:
        rr0, cc0 = net["remote_endpoint"]
        color = int(net["remote_color"])
        for dr, dc in net["offsets"]:
            rr = rr0 + int(dr)
            cc = cc0 + int(dc)
            if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == BG and int(out[rr, cc]) == BG:
                out[rr, cc] = color

    return out


def _random_with_gap(lo: int, hi: int, count: int, gap: int) -> list[int]:
    values = list(range(int(lo), int(hi) + 1))
    for _ in range(200):
        random.shuffle(values)
        picked: list[int] = []
        for value in values:
            if all(abs(value - other) >= int(gap) for other in picked):
                picked.append(int(value))
                if len(picked) == count:
                    return sorted(picked)
    if count == 1:
        return [random.choice(values)]
    if hi - lo < (count - 1) * gap:
        raise ValueError("coordinate range is too small for requested spacing")
    for start in range(int(lo), int(hi) + 1):
        picked = [start + i * int(gap) for i in range(count)]
        if picked[-1] <= int(hi):
            return picked
    raise ValueError("could not pick spaced coordinates")


def _make_template_shape() -> tuple[set[Coord], int]:
    for _ in range(300):
        h = random.choice((3, 4))
        w = random.choice((3, 4))
        anchor_row = random.randrange(h)
        area = random.randint(4, min(7, h * w - 1))
        cells: set[Coord] = {(anchor_row, 0), (anchor_row, 1)}
        while len(cells) < area:
            frontier: set[Coord] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))

        if len(cells) != area:
            continue
        if all(r == anchor_row for r, _ in cells):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return cells, anchor_row
    raise RuntimeError("failed to build a non-symmetric template")


def _endpoint_allowed_dirs(orientation: str, endpoint_side: str) -> tuple[list[str], str]:
    if orientation == "h":
        if endpoint_side == "left":
            return ["N", "S", "W"], "W"
        return ["N", "S", "E"], "E"
    if endpoint_side == "top":
        return ["W", "E", "N"], "N"
    return ["W", "E", "S"], "S"


def _source_dir_candidates(allowed: list[str], outward: str, mode: str) -> list[str]:
    orthogonal = [d for d in allowed if d != outward]
    if mode == "outward":
        return [outward] + orthogonal
    if mode == "orthogonal":
        return orthogonal + [outward]
    pool = [outward, outward] + orthogonal + orthogonal[:]
    random.shuffle(pool)
    seen: list[str] = []
    for item in pool:
        if item not in seen:
            seen.append(item)
    for item in allowed:
        if item not in seen:
            seen.append(item)
    return seen


def _paint_cells(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _make_network_spec(
    shape: tuple[int, int],
    orientation: str,
    axis_pos: int,
    occupied_input: set[Coord],
    reserved_future: set[Coord],
    source_mode: str,
) -> dict:
    h, w = map(int, shape)
    for _ in range(500):
        if orientation == "h":
            left_hi = min(8, w - 14)
            if left_hi < 5:
                continue
            left_c = random.randint(5, left_hi)
            right_lo = left_c + 8
            right_hi = w - 6
            if right_lo > right_hi:
                continue
            right_c = random.randint(right_lo, right_hi)
            left_ep = (axis_pos, left_c)
            right_ep = (axis_pos, right_c)
            source_side = random.choice(("left", "right"))
            source_ep = left_ep if source_side == "left" else right_ep
            remote_ep = right_ep if source_side == "left" else left_ep
            remote_side = "right" if source_side == "left" else "left"
            wire_cells = {(axis_pos, c) for c in range(left_c, right_c + 1)}
        else:
            top_hi = min(8, h - 14)
            if top_hi < 5:
                continue
            top_r = random.randint(5, top_hi)
            bottom_lo = top_r + 8
            bottom_hi = h - 6
            if bottom_lo > bottom_hi:
                continue
            bottom_r = random.randint(bottom_lo, bottom_hi)
            top_ep = (top_r, axis_pos)
            bottom_ep = (bottom_r, axis_pos)
            source_side = random.choice(("top", "bottom"))
            source_ep = top_ep if source_side == "top" else bottom_ep
            remote_ep = bottom_ep if source_side == "top" else top_ep
            remote_side = "bottom" if source_side == "top" else "top"
            wire_cells = {(r, axis_pos) for r in range(top_r, bottom_r + 1)}

        allowed_dirs, outward_dir = _endpoint_allowed_dirs(orientation, source_side)
        remote_dirs, _ = _endpoint_allowed_dirs(orientation, remote_side)
        template_cells, anchor_row = _make_template_shape()
        base_offsets = _template_offsets(template_cells, anchor_row)

        for src_dir in _source_dir_candidates(allowed_dirs, outward_dir, source_mode):
            rotated_offsets = [_rot_from_east(offset, src_dir) for offset in base_offsets]
            src_template = {
                (source_ep[0] + dr, source_ep[1] + dc)
                for dr, dc in rotated_offsets
            }
            future_dest = {
                (remote_ep[0] + dr, remote_ep[1] + dc)
                for dr, dc in rotated_offsets
            }
            src_marker = (
                source_ep[0] + DIR_VEC[src_dir][0],
                source_ep[1] + DIR_VEC[src_dir][1],
            )

            if not _in_bounds(shape, *src_marker):
                continue
            if any(not _in_bounds(shape, r, c) for r, c in src_template | future_dest):
                continue
            if src_marker in wire_cells or src_template & wire_cells:
                continue

            remote_dir_order = remote_dirs[:]
            random.shuffle(remote_dir_order)
            for remote_dir in remote_dir_order:
                remote_marker = (
                    remote_ep[0] + DIR_VEC[remote_dir][0],
                    remote_ep[1] + DIR_VEC[remote_dir][1],
                )
                if not _in_bounds(shape, *remote_marker):
                    continue
                if remote_marker in wire_cells:
                    continue

                input_cells = set(wire_cells) | set(src_template) | {src_marker, remote_marker}
                future_nonwire = future_dest - set(wire_cells)
                if input_cells & occupied_input:
                    continue
                if input_cells & reserved_future:
                    continue
                if future_nonwire & occupied_input:
                    continue
                if future_nonwire & reserved_future:
                    continue
                if future_nonwire & (input_cells - set(wire_cells)):
                    continue

                palette = list(MARKER_COLORS)
                src_color = random.choice(palette)
                dst_color = random.choice([c for c in palette if c != src_color])
                overlap_on_wire = bool(future_dest & set(wire_cells))

                return {
                    "wire_cells": frozenset(wire_cells),
                    "source_endpoint": tuple(source_ep),
                    "remote_endpoint": tuple(remote_ep),
                    "source_marker": tuple(src_marker),
                    "remote_marker": tuple(remote_marker),
                    "source_color": int(src_color),
                    "remote_color": int(dst_color),
                    "template_cells": frozenset(src_template),
                    "offsets": tuple(rotated_offsets),
                    "future_dest": frozenset(future_dest),
                    "overlap_on_wire": overlap_on_wire,
                    "orientation": orientation,
                }

    raise RuntimeError("failed to place one network")


def _spec_key(spec: dict) -> frozenset[Coord]:
    return frozenset(spec["wire_cells"] if "wire_cells" in spec else spec["wire"])


def _verify_scene(grid: np.ndarray, specs: list[dict]) -> None:
    parsed = _parse_networks(grid)
    assert len(parsed) == len(specs), "scene must decode into the intended number of networks"
    parsed_by_wire = {_spec_key(item): item for item in parsed}
    spec_by_wire = {_spec_key(item): item for item in specs}
    assert parsed_by_wire.keys() == spec_by_wire.keys(), "wire pairings must be uniquely recoverable"

    for key, spec in spec_by_wire.items():
        parsed_item = parsed_by_wire[key]
        assert parsed_item["source_endpoint"] == spec["source_endpoint"]
        assert parsed_item["remote_endpoint"] == spec["remote_endpoint"]
        assert parsed_item["source_marker"] == spec["source_marker"]
        assert parsed_item["remote_marker"] == spec["remote_marker"]
        assert parsed_item["remote_color"] == spec["remote_color"]
        assert parsed_item["template_cells"] == spec["template_cells"]

    out = main(grid)
    for spec in specs:
        for r, c in spec["wire_cells"]:
            assert int(out[r, c]) == WIRE, "wire must remain unchanged"
        sr, sc = spec["source_marker"]
        assert int(out[sr, sc]) == BG, "source marker must be erased"
        rr, rc = spec["remote_marker"]
        assert int(out[rr, rc]) == spec["remote_color"], "remote marker must remain"
        for r, c in spec["template_cells"]:
            assert int(out[r, c]) == BG, "source template must be erased"
        for r, c in spec["future_dest"]:
            if int(grid[r, c]) == BG:
                assert int(out[r, c]) == spec["remote_color"], "black destination cells must be stamped"
            else:
                assert int(out[r, c]) == int(grid[r, c]), "non-black destination cells must not be overwritten"


def generate_input() -> np.ndarray:
    for _ in range(400):
        count = random.choice((2, 3))
        orientation = random.choice(("h", "v"))
        if orientation == "h":
            height = random.randint(19, 28) if count == 2 else random.randint(27, 30)
            width = random.randint(20, 30)
            axes = _random_with_gap(5, height - 6, count, 8)
        else:
            height = random.randint(20, 30)
            width = random.randint(19, 28) if count == 2 else random.randint(27, 30)
            axes = _random_with_gap(5, width - 6, count, 8)

        source_modes = ["outward"]
        if count >= 2:
            source_modes.append("orthogonal")
        while len(source_modes) < count:
            source_modes.append("any")
        random.shuffle(source_modes)

        specs: list[dict] = []
        occupied_input: set[Coord] = set()
        reserved_future: set[Coord] = set()
        ok = True
        try:
            for axis_pos, source_mode in zip(axes, source_modes):
                spec = _make_network_spec(
                    (height, width),
                    orientation,
                    axis_pos,
                    occupied_input,
                    reserved_future,
                    source_mode,
                )
                specs.append(spec)
                occupied_input |= set(spec["wire_cells"])
                occupied_input |= set(spec["template_cells"])
                occupied_input.add(spec["source_marker"])
                occupied_input.add(spec["remote_marker"])
                reserved_future |= set(spec["future_dest"]) - set(spec["wire_cells"])
        except RuntimeError:
            ok = False

        if not ok:
            continue
        if not any(spec["overlap_on_wire"] for spec in specs):
            continue
        if not any(not spec["overlap_on_wire"] for spec in specs):
            continue

        grid = np.zeros((height, width), dtype=int)
        for spec in specs:
            _paint_cells(grid, set(spec["wire_cells"]), WIRE)
            _paint_cells(grid, set(spec["template_cells"]), TEMPLATE)
            grid[spec["source_marker"]] = spec["source_color"]
            grid[spec["remote_marker"]] = spec["remote_color"]

        try:
            _verify_scene(grid, specs)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
