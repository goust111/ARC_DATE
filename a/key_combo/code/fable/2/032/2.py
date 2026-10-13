from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
SIDES = ("top", "right", "bottom", "left")
MAX_GENERATION_ATTEMPTS = 800


@dataclass(frozen=True)
class Port:
    side: str
    offset: int
    coord: Coord


@dataclass(frozen=True)
class Frame:
    bbox: BBox
    fill_color: int
    border_color: int
    ports: tuple[Port, ...]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_multicolor_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    fg = grid != int(bg)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(np.array(coords, dtype=int))
    components.sort(key=lambda pts: bbox_coords(pts))
    return components


def _port_coord(bbox: BBox, side: str, offset: int) -> Coord:
    r0, c0, r1, c1 = map(int, bbox)
    off = int(offset)
    if side == "top":
        return int(r0), int(c0 + off)
    if side == "bottom":
        return int(r1), int(c0 + off)
    if side == "left":
        return int(r0 + off), int(c0)
    if side == "right":
        return int(r0 + off), int(c1)
    raise ValueError(f"unknown side: {side}")


def _port_sort_key(port: Port) -> tuple[int, int, Coord]:
    return SIDES.index(port.side), int(port.offset), port.coord


def _line_coords(a: Coord, b: Coord) -> tuple[Coord, ...]:
    if int(a[0]) == int(b[0]):
        step = 1 if int(b[1]) >= int(a[1]) else -1
        return tuple((int(a[0]), int(c)) for c in range(int(a[1]), int(b[1]) + step, step))
    if int(a[1]) == int(b[1]):
        step = 1 if int(b[0]) >= int(a[0]) else -1
        return tuple((int(r), int(a[1])) for r in range(int(a[0]), int(b[0]) + step, step))
    raise ValueError("line is neither horizontal nor vertical")


def _segment_clear(a: Coord, b: Coord, blocked: np.ndarray, allowed: set[Coord]) -> bool:
    for cell in _line_coords(a, b):
        if cell in allowed:
            continue
        if bool(blocked[int(cell[0]), int(cell[1])]):
            return False
    return True


def _route_candidates(a: Coord, b: Coord, blocked: np.ndarray) -> list[tuple[Coord, ...]]:
    routes: list[tuple[Coord, ...]] = []
    if int(a[0]) == int(b[0]) or int(a[1]) == int(b[1]):
        if _segment_clear(a, b, blocked, {a, b}):
            routes.append(_line_coords(a, b))
    else:
        bends = ((int(a[0]), int(b[1])), (int(b[0]), int(a[1])))
        for bend in bends:
            if bool(blocked[int(bend[0]), int(bend[1])]):
                continue
            allowed = {a, b, bend}
            if not _segment_clear(a, bend, blocked, allowed):
                continue
            if not _segment_clear(bend, b, blocked, allowed):
                continue
            first = _line_coords(a, bend)
            second = _line_coords(bend, b)
            routes.append(first + second[1:])
    deduped: list[tuple[Coord, ...]] = []
    seen: set[tuple[Coord, ...]] = set()
    for route in routes:
        if route not in seen:
            seen.add(route)
            deduped.append(route)
    return deduped


def _bbox_obstacles(frames: list[Frame], shape: tuple[int, int]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame.bbox)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    return blocked


def _parse_frames(grid: np.ndarray) -> tuple[int, list[Frame]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _find_multicolor_components(g, bg)
    assert len(components) >= 3

    frames: list[Frame] = []
    border_colors: set[int] = set()
    sizes: set[int] = set()

    for coords in components:
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        size = int(r1 - r0 + 1)
        assert size == int(c1 - c0 + 1)
        assert size >= 5
        patch = crop_bbox(g, bbox)
        sizes.add(size)

        interior = patch[1:-1, 1:-1]
        inner_colors = set(map(int, interior.flatten()))
        assert len(inner_colors) == 1
        fill_color = int(next(iter(inner_colors)))
        assert fill_color != int(bg)

        perimeter_non_bg: list[int] = []
        for rr in range(size):
            for cc in range(size):
                if rr in (0, size - 1) or cc in (0, size - 1):
                    value = int(patch[rr, cc])
                    if value != int(bg):
                        perimeter_non_bg.append(value)
        assert perimeter_non_bg
        border_color = int(Counter(perimeter_non_bg).most_common(1)[0][0])
        assert set(perimeter_non_bg) == {int(border_color)}
        border_colors.add(border_color)

        for rr in range(1, size - 1):
            for cc in range(1, size - 1):
                assert int(patch[rr, cc]) == int(fill_color)
        for corner in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)):
            assert int(patch[int(corner[0]), int(corner[1])]) == int(border_color)

        ports: list[Port] = []
        for off in range(1, size - 1):
            if int(patch[0, off]) == int(bg):
                assert int(patch[1, off]) == int(fill_color)
                ports.append(Port("top", off, (int(r0), int(c0 + off))))
            else:
                assert int(patch[0, off]) == int(border_color)

            if int(patch[size - 1, off]) == int(bg):
                assert int(patch[size - 2, off]) == int(fill_color)
                ports.append(Port("bottom", off, (int(r1), int(c0 + off))))
            else:
                assert int(patch[size - 1, off]) == int(border_color)

            if int(patch[off, 0]) == int(bg):
                assert int(patch[off, 1]) == int(fill_color)
                ports.append(Port("left", off, (int(r0 + off), int(c0))))
            else:
                assert int(patch[off, 0]) == int(border_color)

            if int(patch[off, size - 1]) == int(bg):
                assert int(patch[off, size - 2]) == int(fill_color)
                ports.append(Port("right", off, (int(r0 + off), int(c1))))
            else:
                assert int(patch[off, size - 1]) == int(border_color)

        assert ports
        frames.append(
            Frame(
                bbox=bbox,
                fill_color=int(fill_color),
                border_color=int(border_color),
                ports=tuple(sorted(ports, key=_port_sort_key)),
            )
        )

    assert len(frames) == len(components)
    assert len(sizes) == 1
    assert len(border_colors) == 1
    return int(bg), sorted(frames, key=lambda frame: (frame.bbox[0], frame.bbox[1]))


def _ordered_frames(frames: list[Frame]) -> list[Frame]:
    ordered = sorted(frames, key=lambda frame: (len(frame.ports), frame.bbox[0], frame.bbox[1]))
    counts = [len(frame.ports) for frame in ordered]
    assert counts == list(range(1, len(ordered) + 1))
    return ordered


def _feasible_bridges(src: Frame, dst: Frame, blocked: np.ndarray) -> list[dict]:
    entries: list[dict] = []
    seen: set[tuple[Coord, Coord, tuple[Coord, ...]]] = set()
    for src_port in src.ports:
        for dst_port in dst.ports:
            for path in _route_candidates(src_port.coord, dst_port.coord, blocked):
                key = (src_port.coord, dst_port.coord, path)
                if key in seen:
                    continue
                seen.add(key)
                kind = "straight" if len({cell[0] for cell in path}) == 1 or len({cell[1] for cell in path}) == 1 else "L"
                entries.append(
                    {
                        "src": src_port,
                        "dst": dst_port,
                        "path": path,
                        "kind": kind,
                    }
                )
    entries.sort(
        key=lambda entry: (
            entry["src"].coord,
            entry["dst"].coord,
            0 if entry["kind"] == "straight" else 1,
            entry["path"],
        )
    )
    return entries


def _choose_bridge(src: Frame, dst: Frame, blocked: np.ndarray) -> dict:
    feasible = _feasible_bridges(src, dst, blocked)
    assert feasible
    grouped: dict[tuple[Coord, Coord], list[dict]] = defaultdict(list)
    for entry in feasible:
        grouped[(entry["src"].coord, entry["dst"].coord)].append(entry)
    assert len(grouped) == 1
    only_group = next(iter(grouped.values()))
    return min(only_group, key=lambda entry: (0 if entry["kind"] == "straight" else 1, entry["path"]))


def _ordered_bridge_info(grid: np.ndarray) -> tuple[list[Frame], list[dict]]:
    _bg, frames = _parse_frames(grid)
    ordered = _ordered_frames(frames)
    blocked = _bbox_obstacles(frames, grid.shape)
    bridges = [_choose_bridge(ordered[i], ordered[i + 1], blocked) for i in range(len(ordered) - 1)]
    return ordered, bridges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect framed squares in notch-count order using the unique bridgeable notch pair between each consecutive pair.

    Input structure:
    - The background is the dominant color.
    - Several equal-sized square objects are scattered apart; each object has a single border color and a unique solid inner fill color.
    - Some border cells are missing as one-cell notches, exposing the background at the frame perimeter.

    Rule:
    - Count the notches on each framed square and visit the squares in increasing notch count order 1..N.
    - For each consecutive pair in that order, find the unique pair of notches that can be connected by a one-cell bridge made of a straight segment or one right-angle turn, while treating every square bbox as blocked except for the chosen endpoint notches.
    - Draw each bridge in the previous square's inner color, filling the two used notches while leaving every unused notch open.

    Key insight:
    - The notches carry two independent meanings at once: their count fixes the global visitation order, and their side/position fixes the local bridge ports.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    ordered, bridges = _ordered_bridge_info(g)
    for index, bridge in enumerate(bridges):
        color = int(ordered[index].fill_color)
        for r, c in bridge["path"]:
            out[int(r), int(c)] = int(color)
    return out


def _all_port_specs(size: int) -> list[tuple[str, int]]:
    return [(side, off) for side in SIDES for off in range(1, int(size) - 1)]


def _render_frame(size: int, border_color: int, fill_color: int, ports: set[tuple[str, int]], bg: int) -> np.ndarray:
    patch = np.full((size, size), int(fill_color), dtype=int)
    patch[0, :] = int(border_color)
    patch[-1, :] = int(border_color)
    patch[:, 0] = int(border_color)
    patch[:, -1] = int(border_color)
    for side, offset in ports:
        rr, cc = _port_coord((0, 0, size - 1, size - 1), side, offset)
        patch[int(rr), int(cc)] = int(bg)
    return patch


def _paint_path(grid: np.ndarray, path: tuple[Coord, ...], color: int) -> None:
    for r, c in path:
        grid[int(r), int(c)] = int(color)


def _relation_kind(a: Coord, b: Coord) -> str:
    if int(a[0]) == int(b[0]) or int(a[1]) == int(b[1]):
        return "straight"
    return "L"


def _pair_descriptor(src_bbox: BBox, dst_bbox: BBox) -> dict:
    src_r0, src_c0, _src_r1, _src_c1 = src_bbox
    dst_r0, dst_c0, _dst_r1, _dst_c1 = dst_bbox
    if int(src_r0) == int(dst_r0):
        if int(dst_c0) > int(src_c0):
            return {"kind": "straight-h", "src_side": "right", "dst_side": "left"}
        return {"kind": "straight-h", "src_side": "left", "dst_side": "right"}
    if int(src_c0) == int(dst_c0):
        if int(dst_r0) > int(src_r0):
            return {"kind": "straight-v", "src_side": "bottom", "dst_side": "top"}
        return {"kind": "straight-v", "src_side": "top", "dst_side": "bottom"}

    if random.random() < 0.5:
        return {
            "kind": "diag-hv",
            "src_side": "right" if int(dst_c0) > int(src_c0) else "left",
            "dst_side": "top" if int(dst_r0) > int(src_r0) else "bottom",
        }
    return {
        "kind": "diag-vh",
        "src_side": "bottom" if int(dst_r0) > int(src_r0) else "top",
        "dst_side": "left" if int(dst_c0) > int(src_c0) else "right",
    }


def _assign_planned_bridges(bboxes: list[BBox], size: int) -> tuple[list[set[tuple[str, int]]], list[tuple[Coord, ...]], list[str]] | None:
    blocked = np.zeros((27, 27), dtype=bool)
    for bbox in bboxes:
        r0, c0, r1, c1 = map(int, bbox)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    used_specs = [set() for _ in bboxes]
    paths: list[tuple[Coord, ...]] = []
    path_cells: set[Coord] = set()
    kinds: list[str] = []
    offsets = list(range(1, int(size) - 1))

    for index in range(len(bboxes) - 1):
        descriptor = _pair_descriptor(bboxes[index], bboxes[index + 1])
        candidates: list[tuple[tuple[str, int], tuple[str, int], tuple[Coord, ...], str]] = []

        if descriptor["kind"] in {"straight-h", "straight-v"}:
            random.shuffle(offsets)
            for offset in offsets:
                src_key = (descriptor["src_side"], int(offset))
                dst_key = (descriptor["dst_side"], int(offset))
                if src_key in used_specs[index] or dst_key in used_specs[index + 1]:
                    continue
                src_port = _port_coord(bboxes[index], src_key[0], src_key[1])
                dst_port = _port_coord(bboxes[index + 1], dst_key[0], dst_key[1])
                routes = _route_candidates(src_port, dst_port, blocked)
                if len(routes) != 1:
                    continue
                path = routes[0]
                if any(cell in path_cells for cell in path):
                    continue
                candidates.append((src_key, dst_key, path, "straight"))
        else:
            src_offsets = list(range(1, int(size) - 1))
            dst_offsets = list(range(1, int(size) - 1))
            random.shuffle(src_offsets)
            random.shuffle(dst_offsets)
            for src_offset in src_offsets:
                src_key = (descriptor["src_side"], int(src_offset))
                if src_key in used_specs[index]:
                    continue
                for dst_offset in dst_offsets:
                    dst_key = (descriptor["dst_side"], int(dst_offset))
                    if dst_key in used_specs[index + 1]:
                        continue
                    src_port = _port_coord(bboxes[index], src_key[0], src_key[1])
                    dst_port = _port_coord(bboxes[index + 1], dst_key[0], dst_key[1])
                    routes = _route_candidates(src_port, dst_port, blocked)
                    if len(routes) != 1:
                        continue
                    path = routes[0]
                    if any(cell in path_cells for cell in path):
                        continue
                    candidates.append((src_key, dst_key, path, "L"))

        if not candidates:
            return None
        src_key, dst_key, path, kind = random.choice(candidates)
        used_specs[index].add(src_key)
        used_specs[index + 1].add(dst_key)
        paths.append(path)
        path_cells.update(path)
        kinds.append(kind)

    return used_specs, paths, kinds


def _render_instance(
    bboxes: list[BBox],
    size: int,
    bg: int,
    border_color: int,
    fill_colors: list[int],
    port_specs: list[set[tuple[str, int]]],
) -> np.ndarray:
    grid = np.full((27, 27), int(bg), dtype=int)
    for index, bbox in enumerate(bboxes):
        r0, c0, _r1, _c1 = bbox
        patch = _render_frame(size, border_color, fill_colors[index], port_specs[index], bg)
        grid[r0 : r0 + size, c0 : c0 + size] = patch
    return grid


def _paint_expected_output(
    grid: np.ndarray,
    paths: list[tuple[Coord, ...]],
    fill_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for index, path in enumerate(paths):
        _paint_path(out, path, fill_colors[index])
    return out


def _crop_pair(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != int(bg))
    r0, c0, r1, c1 = bbox_coords(coords)
    top = int(r0 - random.randint(0, min(3, r0)))
    left = int(c0 - random.randint(0, min(3, c0)))
    bottom = int(r1 + random.randint(0, min(3, inp.shape[0] - 1 - r1)))
    right = int(c1 + random.randint(0, min(3, inp.shape[1] - 1 - c1)))
    return (
        inp[top : bottom + 1, left : right + 1].copy(),
        out[top : bottom + 1, left : right + 1].copy(),
    )


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        _bg, frames = _parse_frames(grid)
        ordered = _ordered_frames(frames)
        positions = [(frame.bbox[0], frame.bbox[1]) for frame in ordered]
        row_major = sorted(positions)
        col_major = sorted(positions, key=lambda pos: (pos[1], pos[0]))
        if positions in (row_major, row_major[::-1], col_major, col_major[::-1]):
            return False

        blocked = _bbox_obstacles(frames, grid.shape)
        bridge_info: list[dict] = []
        styles: list[str] = []
        used_ports: set[Coord] = set()
        for index in range(len(ordered) - 1):
            feasible = _feasible_bridges(ordered[index], ordered[index + 1], blocked)
            if len(feasible) != 1:
                return False
            bridge = feasible[0]
            bridge_info.append(bridge)
            styles.append(str(bridge["kind"]))
            used_ports.add(bridge["src"].coord)
            used_ports.add(bridge["dst"].coord)

        if "straight" not in styles or "L" not in styles:
            return False
        if not any(port.coord not in used_ports for frame in ordered for port in frame.ports):
            return False
        return True
    except AssertionError:
        return False


def _choose_layout(size: int, count: int) -> list[BBox] | None:
    anchors = [1, 10, 19]
    slots = [(r, c) for r in anchors for c in anchors]
    for _ in range(200):
        chosen = random.sample(slots, count)
        relation_kinds = [_relation_kind(chosen[i], chosen[i + 1]) for i in range(count - 1)]
        if "straight" not in relation_kinds or "L" not in relation_kinds:
            continue
        row_major = sorted(chosen)
        col_major = sorted(chosen, key=lambda pos: (pos[1], pos[0]))
        if chosen in (row_major, row_major[::-1], col_major, col_major[::-1]):
            continue
        return [(r, c, r + size - 1, c + size - 1) for r, c in chosen]
    return None


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([5, 6])
    count = random.choice([3, 4])
    bboxes = _choose_layout(size, count)
    if bboxes is None:
        return None

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    border_color = random.choice(palette)
    palette.remove(border_color)
    fill_colors = random.sample(palette, count)

    assigned = _assign_planned_bridges(bboxes, size)
    if assigned is None:
        return None
    base_ports, paths, kinds = assigned
    if "straight" not in kinds or "L" not in kinds:
        return None

    all_specs = _all_port_specs(size)
    for _ in range(250):
        port_specs = [set(specs) for specs in base_ports]
        valid = True
        for index in range(count):
            need = index + 1 - len(port_specs[index])
            if need < 0:
                valid = False
                break
            pool = [spec for spec in all_specs if spec not in port_specs[index]]
            if need > len(pool):
                valid = False
                break
            extras = random.sample(pool, need)
            port_specs[index].update(extras)
        if not valid:
            continue

        inp = _render_instance(bboxes, size, bg, border_color, fill_colors, port_specs)
        expected = _paint_expected_output(inp, paths, fill_colors)
        cropped_inp, cropped_expected = _crop_pair(inp, expected, bg)

        if not _validate_semantics(cropped_inp):
            continue
        try:
            produced = main(cropped_inp)
        except AssertionError:
            continue
        if np.array_equal(produced, cropped_expected):
            return cropped_inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
