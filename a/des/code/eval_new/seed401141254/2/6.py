from __future__ import annotations

import numpy as np

BG = 0
GREY = 5
AZURE = 8
MAROON = 9
BODY_COLORS = (1, 2, 3, 4, 6, 7)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _normalize(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}, (int(r0), int(c0))


def _shift(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _expanded(coords: set[tuple[int, int]], pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                out.add((int(rr), int(cc)))
    return out


def _decode_direction(cells: dict[tuple[int, int], int]) -> tuple[int, int]:
    maroons = [pos for pos, color in cells.items() if int(color) == MAROON]
    greys = [pos for pos, color in cells.items() if int(color) == GREY]
    assert len(maroons) == 1, "each region must contain exactly one maroon head"
    assert len(greys) == 2, "each region must contain exactly two grey tail cells"
    head = maroons[0]
    adjacent = [pos for pos in greys if abs(pos[0] - head[0]) + abs(pos[1] - head[1]) == 1]
    assert len(adjacent) == 1, "maroon head must touch the center grey cell"
    line = [head] + greys
    rows = sorted(r for r, _ in line)
    cols = sorted(c for _, c in line)
    assert (len(set(rows)) == 1 and cols[2] - cols[0] == 2) or (
        len(set(cols)) == 1 and rows[2] - rows[0] == 2
    ), "arrow cells must form a contiguous 1x3 or 3x1 line"
    nr, nc = adjacent[0]
    dr = int(head[0] - nr)
    dc = int(head[1] - nc)
    assert abs(dr) + abs(dc) == 1
    return dr, dc


def _parse_objects(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components(grid != BG)
    objects: list[dict[str, object]] = []
    for comp in comps:
        colors = {coord: int(grid[coord]) for coord in comp}
        body_colors = {color for color in colors.values() if color not in (GREY, MAROON)}
        assert len(body_colors) == 1, "each region must have one body color plus the arrow colors"
        direction = _decode_direction(colors)
        bbox = _bbox(comp)
        objects.append(
            {
                "cells": dict(colors),
                "area": int(len(comp)),
                "dir": direction,
                "bbox": tuple(map(int, bbox)),
            }
        )
    assert objects, "expected at least one movable region"
    return objects


def _simulate_objects(
    objects: list[dict[str, object]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    h, w = map(int, shape)
    objs: list[dict[str, object]] = []
    for obj in objects:
        objs.append(
            {
                "cells": dict(obj["cells"]),
                "area": int(obj["area"]),
                "dir": tuple(obj["dir"]),
                "bbox": tuple(obj["bbox"]),
            }
        )

    order = sorted(range(len(objs)), key=lambda i: (objs[i]["area"], objs[i]["bbox"]))
    stats = [
        {"embedded": 0, "clipped": 0, "translated": 0, "final_inbounds": 0}
        for _ in range(len(objs))
    ]

    for pos, idx in enumerate(order):
        dr, dc = objs[idx]["dir"]
        distance = int(objs[idx]["area"])
        translated: dict[tuple[int, int], int] = {}
        for (r, c), color in objs[idx]["cells"].items():
            translated[(int(r) + distance * int(dr), int(c) + distance * int(dc))] = int(color)

        visible: dict[tuple[int, int], int] = {}
        stats[idx]["translated"] = int(len(translated))
        stats[idx]["clipped"] = int(
            sum(1 for (r, c) in translated if not (0 <= r < h and 0 <= c < w))
        )
        later = order[pos + 1 :]
        for coord, color in translated.items():
            overlapped = False
            for other in later:
                if coord in objs[other]["cells"]:
                    objs[other]["cells"][coord] = AZURE
                    overlapped = True
            if overlapped:
                stats[idx]["embedded"] = int(stats[idx]["embedded"]) + 1
            else:
                visible[coord] = int(color)
        objs[idx]["cells"] = visible

    out = np.full((h, w), BG, dtype=int)
    coord_counts: dict[tuple[int, int], int] = {}
    for idx in order:
        inbounds = 0
        for (r, c), color in objs[idx]["cells"].items():
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = int(color)
                coord_counts[(int(r), int(c))] = coord_counts.get((int(r), int(c)), 0) + 1
                inbounds += 1
        stats[idx]["final_inbounds"] = int(inbounds)

    meta = {
        "order": list(order),
        "stats": stats,
        "duplicates": int(sum(1 for count in coord_counts.values() if count > 1)),
        "azure_count": int(np.count_nonzero(out == AZURE)),
    }
    return out, objs, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move arrow-marked colored regions by their own areas and embed smaller movers into larger regions they hit.

    Input structure:
    - Background is black.
    - Several disconnected non-black regions appear; each region has one body color.
    - Inside each region, exactly three cells form an arrow: two grey cells and one maroon end cell.
    - The maroon cell is the arrow head, so the 1x3 or 3x1 arrow points up, down, left, or right.

    Rule:
    - Read each region's movement direction from its internal arrow.
    - Count each region's full area, including the grey and maroon arrow cells, and sort regions from smaller to larger.
    - Process them in that order: translate each region by exactly its own area in its arrow direction.
    - If a moved smaller region lands on cells of a larger region that has not moved yet, those overlapping cells become azure and are absorbed into the larger region, so they move again later with that larger region; non-overlapping cells keep their original colors. Pixels moved outside the grid are discarded.

    Key insight:
    - Three mechanisms interact at once: the maroon arrow head encodes direction, area encodes distance, and size order determines which overlaps become azure embeddings inside later-moving larger regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _parse_objects(g)
    out, _objs, _meta = _simulate_objects(objects, g.shape)
    return out


def _random_shape(
    area: int,
    direction: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    horizontal = direction[0] == 0
    if horizontal:
        cells = {(1, 0), (1, 1), (1, 2)}
        arrow = {(1, 0), (1, 1), (1, 2)}
    else:
        cells = {(0, 1), (1, 1), (2, 1)}
        arrow = {(0, 1), (1, 1), (2, 1)}

    for _ in range(200):
        cur = set(cells)
        while len(cur) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cur:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cur:
                        continue
                    trial = set(cur)
                    trial.add((int(nr), int(nc)))
                    norm, _ = _normalize(trial)
                    r0, c0, r1, c1 = _bbox(norm)
                    if r1 - r0 + 1 <= 5 and c1 - c0 + 1 <= 5:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            center_r = float(sum(r for r, _ in cur)) / len(cur)
            center_c = float(sum(c for _, c in cur)) / len(cur)
            frontier.sort(
                key=lambda pos: (
                    abs(pos[0] - center_r) + abs(pos[1] - center_c),
                    pos[0],
                    pos[1],
                )
            )
            keep = frontier[: max(3, min(8, len(frontier)))]
            choice = keep[int(rng.integers(0, len(keep)))]
            cur.add(choice)

        norm_cells, (off_r, off_c) = _normalize(cur)
        norm_arrow = {(r - off_r, c - off_c) for r, c in arrow}
        if horizontal and rng.random() < 0.5:
            max_r = max(r for r, _ in norm_cells)
            norm_cells = {(max_r - r, c) for r, c in norm_cells}
            norm_arrow = {(max_r - r, c) for r, c in norm_arrow}
        if rng.random() < 0.5:
            max_r = max(r for r, _ in norm_cells)
            max_c = max(c for _, c in norm_cells)
            norm_cells = {(max_r - r, max_c - c) for r, c in norm_cells}
            norm_arrow = {(max_r - r, max_c - c) for r, c in norm_arrow}

        r0, c0, r1, c1 = _bbox(norm_cells)
        if horizontal:
            if (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 3:
                return norm_cells, norm_arrow
        else:
            if (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 2:
                return norm_cells, norm_arrow
    raise AssertionError("failed to grow a valid region shape")


def _head_cell(
    arrow_cells: set[tuple[int, int]],
    direction: tuple[int, int],
) -> tuple[int, int]:
    dr, dc = direction
    if dr == -1:
        return min(arrow_cells)
    if dr == 1:
        return max(arrow_cells)
    if dc == -1:
        return min(arrow_cells, key=lambda p: (p[1], p[0]))
    return max(arrow_cells, key=lambda p: (p[1], p[0]))


def _build_object(
    top: int,
    left: int,
    shape_cells: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    direction: tuple[int, int],
    color: int,
) -> dict[str, object]:
    head = _head_cell(arrow_cells, direction)
    cells: dict[tuple[int, int], int] = {}
    for r, c in shape_cells:
        abs_pos = (int(top + r), int(left + c))
        if (r, c) in arrow_cells:
            cells[abs_pos] = MAROON if (r, c) == head else GREY
        else:
            cells[abs_pos] = int(color)
    bbox = _bbox(list(cells.keys()))
    return {
        "cells": cells,
        "area": int(len(shape_cells)),
        "dir": tuple(direction),
        "bbox": tuple(map(int, bbox)),
    }


def _translated_coords(obj: dict[str, object]) -> set[tuple[int, int]]:
    dr, dc = obj["dir"]
    step = int(obj["area"])
    return {
        (int(r) + step * int(dr), int(c) + step * int(dc))
        for r, c in obj["cells"].keys()
    }


def _place_random(
    h: int,
    w: int,
    shape_cells: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    direction: tuple[int, int],
    color: int,
    occupied_buffer: set[tuple[int, int]],
    require_clip: bool | None,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    _, _, max_r, max_c = _bbox(shape_cells)
    rows = list(range(0, h - max_r))
    cols = list(range(0, w - max_c))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for top in rows:
        for left in cols:
            abs_coords = {(int(top + r), int(left + c)) for r, c in shape_cells}
            if abs_coords & occupied_buffer:
                continue
            obj = _build_object(top, left, shape_cells, arrow_cells, direction, color)
            translated = _translated_coords(obj)
            clip = any(not (0 <= r < h and 0 <= c < w) for r, c in translated)
            inbounds = sum(1 for r, c in translated if 0 <= r < h and 0 <= c < w)
            if require_clip is True and (not clip or inbounds < 3):
                continue
            if require_clip is False and clip:
                continue
            return obj
    return None


def _place_with_overlap(
    h: int,
    w: int,
    shape_cells: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    direction: tuple[int, int],
    color: int,
    occupied_buffer: set[tuple[int, int]],
    target: dict[str, object],
    others: list[dict[str, object]],
    rng: np.random.Generator,
) -> dict[str, object] | None:
    _, _, max_r, max_c = _bbox(shape_cells)
    candidates: list[dict[str, object]] = []
    target_cells = set(target["cells"].keys())
    other_cells = set()
    for other in others:
        other_cells.update(other["cells"].keys())
    for top in range(0, h - max_r):
        for left in range(0, w - max_c):
            abs_coords = {(int(top + r), int(left + c)) for r, c in shape_cells}
            if abs_coords & occupied_buffer:
                continue
            obj = _build_object(top, left, shape_cells, arrow_cells, direction, color)
            translated = _translated_coords(obj)
            overlap = translated & target_cells
            if not (1 <= len(overlap) <= max(1, min(4, len(shape_cells) - 2))):
                continue
            if translated & other_cells:
                continue
            non_overlap_inbounds = sum(
                1
                for r, c in translated - overlap
                if 0 <= r < h and 0 <= c < w
            )
            if non_overlap_inbounds < 2:
                continue
            if any(coord in target_cells for coord in abs_coords):
                continue
            candidates.append(obj)
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _input_from_objects(
    h: int,
    w: int,
    objects: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for (r, c), color in obj["cells"].items():
            grid[r, c] = int(color)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 29))
    w = int(rng.integers(18, 29))
    count = int(rng.integers(3, 5))
    areas = sorted(map(int, rng.choice(np.arange(6, 15), size=count, replace=False)))
    directions = [DIRS[int(rng.integers(0, len(DIRS)))] for _ in range(count)]
    colors = list(map(int, rng.choice(np.array(BODY_COLORS), size=count, replace=False)))

    specs: list[dict[str, object]] = []
    for area, direction, color in zip(areas, directions, colors):
        shape_cells, arrow_cells = _random_shape(area, direction, rng)
        specs.append(
            {
                "area": int(area),
                "dir": tuple(direction),
                "color": int(color),
                "shape_cells": shape_cells,
                "arrow_cells": arrow_cells,
            }
        )

    want_overlap = bool(rng.random() < 0.55)
    source_idx = None
    target_idx = None
    if want_overlap:
        source_idx = int(rng.integers(0, count - 1))
        target_idx = int(rng.integers(source_idx + 1, count))

    spare = [i for i in range(count) if i not in {source_idx, target_idx}]
    want_clip = bool(rng.random() < 0.5)
    clip_idx = None
    if want_clip:
        if spare:
            clip_idx = int(spare[int(rng.integers(0, len(spare)))])
        else:
            clip_idx = int(rng.integers(0, count))

    placed: dict[int, dict[str, object]] = {}
    occupied_buffer: set[tuple[int, int]] = set()

    for idx in range(count - 1, -1, -1):
        spec = specs[idx]
        require_clip = True if idx == clip_idx else False
        if idx == source_idx:
            assert target_idx is not None
            others = [placed[j] for j in placed if j != target_idx]
            obj = _place_with_overlap(
                h,
                w,
                spec["shape_cells"],
                spec["arrow_cells"],
                spec["dir"],
                spec["color"],
                occupied_buffer,
                placed[target_idx],
                others,
                rng,
            )
        else:
            obj = _place_random(
                h,
                w,
                spec["shape_cells"],
                spec["arrow_cells"],
                spec["dir"],
                spec["color"],
                occupied_buffer,
                require_clip,
                rng,
            )
        if obj is None:
            return None
        placed[idx] = obj
        occupied_buffer |= _expanded(set(obj["cells"].keys()), pad=1)

    objects = [placed[i] for i in range(count)]
    inp = _input_from_objects(h, w, objects)

    parsed = _parse_objects(inp)
    if sorted(int(obj["area"]) for obj in parsed) != areas:
        return None

    out, sim_objs, meta = _simulate_objects(parsed, inp.shape)
    stats = meta["stats"]
    if meta["duplicates"] != 0:
        return None

    azure_count = int(meta["azure_count"])
    if want_overlap and azure_count == 0:
        return None
    if not want_overlap and azure_count != 0:
        return None

    total_clipped = int(sum(int(item["clipped"]) for item in stats))
    if want_clip and total_clipped == 0:
        return None
    if not want_clip and total_clipped != 0:
        return None

    if np.count_nonzero(out != BG) < max(10, sum(areas) // 2):
        return None

    for item in stats:
        contributed = int(item["final_inbounds"]) + int(item["embedded"])
        if contributed < 2:
            return None

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
