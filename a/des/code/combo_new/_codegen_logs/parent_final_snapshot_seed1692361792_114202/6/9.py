from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8]
ANCHOR_COLOR_POOL = [1, 2, 3, 4, 6]

LAYOUT_CHOICES = [
    [1],
    [2],
    [1, 1],
    [1, 2],
    [2, 1],
    [1, 1, 1],
    [2, 2],
    [1, 1, 2],
    [1, 2, 1],
    [2, 1, 1],
]
LAYOUT_WEIGHTS = [1, 1, 3, 3, 3, 2, 2, 2, 2, 2]

MAX_GENERATION_ATTEMPTS = 400
FRAME_PLACEMENT_ATTEMPTS = 150
DESTINATION_ATTEMPTS = 180
SOURCE_ATTEMPTS = 220
SHAPE_ATTEMPTS = 120
FRAME_MARGIN = 1
FRAME_GAP = 3


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _neighbors8(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[Coord] | set[Coord] | tuple[Coord, ...]) -> BBox:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_frame(grid: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON


def _segment_cells(a: Coord, b: Coord) -> list[Coord] | None:
    if a[0] == b[0]:
        step = 1 if a[1] <= b[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if a[0] <= b[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    return None


def _boxes_apart(a: BBox, b: BBox, pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _frame_border_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_ports(bbox: BBox, h: int, w: int) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    ports: list[Coord] = []
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0:
            ports.append((r0 - 1, c))
        if r1 + 1 < h:
            ports.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0:
            ports.append((r, c0 - 1))
        if c1 + 1 < w:
            ports.append((r, c1 + 1))
    return ports


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have an interior"
        border = _frame_border_cells((r0, c0, r1, c1))
        assert set(comp) == border, "maroon component must be a rectangular hollow frame"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border": border,
                "anchors": [],
            }
        )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Deliver each connected fragment into its linked target frame by aligning the fragment's matching anchor-colored pixel to the frame anchor.

    Input structure:
    - The background is black.
    - One or more maroon hollow rectangular frames appear on the grid, and each frame interior contains one or two non-black, non-maroon anchor pixels.
    - Separate multicolor 4-connected fragment objects lie outside the frames.
    - Each fragment is linked to exactly one frame by a one-pixel-wide grey path touching the fragment at one end and the frame border at the other.

    Rule:
    - Identify the maroon frames, the anchor pixels inside them, the outside fragments, and the grey linking paths.
    - Use each grey path to pair one fragment with one target frame.
    - Inside the fragment, find the unique pixel whose color matches one anchor color in that paired frame.
    - Translate the whole fragment so that this matching fragment pixel lands exactly on the frame anchor, then erase the original fragment position and the grey path.

    Key insight:
    - The grey path tells which frame receives a fragment, while the shared anchor color tells how to align that fragment once it arrives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    frames = _find_frames(g)
    assert frames, "expected at least one target frame"

    interior_to_frame: dict[Coord, int] = {}
    border_to_frame: dict[Coord, int] = {}
    for fid, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                interior_to_frame[(r, c)] = fid
        for coord in frame["border"]:
            border_to_frame[coord] = fid

    color_mask = (g != BG) & (g != GREY) & (g != MAROON)
    anchor_components: list[tuple[int, Coord, int]] = []
    fragments: list[dict] = []
    for comp in _components(color_mask):
        frame_ids = {interior_to_frame[coord] for coord in comp if coord in interior_to_frame}
        if frame_ids:
            assert all(coord in interior_to_frame for coord in comp), "colored component cannot straddle frame interior and exterior"
            assert len(frame_ids) == 1, "anchor component must belong to exactly one frame"
            assert len(comp) == 1, "anchors must be singleton pixels inside a frame"
            coord = comp[0]
            anchor_components.append((next(iter(frame_ids)), coord, int(g[coord])))
        else:
            fragments.append({"coords": tuple(sorted(comp))})

    for fid, coord, color in anchor_components:
        frames[fid]["anchors"].append({"coord": coord, "color": color})

    for frame in frames:
        assert 1 <= len(frame["anchors"]) <= 2, "each frame must contain one or two anchors"
        anchor_colors = [anchor["color"] for anchor in frame["anchors"]]
        assert len(anchor_colors) == len(set(anchor_colors)), "anchors inside one frame must have distinct colors"

    fragment_coord_to_id: dict[Coord, int] = {}
    for frag_id, frag in enumerate(fragments):
        for coord in frag["coords"]:
            fragment_coord_to_id[coord] = frag_id

    path_components = _components(g == GREY)
    assert len(path_components) == len(fragments), "expected one grey path per fragment"

    fragment_to_frame: dict[int, int] = {}
    all_path_cells: list[Coord] = []
    for comp in path_components:
        adj_fragments: set[int] = set()
        adj_frames: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in fragment_coord_to_id:
                    adj_fragments.add(fragment_coord_to_id[(nr, nc)])
                if (nr, nc) in border_to_frame:
                    adj_frames.add(border_to_frame[(nr, nc)])
        assert len(adj_fragments) == 1, "each path must touch exactly one fragment"
        assert len(adj_frames) == 1, "each path must touch exactly one frame"
        frag_id = next(iter(adj_fragments))
        frame_id = next(iter(adj_frames))
        assert frag_id not in fragment_to_frame, "each fragment must use exactly one path"
        fragment_to_frame[frag_id] = frame_id
        all_path_cells.extend(comp)

    assert set(fragment_to_frame) == set(range(len(fragments))), "every fragment must be paired to a frame"

    out = g.copy()
    for r, c in all_path_cells:
        out[r, c] = BG
    for frag in fragments:
        for r, c in frag["coords"]:
            out[r, c] = BG

    used_anchors: set[Coord] = set()
    painted: set[Coord] = set()
    for frag_id, frag in enumerate(fragments):
        frame = frames[fragment_to_frame[frag_id]]
        coords = frag["coords"]
        color_to_coords: dict[int, list[Coord]] = {}
        for r, c in coords:
            color_to_coords.setdefault(int(g[r, c]), []).append((r, c))

        matches: list[tuple[dict, Coord]] = []
        for anchor in frame["anchors"]:
            positions = color_to_coords.get(anchor["color"], [])
            if len(positions) == 1:
                matches.append((anchor, positions[0]))
        assert len(matches) == 1, "fragment must match exactly one anchor in its paired frame"

        anchor, source_anchor = matches[0]
        used_anchors.add(anchor["coord"])
        dr = anchor["coord"][0] - source_anchor[0]
        dc = anchor["coord"][1] - source_anchor[1]
        r0, c0, r1, c1 = frame["bbox"]

        for sr, sc in coords:
            rr, cc = sr + dr, sc + dc
            assert 0 <= rr < h and 0 <= cc < w
            assert r0 < rr < r1 and c0 < cc < c1, "translated fragment must stay inside the frame"
            assert (rr, cc) not in painted, "translated fragments must not overlap"
            assert int(out[rr, cc]) in (BG, int(g[sr, sc])), "frame interior must only contain anchors before placement"
            out[rr, cc] = int(g[sr, sc])
            painted.add((rr, cc))

    expected_anchor_count = sum(len(frame["anchors"]) for frame in frames)
    assert len(used_anchors) == expected_anchor_count, "every anchor must receive one fragment"
    return out


def _sample_layout() -> list[int]:
    return list(random.choices(LAYOUT_CHOICES, weights=LAYOUT_WEIGHTS, k=1)[0])


def _sample_canvas_shape(n_frames: int, n_fragments: int) -> tuple[int, int]:
    base_min = 22
    if n_frames >= 3 or n_fragments >= 4:
        base_min = 25
    rows = random.randint(base_min, 30)
    cols = random.randint(base_min, 30)
    if random.random() < 0.65:
        cols = max(base_min, min(30, rows + random.randint(-4, 4)))
    return rows, cols


def _sample_frame_dims(anchor_count: int) -> tuple[int, int]:
    if anchor_count == 1:
        inner_h = random.randint(4, 7)
        inner_w = random.randint(4, 7)
    else:
        inner_h = random.randint(6, 8)
        inner_w = random.randint(6, 8)
    return inner_h + 2, inner_w + 2


def _place_frames(rows: int, cols: int, layout: list[int]) -> list[dict] | None:
    frames = [{"anchor_count": count, "size": _sample_frame_dims(count)} for count in layout]
    order = sorted(range(len(frames)), key=lambda i: frames[i]["size"][0] * frames[i]["size"][1], reverse=True)
    placed: list[dict | None] = [None] * len(frames)
    bboxes: list[BBox] = []

    for idx in order:
        height, width = frames[idx]["size"]
        if rows - height - 1 < FRAME_MARGIN or cols - width - 1 < FRAME_MARGIN:
            return None
        for _ in range(FRAME_PLACEMENT_ATTEMPTS):
            r0 = random.randint(FRAME_MARGIN, rows - height - FRAME_MARGIN)
            c0 = random.randint(FRAME_MARGIN, cols - width - FRAME_MARGIN)
            bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
            if all(_boxes_apart(bbox, other, FRAME_GAP) for other in bboxes):
                bboxes.append(bbox)
                placed[idx] = {
                    "bbox": bbox,
                    "anchor_count": frames[idx]["anchor_count"],
                    "border": _frame_border_cells(bbox),
                }
                break
        else:
            return None

    return [frame for frame in placed if frame is not None]


def _sample_anchor_colors(layout: list[int]) -> list[list[int]]:
    all_colors: list[list[int]] = []
    shared_color = random.choice(ANCHOR_COLOR_POOL) if len(layout) > 1 and random.random() < 0.8 else None
    shared_frames = set(random.sample(range(len(layout)), 2)) if shared_color is not None else set()

    for idx, count in enumerate(layout):
        if idx in shared_frames:
            others = random.sample([c for c in ANCHOR_COLOR_POOL if c != shared_color], count - 1)
            colors = [shared_color] + others
        else:
            colors = random.sample(ANCHOR_COLOR_POOL, count)
        random.shuffle(colors)
        all_colors.append(colors)
    return all_colors


def _random_shape(max_area: int) -> list[Coord] | None:
    for _ in range(SHAPE_ATTEMPTS):
        target_area = random.randint(3, max_area)
        cells: set[Coord] = {(0, 0)}
        stalled = 0
        while len(cells) < target_area and stalled < 20:
            base_r, base_c = random.choice(tuple(cells))
            candidates = [(base_r - 1, base_c), (base_r + 1, base_c), (base_r, base_c - 1), (base_r, base_c + 1)]
            random.shuffle(candidates)
            placed = False
            for cand in candidates:
                if cand in cells:
                    continue
                tentative = cells | {cand}
                r0, c0, r1, c1 = _bbox(tentative)
                if r1 - r0 + 1 <= 4 and c1 - c0 + 1 <= 4:
                    cells = tentative
                    placed = True
                    break
            if placed:
                stalled = 0
            else:
                stalled += 1
        if len(cells) == target_area:
            return sorted(cells)
    return None


def _place_destination_fragments(frame: dict, anchor_colors: list[int]) -> list[dict] | None:
    r0, c0, r1, c1 = frame["bbox"]
    occupied: set[Coord] = set()
    fragments: list[dict] = []
    forbidden_fill_colors = set(anchor_colors)

    for anchor_color in anchor_colors:
        max_area = 5 if len(anchor_colors) == 2 else 6
        placed = False
        for _ in range(DESTINATION_ATTEMPTS):
            offsets = _random_shape(max_area)
            if offsets is None:
                continue
            anchor_r = random.randint(r0 + 1, r1 - 1)
            anchor_c = random.randint(c0 + 1, c1 - 1)
            cells = {(anchor_r + dr, anchor_c + dc) for dr, dc in offsets}
            if not all(r0 < rr < r1 and c0 < cc < c1 for rr, cc in cells):
                continue
            if cells & occupied:
                continue

            expanded: set[Coord] = set()
            for rr, cc in cells:
                expanded.add((rr, cc))
                for nr, nc in _neighbors8(rr, cc, r1 + 1, c1 + 1):
                    if r0 < nr < r1 and c0 < nc < c1:
                        expanded.add((nr, nc))
            if expanded & occupied:
                continue

            fill_palette = [color for color in FRAGMENT_COLORS if color not in forbidden_fill_colors]
            use_count = random.randint(1, min(3, len(fill_palette)))
            fill_choices = random.sample(fill_palette, use_count)
            colors: dict[Coord, int] = {(0, 0): anchor_color}
            for coord in offsets:
                if coord == (0, 0):
                    continue
                colors[coord] = random.choice(fill_choices)

            fragments.append(
                {
                    "frame_bbox": frame["bbox"],
                    "frame_border": frame["border"],
                    "anchor_color": anchor_color,
                    "anchor_coord": (anchor_r, anchor_c),
                    "offset_colors": colors,
                    "final_cells": [((anchor_r + dr, anchor_c + dc), color) for (dr, dc), color in colors.items()],
                }
            )
            occupied |= expanded
            placed = True
            break
        if not placed:
            return None

    return fragments


def _source_fragment_ok(cells: set[Coord], occupied: set[Coord], structural_block: set[Coord], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied or (r, c) in structural_block:
            return False
    for r, c in cells:
        around = {(r, c), *(_neighbors8(r, c, h, w))}
        for coord in around:
            if coord in cells:
                continue
            if coord in occupied or coord in structural_block:
                return False
    return True


def _path_is_valid(
    path: list[Coord],
    source_cells: set[Coord],
    target_border: set[Coord],
    occupied: set[Coord],
    structural_block: set[Coord],
    h: int,
    w: int,
) -> bool:
    if len(path) < 2 or len(path) != len(set(path)):
        return False

    source_touch = 0
    frame_touch = 0
    for r, c in path:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied or (r, c) in structural_block:
            return False

        neighbors = set(_neighbors4(r, c, h, w))
        touch_source = bool(neighbors & source_cells)
        touch_frame = bool(neighbors & target_border)
        if touch_source:
            source_touch += 1
        if touch_frame:
            frame_touch += 1

        disallowed_occupied = (neighbors & occupied) - target_border
        if disallowed_occupied:
            return False
        if any(coord in structural_block and coord not in target_border for coord in neighbors):
            return False

    return source_touch == 1 and frame_touch == 1


def _find_path_candidates(
    source_cells: set[Coord],
    target_border: set[Coord],
    ports: list[Coord],
    occupied: set[Coord],
    structural_block: set[Coord],
    h: int,
    w: int,
) -> dict[str, list[list[Coord]]]:
    candidates: dict[str, list[list[Coord]]] = {"straight": [], "elbow": []}

    adjacency: set[Coord] = set()
    for r, c in source_cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) not in source_cells and (nr, nc) not in occupied and (nr, nc) not in structural_block:
                adjacency.add((nr, nc))

    usable_ports = [port for port in ports if port not in occupied and port not in structural_block]
    for start in adjacency:
        for end in usable_ports:
            straight = _segment_cells(start, end)
            if straight is not None and _path_is_valid(straight, source_cells, target_border, occupied, structural_block, h, w):
                candidates["straight"].append(straight)

            if start[0] == end[0] or start[1] == end[1]:
                continue
            for corner in ((start[0], end[1]), (end[0], start[1])):
                seg_a = _segment_cells(start, corner)
                seg_b = _segment_cells(corner, end)
                if seg_a is None or seg_b is None:
                    continue
                elbow = seg_a + seg_b[1:]
                if _path_is_valid(elbow, source_cells, target_border, occupied, structural_block, h, w):
                    candidates["elbow"].append(elbow)
    return candidates


def _place_sources_and_paths(rows: int, cols: int, frames: list[dict], fragments: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((rows, cols), BG, dtype=int)
    structural_block: set[Coord] = set()
    occupied: set[Coord] = set()

    for frame in frames:
        _draw_frame(grid, frame["bbox"])
        r0, c0, r1, c1 = frame["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                structural_block.add((r, c))
        occupied |= frame["border"]

    for frag in fragments:
        ar, ac = frag["anchor_coord"]
        grid[ar, ac] = frag["anchor_color"]
        occupied.add((ar, ac))

    order = sorted(range(len(fragments)), key=lambda idx: len(fragments[idx]["offset_colors"]), reverse=True)
    for idx in order:
        frag = fragments[idx]
        offsets = list(frag["offset_colors"].keys())
        target_border = frag["frame_border"]
        target_bbox = frag["frame_bbox"]
        ports = _frame_ports(target_bbox, rows, cols)

        placed = False
        for _ in range(SOURCE_ATTEMPTS):
            source_anchor = (random.randint(0, rows - 1), random.randint(0, cols - 1))
            source_cells = {(source_anchor[0] + dr, source_anchor[1] + dc) for dr, dc in offsets}
            if not _source_fragment_ok(source_cells, occupied, structural_block, rows, cols):
                continue

            path_candidates = _find_path_candidates(source_cells, target_border, ports, occupied, structural_block, rows, cols)
            if not path_candidates["straight"] and not path_candidates["elbow"]:
                continue

            desired_kind = random.choice(["straight", "elbow"])
            usable = path_candidates[desired_kind] or path_candidates["straight"] or path_candidates["elbow"]
            path = random.choice(usable)

            source_cells_with_colors: list[tuple[Coord, int]] = []
            for (dr, dc), color in frag["offset_colors"].items():
                coord = (source_anchor[0] + dr, source_anchor[1] + dc)
                source_cells_with_colors.append((coord, color))
                grid[coord] = color
                occupied.add(coord)
            for coord in path:
                grid[coord] = GREY
                occupied.add(coord)

            frag["source_cells"] = source_cells_with_colors
            frag["path_cells"] = list(path)
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for frag in fragments:
        for (r, c), _color in frag["source_cells"]:
            expected[r, c] = BG
        for r, c in frag["path_cells"]:
            expected[r, c] = BG
        for (r, c), color in frag["final_cells"]:
            expected[r, c] = color

    return grid, expected


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    rows, cols = _sample_canvas_shape(len(layout), sum(layout))
    frames = _place_frames(rows, cols, layout)
    if frames is None:
        return None

    anchor_colors_by_frame = _sample_anchor_colors(layout)
    all_fragments: list[dict] = []
    for frame, anchor_colors in zip(frames, anchor_colors_by_frame, strict=False):
        fragments = _place_destination_fragments(frame, anchor_colors)
        if fragments is None:
            return None
        all_fragments.extend(fragments)

    built = _place_sources_and_paths(rows, cols, frames, all_fragments)
    if built is None:
        return None
    grid, expected = built

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connected-fragment delivery puzzle")
