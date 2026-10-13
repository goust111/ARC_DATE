from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 <= dst.shape[0] - h
    assert 0 <= c0 <= dst.shape[1] - w
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (r0, c0) == (0, 0) or (r0, c1) == (0, w - 1) or (r1, c0) == (h - 1, 0) or (r1, c1) == (h - 1, w - 1)


def _program_info(grid: np.ndarray, bg: int, comps: list[np.ndarray]) -> dict:
    candidates = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        flat = list(map(int, patch.ravel()))
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            continue
        if len(coords) != patch.size:
            continue
        if not np.all(patch != bg):
            continue
        if len(set(flat)) != len(flat):
            continue
        if not _touches_corner(bbox, grid.shape):
            continue
        candidates.append({"bbox": bbox, "patch": patch})
    assert len(candidates) == 1, "expected one solid multicolor corner program block"
    return candidates[0]


def _snake_read(block: np.ndarray) -> list[int]:
    seq: list[int] = []
    for r in range(block.shape[0]):
        row = block[r, :] if r % 2 == 0 else block[r, ::-1]
        seq.extend(map(int, row))
    return seq


def _segment_info(grid: np.ndarray, bg: int, coords: np.ndarray) -> dict:
    bbox = _bbox_coords(coords)
    patch = _crop_bbox(grid, bbox)
    mask = patch != bg
    colors = set(map(int, patch[mask]))
    assert len(colors) == 1, "each fragment must be monochrome"
    color = colors.pop()

    degrees: dict[tuple[int, int], int] = {}
    endpoints = []
    for r, c in map(tuple, np.argwhere(mask)):
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and mask[nr, nc]:
                deg += 1
        assert 1 <= deg <= 2, "fragment must be path-like"
        degrees[(int(r), int(c))] = deg
        if deg == 1:
            endpoints.append((int(r), int(c)))

    assert len(endpoints) == 2, "fragment must have exactly two degree-1 endpoints"
    left = [pt for pt in endpoints if pt[1] == 0]
    right = [pt for pt in endpoints if pt[1] == patch.shape[1] - 1]
    assert len(left) == 1 and len(right) == 1, "fragment endpoints must sit on bbox side columns"

    rows_used = {r for r, _c in map(tuple, np.argwhere(mask))}
    return {
        "color": color,
        "bbox": bbox,
        "patch": patch,
        "left_row": left[0][0],
        "right_row": right[0][0],
        "turned": len(rows_used) > 1,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered colored path fragments into a compact chain using the corner program block's boustrophedon order.

    Input structure:
    - The background is the dominant empty color.
    - Exactly one grid corner contains a solid rectangular program block whose cell colors are all distinct.
    - Every other non-background component is a monochrome 4-connected path fragment, and each fragment's own leftmost and rightmost occupied columns contain exactly one degree-1 endpoint.

    Rule:
    - Crop the corner program block and read its colors in boustrophedon order: local row 0 left-to-right, local row 1 right-to-left, alternating by row.
    - Match each read color to the fragment of the same color.
    - Produce an output with the same height as the input and width equal to the sum of fragment widths. Place the first fragment at the left edge with its original top row unchanged. Then place each later fragment immediately to the right, shifting it only vertically so its left endpoint lands on the same row as the previous placed fragment's right endpoint.
    - The program block itself does not appear in the output.

    Key insight:
    - The fragments already reveal how endpoints should touch once an order is chosen, but only the corner block supplies that order; its boustrophedon color sequence is the unique assembly program.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _components(g != bg)
    program = _program_info(g, bg, comps)
    seq = _snake_read(program["patch"])

    segment_by_color: dict[int, dict] = {}
    for coords in comps:
        bbox = _bbox_coords(coords)
        if bbox == program["bbox"]:
            continue
        info = _segment_info(g, bg, coords)
        assert info["color"] not in segment_by_color, "each program color must name one fragment"
        segment_by_color[info["color"]] = info

    assert set(seq) == set(segment_by_color), "program colors must match fragment colors one-to-one"

    h = g.shape[0]
    out_w = sum(segment_by_color[color]["patch"].shape[1] for color in seq)
    out = np.full((h, out_w), bg, dtype=int)

    col = 0
    first = segment_by_color[seq[0]]
    first_top = first["bbox"][0]
    _paste(out, first["patch"], first_top, 0, transparent=bg)
    prev_right_row = first_top + first["right_row"]
    col += first["patch"].shape[1]

    for color in seq[1:]:
        frag = segment_by_color[color]
        top = prev_right_row - frag["left_row"]
        assert 0 <= top <= h - frag["patch"].shape[0], "vertical slide must stay in-bounds"
        _paste(out, frag["patch"], top, col, transparent=bg)
        prev_right_row = top + frag["right_row"]
        col += frag["patch"].shape[1]

    return out


BG = 0
PROGRAM_SHAPES = ((2, 2), (2, 3), (3, 2))
SEGMENT_HEIGHTS = (3, 4, 5)
SEGMENT_WIDTHS = (2, 3)
HEIGHT_RANGE = (10, 16)
SEPARATOR = 1
MAX_GENERATION_ATTEMPTS = 400


def _render_path(path: tuple[tuple[int, int], ...], height: int, width: int, color: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    for r, c in path:
        patch[r, c] = color
    return patch


def _path_valid(path: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    cells = set(path)
    endpoints = []
    for r, c in cells:
        deg = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        if deg not in (1, 2):
            return False
        if deg == 1:
            endpoints.append((r, c))
    if len(endpoints) != 2:
        return False
    left = [pt for pt in endpoints if pt[1] == 0]
    right = [pt for pt in endpoints if pt[1] == w - 1]
    return len(left) == 1 and len(right) == 1


@lru_cache(maxsize=None)
def _path_options(h: int, w: int, left_row: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    start = (left_row, 0)
    max_cells = min(h * w, w + h + 1)
    paths: list[tuple[tuple[int, int], ...]] = []

    def dfs(path: list[tuple[int, int]], goal: tuple[int, int], n_cells: int) -> None:
        if len(path) == n_cells:
            if path[-1] == goal:
                trial = tuple(path)
                if _path_valid(trial, h, w):
                    paths.append(trial)
            return

        remain = n_cells - len(path) - 1
        r, c = path[-1]
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in path:
                continue
            if abs(nr - goal[0]) + abs(nc - goal[1]) > remain:
                continue
            path.append((nr, nc))
            dfs(path, goal, n_cells)
            path.pop()

    for n_cells in range(w + 1, max_cells + 1):
        for right_row in range(h):
            dfs([start], (right_row, w - 1), n_cells)

    uniq = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return tuple(uniq)


def _normalized_path_info(path: tuple[tuple[int, int], ...]) -> dict:
    min_row = min(r for r, _c in path)
    max_row = max(r for r, _c in path)
    shifted = tuple((r - min_row, c) for r, c in path)
    return {
        "path": shifted,
        "height": max_row - min_row + 1,
        "width": max(c for _r, c in shifted) + 1,
        "left_row": shifted[0][0],
        "right_row": shifted[-1][0],
        "turned": len({r for r, _c in shifted}) > 1,
    }


def _choose_program_shape() -> tuple[int, int]:
    return random.choice(PROGRAM_SHAPES)


def _inverse_snake_fill(seq: list[int], h: int, w: int) -> np.ndarray:
    block = np.full((h, w), BG, dtype=int)
    idx = 0
    for r in range(h):
        cols = range(w) if r % 2 == 0 else range(w - 1, -1, -1)
        for c in cols:
            block[r, c] = seq[idx]
            idx += 1
    return block


def _all_segment_candidates(h: int, left_abs_row: int) -> list[dict]:
    candidates = []
    for box_h in SEGMENT_HEIGHTS:
        for box_w in SEGMENT_WIDTHS:
            for left_row in range(box_h):
                for path in _path_options(box_h, box_w, left_row):
                    info = _normalized_path_info(path)
                    top = left_abs_row - info["left_row"]
                    if not (0 <= top <= h - info["height"]):
                        continue
                    candidates.append(
                        {
                            "patch_shape": (info["height"], info["width"]),
                            "path": info["path"],
                            "left_row": info["left_row"],
                            "right_row": info["right_row"],
                            "top": top,
                            "turned": info["turned"],
                        }
                    )
    return candidates


def _build_chain(seq: list[int], h: int) -> list[dict] | None:
    first_candidates = []
    for box_h in SEGMENT_HEIGHTS:
        for box_w in SEGMENT_WIDTHS:
            for left_row in range(box_h):
                for path in _path_options(box_h, box_w, left_row):
                    info = _normalized_path_info(path)
                    for top in range(0, h - info["height"] + 1):
                        first_candidates.append(
                            {
                                "patch_shape": (info["height"], info["width"]),
                                "path": info["path"],
                                "left_row": info["left_row"],
                                "right_row": info["right_row"],
                                "top": top,
                                "turned": info["turned"],
                            }
                        )

    for _ in range(160):
        chain: list[dict] = []
        first = random.choice(first_candidates)
        chain.append(
            {
                "color": seq[0],
                "patch": _render_path(first["path"], *first["patch_shape"], seq[0]),
                "left_row": first["left_row"],
                "right_row": first["right_row"],
                "output_top": first["top"],
                "turned": first["turned"],
            }
        )
        current_row = first["top"] + first["right_row"]

        ok = True
        for color in seq[1:]:
            candidates = _all_segment_candidates(h, current_row)
            if not candidates:
                ok = False
                break
            chosen = random.choice(candidates)
            chain.append(
                {
                    "color": color,
                    "patch": _render_path(chosen["path"], *chosen["patch_shape"], color),
                    "left_row": chosen["left_row"],
                    "right_row": chosen["right_row"],
                    "output_top": chosen["top"],
                    "turned": chosen["turned"],
                }
            )
            current_row = chosen["top"] + chosen["right_row"]

        if not ok:
            continue
        if not any(item["turned"] or item["left_row"] != item["right_row"] for item in chain):
            continue
        return chain
    return None


def _compose_output(chain: list[dict], h: int) -> np.ndarray:
    out_w = sum(item["patch"].shape[1] for item in chain)
    out = np.full((h, out_w), BG, dtype=int)
    col = 0
    for item in chain:
        _paste(out, item["patch"], item["output_top"], col, transparent=BG)
        item["output_left"] = col
        col += item["patch"].shape[1]
    return out


def _choose_input_tops(chain: list[dict], h: int) -> dict[int, int] | None:
    tops = {chain[0]["color"]: int(chain[0]["output_top"])}
    moved = False
    for item in chain[1:]:
        seg_h = item["patch"].shape[0]
        choices = list(range(0, h - seg_h + 1))
        if not choices:
            return None
        preferred = [t for t in choices if t != item["output_top"]]
        if preferred and random.random() < 0.8:
            top = random.choice(preferred)
            moved = True
        else:
            top = random.choice(choices)
            moved |= top != item["output_top"]
        tops[item["color"]] = int(top)

    if moved:
        return tops

    for item in chain[1:]:
        seg_h = item["patch"].shape[0]
        choices = [t for t in range(0, h - seg_h + 1) if t != item["output_top"]]
        if choices:
            tops[item["color"]] = random.choice(choices)
            return tops
    return None


def _input_orders_match_program(placements: dict[int, dict], seq: list[int]) -> bool:
    colors = list(placements)
    naive_orders = [
        tuple(sorted(colors, key=lambda color: placements[color]["c0"])),
        tuple(sorted(colors, key=lambda color: -placements[color]["c0"])),
        tuple(sorted(colors, key=lambda color: (placements[color]["r0"], placements[color]["c0"]))),
        tuple(sorted(colors, key=lambda color: (-placements[color]["r0"], placements[color]["c0"]))),
    ]
    return tuple(seq) in naive_orders


def _make_input(seq: list[int], chain: list[dict], program_block: np.ndarray, h: int) -> np.ndarray | None:
    strip_order = seq[:]
    while True:
        random.shuffle(strip_order)
        if strip_order != seq:
            break

    strip_width = sum(next(item for item in chain if item["color"] == color)["patch"].shape[1] for color in strip_order) + SEPARATOR * (len(strip_order) - 1)
    prog_h, prog_w = program_block.shape
    extra_gap = random.randint(0, 3)
    corner = random.choice(("tl", "tr", "bl", "br"))

    if corner in ("tl", "bl"):
        strip_start = prog_w + 1 + extra_gap
        grid_w = strip_start + strip_width
        program_c0 = 0
    else:
        strip_start = 0
        grid_w = strip_width + 1 + extra_gap + prog_w
        program_c0 = grid_w - prog_w

    if grid_w > 30:
        return None

    program_r0 = 0 if corner in ("tl", "tr") else h - prog_h
    tops = _choose_input_tops(chain, h)
    if tops is None:
        return None

    grid = np.full((h, grid_w), BG, dtype=int)
    _paste(grid, program_block, program_r0, program_c0, transparent=BG)

    placements: dict[int, dict] = {}
    col = strip_start
    by_color = {item["color"]: item for item in chain}
    for index, color in enumerate(strip_order):
        item = by_color[color]
        top = tops[color]
        _paste(grid, item["patch"], top, col, transparent=BG)
        placements[color] = {"r0": top, "c0": col}
        col += item["patch"].shape[1]
        if index + 1 < len(strip_order):
            col += SEPARATOR

    if _input_orders_match_program(placements, seq):
        return None
    if tuple(seq) == tuple(sorted(seq)) or tuple(seq) == tuple(sorted(seq, reverse=True)):
        return None
    return grid


def _validate_generated(inp: np.ndarray, expected: np.ndarray, seq: list[int], chain: list[dict]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    bg = _get_bg(inp)
    comps = _components(inp != bg)
    try:
        program = _program_info(inp, bg, comps)
    except Exception:
        return False

    if _snake_read(program["patch"]) != seq:
        return False

    segment_colors = []
    parsed_infos = {}
    for coords in comps:
        if _bbox_coords(coords) == program["bbox"]:
            continue
        try:
            info = _segment_info(inp, bg, coords)
        except Exception:
            return False
        segment_colors.append(info["color"])
        parsed_infos[info["color"]] = info

    if sorted(segment_colors) != sorted(seq):
        return False

    by_color = {item["color"]: item for item in chain}
    moved = any(
        _bbox_coords(coords)[0] != by_color[_segment_info(inp, bg, coords)["color"]]["output_top"]
        for coords in comps
        if _bbox_coords(coords) != program["bbox"]
        and _segment_info(inp, bg, coords)["color"] != seq[0]
    )
    if not moved:
        return False

    naive_orders = [
        tuple(sorted(parsed_infos, key=lambda color: parsed_infos[color]["bbox"][1])),
        tuple(sorted(parsed_infos, key=lambda color: -parsed_infos[color]["bbox"][1])),
        tuple(sorted(parsed_infos, key=lambda color: (parsed_infos[color]["bbox"][0], parsed_infos[color]["bbox"][1]))),
        tuple(sorted(parsed_infos, key=lambda color: (-parsed_infos[color]["bbox"][0], parsed_infos[color]["bbox"][1]))),
    ]
    if tuple(seq) in naive_orders:
        return False

    if not any(item["turned"] or item["left_row"] != item["right_row"] for item in chain):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    prog_h, prog_w = _choose_program_shape()
    seq = random.sample(list(range(1, 10)), prog_h * prog_w)
    h = random.randint(*HEIGHT_RANGE)
    program_block = _inverse_snake_fill(seq, prog_h, prog_w)

    chain = _build_chain(seq, h)
    if chain is None:
        return None
    expected = _compose_output(chain, h)

    inp = _make_input(seq, chain, program_block, h)
    if inp is None:
        return None
    if not _validate_generated(inp, expected, seq, chain):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
