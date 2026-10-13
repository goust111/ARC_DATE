from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GRAY = 5
SIDE_NAMES = ("top", "bottom", "left", "right")
SIDE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _line_bars(grid: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg), GRAY}):
        for coords in _components(grid == int(color), connectivity=4):
            if len(coords) < 3:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            area = len(coords)
            if r0 == r1 and area == c1 - c0 + 1:
                bars.append(
                    {
                        "orientation": "h",
                        "color": int(color),
                        "row": r0,
                        "c0": c0,
                        "c1": c1,
                        "length": area,
                        "coords": set(coords),
                    }
                )
            elif c0 == c1 and area == r1 - r0 + 1:
                bars.append(
                    {
                        "orientation": "v",
                        "color": int(color),
                        "col": c0,
                        "r0": r0,
                        "r1": r1,
                        "length": area,
                        "coords": set(coords),
                    }
                )
    return bars


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    hbars = [bar for bar in _line_bars(grid, bg) if bar["orientation"] == "h"]
    vbars = [bar for bar in _line_bars(grid, bg) if bar["orientation"] == "v"]
    candidates: list[dict] = []

    for i, h1 in enumerate(hbars):
        for h2 in hbars[i + 1 :]:
            if h1["row"] == h2["row"]:
                continue
            top, bottom = (h1, h2) if h1["row"] < h2["row"] else (h2, h1)
            for j, v1 in enumerate(vbars):
                for v2 in vbars[j + 1 :]:
                    if v1["col"] == v2["col"]:
                        continue
                    left, right = (v1, v2) if v1["col"] < v2["col"] else (v2, v1)

                    tr, br = int(top["row"]), int(bottom["row"])
                    lc, rc = int(left["col"]), int(right["col"])
                    if br - tr < 4 or rc - lc < 4:
                        continue
                    if int(top["length"]) != rc - lc - 1:
                        continue
                    if int(bottom["length"]) != rc - lc - 1:
                        continue
                    if int(left["length"]) != br - tr - 1:
                        continue
                    if int(right["length"]) != br - tr - 1:
                        continue
                    if (int(top["c0"]), int(top["c1"])) != (lc + 1, rc - 1):
                        continue
                    if (int(bottom["c0"]), int(bottom["c1"])) != (lc + 1, rc - 1):
                        continue
                    if (int(left["r0"]), int(left["r1"])) != (tr + 1, br - 1):
                        continue
                    if (int(right["r0"]), int(right["r1"])) != (tr + 1, br - 1):
                        continue
                    colors = [top["color"], bottom["color"], left["color"], right["color"]]
                    if len(set(map(int, colors))) != 4:
                        continue
                    if any(int(grid[r, c]) != int(bg) for r, c in ((tr, lc), (tr, rc), (br, lc), (br, rc))):
                        continue

                    frame_coords = set()
                    for bar in (top, bottom, left, right):
                        frame_coords |= set(bar["coords"])
                    candidates.append(
                        {
                            "top": {"color": int(top["color"]), "line": tr},
                            "bottom": {"color": int(bottom["color"]), "line": br},
                            "left": {"color": int(left["color"]), "line": lc},
                            "right": {"color": int(right["color"]), "line": rc},
                            "r0": tr,
                            "r1": br,
                            "c0": lc,
                            "c1": rc,
                            "coords": frame_coords,
                        }
                    )

    assert len(candidates) == 1, "expected a unique four-color rectangular frame"
    return candidates[0]


def _reflect_cell(side: str, r: int, c: int, frame: dict) -> tuple[int, int]:
    if side == "top":
        return 2 * int(frame["top"]["line"]) - int(r), int(c)
    if side == "bottom":
        return 2 * int(frame["bottom"]["line"]) - int(r), int(c)
    if side == "left":
        return int(r), 2 * int(frame["left"]["line"]) - int(c)
    if side == "right":
        return int(r), 2 * int(frame["right"]["line"]) - int(c)
    raise AssertionError("unknown side")


def _strictly_inside_frame(r: int, c: int, frame: dict) -> bool:
    return int(frame["r0"]) < int(r) < int(frame["r1"]) and int(frame["c0"]) < int(c) < int(frame["c1"])


def _nearest_line_sides(coords: list[tuple[int, int]], frame: dict) -> set[str]:
    r0, c0, r1, c1 = _bbox(coords)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    distances = {
        "top": abs(cr - int(frame["top"]["line"])),
        "bottom": abs(cr - int(frame["bottom"]["line"])),
        "left": abs(cc - int(frame["left"]["line"])),
        "right": abs(cc - int(frame["right"]["line"])),
    }
    best = min(distances.values())
    return {side for side, dist in distances.items() if dist == best}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each gray marked sprite into the frame across the same-colored frame side.

    Input structure:
    - The background is the dominant color.
    - A central rectangular frame has four solid colored sides, four background corners,
      and an empty background interior.
    - Outside the frame are gray sprites; each sprite contains exactly one colored marker
      pixel whose color is one of the four frame-side colors.

    Rule:
    - Match each sprite's marker color to the frame side with that color.
    - Reflect every cell of that whole sprite across the straight line containing the
      matched side, so the reflected sprite lands completely inside the frame.
    - Erase the original outside sprite and recolor the entire reflected image with the
      matched side color, including the former marker cell.
    - Keep the four-colored frame unchanged.

    Key insight:
    - The marker color, not the visually closest side line, chooses the mirror; the
      placement is still a true geometric reflection across that selected side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    frame = _find_frame(g, bg)
    color_to_side = {int(frame[side]["color"]): side for side in SIDE_NAMES}

    movable = g != int(bg)
    for r, c in frame["coords"]:
        movable[int(r), int(c)] = False

    moves: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int]] = []
    for coords in _components(movable, connectivity=8):
        vals = [int(g[r, c]) for r, c in coords]
        if GRAY not in vals:
            continue
        marker_cells = [
            (int(r), int(c), int(g[r, c]))
            for r, c in coords
            if int(g[r, c]) != GRAY and int(g[r, c]) != int(bg)
        ]
        assert len(marker_cells) == 1, "each gray sprite must have exactly one marker pixel"
        _mr, _mc, marker_color = marker_cells[0]
        assert marker_color in color_to_side, "marker color must match one frame side"
        side = color_to_side[marker_color]
        reflected = [_reflect_cell(side, r, c, frame) for r, c in coords]
        assert all(_strictly_inside_frame(r, c, frame) for r, c in reflected), "reflected sprite must land inside the frame"
        moves.append((coords, reflected, marker_color))

    assert moves, "expected at least one marked gray sprite"

    out = g.copy()
    for coords, _reflected, _color in moves:
        for r, c in coords:
            out[int(r), int(c)] = int(bg)

    painted: set[tuple[int, int]] = set()
    for _coords, reflected, color in moves:
        for r, c in reflected:
            rc = (int(r), int(c))
            assert rc not in painted, "reflected sprites must not overlap"
            assert int(out[rc]) == int(bg), "reflected sprite must land on empty interior"
            painted.add(rc)
            out[rc] = int(color)

    return out


def _frame_coords(frame: dict) -> set[tuple[int, int]]:
    tr, br, lc, rc = int(frame["r0"]), int(frame["r1"]), int(frame["c0"]), int(frame["c1"])
    coords = {(tr, c) for c in range(lc + 1, rc)}
    coords |= {(br, c) for c in range(lc + 1, rc)}
    coords |= {(r, lc) for r in range(tr + 1, br)}
    coords |= {(r, rc) for r in range(tr + 1, br)}
    return coords


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    tr, br, lc, rc = int(frame["r0"]), int(frame["r1"]), int(frame["c0"]), int(frame["c1"])
    grid[tr, lc + 1 : rc] = int(frame["top"]["color"])
    grid[br, lc + 1 : rc] = int(frame["bottom"]["color"])
    grid[tr + 1 : br, lc] = int(frame["left"]["color"])
    grid[tr + 1 : br, rc] = int(frame["right"]["color"])


def _touches(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _add_with_padding(occupied: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    occupied.update((int(r), int(c)) for r, c in cells)


def _frontier(cells: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    out = []
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                out.append((rr, cc))
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _make_sprite() -> dict:
    area = random.randint(5, 9)
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    cells: set[tuple[int, int]] = {(height // 2, width // 2)}
    while len(cells) < area:
        choices = _frontier(cells, height, width)
        if not choices:
            return _make_sprite()
        cells.add(random.choice(choices))
    cells = _normalize_cells(cells)

    r0, c0, r1, c1 = _bbox(list(cells))
    marker_pool = [
        (r, c)
        for r, c in cells
        if r0 < r < r1
        and c0 < c < c1
        and sum((r + dr, c + dc) in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))) >= 2
    ]
    if not marker_pool:
        marker_pool = list(cells)
    marker = random.choice(marker_pool)
    return {"cells": sorted(cells), "marker": marker}


def _candidate_offsets(sprite: dict, frame: dict) -> list[tuple[int, int]]:
    cells = sprite["cells"]
    sr0, sc0, sr1, sc1 = _bbox(cells)
    offsets = []
    for r in range(int(frame["r0"]) + 1 - sr0, int(frame["r1"]) - sr1):
        for c in range(int(frame["c0"]) + 1 - sc0, int(frame["c1"]) - sc1):
            offsets.append((r, c))
    random.shuffle(offsets)
    return offsets


def _place_sprite(
    sprite: dict,
    side: str,
    frame: dict,
    input_occ: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
    shape: tuple[int, int],
    need_decoy: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for off_r, off_c in _candidate_offsets(sprite, frame):
        target = [(off_r + r, off_c + c) for r, c in sprite["cells"]]
        source = [_reflect_cell(side, r, c, frame) for r, c in target]
        if not all(_strictly_inside_frame(r, c, frame) for r, c in target):
            continue
        if any(_strictly_inside_frame(r, c, frame) for r, c in source):
            continue
        if _touches(target, output_occ, shape, pad=1):
            continue
        if _touches(source, input_occ, shape, pad=1):
            continue
        nearest = _nearest_line_sides(source, frame)
        if need_decoy and side in nearest:
            continue
        return source, target
    return None


def _random_frame_spec() -> tuple[tuple[int, int], dict]:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    for _ in range(200):
        inner_h = random.randint(9, min(16, h - 10))
        inner_w = random.randint(9, min(16, w - 10))
        tr = random.randint(4, h - inner_h - 6)
        lc = random.randint(4, w - inner_w - 6)
        br = tr + inner_h + 1
        rc = lc + inner_w + 1
        if h - br - 1 >= 4 and w - rc - 1 >= 4:
            colors = random.sample(SIDE_COLOR_POOL, 4)
            frame = {
                "top": {"color": colors[0], "line": tr},
                "bottom": {"color": colors[1], "line": br},
                "left": {"color": colors[2], "line": lc},
                "right": {"color": colors[3], "line": rc},
                "r0": tr,
                "r1": br,
                "c0": lc,
                "c1": rc,
            }
            frame["coords"] = _frame_coords(frame)
            return (h, w), frame
    raise RuntimeError("failed to choose a frame")


def _build_attempt() -> np.ndarray | None:
    shape, frame = _random_frame_spec()
    inp = np.full(shape, BG, dtype=int)
    expected = np.full(shape, BG, dtype=int)
    _draw_frame(inp, frame)
    _draw_frame(expected, frame)

    input_occ = set(frame["coords"])
    output_occ = set(frame["coords"])
    sides = list(SIDE_NAMES)
    random.shuffle(sides)
    for _ in range(random.randint(0, 3)):
        sides.append(random.choice(SIDE_NAMES))

    decoy_index = random.randrange(len(sides))
    had_decoy = False
    placed_by_side = {side: 0 for side in SIDE_NAMES}

    for idx, side in enumerate(sides):
        placed = None
        for _ in range(80):
            sprite = _make_sprite()
            placed = _place_sprite(
                sprite,
                side,
                frame,
                input_occ,
                output_occ,
                shape,
                need_decoy=(idx == decoy_index),
            )
            if placed is not None:
                break
        if placed is None:
            return None

        source, target = placed
        marker_target = None
        for r, c in target:
            local = (r - min(rr for rr, _cc in target), c - min(cc for _rr, cc in target))
            if local == sprite["marker"]:
                marker_target = (r, c)
                break
        if marker_target is None:
            off_r = target[0][0] - sprite["cells"][0][0]
            off_c = target[0][1] - sprite["cells"][0][1]
            marker_target = (off_r + sprite["marker"][0], off_c + sprite["marker"][1])
        marker_source = _reflect_cell(side, marker_target[0], marker_target[1], frame)
        side_color = int(frame[side]["color"])

        for r, c in source:
            inp[r, c] = GRAY
        inp[marker_source] = side_color
        for r, c in target:
            expected[r, c] = side_color

        _add_with_padding(input_occ, source)
        _add_with_padding(output_occ, target)
        placed_by_side[side] += 1
        had_decoy = had_decoy or side not in _nearest_line_sides(source, frame)

    if not all(placed_by_side.values()):
        return None
    if not had_decoy:
        return None
    if int(np.sum(inp == GRAY)) < 12:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/3/021/6 input")
