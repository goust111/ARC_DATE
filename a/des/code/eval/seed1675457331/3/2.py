import numpy as np


DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
WALL_COLOR = 5
GEN_ATTEMPTS = 500


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_2x2(points: list[tuple[int, int]]) -> bool:
    if len(points) != 4:
        return False
    r0, c0, r1, c1 = _bbox(points)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    need = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return set(points) == need


def _edge_segment_ok(points: list[tuple[int, int]], h: int, w: int) -> bool:
    if not (2 <= len(points) <= 4):
        return False
    rows = {r for r, _ in points}
    cols = {c for _, c in points}
    if len(rows) == 1:
        row = next(iter(rows))
        if row not in (0, h - 1):
            return False
        c0 = min(cols)
        c1 = max(cols)
        return set(points) == {(row, c) for c in range(c0, c1 + 1)}
    if len(cols) == 1:
        col = next(iter(cols))
        if col not in (0, w - 1):
            return False
        r0 = min(rows)
        r1 = max(rows)
        return set(points) == {(r, col) for r in range(r0, r1 + 1)}
    return False


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_of_color(grid, color)
        if comps and all(_edge_segment_ok(comp, h, w) for comp in comps):
            candidates.append((sum(len(comp) for comp in comps), color))
    assert candidates, "expected at least one wall color"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return int(candidates[0][1])


def _parse_arrow(points: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True

    if h == 3 and w >= 4 and len(points) == w + 2:
        if local[1].all() and int(local[0].sum()) == 1 and int(local[2].sum()) == 1:
            top_idx = int(np.flatnonzero(local[0])[0])
            bot_idx = int(np.flatnonzero(local[2])[0])
            if top_idx == bot_idx == 0:
                return {
                    "direction": "left",
                    "length": w - 1,
                    "start": (r0 + 1, c1),
                }
            if top_idx == bot_idx == w - 1:
                return {
                    "direction": "right",
                    "length": w - 1,
                    "start": (r0 + 1, c0),
                }

    if w == 3 and h >= 4 and len(points) == h + 2:
        if local[:, 1].all() and int(local[:, 0].sum()) == 1 and int(local[:, 2].sum()) == 1:
            left_idx = int(np.flatnonzero(local[:, 0])[0])
            right_idx = int(np.flatnonzero(local[:, 2])[0])
            if left_idx == right_idx == 0:
                return {
                    "direction": "up",
                    "length": h - 1,
                    "start": (r1, c0 + 1),
                }
            if left_idx == right_idx == h - 1:
                return {
                    "direction": "down",
                    "length": h - 1,
                    "start": (r0, c0 + 1),
                }
    return None


def _extract_scene(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    wall_color = _detect_wall_color(g, bg)
    cargos: list[dict] = []
    arrows: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, wall_color)):
        for comp in _components_of_color(g, color):
            if _is_solid_2x2(comp):
                r0, c0, _, _ = _bbox(comp)
                cargos.append(
                    {
                        "color": color,
                        "cells": set(comp),
                        "top_left": (r0, c0),
                    }
                )
                continue
            arrow = _parse_arrow(comp)
            assert arrow is not None, f"unrecognized non-wall component for color {color}"
            arrow["color"] = color
            arrow["cells"] = set(comp)
            arrows.append(arrow)

    assert cargos and arrows and len(cargos) == len(arrows), "expected one cargo per arrow"

    used: set[int] = set()
    for arrow in arrows:
        dr, dc = DIR_VECS[arrow["direction"]]
        anchor = (arrow["start"][0] - dr, arrow["start"][1] - dc)
        matches = [idx for idx, cargo in enumerate(cargos) if anchor in cargo["cells"]]
        assert len(matches) == 1, "each arrow must match exactly one cargo"
        idx = matches[0]
        assert idx not in used, "a cargo cannot belong to two arrows"
        used.add(idx)
        arrow["cargo"] = cargos[idx]

    return bg, wall_color, cargos, arrows


def _wall_spans(grid: np.ndarray, wall_color: int) -> dict[str, set[int]]:
    h, w = grid.shape
    spans = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    for c in range(w):
        if int(grid[0, c]) == wall_color:
            spans["top"].add(c)
        if int(grid[h - 1, c]) == wall_color:
            spans["bottom"].add(c)
    for r in range(h):
        if int(grid[r, 0]) == wall_color:
            spans["left"].add(r)
        if int(grid[r, w - 1]) == wall_color:
            spans["right"].add(r)
    return spans


def _target_top_left(grid: np.ndarray, wall_color: int, cargo: dict, direction: str) -> tuple[int, int]:
    h, w = grid.shape
    spans = _wall_spans(grid, wall_color)
    r0, c0 = cargo["top_left"]
    if direction == "right":
        blocked = r0 in spans["right"] or (r0 + 1) in spans["right"]
        return r0, (w - 3 if blocked else w - 2)
    if direction == "left":
        blocked = r0 in spans["left"] or (r0 + 1) in spans["left"]
        return r0, (1 if blocked else 0)
    if direction == "down":
        blocked = c0 in spans["bottom"] or (c0 + 1) in spans["bottom"]
        return (h - 3 if blocked else h - 2), c0
    assert direction == "up"
    blocked = c0 in spans["top"] or (c0 + 1) in spans["top"]
    return (1 if blocked else 0), c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each 2x2 cargo along its arrow direction until it reaches the boundary
    or an edge wall, then resolve overlaps by shaft length.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_color, _cargos, arrows = _extract_scene(g)
    out = np.full_like(g, bg)
    out[g == wall_color] = wall_color

    placements = []
    for arrow in arrows:
        cargo = arrow["cargo"]
        tr, tc = _target_top_left(g, wall_color, cargo, arrow["direction"])
        placements.append(
            {
                "length": int(arrow["length"]),
                "color": int(cargo["color"]),
                "top_left": (tr, tc),
            }
        )

    placements.sort(key=lambda item: item["length"])
    for item in placements:
        r0, c0 = item["top_left"]
        out[r0 : r0 + 2, c0 : c0 + 2] = item["color"]
    return out


def _expand(points: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _wall_cells_for_edge(edge: str, start: int, length: int, h: int, w: int) -> set[tuple[int, int]]:
    if edge == "top":
        return {(0, c) for c in range(start, start + length)}
    if edge == "bottom":
        return {(h - 1, c) for c in range(start, start + length)}
    if edge == "left":
        return {(r, 0) for r in range(start, start + length)}
    return {(r, w - 1) for r in range(start, start + length)}


def _choose_edge_segments(
    rng: np.random.Generator,
    h: int,
    w: int,
    clear: dict[str, set[int]],
) -> dict[str, set[tuple[int, int]]]:
    segments: dict[str, set[tuple[int, int]]] = {}
    for edge in ("top", "bottom", "left", "right"):
        axis_len = w if edge in ("top", "bottom") else h
        length = int(rng.integers(2, 5))
        starts = []
        for start in range(axis_len - length + 1):
            covered = set(range(start, start + length))
            if covered & clear.get(edge, set()):
                continue
            starts.append(start)
        if not starts:
            return {}
        start = int(rng.choice(starts))
        segments[edge] = _wall_cells_for_edge(edge, start, length, h, w)
    return segments


def _hit_and_miss_sets(h: int, w: int, walls: dict[str, set[tuple[int, int]]]) -> dict[str, tuple[list[int], list[int]]]:
    top_cols = {c for _, c in walls["top"]}
    bottom_cols = {c for _, c in walls["bottom"]}
    left_rows = {r for r, _ in walls["left"]}
    right_rows = {r for r, _ in walls["right"]}
    result = {
        "up": (
            [c for c in range(w - 1) if c in top_cols or (c + 1) in top_cols],
            [c for c in range(w - 1) if c not in top_cols and (c + 1) not in top_cols],
        ),
        "down": (
            [c for c in range(w - 1) if c in bottom_cols or (c + 1) in bottom_cols],
            [c for c in range(w - 1) if c not in bottom_cols and (c + 1) not in bottom_cols],
        ),
        "left": (
            [r for r in range(h - 1) if r in left_rows or (r + 1) in left_rows],
            [r for r in range(h - 1) if r not in left_rows and (r + 1) not in left_rows],
        ),
        "right": (
            [r for r in range(h - 1) if r in right_rows or (r + 1) in right_rows],
            [r for r in range(h - 1) if r not in right_rows and (r + 1) not in right_rows],
        ),
    }
    return result


def _boxes_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return not (ar + 1 < br or br + 1 < ar or ac + 1 < bc or bc + 1 < ac)


def _candidate_finals(
    direction: str,
    stop: bool,
    h: int,
    w: int,
    edge_hits: dict[str, tuple[list[int], list[int]]],
) -> list[tuple[int, int]]:
    hit_vals, miss_vals = edge_hits[direction]
    vals = hit_vals if stop else miss_vals
    if direction == "right":
        col = w - 3 if stop else w - 2
        return [(r, col) for r in vals]
    if direction == "left":
        col = 1 if stop else 0
        return [(r, col) for r in vals]
    if direction == "down":
        row = h - 3 if stop else h - 2
        return [(row, c) for c in vals]
    row = 1 if stop else 0
    return [(row, c) for c in vals]


def _valid_variants(direction: str, r0: int, c0: int, h: int, w: int) -> list[str]:
    if direction in ("left", "right"):
        variants = []
        if r0 <= h - 3:
            variants.append("above")
        if r0 >= 1:
            variants.append("below")
        return variants
    variants = []
    if c0 <= w - 3:
        variants.append("left")
    if c0 >= 1:
        variants.append("right")
    return variants


def _build_object(
    r0: int,
    c0: int,
    direction: str,
    variant: str,
    length: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cargo = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
    arrow: set[tuple[int, int]]
    if direction == "right":
        shaft_row = r0 + 1 if variant == "above" else r0
        start_c = c0 + 2
        arrow = {(shaft_row, start_c + k) for k in range(length)}
        tip_c = start_c + length
        arrow |= {(shaft_row - 1, tip_c), (shaft_row, tip_c), (shaft_row + 1, tip_c)}
        return cargo, arrow
    if direction == "left":
        shaft_row = r0 + 1 if variant == "above" else r0
        start_c = c0 - 1
        arrow = {(shaft_row, start_c - k) for k in range(length)}
        tip_c = start_c - length
        arrow |= {(shaft_row - 1, tip_c), (shaft_row, tip_c), (shaft_row + 1, tip_c)}
        return cargo, arrow
    if direction == "down":
        shaft_col = c0 + 1 if variant == "left" else c0
        start_r = r0 + 2
        arrow = {(start_r + k, shaft_col) for k in range(length)}
        tip_r = start_r + length
        arrow |= {(tip_r, shaft_col - 1), (tip_r, shaft_col), (tip_r, shaft_col + 1)}
        return cargo, arrow
    shaft_col = c0 + 1 if variant == "left" else c0
    start_r = r0 - 1
    arrow = {(start_r - k, shaft_col) for k in range(length)}
    tip_r = start_r - length
    arrow |= {(tip_r, shaft_col - 1), (tip_r, shaft_col), (tip_r, shaft_col + 1)}
    return cargo, arrow


def _all_in_bounds(points: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in points)


def _initial_options(
    final_top_left: tuple[int, int],
    direction: str,
    length: int,
    h: int,
    w: int,
) -> list[tuple[int, int, str]]:
    fr, fc = final_top_left
    options: list[tuple[int, int, str]] = []
    if direction == "right":
        max_c = min(fc - 1, w - length - 4)
        for c0 in range(1, max_c + 1):
            for variant in _valid_variants(direction, fr, c0, h, w):
                options.append((fr, c0, variant))
        return options
    if direction == "left":
        min_c = max(fc + 1, length + 2)
        for c0 in range(min_c, w - 2):
            for variant in _valid_variants(direction, fr, c0, h, w):
                options.append((fr, c0, variant))
        return options
    if direction == "down":
        max_r = min(fr - 1, h - length - 4)
        for r0 in range(1, max_r + 1):
            for variant in _valid_variants(direction, r0, fc, h, w):
                options.append((r0, fc, variant))
        return options
    min_r = max(fr + 1, length + 2)
    for r0 in range(min_r, h - 2):
        for variant in _valid_variants(direction, r0, fc, h, w):
            options.append((r0, fc, variant))
    return options


def _pick_finals(
    rng: np.random.Generator,
    h: int,
    w: int,
    overlap_mode: bool,
    edge_hits: dict[str, tuple[list[int], list[int]]],
    corner: str | None,
) -> list[dict] | None:
    specs: list[dict] = []

    if overlap_mode:
        assert corner is not None
        if corner == "top_left":
            specs.extend(
                [
                    {"direction": "up", "stop": False, "final": (0, 0)},
                    {"direction": "left", "stop": False, "final": (0, 0)},
                ]
            )
            remaining_dirs = ["right", "down"]
        elif corner == "top_right":
            specs.extend(
                [
                    {"direction": "up", "stop": False, "final": (0, w - 2)},
                    {"direction": "right", "stop": False, "final": (0, w - 2)},
                ]
            )
            remaining_dirs = ["left", "down"]
        elif corner == "bottom_left":
            specs.extend(
                [
                    {"direction": "down", "stop": False, "final": (h - 2, 0)},
                    {"direction": "left", "stop": False, "final": (h - 2, 0)},
                ]
            )
            remaining_dirs = ["up", "right"]
        else:
            specs.extend(
                [
                    {"direction": "down", "stop": False, "final": (h - 2, w - 2)},
                    {"direction": "right", "stop": False, "final": (h - 2, w - 2)},
                ]
            )
            remaining_dirs = ["up", "left"]

        extra_count = int(rng.integers(1, 3))
        if extra_count == 1:
            extra_dirs = [str(rng.choice(remaining_dirs))]
        else:
            extra_dirs = remaining_dirs
        stop_flags = [True] + [bool(rng.integers(0, 2)) for _ in range(len(extra_dirs) - 1)]
        rng.shuffle(stop_flags)
        for direction, stop in zip(extra_dirs, stop_flags):
            candidates = _candidate_finals(direction, stop, h, w, edge_hits)
            rng.shuffle(candidates)
            chosen = None
            for final in candidates:
                if any(_boxes_overlap(final, spec["final"]) for spec in specs):
                    continue
                chosen = final
                break
            if chosen is None:
                return None
            specs.append({"direction": direction, "stop": stop, "final": chosen})
        return specs

    n_arrows = int(rng.integers(2, 5))
    directions = list(rng.choice(np.array(["up", "down", "left", "right"]), size=n_arrows, replace=False))
    stop_flags = [True, False] + [bool(rng.integers(0, 2)) for _ in range(n_arrows - 2)]
    rng.shuffle(stop_flags)
    for direction, stop in zip(directions, stop_flags):
        candidates = _candidate_finals(direction, stop, h, w, edge_hits)
        rng.shuffle(candidates)
        chosen = None
        for final in candidates:
            if any(_boxes_overlap(final, spec["final"]) for spec in specs):
                continue
            chosen = final
            break
        if chosen is None:
            return None
        specs.append({"direction": direction, "stop": stop, "final": chosen})
    return specs


def _clear_requirements(h: int, w: int, overlap_mode: bool, corner: str | None) -> dict[str, set[int]]:
    clear = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    if not overlap_mode or corner is None:
        return clear
    if corner == "top_left":
        clear["top"].update({0, 1})
        clear["left"].update({0, 1})
    elif corner == "top_right":
        clear["top"].update({w - 2, w - 1})
        clear["right"].update({0, 1})
    elif corner == "bottom_left":
        clear["bottom"].update({0, 1})
        clear["left"].update({h - 2, h - 1})
    else:
        clear["bottom"].update({w - 2, w - 1})
        clear["right"].update({h - 2, h - 1})
    return clear


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(14, 19))
    w = int(rng.integers(14, 19))
    bg = int(rng.choice(PALETTE))
    overlap_mode = bool(rng.integers(0, 2))
    corner = None
    if overlap_mode:
        corner = str(rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"]))
    clear = _clear_requirements(h, w, overlap_mode, corner)
    walls = _choose_edge_segments(rng, h, w, clear)
    if not walls:
        return None

    edge_hits = _hit_and_miss_sets(h, w, walls)
    if any(not hit or not miss for hit, miss in edge_hits.values()):
        return None

    specs = _pick_finals(rng, h, w, overlap_mode, edge_hits, corner)
    if specs is None:
        return None

    n_arrows = len(specs)
    max_len = min(6, min(h, w) - 7)
    lengths = list(range(3, max_len + 1))
    if len(lengths) < n_arrows:
        return None
    rng.shuffle(lengths)
    lengths = sorted(lengths[:n_arrows], reverse=True)

    colors = [c for c in PALETTE if c != bg]
    if len(colors) < 2 * n_arrows:
        return None
    rng.shuffle(colors)

    placed_specs = []
    blocked = set()
    wall_cells = set().union(*walls.values())
    blocked |= wall_cells
    for wall in wall_cells:
        blocked.add(wall)

    for idx, spec in enumerate(sorted(specs, key=lambda item: (item["final"][0] + item["final"][1]))):
        length = lengths[idx]
        options = _initial_options(spec["final"], spec["direction"], length, h, w)
        rng.shuffle(options)
        placed = None
        for r0, c0, variant in options:
            cargo, arrow = _build_object(r0, c0, spec["direction"], variant, length)
            cells = cargo | arrow
            if not _all_in_bounds(cells, h, w):
                continue
            if cells & wall_cells:
                continue
            if cells & blocked:
                continue
            placed = {
                **spec,
                "length": length,
                "cargo_color": int(colors[2 * idx]),
                "arrow_color": int(colors[2 * idx + 1]),
                "cargo_cells": cargo,
                "arrow_cells": arrow,
            }
            blocked |= _expand(cells, h, w)
            placed_specs.append(placed)
            break
        if placed is None:
            return None

    grid = np.full((h, w), bg, dtype=int)
    for edge_cells in walls.values():
        for r, c in edge_cells:
            grid[r, c] = WALL_COLOR
    for spec in placed_specs:
        for r, c in spec["cargo_cells"]:
            grid[r, c] = spec["cargo_color"]
        for r, c in spec["arrow_cells"]:
            grid[r, c] = spec["arrow_color"]

    try:
        out = main(grid)
        bg2, wall2, cargos2, arrows2 = _extract_scene(grid)
        assert bg2 == bg
        assert wall2 == WALL_COLOR
        assert len(cargos2) == len(arrows2) == n_arrows
        targets = [_target_top_left(grid, wall2, arrow["cargo"], arrow["direction"]) for arrow in arrows2]
        stop_flags = []
        for arrow, target in zip(arrows2, targets):
            if arrow["direction"] == "right":
                stop_flags.append(target[1] == w - 3)
            elif arrow["direction"] == "left":
                stop_flags.append(target[1] == 1)
            elif arrow["direction"] == "down":
                stop_flags.append(target[0] == h - 3)
            else:
                stop_flags.append(target[0] == 1)
        assert any(stop_flags) and any(not flag for flag in stop_flags)
        has_overlap = False
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                if _boxes_overlap(targets[i], targets[j]):
                    has_overlap = True
        if overlap_mode:
            assert has_overlap
        else:
            assert not has_overlap
        assert np.array_equal(out, main(grid))
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
