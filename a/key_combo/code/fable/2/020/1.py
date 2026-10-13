from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RECT_DIMS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 1),
    (4, 2),
)
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACE_ATTEMPTS = 500
CAPTURE_STEP_CHOICES = (2, 2, 3)
MAX_VISIBLE_STEPS = 6


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _rect_cells(r0: int, c0: int, h: int, w: int, shape: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    cells = {
        (rr, cc)
        for rr in range(r0, r0 + h)
        for cc in range(c0, c0 + w)
    }
    if shape is None:
        return cells
    gh, gw = shape
    return {(r, c) for r, c in cells if 0 <= r < gh and 0 <= c < gw}


def _frame_border_cells(inner_r: int, inner_c: int, h: int, w: int) -> set[tuple[int, int]]:
    top = inner_r - 1
    left = inner_c - 1
    bottom = inner_r + h
    right = inner_c + w
    cells = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    gh, gw = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < gh and 0 <= cc < gw:
                    out.add((rr, cc))
    return out


def _is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return int(comp["area"]) == h * w


def _is_hollow_rect(grid: np.ndarray, comp: dict, bg: int = BG) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or int(comp["area"]) != 2 * h + 2 * w - 4:
        return False
    rel = {(r - r0, c - c0) for r, c in comp["coords"]}
    border = {
        (rr, cc)
        for rr in range(h)
        for cc in range(w)
        if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1
    }
    if rel != border:
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg))


def _trajectory_steps(spec: dict, shape: tuple[int, int]) -> list[dict]:
    gh, gw = shape
    steps: list[dict] = []
    k = 1
    while True:
        r = int(spec["r"]) + k * int(spec["h"])
        c = int(spec["c"]) + k * int(spec["w"])
        if spec["capture_step"] is not None and k == int(spec["capture_step"]):
            steps.append({"kind": "capture", "step": k, "r": r, "c": c, "clipped": False})
            break
        rr0 = max(0, r)
        cc0 = max(0, c)
        rr1 = min(gh, r + int(spec["h"]))
        cc1 = min(gw, c + int(spec["w"]))
        if rr0 >= rr1 or cc0 >= cc1:
            break
        steps.append(
            {
                "kind": "stamp",
                "step": k,
                "r": r,
                "c": c,
                "rr0": rr0,
                "cc0": cc0,
                "rr1": rr1,
                "cc1": cc1,
                "clipped": rr0 != r or cc0 != c or rr1 != r + int(spec["h"]) or cc1 != c + int(spec["w"]),
            }
        )
        k += 1
    return steps


def _solid_entity_cells(spec: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _rect_cells(int(spec["r"]), int(spec["c"]), int(spec["h"]), int(spec["w"]))
    for step in _trajectory_steps(spec, shape):
        if step["kind"] == "capture":
            cells |= _rect_cells(step["r"], step["c"], int(spec["h"]), int(spec["w"]))
            cells |= _frame_border_cells(step["r"], step["c"], int(spec["h"]), int(spec["w"]))
        else:
            cells |= {
                (rr, cc)
                for rr in range(step["rr0"], step["rr1"])
                for cc in range(step["cc0"], step["cc1"])
            }
    return cells


def _frame_box_clear(
    inner_r: int,
    inner_c: int,
    h: int,
    w: int,
    occupied_actual: set[tuple[int, int]],
) -> bool:
    for rr in range(inner_r - 1, inner_r + h + 1):
        for cc in range(inner_c - 1, inner_c + w + 1):
            if (rr, cc) in occupied_actual:
                return False
    return True


def _render_input(solids: list[dict], frames: list[dict], shape: tuple[int, int]) -> np.ndarray:
    gh, gw = shape
    grid = np.full((gh, gw), BG, dtype=int)
    for solid in solids:
        r0 = int(solid["r"])
        c0 = int(solid["c"])
        h = int(solid["h"])
        w = int(solid["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(solid["color"])
    for frame in frames:
        for r, c in _frame_border_cells(int(frame["inner_r"]), int(frame["inner_c"]), int(frame["h"]), int(frame["w"])):
            grid[r, c] = int(frame["color"])
    return grid


def _render_output(solids: list[dict], frames: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = _render_input(solids, frames, shape)
    gh, gw = shape
    frame_lookup = {
        (int(frame["inner_r"]), int(frame["inner_c"]), int(frame["h"]), int(frame["w"])): int(frame["color"])
        for frame in frames
    }
    for solid in sorted(solids, key=lambda item: (int(item["r"]), int(item["c"]), int(item["color"]))):
        h = int(solid["h"])
        w = int(solid["w"])
        color = int(solid["color"])
        for step in _trajectory_steps(solid, shape):
            if step["kind"] == "capture":
                frame_color = frame_lookup[(step["r"], step["c"], h, w)]
                out[step["r"] : step["r"] + h, step["c"] : step["c"] + w] = frame_color
                break
            rr0 = max(0, step["r"])
            cc0 = max(0, step["c"])
            rr1 = min(gh, step["r"] + h)
            cc1 = min(gw, step["c"] + w)
            out[rr0:rr1, cc0:cc1] = color
    for frame in frames:
        for r, c in _frame_border_cells(int(frame["inner_r"]), int(frame["inner_c"]), int(frame["h"]), int(frame["w"])):
            out[r, c] = int(frame["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each solid rectangle down-right until one copy exactly fills a hollow frame interior, then recolor that copy with the frame color.

    Input structure:
    - The background is black.
    - Several monochrome solid axis-aligned rectangles appear on the canvas.
    - Several monochrome one-cell-thick hollow rectangular frames also appear, with empty black interiors.

    Rule:
    - Keep every original solid rectangle and every hollow frame border unchanged.
    - A solid rectangle of height h and width w stamps copies of itself at offsets (h, w), (2h, 2w), ... along the down-right diagonal.
    - Ordinary copies keep the rectangle color and are clipped at the grid border, but if a copy lands exactly on a frame interior of the same size, fill that interior with the frame color instead and stop that rectangle's trajectory there.

    Key insight:
    - Size matching is not enough: a hollow frame is filled only when its interior lies exactly on the diagonal lattice generated by a rectangle's own height-width stride.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    gh, gw = g.shape

    comps = _find_components(g, bg=BG)
    solids: list[dict] = []
    frames: list[dict] = []
    for comp in comps:
        if _is_solid_rect(comp):
            r0, c0, r1, c1 = comp["bbox"]
            solids.append(
                {
                    "color": int(comp["color"]),
                    "r": int(r0),
                    "c": int(c0),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                }
            )
        elif _is_hollow_rect(g, comp, bg=BG):
            r0, c0, r1, c1 = comp["bbox"]
            frames.append(
                {
                    "color": int(comp["color"]),
                    "inner_r": int(r0 + 1),
                    "inner_c": int(c0 + 1),
                    "h": int(r1 - r0 - 1),
                    "w": int(c1 - c0 - 1),
                }
            )
        else:
            raise AssertionError("all foreground components must be solid rectangles or hollow rectangle frames")

    frame_lookup = {
        (int(frame["inner_r"]), int(frame["inner_c"]), int(frame["h"]), int(frame["w"])): int(frame["color"])
        for frame in frames
    }
    assert len(frame_lookup) == len(frames), "frame interiors must be unique"

    out = g.copy()
    for solid in sorted(solids, key=lambda item: (int(item["r"]), int(item["c"]), int(item["color"]))):
        h = int(solid["h"])
        w = int(solid["w"])
        color = int(solid["color"])
        k = 1
        while True:
            r = int(solid["r"]) + k * h
            c = int(solid["c"]) + k * w
            frame_color = frame_lookup.get((r, c, h, w))
            if frame_color is not None:
                out[r : r + h, c : c + w] = frame_color
                break
            rr0 = max(0, r)
            cc0 = max(0, c)
            rr1 = min(gh, r + h)
            cc1 = min(gw, c + w)
            if rr0 >= rr1 or cc0 >= cc1:
                break
            out[rr0:rr1, cc0:cc1] = color
            k += 1
    for frame in frames:
        for r, c in _frame_border_cells(int(frame["inner_r"]), int(frame["inner_c"]), int(frame["h"]), int(frame["w"])):
            out[r, c] = int(frame["color"])
    return out


def _place_captured_solid(
    shape: tuple[int, int],
    color: int,
    frame_color: int,
    occupied_actual: set[tuple[int, int]],
    occupied_buffered: set[tuple[int, int]],
) -> dict | None:
    gh, gw = shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        h, w = random.choice(RECT_DIMS)
        capture_step = random.choice(CAPTURE_STEP_CHOICES)
        max_r = gh - (capture_step + 1) * h - 1
        max_c = gw - (capture_step + 1) * w - 1
        if max_r < 0 or max_c < 0:
            continue
        r0 = random.randint(0, max_r)
        c0 = random.randint(0, max_c)
        spec = {
            "color": color,
            "r": r0,
            "c": c0,
            "h": h,
            "w": w,
            "capture_step": capture_step,
            "frame_color": frame_color,
        }
        cells = _solid_entity_cells(spec, shape)
        if any(cell in occupied_buffered for cell in cells):
            continue
        steps = _trajectory_steps(spec, shape)
        if len(steps) > MAX_VISIBLE_STEPS:
            continue
        return spec
    return None


def _place_uncaptured_solid(
    shape: tuple[int, int],
    color: int,
    occupied_actual: set[tuple[int, int]],
    occupied_buffered: set[tuple[int, int]],
) -> dict | None:
    gh, gw = shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        h, w = random.choice(RECT_DIMS)
        r0 = random.randint(0, gh - h)
        c0 = random.randint(0, gw - w)
        spec = {
            "color": color,
            "r": r0,
            "c": c0,
            "h": h,
            "w": w,
            "capture_step": None,
            "frame_color": None,
        }
        steps = _trajectory_steps(spec, shape)
        if not steps or len(steps) < 2 or len(steps) > MAX_VISIBLE_STEPS:
            continue
        if not any(step["kind"] == "stamp" and step["clipped"] for step in steps):
            continue
        cells = _solid_entity_cells(spec, shape)
        if any(cell in occupied_buffered for cell in cells):
            continue
        return spec
    return None


def _place_decoy_frame(
    shape: tuple[int, int],
    color: int,
    size_options: list[tuple[int, int]],
    occupied_actual: set[tuple[int, int]],
    occupied_buffered: set[tuple[int, int]],
) -> dict | None:
    gh, gw = shape
    option_order = size_options[:]
    random.shuffle(option_order)
    for h, w in option_order:
        candidates = []
        for inner_r in range(1, gh - h):
            for inner_c in range(1, gw - w):
                border = _frame_border_cells(inner_r, inner_c, h, w)
                if any(cell in occupied_buffered for cell in border):
                    continue
                if not _frame_box_clear(inner_r, inner_c, h, w, occupied_actual):
                    continue
                candidates.append((inner_r, inner_c))
        if candidates:
            inner_r, inner_c = random.choice(candidates)
            return {"color": color, "inner_r": inner_r, "inner_c": inner_c, "h": h, "w": w, "kind": "decoy"}
    return None


def _active_hit_map(solids: list[dict], shape: tuple[int, int]) -> dict[tuple[int, int, int, int], set[int]]:
    hit_map: dict[tuple[int, int, int, int], set[int]] = {}
    for idx, solid in enumerate(solids):
        h = int(solid["h"])
        w = int(solid["w"])
        for step in _trajectory_steps(solid, shape):
            if step["kind"] == "capture" or not step["clipped"]:
                key = (int(step["r"]), int(step["c"]), h, w)
                hit_map.setdefault(key, set()).add(idx)
    return hit_map


def _validate_layout(solids: list[dict], frames: list[dict], shape: tuple[int, int], expected: np.ndarray) -> bool:
    if not solids or not frames:
        return False
    hit_map = _active_hit_map(solids, shape)

    captured = [solid for solid in solids if solid["capture_step"] is not None]
    uncaptured = [solid for solid in solids if solid["capture_step"] is None]
    if not captured or not uncaptured:
        return False
    if not any(step["kind"] == "stamp" and step["clipped"] for solid in uncaptured for step in _trajectory_steps(solid, shape)):
        return False

    same_size_decoys = 0
    for frame in frames:
        key = (int(frame["inner_r"]), int(frame["inner_c"]), int(frame["h"]), int(frame["w"]))
        hits = hit_map.get(key, set())
        if frame.get("kind") == "capture":
            if hits != {int(frame["solid_index"])}:
                return False
        else:
            if hits:
                return False
            if any((int(solid["h"]), int(solid["w"])) == (int(frame["h"]), int(frame["w"])) for solid in solids):
                same_size_decoys += 1
    if same_size_decoys < 1:
        return False

    inp = _render_input(solids, frames, shape)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    for frame in frames:
        if frame.get("kind") != "decoy":
            continue
        r = int(frame["inner_r"])
        c = int(frame["inner_c"])
        h = int(frame["h"])
        w = int(frame["w"])
        if np.any(solved[r : r + h, c : c + w] != BG):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    gh = random.randint(*GRID_HEIGHT_RANGE)
    gw = random.randint(*GRID_WIDTH_RANGE)
    shape = (gh, gw)

    solid_count = random.choice((2, 3))
    captured_count = 1 if solid_count == 2 else random.choice((1, 2))
    uncaptured_count = solid_count - captured_count
    extra_decoys = random.choice((0, 1))
    frame_count = captured_count + 1 + extra_decoys

    colors = random.sample(PALETTE, solid_count + frame_count)
    solid_colors = colors[:solid_count]
    frame_colors = colors[solid_count:]

    solids: list[dict] = []
    frames: list[dict] = []
    occupied_actual: set[tuple[int, int]] = set()
    occupied_buffered: set[tuple[int, int]] = set()

    for idx in range(captured_count):
        spec = _place_captured_solid(shape, solid_colors[idx], frame_colors[idx], occupied_actual, occupied_buffered)
        if spec is None:
            return None
        solids.append(spec)
        cells = _solid_entity_cells(spec, shape)
        occupied_actual |= cells
        occupied_buffered |= _expand_cells(cells, shape)
        frame_step = next(step for step in _trajectory_steps(spec, shape) if step["kind"] == "capture")
        frames.append(
            {
                "color": int(spec["frame_color"]),
                "inner_r": int(frame_step["r"]),
                "inner_c": int(frame_step["c"]),
                "h": int(spec["h"]),
                "w": int(spec["w"]),
                "kind": "capture",
                "solid_index": len(solids) - 1,
            }
        )

    for idx in range(uncaptured_count):
        spec = _place_uncaptured_solid(
            shape,
            solid_colors[captured_count + idx],
            occupied_actual,
            occupied_buffered,
        )
        if spec is None:
            return None
        solids.append(spec)
        cells = _solid_entity_cells(spec, shape)
        occupied_actual |= cells
        occupied_buffered |= _expand_cells(cells, shape)

    same_size_options = [(int(solid["h"]), int(solid["w"])) for solid in solids]
    decoy = _place_decoy_frame(shape, frame_colors[captured_count], same_size_options, occupied_actual, occupied_buffered)
    if decoy is None:
        return None
    frames.append(decoy)
    border = _frame_border_cells(int(decoy["inner_r"]), int(decoy["inner_c"]), int(decoy["h"]), int(decoy["w"]))
    occupied_actual |= border
    occupied_buffered |= _expand_cells(border, shape)

    if extra_decoys:
        used_sizes = {(int(solid["h"]), int(solid["w"])) for solid in solids}
        extra_sizes = [dim for dim in RECT_DIMS if dim not in used_sizes] or list(RECT_DIMS)
        decoy2 = _place_decoy_frame(shape, frame_colors[captured_count + 1], extra_sizes, occupied_actual, occupied_buffered)
        if decoy2 is None:
            return None
        frames.append(decoy2)
        border2 = _frame_border_cells(int(decoy2["inner_r"]), int(decoy2["inner_c"]), int(decoy2["h"]), int(decoy2["w"]))
        occupied_actual |= border2
        occupied_buffered |= _expand_cells(border2, shape)

    expected = _render_output(solids, frames, shape)
    if not _validate_layout(solids, frames, shape, expected):
        return None
    return _render_input(solids, frames, shape)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-stamp capture task")
