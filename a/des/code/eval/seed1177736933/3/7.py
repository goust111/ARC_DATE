from __future__ import annotations

from typing import Any

import numpy as np


BG = 0
FRAME_COLOR = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("up", "down", "left", "right")
PALETTE = tuple(color for color in range(10) if color not in {BG, FRAME_COLOR})
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in DIR4:
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


def _scan_colors(grid: np.ndarray, bg: int = BG) -> list[int]:
    seen: list[int] = []
    for value in grid.flat:
        value = int(value)
        if value == bg or value in seen:
            continue
        seen.append(value)
    return seen


def _ray_distance(side: str, inner_h: int, inner_w: int, r: int, c: int) -> int:
    if side == "up":
        return inner_h - r
    if side == "down":
        return r + 1
    if side == "left":
        return inner_w - c
    if side == "right":
        return c + 1
    raise ValueError(f"unknown side: {side}")


def _nearest_side(inner_h: int, inner_w: int, r: int, c: int) -> str:
    distances = {
        "up": r + 1,
        "down": inner_h - r,
        "left": c + 1,
        "right": inner_w - c,
    }
    best = min(distances.values())
    winners = [side for side, dist in distances.items() if dist == best]
    if len(winners) != 1:
        raise AssertionError("defect has ambiguous nearest edge")
    return winners[0]


def _ray_start_and_delta(
    bbox: tuple[int, int, int, int], inner_r: int, inner_c: int, side: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    grid_r = r0 + 1 + inner_r
    grid_c = c0 + 1 + inner_c
    if side == "up":
        return (r0 - 1, grid_c), (-1, 0)
    if side == "down":
        return (r1 + 1, grid_c), (1, 0)
    if side == "left":
        return (grid_r, c0 - 1), (0, -1)
    if side == "right":
        return (grid_r, c1 + 1), (0, 1)
    raise ValueError(f"unknown side: {side}")


def _find_frames(grid: np.ndarray) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for color in _scan_colors(grid):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 4 or width < 4:
                continue
            if len(comp) != 2 * (height + width) - 4:
                continue
            ok = True
            for rr in range(r0, r1 + 1):
                if int(grid[rr, c0]) != color or int(grid[rr, c1]) != color:
                    ok = False
                    break
            if not ok:
                continue
            for cc in range(c0, c1 + 1):
                if int(grid[r0, cc]) != color or int(grid[r1, cc]) != color:
                    ok = False
                    break
            if not ok:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            inner_colors = _scan_colors(inner)
            if len(inner_colors) != 2:
                continue
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "frame_color": color,
                    "inner": inner.copy(),
                    "inner_colors": inner_colors,
                }
            )
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    return frames


def _infer_phase(inner: np.ndarray, colors: list[int]) -> int:
    scores: list[tuple[int, int]] = []
    for phase in (0, 1):
        mismatches = 0
        observed = 0
        for r in range(inner.shape[0]):
            for c in range(inner.shape[1]):
                value = int(inner[r, c])
                if value == BG:
                    continue
                observed += 1
                expected = colors[(r + c + phase) % 2]
                if value != expected:
                    mismatches += 1
        scores.append((mismatches, observed))
    if max(obs for _, obs in scores) == 0:
        raise AssertionError("checkerboard has no visible cells")
    if scores[0][0] < scores[1][0]:
        return 0
    if scores[1][0] < scores[0][0]:
        return 1
    if scores[0][0] != 0:
        raise AssertionError("checkerboard phase is inconsistent")
    raise AssertionError("checkerboard phase is ambiguous")


def _frame_rays(frame: dict[str, Any]) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    inner = frame["inner"]
    colors = frame["inner_colors"]
    phase = _infer_phase(inner, colors)
    inner_h, inner_w = inner.shape
    rays: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            if int(inner[r, c]) != BG:
                continue
            side = _nearest_side(inner_h, inner_w, r, c)
            ray_len = _ray_distance(side, inner_h, inner_w, r, c)
            color = colors[(r + c + phase) % 2]
            start, delta = _ray_start_and_delta(frame["bbox"], r, c, side)
            rays.append((start, delta, ray_len, color))
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find black defects inside checkerboard-filled rectangular frames and project
    outward rays toward the nearest edge. Each ray uses the missing checkerboard
    color and has length equal to the defect's distance to the opposite wall.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one rectangular container"
    for frame in frames:
        for start, delta, ray_len, color in _frame_rays(frame):
            rr, cc = start
            dr, dc = delta
            for _ in range(ray_len):
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = color
                rr += dr
                cc += dc
    return out


def _defect_groups(inner_h: int, inner_w: int) -> dict[str, list[tuple[int, int, int]]]:
    groups = {side: [] for side in SIDES}
    for r in range(inner_h):
        for c in range(inner_w):
            distances = {
                "up": r + 1,
                "down": inner_h - r,
                "left": c + 1,
                "right": inner_w - c,
            }
            best = min(distances.values())
            winners = [side for side, dist in distances.items() if dist == best]
            if len(winners) != 1:
                continue
            side = winners[0]
            ray_len = _ray_distance(side, inner_h, inner_w, r, c)
            groups[side].append((r, c, ray_len))
    return groups


def _choose_defects(rng: np.random.Generator, inner_h: int, inner_w: int) -> list[dict[str, int | str]]:
    groups = _defect_groups(inner_h, inner_w)
    vertical = [side for side in ("up", "down") if groups[side]]
    horizontal = [side for side in ("left", "right") if groups[side]]
    if not vertical or not horizontal:
        raise AssertionError("container must support all direction classes")

    defects: list[dict[str, int | str]] = []
    if float(rng.random()) < 0.5:
        side = str(rng.choice(np.array(SIDES, dtype=object)))
        candidates = [item for item in groups[side] if item[2] >= 2] or groups[side]
        r, c, _ = candidates[int(rng.integers(len(candidates)))]
        defects.append({"side": side, "r": int(r), "c": int(c)})
    else:
        side_v = str(rng.choice(np.array(vertical, dtype=object)))
        side_h = str(rng.choice(np.array(horizontal, dtype=object)))
        for side in (side_v, side_h):
            candidates = [item for item in groups[side] if item[2] >= 2] or groups[side]
            r, c, _ = candidates[int(rng.integers(len(candidates)))]
            defects.append({"side": side, "r": int(r), "c": int(c)})
    return defects


def _make_container_spec(rng: np.random.Generator) -> dict[str, Any]:
    inner_h = int(rng.integers(4, 8))
    inner_w = int(rng.integers(4, 8))
    colors = [int(v) for v in rng.choice(np.array(PALETTE, dtype=int), size=2, replace=False)]
    phase = int(rng.integers(0, 2))
    defects = _choose_defects(rng, inner_h, inner_w)
    margins = {side: 0 for side in SIDES}
    for defect in defects:
        side = str(defect["side"])
        ray_len = _ray_distance(side, inner_h, inner_w, int(defect["r"]), int(defect["c"]))
        margins[side] = max(margins[side], ray_len)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "colors": colors,
        "phase": phase,
        "defects": defects,
        "margins": margins,
        "expanded_h": margins["up"] + outer_h + margins["down"],
        "expanded_w": margins["left"] + outer_w + margins["right"],
    }


def _place_specs(
    rng: np.random.Generator, specs: list[dict[str, Any]], height: int, width: int
) -> list[dict[str, Any]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, Any] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["expanded_h"] * specs[idx]["expanded_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        expanded_h = int(spec["expanded_h"])
        expanded_w = int(spec["expanded_w"])
        if expanded_h + 2 > height or expanded_w + 2 > width:
            return None
        done = False
        for _ in range(200):
            er0 = int(rng.integers(0, height - expanded_h + 1))
            ec0 = int(rng.integers(0, width - expanded_w + 1))
            er1 = er0 + expanded_h - 1
            ec1 = ec0 + expanded_w - 1
            check_r0 = max(0, er0 - 1)
            check_c0 = max(0, ec0 - 1)
            check_r1 = min(height - 1, er1 + 1)
            check_c1 = min(width - 1, ec1 + 1)
            if occupied[check_r0 : check_r1 + 1, check_c0 : check_c1 + 1].any():
                continue
            spec = dict(spec)
            spec["bbox"] = (
                er0 + int(spec["margins"]["up"]),
                ec0 + int(spec["margins"]["left"]),
                er0 + int(spec["margins"]["up"]) + int(spec["outer_h"]) - 1,
                ec0 + int(spec["margins"]["left"]) + int(spec["outer_w"]) - 1,
            )
            occupied[check_r0 : check_r1 + 1, check_c0 : check_c1 + 1] = True
            placed[idx] = spec
            done = True
            break
        if not done:
            return None
    return [spec for spec in placed if spec is not None]


def _render_input(specs: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        inner = np.zeros((spec["inner_h"], spec["inner_w"]), dtype=int)
        for r in range(spec["inner_h"]):
            for c in range(spec["inner_w"]):
                inner[r, c] = int(spec["colors"][(r + c + int(spec["phase"])) % 2])
        for defect in spec["defects"]:
            inner[int(defect["r"]), int(defect["c"])] = BG
        grid[r0 + 1 : r1, c0 + 1 : c1] = inner
    return grid


def _render_expected(grid: np.ndarray, specs: list[dict[str, Any]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        for defect in spec["defects"]:
            r = int(defect["r"])
            c = int(defect["c"])
            side = str(defect["side"])
            color = int(spec["colors"][(r + c + int(spec["phase"])) % 2])
            ray_len = _ray_distance(side, inner_h, inner_w, r, c)
            start, delta = _ray_start_and_delta(spec["bbox"], r, c, side)
            rr, cc = start
            dr, dc = delta
            for _ in range(ray_len):
                if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                    raise AssertionError("ray left the grid")
                if int(grid[rr, cc]) != BG:
                    raise AssertionError("ray overlaps an existing object")
                if int(out[rr, cc]) not in {BG, color}:
                    raise AssertionError("rays overlap with conflicting colors")
                out[rr, cc] = color
                rr += dr
                cc += dc
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = RNG
    count = int(rng.integers(2, 4))
    specs = [_make_container_spec(rng) for _ in range(count)]

    used_sides = {str(defect["side"]) for spec in specs for defect in spec["defects"]}
    if len(used_sides) < 2:
        return None

    total_area = sum(int(spec["expanded_h"]) * int(spec["expanded_w"]) for spec in specs)
    max_h = max(int(spec["expanded_h"]) for spec in specs)
    max_w = max(int(spec["expanded_w"]) for spec in specs)
    side_hint = int(np.ceil(np.sqrt(total_area * 1.8)))
    min_h = max(22, max_h + 2, side_hint - 2)
    min_w = max(22, max_w + 2, side_hint - 2)
    if min_h > 34 or min_w > 34:
        return None
    height = int(rng.integers(min_h, 35))
    width = int(rng.integers(min_w, 35))

    placed = _place_specs(rng, specs, height, width)
    if placed is None or len(placed) != count:
        return None

    grid = _render_input(placed, height, width)
    expected = _render_expected(grid, placed)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if int((actual != grid).sum()) < count * 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 300 attempts")
