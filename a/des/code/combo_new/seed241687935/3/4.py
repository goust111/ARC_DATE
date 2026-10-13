from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_PRIORITY = {"top": 0, "right": 1, "bottom": 2, "left": 3}
DIRS = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _frame_record(r0: int, c0: int, r1: int, c1: int) -> dict[str, int]:
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "it": int(r0 + 1),
        "ib": int(r1 - 1),
        "il": int(c0 + 1),
        "ir": int(c1 - 1),
    }


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((int(r0), int(c)))
        pts.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        pts.add((int(r), int(c0)))
        pts.add((int(r), int(c1)))
    return pts


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for comp in _components(grid == FRAME):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if set(comp) != _rect_border(int(r0), int(c0), int(r1), int(c1)):
            continue
        frames.append(_frame_record(int(r0), int(c0), int(r1), int(c1)))
    frames.sort(key=lambda f: (f["r0"], f["c0"], f["r1"], f["c1"]))
    return frames


def _marker_info(frame: dict[str, int], r: int, c: int) -> dict[str, object] | None:
    it, ib, il, ir = frame["it"], frame["ib"], frame["il"], frame["ir"]
    width = int(ir - il)
    height = int(ib - it)
    candidates: list[tuple[str, int, int]] = []

    if r == it and il <= c <= ir:
        dist = int(min(c - il, ir - c))
        perimeter = int(c - il)
        candidates.append(("top", dist, perimeter))
    if c == ir and it <= r <= ib:
        dist = int(min(r - it, ib - r))
        perimeter = int(width + (r - it))
        candidates.append(("right", dist, perimeter))
    if r == ib and il <= c <= ir:
        dist = int(min(c - il, ir - c))
        perimeter = int(width + height + (ir - c))
        candidates.append(("bottom", dist, perimeter))
    if c == il and it <= r <= ib:
        dist = int(min(r - it, ib - r))
        perimeter = int(width + height + width + (ib - r))
        candidates.append(("left", dist, perimeter))

    if not candidates:
        return None

    side, dist, perimeter = min(candidates, key=lambda item: SIDE_PRIORITY[item[0]])
    return {
        "pos": (int(r), int(c)),
        "side": side,
        "corner_distance": int(dist),
        "order": (int(dist), int(SIDE_PRIORITY[side]), int(perimeter)),
    }


def _collect_markers(grid: np.ndarray, frame: dict[str, int], bg: int) -> list[dict[str, object]]:
    markers: list[dict[str, object]] = []
    colors = set()
    for r in range(frame["it"], frame["ib"] + 1):
        for c in range(frame["il"], frame["ir"] + 1):
            value = int(grid[r, c])
            if value in (bg, FRAME):
                continue
            info = _marker_info(frame, int(r), int(c))
            assert info is not None, "marker must lie on an inner frame edge"
            info["color"] = value
            markers.append(info)
            colors.add(value)
    assert markers, "each frame must contain at least one marker"
    assert len(colors) == len(markers), "marker colors must be unique within a frame"
    return markers


def _fire_rays(
    out: np.ndarray,
    frame: dict[str, int],
    markers: list[dict[str, object]],
    bg: int,
) -> list[dict[str, object]]:
    del frame
    ray_cells: set[tuple[int, int]] = set()
    marker_positions = {tuple(m["pos"]) for m in markers}
    results: list[dict[str, object]] = []

    for marker in sorted(markers, key=lambda m: m["order"]):
        r, c = marker["pos"]
        dr, dc = DIRS[str(marker["side"])]
        cur_r, cur_c = int(r), int(c)
        painted = 0
        reason = "border"

        while True:
            nr = int(cur_r + dr)
            nc = int(cur_c + dc)
            cell = int(out[nr, nc])
            if cell == FRAME:
                reason = "border"
                break
            if (nr, nc) in ray_cells:
                reason = "ray"
                break
            if (nr, nc) in marker_positions or cell != bg:
                reason = "occupied"
                break

            out[nr, nc] = int(marker["color"])
            ray_cells.add((nr, nc))
            painted += 1
            cur_r, cur_c = nr, nc

        results.append(
            {
                "marker": marker,
                "reason": reason,
                "painted": int(painted),
            }
        )

    return results


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project inward rays from colored inner-edge markers inside each gray frame.

    Input structure:
    - The most common color is black background.
    - Several disjoint gray one-cell-thick rectangular hollow frames appear on the grid.
    - Single colored marker pixels lie on the inner top, right, bottom, or left edge of each frame.
    - Every other cell inside each frame is background.

    Rule:
    - For each marker, determine which inner edge it sits on and fire a same-colored ray toward the frame interior.
    - Process markers within each frame from smaller nearest-corner distance to larger; equal distances use the side order top, right, bottom, left.
    - Extend each ray through background cells only, stopping before an already drawn ray pixel or before the opposite gray border.
    - Keep the original gray frame and the original marker pixels unchanged.

    Key insight:
    - The marker side gives the ray direction, but the final ray length depends on the hidden firing order because earlier rays become blockers for later ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    frames = _find_frames(g)
    assert frames, "expected at least one gray frame"

    out = g.copy()
    for frame in frames:
        markers = _collect_markers(g, frame, bg)
        _fire_rays(out, frame, markers, bg)
    return out


def _blank_frame_grid(inner_h: int, inner_w: int) -> np.ndarray:
    grid = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME
    return grid


def _frame_candidates(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    positions.extend((1, c) for c in range(2, inner_w))
    positions.extend((r, inner_w) for r in range(2, inner_h))
    positions.extend((inner_h, c) for c in range(2, inner_w))
    positions.extend((r, 1) for r in range(2, inner_h))
    return [(int(r), int(c)) for r, c in positions]


def _simulate_positions(inner_h: int, inner_w: int, positions: list[tuple[int, int]]) -> tuple[list[dict[str, object]], np.ndarray] | None:
    grid = _blank_frame_grid(int(inner_h), int(inner_w))
    for idx, (r, c) in enumerate(positions):
        grid[int(r), int(c)] = int(PALETTE[idx])

    frame = _frame_record(0, 0, int(inner_h + 1), int(inner_w + 1))
    markers = _collect_markers(grid, frame, BG)
    if len({(m["side"], m["corner_distance"]) for m in markers}) != len(markers):
        return None

    out = grid.copy()
    results = _fire_rays(out, frame, markers, BG)
    return results, out


def _sample_simple_positions(inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    candidates = _frame_candidates(int(inner_h), int(inner_w))
    max_markers = min(4, len(candidates))
    for _ in range(250):
        count = random.randint(2, max_markers)
        positions = random.sample(candidates, count)
        sim = _simulate_positions(int(inner_h), int(inner_w), positions)
        if sim is None:
            continue
        results, _out = sim
        if any(int(res["painted"]) == 0 for res in results):
            continue
        if any(str(res["reason"]) != "border" for res in results):
            continue
        if len({str(res["marker"]["side"]) for res in results}) < 2:
            continue
        return [(int(r), int(c)) for r, c in positions]
    return None


def _collision_seed_options(inner_h: int, inner_w: int) -> list[list[tuple[int, int]]]:
    max_d = min((inner_h - 1) // 2, (inner_w - 1) // 2)
    options: list[list[tuple[int, int]]] = []
    for d in range(2, max_d + 1):
        options.append([(1, 1 + d), (1 + d, 1)])
        options.append([(1, inner_w - d), (1 + d, inner_w)])
        options.append([(inner_h - d, inner_w), (inner_h, inner_w - d)])
        options.append([(inner_h, 1 + d), (inner_h - d, 1)])
    return [[(int(r), int(c)) for r, c in pair] for pair in options]


def _sample_collision_positions(inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    options = _collision_seed_options(int(inner_h), int(inner_w))
    if not options:
        return None
    candidates = _frame_candidates(int(inner_h), int(inner_w))
    random.shuffle(options)
    for base_pair in options:
        remaining = [pos for pos in candidates if pos not in base_pair]
        max_markers = min(4, len(remaining) + len(base_pair))
        for _ in range(180):
            count = random.randint(2, max_markers)
            extras = random.sample(remaining, count - len(base_pair))
            positions = base_pair + extras
            sim = _simulate_positions(int(inner_h), int(inner_w), positions)
            if sim is None:
                continue
            results, _out = sim
            reasons = [str(res["reason"]) for res in results]
            if "occupied" in reasons or "ray" not in reasons:
                continue
            if any(int(res["painted"]) == 0 for res in results):
                continue
            return [(int(r), int(c)) for r, c in positions]
    return None


def _make_frame_spec(want_collision: bool) -> dict[str, object] | None:
    for _ in range(120):
        if want_collision:
            inner_h = random.randint(5, 10)
            inner_w = random.randint(5, 10)
            positions = _sample_collision_positions(int(inner_h), int(inner_w))
        else:
            inner_h = random.randint(4, 10)
            inner_w = random.randint(4, 10)
            positions = _sample_simple_positions(int(inner_h), int(inner_w))
        if positions is None:
            continue

        colors = random.sample(PALETTE, len(positions))
        grid = _blank_frame_grid(int(inner_h), int(inner_w))
        for color, (r, c) in zip(colors, positions):
            grid[int(r), int(c)] = int(color)

        sim = _simulate_positions(int(inner_h), int(inner_w), positions)
        assert sim is not None
        results, out = sim
        reasons = [str(res["reason"]) for res in results]
        if want_collision and "ray" not in reasons:
            continue
        if not want_collision and any(reason != "border" for reason in reasons):
            continue

        return {
            "grid": grid,
            "out": out,
            "shape": tuple(int(v) for v in grid.shape),
            "collision": bool(want_collision),
        }
    return None


def _place_frames(scene_h: int, scene_w: int, specs: list[dict[str, object]]) -> list[tuple[dict[str, object], int, int]] | None:
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    placements: list[tuple[dict[str, object], int, int]] = []
    ordered = sorted(specs, key=lambda spec: int(np.prod(spec["shape"])), reverse=True)

    for spec in ordered:
        frame_h, frame_w = spec["shape"]
        if frame_h > scene_h or frame_w > scene_w:
            return None
        min_r = 1 if scene_h - frame_h >= 2 else 0
        max_r = scene_h - frame_h - (1 if scene_h - frame_h >= 2 else 0)
        min_c = 1 if scene_w - frame_w >= 2 else 0
        max_c = scene_w - frame_w - (1 if scene_w - frame_w >= 2 else 0)
        if max_r < min_r or max_c < min_c:
            return None

        placed = False
        for _ in range(250):
            r0 = random.randint(int(min_r), int(max_r))
            c0 = random.randint(int(min_c), int(max_c))
            rr0 = max(0, int(r0 - 1))
            cc0 = max(0, int(c0 - 1))
            rr1 = min(scene_h, int(r0 + frame_h + 1))
            cc1 = min(scene_w, int(c0 + frame_w + 1))
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue
            occupied[r0 : r0 + frame_h, c0 : c0 + frame_w] = True
            placements.append((spec, int(r0), int(c0)))
            placed = True
            break
        if not placed:
            return None

    return placements


def _render_scene(scene_h: int, scene_w: int, placements: list[tuple[dict[str, object], int, int]]) -> np.ndarray:
    grid = np.full((scene_h, scene_w), BG, dtype=int)
    for spec, r0, c0 in placements:
        patch = np.asarray(spec["grid"], dtype=int)
        frame_h, frame_w = patch.shape
        window = grid[r0 : r0 + frame_h, c0 : c0 + frame_w]
        mask = patch != BG
        window[mask] = patch[mask]
    return grid


def _gen_attempt() -> np.ndarray | None:
    need_collision = bool(random.getrandbits(1))
    n_frames = random.choice([1, 2, 2, 3, 3, 4])
    collision_index = random.randrange(n_frames) if need_collision else -1

    specs: list[dict[str, object]] = []
    for idx in range(n_frames):
        spec = _make_frame_spec(idx == collision_index)
        if spec is None:
            return None
        specs.append(spec)

    total_area = sum(int(np.prod(spec["shape"])) for spec in specs)
    max_h = max(int(spec["shape"][0]) for spec in specs)
    max_w = max(int(spec["shape"][1]) for spec in specs)
    base = int(np.ceil(np.sqrt(total_area * 2.2)))
    scene_h_min = max(12, max_h + 2, base - 2)
    scene_w_min = max(12, max_w + 2, base - 2)
    scene_h_max = min(30, max(scene_h_min, base + 6))
    scene_w_max = min(30, max(scene_w_min, base + 6))
    if scene_h_min > 30 or scene_w_min > 30:
        return None

    for _ in range(60):
        scene_h = random.randint(scene_h_min, scene_h_max)
        scene_w = random.randint(scene_w_min, scene_w_max)
        placements = _place_frames(int(scene_h), int(scene_w), specs)
        if placements is None:
            continue

        grid = _render_scene(int(scene_h), int(scene_w), placements)
        if _most_common_color(grid) != BG:
            continue
        if len(_find_frames(grid)) != n_frames:
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape or bool(np.array_equal(out, grid)):
            continue
        if need_collision and not any(bool(spec["collision"]) for spec in specs):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
