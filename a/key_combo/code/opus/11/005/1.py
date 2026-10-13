from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GENERATION_ATTEMPTS = 600
FRAME_RETRIES = 120
SOURCE_SELECTION_RETRIES = 220
TEMP_SOURCE_COLOR = 9


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for coords in _component_coords(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        color = int(grid[coords[0]])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        area = len(coords)
        if area != 2 * (h + w) - 4:
            continue
        border_ok = True
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border and int(grid[rr, cc]) != color:
                    border_ok = False
                    break
                if (not on_border) and int(grid[rr, cc]) != bg:
                    border_ok = False
                    break
            if not border_ok:
                break
        if not border_ok:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "capacity": ((w - 2) + 1) // 2,
            }
        )
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _detect_sources(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    frame_mask = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_mask[r0 : r1 + 1, c0 : c1 + 1] |= (
            (np.arange(r0, r1 + 1)[:, None] == r0)
            | (np.arange(r0, r1 + 1)[:, None] == r1)
            | (np.arange(c0, c1 + 1)[None, :] == c0)
            | (np.arange(c0, c1 + 1)[None, :] == c1)
        )

    src_mask = (grid != bg) & ~frame_mask
    sources: list[dict] = []
    for coords in _component_coords(src_mask):
        if len(coords) != 1:
            raise ValueError("non-frame foreground components must be isolated source pixels")
        r, c = coords[0]
        sources.append({"pos": (r, c), "color": int(grid[r, c])})
    sources.sort(key=lambda s: s["pos"])
    return sources


def _interior_center_row(frame: dict) -> int:
    r0, _, r1, _ = frame["inner_bbox"]
    return (r0 + r1) // 2


def _dot_columns(frame: dict) -> list[int]:
    _, c0, _, c1 = frame["inner_bbox"]
    return list(range(c1, c0 - 1, -2))


def _simulate(
    grid: np.ndarray,
    *,
    allow_frame_entry: bool = True,
    continue_through_frames: bool = False,
    source_order: str = "forward",
) -> dict:
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    frames = _detect_frames(g, bg)
    sources = _detect_sources(g, bg, frames)
    border_to_frame: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for rr in range(r0, r1 + 1):
            border_to_frame[(rr, c0)] = idx
            border_to_frame[(rr, c1)] = idx
        for cc in range(c0, c1 + 1):
            border_to_frame[(r0, cc)] = idx
            border_to_frame[(r1, cc)] = idx

    out = g.copy()
    counts = [0] * len(frames)
    ray_infos: list[dict] = []
    ordered_sources = sources if source_order == "forward" else list(reversed(sources))

    for src in ordered_sources:
        sr, sc = src["pos"]
        color = int(src["color"])
        for dr, dc in DIAGONALS:
            r, c = sr, sc
            active_frame = None
            painted: list[tuple[int, int]] = []
            entered_frame = None
            stop_reason = "boundary"
            while True:
                nr, nc = r + dr, c + dc
                if not _in_bounds(out.shape, nr, nc):
                    stop_reason = "boundary"
                    break

                val = int(out[nr, nc])
                frame_idx = border_to_frame.get((nr, nc))
                if val != bg:
                    if (
                        allow_frame_entry
                        and frame_idx is not None
                        and active_frame is None
                    ):
                        rr, cc = nr + dr, nc + dc
                        if _in_bounds(out.shape, rr, cc):
                            ir0, ic0, ir1, ic1 = frames[frame_idx]["inner_bbox"]
                            if ir0 <= rr <= ir1 and ic0 <= cc <= ic1 and int(out[rr, cc]) == bg:
                                out[rr, cc] = color
                                painted.append((rr, cc))
                                active_frame = frame_idx
                                entered_frame = frame_idx
                                r, c = rr, cc
                                continue
                    if active_frame is not None and frame_idx == active_frame and continue_through_frames:
                        rr, cc = nr + dr, nc + dc
                        if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == bg:
                            out[rr, cc] = color
                            painted.append((rr, cc))
                            r, c = rr, cc
                            active_frame = None
                            continue
                    stop_reason = "blocker"
                    break

                out[nr, nc] = color
                painted.append((nr, nc))
                r, c = nr, nc

            if entered_frame is not None:
                counts[entered_frame] += 1
            ray_infos.append(
                {
                    "source": src["pos"],
                    "dir": (dr, dc),
                    "entered_frame": entered_frame,
                    "painted": painted,
                    "stop_reason": stop_reason,
                }
            )

    for frame, count in zip(frames, counts):
        row = _interior_center_row(frame)
        for col in _dot_columns(frame)[:count]:
            out[row, col] = int(frame["color"])

    return {
        "output": out,
        "counts": counts,
        "frames": frames,
        "sources": sources,
        "rays": ray_infos,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from isolated pixels, count the frame interiors they traverse, and display each count as dots inside that frame.

    Input structure:
    - The grid has black background, 2-4 isolated colored source pixels, and 2-4 one-cell-thick hollow rectangular frames.
    - Every frame is monochrome, non-overlapping, and its interior is black in the input.
    - Source pixels are outside all frames and do not share rows or columns.

    Rule:
    - From each source pixel, emit four diagonal rays. A ray normally paints black cells until the next step would leave the grid or land on a non-black pixel.
    - When a ray's next step is a frame-border cell and the following diagonal step would enter that frame's interior, the ray skips the near border, paints through the interior, and then stops at the next blocker; this makes nearer frames occlude farther ones.
    - Count, for each frame, how many source-direction rays paint at least one interior cell of that frame.
    - Keep the original sources and frame borders, keep all ray pixels, and place that many same-colored dots on the frame's interior center row from the rightmost interior column leftward with a one-column gap. If a dot overlaps a ray pixel, keep the dot.

    Key insight:
    - The rays are determined by source position and diagonal direction, but the visible counts come from where those rays can actually enter a frame interior before being truncated by a frame boundary, so frame placement changes both traversal and tally.
    """
    return _simulate(grid)["output"]


def _frame_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells = set()
    for rr in range(r0, r1 + 1):
        cells.add((rr, c0))
        cells.add((rr, c1))
    for cc in range(c0, c1 + 1):
        cells.add((r0, cc))
        cells.add((r1, cc))
    return cells


def _cell_is_isolated_bg(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    for nr, nc in _neighbors8(r, c):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG:
            return False
    return True


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_frames(rng: random.Random, h: int, w: int, n_frames: int) -> list[dict] | None:
    frame_colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], n_frames)
    sizes = [(rng.randint(5, 8), rng.randint(6, 8)) for _ in range(n_frames)]
    sizes.sort(key=lambda sz: sz[0] * sz[1], reverse=True)

    placed: list[dict] = []

    def clear(box: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = box
        for other in placed:
            or0, oc0, or1, oc1 = other["bbox"]
            if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
                return False
        return True

    def search(i: int) -> bool:
        if i == n_frames:
            return True
        fh, fw = sizes[i]
        positions = [(r, c) for r in range(1, h - fh) for c in range(1, w - fw)]
        rng.shuffle(positions)
        for r0, c0 in positions:
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if not clear(box):
                continue
            frame = {
                "bbox": box,
                "color": frame_colors[i],
                "inner_bbox": (box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1),
                "capacity": ((fw - 2) + 1) // 2,
            }
            placed.append(frame)
            if search(i + 1):
                return True
            placed.pop()
        return False

    if not search(0):
        return None
    placed.sort(key=lambda f: f["bbox"])
    return placed


def _single_source_candidates(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    trial = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not _cell_is_isolated_bg(grid, r, c):
                continue
            trial[r, c] = TEMP_SOURCE_COLOR
            try:
                sim = _simulate(trial)
            except ValueError:
                trial[r, c] = BG
                continue
            trial[r, c] = BG
            total = sum(sim["counts"])
            if total == 0:
                continue
            if any(count > frame["capacity"] for count, frame in zip(sim["counts"], frames)):
                continue
            cover = frozenset(i for i, count in enumerate(sim["counts"]) if count > 0)
            candidates.append(
                {
                    "pos": (r, c),
                    "counts": tuple(sim["counts"]),
                    "cover": cover,
                    "total": total,
                }
            )
    return candidates


def _far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _choose_sources(
    rng: random.Random,
    base_grid: np.ndarray,
    frames: list[dict],
    need_shadow: bool,
    need_multicount: bool,
) -> np.ndarray | None:
    candidates = _single_source_candidates(base_grid, frames)
    if len(candidates) < 2:
        return None

    all_colors = set(range(1, 10))
    frame_colors = {int(frame["color"]) for frame in frames}
    source_colors_pool = sorted(all_colors - frame_colors)
    if len(source_colors_pool) < 2:
        return None

    for _ in range(SOURCE_SELECTION_RETRIES):
        n_sources = rng.randint(2, 4)
        chosen: list[dict] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        uncovered = set(range(len(frames)))
        pool = candidates[:]
        rng.shuffle(pool)

        while len(chosen) < n_sources:
            feasible = [
                cand
                for cand in pool
                if cand["pos"][0] not in used_rows
                and cand["pos"][1] not in used_cols
                and all(_far_enough(cand["pos"], prev["pos"]) for prev in chosen)
            ]
            if not feasible:
                break
            coverers = [cand for cand in feasible if cand["cover"] & uncovered]
            if coverers:
                coverers.sort(key=lambda cand: (len(cand["cover"] & uncovered), cand["total"]), reverse=True)
                top_k = coverers[: min(8, len(coverers))]
                cand = rng.choice(top_k)
            else:
                feasible.sort(key=lambda cand: cand["total"], reverse=True)
                cand = rng.choice(feasible[: min(8, len(feasible))])
            chosen.append(cand)
            used_rows.add(cand["pos"][0])
            used_cols.add(cand["pos"][1])
            uncovered -= set(cand["cover"])
            pool.remove(cand)

            if not uncovered and len(chosen) >= 2 and rng.random() < 0.4:
                break

        if uncovered or len(chosen) < 2:
            continue

        if len(source_colors_pool) < len(chosen):
            continue
        source_colors = rng.sample(source_colors_pool, len(chosen))
        inp = base_grid.copy()
        for cand, color in zip(chosen, source_colors):
            r, c = cand["pos"]
            inp[r, c] = color

        try:
            sim = _simulate(inp)
            sim_rev = _simulate(inp, source_order="reverse")
        except ValueError:
            continue

        counts = sim["counts"]
        if len(sim["sources"]) != len(chosen):
            continue
        if counts != sim_rev["counts"] or not np.array_equal(sim["output"], sim_rev["output"]):
            continue
        if any(count == 0 for count in counts):
            continue
        if any(count > frame["capacity"] for count, frame in zip(counts, frames)):
            continue
        if need_multicount and max(counts) < 2:
            continue
        if sum(1 for ray in sim["rays"] if ray["entered_frame"] is not None) != sum(counts):
            continue
        if sum(counts) >= len(chosen) * 4:
            continue
        if np.array_equal(inp, sim["output"]):
            continue

        shadow_sim = _simulate(inp, continue_through_frames=True)
        if need_shadow and shadow_sim["counts"] == counts:
            continue

        edge_stops = sum(1 for ray in sim["rays"] if ray["stop_reason"] == "boundary")
        if edge_stops == 0:
            continue

        return inp
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    n_frames = rng.randint(2, 4)
    need_shadow = rng.random() < 0.65
    need_multicount = rng.random() < 0.7

    for _ in range(FRAME_RETRIES):
        frames = _place_frames(rng, h, w, n_frames)
        if frames is None:
            continue
        base = np.zeros((h, w), dtype=int)
        for frame in frames:
            _draw_frame(base, frame)
        inp = _choose_sources(rng, base, frames, need_shadow, need_multicount)
        if inp is None:
            continue
        final = _simulate(inp)
        if len(final["frames"]) != n_frames:
            continue
        if len(final["sources"]) < 2 or len(final["sources"]) > 4:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
