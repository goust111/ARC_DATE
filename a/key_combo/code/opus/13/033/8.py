from __future__ import annotations

from collections import Counter, deque
import hashlib
import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PARTITION_TEMPLATES = (
    np.array([[0, 1, 2], [0, 3, 2], [4, 3, 5]], dtype=int),
    np.array([[0, 1, 1], [2, 3, 1], [2, 3, 4]], dtype=int),
    np.array([[0, 1, 2], [3, 1, 4], [5, 6, 4]], dtype=int),
    np.array([[0, 1, 2], [0, 3, 2], [4, 5, 5]], dtype=int),
)
CELL_SIZE_RANGE = (2, 4)
GEN_ATTEMPTS = 400
ASSIGN_ATTEMPTS = 120

MODE_SPECS = (
    {"need_isolated": True, "need_same_color_adj": False, "need_multi": False},
    {"need_isolated": False, "need_same_color_adj": True, "need_multi": False},
    {"need_isolated": True, "need_same_color_adj": False, "need_multi": True},
    {"need_isolated": False, "need_same_color_adj": True, "need_multi": True},
)

_GEN_CALLS = 0
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _sig(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=np.int8)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _dist2(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(a[0]) - int(b[0])
    dc = int(a[1]) - int(b[1])
    return dr * dr + dc * dc


def _render_partition(coarse_labels: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    rows, cols = coarse_labels.shape
    row_starts: list[int] = []
    col_starts: list[int] = []
    pos = 0
    for h in row_heights:
        row_starts.append(pos)
        pos += h + 1
    total_h = pos - 1
    pos = 0
    for w in col_widths:
        col_starts.append(pos)
        pos += w + 1
    total_w = pos - 1

    grid = np.zeros((total_h, total_w), dtype=int)
    for r in range(rows):
        for c in range(cols - 1):
            if int(coarse_labels[r, c]) == int(coarse_labels[r, c + 1]):
                continue
            wall_c = col_starts[c] + col_widths[c]
            r0 = row_starts[r]
            r1 = r0 + row_heights[r]
            grid[r0:r1, wall_c] = GREY
    for r in range(rows - 1):
        for c in range(cols):
            if int(coarse_labels[r, c]) == int(coarse_labels[r + 1, c]):
                continue
            wall_r = row_starts[r] + row_heights[r]
            c0 = col_starts[c]
            c1 = c0 + col_widths[c]
            grid[wall_r, c0:c1] = GREY
    return grid


def _partition_meta(grid: np.ndarray) -> dict | None:
    comps = _components(grid != GREY)
    if not (4 <= len(comps) <= 7):
        return None

    region_id = np.full(grid.shape, -1, dtype=int)
    regions = []
    for rid, coords in enumerate(comps):
        region_id[coords[:, 0], coords[:, 1]] = rid
        regions.append({"coords": coords})

    adjacency = {rid: set() for rid in range(len(regions))}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY:
                continue
            if 0 < r < h - 1:
                a = int(region_id[r - 1, c])
                b = int(region_id[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
            if 0 < c < w - 1:
                a = int(region_id[r, c - 1])
                b = int(region_id[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    for rid, region in enumerate(regions):
        coords = region["coords"]
        inner = [
            (int(r), int(c))
            for r, c in map(tuple, coords)
            if 0 < int(r) < h - 1 and 0 < int(c) < w - 1
        ]
        if not inner:
            return None
        region["inner"] = inner
        region["bbox"] = (
            int(coords[:, 0].min()),
            int(coords[:, 1].min()),
            int(coords[:, 0].max()),
            int(coords[:, 1].max()),
        )
    return {"regions": regions, "adjacency": adjacency, "region_id": region_id}


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30
    present = {int(v) for v in np.unique(g)}
    assert present <= {BG, BLUE, RED, GREEN, YELLOW, GREY}
    assert GREY in present

    comps = _components(g != GREY)
    region_id = np.full(g.shape, -1, dtype=int)
    regions = []
    for rid, coords in enumerate(comps):
        region_id[coords[:, 0], coords[:, 1]] = rid
        marker_cells = [
            (int(r), int(c))
            for r, c in map(tuple, coords)
            if int(g[r, c]) in MARKER_COLORS
        ]
        assert len(marker_cells) <= 1, "each region must contain at most one marker"
        marker = marker_cells[0] if marker_cells else None
        if marker is not None:
            assert 0 < marker[0] < h - 1 and 0 < marker[1] < w - 1
        regions.append(
            {
                "coords": coords,
                "marker": marker,
                "color": int(g[marker]) if marker is not None else None,
                "bbox": (
                    int(coords[:, 0].min()),
                    int(coords[:, 1].min()),
                    int(coords[:, 0].max()),
                    int(coords[:, 1].max()),
                ),
            }
        )

    adjacency = {rid: set() for rid in range(len(regions))}
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != GREY:
                continue
            if 0 < r < h - 1:
                a = int(region_id[r - 1, c])
                b = int(region_id[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
            if 0 < c < w - 1:
                a = int(region_id[r, c - 1])
                b = int(region_id[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    marked_ids = [rid for rid, region in enumerate(regions) if region["marker"] is not None]
    marked_ids.sort(key=lambda rid: regions[rid]["marker"])
    marked_neighbors = {
        rid: sorted(
            [nid for nid in adjacency[rid] if regions[nid]["marker"] is not None],
            key=lambda nid: regions[nid]["marker"],
        )
        for rid in marked_ids
    }

    taken: set[int] = set()
    pairs: list[tuple[int, int]] = []
    candidate_history: dict[int, dict] = {}
    pair_partner: dict[int, int] = {}

    for rid in marked_ids:
        if rid in taken:
            continue
        own_marker = regions[rid]["marker"]
        own_color = regions[rid]["color"]
        assert own_marker is not None and own_color is not None

        candidates = []
        for nid in adjacency[rid]:
            if nid in taken:
                continue
            other_marker = regions[nid]["marker"]
            other_color = regions[nid]["color"]
            if other_marker is None or other_color is None or other_color == own_color:
                continue
            candidates.append(
                {
                    "rid": nid,
                    "dist2": _dist2(own_marker, other_marker),
                    "marker": other_marker,
                    "color": other_color,
                }
            )
        candidates.sort(key=lambda item: (item["dist2"], item["marker"][0], item["marker"][1]))
        unique_nearest = True
        if len(candidates) >= 2 and candidates[0]["dist2"] == candidates[1]["dist2"]:
            unique_nearest = False
        candidate_history[rid] = {
            "candidates": tuple((item["rid"], item["dist2"]) for item in candidates),
            "unique_nearest": unique_nearest,
        }
        if not candidates:
            continue
        chosen = int(candidates[0]["rid"])
        taken.add(rid)
        taken.add(chosen)
        pair_partner[rid] = chosen
        pair_partner[chosen] = rid
        pairs.append((rid, chosen))

    fill_colors: dict[int, int] = {}
    for rid, region in enumerate(regions):
        marker = region["marker"]
        color = region["color"]
        if marker is None:
            continue
        if rid in pair_partner:
            fill_colors[rid] = int(regions[pair_partner[rid]]["color"])
        else:
            fill_colors[rid] = int(color)

    out = g.copy()
    for rid, region in enumerate(regions):
        color = fill_colors.get(rid)
        if color is None:
            continue
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    out[g == GREY] = GREY

    return {
        "grid": g,
        "output": out,
        "regions": regions,
        "adjacency": adjacency,
        "marked_ids": marked_ids,
        "marked_neighbors": marked_neighbors,
        "pairs": pairs,
        "pair_partner": pair_partner,
        "candidate_history": candidate_history,
        "fill_colors": fill_colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-bounded regions by swapping colors between adjacent differently marked regions and self-filling every unmatched marked region.

    Input structure:
    - A black background is partitioned by one-pixel grey wall segments into 4-connected enclosed regions; grid edges can also bound a region.
    - Some regions contain exactly one interior marker pixel colored blue, red, green, or yellow, while other region cells stay black.
    - Each marked region is identified both by its wall-defined geometry and by its single marker color.

    Rule:
    - Treat grey as the divider and find all 4-connected non-grey regions.
    - Build a region-adjacency graph: two regions are adjacent only when a grey wall pixel has one region on one side and the other region on the opposite side.
    - Visit marked regions in marker-coordinate order. An unpaired region greedily exchanges with one unpaired adjacent marked region of a different color: if several are available, pick the one whose marker is nearest by Euclidean distance.
    - For every exchange pair, fill each whole region with the partner's marker color. Every marked region left unmatched fills with its own color. Unmarked regions stay black, and grey walls stay grey.

    Key insight:
    - The answer depends on two coupled structures at once: walls define which regions can interact, and marker colors decide whether an interaction is an exchange or a self-fill.
    """
    return _analyze(grid)["output"]


def _pick_marker(region: dict, rng: random.Random) -> tuple[int, int]:
    pts = list(region["inner"])
    style = rng.randrange(4)
    if style == 0:
        return min(pts)
    if style == 1:
        return max(pts)
    if style == 2:
        r0, c0, r1, c1 = region["bbox"]
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        pts.sort(key=lambda rc: ((rc[0] - center[0]) ** 2 + (rc[1] - center[1]) ** 2, rc[0], rc[1]))
        return pts[0]
    return rng.choice(pts)


def _random_partition(rng: random.Random) -> np.ndarray:
    coarse = np.array(rng.choice(PARTITION_TEMPLATES), dtype=int)
    if rng.random() < 0.5:
        coarse = np.fliplr(coarse)
    if rng.random() < 0.5:
        coarse = np.flipud(coarse)
    if rng.random() < 0.5:
        coarse = coarse.T
    unique = {int(v): idx for idx, v in enumerate(sorted(set(map(int, coarse.flatten()))))}
    return np.vectorize(unique.get)(coarse).astype(int)


def _random_assignment(base_grid: np.ndarray, part: dict, mode: dict, rng: random.Random) -> np.ndarray | None:
    num_regions = len(part["regions"])
    max_marked = min(5, num_regions - 1)
    if max_marked < 3:
        return None

    region_ids = list(range(num_regions))
    for _ in range(ASSIGN_ATTEMPTS):
        marked_count = rng.randint(3, max_marked)
        marked_ids = set(rng.sample(region_ids, marked_count))
        grid = base_grid.copy()

        for rid in marked_ids:
            marker = _pick_marker(part["regions"][rid], rng)
            grid[marker] = rng.choice(MARKER_COLORS)

        info = _analyze(grid)
        if _semantic_ok(info, mode):
            return grid
    return None


def _semantic_ok(info: dict, mode: dict) -> bool:
    grid = info["grid"]
    out = info["output"]
    regions = info["regions"]
    marked_ids = info["marked_ids"]
    pairs = info["pairs"]
    pair_partner = info["pair_partner"]

    if not (4 <= len(regions) <= 7):
        return False
    if not (3 <= len(marked_ids) <= 5):
        return False
    if len(marked_ids) >= len(regions):
        return False
    if not pairs:
        return False
    if np.array_equal(grid, out):
        return False
    if not np.array_equal(grid[grid == GREY], out[grid == GREY]):
        return False

    self_filled = [rid for rid in marked_ids if rid not in pair_partner]
    if not self_filled:
        return False

    output_colors = set()
    for rid, region in enumerate(regions):
        coords = region["coords"]
        vals = {int(v) for v in out[coords[:, 0], coords[:, 1]]}
        if rid in marked_ids:
            if len(vals) != 1:
                return False
            val = next(iter(vals))
            output_colors.add(val)
            if rid in pair_partner:
                if val != int(regions[pair_partner[rid]]["color"]):
                    return False
            else:
                if val != int(region["color"]):
                    return False
        else:
            if vals != {BG}:
                return False

    if len(output_colors) < 2:
        return False

    if mode["need_isolated"]:
        if not any(len(info["marked_neighbors"][rid]) == 0 for rid in marked_ids):
            return False

    if mode["need_same_color_adj"]:
        same_color_adj = False
        for rid in marked_ids:
            own = int(regions[rid]["color"])
            for nid in info["marked_neighbors"][rid]:
                if int(regions[nid]["color"]) == own:
                    same_color_adj = True
                    break
            if same_color_adj:
                break
        if not same_color_adj:
            return False

    if mode["need_multi"]:
        multi = False
        for rid, hist in info["candidate_history"].items():
            if len(hist["candidates"]) >= 2 and hist["unique_nearest"]:
                multi = True
                break
        if not multi:
            return False

    return True


def _gen_partition_attempt(rng: random.Random) -> np.ndarray | None:
    coarse = _random_partition(rng)
    rows, cols = coarse.shape
    row_heights = [rng.randint(*CELL_SIZE_RANGE) for _ in range(rows)]
    col_widths = [rng.randint(*CELL_SIZE_RANGE) for _ in range(cols)]
    grid = _render_partition(coarse, row_heights, col_widths)
    if max(grid.shape) > 30:
        return None
    part = _partition_meta(grid)
    if part is None:
        return None
    return grid


def _gen_attempt(mode: dict, seed: int) -> np.ndarray | None:
    rng = random.Random(seed)
    base_grid = _gen_partition_attempt(rng)
    if base_grid is None:
        return None
    part = _partition_meta(base_grid)
    if part is None:
        return None
    candidate = _random_assignment(base_grid, part, mode, rng)
    if candidate is None:
        return None

    inp_sig = _sig(candidate)
    out_sig = _sig(main(candidate))
    if inp_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return None
    return candidate


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    mode = MODE_SPECS[_GEN_CALLS % len(MODE_SPECS)]
    seed_base = random.randrange(1 << 30)

    for attempt in range(GEN_ATTEMPTS):
        grid = _gen_attempt(mode, seed_base + attempt)
        if grid is None:
            continue
        _GEN_CALLS += 1
        _SEEN_INPUT_SIGS.add(_sig(grid))
        _SEEN_OUTPUT_SIGS.add(_sig(main(grid)))
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
