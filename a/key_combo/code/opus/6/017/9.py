from __future__ import annotations

from itertools import combinations
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

BAR_COLORS = (BLUE, GREEN, YELLOW, PINK, ORANGE)
GEN_MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _aligned_left(anchor_c0: int, width: int) -> int:
    # Integer placement closest to center alignment; odd widths use the left-leaning tie.
    return (2 * int(anchor_c0) + 2 - int(width)) // 2


def _bar_shape_cells(width: int, protrusions: tuple[int, ...]) -> list[tuple[int, int]]:
    cells = [(1, dc) for dc in range(width)]
    cells.extend((0, dc) for dc in protrusions)
    return cells


def _find_anchors(grid: np.ndarray) -> list[dict]:
    anchors = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if coords.shape[0] != 4 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        markers = []
        if r0 > 0:
            for c in range(max(0, c0 - 1), min(grid.shape[1], c1 + 2)):
                if int(grid[r0 - 1, c]) == RED:
                    markers.append(int(c))
        anchors.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "markers": tuple(sorted(markers)),
            }
        )
    anchors.sort(key=lambda a: (a["r0"], a["c0"]))
    return anchors


def _find_bars(grid: np.ndarray) -> list[dict]:
    bars = []
    for color in sorted(set(map(int, np.unique(grid))) - {BG, RED, GREY}):
        for coords in _components(grid == color):
            rows = sorted(set(int(r) for r, _ in coords))
            body_row = max(rows)
            body_cols = sorted(int(c) for r, c in coords if int(r) == body_row)
            if not body_cols:
                continue
            if body_cols != list(range(body_cols[0], body_cols[-1] + 1)):
                continue
            width = len(body_cols)
            left = body_cols[0]
            protrusions = sorted(int(c) - left for r, c in coords if int(r) == body_row - 1)
            bars.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "body_row": int(body_row),
                    "left": int(left),
                    "width": int(width),
                    "right": int(left + width - 1),
                    "protrusions": tuple(protrusions),
                }
            )
    bars.sort(key=lambda b: (b["body_row"], b["left"], b["color"]))
    return bars


def _target_anchor_index(bar: dict, anchors: list[dict]) -> int:
    body_center_r2 = 2 * int(bar["body_row"])
    body_center_c2 = int(bar["left"] + bar["right"])
    best_idx = -1
    best_key = None
    for idx, anchor in enumerate(anchors):
        dist = abs(body_center_r2 - (2 * anchor["r0"] + 1)) + abs(body_center_c2 - (2 * anchor["c0"] + 1))
        key = (dist, anchor["c0"], anchor["r0"])
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx
    return best_idx


def _match_score(bar: dict, anchor: dict) -> int:
    dest_left = _aligned_left(anchor["c0"], bar["width"])
    cols = {dest_left + dc for dc in bar["protrusions"]}
    return len(cols & set(anchor["markers"]))


def _analyze(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, list[dict]]]:
    anchors = _find_anchors(grid)
    bars = _find_bars(grid)
    by_anchor: dict[int, list[dict]] = {i: [] for i in range(len(anchors))}
    for bar in bars:
        target = _target_anchor_index(bar, anchors)
        score = _match_score(bar, anchors[target])
        bar["target"] = int(target)
        bar["score"] = int(score)
        by_anchor[target].append(bar)
    for idx, group in by_anchor.items():
        group.sort(key=lambda b: (-b["score"], b["width"], b["body_row"], b["left"], b["color"]))
    return anchors, bars, by_anchor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Snap competing toothed bars onto the nearest grey anchors, keeping only the best projection-column match.

    Input structure:
    - The background is black.
    - There are 2x2 grey anchor blocks, each with several red marker pixels one row above it.
    - There are 3-5 colored horizontal bars; each bar has a 1-row body and some same-color upward protrusions.

    Rule:
    - For each colored bar, choose the nearest anchor by Manhattan distance from body center to anchor center; column-leftmost anchor breaks ties.
    - Score that bar against its chosen anchor by centering it horizontally above the anchor and counting how many protrusion columns overlap the anchor's red marker columns.
    - If multiple bars chose the same anchor, keep the one with highest score; equal scores prefer the shorter body, then the higher original row.
    - Erase every bar from its input position; move each winning bar to sit directly above its anchor with the body two rows above the anchor top, preserving its protrusions.

    Key insight:
    - Distance only chooses which anchor a bar competes for; the actual winner is determined by projection-to-marker column overlap.
    """
    g = np.asarray(grid, dtype=int)
    anchors, bars, by_anchor = _analyze(g)

    out = g.copy()
    for bar in bars:
        for r, c in bar["coords"]:
            out[int(r), int(c)] = BG

    for idx, anchor in enumerate(anchors):
        if not by_anchor[idx]:
            continue
        winner = by_anchor[idx][0]
        body_row = anchor["r0"] - 2
        left = _aligned_left(anchor["c0"], winner["width"])
        out[body_row, left : left + winner["width"]] = winner["color"]
        for dc in winner["protrusions"]:
            out[body_row - 1, left + dc] = winner["color"]

    return out


def _place_anchors(rng: np.random.Generator, h: int, w: int, n_anchors: int) -> list[dict] | None:
    anchors = []
    attempts = 0
    while len(anchors) < n_anchors and attempts < 400:
        attempts += 1
        r0 = int(rng.integers(4, h - 2))
        c0 = int(rng.integers(2, w - 3))
        rect = (r0 - 3, c0 - 2, r0 + 1, c0 + 4)
        if any(not (rect[2] < old[0] or old[2] < rect[0] or rect[3] < old[1] or old[3] < rect[1]) for old in [a["zone"] for a in anchors]):
            continue
        marker_candidates = [c0 - 1, c0, c0 + 1, c0 + 2]
        marker_count = int(rng.integers(2, 5))
        markers = tuple(sorted(int(v) for v in rng.choice(marker_candidates, size=marker_count, replace=False)))
        anchors.append({"r0": r0, "c0": c0, "markers": markers, "zone": rect})
    if len(anchors) != n_anchors:
        return None
    anchors.sort(key=lambda a: (a["r0"], a["c0"]))
    return anchors


def _shape_candidates(anchor: dict) -> list[dict]:
    out = []
    for width in (3, 4, 5):
        for k in (1, 2, 3):
            if k > width:
                continue
            for protrusions in combinations(range(width), k):
                bar = {"width": width, "protrusions": tuple(int(v) for v in protrusions)}
                score = _match_score(bar, anchor)
                out.append({"width": width, "protrusions": bar["protrusions"], "score": int(score)})
    return out


def _pick_pair(cands: list[dict], mode: str, rng: np.random.Generator) -> tuple[dict, dict] | None:
    pairs = []
    for a, b in combinations(cands, 2):
        if mode == "score":
            if a["score"] != b["score"]:
                winner, loser = (a, b) if a["score"] > b["score"] else (b, a)
                pairs.append((winner, loser))
        elif mode == "length":
            if a["score"] == b["score"] and a["width"] != b["width"]:
                winner, loser = (a, b) if a["width"] < b["width"] else (b, a)
                pairs.append((winner, loser))
        else:
            if a["score"] == b["score"] and a["width"] == b["width"]:
                pairs.append((a, b))
    if not pairs:
        return None
    winner, loser = pairs[int(rng.integers(0, len(pairs)))]
    return dict(winner), dict(loser)


def _fixed_masks(h: int, w: int, anchors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.zeros((h, w), dtype=bool)
    for anchor in anchors:
        r0, c0 = anchor["r0"], anchor["c0"]
        fixed[r0 : r0 + 2, c0 : c0 + 2] = True
        fixed[r0 - 1, list(anchor["markers"])] = True
    return fixed, _dilate8(fixed)


def _candidate_positions(
    h: int,
    w: int,
    shape: dict,
    target_idx: int,
    anchors: list[dict],
    fixed_dilated: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    width = shape["width"]
    cells = _bar_shape_cells(width, shape["protrusions"])
    out = []
    for top in range(h - 1):
        for left in range(w - width + 1):
            coords = [(top + dr, left + dc) for dr, dc in cells]
            if any(fixed_dilated[r, c] for r, c in coords):
                continue
            bar = {
                "body_row": top + 1,
                "left": left,
                "right": left + width - 1,
                "width": width,
                "protrusions": shape["protrusions"],
            }
            if _target_anchor_index(bar, anchors) != target_idx:
                continue
            bbox = (top, left, top + 1, left + width - 1)
            out.append({"top": top, "left": left, "coords": coords, "bbox": bbox})
    rng.shuffle(out)
    return out[:120]


def _place_bars(
    h: int,
    w: int,
    specs: list[dict],
    anchors: list[dict],
    fixed_dilated: np.ndarray,
    rng: np.random.Generator,
) -> list[dict] | None:
    candidates = []
    for spec in specs:
        cand = _candidate_positions(h, w, spec, spec["target"], anchors, fixed_dilated, rng)
        if not cand:
            return None
        candidates.append(cand)

    order = sorted(range(len(specs)), key=lambda i: (len(candidates[i]), -specs[i]["width"]))
    placed: dict[int, dict] = {}
    occupied = np.zeros((h, w), dtype=bool)

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for cand in candidates[idx]:
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cand["coords"]:
                mask[r, c] = True
            if (_dilate8(mask) & occupied).any():
                continue
            placed[idx] = cand
            occupied[:] |= mask
            if rec(pos + 1):
                return True
            occupied[:] ^= mask
            placed.pop(idx)
        return False

    if not rec(0):
        return None

    bars = []
    for idx, spec in enumerate(specs):
        pos = placed[idx]
        bars.append(
            {
                "color": spec["color"],
                "top": pos["top"],
                "left": pos["left"],
                "width": spec["width"],
                "protrusions": spec["protrusions"],
                "target": spec["target"],
                "mode": spec["mode"],
                "role": spec["role"],
            }
        )
    return bars


def _render_input(h: int, w: int, anchors: list[dict], bars: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for anchor in anchors:
        r0, c0 = anchor["r0"], anchor["c0"]
        grid[r0 : r0 + 2, c0 : c0 + 2] = GREY
        grid[r0 - 1, list(anchor["markers"])] = RED
    for bar in bars:
        color = int(bar["color"])
        grid[bar["top"] + 1, bar["left"] : bar["left"] + bar["width"]] = color
        for dc in bar["protrusions"]:
            grid[bar["top"], bar["left"] + dc] = color
    return grid


def _build_specs(rng: np.random.Generator, anchors: list[dict]) -> tuple[list[dict], dict[int, str]] | None:
    n_anchors = len(anchors)
    n_bars = (3 if n_anchors == 2 else 4) + int(rng.integers(0, 2))
    conflict_idx = int(rng.integers(0, n_anchors))
    counts = [1] * n_anchors
    counts[conflict_idx] += 1
    if n_bars == n_anchors + 2:
        extra_choices = [i for i in range(n_anchors) if i != conflict_idx]
        counts[int(rng.choice(extra_choices))] += 1

    pair_modes: dict[int, str] = {}
    specs = []
    colors = [int(v) for v in rng.choice(np.array(BAR_COLORS, dtype=int), size=n_bars, replace=False)]
    color_idx = 0

    for anchor_idx, count in enumerate(counts):
        cands = _shape_candidates(anchors[anchor_idx])
        if count == 1:
            shape = dict(cands[int(rng.integers(0, len(cands)))])
            specs.append(
                {
                    "color": colors[color_idx],
                    "target": anchor_idx,
                    "width": shape["width"],
                    "protrusions": shape["protrusions"],
                    "mode": "single",
                    "role": "single",
                }
            )
            color_idx += 1
            continue

        mode = ("score", "length", "row")[int(rng.integers(0, 3))] if anchor_idx == conflict_idx else "score"
        pair = _pick_pair(cands, mode, rng)
        if pair is None:
            return None
        pair_modes[anchor_idx] = mode
        winner_shape, loser_shape = pair
        specs.append(
            {
                "color": colors[color_idx],
                "target": anchor_idx,
                "width": winner_shape["width"],
                "protrusions": winner_shape["protrusions"],
                "mode": mode,
                "role": "winner",
            }
        )
        specs.append(
            {
                "color": colors[color_idx + 1],
                "target": anchor_idx,
                "width": loser_shape["width"],
                "protrusions": loser_shape["protrusions"],
                "mode": mode,
                "role": "loser",
            }
        )
        color_idx += 2

    return specs, pair_modes


def _validate_generation(grid: np.ndarray, pair_modes: dict[int, str]) -> bool:
    anchors, bars, by_anchor = _analyze(grid)
    if not (2 <= len(anchors) <= 3 and 3 <= len(bars) <= 5):
        return False
    if not any(len(group) >= 2 for group in by_anchor.values()):
        return False
    for bar in bars:
        if not (3 <= bar["width"] <= 5 and 1 <= len(bar["protrusions"]) <= 3):
            return False
    for idx, mode in pair_modes.items():
        group = by_anchor[idx]
        if len(group) != 2:
            return False
        a, b = group
        if mode == "score" and not (a["score"] > b["score"]):
            return False
        if mode == "length" and not (a["score"] == b["score"] and a["width"] < b["width"]):
            return False
        if mode == "row" and not (a["score"] == b["score"] and a["width"] == b["width"] and a["body_row"] < b["body_row"]):
            return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        h = int(rng.integers(14, 19))
        w = int(rng.integers(17, 23))
        n_anchors = int(rng.integers(2, 4))
        anchors = _place_anchors(rng, h, w, n_anchors)
        if anchors is None:
            continue
        built = _build_specs(rng, anchors)
        if built is None:
            continue
        specs, pair_modes = built
        _fixed, fixed_dilated = _fixed_masks(h, w, anchors)
        bars = _place_bars(h, w, specs, anchors, fixed_dilated, rng)
        if bars is None:
            continue
        grid = _render_input(h, w, anchors, bars)
        if _validate_generation(grid, pair_modes):
            return grid
    raise RuntimeError("generate_input failed to build a valid puzzle instance")
