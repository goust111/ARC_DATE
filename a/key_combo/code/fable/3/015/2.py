from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


# Ensure arc_utils can be imported when this file is loaded through SourceFileLoader.
_UTILS_PATH = Path(__file__).resolve().parents[6] / "new" / "solvers" / "arc_utils.py"
if not _UTILS_PATH.exists():
    for p in Path(__file__).resolve().parents:
        candidate = p / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            _UTILS_PATH = candidate
            break
_UTILS_DIR = _UTILS_PATH.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg


def _extract_blocks(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    """
    Parse solid yellow rectangles, each containing one unique non-yellow control color.
    """
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert components, "expected at least one non-background component"

    blocks: list[dict[str, Any]] = []
    used_counts: set[int] = set()

    for comp in components:
        coords = comp["coords"]
        r0, c0, r1, c1 = bbox_coords(coords)
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if region.size != int(coords.shape[0]):
            # Not a solid rectangle.
            raise AssertionError("each block must be a solid rectangle")

        colors = set(map(int, np.unique(region)))
        if Color.YELLOW not in colors:
            raise AssertionError("expected yellow inside each block")
        control_colors = colors - {Color.YELLOW}
        if len(control_colors) != 1:
            raise AssertionError("each block must contain exactly one control color")

        control_color = next(iter(control_colors))
        control_count = int(np.sum(region == control_color))
        if control_count <= 0:
            raise AssertionError("control color count must be positive")
        if control_count in used_counts:
            raise AssertionError("control counts must be unique")
        used_counts.add(control_count)

        if int(np.sum(region == bg)) != 0:
            raise AssertionError("block region must be fully non-background")

        h, w = region.shape
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True

        blocks.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "h": int(h),
                "w": int(w),
                "count": control_count,
                "color": int(control_color),
                "init_area": int(region.size),
                "full_area": int((h + 2 * control_count) * (w + 2 * control_count)),
            }
        )

    return blocks


def _one_step_frontier(mask: np.ndarray) -> np.ndarray:
    """
    Return the 4-neighborhood ring around `mask` that is not already in mask.
    """
    padded = np.pad(mask, 1, constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return (up | down | left | right) & (~mask)


def _run_growth(
    grid: np.ndarray, blocks: list[dict[str, Any]], *, track_conflict: bool = False
) -> tuple[np.ndarray, dict[str, bool]]:
    out = np.array(grid, copy=True)
    bg = int(get_bg(out))
    H, W = out.shape

    block_masks = [np.array(block["mask"], copy=True) for block in blocks]
    max_round = max(block["count"] for block in blocks)
    same_round_conflict = False

    for t in range(max_round):
        claims: dict[tuple[int, int], tuple[int, int]] = {}
        claim_counts: dict[tuple[int, int], int] = {}

        for i, block in enumerate(blocks):
            if t >= int(block["count"]):
                continue

            frontier = _one_step_frontier(block_masks[i])
            frontier = frontier & (out == bg)
            if not bool(frontier.any()):
                continue

            rows, cols = np.where(frontier)
            for r, c in zip(rows.tolist(), cols.tolist()):
                key = (int(r), int(c))
                cnt = claim_counts.get(key, 0) + 1
                claim_counts[key] = cnt
                if cnt >= 2:
                    same_round_conflict = True

                prev = claims.get(key)
                if prev is None or int(block["count"]) > blocks[prev[0]]["count"]:
                    claims[key] = (i, int(block["count"]))

        if not claims:
            continue

        for (r, c), (winner, _winner_count) in claims.items():
            out[r, c] = int(blocks[winner]["color"])
            block_masks[winner][r, c] = True

    if any(int(mask.sum()) > 0 for mask in block_masks):
        for block, mask in zip(blocks, block_masks):
            possible = int(block["full_area"])
            occupied = int(mask.sum())
            if occupied < possible:
                # Block cannot reach its full bbox; it has been blocked by another block.
                break
        else:
            blocked = False
    else:
        blocked = False

    stats = {"same_round_conflict": same_round_conflict, "blocked": False}
    if len(blocks) > 1:
        # Recompute blockage from the explicit area comparison above.
        for block, mask in zip(blocks, block_masks):
            if int(mask.sum()) < int(block["full_area"]):
                stats["blocked"] = True
                break

    return out, stats if track_conflict else (out, {"same_round_conflict": False, "blocked": False})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each solid yellow rectangle outward for a number of rounds equal to its control count.

    Input structure:
    - The background is the dominant color.
    - Multiple disjoint solid yellow rectangles are present.
    - Each rectangle contains only yellow plus one other control color, and all control counts are different.

    Rule:
    - Count control-color pixels in each block; that count is both the number of expansion rounds and priority.
    - All blocks expand simultaneously one-cell wide each round, only into background cells.
    - In the same round, if several blocks target the same background cell, the block with larger control count paints it.
    - Original yellow cells and control pixels are preserved.

    Key insight:
    - The control count serves as both growth time and tie-break priority, so synchronous waves can block a farther, higher-count block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    blocks = _extract_blocks(g, bg)
    out, _ = _run_growth(g, blocks, track_conflict=False)
    return out


_BG_OPTIONS = [0, 1, 2, 3, 5, 6, 7, 8, 9]
_MIN_GRID = 16
_MAX_GRID = 30
_MIN_BLOCKS = 2
_MAX_BLOCKS = 4
_MAX_ATTEMPTS = 500


def _random_block_specs(block_count: int, bg: int) -> list[dict[str, int]] | None:
    # Distinct control counts from 1..8.
    count_pool = random.sample(range(1, 9), block_count)
    colors = [c for c in range(10) if c != bg and c != Color.YELLOW]
    if len(colors) < block_count:
        return None
    random.shuffle(count_pool)
    random.shuffle(colors)
    specs: list[dict[str, int]] = []

    for count, color in zip(count_pool, colors):
        for _ in range(40):
            h = int(random.randint(3, 8))
            w = int(random.randint(3, 8))
            if (h - 2) * (w - 2) >= count:
                specs.append({"h": h, "w": w, "count": count, "color": int(color)})
                break
        else:
            return None
    return specs


def _can_place(
    occupancy: np.ndarray,
    r: int,
    c: int,
    h: int,
    w: int,
    *,
    allow_margin: int = 1,
) -> bool:
    H, W = occupancy.shape
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    r0 = max(0, r - allow_margin)
    c0 = max(0, c - allow_margin)
    r1 = min(H, r + h + allow_margin)
    c1 = min(W, c + w + allow_margin)
    return not bool(occupancy[r0:r1, c0:c1].any())


def _can_place_expanded(
    occupancy: np.ndarray,
    r: int,
    c: int,
    h: int,
    w: int,
    k: int,
    *,
    allow_margin: int = 0,
) -> bool:
    H, W = occupancy.shape
    r0 = r - k
    c0 = c - k
    r1 = r + h - 1 + k + 1
    c1 = c + w - 1 + k + 1
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        return False
    r0m = max(0, r0 - allow_margin)
    c0m = max(0, c0 - allow_margin)
    r1m = min(H, r1 + allow_margin)
    c1m = min(W, c1 + allow_margin)
    return not bool(occupancy[r0m:r1m, c0m:c1m].any())


def _place_blocks(
    specs: list[dict[str, int]],
    H: int,
    W: int,
    mode: str,
) -> list[tuple[int, int]] | None:
    occupancy = np.zeros((H, W), dtype=bool)
    expanded_occupancy = np.zeros((H, W), dtype=bool)
    order = list(range(len(specs)))
    random.shuffle(order)
    placements: list[tuple[int, int] | None] = [None] * len(specs)

    for idx in order:
        spec = specs[idx]
        h = int(spec["h"])
        w = int(spec["w"])
        k = int(spec["count"])
        candidates: list[tuple[int, int]] = []

        for r in range(H - h + 1):
            for c in range(W - w + 1):
                if not _can_place(occupancy, r, c, h, w, allow_margin=1):
                    continue
                if mode == "clean":
                    if not _can_place_expanded(
                        expanded_occupancy, r, c, h, w, k, allow_margin=0
                    ):
                        continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements[idx] = (r, c)
        occupancy[r : r + h, c : c + w] = True
        if mode == "clean":
            expanded_occupancy[r - k : r + h + k, c - k : c + w + k] = True

    return [p for p in placements]


def _draw_block(
    grid: np.ndarray,
    top: int,
    left: int,
    spec: dict[str, int],
) -> None:
    h = spec["h"]
    w = spec["w"]
    k = spec["count"]
    control = spec["color"]

    block = np.full((h, w), Color.YELLOW, dtype=int)
    interior = [(rr, cc) for rr in range(1, h - 1) for cc in range(1, w - 1)]
    chosen = random.sample(interior, k)
    for rr, cc in chosen:
        block[rr, cc] = control
    grid[top : top + h, left : left + w] = block


def _run_stats(inp: np.ndarray, blocks: list[dict[str, Any]]) -> dict[str, bool]:
    out, stats = _run_growth(inp, blocks, track_conflict=True)
    stats["changed"] = bool(np.any(inp != out))
    return stats


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        H = int(random.randint(_MIN_GRID, _MAX_GRID))
        W = int(random.randint(_MIN_GRID, _MAX_GRID))
        bg = int(random.choice(_BG_OPTIONS))
        mode_roll = random.random()
        if mode_roll < 0.2:
            mode = "single"
            block_count = 1
        elif mode_roll < 0.5:
            mode = "clean"
            block_count = int(random.randint(_MIN_BLOCKS, _MAX_BLOCKS))
        else:
            mode = "conflict"
            block_count = int(random.randint(_MIN_BLOCKS, _MAX_BLOCKS))

        specs = _random_block_specs(block_count, bg)
        if specs is None:
            continue

        placements = _place_blocks(specs, H, W, mode="clean" if mode == "clean" else "conflict")
        if placements is None:
            continue

        inp = np.full((H, W), bg, dtype=int)
        for spec, pos in zip(specs, placements):
            _draw_block(inp, pos[0], pos[1], spec)

        try:
            blocks = _extract_blocks(inp, bg)
            out, stats = _run_growth(inp, blocks, track_conflict=True)
        except Exception:
            continue

        if mode == "single":
            if np.array_equal(main(inp), out):
                return inp
            continue

        if mode == "clean":
            if not stats["same_round_conflict"] and not stats["blocked"] and np.array_equal(main(inp), out):
                return inp
            continue

        if mode == "conflict":
            if (stats["same_round_conflict"] or stats["blocked"]) and np.array_equal(main(inp), out):
                return inp
            continue

    raise RuntimeError("failed to generate valid 2.py input")
