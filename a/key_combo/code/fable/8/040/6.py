from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BG = int(Color.BLACK)
PALETTE = tuple(int(c) for c in Color.NOT_BLACK)
GEN_ATTEMPTS = 400


def _majority_color(values: np.ndarray) -> tuple[int, int, bool]:
    counts = Counter(int(v) for v in values.ravel())
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    winners.sort()
    return int(winners[0]), int(best_count), len(winners) > 1


def _repair_from_period(rect: np.ndarray, pr: int, pc: int) -> dict:
    h, w = rect.shape
    tile = np.zeros((pr, pc), dtype=int)
    defects = 0
    ambiguous = False
    for rr in range(pr):
        for cc in range(pc):
            color, keep, tie = _majority_color(rect[rr::pr, cc::pc])
            tile[rr, cc] = int(color)
            defects += int(rect[rr::pr, cc::pc].size - keep)
            ambiguous |= bool(tie)

    repaired = tile[np.arange(h)[:, None] % pr, np.arange(w)[None, :] % pc]
    return {
        "period": (int(pr), int(pc)),
        "tile": tile,
        "repaired": repaired,
        "defects": int(defects),
        "ambiguous": bool(ambiguous),
    }


def _infer_periodic_rectangle(rect: np.ndarray) -> dict:
    h, w = rect.shape
    candidates = []
    for pr in range(1, h + 1):
        if h % pr != 0:
            continue
        for pc in range(1, w + 1):
            if w % pc != 0:
                continue
            if pr == h and pc == w:
                continue
            cand = _repair_from_period(rect, pr, pc)
            key = (
                int(cand["defects"]),
                int(cand["ambiguous"]),
                int(pr * pc),
                int(pr + pc),
                int(pr),
                int(pc),
            )
            candidates.append((key, cand))

    if not candidates:
        return _repair_from_period(rect, h, w)

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground rectangle"

    rectangles = []
    for comp in comps:
        crop = crop_bbox(g, comp["bbox"])
        assert int(comp["area"]) == int(crop.size), "foreground objects must be solid rectangles"
        analysis = _infer_periodic_rectangle(crop)
        rectangles.append(
            {
                "bbox": tuple(int(x) for x in comp["bbox"]),
                "crop": crop,
                "repaired": analysis["repaired"],
                "period": analysis["period"],
                "tile": analysis["tile"],
                "defects": int(analysis["defects"]),
            }
        )

    min_defects = min(int(item["defects"]) for item in rectangles)
    winners = [item for item in rectangles if int(item["defects"]) == min_defects]
    assert len(winners) == 1, "expected a unique rectangle with the fewest defects"
    return winners[0]["repaired"], {"rectangles": rectangles, "winner_bbox": winners[0]["bbox"]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the periodic rectangle with the fewest defects, then repair it to its perfect repeating pattern.

    Input structure:
    - The grid has a uniform background with several separated solid rectangles.
    - Each rectangle is fully filled and follows its own 2D periodic color pattern.
    - A few pixels inside each rectangle violate that rectangle's otherwise regular pattern.

    Rule:
    - Detect every solid foreground rectangle and crop it to its bbox.
    - For each crop, infer the repeating row/column period that yields the smallest number of contradictory pixels.
    - Count those contradictions as defects, select the unique rectangle with the fewest defects, and restore all of its cells to the inferred perfect pattern.

    Key insight:
    - Defects are defined relative to an implied periodic tiling, so they cannot be found reliably from raw color frequency alone; the hidden repeat must be reconstructed first.
    """
    out, _details = _solve_details(grid)
    return out


def _periodic_fill(tile: np.ndarray, reps_r: int, reps_c: int) -> np.ndarray:
    return np.tile(tile, (int(reps_r), int(reps_c))).astype(int)


def _is_exact_repeat(arr: np.ndarray, pr: int, pc: int) -> bool:
    h, w = arr.shape
    rebuilt = arr[np.arange(h)[:, None] % pr, np.arange(w)[None, :] % pc]
    return bool(np.array_equal(arr, rebuilt))


def _minimal_period(arr: np.ndarray) -> tuple[int, int]:
    h, w = arr.shape
    best = (h, w)
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if _is_exact_repeat(arr, pr, pc):
                if (pr * pc, pr + pc, pr, pc) < (best[0] * best[1], best[0] + best[1], best[0], best[1]):
                    best = (int(pr), int(pc))
    return best


def _sample_tile(pr: int, pc: int, palette: tuple[int, ...]) -> np.ndarray:
    for _ in range(300):
        tile = np.random.choice(np.array(palette, dtype=int), size=(pr, pc), replace=True)
        if len(set(int(v) for v in tile.ravel())) < 2:
            continue
        if _minimal_period(tile) != (pr, pc):
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a non-degenerate tile")


def _inject_defects(perfect: np.ndarray, tile: np.ndarray, defect_count: int) -> np.ndarray:
    pr, pc = tile.shape
    noisy = perfect.copy()
    palette = tuple(sorted(set(int(v) for v in tile.ravel())))
    assert len(palette) >= 2

    by_residue: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(perfect.shape[0]):
        for c in range(perfect.shape[1]):
            by_residue.setdefault((r % pr, c % pc), []).append((int(r), int(c)))
    assert defect_count <= len(by_residue)

    for rr, cc in random.sample(list(by_residue.keys()), defect_count):
        r, c = random.choice(by_residue[(rr, cc)])
        expected = int(tile[rr, cc])
        choices = [color for color in palette if int(color) != expected]
        noisy[r, c] = int(random.choice(choices))
    return noisy


def _dominant_fill_mismatch(rect: np.ndarray) -> int:
    counts = Counter(int(v) for v in rect.ravel())
    return int(rect.size - max(counts.values()))


def _random_rectangle_spec(defect_count: int) -> dict:
    for _ in range(200):
        pr = random.randint(2, 4)
        pc = random.randint(2, 4)
        if pr * pc >= int(defect_count):
            break
    else:
        raise RuntimeError("failed to sample a large-enough period")
    reps_r = random.randint(2, 3)
    reps_c = random.randint(2, 3)
    palette_size = random.randint(2, min(4, pr * pc))
    palette = tuple(random.sample(PALETTE, palette_size))
    tile = _sample_tile(pr, pc, palette)
    perfect = _periodic_fill(tile, reps_r, reps_c)
    noisy = _inject_defects(perfect, tile, defect_count)
    return {
        "period": (int(pr), int(pc)),
        "tile": tile,
        "perfect": perfect,
        "noisy": noisy,
        "defect_count": int(defect_count),
    }


def _boxes_are_separated(b0: tuple[int, int, int, int], b1: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = b0
    rr0, cc0, rr1, cc1 = b1
    return (
        r1 + pad < rr0
        or rr1 + pad < r0
        or c1 + pad < cc0
        or cc1 + pad < c0
    )


def _place_specs(specs: list[dict]) -> np.ndarray | None:
    H = 30
    W = 30
    grid = np.full((H, W), BG, dtype=int)

    order = list(range(len(specs)))
    random.shuffle(order)
    placed_boxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        patch = specs[idx]["noisy"]
        h, w = patch.shape
        placed = False
        for _ in range(250):
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            box = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            if all(_boxes_are_separated(box, other, pad=1) for other in placed_boxes):
                paste(grid, patch, r0, c0, transparent=BG, clip=False)
                specs[idx]["bbox"] = box
                placed_boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return grid


def _validate_instance(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        expected_winner = min(range(len(specs)), key=lambda i: int(specs[i]["defect_count"]))
        if sum(int(spec["defect_count"]) == int(specs[expected_winner]["defect_count"]) for spec in specs) != 1:
            return False

        dominant_scores = [_dominant_fill_mismatch(spec["noisy"]) for spec in specs]
        if min(dominant_scores) >= dominant_scores[expected_winner]:
            return False

        out, details = _solve_details(grid)
        if not np.array_equal(out, specs[expected_winner]["perfect"]):
            return False

        by_bbox = {tuple(int(x) for x in spec["bbox"]): spec for spec in specs}
        for rect in details["rectangles"]:
            bbox = tuple(int(x) for x in rect["bbox"])
            if bbox not in by_bbox:
                return False
            spec = by_bbox[bbox]
            if int(rect["defects"]) != int(spec["defect_count"]):
                return False
            if not np.array_equal(rect["repaired"], spec["perfect"]):
                return False
            if _minimal_period(spec["tile"]) != tuple(spec["period"]):
                return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(3, 4)
    base = random.randint(1, 2)
    counts = random.sample(list(range(base, base + 5)), rect_count)
    winner_count = min(counts)
    specs = [_random_rectangle_spec(defect_count) for defect_count in counts]
    if sum(int(spec["defect_count"]) == winner_count for spec in specs) != 1:
        return None

    grid = _place_specs(specs)
    if grid is None:
        return None

    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return None
    if not _validate_instance(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-rectangle defect puzzle")
