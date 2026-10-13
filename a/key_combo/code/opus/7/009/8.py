from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, label_mask, paste, random_free_location


BG = Color.BLACK
IN_COLOR = Color.GREY
MAJ_COLOR = Color.TEAL
MIN_COLOR = Color.RED
GRID_MIN = 16
GRID_MAX = 24
SPRITE_SIZE = 3
TOTAL_COUNT_RANGE = (8, 15)
GEN_RETRIES = 400
ORIENTATIONS = ("nw_se", "ne_sw")


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, _r1, _c1 = component["bbox"]
    h = bbox_h(component["bbox"])
    w = bbox_w(component["bbox"])
    assert (h, w) == (SPRITE_SIZE, SPRITE_SIZE)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _diag_counts(mask: np.ndarray) -> tuple[int, int]:
    nw_se = 0
    ne_sw = 0
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            if mask[r, c] and mask[r + 1, c + 1]:
                nw_se += 1
    for r in range(mask.shape[0] - 1):
        for c in range(1, mask.shape[1]):
            if mask[r, c] and mask[r + 1, c - 1]:
                ne_sw += 1
    return nw_se, ne_sw


def _orientation(mask: np.ndarray) -> str:
    nw_se, ne_sw = _diag_counts(mask)
    return "nw_se" if nw_se > ne_sw else "ne_sw"


def _family_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    key = _mask_key(mask)
    mirror_key = _mask_key(np.fliplr(mask))
    return min(key, mirror_key)


def _render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _extract_instances(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    instances = []
    for component in components:
        mask = _component_mask(component)
        instances.append(
            {
                "bbox": component["bbox"],
                "coords": component["coords"],
                "mask": mask,
                "family": _family_key(mask),
                "orientation": _orientation(mask),
            }
        )
    return instances


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    instances = _extract_instances(g)
    assert instances

    families: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for inst in instances:
        families[inst["family"]].append(inst)

    winner_count = max(len(group) for group in families.values())
    winners = [family for family, group in families.items() if len(group) == winner_count]
    assert len(winners) == 1
    winner_family = winners[0]
    winner_instances = families[winner_family]
    assert len(winner_instances) >= 4

    orient_counts = {name: 0 for name in ORIENTATIONS}
    for inst in winner_instances:
        orient_counts[inst["orientation"]] += 1
    assert orient_counts["nw_se"] != orient_counts["ne_sw"]
    majority_orientation = "nw_se" if orient_counts["nw_se"] > orient_counts["ne_sw"] else "ne_sw"
    minority_orientation = "ne_sw" if majority_orientation == "nw_se" else "nw_se"

    out = np.full_like(g, bg)
    for inst in winner_instances:
        color = MAJ_COLOR if inst["orientation"] == majority_orientation else MIN_COLOR
        coords = inst["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    details = {
        "instances": instances,
        "winner_family": winner_family,
        "winner_instances": winner_instances,
        "orient_counts": orient_counts,
        "majority_orientation": majority_orientation,
        "minority_orientation": minority_orientation,
    }
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only the most frequent mirrored 3x3 sprite family, then recolor its two diagonal orientations by majority.

    Input structure:
    - The grid has a dominant black background and several separated monochrome 3x3 sprites.
    - Every sprite is an 8-connected component whose bounding box is exactly 3x3.
    - Mirror-image variants of the same occupancy pattern may both appear, while other 3x3 families act as distractors.

    Rule:
    - Extract each foreground component and convert it to its 3x3 boolean occupancy mask.
    - Group sprites by mask up to left-right mirroring, and select the uniquely most frequent family.
    - For each instance in that winning family, count adjacent occupied pairs on the NW-SE and NE-SW diagonals; if NW-SE is larger, classify it as NW-SE, otherwise as NE-SW.
    - Recolor the more common winning orientation teal and the less common winning orientation red, erase every non-winning family to background, and keep each surviving sprite in place.

    Key insight:
    - The task has two coupled filters at different granularities: global frequency decides which mirrored sprite family survives, and only then do local diagonal-adjacency counts decide the surviving instances' colors.
    """
    out, _details = _solve_details(np.asarray(grid, dtype=int))
    return out


def _enumerate_family_catalog() -> list[dict]:
    families: dict[tuple[tuple[int, ...], ...], dict[str, np.ndarray]] = {}
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        for k in range(SPRITE_SIZE * SPRITE_SIZE):
            if bits & (1 << k):
                mask[k // SPRITE_SIZE, k % SPRITE_SIZE] = True
        if mask.sum() < 4 or mask.sum() > 6:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        _labels, ncomp = label_mask(mask, connectivity=8)
        if ncomp != 1:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if _orientation(mask) == _orientation(np.fliplr(mask)):
            continue
        family = _family_key(mask)
        families.setdefault(family, {})[_orientation(mask)] = mask.copy()

    catalog = []
    for family, variants in families.items():
        if set(variants) == set(ORIENTATIONS):
            catalog.append({"family": family, "variants": variants})
    return catalog


FAMILY_CATALOG = _enumerate_family_catalog()


def _sample_family_counts() -> tuple[int, list[int]]:
    for _ in range(200):
        n_families = random.choice((2, 3))
        winner_count = random.randint(4, 8)
        other_counts = [random.randint(1, winner_count - 1) for _ in range(n_families - 1)]
        total = winner_count + sum(other_counts)
        if TOTAL_COUNT_RANGE[0] <= total <= TOTAL_COUNT_RANGE[1]:
            return winner_count, other_counts
    raise RuntimeError("failed to sample family counts")


def _sample_orientation_counts(count: int, force_both: bool) -> tuple[str, dict[str, int]]:
    if force_both:
        minority = random.randint(1, (count - 1) // 2)
        majority = count - minority
        majority_orientation = random.choice(ORIENTATIONS)
        counts = {
            majority_orientation: majority,
            ORIENTATIONS[1] if majority_orientation == ORIENTATIONS[0] else ORIENTATIONS[0]: minority,
        }
        return majority_orientation, counts

    split = random.randint(0, count)
    majority_orientation = random.choice(ORIENTATIONS)
    counts = {majority_orientation: split}
    other = ORIENTATIONS[1] if majority_orientation == ORIENTATIONS[0] else ORIENTATIONS[0]
    counts[other] = count - split
    return majority_orientation, counts


def _instance_specs_for_families(families: list[dict], winner_count: int, other_counts: list[int]) -> tuple[list[dict], dict]:
    winner_family = families[0]
    majority_orientation, winner_orient_counts = _sample_orientation_counts(winner_count, force_both=True)

    specs = []
    for orientation, count in winner_orient_counts.items():
        for _ in range(count):
            specs.append(
                {
                    "family": winner_family["family"],
                    "mask": winner_family["variants"][orientation],
                    "orientation": orientation,
                }
            )

    family_counts = {winner_family["family"]: winner_count}
    for family, count in zip(families[1:], other_counts):
        _unused_majority, orient_counts = _sample_orientation_counts(count, force_both=False)
        if orient_counts["nw_se"] == 0:
            orient_counts["ne_sw"] = count
        if orient_counts["ne_sw"] == 0:
            orient_counts["nw_se"] = count
        for orientation, orient_count in orient_counts.items():
            for _ in range(orient_count):
                specs.append(
                    {
                        "family": family["family"],
                        "mask": family["variants"][orientation],
                        "orientation": orientation,
                    }
                )
        family_counts[family["family"]] = count

    random.shuffle(specs)
    meta = {
        "winner_family": winner_family["family"],
        "winner_majority_orientation": majority_orientation,
        "winner_orientation_counts": winner_orient_counts,
        "family_counts": family_counts,
    }
    return specs, meta


def _build_expected(shape: tuple[int, int], placements: list[dict], winner_family, majority_orientation: str) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for placed in placements:
        if placed["family"] != winner_family:
            continue
        color = MAJ_COLOR if placed["orientation"] == majority_orientation else MIN_COLOR
        paste(out, _render_sprite(placed["mask"], color), placed["r"], placed["c"], transparent=BG)
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    actual, details = _solve_details(grid)
    if not np.array_equal(actual, expected):
        return False
    if details["winner_family"] != meta["winner_family"]:
        return False
    if details["majority_orientation"] != meta["winner_majority_orientation"]:
        return False

    counts = details["orient_counts"]
    if min(counts.values()) <= 0:
        return False
    if len({tuple(inst["family"]) for inst in details["instances"]}) < 2:
        return False

    erase_only = np.full_like(grid, BG)
    for inst in details["winner_instances"]:
        erase_only[inst["coords"][:, 0], inst["coords"][:, 1]] = int(MAJ_COLOR)
    if np.array_equal(erase_only, expected):
        return False

    recolor_all = np.full_like(grid, BG)
    for inst in details["instances"]:
        color = MAJ_COLOR if inst["orientation"] == details["majority_orientation"] else MIN_COLOR
        recolor_all[inst["coords"][:, 0], inst["coords"][:, 1]] = int(color)
    if np.array_equal(recolor_all, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    winner_count, other_counts = _sample_family_counts()
    n_families = 1 + len(other_counts)
    families = random.sample(FAMILY_CATALOG, n_families)
    specs, meta = _instance_specs_for_families(families, winner_count, other_counts)

    total = len(specs)
    side_min = max(GRID_MIN, int(np.ceil(np.sqrt(total * 18))))
    height = random.randint(side_min, GRID_MAX)
    width = random.randint(side_min, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)

    placements = []
    for spec in specs:
        patch = _render_sprite(spec["mask"], IN_COLOR)
        try:
            r, c = random_free_location(grid, patch, transparent=BG, padding=1, padding_connectivity=8)
        except ValueError:
            return None
        paste(grid, patch, r, c, transparent=BG)
        placements.append({**spec, "r": r, "c": c})

    expected = _build_expected(grid.shape, placements, meta["winner_family"], meta["winner_majority_orientation"])
    if not _semantic_checks(grid, expected, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
