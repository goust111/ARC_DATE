from __future__ import annotations

import random
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("arc_utils_for_opus_4_055_4", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox


BG = Color.BLACK
RED = Color.RED
YELLOW = Color.YELLOW
GREY = Color.GREY

QUADRANT_NAMES = ("tl", "tr", "bl", "br")

TEMPLATE_SPECS = (
    {"mask": np.array([[1, 0], [1, 1]], dtype=bool), "offset": (-1, 0)},
    {"mask": np.array([[0, 1], [1, 1]], dtype=bool), "offset": (-1, -1)},
    {"mask": np.array([[1, 0, 0], [1, 1, 1]], dtype=bool), "offset": (-1, 0)},
    {"mask": np.array([[0, 0, 1], [1, 1, 1]], dtype=bool), "offset": (-1, -2)},
    {"mask": np.array([[1, 1, 1], [1, 0, 0]], dtype=bool), "offset": (0, 0)},
    {"mask": np.array([[1, 1, 1], [0, 0, 1]], dtype=bool), "offset": (0, -2)},
)

FRAGMENTS = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)

MAX_GENERATION_ATTEMPTS = 500


def _split_quadrants(grid: np.ndarray) -> dict[str, np.ndarray]:
    h, w = grid.shape
    assert h % 2 == 1 and w % 2 == 1
    mid_r = h // 2
    mid_c = w // 2
    return {
        "tl": grid[:mid_r, :mid_c].copy(),
        "tr": grid[:mid_r, mid_c + 1 :].copy(),
        "bl": grid[mid_r + 1 :, :mid_c].copy(),
        "br": grid[mid_r + 1 :, mid_c + 1 :].copy(),
    }


def _majority_and_outlier(quadrants: dict[str, np.ndarray]) -> tuple[str, str]:
    enc = {name: quad.tobytes() for name, quad in quadrants.items()}
    counts = Counter(enc.values())
    majority_keys = [key for key, count in counts.items() if count == 3]
    outlier_keys = [key for key, count in counts.items() if count == 1]
    assert len(majority_keys) == 1 and len(outlier_keys) == 1

    majority_name = next(name for name in QUADRANT_NAMES if enc[name] == majority_keys[0])
    outlier_name = next(name for name in QUADRANT_NAMES if enc[name] == outlier_keys[0])
    return majority_name, outlier_name


def _extract_template_mask(grid: np.ndarray) -> np.ndarray:
    yellow_mask = grid == YELLOW
    box = bbox(yellow_mask)
    assert box is not None
    out = crop_bbox(yellow_mask, box).astype(bool)
    assert out.shape in {(2, 2), (2, 3), (3, 2)}
    assert out[0].any() and out[-1].any() and out[:, 0].any() and out[:, -1].any()
    return out


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for flipped in (False, True):
        base = np.fliplr(mask) if flipped else mask
        for k in range(4):
            transformed = np.rot90(base, k).astype(bool)
            key = (tuple(transformed.shape), transformed.tobytes())
            variants.setdefault(key, transformed.copy())
    return list(variants.values())


def _find_matches(quadrant: np.ndarray, template_mask: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    red_mask = quadrant == RED
    qh, qw = quadrant.shape
    matches: list[tuple[np.ndarray, int, int]] = []
    for variant in _dihedral_variants(template_mask):
        h, w = variant.shape
        for r0 in range(qh - h + 1):
            for c0 in range(qw - w + 1):
                window = red_mask[r0 : r0 + h, c0 : c0 + w]
                if bool(window[variant].all()):
                    matches.append((variant.copy(), int(r0), int(c0)))
    return matches


def _paint_match(quadrant: np.ndarray, match: tuple[np.ndarray, int, int]) -> np.ndarray:
    variant, r0, c0 = match
    out = quadrant.copy()
    h, w = variant.shape
    patch = out[r0 : r0 + h, c0 : c0 + w]
    patch[variant] = YELLOW
    return out


def _analyze_instance(grid: np.ndarray) -> tuple[str, np.ndarray]:
    quadrants = _split_quadrants(grid)
    majority_name, outlier_name = _majority_and_outlier(quadrants)
    template_mask = _extract_template_mask(grid)

    for name in QUADRANT_NAMES:
        matches = _find_matches(quadrants[name], template_mask)
        if name == outlier_name:
            assert len(matches) == 1
            out = _paint_match(quadrants[name], matches[0])
        else:
            assert len(matches) == 0
    assert majority_name != outlier_name
    return outlier_name, out


def main(grid: np.ndarray) -> np.ndarray:
    """Find the outlier quadrant, locate the template match inside it, recolor that match yellow, and return only that quadrant.

    Input structure:
    - The grid has odd height and width, with a grey central row and grey central column forming a divider cross.
    - The four quadrants around that cross are equal-sized black-background regions containing red foreground patterns.
    - Three quadrants are pixelwise identical, while the remaining quadrant is the unique outlier.
    - A small yellow binary template is embedded on the divider cross and is cropped by its own yellow bounding box.

    Rule:
    - Compare the four quadrant matrices and identify the single quadrant that differs from the majority pattern.
    - Extract the yellow template from the divider cross.
    - Search only inside the outlier quadrant for a unique subset match of that template under all eight dihedral transforms.
    - Recolor the matched red cells to yellow and output only the modified outlier quadrant.

    Key insight:
    - The quadrant comparison reduces the search space to one region, and only inside that outlier does the cross template have a unique dihedral subset match.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1
    mid_r = h // 2
    mid_c = w // 2

    cross_mask = np.zeros_like(g, dtype=bool)
    cross_mask[mid_r, :] = True
    cross_mask[:, mid_c] = True
    assert bool(((g == YELLOW) <= cross_mask).all())
    assert bool(np.isin(g[cross_mask], [GREY, YELLOW]).all())
    assert set(map(int, np.unique(g))).issubset({BG, RED, YELLOW, GREY})

    _outlier_name, out = _analyze_instance(g)
    return out


def _random_fragment() -> np.ndarray:
    base = random.choice(FRAGMENTS)
    transformed = np.rot90(base, random.randrange(4))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return transformed.astype(bool)


def _can_place(mask: np.ndarray, fragment: np.ndarray, r0: int, c0: int, gap: int) -> bool:
    h, w = fragment.shape
    qh, qw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > qh or c0 + w > qw:
        return False
    if bool(mask[r0 : r0 + h, c0 : c0 + w][fragment].any()):
        return False

    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(qh, r0 + h + gap)
    cc1 = min(qw, c0 + w + gap)
    if gap > 0 and bool(mask[rr0:rr1, cc0:cc1].any()):
        return False
    return True


def _place_fragment(mask: np.ndarray, fragment: np.ndarray, r0: int, c0: int) -> None:
    h, w = fragment.shape
    mask[r0 : r0 + h, c0 : c0 + w] |= fragment


def _template_cells(variant: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    coords = np.argwhere(variant)
    return {(int(r0 + dr), int(c0 + dc)) for dr, dc in coords}


def _candidate_target_positions(common: np.ndarray, template_mask: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    qh, qw = common.shape
    candidates: list[tuple[int, np.ndarray, int, int]] = []
    for variant in _dihedral_variants(template_mask):
        h, w = variant.shape
        for r0 in range(qh - h + 1):
            for c0 in range(qw - w + 1):
                cells = _template_cells(variant, r0, c0)
                if all(bool(common[r, c]) for r, c in cells):
                    continue
                overlap = sum(int(common[r, c]) for r, c in cells)
                adjacency = 0
                for r, c in cells:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < qh and 0 <= cc < qw and common[rr, cc] and (rr, cc) not in cells:
                            adjacency += 1
                candidates.append((overlap + adjacency, variant.copy(), int(r0), int(c0)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(variant, r0, c0) for _score, variant, r0, c0 in candidates]


def _augment_outlier(common: np.ndarray, target_cells: set[tuple[int, int]], template_mask: np.ndarray) -> np.ndarray | None:
    outlier = common.copy()
    for r, c in target_cells:
        outlier[r, c] = True
    if len(_find_matches(outlier.astype(int) * RED, template_mask)) != 1:
        return None

    edits_needed = random.randint(1, 2)
    extra_changed = 0
    candidates = [(r, c) for r in range(outlier.shape[0]) for c in range(outlier.shape[1]) if (r, c) not in target_cells]
    random.shuffle(candidates)
    for r, c in candidates:
        proposal = outlier.copy()
        proposal[r, c] = not proposal[r, c]
        if proposal.sum() < 4:
            continue
        if len(_find_matches(proposal.astype(int) * RED, template_mask)) != 1:
            continue
        outlier = proposal
        extra_changed += 1
        if extra_changed >= edits_needed:
            break

    if extra_changed < 1:
        return None
    return outlier


def _build_common_quadrant(size: int, template_mask: np.ndarray) -> np.ndarray | None:
    for _ in range(200):
        mask = np.zeros((size, size), dtype=bool)
        fragment_count = random.randint(2, 4)
        for _ in range(fragment_count):
            fragment = _random_fragment()
            gap = random.choice((0, 1))
            placed = False
            for _ in range(40):
                r0 = random.randint(0, size - fragment.shape[0])
                c0 = random.randint(0, size - fragment.shape[1])
                if _can_place(mask, fragment, r0, c0, gap):
                    _place_fragment(mask, fragment, r0, c0)
                    placed = True
                    break
            if not placed:
                break
        else:
            if 5 <= int(mask.sum()) <= size * size - 4 and len(_find_matches(mask.astype(int) * RED, template_mask)) == 0:
                return mask
    return None


def _build_outlier(common: np.ndarray, template_mask: np.ndarray) -> np.ndarray | None:
    for variant, r0, c0 in _candidate_target_positions(common, template_mask):
        target_cells = _template_cells(variant, r0, c0)
        outlier = _augment_outlier(common, target_cells, template_mask)
        if outlier is None:
            continue
        if np.array_equal(outlier, common):
            continue
        return outlier
    return None


def _embed_template(grid: np.ndarray, spec: dict[str, object]) -> None:
    mask = np.asarray(spec["mask"], dtype=bool)
    dr, dc = spec["offset"]
    mid_r = grid.shape[0] // 2
    mid_c = grid.shape[1] // 2
    top = mid_r + int(dr)
    left = mid_c + int(dc)
    coords = np.argwhere(mask)
    for rr, cc in coords:
        grid[top + int(rr), left + int(cc)] = YELLOW


def _render_grid(common: np.ndarray, outlier: np.ndarray, template_spec: dict[str, object], outlier_name: str) -> np.ndarray:
    size = common.shape[0]
    grid = np.zeros((2 * size + 1, 2 * size + 1), dtype=int)
    mid = size
    grid[mid, :] = GREY
    grid[:, mid] = GREY

    quadrants = {"tl": common, "tr": common, "bl": common, "br": common}
    quadrants[outlier_name] = outlier
    grid[:mid, :mid] = quadrants["tl"] * RED
    grid[:mid, mid + 1 :] = quadrants["tr"] * RED
    grid[mid + 1 :, :mid] = quadrants["bl"] * RED
    grid[mid + 1 :, mid + 1 :] = quadrants["br"] * RED
    _embed_template(grid, template_spec)
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        quadrants = _split_quadrants(grid)
        majority_name, outlier_name = _majority_and_outlier(quadrants)
        template_mask = _extract_template_mask(grid)
        for name in QUADRANT_NAMES:
            matches = _find_matches(quadrants[name], template_mask)
            if name == outlier_name:
                if len(matches) != 1:
                    return False
                expected = _paint_match(quadrants[name], matches[0])
            else:
                if len(matches) != 0:
                    return False
        if majority_name == outlier_name:
            return False
        return bool(np.array_equal(main(grid), expected))
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        template_spec = random.choice(TEMPLATE_SPECS)
        template_mask = np.asarray(template_spec["mask"], dtype=bool)
        size = random.randint(5, 8)
        common = _build_common_quadrant(size, template_mask)
        if common is None:
            continue
        outlier = _build_outlier(common, template_mask)
        if outlier is None:
            continue
        outlier_name = random.choice(QUADRANT_NAMES)
        grid = _render_grid(common, outlier, template_spec, outlier_name)
        if _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
