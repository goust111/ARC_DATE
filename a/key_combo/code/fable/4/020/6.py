from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import TypedDict

import numpy as np
from scipy import ndimage


_THIS_FILE = Path(__file__).resolve()
for _candidate in [_THIS_FILE, *_THIS_FILE.parents]:
    _solvers_root = _candidate / "new" / "solvers"
    if _solvers_root.exists() and (_solvers_root / "arc_utils.py").exists():
        if str(_solvers_root) not in sys.path:
            sys.path.insert(0, str(_solvers_root))
        break
else:
    raise ImportError("Could not locate arc_utils.py from project tree")

from arc_utils import find_components, get_bg, crop_bbox, paste  # noqa: E402


_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


class _ObjectInfo(TypedDict):
    bbox: tuple[int, int, int, int]
    local: np.ndarray
    outer_color: int
    inner_color: int


def _object_pair_info(grid: np.ndarray, comp: dict, bg: int) -> _ObjectInfo:
    bbox = comp["bbox"]
    r0, c0, r1, c1 = map(int, bbox)
    local = crop_bbox(grid, bbox)
    values = local[local != bg]
    colors = set(map(int, np.unique(values)))
    if len(colors) != 2:
        raise AssertionError("each object must use exactly two non-background colors")

    rr = comp["coords"][:, 0] - r0
    cc = comp["coords"][:, 1] - c0
    component_mask = np.zeros(local.shape, dtype=bool)
    component_mask[rr, cc] = True

    border = component_mask & ~ndimage.binary_erosion(component_mask, structure=_STRUCT4)
    if not border.any():
        border = component_mask

    outer_color = None
    boundary_hits = {}
    for color in colors:
        boundary_hits[color] = int(np.sum((local == color) & border))

    max_hits = max(boundary_hits.values())
    tied = [color for color, hits in boundary_hits.items() if hits == max_hits]
    if len(tied) == 1:
        outer_color = tied[0]
    else:
        edge = np.zeros_like(component_mask, dtype=bool)
        edge[0, :] = component_mask[0, :]
        edge[-1, :] = component_mask[-1, :]
        edge[:, 0] = edge[:, 0] | component_mask[:, 0]
        edge[:, -1] = edge[:, -1] | component_mask[:, -1]
        edge_hits = {
            color: int(np.sum((local == color) & edge))
            for color in colors
        }
        max_edge = max(edge_hits.values())
        tied = [color for color, hits in edge_hits.items() if hits == max_edge]
        if len(tied) != 1:
            raise AssertionError("cannot disambiguate border and core colors")
        outer_color = tied[0]

    outer_color = int(outer_color)
    inner_color = int(next(color for color in colors if int(color) != outer_color))
    return _ObjectInfo(
        bbox=(r0, c0, r1, c1),
        local=local,
        outer_color=outer_color,
        inner_color=inner_color,
    )


def _find_outlier_pair(objects: list[_ObjectInfo]) -> tuple[int, int]:
    pairs = Counter((obj["outer_color"], obj["inner_color"]) for obj in objects)
    candidates = [
        pair for pair, count in pairs.items()
        if count == 1 and pairs[(pair[1], pair[0])] >= 2
    ]
    if len(candidates) == 1:
        return candidates[0]

    # Robust fallback for unexpected but still close cases:
    if len(candidates) == 0:
        singles = [pair for pair, count in pairs.items() if count == 1]
        if len(singles) == 1:
            return singles[0]
        if len(singles) == 0:
            raise AssertionError("no inverted pair candidate found")
        candidates = singles

    # Pick the candidate whose reverse orientation is strongest.
    candidates.sort(key=lambda pair: pairs[(pair[1], pair[0])], reverse=True)
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Fix the unique inverted frame-core object by swapping its two colors and crop it.

    Input structure:
    - The background is the most common color.
    - Every foreground object is one separated 4-connected component with exactly two colors.
    - In each valid object, one color forms the outer frame and the other color fills the core.

    Rule:
    - For each component, infer ordered pairing as (outer_color, inner_color) from boundary pixels.
    - Find the globally unique ordered pair that is inverted relative to the repeated legal pairs.
    - Swap outer and inner colors only inside that object, then return the repaired object as a tight crop.

    Key insight:
    - The anomaly is only visible after aggregating the entire set of object color pairings,
      then correcting the single object whose pairing is the sole inverse of a majority rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert components, "expected at least one foreground component"

    object_infos: list[_ObjectInfo] = []
    for comp in components:
        info = _object_pair_info(g, comp, bg=bg)
        object_infos.append(info)

    target_pair = _find_outlier_pair(object_infos)
    target = next(
        info for info in object_infos
        if (info["outer_color"], info["inner_color"]) == target_pair
    )

    local = target["local"]
    repaired = local.copy()
    outer_mask = local == target["outer_color"]
    inner_mask = local == target["inner_color"]
    repaired[outer_mask] = target["inner_color"]
    repaired[inner_mask] = target["outer_color"]
    return repaired


def _sample_legal_pairs(rng: np.random.Generator, bg: int) -> list[tuple[int, int]]:
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    n_pairs = 2 if rng.random() < 0.75 else 3
    return [(int(palette[i]), int(palette[i + 1])) for i in range(0, n_pairs * 2, 2)]


def _make_object_patch(
    outer: int,
    inner: int,
    rng: np.random.Generator,
) -> np.ndarray:
    thickness = int(rng.choice([1, 1, 2]))
    h = int(rng.integers(5, 9))
    w = int(rng.integers(5, 9))
    h = max(h, 2 * thickness + 3)
    w = max(w, 2 * thickness + 3)

    patch = np.full((h, w), inner, dtype=int)
    for i in range(thickness):
        patch[i, :] = outer
        patch[-1 - i, :] = outer
        patch[:, i] = outer
        patch[:, -1 - i] = outer

    if rng.random() < 0.4:
        patch = np.rot90(patch, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _place_object(
    grid: np.ndarray,
    occupancy: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    H, W = map(int, grid.shape)
    h, w = map(int, patch.shape)
    if h + 2 > H or w + 2 > W:
        return None

    for _ in range(500):
        r = int(rng.integers(0, H - h + 1))
        c = int(rng.integers(0, W - w + 1))
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(H, r + h + 1)
        c1 = min(W, c + w + 1)
        if occupancy[r0:r1, c0:c1].any():
            continue

        paste(grid, patch, r, c)
        occupancy[r : r + h, c : c + w] = True
        return r, c
    return None


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    legal_pairs = _sample_legal_pairs(rng, bg=bg)
    legal_counts = [int(rng.integers(2, 4)) for _ in legal_pairs]

    outlier_idx = int(rng.integers(0, len(legal_pairs)))
    specs = []
    for pair, count in zip(legal_pairs, legal_counts):
        for _ in range(count):
            specs.append(pair)
    specs.append((legal_pairs[outlier_idx][1], legal_pairs[outlier_idx][0]))

    rng.shuffle(specs)

    object_count = len(specs)
    base_size = min(30, 12 + object_count * 3)
    min_size = min(30, base_size - 2)
    H = int(rng.integers(max(10, min_size), 31))
    W = int(rng.integers(max(10, min_size), 31))

    grid = np.full((H, W), bg, dtype=int)
    occupancy = np.zeros((H, W), dtype=bool)
    patches = []
    for outer, inner in specs:
        patches.append((outer, inner, _make_object_patch(outer, inner, rng)))
    patches.sort(key=lambda item: item[2].size, reverse=True)

    for outer, inner, patch in patches:
        if _place_object(grid, occupancy, patch, rng) is None:
            return None
        _ = (outer, inner)  # keep pair metadata explicit for readability
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        candidate = _generate_candidate(rng)
        if candidate is None:
            continue
        try:
            out = main(candidate)
        except Exception:
            continue
        if not isinstance(out, np.ndarray):
            continue
        if out.ndim != 2 or out.size == 0:
            continue
        return candidate
    raise RuntimeError("failed to generate a valid 6 input")


