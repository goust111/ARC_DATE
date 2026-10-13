from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_arc_utils_fable_1_008_5", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components


BG = Color.BLACK
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400
UNSET = -1


def _corner_components(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert len(comps) == 4, "expected exactly four visible fragments"

    h, w = map(int, grid.shape)
    by_corner: dict[str, dict] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        hits = []
        if r0 == 0:
            hits.append("T")
        if r1 == h - 1:
            hits.append("B")
        if c0 == 0:
            hits.append("L")
        if c1 == w - 1:
            hits.append("R")
        key = "".join(hits)
        assert key in {"TL", "TR", "BL", "BR"}, "each fragment must occupy one crop corner"
        assert key not in by_corner, "expected one fragment per corner"
        by_corner[key] = comp

    assert set(by_corner) == {"TL", "TR", "BL", "BR"}
    return tuple(crop_bbox(grid, by_corner[key]["bbox"]) for key in ("TL", "TR", "BL", "BR"))


def _valid_row_overlaps(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
) -> list[int]:
    ht = int(top_left.shape[0])
    hb = int(bottom_left.shape[0])
    assert ht == int(top_right.shape[0])
    assert hb == int(bottom_right.shape[0])
    overlaps = []
    for overlap in range(1, min(ht, hb) + 1):
        if np.array_equal(top_left[:overlap, :], bottom_left[hb - overlap :, :]) and np.array_equal(
            top_right[:overlap, :], bottom_right[hb - overlap :, :]
        ):
            overlaps.append(overlap)
    return overlaps


def _valid_col_overlaps(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
) -> list[int]:
    wl = int(top_left.shape[1])
    wr = int(top_right.shape[1])
    assert wl == int(bottom_left.shape[1])
    assert wr == int(bottom_right.shape[1])
    overlaps = []
    for overlap in range(1, min(wl, wr) + 1):
        if np.array_equal(top_left[:, :overlap], top_right[:, wr - overlap :]) and np.array_equal(
            bottom_left[:, :overlap], bottom_right[:, wr - overlap :]
        ):
            overlaps.append(overlap)
    return overlaps


def _stamp(out: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, patch.shape)
    view = out[r0 : r0 + h, c0 : c0 + w]
    clash = (view != UNSET) & (view != patch)
    if np.any(clash):
        return False
    write_mask = view == UNSET
    view[write_mask] = patch[write_mask]
    return True


def _assemble(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
    row_overlap: int,
    col_overlap: int,
) -> np.ndarray | None:
    ht, wl = map(int, top_left.shape)
    hb, wr = map(int, bottom_right.shape)
    sh = ht + hb - int(row_overlap)
    sw = wl + wr - int(col_overlap)
    assert sh > ht and sh > hb and sw > wl and sw > wr

    out = np.full((sh, sw), UNSET, dtype=int)
    placements = (
        (bottom_right, 0, 0),
        (bottom_left, 0, sw - wl),
        (top_right, sh - ht, 0),
        (top_left, sh - ht, sw - wl),
    )
    for patch, r0, c0 in placements:
        if not _stamp(out, patch, int(r0), int(c0)):
            return None
    if np.any(out == UNSET):
        return None
    return out


def _candidate_templates(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
) -> list[np.ndarray]:
    candidates = []
    for row_overlap in _valid_row_overlaps(top_left, top_right, bottom_left, bottom_right):
        for col_overlap in _valid_col_overlaps(top_left, top_right, bottom_left, bottom_right):
            template = _assemble(top_left, top_right, bottom_left, bottom_right, row_overlap, col_overlap)
            if template is None:
                continue
            if not any(np.array_equal(template, prev) for prev in candidates):
                candidates.append(template)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the full rectangular sprite by folding four periodic crop fragments back onto one template.

    Input structure:
    - The background is black.
    - Four multicolor rectangular fragments sit in the four crop corners, separated by black gaps.
    - Each fragment is the visible part of one copy of the same all-nonblack sprite in a 2D periodic stamp lattice, and every copy is clipped by the crop boundary.

    Rule:
    - Treat the four corner fragments as suffix/prefix crops of one shared sprite along both axes.
    - Infer how much the top fragments overlap the bottom fragments, and how much the left fragments overlap the right fragments, by matching their shared colored rows and columns.
    - Fold the four fragments onto one rectangle using that unique row and column overlap, and output the reconstructed sprite.

    Key insight:
    - No complete copy is visible; the answer appears only after identifying which rows and columns are equivalent across period-shifted corner remnants and collapsing them into one template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    top_left, top_right, bottom_left, bottom_right = _corner_components(g)
    candidates = _candidate_templates(top_left, top_right, bottom_left, bottom_right)
    assert len(candidates) == 1, "expected a unique reconstruction"
    return candidates[0]


def _random_template() -> np.ndarray:
    sh = int(RNG.integers(5, 10))
    sw = int(RNG.integers(5, 10))
    color_count = int(RNG.integers(3, min(7, sh * sw) + 1))
    colors = RNG.choice(np.arange(1, 10), size=color_count, replace=False)
    template = RNG.choice(colors, size=(sh, sw), replace=True).astype(int)

    # Break cheap translational symmetries that would make overlap inference ambiguous.
    for r in range(sh):
        template[r, (2 * r + 1) % sw] = int(colors[r % color_count])
    for c in range(sw):
        template[(3 * c + 2) % sh, c] = int(colors[(c + 1) % color_count])
    return template


def _build_input(template: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    sh, sw = map(int, template.shape)
    a = int(RNG.integers(1, sh - 1))
    b = int(RNG.integers(1, sw - 1))
    ht = sh - a
    wl = sw - b

    row_overlap = int(RNG.integers(1, ht))
    col_overlap = int(RNG.integers(1, wl))
    hb = a + row_overlap
    wr = b + col_overlap

    gap_r = int(RNG.integers(1, 5))
    gap_c = int(RNG.integers(1, 5))

    height = ht + gap_r + hb
    width = wl + gap_c + wr
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BG, dtype=int)
    grid[:ht, :wl] = template[a:, b:]
    grid[:ht, width - wr :] = template[a:, :wr]
    grid[height - hb :, :wl] = template[:hb, b:]
    grid[height - hb :, width - wr :] = template[:hb, :wr]

    meta = {
        "a": a,
        "b": b,
        "hb": hb,
        "wr": wr,
        "ht": ht,
        "wl": wl,
        "row_overlap": row_overlap,
        "col_overlap": col_overlap,
        "gap_r": gap_r,
        "gap_c": gap_c,
    }
    return grid, meta


def _foreground_components(grid: np.ndarray) -> list[dict]:
    return find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")


def _validate_instance_geometry(grid: np.ndarray, template: np.ndarray, meta: dict[str, int]) -> bool:
    comps = _foreground_components(grid)
    if len(comps) != 4:
        return False

    h, w = map(int, grid.shape)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            return False
        frag_h = r1 - r0 + 1
        frag_w = c1 - c0 + 1
        if frag_h >= template.shape[0] or frag_w >= template.shape[1]:
            return False

    top_left, top_right, bottom_left, bottom_right = _corner_components(grid)
    row_candidates = _valid_row_overlaps(top_left, top_right, bottom_left, bottom_right)
    col_candidates = _valid_col_overlaps(top_left, top_right, bottom_left, bottom_right)
    return row_candidates == [meta["row_overlap"]] and col_candidates == [meta["col_overlap"]]


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        template = _random_template()
        grid, meta = _build_input(template)
        if not _validate_instance_geometry(grid, template, meta):
            continue

        candidates = _candidate_templates(*_corner_components(grid))
        if len(candidates) != 1:
            continue
        if not np.array_equal(candidates[0], template):
            continue
        if not np.array_equal(main(grid), template):
            continue
        if meta["row_overlap"] < 1 or meta["col_overlap"] < 1:
            continue
        return grid
    raise RuntimeError("failed to generate a valid periodic-fragment reconstruction input")
