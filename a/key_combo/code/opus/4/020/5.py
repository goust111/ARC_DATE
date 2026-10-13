from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


def _install_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        candidate = base / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            return

    fallback = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if fallback.exists():
        utils_dir = str(fallback.parent)
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        return

    raise ImportError("could not locate new/solvers/arc_utils.py")


_install_arc_utils_path()

from arc_utils import Color, find_components  # noqa: E402


Coord = tuple[int, int]
Motif = tuple[tuple[Coord, ...], tuple[Coord, ...]]

GEN_ATTEMPTS = 500
HALF_HEIGHT_RANGE = (7, 14)
WIDTH_RANGE = (7, 13)
EXTRA_MOTIF_COUNT_CHOICES = (0, 1, 1, 2, 2, 3)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_coords(cells: set[Coord]) -> tuple[Coord, ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_bbox(cells: tuple[Coord, ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _random_connected_shape(size: int) -> tuple[Coord, ...]:
    cells: set[Coord] = {(0, 0)}
    while len(cells) < int(size):
        frontier: list[Coord] = []
        frontier_seen: set[Coord] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cells or (nr, nc) in frontier_seen:
                    continue
                frontier.append((nr, nc))
                frontier_seen.add((nr, nc))
        cells.add(random.choice(frontier))
    return _normalize_coords(cells)


def _make_singleton_diff() -> Motif:
    if random.random() < 0.5:
        return (((0, 0),), ())
    return ((), ((0, 0),))


def _make_exclusive_shape(owner: str | None = None) -> Motif:
    owner = owner or random.choice(("upper", "lower"))
    shape = _random_connected_shape(random.randint(1, 5))
    if owner == "upper":
        return (shape, ())
    return ((), shape)


def _make_overlap_shape() -> Motif:
    shape = _random_connected_shape(random.randint(1, 4))
    return (shape, shape)


def _make_mixed_diff() -> Motif:
    shape = list(_random_connected_shape(random.randint(2, 5)))
    random.shuffle(shape)
    split = random.randint(1, len(shape) - 1)
    upper = tuple(sorted(shape[:split]))
    lower = tuple(sorted(shape[split:]))
    return upper, lower


def _components_from_mask(mask: np.ndarray) -> list[dict]:
    comp_grid = np.where(mask, Color.BLUE, Color.BLACK).astype(int)
    return find_components(
        comp_grid,
        bg=Color.BLACK,
        colors=[Color.BLUE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )


def _solve_from_halves(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    both_bg = (upper == Color.BLACK) & (lower == Color.BLACK)
    both_fg = (upper == Color.GREEN) & (lower == Color.ORANGE)
    diff_mask = ((upper == Color.GREEN) ^ (lower == Color.ORANGE))

    out = np.full(upper.shape, Color.BLACK, dtype=int)
    out[both_bg] = Color.GREY
    out[both_fg] = Color.BLACK

    for component in _components_from_mask(diff_mask):
        coords = component["coords"]
        fill = Color.BLUE if int(component["area"]) == 1 else Color.RED
        out[coords[:, 0], coords[:, 1]] = int(fill)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compare the stacked masks, then recolor difference components by 4-connected size.

    Input structure:
    - The grid has an even number of rows and a black background.
    - The top half is a binary mask whose foreground color is green.
    - The bottom half is a same-size binary mask whose foreground color is orange.

    Rule:
    - Split the input into equal top and bottom halves and compare corresponding cells.
    - If both cells are background, output grey; if both are foreground, output black.
    - Merge all upper-only and lower-only cells into a single difference mask.
    - Partition that difference mask into 4-connected components, recoloring size-1 components blue and size-2-or-larger components red.

    Key insight:
    - The exclusive cells from both halves must be combined before classification: output color depends on the merged difference component size, not on which half supplied each cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % 2 == 0 and g.shape[0] > 0

    half = g.shape[0] // 2
    upper = g[:half, :]
    lower = g[half:, :]

    assert set(map(int, np.unique(upper))) <= {Color.BLACK, Color.GREEN}
    assert set(map(int, np.unique(lower))) <= {Color.BLACK, Color.ORANGE}

    return _solve_from_halves(upper, lower)


def _candidate_positions(used: np.ndarray, motif: Motif) -> list[Coord]:
    half_h, width = used.shape
    union_cells = tuple(sorted(set(motif[0]) | set(motif[1])))
    shape_h, shape_w = _shape_bbox(union_cells)
    candidates: list[Coord] = []

    for r0 in range(half_h - shape_h + 1):
        for c0 in range(width - shape_w + 1):
            ok = True
            for dr, dc in union_cells:
                rr, cc = r0 + dr, c0 + dc
                if used[rr, cc]:
                    ok = False
                    break
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < half_h and 0 <= nc < width and used[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))

    random.shuffle(candidates)
    return candidates


def _place_motif(upper: np.ndarray, lower: np.ndarray, used: np.ndarray, motif: Motif) -> bool:
    for r0, c0 in _candidate_positions(used, motif):
        for dr, dc in motif[0]:
            upper[r0 + dr, c0 + dc] = Color.GREEN
            used[r0 + dr, c0 + dc] = True
        for dr, dc in motif[1]:
            lower[r0 + dr, c0 + dc] = Color.ORANGE
            used[r0 + dr, c0 + dc] = True
        return True
    return False


def _build_candidate() -> np.ndarray | None:
    half_h = random.randint(*HALF_HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    upper = np.full((half_h, width), Color.BLACK, dtype=int)
    lower = np.full((half_h, width), Color.BLACK, dtype=int)
    used = np.zeros((half_h, width), dtype=bool)

    motifs: list[Motif] = [
        _make_singleton_diff(),
        _make_mixed_diff(),
        _make_overlap_shape(),
    ]

    extra_count = random.choice(EXTRA_MOTIF_COUNT_CHOICES)
    for _ in range(extra_count):
        motif_kind = random.choice(("upper", "lower", "mixed", "overlap"))
        if motif_kind == "upper":
            motifs.append(_make_exclusive_shape("upper"))
        elif motif_kind == "lower":
            motifs.append(_make_exclusive_shape("lower"))
        elif motif_kind == "mixed":
            motifs.append(_make_mixed_diff())
        else:
            motifs.append(_make_overlap_shape())

    random.shuffle(motifs)
    for motif in motifs:
        if not _place_motif(upper, lower, used, motif):
            return None

    return np.vstack([upper, lower]).astype(int)


def _validate_generated_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] % 2 != 0:
        return False

    half = g.shape[0] // 2
    upper = g[:half, :]
    lower = g[half:, :]

    if set(map(int, np.unique(upper))) - {Color.BLACK, Color.GREEN}:
        return False
    if set(map(int, np.unique(lower))) - {Color.BLACK, Color.ORANGE}:
        return False

    upper_components = _components_from_mask(upper == Color.GREEN)
    lower_components = _components_from_mask(lower == Color.ORANGE)
    if not upper_components or not lower_components:
        return False
    if any(not (1 <= int(comp["area"]) <= 5) for comp in upper_components + lower_components):
        return False

    both_bg = (upper == Color.BLACK) & (lower == Color.BLACK)
    both_fg = (upper == Color.GREEN) & (lower == Color.ORANGE)
    diff_mask = ((upper == Color.GREEN) ^ (lower == Color.ORANGE))
    diff_components = _components_from_mask(diff_mask)

    if not both_bg.any() or not both_fg.any():
        return False
    if not any(int(comp["area"]) == 1 for comp in diff_components):
        return False
    if not any(int(comp["area"]) >= 2 for comp in diff_components):
        return False

    mixed_component_found = False
    for component in diff_components:
        coords = component["coords"]
        has_upper_only = bool(np.any(upper[coords[:, 0], coords[:, 1]] == Color.GREEN))
        has_lower_only = bool(np.any(lower[coords[:, 0], coords[:, 1]] == Color.ORANGE))
        if has_upper_only and has_lower_only:
            mixed_component_found = True
            break
    if not mixed_component_found:
        return False

    out = main(g)
    if out.shape != upper.shape:
        return False
    if set(map(int, np.unique(out))) != {Color.BLACK, Color.BLUE, Color.RED, Color.GREY}:
        return False

    expected = np.full(out.shape, Color.BLACK, dtype=int)
    expected[both_bg] = Color.GREY
    expected[both_fg] = Color.BLACK
    for component in diff_components:
        coords = component["coords"]
        fill = Color.BLUE if int(component["area"]) == 1 else Color.RED
        expected[coords[:, 0], coords[:, 1]] = int(fill)

    return np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None and _validate_generated_input(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 5")
