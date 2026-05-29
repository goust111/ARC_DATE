# Combo variant of 269e22fb (index 9)
# combo_spec: D1/3+D7/0
# Name: 小窗口轮廓线匹配

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import colors_present


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Original task:
#   a medium-size two-color crop exposed every pixel inside the window, so the
#   solver could compare the full fg/bg bitmap against every offset of every
#   dihedral orientation.
# - Combo change 1 (D1/3):
#   the crop is converted into a three-state contour view. Cells on a local
#   fg/bg boundary keep the motif's fg or bg color, while every non-edge cell
#   becomes one uniform filler color. Matching therefore uses the line geometry
#   `{bg-edge, fg-edge, filler}` instead of raw region occupancy.
# - Combo change 2 (D7/0):
#   the crop is also shrunk to a tiny window family:
#   `(5x7, 6x6, 6x7, 7x5, 7x6, 7x7)`.
# - Concrete trace:
#   suppose the hidden source window were a 6x6 patch whose true binary content
#   contains a short zig-zag boundary. In the original task, all 36 cells would
#   be visible in two colors and exact equality would identify the offset. In
#   the combo task, only the cells touching that boundary remain colored on the
#   fg side or bg side; flat interior cells are replaced by the filler. The
#   solver must first separate the filler, then compare the sparse contour
#   signature of that tiny patch against all tiny windows from the 8 motif poses.
#
# b) Complexity calibration
# - The crop sizes are deliberately far below the original `8-16 x 8-12` regime,
#   so the task really uses "extreme partial evidence" rather than a lightly
#   modified original crop.
# - To keep this solvable, the window sizes are restricted to a small repeated
#   family instead of an unconstrained 1-7 range; this preserves visual variety
#   without exploding ambiguity.
# - To keep the contour visibly sparse, the generator further keeps only windows
#   whose colored edge pixels cover at most 45% of the crop area.
# - The generator precomputes which tiny contour windows are uniquely solvable
#   and samples only from that catalog, with explicit coverage across all 6 tiny
#   sizes and all 8 dihedral poses. This keeps the fusion hard but not noisy.
#
# c) Implicit-assumption audit of main()
# - `assert len(colors) == 2` from the base solver breaks immediately because the
#   filler introduces a third role and third color.
# - `visible = g == fg` no longer reconstructs the hidden binary crop, because
#   filler cells are not "background"; they mean "unknown interior".
# - Per-pixel equality against `pattern[r:r+H, c:c+W]` is invalid. The solver
#   must derive a contour signature for candidate windows and compare that to the
#   observed three-state pattern.
# - D7 further breaks the old "medium crops are unique by size alone" intuition.
#   With 5-7 sized windows, uniqueness must be enforced explicitly by generation.
#
# d) Placement legality
# - Every input is still one axis-aligned crop fully contained inside a 20x20
#   dihedral orientation of the fixed motif, so there is no overlap or clipping.
# - The contour extraction itself is local to the crop, using only in-window
#   4-neighbor disagreements. This keeps the rendered input consistent with the
#   visible rectangle and avoids any dependence on hidden outside pixels.
#
# e) Visual observability
# - A human sees a strong, repeated visual pattern across examples: tiny windows
#   with two thin outline colors and a dominant third filler color.
# - The change is structural rather than numeric. No inference depends on ARC
#   color ids or any hidden ordering of the palette.
#
# f) Visual cleanliness
# - Only three visual roles exist: foreground edge, background edge, and filler.
#   The outline is one cell thick, and the filler occupies the interior, so the
#   sparse window stays readable despite the high ambiguity of its size.
#
# g) Inferability of the changes
# - Color remapping remains simple: once the motif pose is recovered, the full
#   20x20 output is painted with the two non-filler colors in motif fg/bg roles.
# - Because the filler never appears in the output, the examples consistently
#   teach that the third color is merely a masking aid for hidden interiors.
#
# h) Anti-degeneration guarantees
# - Every generated sample uses one of the tiny window sizes above, so the D7
#   shrinkage is always active.
# - Every generated sample contains all three contour states, so the D1 contour
#   encoding is always active and never collapses back to a plain two-color crop.
# - Every generated sample also has filler-majority area by construction, so the
#   contour view stays sparse rather than reverting to an almost fully visible
#   crop in disguise.
#
# i) New solver assumptions and generator support
# - Assumption: the input size is one of the supported tiny sizes.
#   Generator support: it samples only from that fixed size family.
# - Assumption: the observed contour signature is uniquely solvable.
#   Generator support: it samples only from the precomputed unique-signature
#   catalog across all 8 orientations and all legal offsets, while balancing the
#   size and orientation branches during repeated generation.
# - Assumption: both contour-side colors and the filler all appear at least once.
#   Generator support: the catalog excludes windows missing any of the three
#   states before any recoloring happens.
# - Assumption: the filler truly hides substantial interior area.
#   Generator support: the catalog excludes windows whose visible contour pixels
#   would exceed 45% of the crop.


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)

_PALETTE = tuple(range(10))
_SMALL_SIZES = ((5, 7), (6, 6), (6, 7), (7, 5), (7, 6), (7, 7))
_MAX_VISIBLE_FRACTION = 0.45


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _edge_mask(window: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(window, dtype=bool)
    vertical = window[:-1, :] != window[1:, :]
    horizontal = window[:, :-1] != window[:, 1:]
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    return edge


def _contour_signature(window: np.ndarray) -> np.ndarray:
    edge = _edge_mask(window)
    sig = np.full(window.shape, 2, dtype=np.int8)
    sig[edge & window] = 1
    sig[edge & ~window] = 0
    return sig


def _render_contour_input(signature: np.ndarray, fg: int, bg: int, filler: int) -> np.ndarray:
    out = np.full(signature.shape, int(filler), dtype=int)
    out[signature == 1] = int(fg)
    out[signature == 0] = int(bg)
    return out


def _build_catalog() -> tuple[dict[tuple[int, int], dict[bytes, tuple[int, int, int]]], list[dict[str, object]]]:
    lookup: dict[tuple[int, int], dict[bytes, tuple[int, int, int]]] = {}
    unique_entries: list[dict[str, object]] = []

    for H, W in _SMALL_SIZES:
        buckets: dict[bytes, list[tuple[int, int, int]]] = {}
        for variant_idx, pattern in enumerate(_VARIANTS):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    signature = _contour_signature(pattern[r : r + H, c : c + W])
                    visible = int(np.count_nonzero(signature != 2))
                    if visible == 0 or visible / (H * W) > _MAX_VISIBLE_FRACTION:
                        continue
                    counts = tuple(int(np.count_nonzero(signature == value)) for value in (0, 1, 2))
                    if min(counts) == 0:
                        continue
                    key = signature.tobytes()
                    buckets.setdefault(key, []).append((int(variant_idx), int(r), int(c)))

        lookup[(H, W)] = {}
        for key, matches in buckets.items():
            if len(matches) != 1:
                continue
            variant_idx, r, c = matches[0]
            lookup[(H, W)][key] = (variant_idx, r, c)
            unique_entries.append(
                {
                    "shape": (H, W),
                    "variant_idx": int(variant_idx),
                    "offset": (int(r), int(c)),
                    "signature": np.frombuffer(key, dtype=np.int8).reshape(H, W).copy(),
                }
            )

    return lookup, unique_entries


def _group_entries(
    entries: list[dict[str, object]]
) -> dict[tuple[int, int], dict[int, list[dict[str, object]]]]:
    grouped: dict[tuple[int, int], dict[int, list[dict[str, object]]]] = {
        shape: {variant_idx: [] for variant_idx in range(len(_VARIANTS))}
        for shape in _SMALL_SIZES
    }
    for entry in entries:
        shape = tuple(entry["shape"])
        variant_idx = int(entry["variant_idx"])
        grouped[shape][variant_idx].append(entry)
    return grouped


_BASE_MASK = _pattern_mask()
_VARIANTS = _dihedral_variants(_BASE_MASK)
_SIGNATURE_LOOKUP, _UNIQUE_WINDOWS = _build_catalog()
_ENTRIES_BY_SHAPE_AND_VARIANT = _group_entries(_UNIQUE_WINDOWS)


def _candidate_matches(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    size_lookup = _SIGNATURE_LOOKUP.get((H, W))
    assert size_lookup is not None

    colors = colors_present(g)
    matches: list[tuple[int, int, np.ndarray]] = []
    for filler in colors:
        edge_colors = [int(color) for color in colors if int(color) != int(filler)]
        if len(edge_colors) != 2:
            continue
        for fg in edge_colors:
            bg = int(edge_colors[0] if edge_colors[1] == fg else edge_colors[1])
            observed = np.full((H, W), 2, dtype=np.int8)
            observed[g == bg] = 0
            observed[g == fg] = 1
            match = size_lookup.get(observed.tobytes())
            if match is None:
                continue
            variant_idx, _r, _c = match
            matches.append((int(fg), int(bg), _VARIANTS[int(variant_idx)]))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the full 20x20 motif from a tiny contour-only crop of one dihedral pose.

    Input structure:
    - The grid is a very small rectangular crop chosen from one rotation/reflection of the shared 20x20 motif.
    - Only local fg/bg boundary cells keep motif colors; every non-boundary cell inside the crop is replaced by a third filler color.

    Rule:
    - Treat each input color as the possible filler, and the other two colors as the two contour-side labels.
    - Convert the observed crop into a three-state contour signature and compare it against every legal tiny window of every dihedral motif orientation.
    - Use the unique matching signature to recover foreground/background polarity, then repaint the whole 20x20 motif with those two non-filler colors.

    Key insight:
    - Even though the window is much smaller than in the base task and most cells are hidden by filler, the remaining two-sided outline geometry is still distinctive enough to identify one motif orientation and offset.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20
    assert (H, W) in _SIGNATURE_LOOKUP

    colors = colors_present(g)
    assert len(colors) == 3

    matches = _candidate_matches(g)
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(200):
        shape = _SMALL_SIZES[int(rng.integers(len(_SMALL_SIZES)))]
        variant_idx = int(rng.integers(len(_VARIANTS)))
        bucket = _ENTRIES_BY_SHAPE_AND_VARIANT[shape][variant_idx]
        assert bucket
        entry = bucket[int(rng.integers(len(bucket)))]
        fg, bg, filler = map(int, rng.choice(_PALETTE, size=3, replace=False))
        signature = np.asarray(entry["signature"], dtype=np.int8)
        contour_grid = _render_contour_input(signature, fg, bg, filler)
        expected = np.where(_VARIANTS[int(entry["variant_idx"])], int(fg), int(bg)).astype(int)

        try:
            solved = main(contour_grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return contour_grid

    raise RuntimeError("failed to generate a unique tiny contour crop")
