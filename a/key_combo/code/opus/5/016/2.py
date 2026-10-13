from __future__ import annotations

import random
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_016_2", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds

BG = Color.BLACK
SEP = Color.GREY
FG = Color.TEAL
REF_ERASE = Color.RED
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = random.Random()


def _fg_neighbor_dirs(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            dirs.append((dr, dc))
    return dirs


def _row_span(mask: np.ndarray, r: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(mask[int(r)])
    if cols.size == 0:
        return None
    return int(cols[0]), int(cols[-1])


def _row_feature(mask: np.ndarray, r: int) -> dict | None:
    span = _row_span(mask, r)
    if span is None:
        return None
    left, right = span
    candidates: list[dict] = []

    left_dirs = _fg_neighbor_dirs(mask, r, left)
    if left_dirs == [(0, 1)]:
        candidates.append({"row": int(r), "side": "left", "kind": "tab", "cell": (int(r), int(left))})

    right_dirs = _fg_neighbor_dirs(mask, r, right)
    if right_dirs == [(0, -1)]:
        candidates.append({"row": int(r), "side": "right", "kind": "tab", "cell": (int(r), int(right))})

    notch_c = int(left - 1)
    if notch_c >= 0 and not bool(mask[r, notch_c]):
        notch_dirs = set(_fg_neighbor_dirs(mask, r, notch_c))
        if notch_dirs == {(-1, 0), (1, 0), (0, 1)}:
            candidates.append({"row": int(r), "side": "left", "kind": "notch", "cell": (int(r), int(notch_c))})

    notch_c = int(right + 1)
    if notch_c < mask.shape[1] and not bool(mask[r, notch_c]):
        notch_dirs = set(_fg_neighbor_dirs(mask, r, notch_c))
        if notch_dirs == {(-1, 0), (1, 0), (0, -1)}:
            candidates.append({"row": int(r), "side": "right", "kind": "notch", "cell": (int(r), int(notch_c))})

    if len(candidates) != 1:
        return None
    return candidates[0]


def _ref_rows_and_colors(grid: np.ndarray, sep_col: int) -> list[tuple[int, int]]:
    refs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r, sep_col + 1 :] != BG)
        if cols.size == 0:
            continue
        color = int(grid[r, sep_col + 1 + int(cols[0])])
        refs.append((int(r), color))
    return refs


def _largest_fg_mask(left_grid: np.ndarray) -> np.ndarray:
    comps = find_components(
        left_grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FG],
        return_format="dict",
    )
    if not comps:
        return left_grid == FG
    comp = max(comps, key=lambda item: int(item["area"]))
    mask = np.zeros(left_grid.shape, dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r), int(c)] = True
    return mask


def _apply_completion(out: np.ndarray, feature: dict) -> None:
    r = int(feature["row"])
    span = _row_span(out == FG, r)
    if span is None:
        return
    left, right = span
    if feature["kind"] == "tab":
        target_c = int(right + 1 if feature["side"] == "left" else left - 1)
        if 0 <= target_c < out.shape[1]:
            out[r, target_c] = FG
        return

    target_c = int(right if feature["side"] == "left" else left)
    if 0 <= target_c < out.shape[1]:
        out[r, target_c] = BG


def _apply_erase(out: np.ndarray, feature: dict) -> None:
    r, c = map(int, feature["cell"])
    if feature["kind"] == "tab":
        out[r, c] = BG
    else:
        out[r, c] = FG


def _simulate_from_refs(left_grid: np.ndarray, refs: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(left_grid, dtype=int, copy=True)
    for r, ref_color in sorted(refs):
        feature = _row_feature(out == FG, r)
        if feature is None:
            continue
        if int(ref_color) == FG:
            _apply_completion(out, feature)
        elif int(ref_color) == REF_ERASE:
            _apply_erase(out, feature)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use row-aligned reference colors to either mirror-complete or erase one-pixel side features of a thick teal polygon.

    Input structure:
    - Black (0) is background. A full-height grey (5) separator column splits the grid into a left shape region and a right reference region.
    - The left region contains one thick teal (8) rectilinear polygon with single-pixel tabs and notches on its left and right side boundaries.
    - The right region contains one reference pixel on each feature row; each such pixel is teal (8) or red (2).

    Rule:
    - Read the feature rows from top to bottom. On each referenced row, identify the current side-boundary feature in the left polygon.
    - If that row's reference pixel is teal (8), complete the shape symmetrically across the polygon's row span: a left/right tab creates a matching tab outside the opposite side, and a left/right notch removes the opposite edge pixel to create a matching notch.
    - If that row's reference pixel is red (2), erase the feature locally: delete a tab pixel or fill a notch pixel so the boundary becomes smooth.
    - After processing all referenced rows, discard the separator and the entire right reference region; output only the transformed left region.

    Key insight:
    - The reference color does not choose a destination location directly; it chooses between two different operations on the aligned boundary feature, while the actual mirrored target comes from that row's current left-to-right span through the teal polygon.
    """
    g = np.array(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    sep_cols = [c for c in range(g.shape[1]) if bool((g[:, c] == SEP).all())]
    assert sep_cols, "expected a full-height grey separator"
    sep_col = int(sep_cols[0])

    left = np.full((g.shape[0], sep_col), BG, dtype=int)
    fg_mask = _largest_fg_mask(g[:, :sep_col])
    left[fg_mask] = FG

    refs = _ref_rows_and_colors(g, sep_col)
    out = _simulate_from_refs(left, refs)
    return out


def _build_clean_shape(height: int, width: int) -> tuple[np.ndarray, list[int], list[int]]:
    mask = np.zeros((height, width), dtype=bool)

    band_heights = []
    remaining = int(height - 2)
    for i in range(3):
        min_h = 3
        max_h = remaining - 3 * (2 - i)
        chosen = _RNG.randint(min_h, max_h)
        band_heights.append(chosen)
        remaining -= chosen

    mid_left = _RNG.randint(2, 4)
    mid_right = _RNG.randint(width - 7, width - 4)
    top_left = max(1, min(mid_left + _RNG.choice([-2, -1, 0, 1]), mid_right - 5))
    top_right = min(width - 2, max(mid_right + _RNG.choice([-1, 0, 1, 2]), top_left + 5))
    bot_left = max(1, min(mid_left + _RNG.choice([-2, -1, 0, 1]), mid_right - 5))
    bot_right = min(width - 2, max(mid_right + _RNG.choice([-1, 0, 1, 2]), bot_left + 5))

    row_lefts = [0] * height
    row_rights = [0] * height
    row = 1
    specs = [
        (band_heights[0], top_left, top_right),
        (band_heights[1], mid_left, mid_right),
        (band_heights[2], bot_left, bot_right),
    ]
    for band_h, left, right in specs:
        for _ in range(band_h):
            row_lefts[row] = int(left)
            row_rights[row] = int(right)
            mask[row, left : right + 1] = True
            row += 1

    return mask, row_lefts, row_rights


def _feature_candidates(clean_mask: np.ndarray, row_lefts: list[int], row_rights: list[int]) -> dict[str, list[int]]:
    candidates = {"left": [], "right": []}
    for r in range(1, clean_mask.shape[0] - 1):
        left = int(row_lefts[r])
        right = int(row_rights[r])
        if left == 0 and right == 0:
            continue
        if left == int(row_lefts[r - 1]) == int(row_lefts[r + 1]) and right - left + 1 >= 6 and left >= 1:
            candidates["left"].append(int(r))
        if right == int(row_rights[r - 1]) == int(row_rights[r + 1]) and right - left + 1 >= 6 and right + 1 < clean_mask.shape[1]:
            candidates["right"].append(int(r))
    return candidates


def _rows_far_enough(rows: list[int]) -> bool:
    return all(int(b) - int(a) >= 2 for a, b in zip(sorted(rows), sorted(rows)[1:]))


def _place_required_features(candidates: dict[str, list[int]]) -> list[dict] | None:
    required = [
        {"side": "left", "kind": "tab", "ref": FG},
        {"side": "left", "kind": "notch", "ref": REF_ERASE},
        {"side": "right", "kind": "notch", "ref": FG},
        {"side": "right", "kind": "tab", "ref": REF_ERASE},
    ]
    picks: list[dict] = []
    used_rows: list[int] = []
    for need in required:
        valid_rows = [
            r
            for r in candidates[need["side"]]
            if r not in used_rows and _rows_far_enough(used_rows + [r])
        ]
        if not valid_rows:
            return None
        row = int(_RNG.choice(valid_rows))
        used_rows.append(row)
        picks.append({"row": row, **need})
    return picks


def _add_optional_features(features: list[dict], candidates: dict[str, list[int]]) -> list[dict]:
    all_rows = {int(feature["row"]) for feature in features}
    target_count = _RNG.randint(4, 6)
    options: list[dict] = []
    for side in ("left", "right"):
        for row in candidates[side]:
            if int(row) in all_rows:
                continue
            for kind in ("tab", "notch"):
                for ref in (FG, REF_ERASE):
                    options.append({"row": int(row), "side": side, "kind": kind, "ref": int(ref)})
    _RNG.shuffle(options)
    for option in options:
        if len(features) >= target_count:
            break
        row = int(option["row"])
        if row in all_rows:
            continue
        if not _rows_far_enough(list(all_rows) + [row]):
            continue
        features.append(option)
        all_rows.add(row)
    return sorted(features, key=lambda item: int(item["row"]))


def _apply_features_to_clean(clean_mask: np.ndarray, features: list[dict], row_lefts: list[int], row_rights: list[int]) -> np.ndarray:
    mask = clean_mask.copy()
    for feature in features:
        r = int(feature["row"])
        if feature["side"] == "left":
            boundary = int(row_lefts[r])
            if feature["kind"] == "tab":
                mask[r, boundary - 1] = True
            else:
                mask[r, boundary] = False
        else:
            boundary = int(row_rights[r])
            if feature["kind"] == "tab":
                mask[r, boundary + 1] = True
            else:
                mask[r, boundary] = False
    return mask


def _mask_to_grid(mask: np.ndarray) -> np.ndarray:
    grid = np.full(mask.shape, BG, dtype=int)
    grid[mask] = FG
    return grid


def _plan_oracle(input_mask: np.ndarray, features: list[dict]) -> np.ndarray:
    work = _mask_to_grid(input_mask)
    for feature in sorted(features, key=lambda item: int(item["row"])):
        current = dict(feature)
        if current["kind"] == "tab":
            span = _row_span(work == FG, int(current["row"]))
            assert span is not None
            left, right = span
            current["cell"] = (
                int(current["row"]),
                int(left if current["side"] == "left" else right),
            )
        else:
            span = _row_span(work == FG, int(current["row"]))
            assert span is not None
            left, right = span
            current["cell"] = (
                int(current["row"]),
                int(left - 1 if current["side"] == "left" else right + 1),
            )

        if int(feature["ref"]) == FG:
            _apply_completion(work, current)
        else:
            _apply_erase(work, current)
    return work


def _all_complete_oracle(input_mask: np.ndarray, features: list[dict]) -> np.ndarray:
    return _plan_oracle(input_mask, [{**feature, "ref": FG} for feature in features])


def _all_erase_oracle(input_mask: np.ndarray, features: list[dict]) -> np.ndarray:
    return _plan_oracle(input_mask, [{**feature, "ref": REF_ERASE} for feature in features])


def _validate_semantics(grid: np.ndarray, clean_mask: np.ndarray, input_mask: np.ndarray, features: list[dict]) -> bool:
    refs = [(int(feature["row"]), int(feature["ref"])) for feature in features]
    rows = sorted(int(feature["row"]) for feature in features)
    if not _rows_far_enough(rows):
        return False
    if {int(feature["ref"]) for feature in features} != {FG, REF_ERASE}:
        return False
    if {feature["kind"] for feature in features} != {"tab", "notch"}:
        return False
    if {feature["side"] for feature in features} != {"left", "right"}:
        return False

    detected_rows = []
    for r, _color in refs:
        feature = _row_feature(input_mask, r)
        if feature is None:
            return False
        detected_rows.append(int(feature["row"]))
    if detected_rows != rows:
        return False

    spans = []
    for r in rows:
        span = _row_span(clean_mask, r)
        if span is None:
            return False
        spans.append(int(span[1] - span[0] + 1))
    if len(set(spans)) < 2:
        return False

    expected = _plan_oracle(input_mask, features)
    solved = main(grid)
    if not np.array_equal(expected, solved):
        return False
    if np.array_equal(expected, _all_complete_oracle(input_mask, features)):
        return False
    if np.array_equal(expected, _all_erase_oracle(input_mask, features)):
        return False
    if int((solved == FG).sum()) == int((input_mask == FG).sum()):
        return False
    return True


def _build_input_grid(left_grid: np.ndarray, features: list[dict], height: int, left_width: int, right_width: int) -> np.ndarray:
    grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, :left_width] = left_grid
    grid[:, left_width] = SEP
    ref_col = int(left_width + 1 + right_width // 2)
    for feature in features:
        grid[int(feature["row"]), ref_col] = int(feature["ref"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = _RNG.randint(13, 18)
    left_width = _RNG.randint(10, 14)
    right_width = _RNG.randint(4, 6)
    clean_mask, row_lefts, row_rights = _build_clean_shape(height, left_width)
    candidates = _feature_candidates(clean_mask, row_lefts, row_rights)

    base_features = _place_required_features(candidates)
    if base_features is None:
        return None
    features = _add_optional_features(base_features, candidates)

    input_mask = _apply_features_to_clean(clean_mask, features, row_lefts, row_rights)
    refs = [(int(feature["row"]), int(feature["ref"])) for feature in features]
    detected = [row for row, _color in refs if _row_feature(input_mask, row) is not None]
    if len(detected) != len(features):
        return None

    left_grid = _mask_to_grid(input_mask)
    grid = _build_input_grid(left_grid, features, height, left_width, right_width)
    if max(grid.shape) > 30:
        return None
    if not _validate_semantics(grid, clean_mask, input_mask, features):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated-symmetry ARC input")
